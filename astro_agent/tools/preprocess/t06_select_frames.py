"""
T06 — select_frames

Accept or reject frames based on per-frame registration metrics from T05.

Sigma-clipping on FWHM, wFWHM, roundness, star count, background, and quality.
Each criterion can be independently tuned — the agent should adjust thresholds
based on T05 summary statistics and the specific dataset.

This mirrors how a human uses Siril's Plot tab and stacking UI: examine the
per-frame data, decide which metrics to filter on, and choose how aggressively
to cut.  The agent has the same levers:

  - max_fwhm_sigma / max_wfwhm_sigma: tighten for consistent seeing, loosen
    for variable conditions or small datasets
  - min_roundness: tighten to reject tracking errors
  - min_star_count: reject frames where detection failed or clouds intervened
  - max_background_sigma: reject frames with anomalous sky brightness
  - min_quality: Siril's composite quality metric (0-1), higher is better

Safety: never returns an empty accepted_frames list.
"""

from __future__ import annotations

import statistics

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SelectionCriteria(BaseModel):
    max_fwhm_sigma: float = Field(
        default=2.0,
        description=(
            "Reject frames with FWHM > median + max_fwhm_sigma × std. "
            "Use 3.0 for small datasets (< 15 subs) to preserve integration time. "
            "Use 1.5 for excellent seeing when you want only the sharpest frames."
        ),
    )
    max_wfwhm_sigma: float | None = Field(
        default=None,
        description=(
            "Reject frames with weighted FWHM > median + max_wfwhm_sigma × std. "
            "wFWHM accounts for star brightness — often a better quality indicator "
            "than raw FWHM.  None = no wFWHM filtering (default)."
        ),
    )
    min_roundness: float = Field(
        default=0.5,
        description=(
            "Reject frames with roundness below this (0–1). "
            "1.0 = perfect circles. Below 0.6 indicates significant elongation. "
            "Tighten to 0.7 for quality-focused selection."
        ),
    )
    min_star_count: int = Field(
        description=(
            "Reject frames with fewer detected stars than this threshold. "
            "Low star count indicates clouds, fog, or detection failure. "
            "Must be derived from T05 summary.median_star_count — "
            "set to ~50% of median for a balanced filter that catches "
            "problem frames without discarding variable-but-valid ones. "
            "Example: median_star_count=402 → min_star_count=200."
        ),
    )
    max_background_sigma: float = Field(
        default=3.0,
        description=(
            "Reject frames with background > median + max_background_sigma × std. "
            "Catches frames affected by passing clouds, moon, or light pollution spikes."
        ),
    )
    min_quality: float | None = Field(
        default=None,
        description=(
            "Reject frames with Siril quality score below this threshold (0–1). "
            "Quality is Siril's composite metric combining FWHM, roundness, and "
            "star count.  None = no quality filtering (default).  Set to ~50% "
            "of T05 summary.median_quality for moderate filtering."
        ),
    )


class SelectFramesInput(BaseModel):
    frame_metrics: dict = Field(
        description=(
            "Per-frame quality metrics dict from analyze_frames (T05). "
            "Keys are frame indices (strings); values contain fwhm, "
            "weighted_fwhm, roundness, quality, number_of_stars, "
            "background_lvl, and pixel stats."
        )
    )
    criteria: SelectionCriteria = Field(
        description=(
            "Rejection thresholds. All sigma-based thresholds adapt to the dataset "
            "distribution, but min_star_count must be set explicitly from T05 "
            "summary.median_star_count (~50% of median is a safe starting point)."
        ),
    )


# ── Selection logic ───────────────────────────────────────────────────────────

def _sigma_threshold(values: list[float], sigma_mult: float) -> float:
    """Compute median + sigma_mult × stdev threshold."""
    if len(values) < 2:
        return float("inf")
    med = statistics.median(values)
    std = statistics.stdev(values)
    return med + sigma_mult * std


def _select_frames(
    frame_metrics: dict[str, dict],
    criteria: SelectionCriteria,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Sigma-clipping selection on registration metrics.
    Returns (accepted, rejected_list, rejection_reasons).
    Safety: if all frames would be rejected, returns all as accepted.
    """
    rejected: dict[str, str] = {}

    fwhms = [v["fwhm"] for v in frame_metrics.values() if v.get("fwhm") is not None and v["fwhm"] > 0]
    bgs   = [v["background_lvl"] for v in frame_metrics.values() if v.get("background_lvl") is not None]

    fwhm_threshold = _sigma_threshold(fwhms, criteria.max_fwhm_sigma)
    bg_threshold   = _sigma_threshold(bgs, criteria.max_background_sigma)

    wfwhm_threshold = float("inf")
    if criteria.max_wfwhm_sigma is not None:
        wfwhms = [v["weighted_fwhm"] for v in frame_metrics.values()
                  if v.get("weighted_fwhm") is not None and v["weighted_fwhm"] > 0]
        wfwhm_threshold = _sigma_threshold(wfwhms, criteria.max_wfwhm_sigma)

    for frame_key, metrics in frame_metrics.items():
        reasons: list[str] = []

        fwhm = metrics.get("fwhm")
        if fwhm is not None and fwhm > fwhm_threshold:
            reasons.append(f"FWHM {fwhm:.3f} > threshold {fwhm_threshold:.3f}")

        wfwhm = metrics.get("weighted_fwhm")
        if criteria.max_wfwhm_sigma is not None and wfwhm is not None and wfwhm > wfwhm_threshold:
            reasons.append(f"wFWHM {wfwhm:.3f} > threshold {wfwhm_threshold:.3f}")

        roundness = metrics.get("roundness")
        if roundness is not None and roundness < criteria.min_roundness:
            reasons.append(f"roundness {roundness:.3f} < min {criteria.min_roundness}")

        star_count = metrics.get("number_of_stars")
        if star_count is not None and star_count < criteria.min_star_count:
            reasons.append(f"star_count {star_count} < min {criteria.min_star_count}")

        bg = metrics.get("background_lvl")
        if bg is not None and bg > bg_threshold:
            reasons.append(f"background {bg:.6f} > threshold {bg_threshold:.6f}")

        quality = metrics.get("quality")
        if criteria.min_quality is not None and quality is not None and quality < criteria.min_quality:
            reasons.append(f"quality {quality:.6f} < min {criteria.min_quality}")

        if reasons:
            rejected[frame_key] = "; ".join(reasons)

    accepted = [f for f in frame_metrics if f not in rejected]

    if not accepted:
        accepted = list(frame_metrics.keys())
        rejected = {}

    return accepted, list(rejected.keys()), rejected


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SelectFramesInput)
def select_frames(
    frame_metrics: dict,
    criteria: dict | None = None,
) -> dict:
    """
    Accept or reject frames based on per-frame registration metrics from T05.

    Sigma-clipping on FWHM, wFWHM, roundness, star count, background, and
    quality.  Each threshold is independently tunable.

    Recommended workflow:
      1. Run T05 to get summary statistics
      2. Review summary — median_fwhm, tracking_quality, sky_consistency
      3. Set criteria based on the data:
         - Small dataset (< 15 subs): max_fwhm_sigma=3.0 (preserve integration)
         - Variable seeing (seeing_stability > 0.3): max_fwhm_sigma=1.5 (be selective)
         - Tracking issues (tracking_quality < 0.7): raise min_roundness to 0.6
         - Dense field: lower min_star_count based on median_star_count

    Never returns an empty accepted_frames list — if all frames fail every
    criterion, all are returned with a warning.
    """
    if not frame_metrics:
        raise ValueError(
            "frame_metrics is empty. Run analyze_frames (T05) first and pass "
            "its frame_metrics output."
        )

    if isinstance(criteria, SelectionCriteria):
        parsed_criteria = criteria
    else:
        parsed_criteria = SelectionCriteria(**(criteria or {}))

    accepted, rejected_list, rejection_reasons = _select_frames(
        frame_metrics, parsed_criteria
    )

    n_total = len(frame_metrics)
    acceptance_rate = len(accepted) / n_total if n_total > 0 else 1.0

    warnings: list[str] = []

    if acceptance_rate < 0.5:
        warnings.append(
            f"Acceptance rate is {acceptance_rate:.0%} ({len(accepted)}/{n_total} frames). "
            "More than 50% rejected — review rejection reasons and consider loosening thresholds."
        )

    if len(accepted) < 3 and n_total >= 3:
        warnings.append(
            f"Only {len(accepted)} frames accepted from {n_total}. "
            "Stacking quality will be limited. Consider loosening criteria."
        )

    return {
        "accepted_frames":   accepted,
        "rejected_frames":   rejected_list,
        "rejection_reasons": rejection_reasons,
        "acceptance_rate":   round(acceptance_rate, 4),
        "warnings":          warnings,
    }
