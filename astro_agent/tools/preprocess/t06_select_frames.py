"""
T06 — select_frames

Accept or reject frames based on per-frame quality metrics from T05.

## Two selection paths

### Star-metrics path (has_star_metrics=True, default)
Sigma-clipping on FWHM, roundness, star count, and background level.
Use when T05 detected stars and populated R-lines (normal long-exposure subs).

### Laplacian-percentile path (has_star_metrics=False)
Used when Siril could not detect stars — typical for sub-second exposures,
sparse star fields, or short DSO subs. Instead of hard thresholds, frames
are ranked by Laplacian sharpness (blur/trailing proxy) and the bottom
(1 - keep_percentile) fraction is dropped. Default keeps the best 85%.

This matches the PixInsight SubframeSelector philosophy: rank frames by a
quality metric, keep a percentage, let stacking rejection handle residual
outliers. It is deliberately conservative — keeping more rather than fewer
is safer than aggressive pre-stack rejection on short exposures.

Safety constraint: never returns an empty accepted_frames list.
"""

from __future__ import annotations

import math
import statistics

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.graph.state import FrameMetrics


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SelectionCriteria(BaseModel):
    max_fwhm_sigma: float = Field(
        default=2.0,
        description=(
            "Reject frames with FWHM > median + max_fwhm_sigma * std_fwhm. "
            "Use 3.0 for small datasets (< 15 subs) to preserve integration time. "
            "Only used in the star-metrics path (has_star_metrics=True)."
        ),
    )
    min_roundness: float = Field(
        default=0.5,
        description=(
            "Reject frames with roundness below this threshold (0–1). "
            "Low roundness indicates tracking errors or elongated stars. "
            "Only used in the star-metrics path."
        ),
    )
    min_star_count: int = Field(
        default=30,
        description=(
            "Reject frames with fewer detected stars than this minimum. "
            "Only used in the star-metrics path."
        ),
    )
    max_background_sigma: float = Field(
        default=3.0,
        description=(
            "Reject frames with background level > median + max_background_sigma * std. "
            "Catches frames with passing clouds or moon interference. "
            "Only used in the star-metrics path."
        ),
    )
    keep_percentile: float = Field(
        default=0.85,
        description=(
            "Fraction of frames to keep when using the Laplacian-percentile path "
            "(has_star_metrics=False). 0.85 = keep the best 85% by sharpness. "
            "Range: 0.5–1.0. Lower values are more aggressive; default is conservative "
            "to preserve integration time on short subs."
        ),
    )


class SelectFramesInput(BaseModel):
    frame_metrics: dict = Field(
        description=(
            "Per-frame quality metrics dict from analyze_frames (T05). "
            "Keys are frame indices (strings); values have fwhm, roundness, "
            "number_of_stars, background_lvl, sigma, bgnoise, and "
            "laplacian_sharpness fields."
        )
    )
    criteria: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Rejection thresholds. Defaults are appropriate for most datasets.",
    )
    has_star_metrics: bool = Field(
        default=True,
        description=(
            "Pass summary.has_star_metrics from T05. "
            "True → standard FWHM/roundness/star_count sigma-clipping. "
            "False → Laplacian-percentile path (for short exposures, sparse fields)."
        ),
    )


# ── Star-metrics path ──────────────────────────────────────────────────────────

def _sigma_threshold(values: list[float], sigma_mult: float) -> float:
    if len(values) < 2:
        return float("inf")
    med = statistics.median(values)
    std = statistics.stdev(values)
    return med + sigma_mult * std


def _select_by_star_metrics(
    frame_metrics: dict[str, dict],
    criteria: SelectionCriteria,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Standard sigma-clipping path. Requires FWHM, roundness, star_count.
    Returns (accepted, rejected_list, rejection_reasons).
    Safety: if all frames would be rejected, returns all as accepted.
    """
    rejected: dict[str, str] = {}

    fwhms = [v["fwhm"] for v in frame_metrics.values() if v.get("fwhm") is not None]
    bgs   = [v["background_lvl"] for v in frame_metrics.values()
             if v.get("background_lvl") is not None]

    fwhm_threshold = _sigma_threshold(fwhms, criteria.max_fwhm_sigma)
    bg_threshold   = _sigma_threshold(bgs,   criteria.max_background_sigma)

    for filename, metrics in frame_metrics.items():
        reasons: list[str] = []

        fwhm = metrics.get("fwhm")
        if fwhm is not None and fwhm > fwhm_threshold:
            reasons.append(f"FWHM {fwhm:.3f} > threshold {fwhm_threshold:.3f}")

        roundness = metrics.get("roundness")
        if roundness is not None and roundness < criteria.min_roundness:
            reasons.append(f"roundness {roundness:.3f} < min {criteria.min_roundness}")

        star_count = metrics.get("number_of_stars")
        if star_count is not None and star_count < criteria.min_star_count:
            reasons.append(f"star_count {star_count} < min {criteria.min_star_count}")

        bg = metrics.get("background_lvl")
        if bg is not None and bg > bg_threshold:
            reasons.append(f"background {bg:.4f} > threshold {bg_threshold:.4f}")

        if reasons:
            rejected[filename] = "; ".join(reasons)

    accepted = [f for f in frame_metrics if f not in rejected]

    if not accepted:
        accepted = list(frame_metrics.keys())
        rejected = {}

    return accepted, list(rejected.keys()), rejected


# ── Laplacian-percentile path ──────────────────────────────────────────────────

def _select_by_ranking(
    frame_metrics: dict[str, dict],
    keep_percentile: float,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Percentile-based selection using Laplacian sharpness as quality metric.

    Ranks all frames with available sharpness data from best (highest
    Laplacian variance) to worst. Drops the bottom (1 - keep_percentile)
    fraction. Frames without sharpness data are always kept — we can't
    evaluate what we can't measure.

    Returns (accepted, rejected_list, rejection_reasons).
    Safety: if all frames would be rejected, returns all as accepted.
    """
    # Partition: frames with and without sharpness data
    with_sharpness = [
        (k, v["laplacian_sharpness"])
        for k, v in frame_metrics.items()
        if v.get("laplacian_sharpness") is not None
    ]
    without_sharpness = [
        k for k, v in frame_metrics.items()
        if v.get("laplacian_sharpness") is None
    ]

    # Sort by sharpness descending (best = highest Laplacian variance)
    with_sharpness.sort(key=lambda x: x[1], reverse=True)

    n_with = len(with_sharpness)
    n_keep = max(1, math.ceil(n_with * keep_percentile))

    accepted_ranked = [k for k, _ in with_sharpness[:n_keep]]
    rejected_ranked = [k for k, _ in with_sharpness[n_keep:]]

    # Frames without sharpness data are always accepted (can't evaluate)
    accepted = accepted_ranked + without_sharpness
    rejected_reasons: dict[str, str] = {
        k: (
            f"Laplacian sharpness in bottom {(1 - keep_percentile):.0%} of batch "
            f"(sharpness={v:.1f})"
        )
        for k, v in with_sharpness[n_keep:]
    }

    if not accepted:
        accepted = list(frame_metrics.keys())
        rejected_ranked = []
        rejected_reasons = {}

    return accepted, rejected_ranked, rejected_reasons


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SelectFramesInput)
def select_frames(
    frame_metrics: dict,
    criteria: dict | None = None,
    has_star_metrics: bool = True,
) -> dict:
    """
    Accept or reject frames based on per-frame quality metrics from T05.

    Pass summary.has_star_metrics from T05 to select the right strategy:

    has_star_metrics=True (default):
      Sigma-clipping on FWHM, roundness, star count, background.
      Standard path for normal long-exposure subs.

    has_star_metrics=False:
      Percentile ranking on Laplacian sharpness. Keeps the best 85% of frames
      (configurable via criteria.keep_percentile). Use for sub-second exposures
      or any dataset where Siril couldn't detect stars per-frame.
      Conservative by default — stacking with sigma-clipping (T07) handles
      residual outliers.

    Never returns an empty accepted_frames list.
    For small datasets (< 15 subs), set criteria.max_fwhm_sigma=3.0 to
    preserve integration time.
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

    if has_star_metrics:
        accepted, rejected_list, rejection_reasons = _select_by_star_metrics(
            frame_metrics, parsed_criteria
        )
        selection_path = "star_metrics"
    else:
        accepted, rejected_list, rejection_reasons = _select_by_ranking(
            frame_metrics, parsed_criteria.keep_percentile
        )
        selection_path = f"laplacian_percentile_{parsed_criteria.keep_percentile:.0%}"

    n_total = len(frame_metrics)
    acceptance_rate = len(accepted) / n_total if n_total > 0 else 1.0

    warnings: list[str] = []
    if acceptance_rate < 0.5:
        warnings.append(
            f"Acceptance rate is {acceptance_rate:.0%} ({len(accepted)}/{n_total} frames). "
            "Consider loosening thresholds or increasing keep_percentile."
        )

    return {
        "accepted_frames":   accepted,
        "rejected_frames":   rejected_list,
        "rejection_reasons": rejection_reasons,
        "acceptance_rate":   round(acceptance_rate, 4),
        "selection_path":    selection_path,
        "warnings":          warnings,
    }
