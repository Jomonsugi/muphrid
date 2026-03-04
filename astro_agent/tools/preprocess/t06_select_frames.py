"""
T06 — select_frames

Accept or reject frames based on per-frame quality metrics from T05.
Rejection uses sigma-clipping relative to the median of each metric.

Pure Python — no Siril invocation. Operates on the frame_metrics dict
returned by analyze_frames.

Safety constraint: never returns an empty accepted_frames list.
"""

from __future__ import annotations

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
            "Use 3.0 for small datasets (< 15 subs) to preserve integration time."
        ),
    )
    min_roundness: float = Field(
        default=0.5,
        description=(
            "Reject frames with roundness below this threshold (0–1). "
            "Low roundness indicates tracking errors or elongated stars."
        ),
    )
    min_star_count: int = Field(
        default=30,
        description="Reject frames with fewer detected stars than this minimum.",
    )
    max_background_sigma: float = Field(
        default=3.0,
        description=(
            "Reject frames with background level > median + max_background_sigma * std. "
            "Catches frames with passing clouds or moon interference."
        ),
    )


class SelectFramesInput(BaseModel):
    frame_metrics: dict = Field(
        description=(
            "Per-frame quality metrics dict from analyze_frames (T05). "
            "Keys are filenames; values have fwhm, roundness, star_count, "
            "background_level fields."
        )
    )
    criteria: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Rejection thresholds. Defaults are appropriate for most datasets.",
    )


# ── Rejection logic ────────────────────────────────────────────────────────────

def _sigma_threshold(values: list[float], sigma_mult: float) -> float:
    if len(values) < 2:
        return float("inf")
    med = statistics.median(values)
    std = statistics.stdev(values)
    return med + sigma_mult * std


def _select(
    frame_metrics: dict[str, dict],
    criteria: SelectionCriteria,
) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Returns (accepted, rejected, rejection_reasons).
    Safety: if all frames would be rejected, returns all as accepted with a warning.
    """
    rejected: dict[str, str] = {}

    # Pre-compute thresholds
    fwhms = [v["fwhm"] for v in frame_metrics.values() if v.get("fwhm") is not None]
    bgs   = [v["background_level"] for v in frame_metrics.values() if v.get("background_level") is not None]

    fwhm_threshold = _sigma_threshold(fwhms, criteria.max_fwhm_sigma)
    bg_threshold   = _sigma_threshold(bgs,   criteria.max_background_sigma)

    for filename, metrics in frame_metrics.items():
        reasons: list[str] = []

        fwhm = metrics.get("fwhm")
        if fwhm is not None and fwhm > fwhm_threshold:
            reasons.append(
                f"FWHM {fwhm:.3f} > threshold {fwhm_threshold:.3f}"
            )

        roundness = metrics.get("roundness")
        if roundness is not None and roundness < criteria.min_roundness:
            reasons.append(
                f"roundness {roundness:.3f} < min {criteria.min_roundness}"
            )

        star_count = metrics.get("star_count")
        if star_count is not None and star_count < criteria.min_star_count:
            reasons.append(
                f"star_count {star_count} < min {criteria.min_star_count}"
            )

        bg = metrics.get("background_level")
        if bg is not None and bg > bg_threshold:
            reasons.append(
                f"background {bg:.4f} > threshold {bg_threshold:.4f}"
            )

        if reasons:
            rejected[filename] = "; ".join(reasons)

    accepted = [f for f in frame_metrics if f not in rejected]

    # Safety: never return empty accepted list
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
    Accept or reject frames based on sigma-clipped quality metric thresholds.
    Returns accepted_frames list to pass to siril_stack (T07).

    Never returns an empty accepted_frames — if all frames fail thresholds,
    all are accepted and a warning is returned. For small datasets (< 15 subs),
    use max_fwhm_sigma=3.0 to preserve integration time.
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

    accepted, rejected_list, rejection_reasons = _select(frame_metrics, parsed_criteria)

    n_total = len(frame_metrics)
    acceptance_rate = len(accepted) / n_total if n_total > 0 else 1.0

    warnings: list[str] = []
    if acceptance_rate < 0.5:
        warnings.append(
            f"Acceptance rate is {acceptance_rate:.0%} ({len(accepted)}/{n_total} frames). "
            "Consider loosening thresholds (increase max_fwhm_sigma or lower min_star_count)."
        )
    if not rejection_reasons and len(accepted) == n_total:
        # All frames accepted — may be because safety fallback fired
        pass

    return {
        "accepted_frames":    accepted,
        "rejected_frames":    rejected_list,
        "rejection_reasons":  rejection_reasons,
        "acceptance_rate":    round(acceptance_rate, 4),
        "warnings":           warnings,
    }
