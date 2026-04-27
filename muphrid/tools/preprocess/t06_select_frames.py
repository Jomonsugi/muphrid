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
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState


# ── Selection criteria schema ──────────────────────────────────────────────────

class SelectionCriteria(BaseModel):
    max_fwhm_sigma: float | None = Field(
        default=2.0,
        description=(
            "Reject frames with FWHM > median + max_fwhm_sigma × std. "
            "Use 3.0 for small datasets (< 15 subs) to preserve integration time. "
            "Use 1.5 for excellent seeing when you want only the sharpest frames. "
            "None = no FWHM threshold (bypass this filter entirely)."
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
    min_roundness: float | None = Field(
        default=0.5,
        description=(
            "Reject frames with roundness below this (0–1). "
            "1.0 = perfect circles. Below 0.6 indicates significant elongation. "
            "Tighten to 0.7 for quality-focused selection. "
            "None = no roundness threshold (bypass this filter entirely)."
        ),
    )
    min_star_count: int | None = Field(
        default=None,
        description=(
            "Reject frames with fewer detected stars than this threshold. "
            "Low star count indicates clouds, fog, or detection failure. "
            "Derive from T05 summary.median_star_count — set to ~50% of "
            "median for a balanced filter that catches problem frames "
            "without discarding variable-but-valid ones. "
            "Example: median_star_count=402 → min_star_count=200. "
            "None = no star-count threshold (bypass this filter entirely)."
        ),
    )
    max_background_sigma: float | None = Field(
        default=3.0,
        description=(
            "Reject frames with background > median + max_background_sigma × std. "
            "Catches frames affected by passing clouds, moon, or light pollution spikes. "
            "None = no background threshold (bypass this filter entirely)."
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

    # Any None threshold means "bypass this filter". Using +inf for the
    # comparison keeps the selection loop branchless without leaking the
    # None type into numeric operations.
    fwhm_threshold = (
        _sigma_threshold(fwhms, criteria.max_fwhm_sigma)
        if criteria.max_fwhm_sigma is not None else float("inf")
    )
    bg_threshold = (
        _sigma_threshold(bgs, criteria.max_background_sigma)
        if criteria.max_background_sigma is not None else float("inf")
    )

    wfwhm_threshold = float("inf")
    if criteria.max_wfwhm_sigma is not None:
        wfwhms = [v["weighted_fwhm"] for v in frame_metrics.values()
                  if v.get("weighted_fwhm") is not None and v["weighted_fwhm"] > 0]
        wfwhm_threshold = _sigma_threshold(wfwhms, criteria.max_wfwhm_sigma)

    for frame_key, metrics in frame_metrics.items():
        reasons: list[str] = []

        fwhm = metrics.get("fwhm")
        # Unconditional precondition: a frame without registration data was
        # dropped by registration (no R-line in the .seq file). It still
        # appears in frame_stats because calibration M-line stats are
        # written independently. Such a frame cannot be stacked — reject it
        # regardless of the sigma / min thresholds so the accepted list is
        # always mappable to positions in the registered .seq.
        if fwhm is None:
            reasons.append(
                "no registration data — frame dropped during registration "
                "(no R-line in the .seq file); cannot be stacked"
            )
        elif fwhm > fwhm_threshold:
            reasons.append(f"FWHM {fwhm:.3f} > threshold {fwhm_threshold:.3f}")

        wfwhm = metrics.get("weighted_fwhm")
        if criteria.max_wfwhm_sigma is not None and wfwhm is not None and wfwhm > wfwhm_threshold:
            reasons.append(f"wFWHM {wfwhm:.3f} > threshold {wfwhm_threshold:.3f}")

        roundness = metrics.get("roundness")
        if (criteria.min_roundness is not None
                and roundness is not None
                and roundness < criteria.min_roundness):
            reasons.append(f"roundness {roundness:.3f} < min {criteria.min_roundness}")

        star_count = metrics.get("number_of_stars")
        if (criteria.min_star_count is not None
                and star_count is not None
                and star_count < criteria.min_star_count):
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

    # Safety escape: if every frame was rejected by the user-configurable
    # criteria, keep all frames that still have registration data (fwhm not
    # None) so stacking can proceed. Frames without registration data remain
    # rejected even in the fallback — they have no .seq position to stack.
    if not accepted:
        registered = [
            f for f, m in frame_metrics.items() if m.get("fwhm") is not None
        ]
        if registered:
            accepted = registered
            # Preserve the "no registration data" rejections; drop the rest.
            rejected = {
                k: v for k, v in rejected.items()
                if k not in registered
            }

    return accepted, list(rejected.keys()), rejected


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool
def select_frames(
    criteria: SelectionCriteria,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AstroState, InjectedState],
) -> Command:
    """
    Accept or reject frames based on per-frame registration metrics from analyze_frames.

    Frame metrics are read from state. Sigma-clipping on FWHM, wFWHM, roundness,
    star count, background, and quality. Accepted frame keys are written to state
    for siril_stack (T07) to consume.

    Args:
        criteria: Rejection thresholds. Tune based on T05 summary statistics:
          - Small dataset (< 15 subs): max_fwhm_sigma=3.0 (preserve integration time)
          - Variable seeing (seeing_stability > 0.3): max_fwhm_sigma=1.5 (be selective)
          - Tracking issues (tracking_quality < 0.7): raise min_roundness to 0.6
          - min_star_count: set to ~50% of median_star_count from T05 summary

    Never produces an empty accepted list — if all frames fail, all are kept.
    """
    frame_metrics = state["metrics"].get("frame_stats", {})
    if not frame_metrics:
        raise ValueError(
            "frame_stats is empty in state. Run analyze_frames (T05) first."
        )

    accepted, rejected_list, rejection_reasons = _select_frames(frame_metrics, criteria)

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

    import json
    result = {
        "accepted_count": len(accepted),
        "rejected_count": len(rejected_list),
        "total_count": n_total,
        "acceptance_rate": round(acceptance_rate, 3),
        "rejection_reasons": rejection_reasons,
        "warnings": warnings,
    }
    return Command(update={
        "paths": {**state["paths"], "selected_frames": accepted},
        "messages": [ToolMessage(content=json.dumps(result, indent=2), tool_call_id=tool_call_id)],
    })
