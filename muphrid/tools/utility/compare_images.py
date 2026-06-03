"""
compare_images

Read-only side-by-side metric comparison of images the agent already has
handles for. The agent never supplies a path — each reference is a handle it
can see in its own state:

  - a variant id from the variant pool (e.g. "remove_gradient_v2"),
  - a checkpoint name from metadata.checkpoints (e.g. "pre_gradient_linear"),
  - the literal "current" for the active working image.

Each handle resolves to a path internally. Metrics are computed fresh per
image via analyze_image's core (invoked read-only on a synthetic state), so
the comparison reflects each image's actual current state rather than a stale
captured slice. Output is metric-major — each metric lines up across images so
contrasts are read in one row — and each image is tagged with its render space
so linear and display images are not silently mixed.

This tool changes no state: it does not promote, restore, or re-point the
working image, and it does not touch the regression baseline. Use it to decide
between candidates; act on the decision with restore_checkpoint /
select_stretch_variant / commit_variant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.tools.utility.analyze import analyze_image


# Metric groups → keys in analyze_image's metric dict. The agent may pass a
# group name, an individual metric key, or "all".
_METRIC_GROUPS: dict[str, list[str]] = {
    "background": [
        "current_background", "background_flatness", "gradient_magnitude",
        "background_quadrants",
    ],
    "color": [
        "per_channel_bg", "green_excess", "channel_imbalance", "channel_stats",
        "mean_saturation", "median_saturation",
    ],
    "noise": ["current_noise", "wavelet_noise", "wavelet_noise_scales", "channel_snr"],
    "snr": ["snr_estimate", "dynamic_range_db"],
    "clipping": ["clipped_shadows_pct", "clipped_highlights_pct", "clipping_per_channel"],
    "stars": ["star_count", "current_fwhm", "fwhm_std", "median_star_peak_ratio", "star_distribution"],
    "dynamic_range": ["dynamic_range_db", "contrast_ratio"],
    "histogram": ["histogram", "histogram_skewness"],
    "linearity": ["is_linear_estimate", "linearity_confidence", "histogram_skewness"],
}
_METRIC_GROUPS["color_balance"] = _METRIC_GROUPS["color"]  # alias

# Default decision panel when no metrics are requested: the headline scalars
# compared across candidates at any stage of the workflow.
_DEFAULT_PANEL: list[str] = [
    "snr_estimate", "current_background", "background_flatness", "gradient_magnitude",
    "green_excess", "channel_imbalance", "per_channel_bg",
    "current_noise", "wavelet_noise",
    "clipped_shadows_pct", "clipped_highlights_pct",
    "current_fwhm", "star_count", "dynamic_range_db", "contrast_ratio",
]


class CompareImagesInput(BaseModel):
    refs: list[str] = Field(
        description=(
            "Images to compare, by handle. Each entry is a variant id from the "
            "variant pool (e.g. 'remove_gradient_v2'), a checkpoint name from "
            "metadata.checkpoints (e.g. 'pre_gradient_linear'), or the literal "
            "'current' for the active working image. Two or more is the usual "
            "case; one is allowed and returns that image's metrics read-only."
        ),
    )
    metrics: list[str] | None = Field(
        default=None,
        description=(
            "Which metrics to line up across the images. Each entry is a metric "
            "group (background, color, noise, snr, clipping, stars, "
            "dynamic_range, histogram, linearity), an individual metric key "
            "(e.g. 'green_excess'), or 'all' for every metric. None returns a "
            "default panel of the most-compared scalars."
        ),
    )
    detect_stars: bool = Field(
        default=True,
        description=(
            "Compute star metrics (count, FWHM) for each image. Star detection "
            "is the slowest part — set False when comparing only "
            "background/color/noise metrics."
        ),
    )
    shadow_thresholds: list[float] | None = Field(
        default=None,
        description="Shadow clip thresholds, passed through to each image's analysis (see analyze_image).",
    )
    highlight_thresholds: list[float] | None = Field(
        default=None,
        description="Highlight clip thresholds, passed through to each image's analysis (see analyze_image).",
    )
    valid_pixel_min: float = Field(
        default=0.0,
        description="Lower valid-pixel bound, passed through to each image's analysis (see analyze_image).",
    )
    valid_pixel_max: float = Field(
        default=0.999,
        description="Upper valid-pixel bound, passed through to each image's analysis (see analyze_image).",
    )


def _resolve_refs(refs: list[str], state: dict) -> tuple[list[dict], list[str], dict, dict]:
    """Map each ref to {ref, kind, path, image_space, label, params}; collect unknowns.

    A ref resolves against the variant pool, then metadata.checkpoints, then the
    literal "current". The agent only ever names handles it can see; paths are
    resolved here and never surfaced back.
    """
    pool = {
        v.get("id"): v
        for v in (state.get("variant_pool") or [])
        if isinstance(v, dict) and v.get("id")
    }
    checkpoints = (state.get("metadata") or {}).get("checkpoints") or {}
    current_path = (state.get("paths") or {}).get("current_image")
    current_space = (state.get("metadata") or {}).get("image_space")

    resolved: list[dict] = []
    unknown: list[str] = []
    for ref in refs:
        if ref in pool:
            v = pool[ref]
            resolved.append({
                "ref": ref, "kind": "variant", "path": v.get("file_path"),
                "image_space": v.get("image_space"), "label": v.get("label"),
                "params": v.get("params"),
            })
        elif ref in checkpoints:
            entry = checkpoints[ref]
            resolved.append({
                "ref": ref, "kind": "checkpoint", "path": entry.get("path"),
                "image_space": entry.get("image_space"), "label": ref, "params": None,
            })
        elif ref == "current":
            resolved.append({
                "ref": "current", "kind": "current", "path": current_path,
                "image_space": current_space, "label": "current working image",
                "params": None,
            })
        else:
            unknown.append(ref)
    return resolved, unknown, pool, checkpoints


def _select_keys(metrics: list[str] | None, per_ref: dict[str, dict]) -> list[str]:
    """Resolve the requested metric selection into an ordered key list."""
    if not metrics:
        return list(_DEFAULT_PANEL)
    if any(m == "all" for m in metrics):
        return sorted({k for m in per_ref.values() for k in m})
    keys: list[str] = []
    for entry in metrics:
        if entry in _METRIC_GROUPS:
            keys.extend(_METRIC_GROUPS[entry])
        else:
            keys.append(entry)  # treat as a literal metric key
    seen: set[str] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


@tool(args_schema=CompareImagesInput)
def compare_images(
    refs: list[str],
    metrics: list[str] | None = None,
    detect_stars: bool = True,
    shadow_thresholds: list[float] | None = None,
    highlight_thresholds: list[float] | None = None,
    valid_pixel_min: float = 0.0,
    valid_pixel_max: float = 0.999,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Compare metrics of two or more images side by side, by handle.

    refs: variant ids (from the pool), checkpoint names (from your
    checkpoints), and/or "current" (the active image). metrics: groups
    (background, color, noise, snr, clipping, stars, dynamic_range, histogram,
    linearity), individual metric keys, or "all"; omit for a default panel.

    Returns a metric-major table — each metric lines up across the images —
    plus a legend tagging each image's render space. Read-only: it computes
    each image's metrics fresh and changes nothing in state (no promotion,
    restore, or baseline update).
    """
    resolved, unknown, pool, checkpoints = _resolve_refs(refs, state)

    if unknown:
        return Command(update={"messages": [ToolMessage(
            content=json.dumps({
                "error": f"Unknown reference(s): {unknown}.",
                "available_variant_ids": sorted(pool.keys()) or ["(none)"],
                "available_checkpoints": sorted(checkpoints.keys()) or ["(none)"],
                "hint": "Use 'current' for the active working image.",
            }, indent=2),
            tool_call_id=tool_call_id,
        )]})

    if not resolved:
        return Command(update={"messages": [ToolMessage(
            content=json.dumps({"error": "No references provided to compare."}),
            tool_call_id=tool_call_id,
        )]})

    working_dir = state["dataset"]["working_dir"]
    phase = state.get("phase")

    per_ref: dict[str, dict] = {}
    legend: list[dict] = []
    unavailable: list[dict] = []

    for item in resolved:
        path = item["path"]
        if not path or not Path(path).exists():
            unavailable.append({"ref": item["ref"], "reason": "file not found" if path else "no path on record"})
            continue
        # Read-only analysis on a synthetic state: analyze_image only reads
        # dataset.working_dir, paths.current_image, metadata.last_analysis_snapshot,
        # regression_warnings, and phase. An empty baseline yields no warnings,
        # and the returned Command is inspected — never applied — so real state
        # is untouched. (The synthetic-state .func() pattern, see CLAUDE.md.)
        synth_state = {
            "dataset": {"working_dir": working_dir},
            "paths": {"current_image": path},
            "metadata": {},
            "regression_warnings": [],
            "phase": phase,
        }
        try:
            cmd = analyze_image.func(
                detect_stars=detect_stars,
                compute_histogram=True,
                shadow_thresholds=shadow_thresholds,
                highlight_thresholds=highlight_thresholds,
                valid_pixel_min=valid_pixel_min,
                valid_pixel_max=valid_pixel_max,
                state=synth_state,
                tool_call_id="compare_images:analyze",
            )
            per_ref[item["ref"]] = cmd.update.get("metrics", {}) or {}
        except Exception as e:  # one bad image must not sink the whole comparison
            unavailable.append({"ref": item["ref"], "reason": f"analysis failed: {type(e).__name__}: {e}"})
            continue
        legend.append({k: item[k] for k in ("ref", "kind", "label", "image_space", "params")})

    keys = _select_keys(metrics, per_ref)
    table: dict[str, dict] = {}
    for k in keys:
        row = {ref: m[k] for ref, m in per_ref.items() if m.get(k) is not None}
        if row:
            table[k] = row

    payload: dict = {"compared": legend, "metrics": table}
    if unavailable:
        payload["unavailable"] = unavailable
    spaces = {item.get("image_space") for item in legend if item.get("image_space")}
    if len(spaces) > 1:
        payload["image_space_warning"] = (
            f"References span render spaces {sorted(spaces)}; background, SNR, "
            f"and clipping are not directly comparable across linear and display."
        )

    return Command(update={"messages": [ToolMessage(
        content=json.dumps(payload, indent=2, default=str),
        tool_call_id=tool_call_id,
    )]})
