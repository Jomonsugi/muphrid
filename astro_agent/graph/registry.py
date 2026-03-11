"""
Tool registry — maps processing phases to their available tools.

The phase_router node reads `state["phase"]` and calls `tools_for_phase()` to
get the tool list for `model.bind_tools()`. Utility tools are included in
every phase gate automatically.

See graph_design.md §Phase Gates & Tool Groupings for the rationale.
"""

from __future__ import annotations

from astro_agent.graph.state import ProcessingPhase

# ── Imports: Preprocessing ────────────────────────────────────────────────────

from astro_agent.tools.preprocess.t01_ingest import ingest_dataset
from astro_agent.tools.preprocess.t02_masters import build_masters
from astro_agent.tools.preprocess.t02b_convert_sequence import convert_sequence
from astro_agent.tools.preprocess.t03_calibrate import calibrate
from astro_agent.tools.preprocess.t04_register import siril_register
from astro_agent.tools.preprocess.t05_analyze_frames import analyze_frames
from astro_agent.tools.preprocess.t06_select_frames import select_frames
from astro_agent.tools.preprocess.t07_stack import siril_stack
from astro_agent.tools.preprocess.t08_crop import auto_crop

# ── Imports: Linear ───────────────────────────────────────────────────────────

from astro_agent.tools.linear.t09_gradient import remove_gradient
from astro_agent.tools.linear.t10_color_calibrate import color_calibrate
from astro_agent.tools.linear.t11_green_noise import remove_green_noise
from astro_agent.tools.linear.t12_noise_reduction import noise_reduction
from astro_agent.tools.linear.t13_deconvolution import deconvolution

# ── Imports: Stretch ──────────────────────────────────────────────────────────

from astro_agent.tools.nonlinear.t14_stretch import stretch_image

# ── Imports: Non-linear ───────────────────────────────────────────────────────

from astro_agent.tools.nonlinear.t15_star_removal import star_removal
from astro_agent.tools.nonlinear.t16_curves import curves_adjust
from astro_agent.tools.nonlinear.t17_local_contrast import local_contrast_enhance
from astro_agent.tools.nonlinear.t18_saturation import saturation_adjust
from astro_agent.tools.nonlinear.t19_star_restoration import star_restoration
from astro_agent.tools.scikit.t25_create_mask import create_mask
from astro_agent.tools.scikit.t26_reduce_stars import reduce_stars
from astro_agent.tools.scikit.t27_multiscale import multiscale_process

# ── Imports: Export ───────────────────────────────────────────────────────────

from astro_agent.tools.utility.t24_export import export_final

# ── Imports: Utility (available in every gate) ────────────────────────────────

from astro_agent.tools.utility.t20_analyze import analyze_image
from astro_agent.tools.utility.t21_plate_solve import plate_solve
from astro_agent.tools.utility.t23_pixel_math import pixel_math
from astro_agent.tools.utility.t28_extract_narrowband import extract_narrowband
from astro_agent.tools.utility.t29_resolve_target import resolve_target


# ── Tool groups ───────────────────────────────────────────────────────────────

UTILITY_TOOLS = [
    analyze_image,
    plate_solve,
    pixel_math,
    extract_narrowband,
    resolve_target,
]

PREPROCESS_TOOLS = [
    ingest_dataset,
    build_masters,
    convert_sequence,
    calibrate,
    siril_register,
    analyze_frames,
    select_frames,
    siril_stack,
    auto_crop,
]

LINEAR_TOOLS = [
    remove_gradient,
    color_calibrate,
    remove_green_noise,
    noise_reduction,
    deconvolution,
]

STRETCH_TOOLS = [
    stretch_image,
]

NONLINEAR_TOOLS = [
    star_removal,
    curves_adjust,
    local_contrast_enhance,
    saturation_adjust,
    star_restoration,
    create_mask,
    reduce_stars,
    multiscale_process,
]

EXPORT_TOOLS = [
    export_final,
]


# ── Phase → tool list mapping ────────────────────────────────────────────────

_PHASE_TO_GATE: dict[ProcessingPhase, list] = {
    # Gate 1: PREPROCESS (covers INGEST through STACKING)
    ProcessingPhase.INGEST:       PREPROCESS_TOOLS,
    ProcessingPhase.CALIBRATION:  PREPROCESS_TOOLS,
    ProcessingPhase.REGISTRATION: PREPROCESS_TOOLS,
    ProcessingPhase.ANALYSIS:     PREPROCESS_TOOLS,
    ProcessingPhase.STACKING:     PREPROCESS_TOOLS,
    # Gate 2: LINEAR
    ProcessingPhase.LINEAR:       LINEAR_TOOLS,
    # Gate 3: STRETCH
    ProcessingPhase.STRETCH:      STRETCH_TOOLS,
    # Gate 4: NONLINEAR
    ProcessingPhase.NONLINEAR:    NONLINEAR_TOOLS,
    # Gate 5: EXPORT
    ProcessingPhase.EXPORT:       EXPORT_TOOLS,
    # Terminal
    ProcessingPhase.COMPLETE:     [],
    ProcessingPhase.REVIEW:       [],
}


def tools_for_phase(phase: ProcessingPhase) -> list:
    """Return the full tool list for a given phase: gate tools + utility tools."""
    gate_tools = _PHASE_TO_GATE.get(phase, [])
    return gate_tools + UTILITY_TOOLS


def all_tools() -> list:
    """Return every registered tool. Used for ToolNode initialization."""
    seen = set()
    tools = []
    for group in [PREPROCESS_TOOLS, LINEAR_TOOLS, STRETCH_TOOLS,
                  NONLINEAR_TOOLS, EXPORT_TOOLS, UTILITY_TOOLS]:
        for t in group:
            if t.name not in seen:
                seen.add(t.name)
                tools.append(t)
    return tools
