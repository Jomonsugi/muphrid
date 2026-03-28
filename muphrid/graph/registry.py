"""
Tool registry — maps processing phases to their available tools.

The phase_router node reads `state["phase"]` and calls `tools_for_phase()` to
get the tool list for `model.bind_tools()`. Utility tools are included in
every phase gate automatically.

See graph_design.md §Phase Gates & Tool Groupings for the rationale.
"""

from __future__ import annotations

from typing import get_type_hints

from muphrid.graph.state import ProcessingPhase

# ── Imports: Preprocessing ────────────────────────────────────────────────────

from muphrid.tools.preprocess.t02_masters import build_masters
from muphrid.tools.preprocess.t02b_convert_sequence import convert_sequence
from muphrid.tools.preprocess.t03_calibrate import calibrate
from muphrid.tools.preprocess.t04_register import siril_register
from muphrid.tools.preprocess.t05_analyze_frames import analyze_frames
from muphrid.tools.preprocess.t06_select_frames import select_frames
from muphrid.tools.preprocess.t07_stack import siril_stack
from muphrid.tools.preprocess.t08_crop import auto_crop

# ── Imports: Linear ───────────────────────────────────────────────────────────

from muphrid.tools.linear.t09_gradient import remove_gradient
from muphrid.tools.linear.t10_color_calibrate import color_calibrate
from muphrid.tools.linear.t11_green_noise import remove_green_noise
from muphrid.tools.linear.t12_noise_reduction import noise_reduction
from muphrid.tools.linear.t13_deconvolution import deconvolution

# ── Imports: Stretch ──────────────────────────────────────────────────────────

from muphrid.tools.nonlinear.t14_stretch import stretch_image, select_stretch_variant

# ── Imports: Non-linear ───────────────────────────────────────────────────────

from muphrid.tools.nonlinear.t15_star_removal import star_removal
from muphrid.tools.nonlinear.t16_curves import curves_adjust
from muphrid.tools.nonlinear.t17_local_contrast import local_contrast_enhance
from muphrid.tools.nonlinear.t18_saturation import saturation_adjust
from muphrid.tools.nonlinear.t19_star_restoration import star_restoration
from muphrid.tools.scikit.t25_create_mask import create_mask
from muphrid.tools.scikit.t26_reduce_stars import reduce_stars
from muphrid.tools.scikit.t27_multiscale import multiscale_process
from muphrid.tools.nonlinear.t31_checkpoint import save_checkpoint, restore_checkpoint

# ── Imports: Export ───────────────────────────────────────────────────────────

from muphrid.tools.utility.t24_export import export_final

# ── Imports: Utility (available in every gate) ────────────────────────────────

from muphrid.tools.utility.t20_analyze import analyze_image
from muphrid.tools.utility.t21_plate_solve import plate_solve
from muphrid.tools.utility.t23_pixel_math import pixel_math
from muphrid.tools.utility.t28_extract_narrowband import extract_narrowband
from muphrid.tools.utility.t29_resolve_target import resolve_target
from muphrid.tools.utility.t30_advance_phase import advance_phase
from muphrid.tools.utility.t32_present_images import present_images


# ── Tool groups ───────────────────────────────────────────────────────────────

UTILITY_TOOLS = [
    analyze_image,
    plate_solve,
    pixel_math,
    extract_narrowband,
    resolve_target,
    advance_phase,
    present_images,
]


def register_memory_tool():
    """Conditionally add memory_search to UTILITY_TOOLS when memory is enabled."""
    from muphrid.graph.hitl import is_memory_enabled
    if is_memory_enabled():
        from muphrid.tools.utility.t33_memory_search import memory_search
        if memory_search not in UTILITY_TOOLS:
            UTILITY_TOOLS.append(memory_search)


# Preprocessing: strict per-phase gating. Each step physically depends on the
# output of the previous — no cross-phase backtracking. Within-phase iteration
# is valid (e.g. rebuild a master, re-register with different params, re-select
# + restack). Research confirms this is how PixInsight, Siril, and expert
# astrophotographers work.

CALIBRATION_TOOLS = [
    build_masters,
    convert_sequence,
    calibrate,
]

REGISTRATION_TOOLS = [
    siril_register,
]

ANALYSIS_TOOLS = [
    analyze_frames,
]

STACKING_TOOLS = [
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
    select_stretch_variant,
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
    save_checkpoint,
    restore_checkpoint,
]

EXPORT_TOOLS = [
    export_final,
]


# ── Phase → tool list mapping ────────────────────────────────────────────────

_PHASE_TO_GATE: dict[ProcessingPhase, list] = {
    # Preprocessing: strict per-phase gating (sequential physical dependencies)
    ProcessingPhase.INGEST:       [],                # resolve_target via UTILITY_TOOLS
    ProcessingPhase.CALIBRATION:  CALIBRATION_TOOLS,
    ProcessingPhase.REGISTRATION: REGISTRATION_TOOLS,
    ProcessingPhase.ANALYSIS:     ANALYSIS_TOOLS,
    ProcessingPhase.STACKING:     STACKING_TOOLS,
    # Post-preprocessing: fluid iteration, backtracking, creativity
    ProcessingPhase.LINEAR:       LINEAR_TOOLS,
    ProcessingPhase.STRETCH:      STRETCH_TOOLS,
    ProcessingPhase.NONLINEAR:    NONLINEAR_TOOLS,
    ProcessingPhase.EXPORT:       EXPORT_TOOLS,
    # Terminal
    ProcessingPhase.COMPLETE:     [],
    ProcessingPhase.REVIEW:       [],
}


def tools_for_phase(phase: ProcessingPhase) -> list:
    """Return the full tool list for a given phase: gate tools + utility tools."""
    gate_tools = _PHASE_TO_GATE.get(phase, [])
    return gate_tools + UTILITY_TOOLS


def _fix_injected_annotations() -> None:
    """
    Fix InjectedState/InjectedToolCallId detection for tools defined in files
    that use 'from __future__ import annotations'.

    With PEP 563 (future annotations), all annotations are stored as strings
    rather than evaluated types. LangChain's _injected_args_keys is a
    cached_property that uses inspect.signature() — which returns the raw string
    annotations — then checks _is_injected_arg_type(), which fails on strings.
    The result: _injected_args_keys = frozenset() and state defaults to None.

    LangGraph's _get_all_injected_args uses get_type_hints() which resolves
    strings correctly, so ToolNode *does* inject state into tool_call["args"].
    But _parse_input() strips extra args (via Pydantic validation against
    args_schema) then tries to add them back via _injected_args_keys — and
    since that's empty, state is lost before the function is called.

    Fix: overwrite func.__annotations__ with the fully-evaluated types from
    get_type_hints() before _injected_args_keys is ever accessed (it's a
    cached_property — first access wins).
    """
    all_groups = [CALIBRATION_TOOLS, REGISTRATION_TOOLS, ANALYSIS_TOOLS,
                  STACKING_TOOLS, LINEAR_TOOLS, STRETCH_TOOLS,
                  NONLINEAR_TOOLS, EXPORT_TOOLS, UTILITY_TOOLS]
    seen: set[str] = set()
    for group in all_groups:
        for t in group:
            if t.name in seen:
                continue
            seen.add(t.name)
            func = getattr(t, "func", None) or getattr(t, "coroutine", None)
            if func is None:
                continue
            try:
                t.func.__annotations__ = get_type_hints(func, include_extras=True)
            except Exception:
                pass


_fix_injected_annotations()


def all_tools() -> list:
    """Return every registered tool. Used for ToolNode initialization."""
    seen = set()
    tools = []
    for group in [CALIBRATION_TOOLS, REGISTRATION_TOOLS, ANALYSIS_TOOLS,
                  STACKING_TOOLS, LINEAR_TOOLS, STRETCH_TOOLS,
                  NONLINEAR_TOOLS, EXPORT_TOOLS, UTILITY_TOOLS]:
        for t in group:
            if t.name not in seen:
                seen.add(t.name)
                tools.append(t)
    return tools
