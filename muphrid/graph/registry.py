"""
Tool registry — maps processing phases to their available tools.

The phase_router node reads `state["phase"]` and calls `tools_for_phase()` to
get the tool list for `model.bind_tools()`. Utility tools are included in
every phase gate automatically.

See graph_design.md §Phase Gates & Tool Groupings for the rationale.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

from langchain_core.tools.base import _is_injected_arg_type

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
from muphrid.tools.nonlinear.t38_hsv_adjust import hsv_adjust
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
from muphrid.tools.utility.t31_commit_variant import commit_variant
from muphrid.tools.utility.t32_present_images import present_images
from muphrid.tools.utility.t34_masked_process import masked_process
from muphrid.tools.utility.t35_hdr_composite import hdr_composite
from muphrid.tools.utility.t36_rewind_phase import rewind_phase
from muphrid.tools.utility.t37_flag_dataset_issue import flag_dataset_issue
from muphrid.tools.utility.t39_present_for_review import present_for_review


# ── Tool groups ───────────────────────────────────────────────────────────────

UTILITY_TOOLS = [
    analyze_image,
    plate_solve,
    pixel_math,
    extract_narrowband,
    resolve_target,
    advance_phase,
    rewind_phase,
    flag_dataset_issue,
    masked_process,
    hdr_composite,
    commit_variant,
    present_images,
    present_for_review,
    # analyze_frames and create_mask used to live in ANALYSIS and NONLINEAR
    # respectively, but they are phase-agnostic diagnostics/primitives:
    #   - analyze_frames reads the registration cache; agents reach for it
    #     during REGISTRATION to check register quality, not just ANALYSIS.
    #   - create_mask is a pure companion to pixel_math (which is already a
    #     utility). Masked blending is legitimate in STRETCH (HDR masked
    #     stretch) and LINEAR (masked gradient removal), not just NONLINEAR.
    # Moving them here removes phase-gate friction for workflows a human
    # would reach for without thinking about phases. See v2_framework_fixes
    # Issue #3b / #3c.
    analyze_frames,
    create_mask,
]


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

ANALYSIS_TOOLS: list = [
    # analyze_frames moved to UTILITY_TOOLS — see that list for rationale.
    # The ANALYSIS phase still exists as a distinct stop in the preprocessing
    # order (between REGISTRATION and STACKING); it just relies on utility
    # tools for its work.
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
    save_checkpoint,
    restore_checkpoint,
]

STRETCH_TOOLS = [
    stretch_image,
    select_stretch_variant,
    save_checkpoint,
    restore_checkpoint,
]

NONLINEAR_TOOLS = [
    star_removal,
    curves_adjust,
    local_contrast_enhance,
    saturation_adjust,
    hsv_adjust,
    star_restoration,
    # create_mask moved to UTILITY_TOOLS — see that list for rationale.
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


def _assert_no_schema_drift() -> None:
    """
    Hard-fail at module import if any registered tool has schema/function drift.

    The args_schema and the wrapped function signature MUST agree on every
    non-injected parameter. When they don't:

      - Pydantic validates incoming LLM args against the schema, filling in
        defaults for every schema field (even ones the function doesn't
        accept).
      - LangChain then calls `func(**validated_input)`, and Python raises
        `TypeError: <name>() got an unexpected keyword argument 'X'`.
      - The agent sees a non-actionable error mentioning a parameter it
        didn't pass, retries with the same args, hits the stuck-loop
        detector, and the run aborts.

    The bug is invisible to the agent and to ordinary code review (the schema
    looks reasonable; the function looks reasonable; only the comparison
    reveals the mismatch). Catching it on import means the system can't even
    start with a broken tool registered — the worst-case outcome shifts from
    "stuck loop in production" to "loud ImportError before the agent boots."

    Symmetrically, if the function declares a non-injected parameter that
    isn't in the schema, the LLM has no way to set it — also a bug, also
    surfaced here.

    Injected params (state, tool_call_id, config, runtime) are excluded from
    both sides since they are filled by the framework, not the LLM.
    """
    drifts: list[str] = []
    seen: set[str] = set()
    injected_arg_names = {"state", "tool_call_id", "config", "runtime"}
    for group in [CALIBRATION_TOOLS, REGISTRATION_TOOLS, ANALYSIS_TOOLS,
                   STACKING_TOOLS, LINEAR_TOOLS, STRETCH_TOOLS,
                   NONLINEAR_TOOLS, EXPORT_TOOLS, UTILITY_TOOLS]:
        for t in group:
            if t.name in seen:
                continue
            seen.add(t.name)

            schema = getattr(t, "args_schema", None)
            if schema is None or not inspect.isclass(schema):
                continue
            schema_fields = set(getattr(schema, "model_fields", {}).keys())

            func = getattr(t, "func", None) or getattr(t, "coroutine", None)
            if func is None:
                continue
            try:
                hints = get_type_hints(func, include_extras=True)
            except Exception:
                hints = {}
            sig = inspect.signature(func)
            func_params: set[str] = set()
            for name, param in sig.parameters.items():
                if name in injected_arg_names:
                    continue
                # Also skip params explicitly typed as injected via Annotated.
                ann = hints.get(name, param.annotation)
                if _is_injected_arg_type(ann):
                    continue
                func_params.add(name)

            schema_only = schema_fields - func_params
            func_only = func_params - schema_fields
            if schema_only or func_only:
                drifts.append(
                    f"  - {t.name}: schema-only={sorted(schema_only) or '∅'}, "
                    f"func-only={sorted(func_only) or '∅'}"
                )

    if drifts:
        msg = (
            "Schema/function drift detected in registered tools. "
            "Every @tool's args_schema fields must match its function "
            "signature parameters (excluding injected args). When they "
            "diverge, Pydantic fills in defaults for schema-only fields "
            "and the function call raises TypeError — invisible to the "
            "agent, which retries until the stuck-loop detector aborts. "
            "Either add the missing parameter to the function signature "
            "or remove the field from the schema.\n\nDrifting tools:\n"
            + "\n".join(drifts)
        )
        raise ImportError(msg)


_assert_no_schema_drift()


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
