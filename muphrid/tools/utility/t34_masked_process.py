"""
T34 — masked_process

Compound tool: apply an image-modifying tool to a masked region of the
working image. Collapses the create_mask → tool → pixel_math sequence
into a single agent call while preserving every parameter the inner
tool exposes.

Architecture
------------
The orchestration is implemented as a LangGraph subgraph that runs
synchronously inside the @tool wrapper. Three nodes:

    build_mask  →  run_inner  →  blend  →  END

Each node calls the underlying tool's pure function (`tool.func(...)`)
with explicit state, so the inner work happens inline rather than
through the parent graph's action / hitl_check routing. As a side
effect, any HITL gate the inner tool would normally trigger is
suppressed — the compound tool is the unit the agent decided to
execute, and inner-tool messages do not flow into the parent message
stream. The compound tool itself can be HITL-mapped if a gate after
the whole operation is desired.

Supported inner tools
---------------------
Image-modifying tools whose Command update writes a single new
current_image FITS. See `_SUPPORTED_INNER_TOOLS`. Tools that produce
sibling outputs (star_removal's starless + star_mask, star_restoration)
are intentionally not supported under this wrapper — their use cases
involve their sibling files in ways masked blending doesn't capture.

Mask sources
------------
`mask` accepts either a path string (pre-built mask FITS) or an inline
`MaskSpec` mirroring `create_mask`'s parameters. The MaskSpec path
delegates to `create_mask.func` inside the build_mask node, collapsing
the mask-creation step into the same call.

Output
------
A blended FITS where masked pixels carry the inner-tool's result and
unmasked pixels carry the baseline. The result is promoted to
`paths.current_image`; the pre-call image is preserved as
`paths.previous_image`. Intermediate files (the raw inner-tool result,
the mask FITS) remain on disk and are inspectable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState

logger = logging.getLogger(__name__)


# ── Inner tool registry ───────────────────────────────────────────────────────
# Lazy imports inside a getter to avoid circular-import risk: every inner
# tool module imports from muphrid.graph.state, and this module is itself
# imported into the registry. Resolving on first call keeps the module
# graph clean.

_SUPPORTED_INNER_TOOL_NAMES = (
    "curves_adjust",
    "saturation_adjust",
    "local_contrast_enhance",
    "noise_reduction",
    "deconvolution",
    "stretch_image",
    "color_calibrate",
    "remove_gradient",
    "remove_green_noise",
)


def _resolve_inner_tool(name: str):
    """Look up an inner tool by name. Returns the @tool-decorated callable."""
    if name == "curves_adjust":
        from muphrid.tools.nonlinear.t16_curves import curves_adjust
        return curves_adjust
    if name == "saturation_adjust":
        from muphrid.tools.nonlinear.t18_saturation import saturation_adjust
        return saturation_adjust
    if name == "local_contrast_enhance":
        from muphrid.tools.nonlinear.t17_local_contrast import local_contrast_enhance
        return local_contrast_enhance
    if name == "noise_reduction":
        from muphrid.tools.linear.t12_noise_reduction import noise_reduction
        return noise_reduction
    if name == "deconvolution":
        from muphrid.tools.linear.t13_deconvolution import deconvolution
        return deconvolution
    if name == "stretch_image":
        from muphrid.tools.nonlinear.t14_stretch import stretch_image
        return stretch_image
    if name == "color_calibrate":
        from muphrid.tools.linear.t10_color_calibrate import color_calibrate
        return color_calibrate
    if name == "remove_gradient":
        from muphrid.tools.linear.t09_gradient import remove_gradient
        return remove_gradient
    if name == "remove_green_noise":
        from muphrid.tools.linear.t11_green_noise import remove_green_noise
        return remove_green_noise
    raise ValueError(
        f"masked_process: inner_tool '{name}' is not supported. "
        f"Valid: {', '.join(_SUPPORTED_INNER_TOOL_NAMES)}."
    )


# ── Pydantic input schemas ────────────────────────────────────────────────────

class MaskSpec(BaseModel):
    """
    Inline mask-creation parameters. Mirrors `create_mask`'s input shape.
    Only `mask_type` is strictly required; everything else has the same
    default as create_mask. See create_mask's docstring for the full
    semantics of each field — this wrapper passes them through unchanged.
    """
    mask_type: str = Field(
        description="One of: luminance, inverted_luminance, range, channel_diff."
    )
    luminance_options: dict | None = Field(
        default=None,
        description="LuminanceOptions kwargs. Used when mask_type is luminance or inverted_luminance.",
    )
    range_options: dict | None = Field(
        default=None,
        description="RangeOptions kwargs. Used when mask_type is range.",
    )
    channel_diff_options: dict | None = Field(
        default=None,
        description="ChannelDiffOptions kwargs. Used when mask_type is channel_diff.",
    )
    region_spec: dict | None = Field(
        default=None,
        description="RegionSpec kwargs (shape + coords) for spatial constraint.",
    )
    region_combine: str = Field(
        default="and",
        description="Combine mode for region + statistical mask: and / or / subtract.",
    )
    feather_radius: float = Field(default=5.0, description="Gaussian blur sigma applied to the final mask.")
    feather_truncate: float = Field(default=4.0, description="Gaussian truncation in standard deviations.")
    feather_mode: str = Field(default="nearest", description="Boundary handling for the Gaussian filter.")
    expand_px: int = Field(default=0, description="Morphological dilation footprint radius.")
    contract_px: int = Field(default=0, description="Morphological erosion footprint radius.")
    morphology_iterations: int = Field(default=1, description="Number of dilate+erode passes.")
    structuring_element: str = Field(default="disk", description="Shape of the morphological footprint.")
    luminance_model: str = Field(default="rec709", description="Luminance weighting (rec709 / equal / max).")
    invert: bool = Field(default=False, description="Invert the final mask after morphology + feathering.")


class MaskedProcessInput(BaseModel):
    inner_tool: str = Field(
        description=(
            "Name of the image-modifying tool to apply under the mask. "
            "Supported: curves_adjust, saturation_adjust, "
            "local_contrast_enhance, noise_reduction, deconvolution, "
            "stretch_image, color_calibrate, remove_gradient, "
            "remove_green_noise."
        ),
    )
    inner_params: dict = Field(
        default_factory=dict,
        description=(
            "Keyword arguments forwarded verbatim to the inner tool. The "
            "schema is the inner tool's args_schema — consult that tool's "
            "docstring for valid parameters. Empty dict invokes the inner "
            "tool with all-defaults."
        ),
    )
    mask: MaskSpec | str = Field(
        description=(
            "Mask source. Either a MaskSpec (inline mask creation: "
            "create_mask runs internally before the inner tool) or a path "
            "string to an existing mask FITS on disk."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description=(
            "Output FITS stem for the blended result. Defaults to "
            "'{baseline_stem}_masked_{inner_tool}'."
        ),
    )


# ── Subgraph nodes ────────────────────────────────────────────────────────────

def _mp_build_mask_node(state: AstroState) -> dict:
    """
    Produce the mask FITS path. If `mask` is already a string path, just
    record it in paths.latest_mask. If it's a MaskSpec dict, run create_mask
    inline and propagate the path it writes.
    """
    mp_inputs: dict = state.get("_mp_inputs", {})  # type: ignore[assignment]
    mask_input = mp_inputs.get("mask")

    if isinstance(mask_input, str):
        return {"paths": {"latest_mask": mask_input}}

    # mask_input is a dict (a MaskSpec model_dump) — call create_mask.func
    from muphrid.tools.scikit.t25_create_mask import create_mask

    cmd = create_mask.func(
        **{k: v for k, v in mask_input.items() if v is not None or k == "mask_type"},
        state=state,
        tool_call_id="_mp_build_mask",
    )
    # Propagate only path updates (latest_mask). Drop the inner ToolMessage.
    return {"paths": cmd.update.get("paths", state["paths"])}


def _mp_run_inner_node(state: AstroState) -> dict:
    """
    Invoke the inner tool with the agent's `inner_params`. Capture the
    pre-call current_image as the baseline so the blend node can
    reconstruct the unmasked source.
    """
    mp_inputs: dict = state.get("_mp_inputs", {})  # type: ignore[assignment]
    inner_tool_name = mp_inputs["inner_tool"]
    inner_params = mp_inputs.get("inner_params") or {}

    inner_tool = _resolve_inner_tool(inner_tool_name)
    baseline_path = state["paths"].get("current_image")

    cmd = inner_tool.func(
        **inner_params,
        state=state,
        tool_call_id="_mp_inner",
    )

    # Stash the baseline path for the blend node. Only path / metadata
    # updates flow forward — messages stay inside the subgraph.
    update: dict = {
        "_mp_inputs": {**mp_inputs, "_baseline_path": baseline_path},
    }
    if "paths" in cmd.update:
        update["paths"] = cmd.update["paths"]
    if "metadata" in cmd.update:
        update["metadata"] = cmd.update["metadata"]
    return update


def _mp_blend_node(state: AstroState) -> dict:
    """
    Blend the inner tool's output against the baseline using pixel_math:
        $baseline$ * (1 - $mask$) + $processed$ * $mask$
    The blended FITS becomes the new current_image; previous_image is
    set to the original (pre-call) current_image so a subsequent
    restore_checkpoint or rewind has the original to fall back to.
    """
    from muphrid.tools.utility.t23_pixel_math import pixel_math

    mp_inputs: dict = state.get("_mp_inputs", {})  # type: ignore[assignment]
    baseline_path: str = mp_inputs.get("_baseline_path")  # type: ignore[assignment]
    processed_path: str = state["paths"]["current_image"]
    mask_path: str | None = state["paths"].get("latest_mask")

    if mask_path is None:
        raise RuntimeError(
            "masked_process: latest_mask is missing after build_mask step "
            "— mask source did not produce a usable FITS path."
        )
    if baseline_path is None:
        raise RuntimeError(
            "masked_process: baseline path was not captured before the "
            "inner tool ran."
        )

    baseline_stem = Path(baseline_path).stem
    processed_stem = Path(processed_path).stem
    mask_stem = Path(mask_path).stem

    output_stem = mp_inputs.get("output_stem") or (
        f"{baseline_stem}_masked_{mp_inputs['inner_tool']}"
    )

    expression = (
        f"${baseline_stem}$ * (1 - ${mask_stem}$) + "
        f"${processed_stem}$ * ${mask_stem}$"
    )

    cmd = pixel_math.func(
        expression=expression,
        output_stem=output_stem,
        rescale=False,
        rescale_low=0.0,
        rescale_high=1.0,
        nosum=True,
        state=state,
        tool_call_id="_mp_blend",
    )

    # pixel_math sets current_image to its result and previous_image to
    # the pre-call current_image (which, at this point in the chain, is
    # the inner tool's processed file). Override previous_image with the
    # ORIGINAL baseline so the agent's view of "before this call" is the
    # pre-masked-process state, not the inner tool's intermediate.
    new_paths = dict(cmd.update.get("paths", {}))
    new_paths["previous_image"] = baseline_path
    return {"paths": new_paths}


# ── Subgraph compilation (module load) ───────────────────────────────────────

def _build_subgraph():
    g = StateGraph(AstroState)
    g.add_node("build_mask", _mp_build_mask_node)
    g.add_node("run_inner", _mp_run_inner_node)
    g.add_node("blend", _mp_blend_node)
    g.set_entry_point("build_mask")
    g.add_edge("build_mask", "run_inner")
    g.add_edge("run_inner", "blend")
    g.add_edge("blend", END)
    return g.compile()


_masked_process_subgraph = _build_subgraph()


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=MaskedProcessInput)
def masked_process(
    inner_tool: str,
    mask: MaskSpec | str,
    inner_params: dict | None = None,
    output_stem: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Apply an image-modifying tool to a masked region of the working image.

    Internally orchestrates: optional create_mask (when `mask` is a
    MaskSpec) → inner_tool (with inner_params) → pixel_math blend
    ($baseline * (1 - mask) + $processed * mask). Each step runs as a
    node in a compiled LangGraph subgraph; inner-tool ToolMessages do
    not flow into the parent message stream and any HITL gate the inner
    tool would normally trigger is suppressed. The compound tool itself
    can be HITL-mapped if a gate after the whole operation is desired.

    Supported inner tools: curves_adjust, saturation_adjust,
    local_contrast_enhance, noise_reduction, deconvolution,
    stretch_image, color_calibrate, remove_gradient, remove_green_noise.

    Post-condition:
      - paths.current_image is the blended result FITS
      - paths.previous_image is the pre-call current_image
      - paths.latest_mask is the mask FITS used (whether built inline
        or supplied as a path)
    """
    # Resolve the inner-tool name early for a clear error message before
    # any subgraph state is constructed.
    _resolve_inner_tool(inner_tool)

    if isinstance(mask, MaskSpec):
        mask_payload = mask.model_dump(exclude_none=True)
    elif isinstance(mask, dict):
        # Pydantic union resolution may already have produced a dict.
        mask_payload = {k: v for k, v in mask.items() if v is not None}
    else:
        mask_payload = mask  # str path

    enriched_state: dict = {
        **state,
        "_mp_inputs": {
            "inner_tool": inner_tool,
            "inner_params": inner_params or {},
            "mask": mask_payload,
            "output_stem": output_stem,
        },
    }

    final_state = _masked_process_subgraph.invoke(enriched_state)

    summary = {
        "inner_tool": inner_tool,
        "inner_params": inner_params or {},
        "mask_path": final_state["paths"].get("latest_mask"),
        "result_image_path": final_state["paths"].get("current_image"),
        "previous_image": final_state["paths"].get("previous_image"),
        "output_stem": output_stem,
    }

    logger.info(
        f"masked_process: inner_tool={inner_tool} → "
        f"{final_state['paths'].get('current_image')}"
    )

    update: dict = {
        "paths": final_state["paths"],
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    }
    # Propagate image_space from the inner subgraph: the inner tool wrote
    # its image_space into the subgraph state, and the blend (pixel_math)
    # preserved it. State authority demands we re-emit it as a delta on
    # the outer Command.update — without this, the outer state's
    # image_space would not reflect what the inner tool produced. Refuse
    # if the subgraph state's image_space is invalid (legacy/missing
    # writer). See Metadata.image_space.
    final_image_space = final_state.get("metadata", {}).get("image_space")
    if final_image_space not in ("linear", "display"):
        raise RuntimeError(
            "masked_process: inner subgraph produced no valid metadata.image_space "
            f"(got {final_image_space!r}). The inner tool should set image_space "
            "in its Command.update; this looks like a writer that skipped its "
            "bookkeeping."
        )
    metadata_delta: dict = {"image_space": final_image_space}
    # If the inner subgraph also flipped diagnostic metrics (e.g. inner
    # was stretch_image), surface that too so it survives at the parent.
    inner_metrics = final_state.get("metrics") or {}
    if "is_linear_estimate" in inner_metrics:
        update["metrics"] = {"is_linear_estimate": inner_metrics["is_linear_estimate"]}
    update["metadata"] = metadata_delta

    return Command(update=update)
