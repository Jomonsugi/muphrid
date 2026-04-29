"""
T35 — hdr_composite

Combine two stretched versions of the same image into a single composite,
gated by a mask. Typical use is pairing a gentle stretch (preserves faint
structure across the broad image) with an aggressive stretch (compresses
bright cores) and letting the mask pick which contribution dominates
where.

The compound tool runs both stretches from the SAME pre-stretch baseline
(state.paths.current_image at call time), so the inputs to the blend are
two genuinely independent stretch results, not chained applications.

Architecture
------------
Implemented as a LangGraph subgraph (build_mask → stretch_base →
stretch_core → blend → END) compiled at module load. Each node calls
the underlying tool's pure function via `.func` with explicit state.
Inner-tool ToolMessages do not flow into the parent message stream;
HITL gates that the inner tools would normally trigger are suppressed.
The compound tool itself can be HITL-mapped if a gate after the whole
operation is desired.

Blend modes
-----------
linear:
    mask-weighted average per channel:
        out = base * (1 - mask) + core * mask
    Implemented via pixel_math (Siril). Same blend mathematically as
    masked_process — included here so the agent can choose hdr_composite
    as the unit of work and pick this mode when chromatic separation
    isn't desired.

luminosity:
    Convert both stretched results to CIE L*a*b*. Replace the L channel
    under the mask: L_out = L_base * (1 - mask) + L_core * mask. Keep
    a* and b* from the BASE stretch unchanged. Convert back to RGB.
    Preserves base's color rendering everywhere while taking core's
    tonal structure where the mask is high. Mono inputs fall back to
    linear blend (no chromatic information to preserve).

Future blend modes (not in v1):
    masked_multiscale — wavelet-based multi-scale blend that takes
    low-frequency content from one input and high-frequency content
    from the other, mask-gated per layer. Adds the algorithmic value
    of selectively combining detail at different spatial scales.

Output
------
The composite FITS becomes paths.current_image. The pre-call image is
preserved as paths.previous_image. The mask FITS used and both stretch
intermediates remain on disk and are inspectable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
from astropy.io import fits as astropy_fits
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState

logger = logging.getLogger(__name__)


# ── Pydantic input schemas ────────────────────────────────────────────────────

class StretchPass(BaseModel):
    """One stretch invocation. Mirrors stretch_image's input shape."""
    method: str = Field(
        default="ghs",
        description="Stretch method: 'ghs', 'asinh', or 'autostretch'.",
    )
    ghs_options: dict | None = Field(
        default=None,
        description=(
            "GHSOptions kwargs (stretch_amount, local_intensity, "
            "symmetry_point, shadow_protection, highlight_protection, "
            "color_model, channels, clip_mode). Required when method='ghs'."
        ),
    )
    asinh_options: dict | None = Field(
        default=None,
        description="AsinhOptions kwargs. Used when method='asinh'.",
    )
    autostretch_options: dict | None = Field(
        default=None,
        description="AutostretchOptions kwargs. Used when method='autostretch'.",
    )


class HDRMaskSpec(BaseModel):
    """
    Inline mask-creation parameters for hdr_composite. Same shape as the
    MaskSpec used by masked_process — see create_mask for full field
    semantics. Either supply this OR a path string for `mask`.
    """
    mask_type: str = Field(
        description="One of: luminance, inverted_luminance, range, channel_diff."
    )
    luminance_options: dict | None = Field(default=None)
    range_options: dict | None = Field(default=None)
    channel_diff_options: dict | None = Field(default=None)
    region_spec: dict | None = Field(default=None)
    region_combine: str = Field(default="and")
    feather_radius: float = Field(default=5.0)
    feather_truncate: float = Field(default=4.0)
    feather_mode: str = Field(default="nearest")
    expand_px: int = Field(default=0)
    contract_px: int = Field(default=0)
    morphology_iterations: int = Field(default=1)
    structuring_element: str = Field(default="disk")
    luminance_model: str = Field(default="rec709")
    invert: bool = Field(default=False)


class HDRCompositeInput(BaseModel):
    base_stretch: StretchPass = Field(
        description=(
            "Stretch parameters for the 'base' pass — typically the "
            "gentler stretch applied across the whole image. The mask "
            "selects this contribution where mask values are LOW."
        ),
    )
    core_stretch: StretchPass = Field(
        description=(
            "Stretch parameters for the 'core' pass — typically the "
            "more aggressive stretch. The mask selects this contribution "
            "where mask values are HIGH."
        ),
    )
    mask: HDRMaskSpec | str = Field(
        description=(
            "Mask source for blending. Either an HDRMaskSpec (inline mask "
            "creation, runs create_mask internally before blending) or a "
            "FITS path string pointing to an existing mask on disk. The "
            "mask is built FROM the pre-stretch baseline image, not from "
            "either stretch result."
        ),
    )
    blend_mode: str = Field(
        default="linear",
        description=(
            "How base and core are combined under the mask. "
            "'linear': per-channel mask-weighted average "
            "(out = base*(1-mask) + core*mask). "
            "'luminosity': replace L channel under mask in CIE L*a*b*; "
            "preserve a*b* from base. Mono inputs fall back to linear."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description=(
            "Output FITS stem for the composite. Defaults to "
            "'{baseline_stem}_hdr'."
        ),
    )


# ── Subgraph nodes ────────────────────────────────────────────────────────────

def _hc_build_mask_node(state: AstroState) -> dict:
    """
    Build the mask from the pre-stretch baseline (linear) image. The
    baseline is read from `_hc_inputs._baseline_path` (captured by the
    @tool wrapper before subgraph invocation), so the mask reflects the
    image state PRIOR to either stretch — not either stretch result.
    """
    inputs: dict = state.get("_hc_inputs", {})  # type: ignore[assignment]
    mask_input = inputs["mask"]

    if isinstance(mask_input, str):
        return {"paths": {"latest_mask": mask_input}}

    from muphrid.tools.scikit.t25_create_mask import create_mask

    # Run create_mask against the pre-stretch baseline. We force the
    # synthetic state's current_image to baseline so create_mask reads
    # the right file regardless of any prior path mutations.
    baseline = inputs["_baseline_path"]
    # Build a synthetic state for direct tool-function invocation.
    # The full paths dict must be preserved here — the inner tool reads
    # state["paths"] directly without going through the deep-merge
    # reducer, so this is NOT a Command.update payload and the
    # delta-only rule does not apply. Override only current_image so
    # the inner tool processes the baseline (not whatever current_image
    # happened to be when the subgraph started).
    synth = {**state, "paths": {**state["paths"], "current_image": baseline}}
    cmd = create_mask.func(
        **{k: v for k, v in mask_input.items() if v is not None or k == "mask_type"},
        state=synth,
        tool_call_id="_hc_build_mask",
    )
    return {"paths": cmd.update.get("paths", state["paths"])}


def _hc_run_stretch(state: AstroState, pass_key: str, output_suffix: str) -> str:
    """
    Run stretch_image on the pre-stretch baseline with the parameters
    for either the base or core pass. Returns the path to the stretched
    FITS. Does NOT propagate paths.current_image — the subgraph only
    needs the captured path, and we want the next stretch pass to read
    the same baseline.
    """
    from muphrid.tools.nonlinear.t14_stretch import (
        AsinhOptions,
        AutostretchOptions,
        GHSOptions,
        stretch_image,
    )

    inputs: dict = state.get("_hc_inputs", {})  # type: ignore[assignment]
    baseline: str = inputs["_baseline_path"]
    pass_params: dict = inputs[pass_key]

    method = pass_params.get("method", "ghs")
    ghs = pass_params.get("ghs_options") or None
    asinh = pass_params.get("asinh_options") or None
    autostretch = pass_params.get("autostretch_options") or None

    ghs_obj = GHSOptions(**ghs) if isinstance(ghs, dict) else None
    asinh_obj = AsinhOptions(**asinh) if isinstance(asinh, dict) else AsinhOptions()
    autostretch_obj = (
        AutostretchOptions(**autostretch) if isinstance(autostretch, dict)
        else AutostretchOptions()
    )

    # Build a synthetic state for direct tool-function invocation.
    # The full paths dict must be preserved here — the inner tool reads
    # state["paths"] directly without going through the deep-merge
    # reducer, so this is NOT a Command.update payload and the
    # delta-only rule does not apply. Override only current_image so
    # the inner tool processes the baseline (not whatever current_image
    # happened to be when the subgraph started).
    synth = {**state, "paths": {**state["paths"], "current_image": baseline}}
    cmd = stretch_image.func(
        method=method,
        ghs_options=ghs_obj,
        asinh_options=asinh_obj,
        autostretch_options=autostretch_obj,
        output_suffix=output_suffix,
        state=synth,
        tool_call_id=f"_hc_stretch_{output_suffix}",
    )
    return cmd.update["paths"]["current_image"]


def _hc_stretch_base_node(state: AstroState) -> dict:
    base_path = _hc_run_stretch(state, "base_stretch", "hdr_base")
    inputs: dict = state.get("_hc_inputs", {})  # type: ignore[assignment]
    return {"_hc_inputs": {**inputs, "_base_path": base_path}}


def _hc_stretch_core_node(state: AstroState) -> dict:
    core_path = _hc_run_stretch(state, "core_stretch", "hdr_core")
    inputs: dict = state.get("_hc_inputs", {})  # type: ignore[assignment]
    return {"_hc_inputs": {**inputs, "_core_path": core_path}}


# ── Blend implementations ─────────────────────────────────────────────────────

def _read_fits_float32(path: str) -> np.ndarray:
    """Read a FITS file as float32 in the array shape it was written with."""
    with astropy_fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    if data.max() > 1.0:
        data = data / data.max()
    return data


def _write_fits(path: str, data: np.ndarray) -> None:
    """Write a float32 array as a single-HDU FITS."""
    hdu = astropy_fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def _to_hwc_rgb(arr: np.ndarray) -> np.ndarray:
    """Normalize various FITS layouts to (H, W, 3) for skimage.color."""
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr.transpose(1, 2, 0)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(
        f"_to_hwc_rgb: expected 3-channel array, got shape {arr.shape}"
    )


def _from_hwc_rgb(arr: np.ndarray, original_layout_chw: bool) -> np.ndarray:
    """Reverse of _to_hwc_rgb, matching the input layout."""
    if original_layout_chw:
        return arr.transpose(2, 0, 1)
    return arr


def _blend_linear(
    base: np.ndarray, core: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Per-channel mask-weighted average. Works for mono and RGB."""
    if base.ndim == 3 and base.shape[0] in (1, 3) and mask.ndim == 2:
        # Broadcast mask across leading channel axis
        m = mask[np.newaxis, ...]
        return base * (1.0 - m) + core * m
    return base * (1.0 - mask) + core * mask


def _blend_luminosity(
    base: np.ndarray, core: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Replace L channel under mask in CIE L*a*b*; preserve a*b* from base.
    Mono inputs fall back to linear blend.
    """
    from skimage.color import lab2rgb, rgb2lab

    is_color = (
        (base.ndim == 3 and base.shape[0] == 3)
        or (base.ndim == 3 and base.shape[2] == 3)
    )
    if not is_color:
        return _blend_linear(base, core, mask)

    chw = base.shape[0] == 3 and base.ndim == 3
    base_hwc = _to_hwc_rgb(base).clip(0.0, 1.0)
    core_hwc = _to_hwc_rgb(core).clip(0.0, 1.0)

    base_lab = rgb2lab(base_hwc)
    core_lab = rgb2lab(core_hwc)

    # L channel: mask-weighted between base and core
    out_lab = base_lab.copy()
    out_lab[..., 0] = base_lab[..., 0] * (1.0 - mask) + core_lab[..., 0] * mask

    out_rgb = lab2rgb(out_lab).clip(0.0, 1.0).astype(np.float32)
    return _from_hwc_rgb(out_rgb, original_layout_chw=chw)


def _blend_python(
    base_path: str,
    core_path: str,
    mask_path: str,
    output_path: str,
    blend_mode: str,
) -> None:
    """Execute the blend as a numpy/skimage pipeline (used for luminosity)."""
    base = _read_fits_float32(base_path)
    core = _read_fits_float32(core_path)
    mask = _read_fits_float32(mask_path)

    if mask.ndim == 3:
        # Reduce a 3-channel mask to a single plane (any channel set = mask)
        mask = mask.max(axis=0) if mask.shape[0] in (1, 3) else mask[..., 0]
    mask = np.clip(mask, 0.0, 1.0)

    if blend_mode == "luminosity":
        out = _blend_luminosity(base, core, mask)
    else:
        out = _blend_linear(base, core, mask)

    _write_fits(output_path, np.clip(out, 0.0, 1.0))


def _hc_blend_node(state: AstroState) -> dict:
    """
    Combine base and core per the requested blend_mode, write the
    composite FITS, and update paths.current_image. previous_image is
    set to the original pre-call current_image so a subsequent
    restore_checkpoint or rewind has the original baseline to fall
    back to.
    """
    inputs: dict = state.get("_hc_inputs", {})  # type: ignore[assignment]
    base_path: str = inputs["_base_path"]
    core_path: str = inputs["_core_path"]
    baseline_path: str = inputs["_baseline_path"]
    blend_mode: str = inputs.get("blend_mode", "linear")
    mask_path: str | None = state["paths"].get("latest_mask")

    if mask_path is None:
        raise RuntimeError(
            "hdr_composite: latest_mask is missing after build_mask step."
        )

    working_dir = state["dataset"]["working_dir"]
    output_stem = inputs.get("output_stem") or (
        f"{Path(baseline_path).stem}_hdr"
    )
    output_path = str(Path(working_dir) / f"{output_stem}.fits")

    if blend_mode == "linear":
        # Use Siril pixel_math for color-correct blending in linear mode —
        # matches the masked_process linear blend exactly.
        from muphrid.tools.utility.t23_pixel_math import pixel_math

        base_stem = Path(base_path).stem
        core_stem = Path(core_path).stem
        mask_stem = Path(mask_path).stem
        expression = (
            f"${base_stem}$ * (1 - ${mask_stem}$) + "
            f"${core_stem}$ * ${mask_stem}$"
        )
        cmd = pixel_math.func(
            expression=expression,
            output_stem=output_stem,
            rescale=False,
            rescale_low=0.0,
            rescale_high=1.0,
            nosum=True,
            state=state,
            tool_call_id="_hc_blend",
        )
        new_paths = dict(cmd.update.get("paths", {}))
    elif blend_mode == "luminosity":
        _blend_python(
            base_path=base_path,
            core_path=core_path,
            mask_path=mask_path,
            output_path=output_path,
            blend_mode="luminosity",
        )
        new_paths = {"current_image": output_path}
    else:
        raise ValueError(
            f"hdr_composite: unknown blend_mode '{blend_mode}'. "
            f"Valid: linear, luminosity."
        )

    new_paths["previous_image"] = baseline_path
    return {"paths": new_paths}


# ── Subgraph compilation (module load) ───────────────────────────────────────

def _build_subgraph():
    g = StateGraph(AstroState)
    g.add_node("build_mask", _hc_build_mask_node)
    g.add_node("stretch_base", _hc_stretch_base_node)
    g.add_node("stretch_core", _hc_stretch_core_node)
    g.add_node("blend", _hc_blend_node)
    g.set_entry_point("build_mask")
    g.add_edge("build_mask", "stretch_base")
    g.add_edge("stretch_base", "stretch_core")
    g.add_edge("stretch_core", "blend")
    g.add_edge("blend", END)
    return g.compile()


_hdr_composite_subgraph = _build_subgraph()


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=HDRCompositeInput)
def hdr_composite(
    base_stretch: StretchPass,
    core_stretch: StretchPass,
    mask: HDRMaskSpec | str,
    blend_mode: str = "linear",
    output_stem: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Combine two stretched versions of the working image into a single
    composite, gated by a mask.

    Both stretches run from the SAME pre-call baseline image, so neither
    pass operates on the other's output. Internally:
      build_mask  →  stretch_base  →  stretch_core  →  blend  →  END

    Each step runs as a node in a compiled LangGraph subgraph; inner-tool
    ToolMessages do not flow into the parent message stream and any HITL
    gates the inner tools would normally trigger are suppressed. The
    compound tool itself can be HITL-mapped if a gate after the whole
    operation is desired.

    Blend modes:
      linear     — per-channel mask-weighted average
                   (out = base*(1-mask) + core*mask)
      luminosity — replace L channel under mask in CIE L*a*b*;
                   preserve a*b* from base (mono inputs use linear)

    Post-condition:
      paths.current_image  := composite FITS
      paths.previous_image := pre-call current_image (the baseline)
      paths.latest_mask    := mask FITS used (built inline or supplied)
    """
    if blend_mode not in ("linear", "luminosity"):
        raise ValueError(
            f"hdr_composite: unknown blend_mode '{blend_mode}'. "
            f"Valid: linear, luminosity."
        )

    # Pydantic union may have produced a dict or the typed model — normalize.
    if isinstance(mask, HDRMaskSpec):
        mask_payload = mask.model_dump(exclude_none=True)
    elif isinstance(mask, dict):
        mask_payload = {k: v for k, v in mask.items() if v is not None}
    else:
        mask_payload = mask  # str path

    enriched_state: dict = {
        **state,
        "_hc_inputs": {
            "base_stretch": base_stretch.model_dump(exclude_none=True),
            "core_stretch": core_stretch.model_dump(exclude_none=True),
            "mask": mask_payload,
            "blend_mode": blend_mode,
            "output_stem": output_stem,
            "_baseline_path": state["paths"]["current_image"],
        },
    }

    final_state = _hdr_composite_subgraph.invoke(enriched_state)

    summary = {
        "blend_mode": blend_mode,
        "base_stretch": base_stretch.model_dump(exclude_none=True),
        "core_stretch": core_stretch.model_dump(exclude_none=True),
        "mask_path": final_state["paths"].get("latest_mask"),
        "base_intermediate_path": final_state.get("_hc_inputs", {}).get("_base_path"),
        "core_intermediate_path": final_state.get("_hc_inputs", {}).get("_core_path"),
        "result_image_path": final_state["paths"].get("current_image"),
        "previous_image": final_state["paths"].get("previous_image"),
        "output_stem": output_stem,
    }

    logger.info(
        f"hdr_composite: blend_mode={blend_mode} → "
        f"{final_state['paths'].get('current_image')}"
    )

    update: dict = {
        "paths": final_state["paths"],
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    }
    # Stretch passes flip is_linear via metadata. Propagate.
    if "metadata" in final_state and final_state["metadata"] is not None:
        update["metadata"] = final_state["metadata"]

    return Command(update=update)
