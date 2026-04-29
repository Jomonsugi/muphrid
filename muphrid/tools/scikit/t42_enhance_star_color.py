"""
T42 — enhance_star_color

Boost the saturation of the star contribution and recombine with the
starless image. Reads paths.starless_image and paths.star_mask from
state, multiplies the mask's HSV saturation by `saturation_multiplier`,
then emits `starless + saturated_mask` as a new current_image.

Optional `confine_to_region_mask` reads paths.latest_mask and applies
the saturation boost only inside the region; outside, the original
mask is restored.

Backend: pure numpy + Astropy + skimage.color. No Siril.

Note on composition: this tool produces a new current_image directly.
It does not modify paths.star_mask on disk — a subsequent call to
selective_star_reblend reads the original mask and re-detects on it,
independent of any prior color-boost. To get tier-suppressed AND
color-boosted output, run enhance_star_color first (its output is the
boosted full-restoration), then run selective_star_reblend (which will
build its own weight map; its output replaces enhance_star_color's
current_image).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
from astropy.io import fits as astropy_fits
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from skimage.color import hsv2rgb, rgb2hsv

from muphrid.graph.state import AstroState


# ── Pydantic input schema ──────────────────────────────────────────────────────


class EnhanceStarColorInput(BaseModel):
    saturation_multiplier: float = Field(
        default=1.3,
        description=(
            "HSV saturation multiplier applied to the star mask. "
            "1.0 = neutral (no-op). >1 boosts star colors; <1 desaturates. "
            "2.0 doubles saturation (clipped to [0,1] in HSV)."
        ),
    )
    confine_to_region_mask: bool = Field(
        default=False,
        description=(
            "When true, the saturation boost applies only where "
            "paths.latest_mask > 0.5; outside the region, the original "
            "mask values are kept. Requires create_mask first."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description="Output FITS stem. Defaults to '{starless_stem}_color_boosted'.",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_fits_color(image_path: Path) -> tuple[np.ndarray, bool]:
    """Load FITS, return ((C,H,W) float32, is_color)."""
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)
    if data.ndim == 3 and data.shape[0] == 3:
        return data, True
    if data.ndim == 3 and data.shape[2] == 3:
        return np.moveaxis(data, -1, 0), True
    return data.squeeze()[np.newaxis, :, :], False


def _saturate_chw(chw: np.ndarray, multiplier: float) -> np.ndarray:
    """Multiply HSV saturation by multiplier; input (3,H,W) float in [0,1]."""
    hwc = np.moveaxis(chw, 0, -1)
    hwc = np.clip(hwc, 0.0, 1.0)
    hsv = rgb2hsv(hwc)
    hsv[..., 1] = np.clip(hsv[..., 1] * multiplier, 0.0, 1.0)
    rgb = hsv2rgb(hsv)
    return np.moveaxis(rgb, -1, 0).astype(np.float32)


# ── LangChain tool ─────────────────────────────────────────────────────────────


@tool(args_schema=EnhanceStarColorInput)
def enhance_star_color(
    saturation_multiplier: float = 1.3,
    confine_to_region_mask: bool = False,
    output_stem: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Boost the saturation of the star mask and recombine with the starless.

    Multiplies the star mask's HSV saturation by `saturation_multiplier`,
    then emits `starless + saturated_mask` as the new current_image. With
    `confine_to_region_mask`, the boost applies only inside paths.latest_mask;
    outside, the original mask values are preserved.

    Reads paths.starless_image and paths.star_mask from state — call
    star_removal first. Refuses on mono starless or mono mask: saturating
    a single-channel array is meaningless. paths.star_mask on disk is
    not modified; the boost is applied transiently in the recombination.
    """
    working_dir = state["dataset"]["working_dir"]
    starless_p = state["paths"].get("starless_image")
    mask_p = state["paths"].get("star_mask")
    current_p = state["paths"].get("current_image")

    if not starless_p or not Path(starless_p).exists():
        raise FileNotFoundError(
            "enhance_star_color: paths.starless_image is missing or the "
            "file does not exist on disk. Call star_removal first."
        )
    if not mask_p or not Path(mask_p).exists():
        raise FileNotFoundError(
            "enhance_star_color: paths.star_mask is missing or the file "
            "does not exist on disk. Call star_removal first."
        )

    incoming_image_space = state["metadata"].get("image_space")
    if incoming_image_space not in ("linear", "display"):
        raise RuntimeError(
            "enhance_star_color: state.metadata.image_space is missing or "
            f"invalid (got {incoming_image_space!r}). Refusing to guess — "
            "restart from a fresh checkpoint."
        )

    starless, starless_is_color = _load_fits_color(Path(starless_p))
    mask, mask_is_color = _load_fits_color(Path(mask_p))
    if starless.max() > 1.0:
        starless = starless / starless.max()
    if mask.max() > 1.0:
        mask = mask / mask.max()

    if not (starless_is_color and mask_is_color):
        raise ValueError(
            "enhance_star_color requires color (3-channel) starless and "
            "star_mask. Saturating a single-channel array is meaningless. "
            "Trace back to star_removal — StarNet must run on a color input."
        )
    if mask.shape[1:] != starless.shape[1:]:
        raise ValueError(
            f"starless ({starless.shape[1:]}) and star_mask ({mask.shape[1:]}) "
            "shapes disagree."
        )

    region_2d: np.ndarray | None = None
    if confine_to_region_mask:
        region_p = state["paths"].get("latest_mask")
        if not region_p or not Path(region_p).exists():
            raise FileNotFoundError(
                "enhance_star_color: confine_to_region_mask=True but "
                "paths.latest_mask is missing. Call create_mask first."
            )
        with astropy_fits.open(region_p) as hdul:
            rm = hdul[0].data.astype(np.float32)
        if rm.max() > 1.0:
            rm = rm / rm.max()
        rm = np.squeeze(rm)
        if rm.ndim == 3:
            rm = rm.max(axis=0) if rm.shape[0] == 3 else rm.max(axis=2)
        if rm.shape != starless.shape[1:]:
            raise ValueError(
                f"region mask ({rm.shape}) and image ({starless.shape[1:]}) "
                "shapes disagree."
            )
        region_2d = (rm > 0.5).astype(np.float32)

    saturated = _saturate_chw(mask, saturation_multiplier)

    if region_2d is not None:
        # Inside region: saturated. Outside: original mask.
        out_mask = (
            saturated * region_2d[np.newaxis, :, :]
            + mask * (1.0 - region_2d[np.newaxis, :, :])
        )
    else:
        out_mask = saturated

    result = np.clip(starless + out_mask, 0.0, 1.0).astype(np.float32)

    starless_stem = Path(starless_p).stem
    out_stem = output_stem or f"{starless_stem}_color_boosted"
    out_path = Path(working_dir) / f"{out_stem}.fits"
    astropy_fits.HDUList(
        [astropy_fits.PrimaryHDU(data=result)]
    ).writeto(out_path, overwrite=True)

    payload = {
        "output_path": str(out_path),
        "saturation_multiplier": saturation_multiplier,
        "confine_to_region_mask": confine_to_region_mask,
        "image_space": incoming_image_space,
    }

    prev_current = current_p
    return Command(update={
        "paths": {
            "current_image": str(out_path),
            "previous_image": prev_current,
        },
        "metadata": {"image_space": incoming_image_space},
        "messages": [ToolMessage(
            content=json.dumps(payload, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    })
