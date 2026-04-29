"""
T38 — hsv_adjust

Adjust an image in HSV color space via per-component curves and an optional
hue-window selector.

Conversion: skimage.color.rgb2hsv / hsv2rgb. Hue, saturation, and value are
each in [0, 1]; hue is cyclic.

Inputs:
  - hue_curve, saturation_curve, value_curve: optional control-point curves,
    each a list of [x, y] points in [0, 1]. At least one curve must be
    provided. Curves are fitted with scipy PchipInterpolator (monotone
    cubic, no oscillation). Hue curve output is wrapped modulo 1.
  - hue_range: optional [low, high] window in [0, 1]. Pixels whose ORIGINAL
    hue falls outside this window are not modified by any of the curves.
    If low > high, the window is treated as wrap-around (e.g., [0.9, 0.1]
    selects pixels in [0.9, 1.0] ∪ [0.0, 0.1]).
  - mask_path: optional FITS mask path. Where mask=1 the curves apply
    fully; where mask=0 the original pixel is preserved; intermediate
    values blend linearly per pixel.

Mono inputs are unsupported (HSV requires three channels).

Backend: pure Python — numpy + scipy + skimage.color. No Siril round-trip.
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
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState

logger = logging.getLogger(__name__)


# ── Pydantic input schema ──────────────────────────────────────────────────────

class HSVAdjustInput(BaseModel):
    hue_curve: list[list[float]] | None = Field(
        default=None,
        description=(
            "Curve applied to hue. List of [x, y] control points in [0, 1] "
            "(both axes), at least 2 points. Hue is cyclic, so the output "
            "is wrapped to [0, 1) — values that exceed 1 wrap to 0. "
            "PchipInterpolator fit, extrapolated linearly to the edges if "
            "x_min > 0 or x_max < 1."
        ),
    )
    saturation_curve: list[list[float]] | None = Field(
        default=None,
        description=(
            "Curve applied to saturation in [0, 1]. List of [x, y] "
            "control points, at least 2 points. Output clipped to [0, 1]."
        ),
    )
    value_curve: list[list[float]] | None = Field(
        default=None,
        description=(
            "Curve applied to value (brightness) in [0, 1]. List of [x, y] "
            "control points, at least 2 points. Output clipped to [0, 1]."
        ),
    )
    hue_range: list[float] | None = Field(
        default=None,
        description=(
            "Optional 2-element window [low, high] in [0, 1] selecting which "
            "pixels are modified, by their ORIGINAL hue. None = apply to "
            "all hues. If low > high, the window wraps around 1 → 0 "
            "(e.g., [0.9, 0.1] selects pixels with hue in [0.9, 1.0] ∪ "
            "[0.0, 0.1])."
        ),
    )
    mask_path: str | None = Field(
        default=None,
        description=(
            "Optional FITS mask path. Mask values in [0, 1] gate where the "
            "adjustment applies: out = mask * adjusted + (1 - mask) * "
            "original, per pixel. None = full-frame application "
            "(equivalent to mask=1 everywhere). Combines with hue_range "
            "via per-pixel multiplication."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description=(
            "Output FITS stem. Defaults to '{source_stem}_hsv'."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_curve(points: list[list[float]]):
    """
    Build a monotone-cubic interpolator from control points. Returns a
    callable that accepts an array of x values and returns the corresponding
    y values, clipped to [0, 1] on output.
    """
    from scipy.interpolate import PchipInterpolator

    if len(points) < 2:
        raise ValueError(
            f"hsv_adjust curve requires at least 2 control points; got {len(points)}"
        )
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"hsv_adjust curve points must be a list of [x, y] pairs; "
            f"got shape {pts.shape}"
        )
    pts = pts[np.argsort(pts[:, 0])]
    keep = np.concatenate([[True], np.diff(pts[:, 0]) > 0])
    pts = pts[keep]
    if pts.shape[0] < 2:
        raise ValueError(
            "hsv_adjust curve has fewer than 2 distinct x values after de-dup."
        )
    interp = PchipInterpolator(pts[:, 0], pts[:, 1], extrapolate=True)

    def apply(values):
        return np.clip(interp(values), 0.0, 1.0)

    return apply


def _hue_range_mask(
    hue_array: np.ndarray, low: float, high: float
) -> np.ndarray:
    """
    Build a boolean (H, W) mask of pixels whose hue falls in [low, high].
    Wrap-around when low > high: selects [low, 1.0] ∪ [0.0, high].
    """
    if low <= high:
        return (hue_array >= low) & (hue_array <= high)
    return (hue_array >= low) | (hue_array <= high)


def _load_rgb_float(image_path: Path) -> tuple[np.ndarray, float, str]:
    """
    Load an RGB FITS as (H, W, 3) float in [0, 1]. Returns (rgb_hwc, scale,
    layout) where layout ∈ {"chw", "hwc"} and scale is the divisor used to
    normalize (1.0 if data was already in [0, 1]).
    """
    with astropy_fits.open(str(image_path)) as hdul:
        data = hdul[0].data.astype(np.float32)

    data_max = float(data.max())
    scale = data_max if data_max > 1.0 else 1.0
    data_n = data / scale if scale != 1.0 else data

    if data_n.ndim == 3 and data_n.shape[0] == 3:
        return data_n.transpose(1, 2, 0), scale, "chw"
    if data_n.ndim == 3 and data_n.shape[2] == 3:
        return data_n, scale, "hwc"
    raise ValueError(
        f"hsv_adjust requires a 3-channel RGB FITS; got shape {data_n.shape}. "
        "Mono inputs are not supported by HSV-space adjustment."
    )


def _load_mask(mask_path: str, target_shape: tuple[int, int]) -> np.ndarray:
    """Load a mask FITS as (H, W) float clipped to [0, 1]."""
    with astropy_fits.open(mask_path) as hdul:
        m = hdul[0].data.astype(np.float32)
    if m.ndim == 3:
        # Reduce a 3-channel mask to a single plane
        m = m.max(axis=0) if m.shape[0] in (1, 3) else m[..., 0]
    if m.shape != target_shape:
        raise ValueError(
            f"hsv_adjust: mask shape {m.shape} does not match image shape "
            f"{target_shape}. Mask must be co-registered to the image."
        )
    return np.clip(m, 0.0, 1.0)


# ── Core adjustment ───────────────────────────────────────────────────────────

def _apply_hsv_adjust(
    image_path: Path,
    *,
    hue_curve: list[list[float]] | None,
    saturation_curve: list[list[float]] | None,
    value_curve: list[list[float]] | None,
    hue_range: list[float] | None,
    mask_path: str | None,
    output_stem: str,
    working_dir: str,
) -> Path:
    from skimage.color import hsv2rgb, rgb2hsv

    if hue_curve is None and saturation_curve is None and value_curve is None:
        raise ValueError(
            "hsv_adjust: at least one of hue_curve, saturation_curve, "
            "value_curve must be provided."
        )
    if hue_range is not None and len(hue_range) != 2:
        raise ValueError(
            f"hsv_adjust: hue_range must be a 2-element list [low, high]; "
            f"got {len(hue_range)} values."
        )

    rgb_hwc, scale, layout = _load_rgb_float(image_path)
    H, W, _ = rgb_hwc.shape

    hsv = rgb2hsv(rgb_hwc.clip(0.0, 1.0))
    h_orig = hsv[..., 0].copy()
    s_orig = hsv[..., 1].copy()
    v_orig = hsv[..., 2].copy()

    # Build per-component output starting from the originals; curves override
    # below where they apply.
    h_new, s_new, v_new = h_orig.copy(), s_orig.copy(), v_orig.copy()

    if hue_curve is not None:
        hue_fn = _build_curve(hue_curve)
        h_curved = hue_fn(h_orig)
        # Cyclic: keep [0, 1). _build_curve already clipped to [0, 1] which
        # means h_curved=1.0 wraps to 0.
        h_new = np.where(h_curved >= 1.0, 0.0, h_curved)

    if saturation_curve is not None:
        sat_fn = _build_curve(saturation_curve)
        s_new = sat_fn(s_orig)

    if value_curve is not None:
        val_fn = _build_curve(value_curve)
        v_new = val_fn(v_orig)

    # Build the gating field that combines hue_range + mask.
    gate = np.ones((H, W), dtype=np.float32)
    if hue_range is not None:
        in_range = _hue_range_mask(h_orig, float(hue_range[0]), float(hue_range[1]))
        gate = gate * in_range.astype(np.float32)
    if mask_path is not None:
        m = _load_mask(mask_path, (H, W))
        gate = gate * m

    # Per-component blend: gated curve output where gate is 1, original
    # where gate is 0, linear blend in between.
    if hue_curve is not None:
        # Cyclic blend: take the shorter angular distance between h_orig and
        # h_new before blending so we don't crossfade through the long way
        # around the hue wheel.
        delta = h_new - h_orig
        delta = np.where(delta > 0.5, delta - 1.0, delta)
        delta = np.where(delta < -0.5, delta + 1.0, delta)
        h_blended = h_orig + gate * delta
        h_blended = np.mod(h_blended, 1.0)
        hsv[..., 0] = h_blended
    if saturation_curve is not None:
        hsv[..., 1] = s_orig + gate * (s_new - s_orig)
    if value_curve is not None:
        hsv[..., 2] = v_orig + gate * (v_new - v_orig)

    rgb_out = hsv2rgb(hsv.clip(0.0, 1.0)).clip(0.0, 1.0)
    if scale != 1.0:
        rgb_out = rgb_out * scale

    # Restore original layout
    if layout == "chw":
        result = rgb_out.transpose(2, 0, 1)
    else:
        result = rgb_out

    output_path = Path(working_dir) / f"{output_stem}.fits"
    hdu = astropy_fits.PrimaryHDU(data=result.astype(np.float32))
    hdu.writeto(str(output_path), overwrite=True)
    return output_path


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=HSVAdjustInput)
def hsv_adjust(
    hue_curve: list[list[float]] | None = None,
    saturation_curve: list[list[float]] | None = None,
    value_curve: list[list[float]] | None = None,
    hue_range: list[float] | None = None,
    mask_path: str | None = None,
    output_stem: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Adjust the working image in HSV color space.

    The image is converted to HSV via skimage. Per-component curves (hue,
    saturation, value) are fitted with scipy PchipInterpolator and applied
    to the corresponding channels. The hue curve is cyclic (output wrapped
    modulo 1); saturation and value outputs are clipped to [0, 1].

    `hue_range` and `mask_path` together gate WHERE the curves apply. The
    gate is the per-pixel product of the hue-window membership and the
    mask. Where the gate is 1, the curved values fully replace the
    originals; where 0, the originals are preserved; intermediate values
    blend linearly. Hue blending uses the shorter angular distance to
    avoid crossfading the long way around the hue wheel.

    At least one curve must be provided. Mono inputs are not supported.
    Output is promoted to paths.current_image.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_stem = output_stem or f"{img_path.stem}_hsv"

    output_path = _apply_hsv_adjust(
        img_path,
        hue_curve=hue_curve,
        saturation_curve=saturation_curve,
        value_curve=value_curve,
        hue_range=hue_range,
        mask_path=mask_path,
        output_stem=out_stem,
        working_dir=working_dir,
    )

    summary: dict = {
        "output_path": str(output_path),
        "applied_components": [
            name for name, val in (
                ("hue_curve", hue_curve),
                ("saturation_curve", saturation_curve),
                ("value_curve", value_curve),
            )
            if val is not None
        ],
        "hue_range": hue_range,
        "mask_path": mask_path,
    }

    return Command(update={
        "paths": {"current_image": str(output_path)},
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    })
