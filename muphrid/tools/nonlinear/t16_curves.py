"""
T16 — curves_adjust

Apply tonal curves to the working image. Three methods:

  - mtf    : Siril Midtone Transfer Function. Three-point curve defined by
             black_point / midtone / white_point.
  - ght    : Siril Generalized Hyperbolic Stretch. Parametric curve with
             D / B / SP / LP / HP controls.
  - points : Arbitrary-points curve. Per-channel curves built from agent-
             provided control points via monotone-cubic interpolation
             (scipy PchipInterpolator). Supports rgb / R / G / B / L_star
             keys in a single call; each curve is applied in a fixed order
             so chained shaping happens within one tool invocation. Pure
             Python — no Siril round-trip.

Backend: Siril CLI for mtf and ght; numpy + scipy + skimage for points.
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

from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script

# ── Pydantic input schema ──────────────────────────────────────────────────────

class MTFOptions(BaseModel):
    black_point: float = Field(
        default=0.0,
        description=(
            "Shadow clipping point (0–1). 0.0 = no black-point shift. "
            "Raising it lifts the black level."
        ),
    )
    midtone: float = Field(
        default=0.5,
        description=(
            "Midtone transfer point (0–1). 0.5 = linear (no change). "
            "< 0.5 brightens midtones, > 0.5 darkens midtones."
        ),
    )
    white_point: float = Field(
        default=1.0,
        description=(
            "Highlight clipping point (0–1). 1.0 = no highlight clip. "
            "Lower values clip progressively more of the bright end."
        ),
    )
    channels: str = Field(
        default="all",
        description=(
            "Which channels receive the adjustment. "
            "all: all channels with the same parameters. "
            "R, G, B: single channel. RG, RB, GB: channel pairs."
        ),
    )


class GHTCurvesOptions(BaseModel):
    stretch_amount: float = Field(
        description=(
            "D: stretch strength (0–10). Primary control."
        ),
    )
    local_intensity: float = Field(
        default=5.0,
        description=(
            "B: tonal focus (-5 to 15). Higher values tightly focus the "
            "adjustment around symmetry_point."
        ),
    )
    symmetry_point: float = Field(
        default=0.3,
        description=(
            "SP: the brightness value to target (0–1). The adjustment peaks "
            "at this value."
        ),
    )
    shadow_protection: float = Field(
        default=0.05,
        description="LP: shadow protection floor (0–SP).",
    )
    highlight_protection: float = Field(
        default=0.9,
        description="HP: highlight protection ceiling (HP–1).",
    )
    channels: str = Field(
        default="all",
        description="Which channels to apply to. all, R, G, B, RG, RB, GB.",
    )
    clip_mode: str = Field(
        default="rgbblend",
        description=(
            "How out-of-range values are handled. "
            "'clip', 'rescale', 'rgbblend' (default), 'globalrescale'."
        ),
    )


class PointsCurvesOptions(BaseModel):
    """
    Arbitrary-points curve adjustment.

    `curves` is a dict whose keys identify a channel or color space and whose
    values are lists of [x, y] control points (each in [0, 1]). The curve
    for each key is fitted with monotone-cubic interpolation (scipy
    PchipInterpolator) and applied in a fixed order — see field doc.
    """
    curves: dict[str, list[list[float]]] = Field(
        description=(
            "Map of channel-or-space key → control points.\n"
            "Each value is a list of [x, y] pairs, x and y both in [0, 1], "
            "minimum 2 points. Keys (applied in this order when multiple "
            "are present):\n"
            "  'rgb'    — single curve applied to R, G, and B identically\n"
            "  'R'      — single-channel curve on R\n"
            "  'G'      — single-channel curve on G\n"
            "  'B'      — single-channel curve on B\n"
            "  'L_star' — curve applied to L* in CIE L*a*b*; chroma "
            "preserved (a*, b* unchanged), then converted back to RGB.\n"
            "Points are sorted by x; if x_min > 0 or x_max < 1, the curve "
            "extrapolates linearly (PchipInterpolator) to the [0, 1] edges. "
            "Output values are clipped to [0, 1]."
        ),
    )


class CurvesAdjustInput(BaseModel):
    method: str = Field(
        default="mtf",
        description=(
            "Curve method:\n"
            "  'mtf'    — Midtone Transfer Function (Siril, three-point).\n"
            "  'ght'    — Generalized Hyperbolic Stretch (Siril, parametric).\n"
            "  'points' — Arbitrary control-point curves per channel/space "
            "(pure Python, monotone-cubic spline)."
        ),
    )
    mtf_options: MTFOptions = Field(default_factory=MTFOptions)
    ght_options: GHTCurvesOptions | None = Field(
        default=None,
        description="GHT parameters. Required when method=ght.",
    )
    points_options: PointsCurvesOptions | None = Field(
        default=None,
        description="Arbitrary-points curve parameters. Required when method=points.",
    )


# ── Command builders ───────────────────────────────────────────────────────────

def _build_mtf_cmd(opts: MTFOptions) -> str:
    cmd = f"mtf {opts.black_point} {opts.midtone} {opts.white_point}"
    if opts.channels != "all":
        cmd += f" {opts.channels}"
    return cmd


def _build_ght_curves_cmd(opts: GHTCurvesOptions) -> str:
    cmd = f"ght -D={opts.stretch_amount} -B={opts.local_intensity}"
    cmd += f" -SP={opts.symmetry_point}"
    if opts.shadow_protection != 0.0:
        cmd += f" -LP={opts.shadow_protection}"
    if opts.highlight_protection != 1.0:
        cmd += f" -HP={opts.highlight_protection}"
    if opts.clip_mode != "rgbblend":
        cmd += f" -clipmode={opts.clip_mode}"
    cmd += " -human"
    if opts.channels != "all":
        cmd += f" {opts.channels}"
    return cmd


# ── Points-curves backend ─────────────────────────────────────────────────────

_POINTS_CURVE_ORDER = ("rgb", "R", "G", "B", "L_star")
_POINTS_CURVE_VALID_KEYS = set(_POINTS_CURVE_ORDER)


def _build_curve(points: list[list[float]]):
    """
    Build a monotone-cubic interpolator from control points. Returns a
    callable that accepts an array of x values in [0, 1] and returns the
    corresponding y values, clipped to [0, 1] on output.
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    if len(points) < 2:
        raise ValueError(
            f"points-curve requires at least 2 control points; got {len(points)}"
        )
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points-curve control points must be a list of [x, y] pairs; "
            f"got shape {pts.shape}"
        )
    # Sort by x and de-duplicate (PchipInterpolator requires strictly increasing x).
    pts = pts[np.argsort(pts[:, 0])]
    keep = np.concatenate([[True], np.diff(pts[:, 0]) > 0])
    pts = pts[keep]
    if pts.shape[0] < 2:
        raise ValueError(
            "points-curve has fewer than 2 distinct x values after de-duplication."
        )
    interp = PchipInterpolator(pts[:, 0], pts[:, 1], extrapolate=True)

    def apply(values):
        out = interp(values)
        return np.clip(out, 0.0, 1.0)

    return apply


def _apply_points_curves(
    image_path: Path,
    options: PointsCurvesOptions,
    output_stem: str,
    working_dir: str,
) -> Path:
    """
    Apply per-channel and per-space curves to the FITS image. Curves are
    applied in fixed order: rgb, R, G, B, L_star. Each curve is built with
    PchipInterpolator and clipped to [0, 1].

    The image is loaded as float32, normalized to [0, 1] for curve
    application, then written back at the same scale. RGB layouts (3, H, W)
    and (H, W, 3) are both supported; mono inputs accept only the 'rgb' key
    (R/G/B/L_star are silently skipped).
    """
    import numpy as np
    from astropy.io import fits as astropy_fits

    unknown = set(options.curves.keys()) - _POINTS_CURVE_VALID_KEYS
    if unknown:
        raise ValueError(
            f"points-curves unknown keys: {sorted(unknown)}. "
            f"Valid: {sorted(_POINTS_CURVE_VALID_KEYS)}."
        )

    with astropy_fits.open(str(image_path)) as hdul:
        data = hdul[0].data.astype(np.float32)

    data_max = float(data.max())
    if data_max > 1.0:
        scale_back = data_max
        data_norm = data / data_max
    else:
        scale_back = 1.0
        data_norm = data.copy()

    is_chw_color = (data_norm.ndim == 3 and data_norm.shape[0] == 3)
    is_hwc_color = (data_norm.ndim == 3 and data_norm.shape[2] == 3)
    is_color = is_chw_color or is_hwc_color
    is_mono = data_norm.ndim == 2

    # Normalize to a CHW float work buffer for uniform indexing
    if is_chw_color:
        work = data_norm.copy()
    elif is_hwc_color:
        work = data_norm.transpose(2, 0, 1).copy()
    elif is_mono:
        work = data_norm[None, :, :].repeat(1, axis=0)
    else:
        raise ValueError(f"Unsupported image shape: {data_norm.shape}")

    for key in _POINTS_CURVE_ORDER:
        if key not in options.curves:
            continue
        curve = _build_curve(options.curves[key])

        if key == "rgb":
            for c in range(work.shape[0]):
                work[c] = curve(work[c])
        elif key == "R":
            if is_color:
                work[0] = curve(work[0])
        elif key == "G":
            if is_color:
                work[1] = curve(work[1])
        elif key == "B":
            if is_color:
                work[2] = curve(work[2])
        elif key == "L_star":
            if is_color:
                from skimage.color import lab2rgb, rgb2lab
                hwc = work.transpose(1, 2, 0).clip(0.0, 1.0)
                lab = rgb2lab(hwc)
                # CIE L* is in [0, 100]; map to [0, 1] for the curve, then back.
                L_norm = lab[..., 0] / 100.0
                lab[..., 0] = curve(L_norm) * 100.0
                rgb = lab2rgb(lab).clip(0.0, 1.0).astype(np.float32)
                work = rgb.transpose(2, 0, 1)
            # Mono: skip — there's no separable luminance from a single channel
            # and the rgb key already covers single-curve adjustment.

    # Convert back to original layout
    if is_chw_color:
        result = work
    elif is_hwc_color:
        result = work.transpose(1, 2, 0)
    else:  # mono
        result = work[0]

    if scale_back != 1.0:
        result = result * scale_back

    output_path = Path(working_dir) / f"{output_stem}.fits"
    hdu = astropy_fits.PrimaryHDU(data=result.astype(np.float32))
    hdu.writeto(str(output_path), overwrite=True)
    return output_path


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=CurvesAdjustInput)
def curves_adjust(
    method: str = "mtf",
    mtf_options: MTFOptions | None = None,
    ght_options: GHTCurvesOptions | None = None,
    points_options: PointsCurvesOptions | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Apply tonal curves to the working image.

    method='mtf'    — three-point Midtone Transfer Function (Siril).
                      Knobs in mtf_options: black_point, midtone, white_point,
                      channels.
    method='ght'    — Generalized Hyperbolic Stretch (Siril). Knobs in
                      ght_options: stretch_amount, local_intensity,
                      symmetry_point, shadow_protection, highlight_protection,
                      channels, clip_mode.
    method='points' — Arbitrary control-point curves (numpy + scipy). Knobs in
                      points_options.curves: dict keyed by 'rgb' / 'R' / 'G' /
                      'B' / 'L_star', each value a list of [x, y] points
                      fitted with PchipInterpolator.

    Output is promoted to paths.current_image.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    method_norm = method.lower()
    if method_norm not in ("mtf", "ght", "points"):
        raise ValueError(
            f"curves_adjust: unknown method '{method}'. "
            f"Valid: mtf, ght, points."
        )

    if mtf_options is None:
        mtf_options = MTFOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if method_norm == "ght" and ght_options is None:
        raise ValueError(
            "curves_adjust: method='ght' requires ght_options.stretch_amount."
        )
    if method_norm == "points" and points_options is None:
        raise ValueError(
            "curves_adjust: method='points' requires points_options.curves."
        )

    stem = img_path.stem
    output_stem = f"{stem}_curves"

    summary: dict = {"output_path": None, "method": method_norm}

    if method_norm == "points":
        output_path = _apply_points_curves(
            img_path, points_options, output_stem, working_dir
        )
        summary["output_path"] = str(output_path)
        summary["points_parameters"] = {
            "curves": {k: list(v) for k, v in points_options.curves.items()},
        }
    else:
        # Siril path (mtf or ght)
        if method_norm == "ght":
            adjust_cmd = _build_ght_curves_cmd(ght_options)  # type: ignore[arg-type]
        else:
            adjust_cmd = _build_mtf_cmd(mtf_options)

        commands = [
            f"load {stem}",
            adjust_cmd,
            f"save {output_stem}",
        ]
        run_siril_script(commands, working_dir=working_dir, timeout=60)

        output_path = Path(working_dir) / f"{output_stem}.fit"
        if not output_path.exists():
            output_path = Path(working_dir) / f"{output_stem}.fits"
        if not output_path.exists():
            raise FileNotFoundError(f"curves_adjust did not produce: {output_path}")

        summary["output_path"] = str(output_path)
        summary["siril_command"] = adjust_cmd
        if method_norm == "mtf":
            summary["mtf_parameters"] = {
                "black_point": mtf_options.black_point,
                "midtone": mtf_options.midtone,
                "white_point": mtf_options.white_point,
                "channels": mtf_options.channels,
            }
        else:  # ght
            summary["ght_parameters"] = {
                "stretch_amount": ght_options.stretch_amount,
                "local_intensity": ght_options.local_intensity,
                "symmetry_point": ght_options.symmetry_point,
                "shadow_protection": ght_options.shadow_protection,
                "highlight_protection": ght_options.highlight_protection,
                "channels": ght_options.channels,
                "clip_mode": ght_options.clip_mode,
            }

    return Command(update={
        "paths": {"current_image": str(output_path)},
        # Curves (MTF / GHT / arbitrary-points) are non-linear tonal
        # transforms applied in display space, post-stretch. Output stays
        # in display space. See Metadata.image_space.
        "metadata": {"image_space": "display"},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
