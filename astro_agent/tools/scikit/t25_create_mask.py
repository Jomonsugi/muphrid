"""
T25 — create_mask

Generate a single-channel FITS mask (pixel values 0.0–1.0) from the working
image based on luminance range, tonal window, or channel difference. Masks
confine subsequent processing to specific image regions.

This tool is the prerequisite for all targeted processing in the pipeline.
Without masks, every tool applies globally and risks degrading regions it
should not touch.

Masked-application pattern (the standard three-step sequence):
  1. [T25 create_mask]        → mask.fits
  2. [processing tool, any T] → processed.fits  (applied globally)
  3. [T23 pixel_math]         "$processed$ * $mask$ + $original$ * (1 - $mask$)"
                              → targeted_result.fits

Backend: Pure Python — scikit-image + Astropy. No Siril invocation.
All I/O via astropy.io.fits so FITS remains the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits as astropy_fits
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, binary_erosion, diamond, disk, square

_FOOTPRINT_BUILDERS = {
    "disk": disk,
    "square": square,
    "diamond": diamond,
}


# ── Pydantic input schema ──────────────────────────────────────────────────────

class LuminanceOptions(BaseModel):
    low: float = Field(default=0.0, description="Minimum luminance value to include in mask (0–1).")
    high: float = Field(default=1.0, description="Maximum luminance value to include in mask (0–1).")


class RangeOptions(BaseModel):
    low: float = Field(description="Minimum channel value to include (0–1).")
    high: float = Field(description="Maximum channel value to include (0–1).")
    channel: str = Field(
        default="luminance",
        description="Channel to threshold: 'luminance', 'R', 'G', or 'B'.",
    )


class ChannelDiffOptions(BaseModel):
    channel_a: str = Field(description="First channel: 'R', 'G', or 'B'.")
    channel_b: str = Field(description="Second channel: 'R', 'G', or 'B'.")
    threshold: float = Field(
        default=0.01,
        description=(
            "Minimum difference (channel_a - channel_b) to include in mask. "
            "channel_a=R, channel_b=B, threshold=0.05 isolates H-alpha emission."
        ),
    )


class CreateMaskInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the working directory for output FITS."
    )
    image_path: str = Field(
        description="Absolute path to the source FITS image."
    )
    mask_type: str = Field(
        description=(
            "Type of mask to generate:\n"
            "  'luminance': bright pixels=1, dark=0. Use to protect stars or "
            "                target bright nebula cores.\n"
            "  'inverted_luminance': dark pixels=1, bright=0. Use to target "
            "                        faint nebulosity or protect sky from saturation.\n"
            "  'range': pixels within [low, high] tonal window=1. Isolates "
            "           midtone nebulosity (e.g. low=0.2, high=0.7).\n"
            "  'channel_diff': where channel_a > channel_b + threshold. "
            "                  Isolates Ha emission (R>B), OIII (B>R)."
        )
    )
    luminance_options: LuminanceOptions = Field(
        default_factory=LuminanceOptions,
        description="Used when mask_type is 'luminance' or 'inverted_luminance'.",
    )
    range_options: RangeOptions = Field(
        default_factory=lambda: RangeOptions(low=0.2, high=0.7),
        description="Used when mask_type is 'range'.",
    )
    channel_diff_options: ChannelDiffOptions = Field(
        default_factory=lambda: ChannelDiffOptions(channel_a="R", channel_b="B"),
        description="Used when mask_type is 'channel_diff'.",
    )
    feather_radius: float = Field(
        default=5.0,
        description=(
            "Gaussian blur radius (sigma) applied to mask edges. "
            "Always > 0 to prevent hard-edged banding in blended results. "
            "5px: conservative (stars, fine detail). 15–30px: large-scale regions."
        ),
    )
    feather_truncate: float = Field(
        default=4.0,
        description=(
            "Truncate the Gaussian filter at this many standard deviations. "
            "4.0: default, smooth tail. 2.0: faster, sharper cutoff. "
            "Higher values extend the transition zone but cost more compute."
        ),
    )
    feather_mode: str = Field(
        default="nearest",
        description=(
            "Boundary handling for the Gaussian feather filter. "
            "'nearest': extend edge pixels (default, good for most masks). "
            "'reflect': mirror at boundary. 'constant': pad with zeros. 'wrap': periodic."
        ),
    )
    expand_px: int = Field(
        default=0,
        description=(
            "Structuring element radius for morphological dilation before "
            "feathering. Use 5 on a star mask to ensure halos are included."
        ),
    )
    contract_px: int = Field(
        default=0,
        description=(
            "Structuring element radius for morphological erosion before "
            "feathering. Use to tighten a mask bleeding into adjacent regions."
        ),
    )
    morphology_iterations: int = Field(
        default=1,
        description=(
            "Number of times to repeat dilation/erosion. "
            "expand_px=2, iterations=3 gives a very different result from "
            "expand_px=6, iterations=1 — repeated small operations are rounder, "
            "single large operations follow the footprint shape more strictly."
        ),
    )
    structuring_element: str = Field(
        default="disk",
        description=(
            "Shape of the morphological footprint for expand/contract. "
            "'disk': circular (default, isotropic). "
            "'square': axis-aligned box. "
            "'diamond': 45° rotated square (Manhattan distance ball)."
        ),
    )
    luminance_model: str = Field(
        default="rec709",
        description=(
            "Luminance weighting for mask computation. "
            "'rec709': standard Rec.709 weights (0.2126R + 0.7152G + 0.0722B). "
            "'equal': equal weight (R+G+B)/3. "
            "'max': per-pixel max(R,G,B) — captures any channel being bright."
        ),
    )
    invert: bool = Field(
        default=False,
        description=(
            "Invert the final mask (0→1, 1→0). Apply after feathering. "
            "Alternative to using 'inverted_luminance' directly."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description=(
            "Output FITS stem (no extension). "
            "Defaults to '{source_stem}_mask_{mask_type}'."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_channels(
    image_path: Path,
    luminance_model: str = "rec709",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FITS, return (lum, r, g, b) all as float32 (H, W)."""
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.max() > 1.0:
        data = data / data.max()

    if data.ndim == 3 and data.shape[0] == 3:
        r, g, b = data[0], data[1], data[2]
    elif data.ndim == 3 and data.shape[2] == 3:
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    else:
        mono = data.squeeze()
        r = g = b = mono

    if luminance_model == "equal":
        lum = (r + g + b) / 3.0
    elif luminance_model == "max":
        lum = np.maximum(np.maximum(r, g), b)
    else:
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    return lum, r, g, b


def _channel_by_name(name: str, r, g, b, lum) -> np.ndarray:
    mapping = {"R": r, "G": g, "B": b, "luminance": lum}
    ch = mapping.get(name.upper() if name.upper() in ("R", "G", "B") else name)
    if ch is None:
        raise ValueError(f"Unknown channel '{name}'. Valid: R, G, B, luminance.")
    return ch


def _build_binary_mask(
    mask_type: str,
    lum: np.ndarray,
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    luminance_options: LuminanceOptions,
    range_options: RangeOptions,
    channel_diff_options: ChannelDiffOptions,
) -> np.ndarray:
    """Compute raw binary mask (True/False, H×W)."""
    mt = mask_type.lower()

    if mt == "luminance":
        return (lum >= luminance_options.low) & (lum <= luminance_options.high)

    elif mt == "inverted_luminance":
        inner = (lum >= luminance_options.low) & (lum <= luminance_options.high)
        return ~inner

    elif mt == "range":
        channel = _channel_by_name(range_options.channel, r, g, b, lum)
        return (channel >= range_options.low) & (channel <= range_options.high)

    elif mt == "channel_diff":
        ch_a = _channel_by_name(channel_diff_options.channel_a, r, g, b, lum)
        ch_b = _channel_by_name(channel_diff_options.channel_b, r, g, b, lum)
        return (ch_a - ch_b) >= channel_diff_options.threshold

    else:
        raise ValueError(
            f"Unknown mask_type '{mask_type}'. "
            "Valid: luminance, inverted_luminance, range, channel_diff."
        )


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=CreateMaskInput)
def create_mask(
    working_dir: str,
    image_path: str,
    mask_type: str = "luminance",
    luminance_options: LuminanceOptions | None = None,
    range_options: RangeOptions | None = None,
    channel_diff_options: ChannelDiffOptions | None = None,
    feather_radius: float = 5.0,
    feather_truncate: float = 4.0,
    feather_mode: str = "nearest",
    expand_px: int = 0,
    contract_px: int = 0,
    morphology_iterations: int = 1,
    structuring_element: str = "disk",
    luminance_model: str = "rec709",
    invert: bool = False,
    output_stem: str | None = None,
) -> dict:
    """
    Generate a float32 FITS mask (values 0.0–1.0) for targeted processing.

    Masks are the foundation of the masked-application pattern — they confine
    any subsequent tool's effect to specific tonal regions without a permanent
    separate processing path.

    Mask type guide:
    - 'luminance': bright=1, dark=0. Protect stars during noise reduction;
      target bright nebula cores for subtle curves adjustment.
    - 'inverted_luminance': dark=1, bright=0. Target faint nebulosity for
      saturation/sharpness; keep dark sky from being affected.
    - 'range': pixels in [low, high] range=1. Midtone nebulosity isolation:
      low=0.2, high=0.7 selects the nebula without stars or pure sky.
    - 'channel_diff': channel_a > channel_b + threshold. Use R>B to isolate
      Hα emission; B>R for OIII; G>R for OIII in narrowband mapped images.

    The mask is saved as a single-channel float32 FITS. Use it immediately in
    T23 pixel_math: "$processed$ * $mask$ + $original$ * (1 - $mask$)"
    """
    if luminance_options is None:
        luminance_options = LuminanceOptions()
    if range_options is None:
        range_options = RangeOptions(low=0.2, high=0.7)
    if channel_diff_options is None:
        channel_diff_options = ChannelDiffOptions(channel_a="R", channel_b="B")

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    footprint_fn = _FOOTPRINT_BUILDERS.get(structuring_element, disk)

    lum, r, g, b = _load_channels(img_path, luminance_model)

    binary = _build_binary_mask(
        mask_type, lum, r, g, b,
        luminance_options, range_options, channel_diff_options,
    )

    for _ in range(morphology_iterations):
        if expand_px > 0:
            binary = binary_dilation(binary, footprint_fn(expand_px))
        if contract_px > 0:
            binary = binary_erosion(binary, footprint_fn(contract_px))

    mask = binary.astype(np.float32)

    if feather_radius > 0:
        mask = gaussian(
            mask, sigma=feather_radius, truncate=feather_truncate, mode=feather_mode,
        ).astype(np.float32)
        mask = np.clip(mask, 0.0, 1.0)

    # Optional inversion
    if invert:
        mask = 1.0 - mask

    # Write single-channel float32 FITS
    out_stem = output_stem or f"{img_path.stem}_mask_{mask_type}"
    out_path = Path(working_dir) / f"{out_stem}.fits"

    hdu = astropy_fits.PrimaryHDU(data=mask)
    hdu.header["BUNIT"] = "mask"
    hdu.header["MASKTYPE"] = mask_type
    hdul = astropy_fits.HDUList([hdu])
    hdul.writeto(out_path, overwrite=True)

    coverage_pct = float(np.mean(mask > 0.5) * 100)
    mean_value = float(np.mean(mask))

    return {
        "mask_path": str(out_path),
        "coverage_pct": coverage_pct,
        "mean_value": mean_value,
    }
