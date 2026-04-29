"""
T26 — reduce_stars

Reduce the angular size of stars using morphological erosion applied only
within star regions. Unlike T19 star_restoration with star_weight < 1.0
(which dims stars without changing size), this tool physically shrinks star
disks — eliminating bloom and improving perceived sharpness of background
nebulosity.

Apply to the final combined image after T19, or directly before export.
Not a substitute for good registration; intended for residual star size
reduction after all other processing is complete.

Algorithm:
1. Build binary star mask from T15 output or auto-threshold luminance
2. Optionally dilate mask to exclude innermost star cores from erosion
3. Apply skimage.morphology.erosion(channel, disk(kernel_radius)) for
   `iterations` passes, only within the masked star region
4. Blend eroded result with original via feathered mask + blend_amount
5. Report stars_affected_count and mean_size_reduction_pct

Backend: Pure Python — scikit-image + Astropy. No Siril invocation.
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
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, dilation, diamond, disk, erosion, square
from skimage.feature import peak_local_max

from muphrid.graph.state import AstroState

_FOOTPRINT_BUILDERS = {
    "disk": disk,
    "square": square,
    "diamond": diamond,
}


# ── Pydantic input schema ──────────────────────────────────────────────────────

class ReduceStarsInput(BaseModel):
    detection_threshold: float = Field(
        default=0.6,
        description=(
            "Luminance value above which a pixel is treated as a star when "
            "no star_mask_path is provided. "
            "0.6: standard stretched image. "
            "0.4: catch fainter stars. "
            "0.75: avoid bright nebula cores being incorrectly eroded."
        ),
    )
    kernel_radius: int = Field(
        default=1,
        description=(
            "Radius of the morphological erosion footprint (pixels). "
            "1: gentlest (subtle tightening). "
            "2: moderate (noticeable bloom reduction). "
            "3: aggressive (only for severely bloated stars). "
            "Do not exceed 3 without previewing — stars become pixelated."
        ),
    )
    structuring_element: str = Field(
        default="disk",
        description=(
            "Shape of the erosion footprint. "
            "'disk': circular (default, isotropic shrinkage). "
            "'square': axis-aligned box (faster, minor axis bias). "
            "'diamond': 45° rotated square."
        ),
    )
    erosion_mode: str = Field(
        default="reflect",
        description=(
            "Boundary handling for grayscale erosion. "
            "'reflect': mirror at edge (default). 'nearest': extend edge pixels. "
            "'constant': pad with cval (0.0). 'wrap': periodic."
        ),
    )
    iterations: int = Field(
        default=1,
        description=(
            "Number of erosion passes applied sequentially. "
            "1 pass = minimal. 2 passes = strong. Do not exceed 3."
        ),
    )
    blend_amount: float = Field(
        default=1.0,
        description=(
            "Weight applied to the eroded result in the blend: "
            "1.0 = full reduction applied. "
            "0.5 = half-strength (useful for mildly large stars). "
            "Blended with original: result * blend + original * (1 - blend)."
        ),
    )
    protect_core_radius: int = Field(
        default=0,
        description=(
            "Pixels to dilate the exclusion zone around each star peak, "
            "preventing erosion from modifying the star core. "
            "Use when stars contain important color data (e.g. red giants). "
            "0 = no core protection (default)."
        ),
    )
    peak_threshold_abs: float | None = Field(
        default=None,
        description=(
            "Minimum absolute intensity a peak must have to qualify as a star "
            "center (for core protection). None = no threshold. "
            "0.8: only protect very bright stars. 0.5: protect moderate stars."
        ),
    )
    peak_num_peaks: int | None = Field(
        default=None,
        description=(
            "Maximum number of star peaks to detect for core protection. "
            "None = unlimited. Use 500 to cap computation on dense fields."
        ),
    )
    peak_exclude_border: bool = Field(
        default=True,
        description=(
            "Exclude peaks within min_distance of the image border. "
            "True: default (prevents partial star artifacts at edges). "
            "False: include all peaks including border stars."
        ),
    )
    label_connectivity: int = Field(
        default=2,
        description=(
            "Pixel connectivity for counting distinct stars. "
            "1: 4-connected (strict, may split elongated stars into two). "
            "2: 8-connected (default, counts diagonal-touching pixels as same star)."
        ),
    )
    feather_px: int = Field(
        default=3,
        description=(
            "Gaussian sigma for the transition between eroded star region "
            "and surrounding image. Prevents visible edge rings. "
            "3px: standard. 5px: for larger stars or stronger reduction."
        ),
    )
    output_stem: str | None = Field(
        default=None,
        description="Output FITS stem. Defaults to '{source_stem}_reduced_stars'.",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_fits(image_path: Path) -> tuple[np.ndarray, bool]:
    """
    Load a FITS file, return (array, is_color).
    Color images returned as (3, H, W) float32.
    """
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.max() > 1.0:
        data = data / data.max()

    if data.ndim == 3 and data.shape[0] == 3:
        return data, True
    elif data.ndim == 3 and data.shape[2] == 3:
        return np.moveaxis(data, -1, 0), True
    else:
        return data.squeeze()[np.newaxis, :, :], False


def _compute_luminance(data: np.ndarray) -> np.ndarray:
    """Compute (H, W) luminance from (C, H, W) float32 array."""
    if data.shape[0] == 1:
        return data[0].astype(np.float32)
    return (0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]).astype(np.float32)


def _load_mask_channel(mask_path: Path) -> np.ndarray:
    """Load a single-channel float32 mask FITS, return (H, W) boolean."""
    with astropy_fits.open(mask_path) as hdul:
        mask_data = hdul[0].data.astype(np.float32).squeeze()
    if mask_data.max() > 1.0:
        mask_data = mask_data / mask_data.max()
    return mask_data > 0.5


def _erode_channel(
    channel: np.ndarray,
    star_binary: np.ndarray,
    core_exclusion: np.ndarray,
    kernel_radius: int,
    iterations: int,
    footprint_fn=disk,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply morphological erosion to channel pixels within star_binary,
    excluding pixels in core_exclusion zone.
    """
    struct = footprint_fn(kernel_radius)
    eroded = channel.copy()
    for _ in range(iterations):
        e = erosion(eroded, struct, mode=mode)
        apply_mask = star_binary & ~core_exclusion
        eroded = np.where(apply_mask, e, eroded)
    return eroded


def _count_stars_affected(star_binary: np.ndarray, connectivity: int = 2) -> int:
    """Count distinct connected star regions."""
    labeled = label(star_binary, connectivity=connectivity)
    return int(labeled.max())


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=ReduceStarsInput)
def reduce_stars(
    detection_threshold: float = 0.6,
    kernel_radius: int = 1,
    structuring_element: str = "disk",
    erosion_mode: str = "reflect",
    iterations: int = 1,
    blend_amount: float = 1.0,
    protect_core_radius: int = 0,
    peak_threshold_abs: float | None = None,
    peak_num_peaks: int | None = None,
    peak_exclude_border: bool = True,
    label_connectivity: int = 2,
    feather_px: int = 3,
    output_stem: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Physically reduce the angular size of stars via morphological erosion.

    This tool shrinks star disk diameters without dimming them. The effect is
    confined to the star region mask — background nebulosity is never touched.

    Key settings:
    - kernel_radius=1, iterations=1: gentlest effect, almost always safe
    - kernel_radius=2: moderate bloom reduction (preview result)
    - star_mask_path: preferred over auto-detection — prevents bright nebula
      cores from being incorrectly identified as stars
    - blend_amount=0.5: half-strength, useful when stars are only mildly large
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    footprint_fn = _FOOTPRINT_BUILDERS.get(structuring_element, disk)

    data, is_color = _load_fits(img_path)
    lum = _compute_luminance(data)

    # Use explicit star_mask_path if given, else fall back to state
    effective_mask_path = star_mask_path or state["paths"].get("star_mask")

    if effective_mask_path and Path(effective_mask_path).exists():
        star_binary = _load_mask_channel(Path(effective_mask_path))
        if star_binary.shape != lum.shape:
            from skimage.transform import resize
            star_binary = resize(
                star_binary.astype(np.float32), lum.shape, order=0
            ).astype(bool)
    else:
        star_binary = lum > detection_threshold

    if protect_core_radius > 0:
        plm_kwargs: dict = dict(
            min_distance=max(1, protect_core_radius),
            labels=star_binary,
            exclude_border=peak_exclude_border,
        )
        if peak_threshold_abs is not None:
            plm_kwargs["threshold_abs"] = peak_threshold_abs
        if peak_num_peaks is not None:
            plm_kwargs["num_peaks"] = peak_num_peaks

        peaks = peak_local_max(lum, **plm_kwargs)
        peak_mask = np.zeros_like(star_binary, dtype=bool)
        if len(peaks) > 0:
            peak_mask[peaks[:, 0], peaks[:, 1]] = True
        core_exclusion = binary_dilation(peak_mask, footprint_fn(protect_core_radius))
        core_exclusion = core_exclusion & star_binary
    else:
        core_exclusion = np.zeros_like(star_binary, dtype=bool)

    # Feathered blend mask for smooth edges
    star_float = star_binary.astype(np.float32)
    if feather_px > 0:
        star_blend_mask = gaussian(star_float, sigma=feather_px)
        star_blend_mask = np.clip(star_blend_mask, 0.0, 1.0)
    else:
        star_blend_mask = star_float

    # Track pre-erosion star pixel count for size-reduction reporting
    star_pixels_before = int(np.sum(star_binary))

    output_data = data.copy()
    for ch_idx in range(data.shape[0]):
        eroded = _erode_channel(
            data[ch_idx], star_binary, core_exclusion, kernel_radius, iterations,
            footprint_fn=footprint_fn, mode=erosion_mode,
        )
        # Blend eroded result with original via feathered mask
        blended = eroded * blend_amount + data[ch_idx] * (1.0 - blend_amount)
        output_data[ch_idx] = (
            blended * star_blend_mask + data[ch_idx] * (1.0 - star_blend_mask)
        )

    # Measure size reduction as shrinkage of bright star area.
    output_lum = _compute_luminance(output_data)
    lum_threshold = float(np.mean(lum[star_binary])) if np.any(star_binary) else detection_threshold
    star_pixels_after = int(np.sum((output_lum > lum_threshold) & star_binary))
    mean_size_reduction_pct = (
        float((star_pixels_before - star_pixels_after) / (star_pixels_before + 1e-9) * 100)
        if star_pixels_before > 0
        else 0.0
    )
    stars_affected_count = _count_stars_affected(star_binary, connectivity=label_connectivity)

    # Save output
    out_stem = output_stem or f"{img_path.stem}_reduced_stars"
    out_path = Path(working_dir) / f"{out_stem}.fits"
    hdu = astropy_fits.PrimaryHDU(data=output_data)
    astropy_fits.HDUList([hdu]).writeto(out_path, overwrite=True)

    summary = {
        "output_path": str(out_path),
        "stars_affected_count": stars_affected_count,
        "star_pixels_before": star_pixels_before,
        "star_pixels_after": star_pixels_after,
        "mean_size_reduction_pct": round(mean_size_reduction_pct, 2),
        "settings": {
            "detection_threshold": detection_threshold,
            "kernel_radius": kernel_radius,
            "structuring_element": structuring_element,
            "erosion_mode": erosion_mode,
            "iterations": iterations,
            "blend_amount": blend_amount,
            "protect_core_radius": protect_core_radius,
            "feather_px": feather_px,
        },
    }

    return Command(update={
        "paths": {"current_image": str(out_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
