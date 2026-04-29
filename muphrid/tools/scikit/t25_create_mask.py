"""
T25 — create_mask

Generate a single-channel FITS mask (float32 values in [0, 1]) from the
working image. Masks are statistical (luminance / tonal range / channel
difference), spatial (rectangle / polygon / ellipse regions), or a
combination of both.

The mask is written as a separate FITS file under working_dir and the
path is recorded in paths.latest_mask. The current image is not modified;
applying the mask is done via pixel_math or via the masked_process
compound tool.

Backend: Pure Python — scikit-image + Astropy. No Siril invocation.
All I/O via astropy.io.fits.
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
from skimage.draw import ellipse as sk_ellipse
from skimage.draw import polygon as sk_polygon
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, binary_erosion, diamond, disk, square

from muphrid.graph.state import AstroState

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
            "Minimum value of (channel_a − channel_b) for a pixel to be "
            "included in the mask."
        ),
    )


class RegionSpec(BaseModel):
    """A spatial region constraint applied alongside the statistical mask."""

    shape: str = Field(
        description=(
            "Geometric primitive: 'rectangle', 'polygon', or 'ellipse'. "
            "Determines how `coords` is interpreted."
        ),
    )
    coords: list[float] = Field(
        description=(
            "Pixel coordinates in the source image. Origin is top-left, x "
            "increases right, y increases down. Format depends on shape:\n"
            "  rectangle: [x0, y0, x1, y1] (any opposite corners)\n"
            "  polygon:   [x0, y0, x1, y1, x2, y2, ...] (≥3 vertices, "
            "even total count)\n"
            "  ellipse:   [center_x, center_y, radius_x, radius_y]"
        ),
    )


class CreateMaskInput(BaseModel):
    mask_type: str = Field(
        description=(
            "Type of statistical mask:\n"
            "  'luminance': mask=1 where luminance ∈ [low, high], else 0.\n"
            "  'inverted_luminance': mask=0 where luminance ∈ [low, high], "
            "else 1.\n"
            "  'range': mask=1 where the chosen channel value ∈ [low, high], "
            "else 0.\n"
            "  'channel_diff': mask=1 where (channel_a − channel_b) ≥ "
            "threshold, else 0.\n"
            "Combined with `region_spec` (when provided) per `region_combine`."
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
    region_spec: RegionSpec | None = Field(
        default=None,
        description=(
            "Optional spatial region constraint. When provided, a binary "
            "region mask is built from `region_spec.shape` and `coords` and "
            "combined with the statistical mask via `region_combine` before "
            "morphology and feathering. None = no spatial constraint."
        ),
    )
    region_combine: str = Field(
        default="and",
        description=(
            "How the region mask combines with the statistical mask. "
            "'and': pixel must be in both (intersection). "
            "'or': pixel in either (union). "
            "'subtract': pixel in statistical AND NOT in region "
            "(removes the region from the statistical selection). "
            "Ignored when region_spec is None."
        ),
    )
    feather_radius: float = Field(
        default=5.0,
        description=(
            "Gaussian blur sigma in pixels, applied to the binary mask to "
            "soften edges before write. 0 disables feathering and writes a "
            "hard-edged binary mask. Larger values widen the transition zone."
        ),
    )
    feather_truncate: float = Field(
        default=4.0,
        description=(
            "Number of standard deviations at which the Gaussian filter is "
            "truncated. Smaller values cut off the tail sooner (sharper edge "
            "transition); larger values keep more of the tail at higher "
            "compute cost."
        ),
    )
    feather_mode: str = Field(
        default="nearest",
        description=(
            "Boundary handling for the Gaussian filter. "
            "'nearest' extends edge pixel values; 'reflect' mirrors at the "
            "boundary; 'constant' pads with zeros; 'wrap' is periodic."
        ),
    )
    expand_px: int = Field(
        default=0,
        description=(
            "Morphological dilation footprint radius in pixels, applied "
            "before feathering. 0 = no dilation. Grows the mask outward."
        ),
    )
    contract_px: int = Field(
        default=0,
        description=(
            "Morphological erosion footprint radius in pixels, applied "
            "before feathering. 0 = no erosion. Shrinks the mask inward."
        ),
    )
    morphology_iterations: int = Field(
        default=1,
        description=(
            "Number of times the dilation+erosion pass is applied. "
            "Repeated small operations produce rounder boundaries than a "
            "single pass with the equivalent total footprint."
        ),
    )
    structuring_element: str = Field(
        default="disk",
        description=(
            "Shape of the morphological footprint for dilation and erosion. "
            "'disk': circular (isotropic). "
            "'square': axis-aligned. "
            "'diamond': 45° rotated square (Manhattan-distance ball)."
        ),
    )
    luminance_model: str = Field(
        default="rec709",
        description=(
            "Per-pixel luminance computation. "
            "'rec709': Rec.709 weights (0.2126R + 0.7152G + 0.0722B). "
            "'equal': (R + G + B) / 3. "
            "'max': max(R, G, B) per pixel."
        ),
    )
    invert: bool = Field(
        default=False,
        description=(
            "Invert the final mask (0 ↔ 1) after morphology and feathering."
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


def _build_region_mask(
    region_spec: RegionSpec,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Rasterize a RegionSpec into a (H, W) bool mask.

    Coordinates use top-left origin: x→right, y→down. skimage.draw uses
    (row, col) = (y, x), which we translate when calling the primitives.
    Coordinates outside image bounds are clipped silently.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    shape = region_spec.shape.lower()
    coords = region_spec.coords

    if shape == "rectangle":
        if len(coords) != 4:
            raise ValueError(
                f"rectangle region coords must be [x0, y0, x1, y1]; "
                f"got {len(coords)} values."
            )
        x0, y0, x1, y1 = coords
        x0i, x1i = sorted([int(max(0, min(w, x0))), int(max(0, min(w, x1)))])
        y0i, y1i = sorted([int(max(0, min(h, y0))), int(max(0, min(h, y1)))])
        mask[y0i:y1i, x0i:x1i] = True

    elif shape == "polygon":
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError(
                f"polygon region coords must be [x0, y0, x1, y1, ...] with "
                f"≥3 vertices (≥6 even-count values); got {len(coords)}."
            )
        xs = coords[0::2]
        ys = coords[1::2]
        rr, cc = sk_polygon(np.array(ys), np.array(xs), shape=(h, w))
        mask[rr, cc] = True

    elif shape == "ellipse":
        if len(coords) != 4:
            raise ValueError(
                f"ellipse region coords must be "
                f"[center_x, center_y, radius_x, radius_y]; "
                f"got {len(coords)} values."
            )
        cx, cy, rx, ry = coords
        rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(h, w))
        mask[rr, cc] = True

    else:
        raise ValueError(
            f"Unknown region shape '{region_spec.shape}'. "
            "Valid: rectangle, polygon, ellipse."
        )

    return mask


def _combine_masks(
    stat_mask: np.ndarray,
    region_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Combine the statistical and region binary masks per `mode`."""
    m = mode.lower()
    if m == "and":
        return stat_mask & region_mask
    if m == "or":
        return stat_mask | region_mask
    if m == "subtract":
        return stat_mask & ~region_mask
    raise ValueError(
        f"Unknown region_combine '{mode}'. Valid: and, or, subtract."
    )


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
    mask_type: str = "luminance",
    luminance_options: LuminanceOptions | None = None,
    range_options: RangeOptions | None = None,
    channel_diff_options: ChannelDiffOptions | None = None,
    region_spec: RegionSpec | None = None,
    region_combine: str = "and",
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
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Generate a float32 FITS mask (values in [0, 1]) from the working image.

    Statistical mask types (selected via `mask_type`):
      - luminance / inverted_luminance: gate on luminance value
      - range: gate on a chosen channel within a value window
      - channel_diff: gate on signed difference between two channels

    When `region_spec` is provided, a spatial region (rectangle, polygon,
    or ellipse) is rasterized and combined with the statistical mask per
    `region_combine` ('and' / 'or' / 'subtract').

    Pipeline order: statistical mask → region combine (if any) → morphology
    (dilate/erode × iterations) → feather (Gaussian) → optional invert.

    The result is written to working_dir as a single-channel FITS and the
    path is recorded in paths.latest_mask. The current image is not
    modified — apply the mask via pixel_math or via the masked_process
    compound tool.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

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

    # Spatial region combine — applied to the binary statistical mask
    # before morphology so dilate/erode + feather operate on the final
    # combined selection (the typical desire — soften the edge of where
    # both constraints intersect).
    if region_spec is not None:
        region_mask = _build_region_mask(region_spec, binary.shape)
        binary = _combine_masks(binary, region_mask, region_combine)

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
    mask_output_path = Path(working_dir) / f"{out_stem}.fits"

    hdu = astropy_fits.PrimaryHDU(data=mask)
    hdu.header["BUNIT"] = "mask"
    hdu.header["MASKTYPE"] = mask_type
    hdul = astropy_fits.HDUList([hdu])
    hdul.writeto(mask_output_path, overwrite=True)

    mask_coverage_pct = round(float(np.mean(mask)) * 100, 2)
    mask_min = float(np.min(mask))
    mask_max = float(np.max(mask))

    summary: dict = {
        "output_path": str(mask_output_path),
        "mask_type": mask_type,
        "source_image": str(image_path),
        "mask_coverage_pct": mask_coverage_pct,
        "mask_min": mask_min,
        "mask_max": mask_max,
        "dimensions": {"height": int(mask.shape[0]), "width": int(mask.shape[1])},
        "settings": {
            "feather_radius": feather_radius,
            "feather_truncate": feather_truncate,
            "feather_mode": feather_mode,
            "expand_px": expand_px,
            "contract_px": contract_px,
            "morphology_iterations": morphology_iterations,
            "structuring_element": structuring_element,
            "luminance_model": luminance_model,
            "invert": invert,
        },
    }
    if region_spec is not None:
        summary["region"] = {
            "shape": region_spec.shape,
            "coords": list(region_spec.coords),
            "combine": region_combine,
        }

    return Command(update={
        "paths": {"latest_mask": str(mask_output_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
