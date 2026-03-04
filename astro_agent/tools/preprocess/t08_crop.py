"""
T08 — auto_crop

Remove black border artifacts introduced by registration (edge regions where
not all input frames overlap after alignment and resampling).

Strategy:
    1. Load the stacked FITS with astropy.
    2. Compute a per-pixel signal mask: pixel is "good" if max across channels > threshold.
    3. Find the bounding box of the good region using numpy.
    4. Apply a 5-pixel safety inset on all sides.
    5. Issue Siril `crop <x> <y> <w> <h>` to perform the actual crop.

Always run after siril_stack. If pixels_removed_pct > 15, registration had
poor frame overlap — note in the processing report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class AutoCropInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the stacked master light FITS to crop. "
            "Typically master_light.fit from siril_stack (T07)."
        )
    )
    threshold: float = Field(
        default=0.01,
        description=(
            "Pixel value threshold (0–1 normalized) below which a pixel is "
            "considered a black border artifact. 0.01 works for most stacks."
        ),
    )


# ── Bounding box detection ─────────────────────────────────────────────────────

def _find_crop_bounds(image_path: Path, threshold: float) -> tuple[int, int, int, int]:
    """
    Return (x, y, w, h) bounding box of non-black content, with 5px safety inset.

    For a 3-channel (RGB) FITS, collapses across channels using max.
    For mono, uses the single channel directly.
    """
    with fits.open(str(image_path)) as hdul:
        data = hdul[0].data.astype(np.float32)  # type: ignore[union-attr]

    # Normalize to 0–1 range
    data_max = float(np.max(data))
    if data_max > 0:
        data = data / data_max

    # Collapse to 2D signal mask
    if data.ndim == 3:
        # Siril FITS: shape is (channels, height, width)
        signal = np.max(data, axis=0)
    else:
        signal = data

    mask = signal > threshold

    # Find bounding box of True region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No signal found — return full image bounds
        h, w = signal.shape
        return 0, 0, w, h

    row_min, row_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    col_min, col_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

    # 5-pixel safety inset
    inset = 5
    x = max(0, col_min + inset)
    y = max(0, row_min + inset)
    x2 = min(signal.shape[1] - 1, col_max - inset)
    y2 = min(signal.shape[0] - 1, row_max - inset)

    # If the insets collapse the crop window (small signal island), fall back
    # to the original bounding box without inset so geometry remains valid.
    if x2 < x or y2 < y:
        x, y, x2, y2 = col_min, row_min, col_max, row_max

    # +1 because row_max/col_max are inclusive pixel indices
    w = x2 - x + 1
    h = y2 - y + 1

    return x, y, w, h


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=AutoCropInput)
def auto_crop(
    working_dir: str,
    image_path: str,
    threshold: float = 0.01,
) -> dict:
    """
    Remove black border registration artifacts from the stacked master light.
    Detects the non-black bounding box in Python, then uses Siril crop command.

    Always run after siril_stack. If pixels_removed_pct > 15, check
    registration quality — poor overlap indicates frames may need to be
    re-registered with looser framing settings.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    x, y, w, h = _find_crop_bounds(img_path, threshold)

    with fits.open(str(img_path)) as hdul:
        data = hdul[0].data
        if data.ndim == 3:
            full_h, full_w = data.shape[1], data.shape[2]
        else:
            full_h, full_w = data.shape[0], data.shape[1]

    original_pixels = full_h * full_w
    cropped_pixels  = w * h
    removed_pct     = round((1.0 - cropped_pixels / original_pixels) * 100, 2)

    # Load the image in Siril and apply crop
    image_name = img_path.stem   # e.g. "master_light"
    commands = [
        f"load {image_name}",
        f"crop {x} {y} {w} {h}",
        f"save {image_name}_crop",
    ]

    run_siril_script(commands, working_dir=working_dir, timeout=60)

    cropped_path = Path(working_dir) / f"{image_name}_crop.fit"
    if not cropped_path.exists():
        cropped_path = Path(working_dir) / f"{image_name}_crop.fits"
    if not cropped_path.exists():
        raise FileNotFoundError(
            f"Cropped image not found at {cropped_path}. "
            f"Siril crop geometry was: x={x} y={y} w={w} h={h}"
        )

    return {
        "cropped_image_path": str(cropped_path),
        "crop_geometry": {"x": x, "y": y, "w": w, "h": h},
        "pixels_removed_pct": removed_pct,
    }
