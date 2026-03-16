"""
T08 — crop

Two modes:

  auto (default): detects the non-black signal bounding box in Python, applies
      a 5-pixel safety inset, then issues Siril crop. Run after siril_stack to
      remove registration edge artifacts. If pixels_removed_pct > 15, registration
      had poor frame overlap — note in the processing report.

  manual: crops to explicit (x, y, w, h) pixel coordinates provided by the user.
      Use for compositional reframing or removing a known noisy edge. Coordinates
      follow Siril convention:
      x/y = top-left corner in pixels, w/h = width/height in pixels.

Siril command: crop <x> <y> <w> <h>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
from astropy.io import fits
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from astro_agent.graph.state import AstroState
from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class AutoCropInput(BaseModel):
    mode: str = Field(
        default="auto",
        description=(
            "auto: detect and remove black registration borders automatically. "
            "Run after siril_stack; no user input needed.\n"
            "manual: crop to explicit coordinates provided by the user. "
            "Obtain (x, y, w, h) before calling."
        ),
    )
    threshold: float = Field(
        default=0.01,
        description=(
            "Pixel value threshold (0–1 normalized) below which a pixel is "
            "considered a black border artifact. Only used in auto mode. "
            "0.01 works for most stacks."
        ),
    )
    x: int | None = Field(
        default=None,
        description="Manual mode only. Left edge of the crop rectangle in pixels.",
    )
    y: int | None = Field(
        default=None,
        description="Manual mode only. Top edge of the crop rectangle in pixels.",
    )
    w: int | None = Field(
        default=None,
        description="Manual mode only. Width of the crop rectangle in pixels.",
    )
    h: int | None = Field(
        default=None,
        description="Manual mode only. Height of the crop rectangle in pixels.",
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

@tool
def auto_crop(
    mode: str = "auto",
    threshold: float = 0.01,
    x: int | None = None,
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Crop a FITS image to remove registration borders or reframe to user coordinates.

    Working directory and current image path are read from state.

    Args:
        mode: 'auto' (detect and remove black borders) or 'manual'
            (crop to explicit x/y/w/h coordinates — obtain from user first).
        threshold: Auto mode border detection threshold (default 0.01).
        x, y, w, h: Manual mode crop coordinates (top-left + size in pixels).
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]
    if not image_path:
        raise ValueError("current_image not found in state. Run siril_stack (T07) first.")

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if mode == "manual":
        if any(v is None for v in (x, y, w, h)):
            raise ValueError(
                "manual mode requires all of: x, y, w, h. "
                "Obtain these from the user before calling."
            )
        crop_x, crop_y, crop_w, crop_h = x, y, w, h  # type: ignore[assignment]
    else:
        crop_x, crop_y, crop_w, crop_h = _find_crop_bounds(img_path, threshold)

    with fits.open(str(img_path)) as hdul:
        data = hdul[0].data
        if data.ndim == 3:
            full_h, full_w = data.shape[1], data.shape[2]
        else:
            full_h, full_w = data.shape[0], data.shape[1]

    original_pixels = full_h * full_w
    cropped_pixels  = crop_w * crop_h
    removed_pct     = round((1.0 - cropped_pixels / original_pixels) * 100, 2)

    image_name = img_path.stem
    commands = [
        f"load {image_name}",
        f"crop {crop_x} {crop_y} {crop_w} {crop_h}",
        f"save {image_name}_crop",
    ]

    run_siril_script(commands, working_dir=working_dir, timeout=60)

    cropped_path = Path(working_dir) / f"{image_name}_crop.fit"
    if not cropped_path.exists():
        cropped_path = Path(working_dir) / f"{image_name}_crop.fits"
    if not cropped_path.exists():
        raise FileNotFoundError(
            f"Cropped image not found at {cropped_path}. "
            f"Siril crop geometry was: x={crop_x} y={crop_y} w={crop_w} h={crop_h}"
        )

    summary = {
        "output_path": str(cropped_path),
        "mode": mode,
        "crop_geometry": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_w,
            "height": crop_h,
        },
        "original_dimensions": {
            "width": full_w,
            "height": full_h,
        },
        "pixels_removed_pct": removed_pct,
        "threshold": threshold if mode == "auto" else None,
    }

    return Command(update={
        "paths": {**state["paths"], "current_image": str(cropped_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
