"""
T32 — present_images

Utility tool the agent calls when it wants the human to review or compare
images. The agent provides image paths, labels, and optional per-image metrics.
The presenter layer (Gradio, CLI) intercepts the result and displays the images
appropriately — the agent has no knowledge of how they are rendered.

This is a UTILITY tool (available in all phases).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageEntry(BaseModel):
    """A single image to present for review."""
    path: str = Field(description="Absolute path to the image file (FITS or JPG/PNG).")
    label: str = Field(description="Short label for this image (e.g. 'Gentle stretch', 'Variant A').")
    metrics: dict | None = Field(
        default=None,
        description="Optional key metrics for this image (e.g. {'dynamic_range_db': 6.5, 'background': 0.16}).",
    )


class PresentImagesInput(BaseModel):
    """Input schema for present_images."""
    images: list[ImageEntry] = Field(description="List of images to present for comparison.")
    title: str = Field(description="Title for the image comparison (e.g. 'Stretch Variants').")
    description: str = Field(
        default="",
        description="Summary of what is being shown and why — displayed in the chat.",
    )


@tool(args_schema=PresentImagesInput)
def present_images(
    images: list[ImageEntry],
    title: str,
    description: str = "",
) -> str:
    """
    Present images for visual inspection and comparison.

    analyze_image is your primary diagnostic tool — use it first for
    data-driven decisions. Call present_images when visual inspection
    is needed to evaluate results that metrics alone cannot fully capture:
    stretch character, gradient artifacts, star rendering, color balance.

    Do NOT use as a default step after every tool — most decisions are
    fully served by analyze_image data.
    """
    # Validate paths exist
    valid_images = []
    for img in images:
        p = Path(img.path)
        if p.exists():
            valid_images.append(img)
        else:
            logger.warning(f"present_images: path not found: {img.path}")

    if not valid_images:
        return json.dumps({
            "status": "error",
            "message": "No valid image paths found.",
        })

    # Build result — the presenter layer reads this from the ToolMessage
    result = {
        "status": "presented",
        "title": title,
        "description": description,
        "images": [
            {
                "path": img.path,
                "label": img.label,
                "metrics": img.metrics,
            }
            for img in valid_images
        ],
        "count": len(valid_images),
    }

    logger.info(f"present_images: {title} — {len(valid_images)} images")
    return json.dumps(result)
