"""
T32 — present_images

Utility tool the agent calls when it wants the human to review or compare
images. The agent provides image paths, labels, and optional per-image metrics.
The presenter layer (Gradio, CLI) intercepts the result and displays the images
appropriately — the agent has no knowledge of how they are rendered.

This is a UTILITY tool (available in all phases).

State side effects:
  Beyond the rendered ToolMessage, this tool also updates state.visual_context
  with the presented images so the VLM (when enabled) can see them on the
  agent's next reasoning cycle. visual_context is the live working set the
  agent_node retention helper reads to build the multimodal view. Each call
  REPLACES previous present_images entries — the latest call wins. Other
  sources (hitl_variant, phase_carry) are preserved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState, VisualRef

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


def _to_preview_path(path: str, working_dir: str, is_linear: bool) -> str | None:
    """
    Resolve a path to a JPG/PNG preview suitable for VLM encoding. FITS files
    are converted via generate_preview when possible.
    """
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() in (".fit", ".fits", ".fts"):
        if not working_dir:
            return None
        from muphrid.tools.utility.t22_generate_preview import generate_preview
        expected = Path(working_dir) / "previews" / f"preview_{p.stem}.jpg"
        if expected.exists():
            return str(expected)
        try:
            result = generate_preview(
                working_dir=working_dir,
                fits_path=str(p),
                format="jpg",
                quality=95,
                auto_stretch_linear=is_linear,
            )
            return result.get("preview_path")
        except Exception as e:
            logger.warning(f"present_images preview generation failed for {p.name}: {e}")
            return None
    return str(p)


@tool(args_schema=PresentImagesInput)
def present_images(
    images: list[ImageEntry],
    title: str,
    description: str = "",
    state: Annotated[AstroState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    Present images for visual inspection and comparison.

    analyze_image is your primary diagnostic tool — use it first for
    data-driven decisions. Call present_images when visual inspection
    is needed to evaluate results that metrics alone cannot fully capture:
    stretch character, gradient artifacts, star rendering, color balance.

    Do NOT use as a default step after every tool — most decisions are
    fully served by analyze_image data.
    """
    valid_images: list[ImageEntry] = []
    for img in images:
        if Path(img.path).exists():
            valid_images.append(img)
        else:
            logger.warning(f"present_images: path not found: {img.path}")

    if not valid_images:
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps({"status": "error", "message": "No valid image paths found."}),
                tool_call_id=tool_call_id,
            )],
        })

    # Build the result the presenter layer reads from the ToolMessage stream
    result = {
        "status": "presented",
        "title": title,
        "description": description,
        "images": [
            {"path": img.path, "label": img.label, "metrics": img.metrics}
            for img in valid_images
        ],
        "count": len(valid_images),
    }

    # Build new visual_context: replace previous present_images entries with
    # this call's set; preserve other sources (hitl_variant, phase_carry).
    state = state or {}
    working_dir = state.get("dataset", {}).get("working_dir", "")
    is_linear = state.get("metrics", {}).get("is_linear_estimate", True)
    phase_value = state.get("phase")
    if hasattr(phase_value, "value"):
        phase_value = phase_value.value

    existing = list(state.get("visual_context", []) or [])
    other_sources = [r for r in existing if r.get("source") != "present_images"]

    new_refs: list[VisualRef] = []
    for img in valid_images:
        preview = _to_preview_path(img.path, working_dir, is_linear)
        if preview is None:
            continue
        new_refs.append(VisualRef(
            path=preview,
            label=img.label,
            source="present_images",
            phase=str(phase_value) if phase_value else "",
        ))

    new_visual_context = other_sources + new_refs

    logger.info(
        f"present_images: {title} — {len(valid_images)} images "
        f"(visual_context: +{len(new_refs)} present_images entries)"
    )

    return Command(update={
        "visual_context": new_visual_context,
        "messages": [ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_call_id,
        )],
    })
