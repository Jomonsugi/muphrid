# AstroAgent - LLM agent for autonomous astrophotography post-processing
# Copyright (C) 2026 Micah Shanks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
T09 — remove_gradient

Remove large-scale background gradients caused by light pollution, moon glow,
or vignetting residuals that survived flat calibration. Must be done in linear
space before stretch — gradient removal in non-linear space corrupts color.

Backend: GraXpert AI (direct subprocess). Handles complex, irregular gradients
without manual sample placement. GraXpert automatically uses the latest
available BGE AI model.

"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from astro_agent.config import load_settings
from astro_agent.graph.state import AstroState


# ── Pydantic input schema ──────────────────────────────────────────────────────

class GraXpertBGEOptions(BaseModel):
    correction_type: str = Field(
        default="Subtraction",
        description=(
            "Subtraction: removes additive gradients (light pollution, sky glow) — "
            "correct for the vast majority of images. "
            "Division: removes multiplicative gradients (residual vignetting after "
            "flat calibration). Use only when flat frames did not fully correct "
            "the vignetting pattern."
        ),
    )
    smoothing: float = Field(
        description=(
            "Smoothing strength 0–1 applied to the AI background model. "
            "Controls how closely the model follows pixel data. "
            "Low smoothing produces a fine-grained model that can mistake "
            "extended emission (nebulae, galaxy halos) for background and "
            "subtract it. High smoothing produces a coarse model that only "
            "captures large-scale gradients and preserves extended structure "
            "but may under-fit complex gradient edges. "
            "You must choose this value based on the target and data — "
            "there is no safe default."
        ),
    )
    save_background_model: bool = Field(
        default=True,
        description=(
            "Also save the extracted background model as a separate FITS. "
            "Useful for inspecting what was removed and diagnosing over-correction."
        ),
    )


class RemoveGradientInput(BaseModel):
    graxpert_options: GraXpertBGEOptions = Field(default_factory=GraXpertBGEOptions)
    chain: bool = Field(
        default=False,
        description=(
            "If False (default), processes the original pre-gradient image, "
            "producing an independent variant. Each call with different parameters "
            "creates a fresh result for comparison. "
            "If True, processes the current image, which may already have gradient "
            "removal applied. This enables multi-pass workflows where different "
            "correction types are applied sequentially (e.g. subtraction for light "
            "pollution followed by division for vignetting residual)."
        ),
    )


# ── GraXpert backend ───────────────────────────────────────────────────────────

def _normalize_correction_type(raw: str) -> str:
    """GraXpert requires capitalized correction type: 'Subtraction' or 'Division'."""
    mapping = {"subtraction": "Subtraction", "division": "Division"}
    return mapping.get(raw.lower(), raw.capitalize())


def _auto_suffix(options: GraXpertBGEOptions, chain: bool) -> str:
    """Generate a descriptive suffix from parameters."""
    correction = options.correction_type.lower()[:3]  # "sub" or "div"
    smoothing = f"s{int(options.smoothing * 100):03d}"  # "s060", "s080"
    if chain:
        return f"chain_{correction}_{smoothing}"
    return f"bge_{correction}_{smoothing}"


def _probe_output_path(base_stem: str, parent: Path) -> Path | None:
    """GraXpert appends its own extension — probe common possibilities."""
    candidates = [
        parent / f"{base_stem}.fits",
        parent / f"{base_stem}.fit",
        parent / f"{base_stem}.fits.fits",
        parent / f"{base_stem}.fit.fits",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_graxpert_bge(
    image_path: Path,
    options: GraXpertBGEOptions,
    output_stem: str,
) -> tuple[Path, Path | None]:
    """
    Call GraXpert directly via subprocess for AI-based background extraction.
    Returns (processed_image_path, background_model_path | None).
    Raises RuntimeError on non-zero exit; FileNotFoundError if output is missing.
    """
    settings = load_settings()
    graxpert_bin = settings.graxpert_bin

    correction = _normalize_correction_type(options.correction_type)

    # Pass output path WITHOUT extension — GraXpert appends .fits itself
    output_target = image_path.parent / output_stem

    cmd: list[str] = [
        graxpert_bin,
        str(image_path),
        "-cli",
        "-cmd", "background-extraction",
        "-output", str(output_target),
        "-correction", correction,
        "-smoothing", str(options.smoothing),
    ]
    if options.save_background_model:
        cmd.append("-bg")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(
            f"GraXpert BGE failed (exit {result.returncode}):\n"
            f"{result.stderr or result.stdout}"
        )

    output_path = _probe_output_path(output_stem, image_path.parent)
    if output_path is None:
        raise FileNotFoundError(
            f"GraXpert did not produce expected output matching "
            f"{output_stem}.* in {image_path.parent}\n"
            f"stdout: {result.stdout}"
        )

    bg_model_path = _probe_output_path(f"{output_stem}_background", image_path.parent)
    return output_path, bg_model_path


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=RemoveGradientInput)
def remove_gradient(
    graxpert_options: GraXpertBGEOptions | None = None,
    chain: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Remove large-scale background gradients from a linear FITS image using
    GraXpert AI background extraction.

    Each call produces an independent variant from the original pre-gradient
    image. Variant names are auto-generated from parameters (correction type +
    smoothing). The result includes the variant name and file path.

    To compare variants, call present_images with the paths from each result.

    If chain=True, the tool processes the current image, which may already have
    gradient removal applied. This enables multi-pass workflows where different
    correction types are applied sequentially (e.g. subtraction then division).
    Chained variants are labeled as such in the output.

    GraXpert requires no manual sample placement and handles multi-source,
    irregular gradients that polynomial methods cannot model — light pollution
    gradients, moon glow, and complex vignetting residuals.

    Correction type guidance:
    - Subtraction (default): additive gradients from light pollution or sky glow.
    - Division: multiplicative gradients from residual vignetting after flats.
"""
    working_dir = state["dataset"]["working_dir"]

    if graxpert_options is None:
        graxpert_options = GraXpertBGEOptions()

    # Determine source image
    if chain:
        # Intentional chaining — process the current (possibly already corrected) image
        image_path = state["paths"]["current_image"]
        original_image_path = state["paths"].get("pre_gradient_image") or image_path
    else:
        # Default: always start from the original pre-gradient image
        original_image_path = (
            state["paths"].get("pre_gradient_image")
            or state["paths"]["current_image"]
        )
        image_path = original_image_path

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Auto-generate variant name from parameters
    suffix = _auto_suffix(graxpert_options, chain)
    # Use original image stem for naming (not the chained intermediate)
    original_stem = Path(original_image_path).stem
    output_stem = f"{original_stem}_{suffix}"

    processed_path, bg_model_path = _run_graxpert_bge(img_path, graxpert_options, output_stem)

    # Build descriptive label
    correction_label = graxpert_options.correction_type
    smoothing_label = graxpert_options.smoothing
    source_label = "chained" if chain else "original"
    variant_label = f"{correction_label}, smoothing {smoothing_label} (from {source_label})"

    summary = {
        "output_path": str(processed_path),
        "variant_label": variant_label,
        "source": source_label,
        "pre_gradient_image": str(original_image_path),
        "background_model_path": str(bg_model_path) if bg_model_path else None,
        "settings": {
            "correction_type": graxpert_options.correction_type,
            "smoothing": graxpert_options.smoothing,
            "save_background_model": graxpert_options.save_background_model,
            "chain": chain,
        },
    }

    return Command(update={
        "paths": {**state["paths"], "current_image": str(processed_path), "pre_gradient_image": str(original_image_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
