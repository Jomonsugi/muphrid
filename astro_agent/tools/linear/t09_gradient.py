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


# ── GraXpert backend ───────────────────────────────────────────────────────────

def _normalize_correction_type(raw: str) -> str:
    """GraXpert requires capitalized correction type: 'Subtraction' or 'Division'."""
    mapping = {"subtraction": "Subtraction", "division": "Division"}
    return mapping.get(raw.lower(), raw.capitalize())



def _probe_output_path(base_stem: Path, parent: Path) -> Path | None:
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
    output_stem = image_path.parent / f"{image_path.stem}_bge"

    # Omit -ai_version so GraXpert uses the latest available model automatically.
    cmd: list[str] = [
        graxpert_bin,
        str(image_path),
        "-cli",
        "-cmd", "background-extraction",
        "-output", str(output_stem),
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

    output_path = _probe_output_path(f"{image_path.stem}_bge", image_path.parent)
    if output_path is None:
        raise FileNotFoundError(
            f"GraXpert did not produce expected output matching "
            f"{image_path.stem}_bge.* in {image_path.parent}\n"
            f"stdout: {result.stdout}"
        )

    bg_model_path = _probe_output_path(f"{image_path.stem}_bge_background", image_path.parent)
    return output_path, bg_model_path


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=RemoveGradientInput)
def remove_gradient(
    graxpert_options: GraXpertBGEOptions | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Remove large-scale background gradients from a linear FITS image using
    GraXpert AI background extraction.

    GraXpert requires no manual sample placement and handles multi-source,
    irregular gradients that polynomial methods cannot model — light pollution
    gradients, moon glow, and complex vignetting residuals.

    Correction type guidance:
    - Subtraction (default): additive gradients from light pollution or sky glow.
    - Division: multiplicative gradients from residual vignetting after flats.
"""
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    if graxpert_options is None:
        graxpert_options = GraXpertBGEOptions()

    original_image_path = image_path
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    processed_path, bg_model_path = _run_graxpert_bge(img_path, graxpert_options)

    summary = {
        "output_path": str(processed_path),
        "pre_gradient_image": str(original_image_path),
        "background_model_path": str(bg_model_path) if bg_model_path else None,
        "settings": {
            "correction_type": graxpert_options.correction_type,
            "smoothing": graxpert_options.smoothing,
            "save_background_model": graxpert_options.save_background_model,
        },
    }

    return Command(update={
        "paths": {**state["paths"], "current_image": str(processed_path), "pre_gradient_image": str(original_image_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
