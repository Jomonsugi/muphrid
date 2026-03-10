"""
T09 — remove_gradient

Remove large-scale background gradients caused by light pollution, moon glow,
or vignetting residuals that survived flat calibration. Must be done in linear
space before stretch — gradient removal in non-linear space corrupts color.

Backend: GraXpert AI (direct subprocess). Handles complex, irregular gradients
without manual sample placement. Confirmed working: v3.0.2, CoreML acceleration,
model bge-ai-models/1.0.1.

HITL: requires_visual_review=True by default. Gradient removal can introduce
subtle artefacts near extended emission nebulae or at image edges. Visual
inspection is mandatory in V1 before releasing this checkpoint to automation.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.config import load_settings


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
        default=0.5,
        description=(
            "Smoothing strength 0–1 applied to the AI background model. "
            "Higher values produce a smoother background model but may under-fit "
            "sharp gradient edges. 0.5 is a reliable default. "
            "Increase toward 1.0 for very smooth gradients; reduce toward 0.0 "
            "for complex multi-source gradients."
        ),
    )
    save_background_model: bool = Field(
        default=True,
        description=(
            "Also save the extracted background model as a separate FITS. "
            "Useful for inspecting what was removed and diagnosing over-correction."
        ),
    )
    ai_version: str = Field(
        default="1.0.1",
        description=(
            "BGE AI model version. 1.0.1 is the current best available. "
            "Model is cached at ~/Library/Application Support/GraXpert/bge-ai-models/."
        ),
    )


class RemoveGradientInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the linear FITS image to process. "
            "Must be a stacked, cropped FITS from T08."
        )
    )
    graxpert_options: GraXpertBGEOptions = Field(default_factory=GraXpertBGEOptions)


# ── GraXpert backend ───────────────────────────────────────────────────────────

def _normalize_correction_type(raw: str) -> str:
    """GraXpert requires capitalized correction type: 'Subtraction' or 'Division'."""
    mapping = {"subtraction": "Subtraction", "division": "Division"}
    return mapping.get(raw.lower(), raw.capitalize())


def _validate_ai_version(version: str) -> str:
    """Ensure ai_version matches semver-like N.N.N format required by GraXpert."""
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        raise ValueError(
            f"ai_version must be in N.N.N format (e.g. '1.0.1'). Got: {version!r}"
        )
    return version


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
    ai_version = _validate_ai_version(options.ai_version)

    # Pass output path WITHOUT extension — GraXpert appends .fits itself
    output_stem = image_path.parent / f"{image_path.stem}_bge"

    cmd: list[str] = [
        graxpert_bin,
        str(image_path),
        "-cli",
        "-cmd", "background-extraction",
        "-output", str(output_stem),
        "-correction", correction,
        "-smoothing", str(options.smoothing),
        "-ai_version", ai_version,
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
    working_dir: str,
    image_path: str,
    graxpert_options: GraXpertBGEOptions | None = None,
) -> dict:
    """
    Remove large-scale background gradients from a linear FITS image using
    GraXpert AI background extraction.

    GraXpert requires no manual sample placement and handles multi-source,
    irregular gradients that polynomial methods cannot model — light pollution
    gradients, moon glow, and complex vignetting residuals.

    Correction type guidance:
    - Subtraction (default): additive gradients from light pollution or sky glow.
    - Division: multiplicative gradients from residual vignetting after flats.

    Always run analyze_image before and after to measure the change in
    background_flatness_score and confirm the gradient was removed without
    over-correcting signal near bright nebulae or at image edges.

    HITL visual review is triggered automatically after this tool (V1).
    """
    if graxpert_options is None:
        graxpert_options = GraXpertBGEOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    processed_path, bg_model_path = _run_graxpert_bge(img_path, graxpert_options)

    return {
        "processed_image_path":  str(processed_path),
        "background_model_path": str(bg_model_path) if bg_model_path else None,
        "correction_type":       graxpert_options.correction_type,
        "smoothing":             graxpert_options.smoothing,
    }
