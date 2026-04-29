# Muphrid - LLM agent for autonomous astrophotography post-processing
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

Fit a smooth background model to a FITS image and subtract or divide it out.

Methods:
  - graxpert   : GraXpert AI background extraction (subprocess call). The
                 background model is non-parametric, learned from training
                 data. No manual sample placement.
  - polynomial : Siril `background -gradient` fits a 2D polynomial of
                 user-controlled degree to automatically-placed sample
                 tiles. Deterministic, no AI dependency.

The model is computed in pixel-value space; non-linear inputs produce a
different fit than linear inputs. paths.pre_gradient_image is preserved
across calls so chain=False starts each variant from the same original.

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

from muphrid.config import load_settings
from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class GraXpertBGEOptions(BaseModel):
    correction_type: str = Field(
        default="Subtraction",
        description=(
            "Subtraction: result = input − model. Removes additive offsets "
            "in pixel value. "
            "Division: result = input / model. Removes multiplicative scale "
            "variation in pixel value."
        ),
    )
    smoothing: float = Field(
        description=(
            "Smoothing strength 0–1 applied to the AI background model. "
            "Controls how closely the model follows pixel data. "
            "Low smoothing produces a fine-grained model that tracks small "
            "spatial variations, including extended sources at low contrast. "
            "High smoothing produces a coarse model that captures only "
            "large-scale variation and ignores small-scale structure."
        ),
    )
    save_background_model: bool = Field(
        default=True,
        description=(
            "When True, GraXpert also writes the extracted background "
            "model as a separate FITS alongside the corrected output."
        ),
    )
    ai_version: str = Field(
        default="",
        description=(
            "GraXpert BGE AI model version in n.n.n format (e.g. '2.0.2'). "
            "Empty string lets GraXpert pick its bundled default. Pin a "
            "specific version when you want reproducible runs or want to "
            "compare two models on a problematic target."
        ),
    )
    # Note on GraXpert BGE knobs: -batch_size and -gpu are valid for the
    # `denoising` subcommand (see t12 noise_reduction) but not for the
    # `background-extraction` subcommand on the GraXpert versions we've
    # tested. Those fields were dropped here after the BGE subprocess
    # rejected the flags with a usage error. If a future GraXpert version
    # adds them to BGE, re-introduce as Optional fields and pass only
    # when set.


class PolynomialBGEOptions(BaseModel):
    degree: int = Field(
        default=2,
        ge=1,
        le=4,
        description=(
            "Polynomial degree of the 2D background model. "
            "1 = planar (bias + linear tilt). "
            "2 = quadratic surface. "
            "3 = cubic surface. "
            "4 = quartic surface (more flexible, fits finer-scale variation)."
        ),
    )
    correction_type: str = Field(
        default="Subtraction",
        description=(
            "Subtraction: result = input − model. "
            "Division: result = input / model."
        ),
    )
    samples_per_line: int = Field(
        default=20,
        ge=5,
        le=60,
        description=(
            "Siril -samples parameter. Number of sample tiles laid out per "
            "image side for fitting. Higher = denser sampling grid."
        ),
    )
    tolerance: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Siril -tolerance parameter for sample rejection (sigma units). "
            "Sample tiles whose pixel statistics deviate from neighbors by "
            "more than this are rejected from the fit."
        ),
    )


class RemoveGradientInput(BaseModel):
    method: str = Field(
        default="graxpert",
        description=(
            "Background-extraction method:\n"
            "  'graxpert'   — GraXpert AI BGE (uses graxpert_options).\n"
            "  'polynomial' — Siril 2D polynomial fit (uses polynomial_options)."
        ),
    )
    graxpert_options: GraXpertBGEOptions | None = Field(
        default=None,
        description="GraXpert-method parameters. Required when method='graxpert'.",
    )
    polynomial_options: PolynomialBGEOptions | None = Field(
        default=None,
        description="Polynomial-method parameters. Required when method='polynomial'.",
    )
    chain: bool = Field(
        default=False,
        description=(
            "False: read paths.pre_gradient_image (or current_image if no "
            "pre_gradient_image is set) so each call produces an "
            "independent variant from the same source. "
            "True: read paths.current_image so the operation chains on "
            "top of any prior gradient correction applied this session."
        ),
    )


# ── GraXpert backend ───────────────────────────────────────────────────────────

def _normalize_correction_type(raw: str) -> str:
    """GraXpert requires capitalized correction type: 'Subtraction' or 'Division'."""
    mapping = {"subtraction": "Subtraction", "division": "Division"}
    return mapping.get(raw.lower(), raw.capitalize())


def _auto_suffix_graxpert(options: GraXpertBGEOptions, chain: bool) -> str:
    """Variant suffix encoding the GraXpert call's distinguishing parameters."""
    correction = options.correction_type.lower()[:3]  # "sub" or "div"
    smoothing = f"s{int(options.smoothing * 100):03d}"
    base = f"bge_{correction}_{smoothing}"
    return f"chain_{base}" if chain else base


def _auto_suffix_polynomial(options: PolynomialBGEOptions, chain: bool) -> str:
    """Variant suffix encoding the polynomial call's distinguishing parameters."""
    correction = options.correction_type.lower()[:3]
    base = f"poly_d{options.degree}_n{options.samples_per_line}_{correction}"
    return f"chain_{base}" if chain else base


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

    # GraXpert CLI requires the positional input filename LAST, after all
    # flags. Earlier code placed it second and worked on permissive versions
    # but failed on stricter argparse setups. Match the t12 denoise pattern
    # (which has always worked) by appending the input path at the end.
    cmd: list[str] = [
        graxpert_bin,
        "-cli",
        "-cmd", "background-extraction",
        "-output", str(output_target),
        "-correction", correction,
        "-smoothing", str(options.smoothing),
    ]
    if options.ai_version:
        cmd.extend(["-ai_version", options.ai_version])
    if options.save_background_model:
        cmd.append("-bg")
    cmd.append(str(image_path))

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


# ── Polynomial backend ─────────────────────────────────────────────────────────

def _run_polynomial_bge(
    image_path: Path,
    options: PolynomialBGEOptions,
    output_stem: str,
    working_dir: str,
) -> tuple[Path, None]:
    """
    Fit and remove a 2D polynomial background via Siril `background -gradient`.

    Siril's background command operates on the loaded image and applies the
    correction in place; the corrected result is then saved. Polynomial fits
    do not produce a separate background model file, so the second tuple
    element is always None (matching _run_graxpert_bge's signature).

    Siril command surface (Siril 1.4):
        background -gradient -degree=N -samples=M -tolerance=T [-mul]
            -mul flag selects multiplicative (Division) correction;
            its absence selects additive (Subtraction).
    """
    correction = options.correction_type.lower()
    bg_cmd = (
        f"background -gradient "
        f"-degree={options.degree} "
        f"-samples={options.samples_per_line} "
        f"-tolerance={options.tolerance}"
    )
    if correction == "division":
        bg_cmd += " -mul"

    commands = [
        f"load {image_path.stem}",
        bg_cmd,
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=180)

    output_path = _probe_output_path(output_stem, image_path.parent)
    if output_path is None:
        raise FileNotFoundError(
            f"Siril `background -gradient` did not produce expected output "
            f"matching {output_stem}.* in {image_path.parent}"
        )
    return output_path, None


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=RemoveGradientInput)
def remove_gradient(
    method: str = "graxpert",
    graxpert_options: GraXpertBGEOptions | None = None,
    polynomial_options: PolynomialBGEOptions | None = None,
    chain: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Fit and remove a 2D background model from the working image.

    method='graxpert'  — GraXpert AI BGE via subprocess. Non-parametric model
                         learned by the AI. Knobs in graxpert_options:
                         correction_type, smoothing, save_background_model,
                         ai_version, batch_size, gpu.

    method='polynomial' — Siril `background -gradient` 2D polynomial fit.
                         Knobs in polynomial_options: degree (1–4),
                         correction_type, samples_per_line (5–60),
                         tolerance.

    Source-image selection:
      chain=False — read paths.pre_gradient_image (or current_image if no
                    pre_gradient image is set). Each call produces an
                    independent variant from the same original.
      chain=True  — read paths.current_image. The new operation chains on
                    top of any previous gradient correction applied this
                    session.

    Output is promoted to paths.current_image; the original input is
    preserved as paths.pre_gradient_image. Variant naming is auto-generated
    from method + parameters; pass-through chaining adds a 'chain_' prefix.
    """
    working_dir = state["dataset"]["working_dir"]

    method_norm = method.lower()
    if method_norm not in ("graxpert", "polynomial"):
        raise ValueError(
            f"remove_gradient: unknown method '{method}'. "
            f"Valid: graxpert, polynomial."
        )
    if method_norm == "polynomial" and polynomial_options is None:
        # Polynomial defaults are all sensible; instantiate when omitted.
        polynomial_options = PolynomialBGEOptions()
    if method_norm == "graxpert" and graxpert_options is None:
        raise ValueError(
            "remove_gradient: method='graxpert' requires graxpert_options "
            "(smoothing has no default — agent must choose a value based on "
            "the image's expected gradient complexity)."
        )

    # Determine source image
    if chain:
        image_path = state["paths"]["current_image"]
        original_image_path = state["paths"].get("pre_gradient_image") or image_path
    else:
        original_image_path = (
            state["paths"].get("pre_gradient_image")
            or state["paths"]["current_image"]
        )
        image_path = original_image_path

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_stem = Path(original_image_path).stem

    if method_norm == "graxpert":
        suffix = _auto_suffix_graxpert(graxpert_options, chain)
        output_stem = f"{original_stem}_{suffix}"
        processed_path, bg_model_path = _run_graxpert_bge(
            img_path, graxpert_options, output_stem
        )
        settings_used = {
            "correction_type": graxpert_options.correction_type,
            "smoothing": graxpert_options.smoothing,
            "save_background_model": graxpert_options.save_background_model,
            "ai_version": graxpert_options.ai_version,
        }
        variant_label = (
            f"graxpert {graxpert_options.correction_type}, "
            f"smoothing {graxpert_options.smoothing}"
        )
    else:  # polynomial
        suffix = _auto_suffix_polynomial(polynomial_options, chain)
        output_stem = f"{original_stem}_{suffix}"
        processed_path, bg_model_path = _run_polynomial_bge(
            img_path, polynomial_options, output_stem, working_dir
        )
        settings_used = {
            "correction_type": polynomial_options.correction_type,
            "degree": polynomial_options.degree,
            "samples_per_line": polynomial_options.samples_per_line,
            "tolerance": polynomial_options.tolerance,
        }
        variant_label = (
            f"polynomial degree {polynomial_options.degree}, "
            f"{polynomial_options.correction_type}"
        )

    source_label = "chained" if chain else "original"

    summary = {
        "output_path": str(processed_path),
        "method": method_norm,
        "variant_label": f"{variant_label} (from {source_label})",
        "source": source_label,
        "pre_gradient_image": str(original_image_path),
        "background_model_path": str(bg_model_path) if bg_model_path else None,
        "settings": {**settings_used, "chain": chain},
    }

    return Command(update={
        "paths": {
            **state["paths"],
            "current_image": str(processed_path),
            "pre_gradient_image": str(original_image_path),
        },
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    })
