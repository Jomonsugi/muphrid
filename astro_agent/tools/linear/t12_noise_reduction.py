"""
T12 — noise_reduction

AI-based noise reduction on linear FITS data using GraXpert.

Why GraXpert AI denoising:
  Applied in linear space where noise statistics are approximately Gaussian
  after stacking — this is the mathematically correct moment to denoise.
  Noise reduced here is not amplified by the stretch transformation, enabling
  more aggressive stretching without noise amplification and producing a
  cleaner base for all downstream processing.

  GraXpert's deep-learning denoiser is trained specifically on astrophotography
  data. It preserves faint nebulosity, star halos, and fine structure better
  than classical algorithms (NL-Bayes, wavelets) because the model has learned
  what signal vs. noise looks like in this domain.

GraXpert command used:
    GraXpert <input> -cli -cmd denoising -output <output>
        -ai_version <n.n.n> -strength <0.0-1.0>
        -batch_size <int> [-gpu {true,false}]
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

import numpy as np
from astropy.io import fits as astropy_fits

from astro_agent.config import load_settings
from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class DenoiseOptions(BaseModel):
    strength: float = Field(
        default=0.5,
        description=(
            "Denoising strength 0.0–1.0. "
            "Higher values remove more noise but risk smoothing fine detail. "
            "0.5 is a reliable starting point for most stacks. "
            "0.3–0.4 for high-SNR images with fine structure (galaxy spirals, "
            "tight planetary nebulae). "
            "0.6–0.8 for high-noise images (short exposures, high ISO, < 2h integration)."
        ),
    )
    ai_version: str = Field(
        default="2.0.0",
        description=(
            "Denoise AI model version in n.n.n format. "
            "GraXpert will use a locally cached model or download from remote if needed. "
            "Prefer locally cached versions to avoid network latency. "
            "If a run fails reporting the version unavailable, retry with a version "
            "from the list GraXpert provides in its error output. "
            "Upgrade to a newer remote version (e.g. '3.0.2') only when the user "
            "explicitly requests it — newer models may produce different noise/detail "
            "trade-offs and affect reproducibility."
        ),
    )
    batch_size: int = Field(
        default=4,
        description=(
            "AI inference batch size. Larger values speed up processing but use more "
            "GPU/CPU memory. Reduce to 1 or 2 if you encounter out-of-memory errors."
        ),
    )
    gpu: bool = Field(
        default=True,
        description=(
            "Enable GPU acceleration for AI inference. "
            "Set False to force CPU (slower but useful for debugging or when "
            "GPU memory is insufficient)."
        ),
    )


class NoiseReductionInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the linear FITS image to denoise. "
            "Must be post gradient-removal and color calibration (T09, T10, T11)."
        )
    )
    options: DenoiseOptions = Field(default_factory=DenoiseOptions)


# ── Noise measurement ──────────────────────────────────────────────────────────

def _measure_bgnoise(stem: str, working_dir: str) -> float | None:
    """Run Siril bgnoise on the loaded image and parse the result."""
    from astro_agent.tools._siril import SirilError
    import re
    try:
        result = run_siril_script(
            [f"load {stem}", "bgnoise"],
            working_dir=working_dir,
            timeout=30,
        )
        m = re.search(r"Background\s+noise[^\d]*([\d.e+-]+)", result.stdout, re.IGNORECASE)
        if m:
            return float(m.group(1))
    except (SirilError, ValueError):
        pass
    return None


# ── GraXpert denoising ─────────────────────────────────────────────────────────

# GraXpert / ONNX zero-pixel bug:
# GraXpert's neural network normalizes tiles by channel mean/std internally.
# When a tile or image region contains exact-zero pixels (common in X-Trans
# debayering and after Siril stacking with `framing=min`), the normalization
# divides by zero → NaN propagates through the inference graph. This is a
# known ONNX/GraXpert issue affecting X-Trans sensors and mosaiced images.
# Fix: apply a tiny pedestal (1e-4) to zero-valued pixels before inference.
# At 32-bit float precision this is ~0.01% of the full-scale range and is
# completely invisible in the final image.
_GRAXPERT_ZERO_PEDESTAL = 1e-4


def _apply_zero_pedestal(fits_path: Path) -> int:
    """Lift sub-pedestal and NaN pixels to _GRAXPERT_ZERO_PEDESTAL in-place.

    GraXpert's ONNX inference normalizes tiles by local mean/std.
    Pixels near zero (< ~1e-6 in X-Trans stacks) are mathematically
    zero to the network but cause 0/0 in normalization → NaN output.
    NaN pixels (Siril rejection/framing borders) also propagate as NaN
    through the inference graph and corrupt entire tiles.
    The fix: clamp both to pedestal — lifts the black point by
    ~0.01% of full scale, completely invisible in the final image.

    Returns the number of pixels modified.
    """
    with astropy_fits.open(str(fits_path), mode="update") as hdul:
        data = hdul[0].data
        mask = (data < _GRAXPERT_ZERO_PEDESTAL) | np.isnan(data)
        count = int(np.sum(mask))
        if count > 0:
            data[mask] = _GRAXPERT_ZERO_PEDESTAL
            hdul.flush()
    return count


def _run_graxpert_denoise(
    image_path: Path,
    options: DenoiseOptions,
) -> tuple[Path, int]:
    """Run GraXpert denoising. Returns (output_path, zero_pixels_pedestaled)."""
    settings = load_settings()

    output_path = image_path.parent / f"{image_path.stem}_denoise.fits"

    # GraXpert always appends .fits to the -output argument, so pass the
    # stem without extension to land at the correct final path.
    output_stem = str(output_path.with_suffix(""))

    # Apply pedestal to a working copy so the original input is not modified.
    import shutil
    graxpert_input = image_path.parent / f"{image_path.stem}_graxpert_input.fits"
    shutil.copy2(str(image_path), str(graxpert_input))
    zero_count = _apply_zero_pedestal(graxpert_input)

    # GraXpert CLI positional: filename must come last (after all flags).
    # Use the pedestaled working copy as input; rename output to expected path.
    working_output = image_path.parent / f"{graxpert_input.stem}_denoise.fits"
    working_stem = str(working_output.with_suffix(""))

    cmd = [
        settings.graxpert_bin,
        "-cli",
        "-cmd", "denoising",
        "-output", working_stem,
        "-ai_version", options.ai_version,
        "-strength", str(options.strength),
        "-batch_size", str(options.batch_size),
        "-gpu", "true" if options.gpu else "false",
        str(graxpert_input),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Clean up working copy regardless of outcome
    graxpert_input.unlink(missing_ok=True)

    combined = (result.stderr + result.stdout).strip()

    if result.returncode != 0:
        raise RuntimeError(
            f"GraXpert denoising failed (exit {result.returncode}):\n{combined}"
        )

    if not working_output.exists():
        raise FileNotFoundError(
            f"GraXpert did not produce expected output: {working_output}\n{combined}"
        )

    # Move to the expected output path (named after original stem, not working copy)
    working_output.rename(output_path)

    return output_path, zero_count


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=NoiseReductionInput)
def noise_reduction(
    working_dir: str,
    image_path: str,
    options: DenoiseOptions | None = None,
) -> dict:
    """
    AI-based noise reduction on the linear FITS image using GraXpert.

    Denoising in linear space is the gold standard: noise statistics are
    approximately Gaussian after stacking, and the ML model has the most
    information to work with. Noise reduced here is not amplified by the
    stretch transformation.

    Tune strength based on SNR: start at 0.5, lower for high-SNR images with
    fine structure, raise for short-exposure or high-ISO captures.

    Returns noise_before, noise_after, and noise_reduction_pct for quantitative
    assessment of the denoising result.
    """
    if options is None:
        options = DenoiseOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    noise_before = _measure_bgnoise(img_path.stem, working_dir)

    output_path, zero_pixels_pedestaled = _run_graxpert_denoise(img_path, options)

    noise_after = _measure_bgnoise(output_path.stem, working_dir)

    reduction_pct: float | None = None
    if noise_before and noise_after and noise_before > 0:
        reduction_pct = round((1.0 - noise_after / noise_before) * 100, 1)

    return {
        "denoised_image_path":    str(output_path),
        "noise_before":           noise_before,
        "noise_after":            noise_after,
        "noise_reduction_pct":    reduction_pct,
        "zero_pixels_pedestaled": zero_pixels_pedestaled,
    }
