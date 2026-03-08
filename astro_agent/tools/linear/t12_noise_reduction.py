"""
T12 — noise_reduction

Reduce noise while preserving signal structure. Applied in linear space where
noise statistics are approximately Gaussian after stacking — this is the
mathematically correct moment to denoise. Noise reduced here will not be
amplified by the stretch transformation.

Primary backend: Siril NL-Bayes (`denoise`) — fast, excellent detail
preservation, optionally boosted with DA3D or SOS algorithms.

Alternative backend: GraXpert AI denoising — deep-learning approach, very
effective but slower. Uses denoise-ai-models/2.0.0 (cached locally).
GraXpert denoising strength is controlled via a temporary preferences file
(strength is not a direct CLI flag in GraXpert 3.0.2).

A mild second pass post-stretch on the starless image may be warranted for
very noisy captures, but linear-space denoising is always the primary pass.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.config import load_settings
from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SirilDenoiseOptions(BaseModel):
    modulation: float = Field(
        default=0.9,
        description=(
            "Blend factor between the denoised output and the original (0.0–1.0). "
            "1.0: full denoising (maximum noise reduction). "
            "0.7–0.9: blends in 10–30% of the original — preserves faint detail "
            "and avoids over-smoothing. Recommended range: 0.7–0.95."
        ),
    )
    method: str = Field(
        default="standard",
        description=(
            "standard: NL-Bayes — fast, high quality. Good default for most images. "
            "da3d: NL-Bayes + DA3D boosting — best detail preservation at the cost "
            "of 2–3× processing time. Preferred for images with fine structure "
            "(galaxy spirals, planetary nebulae). "
            "sos: SOS (Sum of Squares) boosting — iterative, use when standard "
            "and da3d leave visible noise in low-SNR regions."
        ),
    )
    sos_iterations: int = Field(
        default=3,
        description=(
            "Number of SOS boosting iterations. Only used when method=sos. "
            "More iterations = stronger denoising but slower. 3–5 is typical."
        ),
    )
    sos_rho: float | None = Field(
        default=None,
        description=(
            "SOS rho parameter (0 < rho < 1). Controls the amount of noisy image "
            "mixed in between iterations. Only used with method=sos. "
            "Lower values = more aggressive denoising per iteration. "
            "Null = Siril default."
        ),
    )
    independent_channels: bool = Field(
        default=False,
        description=(
            "Process each color channel independently (-indep). "
            "Default (False) processes all channels jointly, which is better for "
            "OSC/DSLR color images. Set True only for unusual per-channel noise."
        ),
    )
    use_vst: bool = Field(
        default=False,
        description=(
            "Apply the generalised Anscombe variance stabilising transform (VST) "
            "prior to NL-Bayes (-vst). Useful for photon-starved images such as "
            "single subs, where noise follows a Poisson distribution rather than "
            "Gaussian. For stacked images, VST is usually not beneficial. "
            "Cannot be combined with method=da3d or method=sos."
        ),
    )
    apply_cosmetic: bool = Field(
        default=True,
        description=(
            "Apply automatic cosmetic correction (find_cosme) before denoising. "
            "Removes residual hot and cold pixels that would otherwise be "
            "propagated and amplified by the denoising algorithm. "
            "Disable (-nocosmetic) only if cosmetic correction was already applied "
            "explicitly earlier in the pipeline."
        ),
    )


class GraXpertDenoiseOptions(BaseModel):
    strength: float = Field(
        default=0.5,
        description=(
            "Denoising strength 0.0–1.0. "
            "Higher values remove more noise but risk smoothing fine detail. "
            "0.5 is a reliable starting point for most stacks. "
            "0.3–0.4 for high-SNR images with fine structure (galaxy spirals). "
            "0.6–0.8 for high-noise images (short exposures, high ISO)."
        ),
    )
    ai_version: str = Field(
        default="2.0.0",
        description=(
            "Denoise AI model version. 2.0.0 is the current best available "
            "(linear data, superior structure preservation over 1.0.0). "
            "Model cached at ~/Library/Application Support/GraXpert/denoise-ai-models/."
        ),
    )


class NoiseReductionInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the linear FITS image to denoise. "
            "Should be post gradient-removal and color calibration (T09, T10, T11)."
        )
    )
    backend: str = Field(
        default="siril",
        description=(
            "siril: NL-Bayes denoising via Siril (primary, fast, proven). "
            "graxpert: AI denoising via GraXpert subprocess (alternative, slower, "
            "deep-learning approach — may outperform Siril on very noisy images)."
        ),
    )
    siril_options: SirilDenoiseOptions = Field(default_factory=SirilDenoiseOptions)
    graxpert_options: GraXpertDenoiseOptions = Field(default_factory=GraXpertDenoiseOptions)


# ── Siril backend ──────────────────────────────────────────────────────────────

def _measure_bgnoise(stem: str, working_dir: str) -> float | None:
    """Run Siril bgnoise on the loaded image and parse the result."""
    from astro_agent.tools._siril import SirilError
    try:
        result = run_siril_script(
            [f"load {stem}", "bgnoise"],
            working_dir=working_dir,
            timeout=30,
        )
        import re
        m = re.search(r"Background\s+noise[^\d]*([\d.e+-]+)", result.stdout, re.IGNORECASE)
        if m:
            return float(m.group(1))
    except (SirilError, ValueError):
        pass
    return None


def _run_siril_denoise(
    image_path: Path,
    working_dir: str,
    options: SirilDenoiseOptions,
) -> Path:
    stem = image_path.stem
    output_stem = f"{stem}_denoise"

    denoise_cmd = f"denoise -mod={options.modulation}"
    if options.method == "da3d":
        denoise_cmd += " -da3d"
    elif options.method == "sos":
        denoise_cmd += f" -sos={options.sos_iterations}"
        if options.sos_rho is not None:
            denoise_cmd += f" -rho={options.sos_rho}"
    if options.use_vst and options.method == "standard":
        denoise_cmd += " -vst"
    if options.independent_channels:
        denoise_cmd += " -indep"
    if not options.apply_cosmetic:
        denoise_cmd += " -nocosmetic"

    commands = [
        f"load {stem}",
        denoise_cmd,
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=180)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"Siril denoise did not produce: {output_path}")

    return output_path


# ── GraXpert backend ───────────────────────────────────────────────────────────

def _run_graxpert_denoise(
    image_path: Path,
    options: GraXpertDenoiseOptions,
) -> Path:
    """
    Run GraXpert denoising via subprocess.
    Strength is not a direct CLI flag in GraXpert 3.0.2 — it is written to a
    temporary preferences JSON and passed via -preferences_file.
    """
    settings = load_settings()
    graxpert_bin = settings.graxpert_bin

    output_path = image_path.parent / f"{image_path.stem}_denoise.fits"

    # Build minimal preferences dict with the desired strength
    prefs = {
        "denoise_strength": options.strength,
        "ai_gpu_acceleration": True,
        "ai_batch_size": 4,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(prefs, tmp)
        prefs_path = tmp.name

    try:
        cmd = [
            graxpert_bin,
            str(image_path),
            "-cli",
            "-cmd", "denoising",
            "-output", str(output_path),
            "-ai_version", options.ai_version,
            "-preferences_file", prefs_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    finally:
        Path(prefs_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"GraXpert denoising failed (exit {result.returncode}):\n"
            f"{result.stderr or result.stdout}"
        )

    if not output_path.exists():
        raise FileNotFoundError(
            f"GraXpert did not produce expected output: {output_path}\n"
            f"stdout: {result.stdout}"
        )

    return output_path


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=NoiseReductionInput)
def noise_reduction(
    working_dir: str,
    image_path: str,
    backend: str = "siril",
    siril_options: SirilDenoiseOptions | None = None,
    graxpert_options: GraXpertDenoiseOptions | None = None,
) -> dict:
    """
    Reduce noise in the linear FITS image before stretching.

    Denoising in linear space is the gold standard — noise statistics are
    Gaussian here and the algorithm has the most information to work with.
    Applying denoising after stretch amplifies noise in shadow regions
    first and makes structure recovery harder.

    Backend guidance:
    - siril (default): NL-Bayes, fast, reliable. Use da3d method for best
      detail preservation on images with fine structure.
    - graxpert: Deep-learning denoiser, may outperform NL-Bayes on very noisy
      captures or images with structured noise patterns.

    Use modulation < 1.0 (Siril) or strength < 1.0 (GraXpert) to blend in
    original signal and avoid over-smoothing. Run analyze_image before and
    after to compare noise_before vs noise_after and confirm SNR improvement
    without signal loss.
    """
    if siril_options is None:
        siril_options = SirilDenoiseOptions()
    if graxpert_options is None:
        graxpert_options = GraXpertDenoiseOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    noise_before = _measure_bgnoise(img_path.stem, working_dir)

    if backend == "graxpert":
        output_path = _run_graxpert_denoise(img_path, graxpert_options)
    else:
        output_path = _run_siril_denoise(img_path, working_dir, siril_options)

    noise_after = _measure_bgnoise(output_path.stem, working_dir)

    reduction_pct: float | None = None
    if noise_before and noise_after and noise_before > 0:
        reduction_pct = round((1.0 - noise_after / noise_before) * 100, 1)

    return {
        "denoised_image_path": str(output_path),
        "backend_used": backend,
        "noise_before": noise_before,
        "noise_after": noise_after,
        "noise_reduction_pct": reduction_pct,
    }
