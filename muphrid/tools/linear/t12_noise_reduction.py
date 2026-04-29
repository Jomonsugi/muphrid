"""
T12 — noise_reduction

Reduce per-pixel noise in a FITS image. Three method families:

  - graxpert  : GraXpert AI denoiser (subprocess). Non-parametric model
                trained on astronomical data. Knobs: strength, ai_version,
                batch_size, gpu.
  - siril     : Siril native `denoise` command. Patch-based (NL-Bayes /
                DA3D / SOS algorithms). Knobs: algorithm, modulation,
                use_vst, independent_channels, apply_cosmetic.
  - bilateral : scikit-image bilateral filter. Edge-preserving smoothing
                via local weighted averaging in (intensity × spatial)
                space. Knobs: sigma_color, sigma_spatial, win_size,
                multichannel.

Operates on the working FITS image; output is promoted to
paths.current_image. Noise is measured before and after via Siril
`bgnoise` when possible, returning noise_before / noise_after /
noise_reduction_pct in the tool result.

Stretch is non-linear: a denoise pass applied before stretch produces
different end-state noise than the same pass applied after stretch.
The agent picks when to call this tool; the tool itself is order-agnostic.
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

import numpy as np
from astropy.io import fits as astropy_fits

from muphrid.config import load_settings
from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class GraXpertDenoiseOptions(BaseModel):
    strength: float = Field(
        description=(
            "Denoising strength 0.0–1.0. Higher values produce a stronger "
            "noise-reduction effect at the cost of fine-detail preservation. "
            "No default — the agent chooses based on the image's noise "
            "characteristics."
        ),
    )
    ai_version: str = Field(
        default="2.0.0",
        description=(
            "Denoise AI model version in n.n.n format. GraXpert will use a "
            "locally cached model or download from remote if needed. Empty "
            "string lets GraXpert pick its bundled default."
        ),
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        description=(
            "AI inference batch size. Higher values process more tiles per "
            "GPU/CPU forward pass at the cost of memory; lower values "
            "reduce memory pressure at the cost of throughput."
        ),
    )
    gpu: bool = Field(
        default=True,
        description=(
            "When True, GraXpert uses GPU acceleration if available; when "
            "False, forces CPU inference."
        ),
    )


class SirilDenoiseOptions(BaseModel):
    algorithm: str = Field(
        default="standard",
        description=(
            "Inner algorithm for Siril's `denoise` command:\n"
            "  'standard' — NL-Bayes (non-local means + Bayesian). Default. "
            "Optionally preceded by VST.\n"
            "  'da3d'     — Data-adaptive Dual Domain Denoising. Faster than "
            "standard, different denoising character.\n"
            "  'sos'      — Strengthen-Operate-Subtract iterative refinement. "
            "Uses sos_iterations as the iteration count."
        ),
    )
    modulation: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Siril -mod parameter. 1.0 applies the full denoise effect; "
            "0.0 applies none. Acts as a strength scaling on the algorithm's "
            "output."
        ),
    )
    sos_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of SOS refinement iterations when algorithm='sos'. "
            "Ignored for other algorithms."
        ),
    )
    use_vst: bool = Field(
        default=False,
        description=(
            "When True and algorithm='standard', applies a Variance "
            "Stabilizing Transform before NL-Bayes. Ignored for other "
            "algorithms."
        ),
    )
    independent_channels: bool = Field(
        default=False,
        description=(
            "When True, each channel is denoised independently (-indep). "
            "When False (default), channels are processed jointly."
        ),
    )
    apply_cosmetic: bool = Field(
        default=True,
        description=(
            "When True, Siril applies its cosmetic correction during "
            "denoise. When False, adds the -nocosmetic flag to skip it."
        ),
    )


class BilateralDenoiseOptions(BaseModel):
    sigma_color: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Standard deviation for the intensity (range) Gaussian. Pixels "
            "whose values differ by more than this are treated as belonging "
            "to different regions and contribute less to the smoothed "
            "output. Smaller values preserve more edges; larger values "
            "smooth across edges."
        ),
    )
    sigma_spatial: float = Field(
        default=15.0,
        ge=0.0,
        description=(
            "Standard deviation for the spatial Gaussian, in pixels. "
            "Controls how far the filter looks for similar pixels to "
            "average with. Larger values smooth over larger regions."
        ),
    )
    win_size: int | None = Field(
        default=None,
        description=(
            "Window size in pixels for the filter footprint. None lets "
            "skimage default to ~max(5, 2*ceil(3*sigma_spatial)+1)."
        ),
    )
    multichannel: bool = Field(
        default=True,
        description=(
            "When True (default), color images are treated as multichannel "
            "and the filter weights blend across channels. When False, "
            "each channel is filtered independently. Ignored for mono "
            "inputs."
        ),
    )


class NoiseReductionInput(BaseModel):
    method: str = Field(
        default="graxpert",
        description=(
            "Noise-reduction method:\n"
            "  'graxpert'  — GraXpert AI denoiser (uses graxpert_options).\n"
            "  'siril'     — Siril native denoise: NL-Bayes / DA3D / SOS "
            "(uses siril_options).\n"
            "  'bilateral' — scikit-image bilateral filter "
            "(uses bilateral_options)."
        ),
    )
    graxpert_options: GraXpertDenoiseOptions | None = Field(
        default=None,
        description="GraXpert-method parameters. Required when method='graxpert'.",
    )
    siril_options: SirilDenoiseOptions | None = Field(
        default=None,
        description="Siril-method parameters. Defaults applied when method='siril' and this is None.",
    )
    bilateral_options: BilateralDenoiseOptions | None = Field(
        default=None,
        description="Bilateral-method parameters. Defaults applied when method='bilateral' and this is None.",
    )


# ── Noise measurement ──────────────────────────────────────────────────────────

def _measure_bgnoise(stem: str, working_dir: str) -> tuple[float | None, str | None]:
    """
    Run Siril bgnoise on the loaded image and parse the result.

    Returns (noise_value, error_msg). If measurement succeeds, error_msg is None.
    If it fails, noise_value is None and error_msg explains why.
    """
    from muphrid.tools._siril import SirilError
    import re
    try:
        result = run_siril_script(
            [f"load {stem}", "bgnoise"],
            working_dir=working_dir,
            timeout=30,
        )
        m = re.search(r"Background\s+noise[^\d]*([\d.e+-]+)", result.stdout, re.IGNORECASE)
        if m:
            return float(m.group(1)), None
        return None, f"bgnoise ran but could not parse noise value from output: {result.stdout[-200:]}"
    except SirilError as e:
        return None, f"bgnoise Siril error: {e}"
    except ValueError as e:
        return None, f"bgnoise parse error: {e}"


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
    options: GraXpertDenoiseOptions,
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


# ── Siril denoise backend ─────────────────────────────────────────────────────

def _run_siril_denoise(
    image_path: Path,
    options: SirilDenoiseOptions,
    working_dir: str,
) -> Path:
    """
    Run Siril's native `denoise` command on the loaded image and save the
    result. Algorithm dispatch matches Siril's flag conventions:

        denoise -mod=<float>             # NL-Bayes (default)
        denoise -mod=<float> -da3d       # DA3D
        denoise -mod=<float> -sos=<N>    # SOS iterative

    Optional flags: -vst (only valid with NL-Bayes), -indep, -nocosmetic.
    """
    output_stem = f"{image_path.stem}_siril_denoise"

    cmd = f"denoise -mod={options.modulation}"
    algo = options.algorithm.lower()
    if algo == "da3d":
        cmd += " -da3d"
    elif algo == "sos":
        cmd += f" -sos={options.sos_iterations}"
    elif algo != "standard":
        raise ValueError(
            f"siril denoise: unknown algorithm '{options.algorithm}'. "
            f"Valid: standard, da3d, sos."
        )
    if options.use_vst and algo == "standard":
        cmd += " -vst"
    if options.independent_channels:
        cmd += " -indep"
    if not options.apply_cosmetic:
        cmd += " -nocosmetic"

    commands = [
        f"load {image_path.stem}",
        cmd,
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=300)

    output_path = image_path.parent / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = image_path.parent / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(
            f"Siril denoise did not produce expected output matching "
            f"{output_stem}.* in {image_path.parent}"
        )
    return output_path


# ── Bilateral denoise backend ─────────────────────────────────────────────────

def _run_bilateral_denoise(
    image_path: Path,
    options: BilateralDenoiseOptions,
) -> Path:
    """
    Edge-preserving bilateral filter via scikit-image. Operates on the
    image data directly (no Siril round-trip). The filter expects pixel
    values in [0, 1]; values outside that range are temporarily rescaled
    and restored after filtering.
    """
    from skimage.restoration import denoise_bilateral

    output_stem = f"{image_path.stem}_bilateral_denoise"
    output_path = image_path.parent / f"{output_stem}.fits"

    with astropy_fits.open(str(image_path)) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Normalize to [0, 1] for the bilateral filter, remembering the scale.
    data_max = float(data.max())
    if data_max > 1.0:
        data_norm = data / data_max
        scale_back = data_max
    else:
        data_norm = data
        scale_back = 1.0

    is_chw_color = (data_norm.ndim == 3 and data_norm.shape[0] == 3)
    is_hwc_color = (data_norm.ndim == 3 and data_norm.shape[2] == 3)

    if is_chw_color:
        if options.multichannel:
            data_hwc = data_norm.transpose(1, 2, 0)
            denoised_hwc = denoise_bilateral(
                data_hwc,
                sigma_color=options.sigma_color,
                sigma_spatial=options.sigma_spatial,
                win_size=options.win_size,
                channel_axis=-1,
            )
            result = denoised_hwc.transpose(2, 0, 1)
        else:
            result = np.zeros_like(data_norm)
            for c in range(3):
                result[c] = denoise_bilateral(
                    data_norm[c],
                    sigma_color=options.sigma_color,
                    sigma_spatial=options.sigma_spatial,
                    win_size=options.win_size,
                    channel_axis=None,
                )
    elif is_hwc_color:
        if options.multichannel:
            result = denoise_bilateral(
                data_norm,
                sigma_color=options.sigma_color,
                sigma_spatial=options.sigma_spatial,
                win_size=options.win_size,
                channel_axis=-1,
            )
        else:
            result = np.zeros_like(data_norm)
            for c in range(3):
                result[..., c] = denoise_bilateral(
                    data_norm[..., c],
                    sigma_color=options.sigma_color,
                    sigma_spatial=options.sigma_spatial,
                    win_size=options.win_size,
                    channel_axis=None,
                )
    else:
        # Mono — multichannel flag has no effect
        mono = data_norm.squeeze()
        result = denoise_bilateral(
            mono,
            sigma_color=options.sigma_color,
            sigma_spatial=options.sigma_spatial,
            win_size=options.win_size,
            channel_axis=None,
        )

    if scale_back != 1.0:
        result = result * scale_back

    hdu = astropy_fits.PrimaryHDU(data=result.astype(np.float32))
    hdu.writeto(str(output_path), overwrite=True)
    return output_path


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=NoiseReductionInput)
def noise_reduction(
    method: str = "graxpert",
    graxpert_options: GraXpertDenoiseOptions | None = None,
    siril_options: SirilDenoiseOptions | None = None,
    bilateral_options: BilateralDenoiseOptions | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Noise reduction on the working FITS image.

    method='graxpert'  — GraXpert AI denoiser. Knobs in graxpert_options:
                         strength, ai_version, batch_size, gpu.
    method='siril'     — Siril `denoise` (NL-Bayes / DA3D / SOS). Knobs in
                         siril_options: algorithm, modulation, sos_iterations,
                         use_vst, independent_channels, apply_cosmetic.
    method='bilateral' — scikit-image edge-preserving bilateral filter.
                         Knobs in bilateral_options: sigma_color,
                         sigma_spatial, win_size, multichannel.

    Returns noise_before, noise_after, noise_reduction_pct (Siril `bgnoise`
    measurement before and after the operation, when measurable). For the
    GraXpert path, also returns zero_pixels_pedestaled — pixels lifted to
    a 1e-4 pedestal to work around an ONNX zero-division bug that affects
    images with exact-zero borders (X-Trans debayering, Siril framing=min).
    Siril and bilateral paths do not need the pedestal.

    Output is promoted to paths.current_image. Method-specific suffix on
    the output filename: '_denoise' (graxpert), '_siril_denoise' (siril),
    '_bilateral_denoise' (bilateral).
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    method_norm = method.lower()
    if method_norm not in ("graxpert", "siril", "bilateral"):
        raise ValueError(
            f"noise_reduction: unknown method '{method}'. "
            f"Valid: graxpert, siril, bilateral."
        )

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    zero_pixels_pedestaled: int | None = None
    settings_used: dict = {}

    if method_norm == "graxpert":
        if graxpert_options is None:
            raise ValueError(
                "noise_reduction: method='graxpert' requires graxpert_options "
                "(strength has no default)."
            )
        output_path, zero_pixels_pedestaled = _run_graxpert_denoise(
            img_path, graxpert_options
        )
        settings_used = {
            "strength": graxpert_options.strength,
            "ai_version": graxpert_options.ai_version,
            "batch_size": graxpert_options.batch_size,
            "gpu": graxpert_options.gpu,
        }
    elif method_norm == "siril":
        if siril_options is None:
            siril_options = SirilDenoiseOptions()
        output_path = _run_siril_denoise(img_path, siril_options, working_dir)
        settings_used = {
            "algorithm": siril_options.algorithm,
            "modulation": siril_options.modulation,
            "sos_iterations": siril_options.sos_iterations,
            "use_vst": siril_options.use_vst,
            "independent_channels": siril_options.independent_channels,
            "apply_cosmetic": siril_options.apply_cosmetic,
        }
    else:  # bilateral
        if bilateral_options is None:
            bilateral_options = BilateralDenoiseOptions()
        output_path = _run_bilateral_denoise(img_path, bilateral_options)
        settings_used = {
            "sigma_color": bilateral_options.sigma_color,
            "sigma_spatial": bilateral_options.sigma_spatial,
            "win_size": bilateral_options.win_size,
            "multichannel": bilateral_options.multichannel,
        }

    # Measure noise before/after if possible
    noise_before, before_err = _measure_bgnoise(img_path.stem, working_dir)
    noise_after, after_err = _measure_bgnoise(output_path.stem, working_dir)
    noise_reduction_pct = None
    if noise_before and noise_after and noise_before > 0:
        noise_reduction_pct = round((1.0 - noise_after / noise_before) * 100, 2)

    noise_warnings = []
    if before_err:
        noise_warnings.append(f"Could not measure noise before denoising: {before_err}")
    if after_err:
        noise_warnings.append(f"Could not measure noise after denoising: {after_err}")

    summary = {
        "output_path": str(output_path),
        "method": method_norm,
        "noise_before": noise_before,
        "noise_after": noise_after,
        "noise_reduction_pct": noise_reduction_pct,
        "noise_measurement_warnings": noise_warnings if noise_warnings else None,
        "zero_pixels_pedestaled": zero_pixels_pedestaled,
        "settings": settings_used,
    }

    # Noise reduction is stretch-agnostic (see module docstring): the same
    # tool can run pre- or post-stretch, and the operation preserves image
    # space (smoothing pixel values does not change the value-space). Pass
    # the input image_space through unchanged. State is authoritative — if
    # it's missing, refuse rather than guess. See Metadata.image_space.
    incoming_image_space = state["metadata"].get("image_space")
    if incoming_image_space not in ("linear", "display"):
        raise RuntimeError(
            "noise_reduction: state.metadata.image_space is missing or invalid "
            f"(got {incoming_image_space!r}). Every writer of paths.current_image "
            "must also write metadata.image_space; this looks like a legacy "
            "checkpoint or a writer that skipped its bookkeeping. Refusing to "
            "guess — restart from a fresh checkpoint."
        )

    return Command(update={
        "paths": {"current_image": str(output_path)},
        "metadata": {"image_space": incoming_image_space},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
