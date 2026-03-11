"""
T13 — deconvolution

Sharpen the image by reversing atmospheric and optical blur. Deconvolution
reconstructs detail lost to atmospheric seeing and diffraction limits.

Backend: Siril CLI — `makepsf` to estimate/load/construct the PSF, then `rl`
(Richardson-Lucy) or `wiener` to apply the correction.

Siril commands (verified against Siril 1.4 CLI docs):
    makepsf clear
    makepsf load filename
    makepsf save [filename]
    makepsf blind [-l0] [-si] [-multiscale] [-lambda=] [-comp=] [-ks=] [-savepsf=]
    makepsf stars [-sym] [-ks=] [-savepsf=]
    makepsf manual { -gaussian | -moffat | -disc | -airy }
        [-fwhm=] [-angle=] [-ratio=] [-beta=]
        [-dia=] [-fl=] [-wl=] [-pixelsize=] [-obstruct=]
        [-ks=] [-savepsf=]
    rl [-loadpsf=] [-alpha=] [-iters=] [-stop=] [-gdstep=] [-tv] [-fh] [-mul]
    wiener [-loadpsf=] [-alpha=]

CRITICAL CONSTRAINTS:
  - Only valid in linear space (before stretch).
  - Only attempt when SNR is adequate: snr_estimate > 50 from analyze_image.
  - Start conservative: rl iterations=5–10.
  - Total Variation regularization (-tv) is the default and safest choice.
  - Always run analyze_image after to verify improvement.

"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import fits_has_nan, run_siril_script


# ── Pydantic input schemas ────────────────────────────────────────────────────

class RLOptions(BaseModel):
    iterations: int = Field(
        default=5,
        description=(
            "Number of Richardson-Lucy iterations. "
            "Default 5 is the conservative first pass for any SNR level. "
            "Increase to 10–20 only after running analyze_image and confirming "
            "SNR > 50 and noise_after is acceptable. "
            "Never exceed 30 — diminishing returns and ringing risk amplify noise."
        ),
    )
    regularization: str = Field(
        default="total_variation",
        description=(
            "'total_variation': TV regularization (-tv) — suppresses ringing, "
            "slight edge smoothing. Safest default. "
            "'hessian_frobenius': Frobenius norm of Hessian (-fh) — preserves edges "
            "better, more ringing risk. Good for sharp features. "
            "'none': No regularization — maximum sharpening, high ringing risk."
        ),
    )
    alpha: float = Field(
        default=3000.0,
        description=(
            "Regularization strength. Lower value = MORE regularization = softer "
            "but cleaner. Higher value = less regularization = sharper but ringing. "
            "Range: 500 (aggressive regularization) – 10000 (minimal regularization)."
        ),
    )
    stop: float | None = Field(
        default=None,
        description=(
            "Stopping criterion (-stop=). Terminates when residual change drops "
            "below this value. Typical: 1e-4 to 1e-6. Null = run all iterations."
        ),
    )
    gdstep: float | None = Field(
        default=None,
        description=(
            "Gradient descent step size (-gdstep=). Default: 0.0005. "
            "Larger steps converge faster but risk oscillation. "
            "Smaller steps are more stable but slower."
        ),
    )
    use_multiplicative: bool = Field(
        default=False,
        description=(
            "Use multiplicative RL update instead of gradient descent (-mul). "
            "Can improve convergence on Poisson-distributed data."
        ),
    )


class WienerOptions(BaseModel):
    alpha: float = Field(
        default=0.001,
        description=(
            "Wiener regularization parameter. Controls noise-sharpness tradeoff. "
            "Lower = sharper but more noise. "
            "0.001 is a reasonable start. Range: 0.0001 – 0.01."
        ),
    )


class BlindPsfOptions(BaseModel):
    """Options for makepsf blind — blind PSF estimation from the image."""
    use_l0: bool = Field(
        default=False,
        description="Use L0 descent method (-l0). Alternative to default method.",
    )
    use_spectral_irregularity: bool = Field(
        default=False,
        description="Use spectral irregularity method (-si).",
    )
    multiscale: bool = Field(
        default=False,
        description=(
            "Multi-scale PSF estimation (-multiscale). Only with L0 method. "
            "Better for images with varying PSF across the field."
        ),
    )
    regularization_lambda: float | None = Field(
        default=None,
        description="Regularization constant (-lambda=). Controls smoothness of blind estimate.",
    )
    comp: float | None = Field(
        default=None,
        description="Compression parameter (-comp=).",
    )


class StarsPsfOptions(BaseModel):
    """Options for makepsf stars — PSF from detected stars."""
    symmetric: bool = Field(
        default=False,
        description=(
            "Force circularly symmetric PSF (-sym). Use when tracking is round "
            "but measurement noise causes asymmetry."
        ),
    )


class ManualPsfOptions(BaseModel):
    """Options for makepsf manual — analytically defined PSF."""
    profile: str = Field(
        default="moffat",
        description=(
            "'moffat': Most accurate model for atmospheric seeing. "
            "'gaussian': Simpler, slightly less accurate. "
            "'airy': Theoretical diffraction pattern — for diffraction-limited optics. "
            "'disc': Top-hat disc — for defocused images."
        ),
    )
    fwhm_px: float | None = Field(
        default=None,
        description=(
            "PSF full width at half maximum in pixels. "
            "Typical: 1.5–4.0 px for well-focused images. "
            "For disc profile, sets the disc diameter."
        ),
    )
    moffat_beta: float = Field(
        default=3.5,
        description=(
            "Moffat beta parameter (wing extent). "
            "Lower = broader wings (worse seeing). Higher = narrower (Gaussian-like). "
            "Range: 2.0 (poor seeing) – 5.0 (excellent). Only for profile=moffat."
        ),
    )
    aspect_ratio: float = Field(
        default=1.0,
        description=(
            "PSF aspect ratio (minor/major axis). 1.0 = circular. "
            "< 1.0 = elongated stars from tracking error."
        ),
    )
    angle_deg: float = Field(
        default=0.0,
        description="Angle of PSF major axis in degrees. Only relevant when aspect_ratio < 1.0.",
    )
    airy_diameter_mm: float | None = Field(
        default=None,
        description="Telescope primary aperture diameter in mm. For profile=airy.",
    )
    airy_focal_length_mm: float | None = Field(
        default=None,
        description="Telescope focal length in mm. For profile=airy.",
    )
    airy_wavelength_nm: float = Field(
        default=525.0,
        description="Central wavelength in nm. 525=green, 656=Ha, 500=OIII, 486=Hb.",
    )
    airy_pixelsize_um: float | None = Field(
        default=None,
        description="Sensor pixel size in microns for Airy pattern. For profile=airy.",
    )
    airy_obstruction_pct: float = Field(
        default=0.0,
        description=(
            "Central obstruction as % of aperture area. "
            "0 for refractors. 25–35 for SCT. 20–25 for Newtonians."
        ),
    )


class PsfConfig(BaseModel):
    """Unified PSF configuration for all source types."""
    psf_kernel_size: int | None = Field(
        default=None,
        description="PSF dimension in pixels (-ks=). Must be odd. Null = Siril default.",
    )
    save_psf: str | None = Field(
        default=None,
        description=(
            "Save the generated PSF to this filename (-savepsf=). "
            "Extension must be .fit, .fits, .fts, or .tif. "
            "Useful for reusing a PSF across multiple deconvolution attempts."
        ),
    )


class DeconvolutionInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description="Absolute path to the linear FITS image to sharpen."
    )
    method: str = Field(
        default="richardson_lucy",
        description=(
            "'richardson_lucy': Iterative with optional regularization. Recommended. "
            "'wiener': Linear Wiener filter — faster, less control, for clean high-SNR."
        ),
    )
    psf_source: str = Field(
        default="stars",
        description=(
            "'stars': PSF from detected stars (makepsf stars) — most accurate. "
            "'blind': Blind PSF estimation (makepsf blind) — for low star count. "
            "'manual': Analytical PSF via manual_psf_options. "
            "'from_file': Load pre-computed PSF FITS from psf_file path. "
            "'loadpsf': Load PSF inline with the rl/wiener command via -loadpsf=."
        ),
    )
    psf_file: str | None = Field(
        default=None,
        description=(
            "Path to PSF FITS file. Used when psf_source='from_file' (with makepsf load) "
            "or psf_source='loadpsf' (with rl -loadpsf= / wiener -loadpsf=)."
        ),
    )
    blind_psf_options: BlindPsfOptions = Field(default_factory=BlindPsfOptions)
    stars_psf_options: StarsPsfOptions = Field(default_factory=StarsPsfOptions)
    manual_psf_options: ManualPsfOptions = Field(default_factory=ManualPsfOptions)
    psf_config: PsfConfig = Field(default_factory=PsfConfig)
    rl_options: RLOptions = Field(default_factory=RLOptions)
    wiener_options: WienerOptions = Field(default_factory=WienerOptions)


# ── PSF command builders ──────────────────────────────────────────────────────

def _build_makepsf_stars(opts: StarsPsfOptions, cfg: PsfConfig) -> str:
    cmd = "makepsf stars"
    if opts.symmetric:
        cmd += " -sym"
    if cfg.psf_kernel_size is not None:
        cmd += f" -ks={cfg.psf_kernel_size}"
    if cfg.save_psf:
        cmd += f" -savepsf={cfg.save_psf}"
    return cmd


def _build_makepsf_blind(opts: BlindPsfOptions, cfg: PsfConfig) -> str:
    cmd = "makepsf blind"
    if opts.use_l0:
        cmd += " -l0"
    if opts.use_spectral_irregularity:
        cmd += " -si"
    if opts.multiscale:
        cmd += " -multiscale"
    if opts.regularization_lambda is not None:
        cmd += f" -lambda={opts.regularization_lambda}"
    if opts.comp is not None:
        cmd += f" -comp={opts.comp}"
    if cfg.psf_kernel_size is not None:
        cmd += f" -ks={cfg.psf_kernel_size}"
    if cfg.save_psf:
        cmd += f" -savepsf={cfg.save_psf}"
    return cmd


def _build_makepsf_manual(opts: ManualPsfOptions, cfg: PsfConfig) -> str:
    cmd = f"makepsf manual -{opts.profile}"
    if opts.fwhm_px is not None:
        cmd += f" -fwhm={opts.fwhm_px}"
    if opts.profile == "moffat" and opts.moffat_beta != 3.5:
        cmd += f" -beta={opts.moffat_beta}"
    if opts.aspect_ratio != 1.0:
        cmd += f" -ratio={opts.aspect_ratio} -angle={opts.angle_deg}"
    if opts.profile == "airy":
        if opts.airy_diameter_mm is not None:
            cmd += f" -dia={opts.airy_diameter_mm}"
        if opts.airy_focal_length_mm is not None:
            cmd += f" -fl={opts.airy_focal_length_mm}"
        cmd += f" -wl={opts.airy_wavelength_nm}"
        if opts.airy_pixelsize_um is not None:
            cmd += f" -pixelsize={opts.airy_pixelsize_um}"
        if opts.airy_obstruction_pct > 0:
            cmd += f" -obstruct={opts.airy_obstruction_pct}"
    if cfg.psf_kernel_size is not None:
        cmd += f" -ks={cfg.psf_kernel_size}"
    if cfg.save_psf:
        cmd += f" -savepsf={cfg.save_psf}"
    return cmd


def _build_rl_cmd(opts: RLOptions, loadpsf: str | None = None) -> str:
    cmd = f"rl -iters={opts.iterations}"
    if loadpsf:
        cmd += f" -loadpsf={loadpsf}"
    if opts.regularization == "total_variation":
        cmd += f" -tv -alpha={opts.alpha}"
    elif opts.regularization == "hessian_frobenius":
        cmd += f" -fh -alpha={opts.alpha}"
    if opts.stop is not None:
        cmd += f" -stop={opts.stop}"
    if opts.gdstep is not None:
        cmd += f" -gdstep={opts.gdstep}"
    if opts.use_multiplicative:
        cmd += " -mul"
    return cmd


def _build_wiener_cmd(opts: WienerOptions, loadpsf: str | None = None) -> str:
    cmd = f"wiener -alpha={opts.alpha}"
    if loadpsf:
        cmd += f" -loadpsf={loadpsf}"
    return cmd


# ── Parse helpers ─────────────────────────────────────────────────────────────

def _parse_psf_fwhm(stdout: str) -> float | None:
    m = re.search(r"PSF\s+FWHM[^=]*=\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"FWHM[^=\d]*([\d.]+)\s*(?:px|pixel)", stdout, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=DeconvolutionInput)
def deconvolution(
    working_dir: str,
    image_path: str,
    method: str = "richardson_lucy",
    psf_source: str = "stars",
    psf_file: str | None = None,
    blind_psf_options: BlindPsfOptions | None = None,
    stars_psf_options: StarsPsfOptions | None = None,
    manual_psf_options: ManualPsfOptions | None = None,
    psf_config: PsfConfig | None = None,
    rl_options: RLOptions | None = None,
    wiener_options: WienerOptions | None = None,
) -> dict:
    """
    Sharpen the linear image by deconvolving atmospheric and optical blur.

    Best results require the image to be in linear space with adequate SNR
    (snr_estimate > 50). Low-SNR images produce ringing rather than sharpening.

    PSF source guidance:
      stars  — measures PSF from the image's own stars. Almost always the best
               choice for stacked astrophotography data.
      blind  — blind estimation when star count is very low. Try l0 method
               (-use_l0) or spectral irregularity (-use_spectral_irregularity).
               Use multiscale for spatially varying PSF.
      manual — analytical PSF when you have precise optics data. Moffat is the
               most physically accurate model for atmospheric seeing.
      from_file — load a previously saved PSF via makepsf load.
      loadpsf — load PSF inline with the rl/wiener command (-loadpsf= flag).
               Slightly different from from_file: skips the separate makepsf step.

    Algorithm guidance:
      richardson_lucy with total_variation regularization is the default. Start
      with iterations=5-10. Increase alpha (less regularization) for more
      sharpening on high-SNR images. Use gdstep to tune gradient descent rate.

      wiener is faster for very clean high-SNR images with a well-known PSF.

    PSF can be saved (-savepsf via psf_config.save_psf) for reuse across
    multiple deconvolution attempts with different rl/wiener parameters.
"""
    if blind_psf_options is None:
        blind_psf_options = BlindPsfOptions()
    if stars_psf_options is None:
        stars_psf_options = StarsPsfOptions()
    if manual_psf_options is None:
        manual_psf_options = ManualPsfOptions()
    if psf_config is None:
        psf_config = PsfConfig()
    if rl_options is None:
        rl_options = RLOptions()
    if wiener_options is None:
        wiener_options = WienerOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = img_path.stem
    output_stem = f"{stem}_deconv"

    commands: list[str] = [f"load {stem}"]

    loadpsf_path: str | None = None

    if psf_source == "loadpsf" and psf_file:
        loadpsf_path = psf_file
    elif psf_source == "from_file" and psf_file:
        psf_p = Path(psf_file)
        if not psf_p.exists():
            raise FileNotFoundError(f"PSF file not found: {psf_file}")
        commands.append(f"makepsf load {psf_p.name}")
    elif psf_source == "blind":
        commands.append(_build_makepsf_blind(blind_psf_options, psf_config))
    elif psf_source == "manual":
        commands.append(_build_makepsf_manual(manual_psf_options, psf_config))
    else:
        commands.append(_build_makepsf_stars(stars_psf_options, psf_config))

    if method == "wiener":
        commands.append(_build_wiener_cmd(wiener_options, loadpsf_path))
    else:
        commands.append(_build_rl_cmd(rl_options, loadpsf_path))

    commands.append(f"save {output_stem}")

    result = run_siril_script(commands, working_dir=working_dir, timeout=300)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"Deconvolution did not produce: {output_path}")

    if fits_has_nan(output_path):
        raise RuntimeError(
            f"Deconvolution produced NaN/Inf values in {output_path.name}. "
            "The Richardson-Lucy algorithm diverged — the image is corrupted and "
            "must not be used downstream (stretch will convert NaN → 1.0, "
            "producing an all-white image). "
            "To fix:\n"
            "  1. Reduce iterations (e.g., iterations=3-5 instead of current value).\n"
            "  2. Lower alpha to increase regularization strength (e.g., alpha=500-1000).\n"
            "  3. Switch method='wiener' — more numerically stable on noisy images.\n"
            "  4. Try psf_source='blind' if the manual PSF does not match the actual blur.\n"
            "  5. Skip deconvolution entirely if SNR is low (snr_estimate < 50)."
        )

    psf_fwhm_used = _parse_psf_fwhm(result.stdout)

    return {
        "processed_image_path": str(output_path),
        "sharpened_image_path": str(output_path),
        "method": method,
        "psf_source": psf_source,
        "psf_profile": manual_psf_options.profile if psf_source == "manual" else None,
        "psf_fwhm_used": psf_fwhm_used,
        "psf_saved_to": psf_config.save_psf,
        "iterations_used": rl_options.iterations if method == "richardson_lucy" else None,
        "regularization": rl_options.regularization if method == "richardson_lucy" else None,
        "stop_criterion": rl_options.stop if method == "richardson_lucy" else None,
    }
