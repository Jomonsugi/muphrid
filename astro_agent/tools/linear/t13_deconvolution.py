"""
T13 — deconvolution

Sharpen the image by reversing atmospheric and optical blur. Deconvolution
reconstructs detail lost to atmospheric seeing and diffraction limits.

Backend: Siril CLI — `makepsf stars` to estimate the PSF from detected
stars, then `rl` (Richardson-Lucy) or `wiener` to apply the correction.

CRITICAL CONSTRAINTS:
  - Only valid in linear space (before stretch).
  - Only attempt when SNR is adequate: snr_estimate > 50 from analyze_image.
    Under-exposed or high-noise images will have ringing amplified, not reduced.
  - Start conservative: rl_iterations=5–10. Increase only if the result
    shows clear improvement with no ringing artifacts.
  - Total Variation regularization (-tv) is the default and safest choice —
    it suppresses ringing at the cost of slight smoothing of sharp edges.
  - Always run analyze_image after to verify the image improved.

HITL: requires_visual_review=True by default.
Ringing (concentric halos around point sources) and over-sharpening are
known artefact risks. Visual inspection is mandatory in V1 to catch these
before they propagate through the rest of the pipeline.

Future note: GraXpert 3.1.x will add dedicated stellar and object
deconvolution AI models. Wire in when that release is stable.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class RLOptions(BaseModel):
    iterations: int = Field(
        default=10,
        description=(
            "Number of Richardson-Lucy iterations. "
            "Start conservative: 5–10 iterations for first pass. "
            "10–20 for modest sharpening on adequate-SNR images. "
            "Never exceed 30 — diminishing returns and ringing risk increase sharply. "
            "Higher SNR images tolerate more iterations."
        ),
    )
    regularization: str = Field(
        default="total_variation",
        description=(
            "total_variation: TV regularization (-tv) — suppresses ringing, "
            "slight edge smoothing. Safest choice for most images. "
            "hessian_frobenius: Frobenius norm of the Hessian matrix (-fh) — "
            "preserves edges better than TV at the cost of more ringing risk. "
            "Good for images with sharp linear features (galaxy arms, nebula filaments). "
            "none: No regularization — maximum sharpening, high ringing risk. "
            "Use only for very high-SNR images where TV is over-smoothing."
        ),
    )
    alpha: float = Field(
        default=3000.0,
        description=(
            "Regularization strength. Lower = less regularization = sharper but "
            "more ringing risk. Higher = softer but cleaner. "
            "Note: lower alpha = more regularization (Siril convention). "
            "3000 is a safe default. Range: 500 (aggressive) – 10000 (conservative)."
        ),
    )
    stop: float | None = Field(
        default=None,
        description=(
            "Stopping criterion threshold (-stop=). When the residual change between "
            "iterations drops below this value, RL terminates early. "
            "Prevents over-iterating past the point of improvement. "
            "Typical range: 1e-4 to 1e-6. Null disables early stopping (runs all "
            "iterations). Recommended for automation to prevent over-deconvolution."
        ),
    )
    use_multiplicative: bool = Field(
        default=False,
        description=(
            "Use multiplicative RL update instead of gradient descent (-mul). "
            "Can improve convergence on Poisson-distributed data (raw photon counts). "
            "Usually not needed for stacked, calibrated images."
        ),
    )


class WienerOptions(BaseModel):
    alpha: float = Field(
        default=0.001,
        description=(
            "Wiener regularization parameter. Controls noise-sharpness tradeoff. "
            "Lower values = sharper but more noise amplification. "
            "0.001 is a reasonable starting point. "
            "Range: 0.0001 (aggressive) – 0.01 (conservative)."
        ),
    )


class MakePsfManualOptions(BaseModel):
    profile: str = Field(
        default="moffat",
        description=(
            "PSF function model for manual PSF construction. "
            "moffat: Moffat profile — most accurate model for atmospheric seeing. "
            "Stars in real astrophotos have Moffat-shaped PSFs, not pure Gaussians. "
            "gaussian: Gaussian profile — simpler, slightly less accurate. "
            "airy: Theoretical Airy diffraction pattern — requires telescope optics "
            "parameters (diameter, focal_length). Best for diffraction-limited optics "
            "in space or when seeing is excellent. "
            "disc: Top-hat disc — for defocused or out-of-focus images."
        ),
    )
    fwhm_px: float | None = Field(
        default=None,
        description=(
            "PSF full width at half maximum in pixels. "
            "For moffat/gaussian: typical range 1.5–4.0 px for well-focused images. "
            "Measure from analyze_image star metrics (median_fwhm_px). "
            "If null, Siril uses its own estimate from the image."
        ),
    )
    moffat_beta: float = Field(
        default=3.5,
        description=(
            "Moffat beta parameter controlling the PSF wing extent. "
            "Lower beta = broader wings (worse seeing, more scattered light). "
            "Higher beta = narrower wings (approaches Gaussian). "
            "Typical range for atmospheric seeing: 2.0 (poor) – 5.0 (excellent). "
            "3.5 is a reliable default. Only used when profile=moffat."
        ),
    )
    aspect_ratio: float = Field(
        default=1.0,
        description=(
            "PSF aspect ratio (minor/major axis). 1.0 = circular. "
            "Values < 1.0 model elongated stars from tracking errors or "
            "atmospheric dispersion. Only used for moffat/gaussian."
        ),
    )
    angle_deg: float = Field(
        default=0.0,
        description=(
            "Angle of the PSF major axis in degrees. "
            "Only relevant when aspect_ratio < 1.0."
        ),
    )
    airy_diameter_mm: float | None = Field(
        default=None,
        description=(
            "Telescope primary aperture diameter in mm. Used when profile=airy. "
            "Example: 130mm refractor → 130."
        ),
    )
    airy_focal_length_mm: float | None = Field(
        default=None,
        description=(
            "Telescope focal length in mm. Used when profile=airy. "
            "Example: 900mm f/6.9 refractor → 900."
        ),
    )
    airy_wavelength_nm: float = Field(
        default=525.0,
        description=(
            "Central wavelength in nm for Airy pattern calculation. "
            "525nm = green (broadband default). "
            "656nm = Hα, 500nm = OIII, 486nm = Hβ."
        ),
    )
    airy_obstruction_pct: float = Field(
        default=0.0,
        description=(
            "Central obstruction as a percentage of aperture area (0–100). "
            "0 for refractors and apo lenses. "
            "Typical SCT/Cassegrain: 25–35%. Typical Newt: 20–25%."
        ),
    )
    symmetric: bool = Field(
        default=False,
        description=(
            "Force the stars-based PSF to be circularly symmetric (-sym). "
            "Only used when psf_source=stars. Useful when tracking is known to "
            "be round but measurement noise is asymmetric."
        ),
    )


class DeconvolutionInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the linear FITS image to sharpen. "
            "Image must be noise-reduced (post T12) and have adequate SNR (>50)."
        )
    )
    method: str = Field(
        default="richardson_lucy",
        description=(
            "richardson_lucy: RL with optional regularization. "
            "Recommended — iterative control, handles most cases well with tv. "
            "wiener: Linear Wiener filter — faster, less iterative control, "
            "good for very clean high-SNR images."
        ),
    )
    psf_source: str = Field(
        default="stars",
        description=(
            "stars: Siril measures PSF from detected stars (makepsf stars) — most "
            "accurate for well-focused star fields with sufficient star count. "
            "blind: Blind PSF estimation (makepsf blind) — use when star count "
            "is very low or stars are poorly resolved. "
            "manual: Analytically defined PSF via makepsf_manual_options. Use when "
            "you know the seeing FWHM or telescope optics precisely. "
            "from_file: Load a pre-computed PSF FITS from psf_file path."
        ),
    )
    psf_file: str | None = Field(
        default=None,
        description=(
            "Absolute path to a pre-computed PSF FITS file. "
            "Only used when psf_source=from_file."
        ),
    )
    makepsf_manual_options: MakePsfManualOptions = Field(
        default_factory=MakePsfManualOptions,
    )
    rl_options: RLOptions = Field(default_factory=RLOptions)
    wiener_options: WienerOptions = Field(default_factory=WienerOptions)


# ── PSF estimation ─────────────────────────────────────────────────────────────

def _parse_psf_fwhm(stdout: str) -> float | None:
    """Extract estimated PSF FWHM from Siril makepsf stdout."""
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
    makepsf_manual_options: MakePsfManualOptions | None = None,
    rl_options: RLOptions | None = None,
    wiener_options: WienerOptions | None = None,
) -> dict:
    """
    Sharpen the linear image by deconvolving atmospheric and optical blur.

    Prerequisites (verify with analyze_image before calling):
    - Image must be in linear space (is_linear=True).
    - snr_estimate > 50. Low-SNR images will produce ringing, not sharpening.
    - Image should be noise-reduced (T12 completed).

    Algorithm guidance:
    - richardson_lucy with regularization=total_variation is the default and
      recommended method for most images. Start with iterations=5–10.
    - wiener is appropriate for very clean, high-SNR images.

    PSF: psf_source=stars uses makepsf stars to measure the PSF from the
    image's own stars — almost always superior to a manually specified PSF
    for stacked data. Use psf_source=manual with MakePsfManualOptions when
    the star count is insufficient or you have precise telescope optics data.

    HITL visual review is triggered automatically after this tool (V1).
    If ringing or halos are visible, reduce iterations or increase alpha.
    """
    if rl_options is None:
        rl_options = RLOptions()
    if wiener_options is None:
        wiener_options = WienerOptions()
    if makepsf_manual_options is None:
        makepsf_manual_options = MakePsfManualOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = img_path.stem
    output_stem = f"{stem}_deconv"

    # Build commands
    commands: list[str] = [f"load {stem}"]

    # PSF setup — verified Siril 1.4 makepsf syntax
    if psf_source == "from_file" and psf_file:
        psf_path = Path(psf_file)
        if not psf_path.exists():
            raise FileNotFoundError(f"PSF file not found: {psf_file}")
        commands.append(f"makepsf load {psf_path.name}")
    elif psf_source == "blind":
        commands.append("makepsf blind")
    elif psf_source == "manual":
        mo = makepsf_manual_options
        psf_cmd = f"makepsf manual -{mo.profile}"
        if mo.fwhm_px is not None:
            psf_cmd += f" -fwhm={mo.fwhm_px}"
        if mo.profile == "moffat":
            psf_cmd += f" -beta={mo.moffat_beta}"
        if mo.aspect_ratio != 1.0:
            psf_cmd += f" -ratio={mo.aspect_ratio} -angle={mo.angle_deg}"
        if mo.profile == "airy":
            if mo.airy_diameter_mm is not None:
                psf_cmd += f" -dia={mo.airy_diameter_mm}"
            if mo.airy_focal_length_mm is not None:
                psf_cmd += f" -fl={mo.airy_focal_length_mm}"
            psf_cmd += f" -wl={mo.airy_wavelength_nm}"
            if mo.airy_obstruction_pct > 0:
                psf_cmd += f" -obstruct={mo.airy_obstruction_pct}"
        commands.append(psf_cmd)
    else:
        # stars: measures PSF from detected stars — most accurate for astrophotography
        stars_cmd = "makepsf stars"
        if makepsf_manual_options.symmetric:
            stars_cmd += " -sym"
        commands.append(stars_cmd)

    # Deconvolution command — verified Siril 1.4 rl syntax
    if method == "wiener":
        commands.append(f"wiener -alpha={wiener_options.alpha}")
    else:
        rl_cmd = f"rl -iters={rl_options.iterations}"
        if rl_options.regularization == "total_variation":
            rl_cmd += f" -tv -alpha={rl_options.alpha}"
        elif rl_options.regularization == "hessian_frobenius":
            rl_cmd += f" -fh -alpha={rl_options.alpha}"
        elif rl_options.regularization == "frobenius":
            # legacy alias
            rl_cmd += f" -alpha={rl_options.alpha}"
        if rl_options.stop is not None:
            rl_cmd += f" -stop={rl_options.stop}"
        if rl_options.use_multiplicative:
            rl_cmd += " -mul"
        commands.append(rl_cmd)

    commands.append(f"save {output_stem}")

    result = run_siril_script(commands, working_dir=working_dir, timeout=300)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"Deconvolution did not produce: {output_path}")

    psf_fwhm_used = _parse_psf_fwhm(result.stdout)

    return {
        "processed_image_path": str(output_path),
        "sharpened_image_path": str(output_path),  # backward-compat alias
        "method": method,
        "psf_source": psf_source,
        "psf_profile": makepsf_manual_options.profile if psf_source == "manual" else None,
        "psf_fwhm_used": psf_fwhm_used,
        "iterations_used": rl_options.iterations if method == "richardson_lucy" else None,
        "regularization": rl_options.regularization if method == "richardson_lucy" else None,
        "stop_criterion": rl_options.stop if method == "richardson_lucy" else None,
    }
