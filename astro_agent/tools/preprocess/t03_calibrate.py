"""
T03 — calibrate

Apply master calibration frames (bias, dark, flat) to each raw light sub,
then debayer (demosaic) the result. All operations performed by Siril 1.4
on raw sensor data before any color interpretation occurs.

Why calibration before demosaic:
  Dark current, bias offset, and flat-field vignetting are properties of
  the raw sensor pixels. Correcting them in raw (CFA) space preserves the
  true noise statistics that stacking relies on. Calibrating after demosaic
  contaminates corrections with interpolation artifacts and degrades SNR.
  This is the mathematically correct order for astrophotography.

Siril 1.4 and X-Trans sensors:
  Siril 1.4 automatically detects Fujifilm X-Trans sensors and applies
  the Markesteijn demosaicing algorithm. For best detail preservation,
  set X-Trans demosaicing quality to 3 passes in Siril Preferences →
  Image Processing → Demosaicing → X-Trans quality. This is a global
  preference and cannot be set per-command.

Agent reasoning guide — set flags based on camera_model from T01:

  Fujifilm X-Trans (X-T30, X-T4, X-S10, X-H2, etc.):
    is_cfa=True, debayer=True, fix_xtrans=True, equalize_cfa=True
    fix_xtrans corrects the rectangle artifact from phase-detection AF pixels.
    equalize_cfa prevents color tinting caused by flat field non-uniformity
    across the X-Trans CFA pattern.

  Bayer OSC/DSLR/mirrorless (Canon, Nikon, Sony, ZWO OSC):
    is_cfa=True, debayer=True, fix_xtrans=False, equalize_cfa=False

  Mono (ZWO ASI533MM, Atik, QHY mono, etc.):
    is_cfa=False, debayer=False

Siril commands (Siril 1.4):
    calibrate sequencename
        [-bias=filename | -bias="=<expression>"]
        [-dark=filename]
        [-flat=filename]
        [-cc=dark siglo sighi | -cc=bpm bpmfile]
        [-cfa] [-debayer] [-fix_xtrans] [-equalize_cfa]
        [-opt | -opt=exp]
        [-all] [-prefix=] [-fitseq]
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Shared schema ──────────────────────────────────────────────────────────────

class CosmeticCorrectionOptions(BaseModel):
    """Options for pixel cosmetic correction during calibration."""
    method: str = Field(
        default="dark",
        description=(
            "'dark': Detect hot/cold pixels from the master dark. Requires master_dark. "
            "'bpm': Use a Bad Pixel Map file. Set bpm_file path. "
            "'none': Skip cosmetic correction."
        ),
    )
    cold_sigma: float = Field(
        default=3.0,
        description=(
            "Sigma threshold for cold pixel detection (method='dark'). "
            "0 = disable cold pixel correction."
        ),
    )
    hot_sigma: float = Field(
        default=5.0,
        description="Sigma threshold for hot pixel detection (method='dark').",
    )
    bpm_file: str | None = Field(
        default=None,
        description=(
            "Absolute path to Bad Pixel Map file (method='bpm'). "
            "Generate with find_hot on a master dark."
        ),
    )


# ── Pydantic input schema ──────────────────────────────────────────────────────

class CalibrateInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    lights_sequence: str = Field(
        description=(
            "Name of the lights sequence to calibrate (without .seq extension). "
            "Must already exist in working_dir — produced by T02b convert_sequence. "
            "T04 register expects the output to be 'pp_lights_seq' (default prefix)."
        )
    )
    master_bias: str | None = Field(
        default=None,
        description=(
            "Absolute path to master bias FITS. "
            "OR a uniform level expression (e.g. '=256' or '=64*$OFFSET'). "
            "Null to skip bias subtraction."
        ),
    )
    master_dark: str | None = Field(
        default=None,
        description="Absolute path to master dark FITS. Null to skip dark subtraction.",
    )
    master_flat: str | None = Field(
        default=None,
        description="Absolute path to master flat FITS. Null to skip flat correction.",
    )
    is_cfa: bool = Field(
        description=(
            "True for OSC/DSLR/mirrorless cameras (Bayer or X-Trans CFA). "
            "Always True for camera RAW input. False for monochrome sensors."
        )
    )
    debayer: bool = Field(
        default=True,
        description=(
            "Demosaic (debayer) CFA data after calibration. "
            "True for all color cameras. False only for mono sensors."
        ),
    )
    equalize_cfa: bool = Field(
        default=False,
        description=(
            "Equalize CFA channel mean intensities from the master flat to prevent "
            "color tinting (-equalize_cfa). "
            "Set True for Fujifilm X-Trans cameras — critical for correct color after "
            "flat calibration of the irregular X-Trans CFA pattern. "
            "Not needed for standard Bayer sensors."
        ),
    )
    fix_xtrans: bool = Field(
        default=False,
        description=(
            "Correct the rectangle artifact in darks and biases caused by Fujifilm's "
            "phase-detection autofocus pixels (-fix_xtrans). "
            "Set True for all Fujifilm X-Trans cameras (X-T30, X-T4, X-S10, X-H2, etc.). "
            "Has no effect on Bayer or mono sensors."
        ),
    )
    cosmetic_correction: CosmeticCorrectionOptions = Field(
        default_factory=CosmeticCorrectionOptions,
    )
    optimize_dark: str | None = Field(
        default=None,
        description=(
            "Dark frame scaling optimization. "
            "'auto': Siril computes dark scaling coefficient automatically (-opt). "
            "Requires both bias and dark masters. Use when dark exposure differs "
            "from light exposure. "
            "'exp': Compute coefficient from the EXPTIME FITS keyword (-opt=exp). "
            "Null: No scaling — use when dark and light exposures match exactly."
        ),
    )
    process_all: bool = Field(
        default=False,
        description=(
            "Process all frames including those marked as excluded (-all). "
            "Normally excluded frames are skipped."
        ),
    )
    prefix: str | None = Field(
        default=None,
        description=(
            "Output filename prefix for calibrated frames. "
            "Default 'pp_' produces 'pp_lights_seq.seq'. "
            "Change only if downstream tools expect a different name."
        ),
    )
    output_fitseq: bool = Field(
        default=False,
        description=(
            "Write output as a single multi-image FITS sequence file (-fitseq) "
            "instead of individual per-frame FITS files. "
            "Individual files (default) are more compatible with external tools."
        ),
    )


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_calibrate_output(stdout: str) -> tuple[int, int]:
    """Return (calibrated_count, bad_pixel_count) from Siril stdout."""
    cal_match = re.search(r"(\d+)\s+(?:image[s]?\s+)?calibrated", stdout, re.IGNORECASE)
    calibrated = int(cal_match.group(1)) if cal_match else 0

    pixel_match = re.search(
        r"(\d+)\s+(?:bad|hot|cold)\s+pixel", stdout, re.IGNORECASE
    )
    bad_pixels = int(pixel_match.group(1)) if pixel_match else 0

    return calibrated, bad_pixels


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=CalibrateInput)
def calibrate(
    working_dir: str,
    lights_sequence: str,
    is_cfa: bool,
    master_bias: str | None = None,
    master_dark: str | None = None,
    master_flat: str | None = None,
    debayer: bool = True,
    equalize_cfa: bool = False,
    fix_xtrans: bool = False,
    cosmetic_correction: CosmeticCorrectionOptions | None = None,
    optimize_dark: str | None = None,
    process_all: bool = False,
    prefix: str | None = None,
    output_fitseq: bool = False,
) -> dict:
    """
    Apply master calibration frames to raw light subs in CFA (raw) space,
    then debayer. This is the correct mathematical order: dark current,
    bias, and flat vignetting are sensor-space properties that must be
    corrected before color interpolation.

    Camera-specific settings (derive from T01 camera_model):

      Fujifilm X-Trans (X-T30, X-T4, X-S10, X-H2, etc.):
        is_cfa=True, debayer=True, fix_xtrans=True, equalize_cfa=True
        Siril 1.4 auto-applies Markesteijn for X-Trans. Set quality=3 in
        Siril Preferences once for best detail preservation.

      Bayer OSC/DSLR/mirrorless (Canon, Nikon, Sony, ZWO OSC):
        is_cfa=True, debayer=True, fix_xtrans=False, equalize_cfa=False

      Monochrome (ZWO mono, Atik, QHY mono):
        is_cfa=False, debayer=False

    Bias can be a file path or a uniform level expression (e.g. '=256').
    Cosmetic correction requires master_dark for method='dark'.
    Dark optimization requires both bias and dark for mode='auto'.
    """
    if cosmetic_correction is None:
        cosmetic_correction = CosmeticCorrectionOptions()

    parts: list[str] = ["calibrate", lights_sequence]

    if master_bias:
        if master_bias.startswith("="):
            parts.append(f'-bias="{master_bias}"')
        else:
            parts.append(f"-bias={master_bias}")

    if master_dark:
        parts.append(f"-dark={master_dark}")

    if master_flat:
        parts.append(f"-flat={master_flat}")

    cc = cosmetic_correction
    if cc.method == "dark" and master_dark:
        parts.append(f"-cc=dark {cc.cold_sigma} {cc.hot_sigma}")
    elif cc.method == "bpm" and cc.bpm_file:
        parts.append(f"-cc=bpm {cc.bpm_file}")

    if is_cfa:
        parts.append("-cfa")
    if debayer and is_cfa:
        parts.append("-debayer")
    if equalize_cfa and is_cfa:
        parts.append("-equalize_cfa")
    if fix_xtrans and is_cfa:
        parts.append("-fix_xtrans")

    if optimize_dark == "exp":
        parts.append("-opt=exp")
    elif optimize_dark == "auto":
        parts.append("-opt")

    if process_all:
        parts.append("-all")
    if prefix is not None:
        parts.append(f"-prefix={prefix}")
    if output_fitseq:
        parts.append("-fitseq")

    calibrate_cmd = " ".join(parts)
    result = run_siril_script([calibrate_cmd], working_dir=working_dir, timeout=600)

    calibrated, bad_pixels = _parse_calibrate_output(result.stdout)

    output_prefix = prefix if prefix is not None else "pp_"
    calibrated_seq = f"{output_prefix}{lights_sequence}"
    calibrated_seq_path = Path(working_dir) / f"{calibrated_seq}.seq"

    return {
        "calibrated_sequence":      calibrated_seq,
        "calibrated_sequence_path": str(calibrated_seq_path),
        "calibrated_count":         calibrated,
        "bad_pixel_count":          bad_pixels,
    }
