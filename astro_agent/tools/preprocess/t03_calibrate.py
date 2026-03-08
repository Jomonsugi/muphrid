"""
T03 — siril_calibrate

Apply master calibration frames (bias, dark, flat) to each raw light sub.
Optionally debayers CFA/OSC data and applies cosmetic correction.

Siril commands (verified against Siril 1.4 CLI docs):
    calibrate sequencename
        [-bias=filename]    # or -bias="=<expression>" for uniform level
        [-dark=filename]
        [-flat=filename]
        [-cc=dark [siglo sighi] | -cc=bpm bpmfile]
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


# ── Pydantic input schema ──────────────────────────────────────────────────────

class CosmeticCorrectionOptions(BaseModel):
    """Options for pixel cosmetic correction during calibration."""
    method: str = Field(
        default="dark",
        description=(
            "'dark': Detect hot/cold pixels from the master dark. Requires master_dark. "
            "'bpm': Use a Bad Pixel Map file for correction. Set bpm_file path. "
            "'none': Skip cosmetic correction."
        ),
    )
    cold_sigma: float = Field(
        default=3.0,
        description=(
            "Sigma threshold for cold pixel detection. 0 = disable cold pixel correction. "
            "Only used with method='dark'."
        ),
    )
    hot_sigma: float = Field(
        default=5.0,
        description=(
            "Sigma threshold for hot pixel detection. "
            "Only used with method='dark'."
        ),
    )
    bpm_file: str | None = Field(
        default=None,
        description=(
            "Absolute path to Bad Pixel Map file. "
            "Only used with method='bpm'. Generate with find_hot on a master dark."
        ),
    )


class SirilCalibrateInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    lights_sequence: str = Field(
        description="Name of the lights sequence (without .seq extension)."
    )
    master_bias: str | None = Field(
        default=None,
        description=(
            "Absolute path to master bias FITS, OR a quoted expression for uniform "
            "level (e.g. '=256' or '=64*$OFFSET'). Pass null to skip bias subtraction."
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
            "Always True for camera RAW input."
        )
    )
    debayer: bool = Field(
        default=True,
        description="Debayer (demosaic) CFA data after calibration.",
    )
    equalize_cfa: bool = Field(
        default=True,
        description=(
            "Equalize CFA channel mean intensities from the master flat to "
            "prevent color tinting. Critical for Fujifilm X-Trans."
        ),
    )
    fix_xtrans: bool = Field(
        default=False,
        description=(
            "Apply X-Trans correction on darks and biases to remove rectangle "
            "pattern caused by autofocus (-fix_xtrans). "
            "Enable for Fujifilm X-Trans cameras (X-T30, X-T4, X-S10, etc.)."
        ),
    )
    cosmetic_correction: CosmeticCorrectionOptions = Field(
        default_factory=CosmeticCorrectionOptions,
    )
    optimize_dark: str | None = Field(
        default=None,
        description=(
            "Dark optimization mode. "
            "'auto': Siril computes dark scaling coefficient automatically (-opt). "
            "Requires both bias and dark masters. "
            "'exp': Calculate coefficient from exposure keyword (-opt=exp). "
            "Null: No dark optimization."
        ),
    )
    process_all: bool = Field(
        default=False,
        description="Process all frames including those marked as excluded (-all).",
    )
    prefix: str | None = Field(
        default=None,
        description=(
            "Custom output prefix. Default is 'pp_'. "
            "Set to change the calibrated sequence naming."
        ),
    )
    output_fitseq: bool = Field(
        default=False,
        description="Output as a FITS sequence (single file) instead of individual FITS (-fitseq).",
    )


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_calibrate_output(stdout: str, seq_name: str) -> tuple[int, int]:
    """Return (calibrated_count, bad_pixel_count) from Siril stdout."""
    cal_match = re.search(r"(\d+)\s+(?:image[s]?\s+)?calibrated", stdout, re.IGNORECASE)
    calibrated = int(cal_match.group(1)) if cal_match else 0

    pixel_match = re.search(
        r"(\d+)\s+(?:bad|hot|cold)\s+pixel", stdout, re.IGNORECASE
    )
    bad_pixels = int(pixel_match.group(1)) if pixel_match else 0

    return calibrated, bad_pixels


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SirilCalibrateInput)
def siril_calibrate(
    working_dir: str,
    lights_sequence: str,
    is_cfa: bool,
    master_bias: str | None = None,
    master_dark: str | None = None,
    master_flat: str | None = None,
    debayer: bool = True,
    equalize_cfa: bool = True,
    fix_xtrans: bool = False,
    cosmetic_correction: CosmeticCorrectionOptions | None = None,
    optimize_dark: str | None = None,
    process_all: bool = False,
    prefix: str | None = None,
    output_fitseq: bool = False,
) -> dict:
    """
    Apply master calibration frames to raw light subs. Handles CFA debayering,
    cosmetic correction, and dark optimization.

    For Fujifilm X-Trans cameras: set is_cfa=True, equalize_cfa=True, and
    fix_xtrans=True to correct the rectangle pattern from autofocus.

    Bias can be a file path or a uniform level expression (e.g. '=256').

    Cosmetic correction modes:
      'dark': Detect hot/cold pixels from master dark (requires master_dark).
      'bpm':  Use a Bad Pixel Map file (generate with find_hot on master dark).
      'none': Skip cosmetic correction.

    Dark optimization:
      'auto': Siril computes scaling coefficient (requires bias + dark).
      'exp':  Use exposure keyword for coefficient calculation.
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

    # Cosmetic correction
    cc = cosmetic_correction
    if cc.method == "dark" and master_dark:
        parts.append(f"-cc=dark {cc.cold_sigma} {cc.hot_sigma}")
    elif cc.method == "bpm" and cc.bpm_file:
        parts.append(f"-cc=bpm {cc.bpm_file}")

    # CFA options
    if is_cfa:
        parts.append("-cfa")
    if debayer and is_cfa:
        parts.append("-debayer")
    if equalize_cfa and is_cfa:
        parts.append("-equalize_cfa")
    if fix_xtrans and is_cfa:
        parts.append("-fix_xtrans")

    # Dark optimization
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

    calibrated, bad_pixels = _parse_calibrate_output(result.stdout, lights_sequence)

    output_prefix = prefix if prefix is not None else "pp_"
    calibrated_seq = f"{output_prefix}{lights_sequence}"
    calibrated_seq_path = Path(working_dir) / f"{calibrated_seq}.seq"

    return {
        "calibrated_sequence": calibrated_seq,
        "calibrated_sequence_path": str(calibrated_seq_path),
        "calibrated_count": calibrated,
        "bad_pixel_count": bad_pixels,
    }
