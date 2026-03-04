"""
T03 — siril_calibrate

Apply master calibration frames (bias, dark, flat) to each raw light sub.
Optionally debayers CFA/OSC data and applies cosmetic correction.

For Fujifilm X-Trans (and all camera RAW input), always set is_cfa=True.
set equalize_cfa=True to avoid X-Trans color tinting during flat correction.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SirilCalibrateInput(BaseModel):
    working_dir: str = Field(
        description=(
            "Absolute path to the Siril working directory. "
            "The lights sequence (.seq) must already exist here (created by convert)."
        )
    )
    lights_sequence: str = Field(
        description=(
            "Name of the lights sequence (without .seq extension) as Siril knows it. "
            "e.g. 'lights_seq'"
        )
    )
    master_bias: str | None = Field(
        default=None,
        description="Absolute path to master bias FITS. Pass null to skip bias subtraction.",
    )
    master_dark: str | None = Field(
        default=None,
        description="Absolute path to master dark FITS. Pass null to skip dark subtraction.",
    )
    master_flat: str | None = Field(
        default=None,
        description="Absolute path to master flat FITS. Pass null to skip flat correction.",
    )
    is_cfa: bool = Field(
        description=(
            "True for OSC/DSLR/mirrorless cameras (Bayer or X-Trans CFA). "
            "Set based on AcquisitionMeta.input_format — always True for camera RAW."
        )
    )
    debayer: bool = Field(
        default=True,
        description=(
            "Debayer (demosaic) the CFA data after calibration. "
            "Should be True unless you want a mono CFA output."
        ),
    )
    equalize_cfa: bool = Field(
        default=True,
        description=(
            "Equalize CFA channels before flat correction. "
            "Critical for Fujifilm X-Trans to prevent color tinting."
        ),
    )
    cosmetic_correction: bool = Field(
        default=True,
        description="Remove hot and cold pixels using cosmetic correction.",
    )
    cc_sigma: list[float] = Field(
        default=[3.0, 5.0],
        description="[cold_sigma, hot_sigma] for cosmetic correction thresholds.",
    )
    optimize_dark: bool = Field(
        default=False,
        description=(
            "Scale dark frame to match light exposure. Enable when darks were shot "
            "at a slightly different temperature or exposure time."
        ),
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
    cosmetic_correction: bool = True,
    cc_sigma: list[float] | None = None,
    optimize_dark: bool = False,
) -> dict:
    """
    Apply master calibration frames to raw light subs. Handles CFA debayering
    for camera RAW input. Returns calibrated sequence path and frame counts.

    Always set is_cfa=True for RAF/CR2/OSC camera input.
    """
    if cc_sigma is None:
        cc_sigma = [3.0, 5.0]

    parts: list[str] = ["calibrate", lights_sequence]

    if master_bias:
        parts.append(f"-bias={master_bias}")
    if master_dark:
        dark_flag = f"-dark={master_dark}"
        if optimize_dark:
            dark_flag += " -opt"
        parts.append(dark_flag)
    if master_flat:
        parts.append(f"-flat={master_flat}")
    if is_cfa:
        parts.append("-cfa")
    if debayer and is_cfa:
        parts.append("-debayer")
    if equalize_cfa and is_cfa:
        parts.append("-equalize_cfa")
    if cosmetic_correction:
        cold_s, hot_s = cc_sigma[0], cc_sigma[1]
        parts.append(f"-cc=dark {cold_s} {hot_s}")

    calibrate_cmd = " ".join(parts)

    result = run_siril_script([calibrate_cmd], working_dir=working_dir, timeout=600)

    calibrated, bad_pixels = _parse_calibrate_output(result.stdout, lights_sequence)

    # Siril names calibrated sequence as pp_<seq_name>
    calibrated_seq = f"pp_{lights_sequence}"
    calibrated_seq_path = Path(working_dir) / f"{calibrated_seq}.seq"

    return {
        "calibrated_sequence": calibrated_seq,
        "calibrated_sequence_path": str(calibrated_seq_path),
        "calibrated_count": calibrated,
        "bad_pixel_count": bad_pixels,
    }
