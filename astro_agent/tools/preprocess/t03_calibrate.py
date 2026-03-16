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

import json
import re
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from astro_agent.graph.state import AstroState
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

@tool
def calibrate(
    is_cfa: bool,
    debayer: bool,
    equalize_cfa: bool,
    fix_xtrans: bool,
    cosmetic_correction: CosmeticCorrectionOptions | None,
    optimize_dark: str | None,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AstroState, InjectedState],
) -> Command:
    """
    Apply master calibration frames to raw light subs in CFA (raw) space,
    then debayer. Working directory, lights sequence, and master paths are
    read from state automatically.

    Camera-specific settings (derived from sensor_type in dataset.acquisition_meta):

      Fujifilm X-Trans (X-T30, X-T4, X-S10, X-H2, etc.):
        is_cfa=True, debayer=True, fix_xtrans=True, equalize_cfa=True

      Bayer OSC/DSLR/mirrorless (Canon, Nikon, Sony, ZWO OSC):
        is_cfa=True, debayer=True, fix_xtrans=False, equalize_cfa=False

      Monochrome (ZWO mono, Atik, QHY mono):
        is_cfa=False, debayer=False

    Args:
        is_cfa: True for any OSC/DSLR/mirrorless camera (Bayer or X-Trans).
            False only for monochrome sensors.
        debayer: Demosaic CFA data after calibration. True for all color cameras.
        equalize_cfa: Equalize CFA channel means to prevent color tinting.
            Required for Fujifilm X-Trans; not needed for Bayer.
        fix_xtrans: Correct rectangle artifacts from Fuji phase-detection AF pixels.
            Required for all Fujifilm X-Trans cameras.
        cosmetic_correction: Hot/cold pixel correction options. Null uses defaults.
        optimize_dark: Dark scaling: 'auto' (Siril computes coefficient),
            'exp' (from EXPTIME keyword), or null (no scaling — use when
            dark and light exposures match exactly).
    """
    if cosmetic_correction is None:
        cosmetic_correction = CosmeticCorrectionOptions()

    working_dir = state["dataset"]["working_dir"]
    lights_sequence = state["paths"]["lights_sequence"]
    masters = state["paths"]["masters"]
    master_bias = masters.get("bias")
    master_dark = masters.get("dark")
    master_flat = masters.get("flat")

    if not lights_sequence:
        raise ValueError("lights_sequence not found in state. Run convert_sequence first.")

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

    calibrate_cmd = " ".join(parts)
    result = run_siril_script([calibrate_cmd], working_dir=working_dir, timeout=600)

    calibrated, bad_pixels = _parse_calibrate_output(result.stdout)

    calibrated_seq = f"pp_{lights_sequence}"

    summary = {
        "calibrated_sequence": calibrated_seq,
        "lights_sequence": lights_sequence,
        "calibrated_count": calibrated,
        "bad_pixels_corrected": bad_pixels,
        "masters_applied": {
            "bias": master_bias,
            "dark": master_dark,
            "flat": master_flat,
        },
        "settings": {
            "is_cfa": is_cfa,
            "debayer": debayer,
            "equalize_cfa": equalize_cfa,
            "fix_xtrans": fix_xtrans,
            "cosmetic_correction_method": cc.method,
            "optimize_dark": optimize_dark,
        },
        "siril_command": calibrate_cmd,
    }

    return Command(update={
        "paths": {**state["paths"], "calibrated_sequence": calibrated_seq},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
