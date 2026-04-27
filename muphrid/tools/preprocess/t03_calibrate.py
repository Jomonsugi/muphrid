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

from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script, siril_script_path


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
    """
    Return (calibrated_count, bad_pixel_count) from Siril stdout.

    Note: Siril's exact wording for the per-image calibration line varies
    across versions, datasets, and locales, so callers must NOT rely on
    `calibrated_count` from this function as ground truth for whether
    calibration succeeded — they should count the actual output FITS files
    on disk instead. The stdout-parsed count is kept as a supplementary
    diagnostic for when it happens to match.
    """
    cal_match = re.search(r"(\d+)\s+(?:image[s]?\s+)?calibrated", stdout, re.IGNORECASE)
    calibrated = int(cal_match.group(1)) if cal_match else 0

    pixel_match = re.search(
        r"(\d+)\s+(?:bad|hot|cold)\s+pixel", stdout, re.IGNORECASE
    )
    bad_pixels = int(pixel_match.group(1)) if pixel_match else 0

    return calibrated, bad_pixels


def _count_calibrated_files(working_dir: Path, calibrated_seq: str) -> int:
    """
    Ground-truth success check: count the per-frame output FITS files Siril
    wrote for the calibrated sequence.

    Siril `calibrate <seq>` produces `pp_<seq>_00001.fit`, `pp_<seq>_00002.fit`,
    etc., plus a `pp_<seq>.seq` sequence index. Counting the per-frame FITS
    files is invariant across Siril versions and stdout phrasing — if Siril
    actually did the work, these files exist; if it failed, they do not.

    The fitseq mode produces a single consolidated `pp_<seq>.fit` file
    instead of per-frame files. We handle both cases.
    """
    wd = Path(working_dir)
    # Per-frame layout (default): pp_<seq>_NNNNN.fit or .fits
    per_frame = sorted(wd.glob(f"{calibrated_seq}_*.fit")) + sorted(
        wd.glob(f"{calibrated_seq}_*.fits")
    )
    if per_frame:
        return len(per_frame)

    # fitseq layout: a single consolidated file — probe the sequence index
    # file. The `.seq` file's `I <n>` line records frame count, but reading
    # the consolidated FITS extension count is more direct and ties success
    # to the data, not the index.
    from astropy.io import fits as _fits
    consolidated = wd / f"{calibrated_seq}.fit"
    if not consolidated.exists():
        consolidated = wd / f"{calibrated_seq}.fits"
    if consolidated.exists():
        try:
            with _fits.open(str(consolidated)) as hdul:
                # Primary HDU is typically empty; count image-bearing HDUs.
                return sum(1 for hdu in hdul if hdu.data is not None and hdu.data.size > 0)
        except Exception:
            # Existence alone is a weaker positive signal. Fall through.
            return 1

    return 0


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

    # Siril's script tokenizer splits on whitespace BEFORE honoring quotes,
    # so quoting paths that contain spaces does not work. siril_script_path
    # sidesteps the issue by handing Siril a relative name (master files
    # live inside working_dir) or an auto-created symlink in working_dir.
    # The "=<expression>" branch preserves Siril's synthetic-bias syntax
    # (e.g. -bias="=2048"), which is a value, not a path.
    if master_bias:
        if master_bias.startswith("="):
            parts.append(f'-bias="{master_bias}"')
        else:
            parts.append(f"-bias={siril_script_path(master_bias, working_dir)}")

    if master_dark:
        parts.append(f"-dark={siril_script_path(master_dark, working_dir)}")

    if master_flat:
        parts.append(f"-flat={siril_script_path(master_flat, working_dir)}")

    cc = cosmetic_correction
    if cc.method == "dark" and master_dark:
        parts.append(f"-cc=dark {cc.cold_sigma} {cc.hot_sigma}")
    elif cc.method == "bpm" and cc.bpm_file:
        parts.append(f"-cc=bpm {siril_script_path(cc.bpm_file, working_dir)}")

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

    # Stdout parse stays as a supplementary diagnostic — NOT the source of
    # truth for success. Siril's per-image wording varies across versions and
    # the previous implementation leaked those variances straight into the
    # agent as `calibrated_count: 0`, which looked indistinguishable from a
    # genuine failure and triggered retry loops (see: Cocoon Nebula run).
    calibrated_from_stdout, bad_pixels = _parse_calibrate_output(result.stdout)

    calibrated_seq = f"pp_{lights_sequence}"

    # Ground truth: count the FITS files Siril actually produced. This is
    # invariant across Siril versions, locales, and per-frame output modes.
    wd = Path(working_dir)
    actual_count = _count_calibrated_files(wd, calibrated_seq)

    if actual_count == 0:
        # Nothing on disk — Siril really did not produce output. Raise with
        # enough context (stdout tail + reconstructed command) so the agent
        # can reason about the failure instead of blindly retrying.
        tail = result.stdout[-1200:] if result.stdout else "<empty>"
        raise RuntimeError(
            f"calibrate: Siril produced no calibrated files matching "
            f"'{calibrated_seq}*' in {working_dir}. The calibrate command did "
            f"not write per-frame outputs OR a consolidated fitseq. Check the "
            f"masters, input sequence integrity, and Siril stderr. "
            f"Command: {calibrate_cmd}\n"
            f"Stdout tail:\n{tail}"
        )

    # Cross-check: if stdout reported a different count than what landed on
    # disk, surface the discrepancy as an informational note rather than an
    # error. The disk count is authoritative; the stdout value is kept so
    # the agent can see the regex/Siril-wording drift if it ever matters.
    count_mismatch_note: str | None = None
    if calibrated_from_stdout and calibrated_from_stdout != actual_count:
        count_mismatch_note = (
            f"Siril stdout reported {calibrated_from_stdout} calibrated images "
            f"but {actual_count} output files were written. Using the disk "
            f"count as authoritative."
        )

    summary = {
        "calibrated_sequence": calibrated_seq,
        "lights_sequence": lights_sequence,
        "calibrated_count": actual_count,
        "calibrated_count_from_stdout": calibrated_from_stdout,
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
    if count_mismatch_note:
        summary["note"] = count_mismatch_note

    return Command(update={
        "paths": {**state["paths"], "calibrated_sequence": calibrated_seq},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
