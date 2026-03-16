"""
T02 — build_masters

Stack calibration frames (bias, dark, or flat) into a single master frame.

Order constraint (enforced by the agent, not this tool):
    bias → dark (bias-subtracted) → flat (bias-subtracted)

Siril commands used (verified against Siril 1.4 docs):
    convert <name> -out=<dir> -fitseq         # build sequence
    calibrate <seq> -bias=<path> [-fitseq]     # flat: subtract bias before stacking
    stack <seq> median [-norm=] -out=<name>     # median stack (no rejection args)
    stack <seq> rej <type> <lo> <hi> [-norm=]  # mean stack with rejection

Rejection type codes for 'rej' (mean) stacking:
    s = sigma clipping, w = winsorized, n = none, p = percentile,
    m = median, l = linear-fit, g = generalized ESD, a = k-MAD

Sensor-relative diagnostic thresholds:
    Thresholds for flat and bias quality checks are computed from sensor
    black_level and white_level (passed via acquisition_meta from T01). This
    makes the checks correct for any camera — 14-bit mirrorless, 12-bit DSLR,
    16-bit dedicated astro cam — rather than assuming 16-bit full range.

    If acquisition_meta is not provided, the tool falls back to reading EXIF
    from the first input file, and then to conservative 16-bit defaults.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import numpy as np
from astropy.io import fits
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from astro_agent.graph.state import AstroState
from astro_agent.tools._sensor import (
    flat_fill_state,
    flat_siril_norm_thresholds,
    read_frame_exif,
)
from astro_agent.tools._siril import SirilError, run_siril_script
from astro_agent.tools.preprocess.t02b_convert_sequence import _convert_to_sequence


# ── Pydantic input schema ────────────────────────────────────────────────────

class BuildMastersInput(BaseModel):
    file_type: str = Field(
        description=(
            "Which calibration type to build: 'bias', 'dark', or 'flat'. "
            "Build in order: bias first, then dark, then flat."
        ),
    )
    stack_method: str = Field(
        description=(
            "Stacking method. 'median': lowest noise floor, best for bias and "
            "small frame counts. 'mean': higher SNR when combined with rejection, "
            "preferred for darks/flats with enough frames (> 15). "
            "Choose based on frame count and frame type."
        ),
    )
    rejection_method: str = Field(
        description=(
            "Rejection algorithm (only used when stack_method='mean'). "
            "'winsorized': best for small N (< 15). "
            "'sigma': reliable for medium N (15–50). "
            "'linear': best for large N (> 50) where Gaussian statistics hold. "
            "'none': no rejection. "
            "Choose based on frame count. Ignored when stack_method='median'."
        ),
    )
    rejection_sigma: list[float] = Field(
        default=[3.0, 3.0],
        description=(
            "[low_sigma, high_sigma] for rejection clipping. "
            "Tighter (e.g. [2.5, 2.5]) rejects more aggressively — use with large N. "
            "Looser (e.g. [4.0, 4.0]) preserves more frames — use with small N."
        ),
    )


# ── Maps (verified against Siril 1.4 docs) ─────────────────────────────────────

_TYPE_NORMALIZE = {
    "bias":   "bias",
    "biases": "bias",
    "dark":   "dark",
    "darks":  "dark",
    "flat":   "flat",
    "flats":  "flat",
}

_REJECTION_CODE = {
    "sigma":        "s",
    "sigma_clipping": "s",
    "winsorized":   "w",
    "none":         "n",
    "percentile":   "p",
    "median":       "m",
    "linear":       "l",
    "generalized":  "g",
    "mad":          "a",
}


# ── Command builders ────────────────────────────────────────────────────────────

def _build_stack_cmd(
    seq_name: str,
    stack_method: str,
    rejection_method: str,
    sigma_lo: float,
    sigma_hi: float,
    norm: str,
    out_name: str,
) -> str:
    """Build a correctly-formed Siril stack command line."""
    if stack_method == "median":
        return f"stack {seq_name} median -norm={norm} -out={out_name}"

    rej_code = _REJECTION_CODE.get(rejection_method, "w")
    return (
        f"stack {seq_name} rej {rej_code} {sigma_lo} {sigma_hi} "
        f"-norm={norm} -out={out_name}"
    )


# ── Sensor level resolution ─────────────────────────────────────────────────────

def _resolve_sensor_levels(
    acquisition_meta: dict | None,
    input_files: list[str],
) -> tuple[int, int]:
    """
    Determine sensor black_level and white_level for threshold computation.

    Priority:
      1. acquisition_meta (from T01 — already paid for, most reliable)
      2. EXIF from first input file (fallback if T01 wasn't run / not passed)
      3. Conservative 16-bit defaults (last resort)
    """
    if acquisition_meta:
        black = acquisition_meta.get("black_level")
        white = acquisition_meta.get("white_level")
        if black is not None and white is not None and white > black:
            return int(black), int(white)

    if input_files:
        try:
            frame_exif = read_frame_exif(Path(input_files[0]))
            s = frame_exif.sensor
            if s.white_level > s.black_level:
                return s.black_level, s.white_level
        except Exception:
            pass

    # Conservative 16-bit default — thresholds will be correct for 16-bit
    # and slightly permissive for lower bit-depth cameras, but never wrong.
    return 0, 65535


# ── Per-frame sequence reader ────────────────────────────────────────────────────

def _read_fitseq_per_frame_medians(fitseq_path: Path) -> list[float]:
    """
    Read per-frame medians from a Siril FITSEQ (multi-extension FITS).

    Siril stores each frame as a separate FITS extension. The primary HDU
    (index 0) is usually an empty header with no image data. Returns a list
    of median pixel values, one per frame, in ADU.
    """
    medians: list[float] = []
    try:
        with fits.open(str(fitseq_path)) as hdul:
            for hdu in hdul:
                if hdu.data is None or hdu.data.size == 0:
                    continue
                medians.append(float(np.median(hdu.data.astype(np.float64))))
    except Exception:
        pass
    return medians


# ── Diagnostics ─────────────────────────────────────────────────────────────────

def _compute_diagnostics(
    master_path: Path,
    file_type: str,
    frame_count: int,
    siril_stdout: str,
    master_bias_path: str | None,
    acquisition_meta: dict | None = None,
    input_files: list[str] | None = None,
    seq_path: Path | None = None,
) -> dict:
    """
    Compute quality flags and warnings from the master FITS.

    Diagnostic checks fire only for clearly broken situations — wrong frame type,
    impossible signal levels, or batches too small to stack. Suboptimal-but-usable
    situations are reported as warnings for the agent to reason about.

    Thresholds are sensor-relative so they are correct for any camera.
    """
    quality_flags: dict = {"frame_count": frame_count}
    warnings: list[str] = []
    quality_issues: list[str] = []

    black, white = _resolve_sensor_levels(acquisition_meta, input_files or [])
    quality_flags["sensor_black"] = black
    quality_flags["sensor_white"] = white

    # ── Frame count: warn if clearly insufficient ─────────────────────────
    if frame_count < 2:
        quality_issues.append(
            f"Only {frame_count} {file_type} frame(s) provided. A minimum of 2 is "
            f"required for any meaningful stacking or noise averaging. "
            f"Capture more {file_type} frames before proceeding."
        )
    elif frame_count < 5:
        warnings.append(
            f"Low {file_type} frame count ({frame_count}). Stacking will succeed but "
            f"noise averaging and rejection are limited. 15–30 frames is typical."
        )

    # ── Read master FITS for pixel-level diagnostics ──────────────────────
    try:
        with fits.open(str(master_path)) as hdul:
            data = hdul[0].data.astype(np.float64)

            quality_flags["mean"]   = float(np.mean(data))
            quality_flags["median"] = float(np.median(data))
            quality_flags["std"]    = float(np.std(data))
            quality_flags["uniformity"] = float(
                1.0 - (np.std(data) / np.mean(data)) if np.mean(data) > 0 else 0.0
            )

            if data.max() > 0:
                quality_flags["hot_pixel_pct"] = float(
                    np.sum(data > 0.95 * data.max()) / data.size * 100
                )
            else:
                quality_flags["hot_pixel_pct"] = 0.0

            if file_type == "flat":
                quality_flags["flat_median_normalized"] = quality_flags["median"]

            if file_type == "bias":
                quality_flags["bias_mean"] = quality_flags["mean"]

    except Exception as e:
        warnings.append(f"Could not read master FITS for diagnostics: {e}")

    # ── Parse rejection counts from Siril output ──────────────────────────
    stacked_match  = re.search(r"(\d+)\s+frame[s]?\s+stacked",  siril_stdout, re.IGNORECASE)
    rejected_match = re.search(r"(\d+)\s+frame[s]?\s+rejected", siril_stdout, re.IGNORECASE)
    if stacked_match:
        quality_flags["frames_stacked"]  = int(stacked_match.group(1))
    if rejected_match:
        quality_flags["frames_rejected"] = int(rejected_match.group(1))

    rejection_rate = quality_flags.get("frames_rejected", 0) / max(frame_count, 1)
    quality_flags["rejection_rate"] = rejection_rate

    # ── Type-specific quality checks ──────────────────────────────────────

    if file_type == "flat":
        flat_med = quality_flags.get("flat_median_normalized", 0.0)

        norm_min, norm_max = flat_siril_norm_thresholds(black, white)
        quality_flags["flat_norm_threshold_min"] = round(norm_min, 4)
        quality_flags["flat_norm_threshold_max"] = round(norm_max, 4)

        if flat_med < norm_min or flat_med > norm_max:
            fill_pct_approx = (flat_med * 65535) / max(white - black, 1) * 100
            quality_issues.append(
                f"Flat median ({flat_med:.3f} Siril-norm, ~{fill_pct_approx:.0f}% fill) "
                f"outside sensor-relative safe range [{norm_min:.3f}, {norm_max:.3f}] "
                f"(= 30–55% of usable ADU range: black={black}, white={white}). "
                f"Likely under/overexposed or wrong frame type."
            )

        if master_bias_path:
            try:
                with fits.open(master_bias_path) as bhdul:
                    bias_med = float(np.median(bhdul[0].data.astype(np.float64)))
                    if flat_med > 0 and flat_med < 2 * bias_med:
                        quality_issues.append(
                            f"Flat median ({flat_med:.3f}) < 2× bias median ({bias_med:.3f}) "
                            f"— flat signal too weak for reliable calibration."
                        )
            except Exception as e:
                warnings.append(
                    f"Could not read master bias for flat comparison: {e}"
                )

    if file_type == "bias":
        bias_mean = quality_flags.get("bias_mean", 0.0)

        # Existing check: bias pedestal should not exceed black_level + 5% usable
        sensor_bias_threshold = (black + 0.05 * (white - black)) / 65535.0
        quality_flags["bias_threshold"] = round(sensor_bias_threshold, 5)
        if bias_mean > sensor_bias_threshold:
            quality_issues.append(
                f"Bias mean ({bias_mean:.5f}) exceeds sensor-relative threshold "
                f"({sensor_bias_threshold:.5f} = black_level + 5% of usable range). "
                f"Possible dark or flat mislabeled as bias."
            )

        # New: check whether ALL frames in the sequence are near saturation —
        # the clearest signal that the entire bias batch is wrong frame type
        if seq_path and seq_path.exists():
            per_frame = _read_fitseq_per_frame_medians(seq_path)
            if per_frame:
                quality_flags["bias_per_frame_medians"] = [round(m, 1) for m in per_frame]
                sat_threshold_adu = 0.85 * white
                saturated_count = sum(1 for m in per_frame if m > sat_threshold_adu)
                if saturated_count == len(per_frame):
                    quality_issues.append(
                        f"All {len(per_frame)} bias frames have median ADU > 85% of sensor "
                        f"white level ({white}). These are not bias frames — they are "
                        f"saturated. Check that the bias folder contains minimum-exposure "
                        f"frames, not flats or lights."
                    )
                elif saturated_count > 0:
                    warnings.append(
                        f"{saturated_count}/{len(per_frame)} bias frames are near saturation. "
                        f"Inspect the bias folder for mixed frame types."
                    )

    if file_type == "dark":
        dark_mean = quality_flags.get("mean", 0.0)

        # Dark master mean should not look like a flat or light:
        # dark current on top of bias pedestal is modest for camera sensors;
        # if dark_mean exceeds bias + 70% of usable range, something is wrong.
        dark_max_threshold = (black + 0.70 * (white - black)) / 65535.0
        quality_flags["dark_max_threshold"] = round(dark_max_threshold, 5)
        if dark_mean > dark_max_threshold:
            quality_issues.append(
                f"Dark master mean ({dark_mean:.5f}) exceeds expected maximum "
                f"({dark_max_threshold:.5f} = black_level + 70% of usable range). "
                f"Dark frames appear too bright — possible flat or light mislabeled as dark."
            )

        # Compare to actual master bias if available
        if master_bias_path:
            try:
                with fits.open(master_bias_path) as bhdul:
                    bias_mean_norm = float(np.mean(bhdul[0].data.astype(np.float64)))
                    quality_flags["bias_mean_for_dark_check"] = bias_mean_norm
                    # Dark master (non-bias-subtracted) must have mean ≥ bias mean
                    if dark_mean < bias_mean_norm * 0.95:
                        quality_issues.append(
                            f"Dark master mean ({dark_mean:.5f}) is lower than bias mean "
                            f"({bias_mean_norm:.5f}). Dark frames cannot have less signal "
                            f"than bias — possible accidental calibration or wrong folder."
                        )
            except Exception as e:
                warnings.append(
                    f"Could not read master bias for dark comparison: {e}"
                )

    # ── Universal: high rejection rate ────────────────────────────────────
    if rejection_rate > 0.40:
        quality_issues.append(
            f"Rejection rate ({rejection_rate:.0%}) exceeds 40% — "
            f"majority of frames are outliers; possible capture issue."
        )

    return {
        "quality_flags": quality_flags,
        "warnings": warnings,
        "quality_issues": quality_issues,
    }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=BuildMastersInput)
def build_masters(
    file_type: str,
    stack_method: str,
    rejection_method: str,
    rejection_sigma: list[float] = [3.0, 3.0],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Stack calibration frames (bias/dark/flat) into a master calibration frame.

    File lists and working directory come from the dataset in state. Master bias
    path is automatically injected for darks and flats that need it.

    Args:
        file_type: Which calibration type to build: 'bias', 'dark', or 'flat'.
            Build in order: bias first, then dark, then flat.
        stack_method: 'median' (recommended for bias — lowest noise floor) or
            'mean' (higher SNR for darks/flats with good rejection).
        rejection_method: Rejection algorithm (only for stack_method='mean').
            'sigma', 'winsorized' (best for small N < 15), or 'none'.
        rejection_sigma: [low_sigma, high_sigma] for rejection. Default [3.0, 3.0].
    """
    working_dir = state["dataset"]["working_dir"]
    acquisition_meta = state["dataset"].get("acquisition_meta")

    ft = _TYPE_NORMALIZE.get(file_type.lower().strip(), file_type.lower().strip())
    if ft not in ("bias", "dark", "flat"):
        raise ValueError(f"file_type must be 'bias', 'dark', or 'flat'. Got: {file_type!r}")
    if stack_method not in ("median", "mean"):
        raise ValueError(f"stack_method must be 'median' or 'mean'. Got: {stack_method!r}")

    # Resolve input files from dataset
    files_map = {"bias": "biases", "dark": "darks", "flat": "flats"}
    input_files = state["dataset"]["files"].get(files_map[ft], [])
    if not input_files:
        raise ValueError(f"No {ft} files found in dataset. Check the dataset folder structure.")

    # Inject master_bias_path for darks and flats
    master_bias_path = state["paths"]["masters"].get("bias")

    wdir = Path(working_dir)
    wdir.mkdir(parents=True, exist_ok=True)

    seq_name = f"{ft}_seq"
    master_out = f"master_{ft}"

    # ── Step 1: Convert raw frames to a FITSEQ sequence ───────────────────
    _convert_to_sequence(
        working_dir=str(wdir),
        input_files=input_files,
        sequence_name=seq_name,
        debayer=False,
    )

    # ── Step 2: For flats, calibrate (bias-subtract) before stacking ──────
    stack_seq = seq_name
    if ft == "flat" and master_bias_path:
        run_siril_script(
            [f"calibrate {seq_name} -bias={master_bias_path} -fitseq"],
            working_dir=str(wdir),
            timeout=300,
        )
        stack_seq = f"pp_{seq_name}"

    # ── Step 3: Stack ─────────────────────────────────────────────────────
    norm = "mulscale" if ft == "flat" else "addscale"

    stack_cmd = _build_stack_cmd(
        seq_name=stack_seq,
        stack_method=stack_method,
        rejection_method=rejection_method,
        sigma_lo=rejection_sigma[0],
        sigma_hi=rejection_sigma[1],
        norm=norm,
        out_name=master_out,
    )

    result = run_siril_script([stack_cmd], working_dir=str(wdir), timeout=300)

    # ── Step 4: Locate master output ──────────────────────────────────────
    master_fits = wdir / f"{master_out}.fit"
    if not master_fits.exists():
        master_fits = wdir / f"{master_out}.fits"
    if not master_fits.exists():
        raise FileNotFoundError(
            f"Expected master frame not found at {wdir / master_out}.fit[s]. "
            f"Siril stdout:\n{result.stdout[-1000:]}"
        )

    # ── Step 5: Diagnostics ───────────────────────────────────────────────
    seq_fits = wdir / f"{seq_name}.fit"
    diagnostics = _compute_diagnostics(
        master_path=master_fits,
        file_type=ft,
        frame_count=len(input_files),
        siril_stdout=result.stdout,
        master_bias_path=master_bias_path,
        acquisition_meta=acquisition_meta,
        input_files=input_files,
        seq_path=seq_fits if seq_fits.exists() else None,
    )

    # ── Step 6: Write master path back to state ────────────────────────────
    masters = {**state["paths"]["masters"], ft: str(master_fits)}

    # Build a clear, concise result for the model.
    # Separate quality_issues (critical) from warnings (informational) so the
    # agent can distinguish "low frame count" from "wrong frame type."
    quality = diagnostics.get("quality_flags", {})
    warns = diagnostics.get("warnings", [])
    issues = diagnostics.get("quality_issues", [])

    # Summarize per-frame medians instead of dumping the full array
    per_frame = quality.pop("bias_per_frame_medians", None)
    per_frame_summary = None
    if per_frame:
        unique = sorted(set(round(v, 1) for v in per_frame))
        if len(unique) == 1:
            per_frame_summary = f"all {len(per_frame)} frames at {unique[0]} ADU"
        else:
            per_frame_summary = (
                f"{len(per_frame)} frames, median range {min(unique)}–{max(unique)} ADU"
            )

    result = {
        "file_type": ft,
        "master_path": str(master_fits),
        "frame_count": len(input_files),
        "quality_flags": quality,
    }
    if per_frame_summary:
        result["per_frame_summary"] = per_frame_summary
    if issues:
        result["quality_issues"] = issues
    if warns:
        result["warnings"] = warns

    result_content = json.dumps(result, indent=2)
    return Command(update={
        "paths": {**state["paths"], "masters": masters},
        "messages": [ToolMessage(content=result_content, tool_call_id=tool_call_id)],
    })
