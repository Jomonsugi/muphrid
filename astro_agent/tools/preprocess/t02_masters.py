"""
T02 — build_masters

Stack calibration frames (bias, dark, or flat) into a single master frame.

Order constraint (enforced by the agent, not this tool):
    bias → dark (bias-subtracted) → flat (bias-subtracted)

Siril commands used:
    convert <seq_name> -out=<working_dir>   # link/convert files into a sequence
    stack   <seq_name> <method> ...         # stack into master_<type>.fit
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import SirilError, run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class BuildMastersInput(BaseModel):
    working_dir: str = Field(
        description=(
            "Absolute path to the directory where Siril will write the master frame. "
            "Input files are copied here before conversion."
        )
    )
    file_type: str = Field(
        description="Type of calibration frame: 'bias', 'dark', or 'flat'."
    )
    input_files: list[str] = Field(
        description="Absolute paths to the raw calibration frames to stack."
    )
    master_bias_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to master bias FITS. Required when file_type='flat' "
            "to subtract bias pedestal before flat normalization. Optional for darks."
        ),
    )
    stack_method: str = Field(
        default="median",
        description=(
            "'median' (recommended for bias — lowest noise floor) or "
            "'average' (mean — slightly higher SNR for darks/flats with good rejection)."
        ),
    )
    rejection_method: str = Field(
        default="sigma_clipping",
        description=(
            "Statistical rejection to eliminate cosmic rays and satellite trails. "
            "'sigma_clipping' (default), 'winsorized' (better for small N < 15), "
            "or 'none'."
        ),
    )
    rejection_sigma: list[float] = Field(
        default=[3.0, 3.0],
        description="[low_sigma, high_sigma] for rejection. Default [3.0, 3.0].",
    )


# ── Siril rejection method map ─────────────────────────────────────────────────

_REJECTION_MAP = {
    "sigma_clipping": "rej",
    "winsorized":     "wrej",
    "none":           "norej",
}

_STACK_METHOD_MAP = {
    "median":  "median",
    "average": "mean",
}


# ── Stats extraction ───────────────────────────────────────────────────────────

def _extract_stats(stdout: str) -> dict:
    """Pull mean, median, noise from Siril's stat output lines."""
    stats: dict = {}

    mean_match = re.search(r"Mean:\s*([\d.e+-]+)", stdout, re.IGNORECASE)
    if mean_match:
        stats["mean_value"] = float(mean_match.group(1))

    med_match = re.search(r"Median:\s*([\d.e+-]+)", stdout, re.IGNORECASE)
    if med_match:
        stats["median_value"] = float(med_match.group(1))

    noise_match = re.search(
        r"(?:Noise|Background noise level|bgnoise):\s*([\d.e+-]+)",
        stdout,
        re.IGNORECASE,
    )
    if noise_match:
        stats["noise_estimate"] = float(noise_match.group(1))

    return stats


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=BuildMastersInput)
def build_masters(
    working_dir: str,
    file_type: str,
    input_files: list[str],
    master_bias_path: str | None = None,
    stack_method: str = "median",
    rejection_method: str = "sigma_clipping",
    rejection_sigma: list[float] | None = None,
) -> dict:
    """
    Stack calibration frames (bias/dark/flat) into a master calibration frame.

    Call in order: bias first, then dark (optionally with master_bias),
    then flat (with master_bias). Returns master_path and frame statistics.
    """
    if rejection_sigma is None:
        rejection_sigma = [3.0, 3.0]

    file_type = file_type.lower().rstrip("s")  # normalise "darks" → "dark"
    if file_type not in ("bias", "dark", "flat"):
        raise ValueError(f"file_type must be 'bias', 'dark', or 'flat'. Got: {file_type!r}")

    rej_code = _REJECTION_MAP.get(rejection_method, "rej")
    stack_cmd = _STACK_METHOD_MAP.get(stack_method, "median")

    wdir = Path(working_dir)
    wdir.mkdir(parents=True, exist_ok=True)

    # Copy input files into a temp subdir so Siril's convert can find them
    # with clean, sequential naming it expects for sequence building.
    with tempfile.TemporaryDirectory(dir=wdir, prefix=f"{file_type}_raw_") as tmpdir:
        tmp = Path(tmpdir)
        for i, src in enumerate(sorted(input_files)):
            suffix = Path(src).suffix
            dst = tmp / f"{file_type}_{i:04d}{suffix}"
            shutil.copy2(src, dst)

        seq_name = f"{file_type}_seq"
        master_out = f"master_{file_type}"

        commands: list[str] = [
            f"convert {file_type}_seq -out={wdir} -fitseq",
        ]

        # Build the stack command
        sigma_lo, sigma_hi = rejection_sigma[0], rejection_sigma[1]
        stack_line = (
            f"stack {seq_name} {stack_cmd} {rej_code} {sigma_lo} {sigma_hi} "
            f"-norm=addscale -out={master_out}"
        )

        # For flat frames: subtract master bias before stacking
        if file_type == "flat" and master_bias_path:
            commands.append(
                f"stack {seq_name} {stack_cmd} {rej_code} {sigma_lo} {sigma_hi} "
                f"-bias={master_bias_path} -norm=mulscale -out={master_out}"
            )
        else:
            commands.append(stack_line)

        try:
            result = run_siril_script(commands, working_dir=str(tmp), timeout=300)
        except SirilError:
            # Re-run with working_dir = wdir (sequence was written there by -out)
            result = run_siril_script(
                [
                    f"stack {seq_name} {stack_cmd} {rej_code} {sigma_lo} {sigma_hi} "
                    + (
                        f"-bias={master_bias_path} -norm=mulscale -out={master_out}"
                        if file_type == "flat" and master_bias_path
                        else f"-norm=addscale -out={master_out}"
                    )
                ],
                working_dir=str(wdir),
                timeout=300,
            )

    master_fits = wdir / f"{master_out}.fit"
    if not master_fits.exists():
        # Siril sometimes writes .fits
        master_fits = wdir / f"{master_out}.fits"
    if not master_fits.exists():
        raise FileNotFoundError(
            f"Expected master frame not found at {wdir / master_out}.fit[s]. "
            f"Siril stdout:\n{result.stdout[-1000:]}"
        )

    stats = _extract_stats(result.stdout)
    stats["frame_count"] = len(input_files)

    return {
        "master_path": str(master_fits),
        "stats": stats,
    }
