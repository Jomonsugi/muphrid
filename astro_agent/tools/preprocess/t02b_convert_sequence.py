"""
T02b — convert_sequence

Convert a set of raw frames into a Siril FITSEQ sequence.  This is the
prerequisite for any Siril sequence operation (calibrate, register, stack).

Used by:
  - T02 (build_masters) internally for calibration frames
  - The pipeline for light frames before T03

Siril command (verified against Siril 1.4 docs):
    convert basename [-debayer] [-fitseq] [-ser] [-start=index] [-out=]
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


class ConvertSequenceInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the directory where the .seq and FITSEQ will be written."
    )
    input_files: list[str] = Field(
        description="Absolute paths to the raw frames to convert."
    )
    sequence_name: str = Field(
        description=(
            "Base name for the output sequence (without .seq extension). "
            "E.g. 'lights_seq', 'bias_seq', 'flat_seq'."
        )
    )
    debayer: bool = Field(
        default=False,
        description="Debayer CFA frames during conversion. Usually False for calibration frames.",
    )


def _convert_to_sequence(
    working_dir: str,
    input_files: list[str],
    sequence_name: str,
    debayer: bool = False,
) -> dict:
    """
    Copy input files to a temp directory with sequential naming and run
    Siril convert to produce a FITSEQ sequence in working_dir.
    """
    wdir = Path(working_dir)
    wdir.mkdir(parents=True, exist_ok=True)

    if not input_files:
        raise ValueError("input_files is empty — nothing to convert.")

    with tempfile.TemporaryDirectory(dir=wdir, prefix=f"{sequence_name}_raw_") as tmpdir:
        tmp = Path(tmpdir)
        for i, src in enumerate(sorted(input_files)):
            suffix = Path(src).suffix
            dst = tmp / f"{sequence_name}_{i:04d}{suffix}"
            shutil.copy2(src, dst)

        debayer_flag = " -debayer" if debayer else ""
        run_siril_script(
            [f"convert {sequence_name} -fitseq{debayer_flag}"],
            working_dir=str(tmp),
            timeout=600,
        )

        # For FITSEQ conversion, run in tmp (no -out) so Siril reliably
        # writes both {sequence}.fit and {sequence}.seq together, then move
        # outputs into the target working directory before tmp cleanup.
        for ext in (".seq", ".fit", ".fits"):
            src_file = tmp / f"{sequence_name}{ext}"
            dst_file = wdir / f"{sequence_name}{ext}"
            if src_file.exists() and not dst_file.exists():
                shutil.move(str(src_file), str(dst_file))

    seq_path = wdir / f"{sequence_name}.seq"
    fitseq_path = wdir / f"{sequence_name}.fit"
    if not fitseq_path.exists():
        fitseq_path = wdir / f"{sequence_name}.fits"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Siril convert did not produce {seq_path}. "
            f"Check that input files are valid image formats."
        )

    return {
        "sequence_name": sequence_name,
        "sequence_path": str(seq_path),
        "fitseq_path": str(fitseq_path) if fitseq_path.exists() else None,
        "frame_count": len(input_files),
    }


@tool(args_schema=ConvertSequenceInput)
def convert_sequence(
    working_dir: str,
    input_files: list[str],
    sequence_name: str,
    debayer: bool = False,
) -> dict:
    """
    Convert raw frames into a Siril FITSEQ sequence.

    Copies input files to a temporary directory with clean sequential naming,
    runs Siril convert, and produces a .seq + .fit FITSEQ in working_dir.

    Use before T03 (calibrate) for lights, or internally by T02 for calibration
    frames. The sequence_name must be unique within working_dir.
    """
    return _convert_to_sequence(working_dir, input_files, sequence_name, debayer)
