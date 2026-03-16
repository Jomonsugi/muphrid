"""
T02b — convert_sequence

Convert all light frames into a Siril FITSEQ sequence. This is the
prerequisite for T03 (calibrate). Light files come from state["dataset"]["files"]["lights"].

Used by:
  - T02 (build_masters) internally for calibration frames via _convert_to_sequence
  - The agent for light frames before T03

Siril command (verified against Siril 1.4 docs):
    convert basename [-debayer] [-fitseq] [-ser] [-start=index] [-out=]
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from astro_agent.graph.state import AstroState
from astro_agent.tools._siril import run_siril_script


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


@tool
def convert_sequence(
    sequence_name: str,
    debayer: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Convert all light frames into a Siril FITSEQ sequence.

    Copies light files from the dataset into a temporary directory with clean
    sequential naming, runs Siril convert, and produces a .seq + .fit FITSEQ
    in the dataset working directory. Must be called before T03 (calibrate).

    Args:
        sequence_name: Base name for the output sequence (no .seq extension).
            E.g. 'lights_seq'. Must be unique within the working directory.
        debayer: Debayer CFA frames during conversion. Set True for OSC/DSLR
            RAW files when converting for preview; False for calibration use
            (calibrate does its own debayering internally). Default: False.
    """
    working_dir = state["dataset"]["working_dir"]
    input_files = state["dataset"]["files"]["lights"]

    result = _convert_to_sequence(working_dir, input_files, sequence_name, debayer)

    return Command(update={
        "paths": {**state["paths"], "lights_sequence": result["sequence_name"]},
        "messages": [ToolMessage(content=json.dumps(result, indent=2, default=str), tool_call_id=tool_call_id)],
    })
