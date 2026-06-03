"""
Disk-space preconditions for tools that write large intermediate artifacts.

Tools whose Siril calls produce multi-gigabyte FITSEQ outputs
(siril_register, calibrate's debayered output, build_masters' input
sequence conversion, convert_sequence) call require_free_space() before
running. When the working volume cannot hold the expected output, the
helper fires a typed LangGraph interrupt (type="disk_full") that pauses
the run for human action.

Why an interrupt, not an error: only the human can free disk. The agent
has no recourse — retrying with different parameters cannot conjure
storage. So insufficient disk is treated like flag_dataset_issue:
synchronously surface the shortfall to the operator, halt, and resume
when they've made room.

Lifecycle:
  1. Tool computes an estimate of bytes_needed for its output.
  2. Tool calls require_free_space(working_dir, bytes_needed, what).
  3. Helper polls shutil.disk_usage on the working volume.
  4. If insufficient, calls interrupt(type="disk_full", ...).
       - autonomous CLI: prints the shortfall and exits non-zero so the
         operator returns to a clear signal.
       - attended CLI / Gradio: renders the message and waits for the
         human to free disk and resume.
  5. After interrupt() returns, helper re-checks disk. If still
     insufficient, interrupts again. This way "resume without freeing
     anything" cannot silently re-enter a failing Siril call.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langgraph.types import interrupt


def free_bytes(path: str | Path) -> int:
    """Free bytes on the volume hosting `path`."""
    return shutil.disk_usage(str(path)).free


def _human(b: float) -> str:
    """Format a byte count with appropriate IEC prefix."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if b < 1024:
            return f"{b:.1f} {unit}" if isinstance(b, float) else f"{b} {unit}"
        b = b / 1024
    return f"{b:.1f} PiB"


def require_free_space(
    working_dir: str | Path,
    bytes_needed: int,
    what: str,
    *,
    safety_margin: float = 1.1,
) -> None:
    """
    Pause the run for human action when the working volume cannot hold
    the expected intermediate output.

    Args:
        working_dir: any path on the target filesystem; the helper checks
            free space on that filesystem.
        bytes_needed: best estimate of the bytes the next tool step will
            write (Siril intermediate FITSEQ, output FITS, etc.).
        what: short description shown to the operator ("registered FITSEQ
            r_pp_lights_seq.fit", etc.).
        safety_margin: multiplier on bytes_needed to leave headroom for
            FITS header overhead, Siril temp files, and atomic rename
            staging. Default 1.1.

    Behavior: re-checks free space on each call and on resume. If free is
    sufficient, returns immediately. Otherwise calls interrupt() with a
    typed payload and loops on resume until the shortfall is cleared.
    """
    needed = int(bytes_needed * safety_margin)
    wdir = Path(working_dir).resolve()

    while True:
        free = free_bytes(wdir)
        if free >= needed:
            return

        shortfall = needed - free
        payload = {
            "type": "disk_full",
            "title": "Insufficient disk space",
            "what": what,
            "working_dir": str(wdir),
            "needed_bytes": needed,
            "free_bytes": free,
            "shortfall_bytes": shortfall,
            "agent_text": (
                f"Cannot write {what}: the working volume needs "
                f"{_human(needed)} free, but only {_human(free)} is "
                f"available (short by {_human(shortfall)}).\n\n"
                f"Working dir: {wdir}\n\n"
                f"Free disk on this volume and resume. The agent has no "
                f"way to recover from this — only the operator can."
            ),
            "approval_allowed": False,
            "review_state": "ready",
            "images": [],
        }
        # Block until the operator resumes. On resume execution returns
        # here; the outer loop re-checks free space, so a resume without
        # adequate freeing produces another disk_full interrupt rather
        # than a confusing Siril failure downstream.
        interrupt(payload)
