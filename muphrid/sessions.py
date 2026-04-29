"""
Session-index helpers shared by the CLI and Gradio surfaces.

Each run produces a thread_id (a UUID-style slug) and a working_dir on disk.
The sessions index at ~/.muphrid/sessions.json maps thread_id → working_dir
so a later resume — from either CLI or Gradio — can recover the run's
location without the user having to remember it.

Both surfaces register sessions when starting a new run, and look them up
when resuming. Keeping the helpers in a shared module avoids the CLI
having to import gradio_app (which would pull in gradio as a dependency).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


_SESSIONS_INDEX = Path.home() / ".muphrid" / "sessions.json"


def register_session(thread_id: str, working_dir: str) -> None:
    """Record a thread_id → working_dir mapping for later resume lookup.

    Idempotent: re-registering the same thread updates the path. The
    index file is created on first call.
    """
    _SESSIONS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    index: dict = {}
    if _SESSIONS_INDEX.exists():
        try:
            index = json.loads(_SESSIONS_INDEX.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    index[thread_id] = working_dir
    _SESSIONS_INDEX.write_text(json.dumps(index, indent=2))


def lookup_session_dir(thread_id: str) -> str | None:
    """Return the working_dir registered for a thread_id, or None."""
    if not _SESSIONS_INDEX.exists():
        return None
    try:
        index = json.loads(_SESSIONS_INDEX.read_text())
        return index.get(thread_id)
    except (json.JSONDecodeError, OSError):
        return None
