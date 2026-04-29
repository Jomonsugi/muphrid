"""
T31 — save_checkpoint / restore_checkpoint

Within-phase image-state bookmarks. Image-modifying tools read
current_image, write a new FITS, and promote the new path to
current_image. Calling the same tool repeatedly therefore chains
cumulatively — each call processes the previous output, not the
original input.

The graph automatically records "auto:*" checkpoints before post-stack
image-modifying tools. save_checkpoint is the manual companion for
deliberate named bookmarks: it records a name → current_image path
mapping in metadata.checkpoints. The checkpoint is a pointer to a FITS
that already exists on disk; no copy is made.

restore_checkpoint sets paths.current_image to the recorded path,
making the next image-modifying tool read from the bookmarked file.
The displaced FITS remains on disk and is still recoverable from
prior tool output messages.

Available in post-stack image-processing phases where a human would
reasonably want undo/bookmark behavior: linear, stretch, and nonlinear.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field


class SaveCheckpointInput(BaseModel):
    name: str = Field(
        description=(
            "Identifier for this checkpoint. Used as the lookup key in "
            "restore_checkpoint(name=...). If the name already exists in "
            "metadata.checkpoints, the prior entry is overwritten."
        ),
    )


class RestoreCheckpointInput(BaseModel):
    name: str = Field(
        description=(
            "The name of the checkpoint to restore. Must match a name "
            "previously passed to save_checkpoint."
        ),
    )


@tool(args_schema=SaveCheckpointInput)
def save_checkpoint(
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Record current_image under a deliberate name in metadata.checkpoints.

    The checkpoint stores the FITS path, not the file contents — no copy
    is made. Restoring this checkpoint later sets current_image back to
    the recorded path.

    You do not need to call this before routine image-modifying tools; the
    graph automatically creates "auto:*" checkpoints for normal undo.

    Returns the saved name and the full set of currently-bookmarked
    checkpoints. Re-using an existing name overwrites the prior entry.
    """
    current_image = state["paths"].get("current_image")

    if not current_image:
        return Command(update={
            "messages": [ToolMessage(
                content="Cannot save checkpoint: no current_image is set.",
                tool_call_id=tool_call_id,
            )],
        })

    if not Path(current_image).exists():
        return Command(update={
            "messages": [ToolMessage(
                content=f"Cannot save checkpoint: file does not exist: {current_image}",
                tool_call_id=tool_call_id,
            )],
        })

    existing = state.get("metadata", {}).get("checkpoints") or {}
    updated = {**existing, name: current_image}

    summary = {
        "saved": name,
        "image": Path(current_image).name,
        "overwritten": name in existing,
        "all_checkpoints": {k: Path(v).name for k, v in updated.items()},
    }

    return Command(update={
        "metadata": {"checkpoints": updated},
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            tool_call_id=tool_call_id,
        )],
    })


@tool(args_schema=RestoreCheckpointInput)
def restore_checkpoint(
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Set current_image to the FITS path stored under `name` in
    metadata.checkpoints.

    The next image-modifying tool will read from the restored path. The
    FITS that was current at restore time is unaffected — it remains on
    disk and recoverable through prior tool output messages or by saving
    a checkpoint pointing at it before this call.

    Side effects on a real (non-noop) restore: any outstanding regression
    warnings are cleared and metadata.last_analysis_snapshot is reset, so
    the next analyze_image call establishes a fresh baseline against the
    restored state. The noop branch (checkpoint and current_image already
    point at the same file) leaves state unchanged.
    """
    checkpoints = state.get("metadata", {}).get("checkpoints") or {}

    if not checkpoints:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    "No checkpoints are available yet. The graph creates "
                    "automatic checkpoints before post-stack image-modifying "
                    "tools, and save_checkpoint can create deliberate named "
                    "bookmarks."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if name not in checkpoints:
        available = ", ".join(f"'{k}'" for k in checkpoints)
        return Command(update={
            "messages": [ToolMessage(
                content=f"Checkpoint '{name}' does not exist. Available: {available}",
                tool_call_id=tool_call_id,
            )],
        })

    restore_path = checkpoints[name]
    if not Path(restore_path).exists():
        return Command(update={
            "messages": [ToolMessage(
                content=f"Checkpoint '{name}' file no longer exists: {restore_path}",
                tool_call_id=tool_call_id,
            )],
        })

    # `noop` tells the agent whether this restore actually changed state. The
    # common failure mode it detects: a checkpoint was saved at a moment when
    # current_image was stale (e.g. a sibling-writing tool did not promote its
    # output), so the bookmark and the live current_image already point at the
    # same file. Restoring to it is a no-op — any tool call after this restore
    # will produce the same output as before, and the agent will keep looping
    # unless it sees this signal and branches (fresh checkpoint from the
    # correct starting path, or a different tool).
    prev_current = state["paths"].get("current_image")
    try:
        noop = (
            prev_current is not None
            and Path(prev_current).resolve() == Path(restore_path).resolve()
        )
    except OSError:
        noop = prev_current == restore_path

    summary = {
        "restored": name,
        "current_image": Path(restore_path).name,
        "previous_image": Path(prev_current).name if prev_current else None,
        "noop": noop,
        "all_checkpoints": {k: Path(v).name for k, v in checkpoints.items()},
    }
    if noop:
        summary["note"] = (
            "Restore was a no-op — the checkpoint and the live current_image "
            "already resolved to the same file, so no state changed. This "
            "happens when a checkpoint was saved against a current_image that "
            "did not advance (for example, a sibling-writing tool whose output "
            "was not promoted to current_image). Re-running the same restore "
            "will produce the same no-op result."
        )

    # On a real restore (not the noop branch), any outstanding regression
    # warnings and the stored analysis snapshot are about the state that
    # was just replaced — they no longer describe the live current_image.
    # Clear them so the next analyze_image establishes a fresh baseline
    # against the restored state. The noop case leaves everything as-is
    # (no state actually changed).
    update: dict = {
        "paths": {"current_image": restore_path},
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            tool_call_id=tool_call_id,
        )],
    }
    if not noop:
        update["regression_warnings"] = []
        update["metadata"] = {"last_analysis_snapshot": None}

    return Command(update=update)
