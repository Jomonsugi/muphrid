"""
T31 — save_checkpoint / restore_checkpoint

Non-destructive iteration for the NONLINEAR phase. Every tool reads
current_image, processes it, and updates current_image to the output.
When the agent iterates (trying different curves parameters), each call
chains cumulatively — 4 attempts = 4 baked-in passes degrading the image.

Checkpoints let the agent bookmark the current image at any point and
restore to it later. Checkpoint files are the actual FITS already
produced by tools — no file copies, just path bookmarks.
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
            "A short, descriptive name for this checkpoint. Use names that "
            "describe the processing state: 'starless_base', 'after_curves', "
            "'pre_saturation', 'curves_v2_good'."
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
    Bookmark the current image as a named checkpoint.

    Saves a reference to current_image under a descriptive name. Does NOT
    copy the file — the checkpoint is a pointer to the FITS that already
    exists on disk.

    When to checkpoint:
    - After star_removal produces a clean starless image
    - After a curves/saturation/contrast pass you are satisfied with
    - Before any experimental adjustment you might want to undo

    The rule: if you would regret losing the current image state, checkpoint it.

    If a checkpoint with the same name exists, it is overwritten — this lets
    you update a checkpoint after re-doing a step.
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
    Restore the working image to a previously saved checkpoint.

    Sets current_image to the checkpointed path. The next tool call will
    read from that image instead of the most recent output.

    Use this when:
    - A curves/saturation pass degraded the image — go back and retry
    - Multiple chained adjustments went wrong — restore to before the first bad one
    - You want to try a different processing path from a branch point

    After restoring, the previous current_image file still exists on disk.
    If you checkpointed it, you can switch back. If you didn't, the file
    is still there but you need the path from the tool output history.
    """
    checkpoints = state.get("metadata", {}).get("checkpoints") or {}

    if not checkpoints:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    "No checkpoints saved yet. Call save_checkpoint to bookmark "
                    "the current image before attempting to restore."
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

    summary = {
        "restored": name,
        "current_image": Path(restore_path).name,
        "previous_image": Path(state["paths"]["current_image"]).name if state["paths"].get("current_image") else None,
        "all_checkpoints": {k: Path(v).name for k, v in checkpoints.items()},
    }

    return Command(update={
        "paths": {**state["paths"], "current_image": restore_path},
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            tool_call_id=tool_call_id,
        )],
    })
