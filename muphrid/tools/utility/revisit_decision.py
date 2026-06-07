"""
revisit_decision

Reopen a converged processing step. This is the loop-back primitive that makes
a decision "converged, not irrevocable": when later work reveals that an
earlier choice was wrong, the agent — or, in HITL mode, the human via the
agent — returns to that step with its alternatives intact instead of starting
from scratch.

It is mode-independent. In autonomous mode the agent calls it to back out and
retry, exactly as a coding agent reverts a bad change and tries again; in HITL
mode it additionally reopens the review gate so the human re-enters the
discussion. HITL adds a voice, not the capability.

Mechanics (append-only — the decision record is NOT deleted; revisiting starts
a fresh iteration seeded from it):
  - paths.current_image + metadata.image_space restored to the step's pre-step
    image (what its candidates branched from), when recorded;
  - variant_pool repopulated from the decision's recorded candidates (dangling
    files dropped);
  - the analysis baseline / regression warnings cleared, so the next
    analyze_image establishes a fresh baseline;
  - the review session reopened iff that tool's HITL gate is enabled.

Handles come from metadata.decisions, surfaced in the system prompt. The agent
never supplies a path.
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

from muphrid.graph import review as review_ctl
from muphrid.graph.state import AstroState


class RevisitDecisionInput(BaseModel):
    decision: str = Field(
        description=(
            "Which converged step to reopen. Use a handle from the revisitable "
            "decisions surfaced in your context — either the decision id "
            "('<phase>:<tool>', e.g. 'linear:remove_gradient') or just the tool "
            "name ('remove_gradient'), resolved within the current phase. "
            "Unknown handles are rejected with the available ones listed."
        ),
    )


@tool(args_schema=RevisitDecisionInput)
def revisit_decision(
    decision: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Reopen a converged step with its alternatives, to iterate on it again.

    Restores the step's pre-step image, repopulates the variant pool from the
    decision's recorded candidates, clears the analysis baseline, and (if that
    tool's HITL gate is enabled) reopens the review session. The decision
    record is preserved — this starts a fresh iteration from it, it does not
    undo history. It reopens a converged step so an earlier choice can be
    reconsidered when later work shows it was wrong.
    """
    state = state or {}
    record, available = review_ctl.resolve_decision(state, decision)

    if record is None:
        return Command(update={"messages": [ToolMessage(
            content=json.dumps({
                "status": "error",
                "message": (
                    f"No revisitable decision matches '{decision}'."
                    if available else
                    "No decisions have been recorded yet — there is nothing to "
                    "revisit. Decisions are recorded when a variant is approved "
                    "(HITL) or committed (commit_variant)."
                ),
                "available_decisions": available,
            }, indent=2),
            tool_call_id=tool_call_id,
        )]})

    # The pre-step image is authoritative state: it's what the step's
    # candidates branched from, and restoring it is the whole point of a
    # revisit. Refuse rather than guess if it (or its render space) wasn't
    # recorded — a real decision always has it (auto_checkpoint sets the
    # anchor); its absence means a legacy/broken record.
    pre_step = record.get("pre_step")
    if not (
        isinstance(pre_step, dict)
        and pre_step.get("path")
        and pre_step.get("image_space") in ("linear", "display")
    ):
        return Command(update={"messages": [ToolMessage(
            content=json.dumps({
                "status": "error",
                "decision_id": record.get("decision_id"),
                "message": (
                    "This decision has no valid recorded pre-step image, so it "
                    "cannot be revisited faithfully. Use restore_checkpoint to "
                    "return to a bookmarked image instead."
                ),
            }, indent=2),
            tool_call_id=tool_call_id,
        )]})

    if not Path(pre_step["path"]).exists():
        return Command(update={"messages": [ToolMessage(
            content=json.dumps({
                "status": "error",
                "decision_id": record.get("decision_id"),
                "message": f"Pre-step image no longer exists on disk: {pre_step['path']}",
            }, indent=2),
            tool_call_id=tool_call_id,
        )]})

    # Repopulate the pool from the recorded alternatives, dropping any whose
    # files have since been cleaned up.
    candidates = record.get("candidates", []) or []
    restored = [
        v for v in candidates
        if isinstance(v, dict) and v.get("file_path") and Path(v["file_path"]).exists()
    ]
    dropped = [
        v.get("id") for v in candidates
        if not (isinstance(v, dict) and v.get("file_path") and Path(v["file_path"]).exists())
    ]

    reopened = review_ctl.reopen_review_for_revisit(state, record)
    chosen = record.get("chosen", {}) or {}

    summary = {
        "status": "revisited",
        "decision_id": record.get("decision_id"),
        "tool": record.get("tool_name"),
        "previously_chosen": chosen.get("variant_id"),
        "restored_candidates": [v.get("id") for v in restored],
        "dropped_missing": dropped,
        "gate_reopened": reopened is not None,
        "note": (
            "Restored to before this step ran, with its candidates back in the "
            "pool. The decision record is kept — this is a fresh iteration, not "
            "an undo. Reconsider the options (compare_images), pick a different "
            "one, or run another experiment."
        ),
    }

    # Single update literal: because it advances paths.current_image it MUST
    # re-assert metadata.image_space in the same payload (render-state writer
    # contract — enforced by the registry guard).
    update = {
        "paths": {"current_image": pre_step["path"]},
        "metadata": {
            "image_space": pre_step["image_space"],
            "last_analysis_snapshot": None,
        },
        "variant_pool": restored,
        "regression_warnings": [],
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            tool_call_id=tool_call_id,
        )],
    }
    if reopened is not None:
        update["review_session"] = reopened
        update["active_hitl"] = True

    return Command(update=update)
