"""
T39 — present_for_review

The agent's curation gesture for HITL gates. Pool variants the agent passes
in here become the active "approve set" — the human sees them in the
proposal panel with Approve buttons. Pool variants the agent does NOT
present remain in the pool as read-only history (status of "what I've
been working on") with no Approve affordance.

Two tools, two intents:

  * present_for_review — "I'm asking your input on these specific variants
    from the pool." Approve buttons appear. Variant ids only; type-checked
    against the current pool. This is the curation signal.

  * present_images — "Here's contextual imagery to look at." Reference
    images, before/after pairs, cross-phase comparisons. No Approve
    consequence. (See t32_present_images.)

Why a separate tool: the agent shouldn't have to think about UI mechanics
("does this trigger an approve button?"). The choice of tool encodes the
intent. `present_for_review` is the verb that means "put these up for
human approval"; everything UI-side flows from that.

Modes:

  * mode="add"  (default) — append the supplied ids to the current review
    set. The agent's typical flow: present 3 variants, user asks for one
    more option, agent runs another tool, calls present_for_review with
    the new id in add mode. The earlier 3 stay; the new one joins them.
    Rejected variants visibly persist as comparison reference until the
    human approves a winner.

  * mode="replace" — replace the entire review set with the supplied ids.
    Use when the agent has fundamentally changed direction and earlier
    candidates are no longer relevant.

Per-call rationale: each call carries the agent's commentary at the
moment of presentation. With mode="add", earlier rationales are preserved
on their respective variants; with mode="replace", everything is reset.
This keeps the agent's evolving reasoning visible per variant in the
proposal panel.

Lifecycle: state.presented_for_review is cleared by advance_phase,
commit_variant, hitl_check on variant approval, and rewind_phase. A new
HITL gate starts empty — the agent must consciously surface candidates
each time. This is intentional: the empty state is a real signal that
the agent owes the human a curation decision.

This is a UTILITY tool — available in every phase.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState, PresentedVariant, Replace, _is_replace

logger = logging.getLogger(__name__)


class PresentForReviewInput(BaseModel):
    """Input schema for present_for_review."""

    variant_ids: list[str] = Field(
        description=(
            "List of variant ids from state.variant_pool to surface for human "
            "approval at the current HITL gate. Format: 'T<NN>_v<n>' (e.g. "
            "'T14_v1', 'T14_v3'). Each id must exist in the current pool — "
            "the tool validates and rejects unknown ids with the list of "
            "valid options. Pool entries you don't include here remain "
            "visible as read-only history but do not get Approve buttons."
        ),
    )
    rationale: str = Field(
        description=(
            "Your commentary on these variants — why you're surfacing them, "
            "what you'd like the human's input on, what tradeoffs you see. "
            "Shown alongside each presented variant in the proposal panel "
            "so the human can read your reasoning at a glance. With "
            "mode='add', this rationale is attached to the newly-added "
            "ids only; earlier ids keep the rationale they were "
            "introduced with. With mode='replace', this is the only "
            "rationale shown. Keep it concise — 1-3 sentences typically."
        ),
    )
    mode: str = Field(
        default="add",
        description=(
            "How this call composes with prior calls at the same gate.\n"
            "  'add' (default): append the supplied ids to the current "
            "review set. Use when surfacing additional options after "
            "user feedback ('here's another variant addressing the "
            "highlight clipping you mentioned').\n"
            "  'replace': clear the prior review set and start fresh "
            "with just these ids. Use when the agent has fundamentally "
            "changed direction and earlier candidates are no longer "
            "relevant."
        ),
    )


@tool(args_schema=PresentForReviewInput)
def present_for_review(
    variant_ids: list[str],
    rationale: str,
    mode: str = "add",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Surface specific pool variants for human approval at the current
    HITL gate.

    The agent's curation gesture: from the pool of variants accumulated
    by HITL-mapped tool execution, pick the ones you want the human to
    weigh in on. Those become Approve-able in the proposal panel; pool
    entries you don't pick remain visible as read-only history.

    CRITICAL: surface the variants you want APPROVED, not the ones you
    are recommending against. The proposal panel shows Approve buttons
    next to every variant you list here — if you put a variant you've
    decided against in the proposal, you are inviting the human to
    approve a result you've already labelled bad. That is incoherent
    collaboration.

    Concretely, if your reasoning is "v1 has artifacts, v2 looks better,
    we should go with v2":
      * RIGHT: present_for_review([T??_v2], rationale="v2 addresses the
        artifacts in v1. The pool has both for comparison.")
      * WRONG: present_for_review([T??_v1], rationale="v1 has artifacts,
        try v2 instead") — this offers Approve on the broken result.

    If you want the human to weigh both in (you genuinely don't know
    which is better), present BOTH and frame it as a choice. Don't
    present one and recommend the other in the rationale; that creates
    the same incoherence at smaller scale.

    Pool ids the human sees in the panel match the ids you reference here
    (e.g. T14_v1, T14_v3) — same vocabulary across agent, system, and
    chat, so the human can refer back to specific variants by name when
    giving feedback.
    """
    state = state or {}
    pool = list(state.get("variant_pool", []) or [])
    pool_by_id = {v.get("id"): v for v in pool if v.get("id")}

    # Validate every requested id against the current pool. Surface ALL
    # invalid ids in one error message so the agent doesn't have to retry
    # iteratively to discover them.
    invalid = [vid for vid in variant_ids if vid not in pool_by_id]
    if invalid:
        valid_ids = sorted(pool_by_id.keys())
        error = {
            "status": "error",
            "message": (
                f"present_for_review: variant id(s) {invalid!r} not found in "
                f"the current variant_pool. Valid ids: "
                f"{valid_ids if valid_ids else '(pool is empty)'}."
            ),
            "invalid_ids": invalid,
            "valid_ids": valid_ids,
        }
        if not pool:
            error["hint"] = (
                "The variant pool is currently empty — no HITL-mapped tool "
                "has produced a variant since the last gate cleared. Run "
                "the tool that should produce a variant first, then call "
                "present_for_review with the resulting id."
            )
        logger.warning(
            f"present_for_review: invalid ids {invalid} (valid: {valid_ids})"
        )
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps(error, indent=2),
                tool_call_id=tool_call_id,
            )],
        })

    if mode not in ("add", "replace"):
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps({
                    "status": "error",
                    "message": (
                        f"present_for_review: invalid mode {mode!r}. "
                        f"Valid modes: 'add', 'replace'."
                    ),
                }, indent=2),
                tool_call_id=tool_call_id,
            )],
        })

    # Build new entries for the supplied ids.
    now_iso = datetime.now(timezone.utc).isoformat()
    new_entries: list[PresentedVariant] = [
        PresentedVariant(
            variant_id=vid,
            rationale=rationale,
            presented_at=now_iso,
        )
        for vid in variant_ids
    ]

    # Build the state update payload. With the _list_extend_or_replace
    # reducer:
    #   - add mode → return the delta list; reducer extends.
    #   - replace mode → wrap in Replace(); reducer fully replaces.
    if mode == "add":
        # Dedupe against current presented_for_review by variant_id —
        # if the agent re-presents an already-presented id, we update
        # its rationale by removing the prior entry and appending fresh.
        # Without this, an already-presented variant would get a second
        # entry with a new rationale and the panel would show duplicates.
        current = list(state.get("presented_for_review", []) or [])
        already_present = {e.get("variant_id") for e in current}
        ids_being_added = {e["variant_id"] for e in new_entries}
        if already_present & ids_being_added:
            # Mixed case: some new, some already present. We need to
            # rebuild the whole list — Replace path with the merged
            # result, since extend would leave stale entries in place.
            kept = [
                e for e in current
                if e.get("variant_id") not in ids_being_added
            ]
            merged = kept + new_entries
            update_value = Replace(merged)
            action_summary = "updated rationale and/or appended"
        else:
            # Pure add — emit just the delta and let the reducer extend.
            update_value = new_entries
            action_summary = "appended"
    else:  # replace
        update_value = Replace(new_entries)
        action_summary = "replaced review set with"

    # Compose summary for the agent's next-turn context. List the variants
    # currently up for review (full set), so the agent can see what the
    # human will be looking at without re-deriving from messages.
    if mode == "replace":
        review_set_after = new_entries
    elif _is_replace(update_value):
        review_set_after = update_value.get("value", [])
    else:
        review_set_after = list(state.get("presented_for_review", []) or []) + new_entries

    summary = {
        "status": "presented",
        "action": action_summary,
        "added_or_updated": variant_ids,
        "rationale_for_this_call": rationale,
        "mode": mode,
        "review_set_now": [
            {
                "variant_id": e["variant_id"],
                "rationale": e["rationale"],
            }
            for e in review_set_after
        ],
        "review_set_size": len(review_set_after),
    }

    logger.info(
        f"present_for_review: {action_summary} {variant_ids} "
        f"({len(review_set_after)} now in review set)"
    )

    return Command(update={
        "presented_for_review": update_value,
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            tool_call_id=tool_call_id,
        )],
    })
