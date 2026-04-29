"""
T39 — present_for_review

The agent's curation gesture for HITL collaboration. Use this when you have
looked at the current variant pool and are ready to deliberately share one
or more candidates with the human for a decision. Variants you pass here
become the presented review candidates: Gradio can show them in the main
HITL review viewer with approval controls, and CLI clients can approve them
by id.

Pool variants you do NOT present remain workbench history. They may still be
visible in the app's pool filmstrip for transparency/debugging, and the
human can ask about them in chat, but they are not approval candidates.

Two tools, two intents:

  * present_for_review — "I'm asking your input on these specific variants
    from the pool." This is the curation signal: ids are type-checked
    against the current pool and become the review candidates.

  * present_images — "Here's contextual imagery to look at." Reference
    images, before/after pairs, cross-phase comparisons. No Approve
    consequence. (See t32_present_images.)

Why a separate tool: collaboration intent should be explicit. Calling this
tool means "I have selected these candidate(s), here is my reasoning, and I
want the human to weigh in." The UI follows from that intent, but the tool
is not UI plumbing; it is the agent sharing its work for review.

This tool does not create images, run metrics, or choose for you. It records
your curation into review_session.proposal and returns a summary of the
current review set. It only works while an active HITL gate is open. If
HITL is disabled or you are making the decision autonomously, use
commit_variant instead of presenting for human review.

Modes:

  * mode="add"  (default) — append the supplied ids to the current review
    set. The agent's typical flow: present 3 variants, user asks for one
    more option, agent runs another tool, calls present_for_review with
    the new id in add mode. The earlier presented candidates stay; the new
    one joins them. Use add only when the earlier options remain fair
    candidates for approval.

  * mode="replace" — replace the entire review set with the supplied ids.
    Use when the agent has fundamentally changed direction, the earlier
    candidates are no longer fair approval options, or you want to narrow the
    collaboration to a recommendation.

Per-call rationale: each call carries the agent's commentary at the
moment of presentation. With mode="add", earlier rationales are preserved
on their respective variants; with mode="replace", everything is reset.
This keeps the agent's evolving reasoning visible per variant in the
review surface.

Lifecycle: review_session.proposal is closed/cleared by advance_phase,
commit_variant, hitl_check on variant approval, and rewind_phase. A new HITL
gate starts empty. If the pool contains variants but no candidates have been
presented, HITL remains non-approvable and the agent is expected to curate:
compare the pool against the step goal, explain the tradeoffs, and call this
tool or run another experiment.

This is a UTILITY tool so it can be called during any active HITL gate, but
it returns a blocked result outside HITL collaboration.
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

from muphrid.graph import review as review_ctl

from muphrid.graph.state import AstroState, ReviewProposalCandidate

logger = logging.getLogger(__name__)


class PresentForReviewInput(BaseModel):
    """Input schema for present_for_review."""

    variant_ids: list[str] = Field(
        description=(
            "Variant ids from state.variant_pool that you are deliberately "
            "presenting as candidates for human review. Format: 'T<NN>_v<n>' "
            "(e.g. 'T14_v1', 'T14_v3'). Each id must exist in the current "
            "pool; unknown ids are rejected with the valid options. Include "
            "only variants that are fair approval candidates. Pool entries "
            "you omit remain observational workbench history."
        ),
    )
    rationale: str = Field(
        description=(
            "Your review note for these candidates: why you selected them, "
            "what tradeoffs matter, and which option you recommend if you "
            "have a preference. Shown alongside each presented variant so "
            "the human can understand your reasoning. With mode='add', this "
            "rationale is attached to the newly-added ids only; earlier ids "
            "keep their prior rationale. With mode='replace', this becomes "
            "the rationale for the new review set. Keep it concise — "
            "1-3 sentences typically."
        ),
    )
    recommendation: str | None = Field(
        default=None,
        description=(
            "Optional single variant id you recommend from variant_ids, if you "
            "have a clear preference. Use None when you are genuinely asking "
            "the human to choose between tradeoffs."
        ),
    )
    tradeoffs: list[str] = Field(
        default_factory=list,
        description=(
            "Short tradeoff bullets to show in the review proposal artifact. "
            "Examples: 'v2 protects highlights better', 'v3 has flatter background'."
        ),
    )
    metric_highlights: dict = Field(
        default_factory=dict,
        description=(
            "Optional compact metric highlights relevant to the recommendation. "
            "Use existing metric names/values when helpful; leave empty otherwise."
        ),
    )
    mode: str = Field(
        default="add",
        description=(
            "How this call composes with prior calls at the same gate.\n"
            "  'add' (default): append the supplied ids to the current "
            "review set. Use when surfacing additional candidates after "
            "user feedback and the existing presented candidates are still "
            "valid approval options.\n"
            "  'replace': clear the prior review set and start fresh "
            "with just these ids. Use when you have changed direction, "
            "narrowed to a recommendation, or earlier candidates are no "
            "longer fair choices."
        ),
    )


@tool(args_schema=PresentForReviewInput)
def present_for_review(
    variant_ids: list[str],
    rationale: str,
    recommendation: str | None = None,
    tradeoffs: list[str] | None = None,
    metric_highlights: dict | None = None,
    mode: str = "add",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Present specific pool variants as candidates for human review.

    This is the agent's collaboration gesture at a HITL gate. From the
    variant pool, pick the result(s) you want the human to consider, explain
    why, and call this tool. Those variants become approval candidates in
    the review surface. Pool entries you do not pick remain observational
    workbench history; the human may ask about them, but they are not
    candidates until you present them.

    CRITICAL: present variants that are fair approval options, not variants
    you are recommending against. If you present a variant you've already
    judged bad, you are asking the human to approve a result you do not
    believe in. That is incoherent collaboration.

    Concretely, if your reasoning is "v1 has artifacts, v2 looks better,
    we should go with v2":
      * RIGHT: present_for_review([T??_v2], rationale="v2 addresses the
        artifacts in v1. The pool has both for comparison.")
      * WRONG: present_for_review([T??_v1], rationale="v1 has artifacts,
        try v2 instead") — this presents the broken result as approvable.

    If you want the human to weigh both in (you genuinely don't know
    which is better), present BOTH and frame it as a choice. Don't
    present one and recommend the other in the rationale; that creates
    the same incoherence at smaller scale.

    Pool ids the human sees in the review surface match the ids you
    reference here (e.g. T14_v1, T14_v3) — same vocabulary across agent,
    system, and chat, so the human can refer back to specific variants by
    name when giving feedback.

    In autonomous mode, do not use this to make your own decision. Use
    commit_variant with your chosen variant id and rationale.
    """
    state = state or {}

    review_session = review_ctl.active_review_session(state)
    if not review_session:
        blocked = {
            "status": "blocked",
            "reason": "no_active_hitl_gate",
            "message": (
                "present_for_review is only available while a HITL review gate "
                "is open. Outside HITL, use present_images if you want to show "
                "visual context, use commit_variant for an autonomous variant "
                "decision, or continue to the next tool/phase."
            ),
        }
        logger.warning("present_for_review blocked: no active HITL gate")
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps(blocked, indent=2),
                name="present_for_review",
                tool_call_id=tool_call_id,
            )],
        })

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
                "The variant pool is currently empty — no variant-producing "
                "tool has produced a captured output since the pool was "
                "cleared. Run the tool that should produce a candidate first, "
                "then call present_for_review with the resulting id."
            )
        logger.warning(
            f"present_for_review: invalid ids {invalid} (valid: {valid_ids})"
        )
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps(error, indent=2),
                name="present_for_review",
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
                name="present_for_review",
                tool_call_id=tool_call_id,
            )],
        })

    # Build new entries for the supplied ids.
    now_iso = datetime.now(timezone.utc).isoformat()
    new_entries: list[ReviewProposalCandidate] = [
        ReviewProposalCandidate(
            variant_id=vid,
            rationale=rationale,
            presented_at=now_iso,
        )
        for vid in variant_ids
    ]

    current_artifact = (review_session or {}).get("proposal", {}) or {}
    current_candidates = [
        e for e in (current_artifact.get("candidates", []) or [])
        if isinstance(e, dict) and e.get("variant_id") in pool_by_id
    ]

    if mode == "add":
        # Dedupe against current proposal candidates by variant_id. Re-presenting
        # an id updates its rationale instead of creating duplicates.
        already_present = {e.get("variant_id") for e in current_candidates}
        ids_being_added = {e["variant_id"] for e in new_entries}
        if already_present & ids_being_added:
            kept = [
                e for e in current_candidates
                if e.get("variant_id") not in ids_being_added
            ]
            review_set_after = kept + new_entries
            action_summary = "updated rationale and/or appended"
        else:
            review_set_after = current_candidates + new_entries
            action_summary = "appended"
    else:  # replace
        review_set_after = new_entries
        action_summary = "replaced review set with"

    review_ids = {e.get("variant_id") for e in review_set_after}
    if recommendation and recommendation not in review_ids:
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps({
                    "status": "error",
                    "message": (
                        f"present_for_review: recommendation {recommendation!r} "
                        "is not in the resulting review_session.proposal "
                        f"candidate set: {sorted(review_ids)}."
                    ),
                }, indent=2),
                name="present_for_review",
                tool_call_id=tool_call_id,
            )],
        })

    proposal_artifact = review_ctl.proposal_from_candidates(
        review_set_after,
        recommendation=recommendation,
        rationale=rationale,
        tradeoffs=tradeoffs or [],
        metric_highlights=metric_highlights or {},
    )
    updated_review_session = review_ctl.update_review_session(
        review_session,
        status="awaiting_human_approval",
        proposal=proposal_artifact,
        visible_response_required=False,
    )

    summary = {
        "status": "presented",
        "action": action_summary,
        "added_or_updated": variant_ids,
        "rationale_for_this_call": rationale,
        "recommendation": recommendation,
        "tradeoffs": tradeoffs or [],
        "metric_highlights": metric_highlights or {},
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
        "review_session": updated_review_session,
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2),
            name="present_for_review",
            tool_call_id=tool_call_id,
        )],
    })
