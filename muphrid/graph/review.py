"""Explicit HITL Review Mode controller helpers.

This module keeps review policy as data and pure-ish transitions rather than
burying it in message-history scans inside graph nodes. The canonical
collaboration contract is `state.review_session`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from muphrid.graph.content import text_content
from muphrid.graph.hitl import is_enabled, tool_cfg
from muphrid.graph.state import (
    AstroState,
    ProcessingPhase,
    ReviewHumanEvent,
    ReviewProposal,
    ReviewProposalCandidate,
    ReviewSession,
    Variant,
)


OPEN_REVIEW_STATUSES = {
    "review_open",
    "awaiting_agent_response",
    "awaiting_curation",
    "awaiting_human_approval",
}


def utc_now() -> str:
    """Return a checkpoint-serializable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def phase_value(phase: ProcessingPhase | str | None) -> str:
    """Normalize a phase enum/string to its state representation."""
    return phase.value if hasattr(phase, "value") else str(phase or "")


def review_is_open(review_session: dict | None) -> bool:
    """True when a review_session represents an unresolved human gate."""
    if not isinstance(review_session, dict):
        return False
    return review_session.get("status") in OPEN_REVIEW_STATUSES


def active_review_session(state: AstroState) -> ReviewSession | None:
    """Return the open review session, if one exists."""
    session = state.get("review_session")
    if review_is_open(session):
        return session
    return None


def active_review_tool(state: AstroState) -> tuple[str | None, str | None]:
    """Return (hitl_key, tool_name) for the current explicit review."""
    session = active_review_session(state)
    if session:
        return session.get("hitl_key"), session.get("tool_name")
    return None, None


def active_review_blocks_autonomy(state: AstroState) -> bool:
    """True when agent-only commit/advance actions should be blocked."""
    hitl_key, _tool_name = active_review_tool(state)
    return bool(hitl_key and is_enabled(hitl_key))


def make_review_session(
    *,
    state: AstroState,
    hitl_key: str,
    tool_name: str,
    status: str = "awaiting_agent_response",
) -> ReviewSession:
    """Create a new explicit review session for an opened HITL gate."""
    phase = phase_value(state.get("phase", ProcessingPhase.INGEST))
    cfg = tool_cfg(hitl_key)
    now = utc_now()
    return ReviewSession(
        gate_id=f"{phase}:{hitl_key}:{tool_name}",
        hitl_key=hitl_key,
        tool_name=tool_name,
        phase=phase,
        title=str(cfg.get("title", tool_name)),
        status=status,
        opened_at=now,
        updated_at=now,
        last_human_event=None,
        turn_policy="answer_visible_text_before_action",
        tool_runs_since_human=0,
        visible_response_required=False,
        proposal=empty_proposal(now),
    )


def update_review_session(
    review_session: dict | None,
    **changes: Any,
) -> ReviewSession | None:
    """Return a copied review_session with shallow changes applied."""
    if not isinstance(review_session, dict):
        return None
    updated = dict(review_session)
    updated.update(changes)
    updated["updated_at"] = utc_now()
    return updated


def close_review_session(
    review_session: dict | None,
    *,
    reason: str,
) -> ReviewSession | None:
    """Return a copied review_session marked closed."""
    if not isinstance(review_session, dict):
        return None
    now = utc_now()
    updated = dict(review_session)
    updated.update({
        "status": "closed",
        "updated_at": now,
        "closed_at": now,
        "close_reason": reason,
    })
    return updated


def empty_proposal(now: str | None = None) -> ReviewProposal:
    """Empty proposal artifact."""
    return ReviewProposal(
        candidates=[],
        recommendation=None,
        rationale="",
        tradeoffs=[],
        metric_highlights={},
        updated_at=now or utc_now(),
    )


def proposal_from_candidates(
    candidates_in: list[ReviewProposalCandidate] | list[dict],
    *,
    recommendation: str | None = None,
    rationale: str = "",
    tradeoffs: list[str] | None = None,
    metric_highlights: dict | None = None,
) -> ReviewProposal:
    """Build a canonical proposal artifact from explicit candidate entries."""
    candidates: list[ReviewProposalCandidate] = []
    for entry in candidates_in or []:
        if not isinstance(entry, dict):
            continue
        vid = entry.get("variant_id")
        if not vid:
            continue
        candidates.append(ReviewProposalCandidate(
            variant_id=str(vid),
            rationale=str(entry.get("rationale", "") or ""),
            presented_at=str(entry.get("presented_at", "") or ""),
        ))
    combined_rationale = rationale.strip()
    if not combined_rationale:
        combined_rationale = "\n".join(
            c["rationale"] for c in candidates if c.get("rationale")
        )
    return ReviewProposal(
        candidates=candidates,
        recommendation=recommendation,
        rationale=combined_rationale,
        tradeoffs=list(tradeoffs or []),
        metric_highlights=dict(metric_highlights or {}),
        updated_at=utc_now(),
    )


def proposal_entries_from_session(
    review_session: dict | None,
    pool: list[Variant],
) -> list[dict]:
    """Resolve review_session.proposal candidates into UI/payload entries."""
    if not review_is_open(review_session):
        return []
    artifact = (review_session or {}).get("proposal", {}) or {}
    candidates = artifact.get("candidates", []) or []
    pool_by_id = {v.get("id"): v for v in pool if isinstance(v, dict) and v.get("id")}
    proposal: list[dict] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        vid = candidate.get("variant_id")
        variant = pool_by_id.get(vid)
        if variant is None:
            continue
        proposal.append({
            "variant": variant,
            "rationale": candidate.get("rationale", ""),
            "presented_at": candidate.get("presented_at", ""),
            "recommendation": artifact.get("recommendation"),
            "tradeoffs": artifact.get("tradeoffs", []),
            "metric_highlights": artifact.get("metric_highlights", {}),
            "proposal_rationale": artifact.get("rationale", ""),
        })
    return proposal


def proposal_candidate_ids(review_session: dict | None) -> set[str]:
    """Return approvable variant ids from review_session.proposal."""
    if not review_is_open(review_session):
        return set()
    artifact = (review_session or {}).get("proposal", {}) or {}
    candidates = artifact.get("candidates", []) or []
    ids: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("variant_id"):
            ids.add(str(candidate["variant_id"]))
    return ids


def review_readiness(
    pool: list[Variant],
    proposal: list[dict],
) -> tuple[bool, str, bool]:
    """Return (needs_curation, review_state, approval_allowed)."""
    needs_curation = bool(pool) and not bool(proposal)
    return needs_curation, ("needs_curation" if needs_curation else "ready"), not needs_curation


def should_route_for_curation(
    *,
    review_session: dict | None,
    pool: list[Variant],
    proposal: list[dict],
    force_interrupt: bool,
) -> bool:
    """True when the controller should ask the agent to curate before pausing."""
    needs_curation, _state, _approval = review_readiness(pool, proposal)
    return (
        needs_curation
        and not force_interrupt
        and (review_session or {}).get("status") != "awaiting_curation"
    )


def curation_update(
    *,
    state: AstroState,
    tool_name: str | None,
    pool: list[Variant],
) -> dict:
    """State update that routes back to the agent for candidate curation."""
    return {
        "active_hitl": True,
        "review_session": update_review_session(
            state.get("review_session"),
            status="awaiting_curation",
        ),
        "messages": [build_curation_prompt(tool_name, pool)],
    }


def feedback_update(
    *,
    state: AstroState,
    event: ReviewHumanEvent,
    tool_name: str | None,
    pool: list[Variant],
    proposal: list[dict],
    prefix_messages: list | None = None,
) -> dict:
    """State update for a non-approval human event during review."""
    prefix_messages = list(prefix_messages or [])
    return {
        "messages": prefix_messages + [
            build_human_feedback_message(event, tool_name, pool, proposal)
        ],
        "active_hitl": True,
        "review_session": update_review_session(
            state.get("review_session"),
            status="awaiting_agent_response",
            last_human_event=event,
            tool_runs_since_human=0,
            visible_response_required=True,
        ),
        "user_feedback": {
            **(state.get("user_feedback", {}) or {}),
            "last_hitl_event": event,
        },
    }


def build_interrupt_payload(
    *,
    cfg: dict,
    tool_name: str,
    image_paths: list[str],
    messages: list,
    agent_text: str,
    pool: list[Variant],
    proposal: list[dict],
    review_session: dict | None,
) -> dict:
    """Build the presenter payload for a Review Mode pause."""
    needs_curation, review_state, approval_allowed = review_readiness(pool, proposal)
    return {
        "type": cfg["type"],
        "title": cfg["title"],
        "tool_name": tool_name,
        "images": image_paths,
        "context": messages[-6:],
        "agent_text": agent_text,
        "variant_pool": pool,
        "proposal": proposal,
        "review_session": update_review_session(
            review_session,
            status="awaiting_human_approval" if proposal else "awaiting_curation",
            visible_response_required=False,
        ),
        "review_state": review_state,
        "approval_allowed": approval_allowed,
    }


def extract_latest_agent_text(messages: list) -> str:
    """Return the latest assistant text content from graph messages."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return text_content(msg.content)
    return ""


def build_open_review_prompt(tool_name: str) -> HumanMessage:
    """Prompt injected after the first HITL tool result opens review mode."""
    return HumanMessage(
        content=(
            "HITL REVIEW OPEN\n"
            f"Tool: {tool_name}\n\n"
            "A human is now reviewing your work on this step.\n\n"
            "Analyze the result — what do the metrics show? What changed?\n"
            "Use the variant pool as your workbench. If the result is worth "
            "human review, call present_for_review with the candidate id(s) "
            "you are intentionally presenting. Explain what worked, what "
            "trade-offs you see, and which option you recommend.\n"
            "Call present_images only for extra visual context; it does not "
            "make a variant approvable.\n"
            "The human will give feedback or approve.\n\n"
            "If they give feedback, interpret it in terms of tool parameters, "
            "re-run the tool, and present the updated result with a comparison "
            "of what changed."
        ),
        additional_kwargs={
            "is_hitl_prompt": True,
            "is_hitl_open": True,
        },
    )


def build_curation_prompt(tool_name: str | None, pool: list[Variant]) -> HumanMessage:
    """Instruction asking the agent to curate the current pool."""
    ids = [v.get("id", "?") for v in pool if isinstance(v, dict)]
    id_list = ", ".join(ids) if ids else "(pool is empty)"
    tool_label = tool_name or "this step"
    return HumanMessage(
        content=(
            f"HITL collaboration is open for {tool_label}, and the variant "
            f"pool now contains: {id_list}.\n\n"
            "Before the human can approve anything, select the candidate(s) "
            "you want to share for review. Compare the options against the "
            "goal of this step, explain the tradeoffs, and call "
            "present_for_review with the variant id(s) you are intentionally "
            "presenting. If none of the variants are good enough, say why "
            "and run another experiment instead of presenting a weak option."
        ),
        additional_kwargs={
            "is_hitl_prompt": True,
            "is_hitl_curation_prompt": True,
        },
    )


def parse_human_event(response: Any) -> ReviewHumanEvent:
    """
    Normalize interrupt resume values into typed human events.

    Review Mode clients send dictionaries. Non-dict values are treated as
    plain feedback text; approval is never inferred from sentinel strings.
    """
    now = utc_now()
    if isinstance(response, dict):
        event_type = str(response.get("type", "") or "feedback")
        text = str(response.get("text", "") or "")
        event = ReviewHumanEvent(
            type=event_type,  # type: ignore[typeddict-item]
            text=text,
            received_at=now,
        )
        if response.get("variant_id"):
            event["variant_id"] = str(response["variant_id"])
        if response.get("rationale"):
            event["rationale"] = str(response["rationale"])
        return event

    text = str(response)
    return ReviewHumanEvent(
        type="feedback",
        text=text,
        received_at=now,
    )


def is_approval_event(event: ReviewHumanEvent) -> bool:
    """True when a human event should close or attempt to close review."""
    return event.get("type") in ("approve_variant", "approve_current")


def build_human_feedback_message(
    event: ReviewHumanEvent,
    tool_name: str | None,
    pool: list[Variant],
    proposal: list[dict],
) -> HumanMessage:
    """Build model-visible HITL feedback context for the next agent turn."""
    presented = []
    for entry in proposal:
        variant = entry.get("variant", {}) if isinstance(entry, dict) else {}
        if isinstance(variant, dict) and variant.get("id"):
            presented.append(variant["id"])
    pool_ids = [v.get("id", "?") for v in pool if isinstance(v, dict)]
    event_type = event.get("type", "feedback")
    user_text = event.get("text", "")
    return HumanMessage(
        content=(
            "HUMAN FEEDBACK DURING HITL\n"
            f"Event: {event_type}\n"
            f"Tool: {tool_name or 'unknown'}\n"
            f"User: {user_text}\n"
            f"Presented: {', '.join(presented) if presented else '(none)'}\n"
            f"Pool: {', '.join(pool_ids) if pool_ids else '(empty)'}\n\n"
            "This is feedback, not approval. If the human asked a question, "
            "answer in visible text first; a tool call or present_for_review "
            "is not a substitute for answering. If they requested changes or "
            "a batch of variants, briefly acknowledge what you will try, then "
            "run the needed tool calls. After producing new candidates, compare "
            "the options and call present_for_review(mode='add') or "
            "mode='replace') only when you are deliberately making candidate(s) "
            "approvable. Do not call commit_variant or advance_phase while "
            "this HITL gate is open."
        ),
        additional_kwargs={
            "is_hitl_feedback": True,
            "hitl_event_type": event_type,
        },
    )


def approval_resume_event(variant_id: str, rationale: str = "") -> dict:
    """Structured resume payload for Gradio approval actions."""
    return {
        "type": "approve_variant",
        "variant_id": variant_id,
        "rationale": rationale,
        "text": f"Approve {variant_id}" + (f": {rationale}" if rationale else ""),
    }


def feedback_resume_event(text: str) -> dict:
    """Structured resume payload for Gradio chat feedback/questions."""
    return {
        "type": "feedback",
        "text": text,
    }


def visible_response_required(review_session: dict | None) -> bool:
    """True when the next agent turn must include visible text before tools."""
    return review_is_open(review_session) and bool(
        (review_session or {}).get("visible_response_required")
    )


def ai_message_has_visible_text(message: AIMessage) -> bool:
    """True when an assistant response contains user-visible prose."""
    return bool(text_content(message.content).strip())


def build_visible_response_required_prompt(review_session: dict | None) -> HumanMessage:
    """Policy correction that asks the model to answer before taking action."""
    event = (review_session or {}).get("last_human_event") or {}
    event_text = event.get("text", "") if isinstance(event, dict) else ""
    return HumanMessage(
        content=(
            "HITL TURN POLICY\n"
            "The human sent a review message. Your next assistant turn must "
            "include visible text that acknowledges or answers it before any "
            "tool call. A tool-only response is not acceptable collaboration.\n\n"
            f"Human message: {event_text or '(no text provided)'}\n\n"
            "Respond again with concise visible text first. If action is needed "
            "after that, you may include tool calls in the same assistant turn."
        ),
        additional_kwargs={
            "is_hitl_prompt": True,
            "is_hitl_turn_policy": True,
        },
    )


