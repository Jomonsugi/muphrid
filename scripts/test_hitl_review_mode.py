#!/usr/bin/env python3
"""
Focused smoke tests for explicit HITL Review Mode.

Run from project root:
    uv run python scripts/test_hitl_review_mode.py

Exit 0 = all checks pass. These tests avoid full image processing and exercise
the controller/state contracts that keep human review from becoming a silent
tool loop.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from muphrid.graph import hitl as hitl_mod
from muphrid.graph import nodes
from muphrid.graph import review as review_ctl
from muphrid.graph.state import ProcessingPhase
from muphrid.tools.utility.t31_commit_variant import commit_variant
from muphrid.tools.utility.t39_present_for_review import present_for_review


_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    msg = f"  {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


def temp_fit(name: str = "variant.fit") -> str:
    path = Path(tempfile.mkdtemp(prefix="hitl_review_test_")) / name
    path.write_bytes(b"SIMPLE  =                    T\nEND\n")
    return str(path)


def base_state() -> dict:
    return {
        "phase": ProcessingPhase.STRETCH,
        "dataset": {"working_dir": tempfile.mkdtemp(prefix="hitl_review_wd_")},
        "paths": {"current_image": None},
        "metadata": {},
        "metrics": {"is_linear_estimate": False},
        "messages": [],
        "active_hitl": False,
        "review_session": None,
        "variant_pool": [],
        "visual_context": [],
        "user_feedback": {},
    }


def make_variant(variant_id: str = "T14_v1") -> dict:
    path = temp_fit(f"{variant_id}.fit")
    return {
        "id": variant_id,
        "phase": "stretch",
        "tool_name": "stretch_image",
        "label": f"{variant_id} - stretch",
        "params": {"amount": 0.7},
        "file_path": path,
        "preview_path": None,
        "metrics": {"signal_coverage_pct": 42.0},
        "created_at": review_ctl.utc_now(),
        "rationale": None,
    }


def test_typed_events() -> None:
    question = review_ctl.parse_human_event(
        {"type": "question", "text": "Which one should we use?"}
    )
    check("question event stays typed", question["type"] == "question")

    approval = review_ctl.parse_human_event(
        {"type": "approve_variant", "variant_id": "T14_v2", "rationale": "best balance"}
    )
    check(
        "approval event carries variant id",
        approval["type"] == "approve_variant" and approval["variant_id"] == "T14_v2",
    )
    legacy = review_ctl.parse_human_event('__APPROVE_VARIANT__{"id": "T14_v2"}')
    check("legacy sentinel is plain feedback, not approval", legacy["type"] == "feedback")


def test_review_session_and_prompt() -> None:
    hitl_mod.set_hitl_tool_enabled("T14_stretch", True)
    state = base_state()
    state["messages"] = [
        ToolMessage(
            content=json.dumps({"output_path": temp_fit("stretch.fit")}),
            name="stretch_image",
            tool_call_id="tool-1",
        )
    ]
    update = nodes.hitl_check(state)
    session = update.get("review_session")
    check("hitl_check opens review_session", review_ctl.review_is_open(session))
    check("active_hitl compatibility mirror set", update.get("active_hitl") is True)
    check(
        "review open prompt injected",
        bool(update.get("messages")) and "HITL REVIEW OPEN" in update["messages"][0].content,
    )


def test_present_for_review_artifact() -> None:
    hitl_mod.set_hitl_tool_enabled("T14_stretch", True)
    state = base_state()
    variant = make_variant()
    state["active_hitl"] = True
    state["variant_pool"] = [variant]
    state["messages"] = [
        ToolMessage(
            content=json.dumps({"output_path": variant["file_path"]}),
            name="stretch_image",
            tool_call_id="tool-1",
        )
    ]
    state["review_session"] = review_ctl.make_review_session(
        state=state,
        hitl_key="T14_stretch",
        tool_name="stretch_image",
    )

    command = present_for_review.func(
        variant_ids=["T14_v1"],
        rationale="Best balance of highlights and nebulosity.",
        recommendation="T14_v1",
        tradeoffs=["Protects highlights better than the stronger stretch."],
        metric_highlights={"signal_coverage_pct": 42.0},
        mode="replace",
        tool_call_id="present-1",
        state=state,
    )
    update = command.update
    artifact = update["review_session"]["proposal"]
    check("present_for_review writes proposal artifact", artifact["candidates"][0]["variant_id"] == "T14_v1")
    check("proposal artifact carries recommendation", artifact["recommendation"] == "T14_v1")
    check("proposal artifact carries tradeoffs", bool(artifact["tradeoffs"]))


def test_missing_review_session_is_not_gate() -> None:
    hitl_mod.set_hitl_tool_enabled("T14_stretch", True)
    state = base_state()
    state["active_hitl"] = True
    state["variant_pool"] = [make_variant()]
    command = present_for_review.func(
        variant_ids=["T14_v1"],
        rationale="Should block without explicit session.",
        mode="replace",
        tool_call_id="present-missing-session",
        state=state,
    )
    payload = json.loads(command.update["messages"][0].content)
    check("active_hitl without review_session is not an open gate", payload["status"] == "blocked")


def test_typed_approval_closes_review() -> None:
    hitl_mod.set_hitl_tool_enabled("T14_stretch", True)
    state = base_state()
    variant = make_variant()
    state["active_hitl"] = True
    state["variant_pool"] = [variant]
    state["review_session"] = review_ctl.make_review_session(
        state=state,
        hitl_key="T14_stretch",
        tool_name="stretch_image",
        status="awaiting_human_approval",
    )
    state["review_session"]["proposal"] = review_ctl.proposal_from_candidates(
        [{
            "variant_id": "T14_v1",
            "rationale": "Best balance.",
            "presented_at": review_ctl.utc_now(),
        }],
        recommendation="T14_v1",
        rationale="Best balance.",
    )
    state["messages"] = [
        ToolMessage(
            content=json.dumps({"output_path": variant["file_path"]}),
            name="stretch_image",
            tool_call_id="tool-1",
        ),
        AIMessage(content="I recommend T14_v1."),
    ]

    old_interrupt = nodes.interrupt
    nodes.interrupt = lambda payload: review_ctl.approval_resume_event("T14_v1", "approved")
    try:
        update = nodes.hitl_check(state)
    finally:
        nodes.interrupt = old_interrupt

    check("typed approval closes active_hitl", update.get("active_hitl") is False)
    check("typed approval closes review_session", update.get("review_session", {}).get("status") == "closed")
    check("typed approval promotes current_image", update.get("paths", {}).get("current_image") == variant["file_path"])


def test_unpresented_variant_approval_rejected() -> None:
    hitl_mod.set_hitl_tool_enabled("T14_stretch", True)
    state = base_state()
    v1 = make_variant("T14_v1")
    v2 = make_variant("T14_v2")
    state["active_hitl"] = True
    state["variant_pool"] = [v1, v2]
    state["review_session"] = review_ctl.make_review_session(
        state=state,
        hitl_key="T14_stretch",
        tool_name="stretch_image",
        status="awaiting_human_approval",
    )
    state["review_session"]["proposal"] = review_ctl.proposal_from_candidates(
        [{
            "variant_id": "T14_v1",
            "rationale": "Only v1 has been presented.",
            "presented_at": review_ctl.utc_now(),
        }],
        recommendation="T14_v1",
        rationale="Only v1 has been presented.",
    )
    state["messages"] = [
        ToolMessage(
            content=json.dumps({"output_path": v1["file_path"]}),
            name="stretch_image",
            tool_call_id="tool-1",
        ),
        AIMessage(content="I recommend T14_v1."),
    ]

    old_interrupt = nodes.interrupt
    nodes.interrupt = lambda payload: review_ctl.approval_resume_event("T14_v2", "wrong button")
    try:
        update = nodes.hitl_check(state)
    finally:
        nodes.interrupt = old_interrupt

    message_text = update.get("messages", [HumanMessage(content="")])[0].content
    check("unpresented approval stays in HITL", update.get("active_hitl") is True)
    check(
        "unpresented approval is rejected",
        "not in the current review proposal" in message_text,
    )


def test_visible_answer_required_before_tools() -> None:
    state = base_state()
    session = review_ctl.make_review_session(
        state=state,
        hitl_key="T14_stretch",
        tool_name="stretch_image",
        status="awaiting_agent_response",
    )
    session = review_ctl.update_review_session(
        session,
        last_human_event={"type": "feedback", "text": "Which one is flatter?"},
        visible_response_required=True,
    )
    state["review_session"] = session

    class ToolOnlyModel:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "present_for_review",
                    "args": {"variant_ids": ["T14_v1"], "rationale": "best"},
                    "id": "call-1",
                }],
            )

    agent = nodes.make_agent_node(lambda phase: ToolOnlyModel())
    update = agent(state)
    msg = update["messages"][0]
    check("tool-only response replaced with turn-policy prompt", isinstance(msg, HumanMessage))
    check(
        "turn-policy prompt marked for agent retry",
        getattr(msg, "additional_kwargs", {}).get("is_hitl_turn_policy") is True,
    )


def test_visible_answer_with_tool_calls_is_allowed() -> None:
    """The other half of the visible-text policy: text + tool_calls in the
    same assistant message must pass through, and the visible_response_required
    flag must clear so subsequent agent turns are unrestricted again.

    Without this test, a future tightening of the predicate (e.g. "no
    tool_calls when visible_response_required") would silently turn the
    policy into "agent must answer in text-only and then re-emit tools
    next turn," which doubles HITL latency and never gets surfaced as a
    regression of this specific affordance — it would show up as "agent
    feels sluggish in HITL" with no obvious cause.
    """
    state = base_state()
    session = review_ctl.make_review_session(
        state=state,
        hitl_key="T14_stretch",
        tool_name="stretch_image",
        status="awaiting_agent_response",
    )
    session = review_ctl.update_review_session(
        session,
        last_human_event={"type": "feedback", "text": "Which one is flatter?"},
        visible_response_required=True,
    )
    state["review_session"] = session

    class TextAndToolModel:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    "T14_v2 is flatter — its background quadrants are within "
                    "0.5% of each other vs 2.1% for T14_v1. Surfacing it now."
                ),
                tool_calls=[{
                    "name": "present_for_review",
                    "args": {
                        "variant_ids": ["T14_v2"],
                        "rationale": "Flatter background per quadrant analysis.",
                    },
                    "id": "call-2",
                }],
            )

    agent = nodes.make_agent_node(lambda phase: TextAndToolModel())
    update = agent(state)

    msg = update["messages"][0]
    check(
        "text+tool_calls response passes through as AIMessage",
        isinstance(msg, AIMessage),
    )
    check(
        "tool_calls are preserved (not stripped)",
        bool(getattr(msg, "tool_calls", None))
        and msg.tool_calls[0].get("name") == "present_for_review",
    )
    check(
        "visible text on message is non-empty",
        bool(str(msg.content or "").strip()),
    )

    updated_session = update.get("review_session") or {}
    check(
        "visible_response_required cleared after a valid text response",
        updated_session.get("visible_response_required") is False,
    )


def test_commit_variant_blocked_by_review_session() -> None:
    hitl_mod.set_hitl_tool_enabled("T14_stretch", True)
    state = base_state()
    state["review_session"] = review_ctl.make_review_session(
        state=state,
        hitl_key="T14_stretch",
        tool_name="stretch_image",
        status="awaiting_human_approval",
    )
    command = commit_variant.func(
        variant_id="T14_v1",
        rationale="agent should not self-commit",
        state=state,
        tool_call_id="commit-1",
    )
    content = command.update["messages"][0].content
    check("commit_variant blocked by explicit review_session", "review_session_requires_human_approval" in content)


def main() -> int:
    print("HITL Review Mode smoke tests")
    test_typed_events()
    test_review_session_and_prompt()
    test_present_for_review_artifact()
    test_missing_review_session_is_not_gate()
    test_typed_approval_closes_review()
    test_unpresented_variant_approval_rejected()
    test_visible_answer_required_before_tools()
    test_visible_answer_with_tool_calls_is_allowed()
    test_commit_variant_blocked_by_review_session()
    if _failures:
        print(f"\n{len(_failures)} failure(s): {', '.join(_failures)}")
        return 1
    print("\nAll HITL Review Mode checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
