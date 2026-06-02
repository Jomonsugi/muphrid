#!/usr/bin/env python3
"""
Focused tests for state-diff effect detection (the no-op contract).

Run from project root:
    uv run python scripts/test_tool_effects.py

Exit 0 = all checks pass.

What this exercises:

  1. current_image_writer_names() — the registry derives, structurally, the
     set of tools that advance paths.current_image. Pointer tools
     (select_stretch_variant, restore_checkpoint) are included; tools that
     legitimately leave the working image unchanged (analyze_image,
     save_checkpoint) are excluded.

  2. variant_snapshot — computes, from the authoritative state diff
     (pre_action_image vs paths.current_image), whether each image-advancing
     call changed the working image, and records it in state.tool_effects.

  3. _check_stuck_loop — reads tool_effects (NOT ToolMessage content) to
     collapse no-op calls. This catches the "different argument each time, all
     no-ops" loop, and does so even when nothing in the message text says
     "noop" — proving the source of truth is state, not the transcript.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    msg = f" {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


def _ai(tool_name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _tool(tool_name: str, call_id: str, content: str = "{}") -> ToolMessage:
    return ToolMessage(content=content, name=tool_name, tool_call_id=call_id)


def test_writer_set() -> None:
    print("\n[1] current_image_writer_names() scopes effect detection correctly")
    from muphrid.graph.registry import current_image_writer_names

    w = current_image_writer_names()
    for name in ("select_stretch_variant", "restore_checkpoint", "stretch_image"):
        check(f"{name} is an image-advancing writer", name in w)
    for name in ("analyze_image", "save_checkpoint", "advance_phase"):
        check(f"{name} is NOT treated as image-advancing", name not in w)


def test_variant_snapshot_effect() -> None:
    print("\n[2] variant_snapshot records effect from the state diff")
    from muphrid.graph.nodes import variant_snapshot

    # Changed: pre != after → effect True.
    state_changed = {
        "pre_action_image": "/runs/aggressive.fit",
        "paths": {"current_image": "/runs/gentle.fit"},
        "messages": [
            _ai("select_stretch_variant", {"variant": "gentle"}, "s1"),
            _tool("select_stretch_variant", "s1"),
        ],
    }
    out = variant_snapshot(state_changed)
    check(
        "real select records changed=True",
        out.get("tool_effects") == {"s1": True},
        f"out={out}",
    )

    # No-op: pre == after → effect False.
    state_noop = {
        "pre_action_image": "/runs/gentle.fit",
        "paths": {"current_image": "/runs/gentle.fit"},
        "messages": [
            _ai("select_stretch_variant", {"variant": "gentle"}, "s2"),
            _tool("select_stretch_variant", "s2"),
        ],
    }
    out = variant_snapshot(state_noop)
    check(
        "redundant select records changed=False",
        out.get("tool_effects") == {"s2": False},
        f"out={out}",
    )

    # Non-writer tool: no effect entry at all.
    state_nonwriter = {
        "pre_action_image": "/runs/gentle.fit",
        "paths": {"current_image": "/runs/gentle.fit"},
        "messages": [
            _ai("analyze_image", {}, "a1"),
            _tool("analyze_image", "a1"),
        ],
    }
    out = variant_snapshot(state_nonwriter)
    check(
        "analyze_image produces no tool_effects entry",
        "tool_effects" not in out,
        f"out={out}",
    )

    # Errored writer call: excluded (the detector handles errors separately).
    err_tool = ToolMessage(
        content="boom", name="select_stretch_variant", tool_call_id="s3", status="error"
    )
    state_err = {
        "pre_action_image": "/runs/gentle.fit",
        "paths": {"current_image": "/runs/gentle.fit"},
        "messages": [_ai("select_stretch_variant", {"variant": "x"}, "s3"), err_tool],
    }
    out = variant_snapshot(state_err)
    check(
        "errored writer call is excluded from tool_effects",
        "tool_effects" not in out,
        f"out={out}",
    )


def test_stuck_loop_reads_state() -> None:
    print("\n[3] _check_stuck_loop collapses no-ops from state, not message text")
    os.environ["MAX_CONSECUTIVE_SAME_TOOL"] = "3"
    from muphrid.graph.nodes import StuckLoopError, _check_stuck_loop

    # Three restore_checkpoint calls with DIFFERENT args — args-fingerprint
    # alone would never trip. Message content carries NO "noop" string.
    messages = [
        HumanMessage(content="go"),
        _ai("restore_checkpoint", {"name": "a"}, "r1"),
        _tool("restore_checkpoint", "r1", '{"restored": "a"}'),
        _ai("restore_checkpoint", {"name": "b"}, "r2"),
        _tool("restore_checkpoint", "r2", '{"restored": "b"}'),
        _ai("restore_checkpoint", {"name": "c"}, "r3"),
        _tool("restore_checkpoint", "r3", '{"restored": "c"}'),
    ]
    assert all("noop" not in m.content for m in messages if isinstance(m, ToolMessage))

    # With state marking all three as no-ops → effect-collapse → trips.
    noop_effects = {"r1": False, "r2": False, "r3": False}
    try:
        _check_stuck_loop(messages, noop_effects)
        check("different-args no-ops trip via tool_effects", False, "did not raise")
    except StuckLoopError as e:
        check("different-args no-ops trip via tool_effects", "no-op" in str(e).lower())

    # Same messages, but state says each call changed current_image → distinct
    # args fingerprints, count 1 each → no trip. Proves state is the driver.
    changed_effects = {"r1": True, "r2": True, "r3": True}
    try:
        _check_stuck_loop(messages, changed_effects)
        check("real distinct restores do NOT trip", True)
    except StuckLoopError as e:
        check("real distinct restores do NOT trip", False, f"raised: {e}")

    # The real incident, replayed: one genuine select then redundant
    # re-selects of the already-active variant. The first changed the image;
    # the rest are no-ops. Under the state-driven detector these collapse on
    # the no-op fingerprint and trip with the *actionable no-op* guidance once
    # the no-op count reaches the limit (rather than the generic args abort).
    incident = [
        HumanMessage(content="go"),
        _ai("select_stretch_variant", {"variant": "gentle"}, "g1"),
        _tool("select_stretch_variant", "g1"),  # changed (aggressive -> gentle)
        _ai("select_stretch_variant", {"variant": "gentle"}, "g2"),
        _tool("select_stretch_variant", "g2"),  # no-op
        _ai("select_stretch_variant", {"variant": "gentle"}, "g3"),
        _tool("select_stretch_variant", "g3"),  # no-op
        _ai("select_stretch_variant", {"variant": "gentle"}, "g4"),
        _tool("select_stretch_variant", "g4"),  # no-op -> 3rd no-op, trips
    ]
    try:
        _check_stuck_loop(incident, {"g1": True, "g2": False, "g3": False, "g4": False})
        check("redundant re-selects trip on the no-op path", False, "did not raise")
    except StuckLoopError as e:
        check("redundant re-selects trip on the no-op path", "no-op" in str(e).lower())

    # Identical args that genuinely change each time still trip on the args
    # fingerprint — the args-based path is unaffected by the new no-op path.
    allchg = [
        HumanMessage(content="go"),
        _ai("curves_adjust", {"k": 1}, "c1"),
        _tool("curves_adjust", "c1"),
        _ai("curves_adjust", {"k": 1}, "c2"),
        _tool("curves_adjust", "c2"),
        _ai("curves_adjust", {"k": 1}, "c3"),
        _tool("curves_adjust", "c3"),
    ]
    try:
        _check_stuck_loop(allchg, {"c1": True, "c2": True, "c3": True})
        check("identical-args effective repeats trip on args path", False, "did not raise")
    except StuckLoopError:
        check("identical-args effective repeats trip on args path", True)


def main() -> int:
    test_writer_set()
    test_variant_snapshot_effect()
    test_stuck_loop_reads_state()
    print()
    if _failures:
        print(f"FAIL — {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f" - {f}")
        return 1
    print("All tool-effect contract tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
