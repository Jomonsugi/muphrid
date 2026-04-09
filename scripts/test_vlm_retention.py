#!/usr/bin/env python3
"""
Unit tests for the VLM image retention policy.

Run from project root:
    uv run python scripts/test_vlm_retention.py

Exit 0 = all cases pass. Exit 1 = one or more failed.

Builds synthetic message lists with stub base64 image_url blocks (no real
encoding required) and asserts on the output of _apply_vlm_retention_policy.
Also covers the helpers it depends on: _last_phase_boundary_index,
_current_hitl_gate_start, _is_vlm_hitl_message.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from muphrid.graph import hitl as hitl_mod
from muphrid.graph.content import image_blocks
from muphrid.graph.nodes import (
    _apply_vlm_retention_policy,
    _current_hitl_gate_start,
    _is_vlm_hitl_message,
    _last_phase_boundary_index,
)


_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "\u2713" if ok else "\u2717"
    msg = f"  {status} {name}"
    if detail:
        msg += f" \u2014 {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


def stub_image() -> dict:
    """Return a stub image_url content block (tiny fake base64 payload)."""
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,XYZ"},
    }


def vlm_hitl_msg(n_images: int, label: str = "test") -> HumanMessage:
    """Build a [VLM-HITL]-tagged multimodal HumanMessage with n stub images."""
    content = [{"type": "text", "text": f"[VLM-HITL] {label}"}]
    for _ in range(n_images):
        content.append(stub_image())
    return HumanMessage(content=content)


def vlm_auto_msg(n_images: int, label: str = "test") -> HumanMessage:
    """Build a [VLM-AUTO]-tagged multimodal HumanMessage with n stub images."""
    content = [{"type": "text", "text": f"[VLM-AUTO] {label}"}]
    for _ in range(n_images):
        content.append(stub_image())
    return HumanMessage(content=content)


def gate_closed_msg(text: str = "Approved T09_v3") -> HumanMessage:
    """Build a HumanMessage tagged with the gate_closed event marker."""
    return HumanMessage(
        content=text,
        additional_kwargs={"event": "gate_closed"},
    )


def advance_phase_tool_msg(idx: int = 0) -> ToolMessage:
    """Build a successful advance_phase ToolMessage."""
    return ToolMessage(
        content='{"status": "advanced", "from": "linear", "to": "stretch"}',
        tool_call_id=f"call_advance_{idx}",
        name="advance_phase",
    )


def total_image_count(messages: list) -> int:
    return sum(len(image_blocks(m.content)) for m in messages if hasattr(m, "content"))


def reset_vlm_modes(hitl: bool, auto: bool, cap: int = 8) -> None:
    """Set VLM mode flags via the runtime override globals."""
    hitl_mod._RUNTIME_VLM_HITL = hitl
    hitl_mod._RUNTIME_VLM_AUTONOMOUS = auto
    hitl_mod._RUNTIME_VLM_RETENTION_MAX = cap


# ── Helper-level tests ───────────────────────────────────────────────────────

print("Helper checks\n" + "=" * 40)

# _last_phase_boundary_index
msgs = [
    SystemMessage(content="sys"),
    HumanMessage(content="human"),
    advance_phase_tool_msg(0),
    AIMessage(content="moving on"),
    advance_phase_tool_msg(1),
    AIMessage(content="continuing"),
]
boundary = _last_phase_boundary_index(msgs)
check(
    "_last_phase_boundary_index returns index after most recent advance_phase",
    boundary == 5,
    f"got {boundary}, expected 5",
)

# Failed advance_phase should NOT count
msgs2 = [
    HumanMessage(content="h"),
    ToolMessage(
        content="Cannot advance: missing requirement",
        tool_call_id="x",
        name="advance_phase",
    ),
    AIMessage(content="ok"),
]
boundary2 = _last_phase_boundary_index(msgs2)
check(
    "_last_phase_boundary_index ignores failed advance_phase",
    boundary2 == 0,
    f"got {boundary2}, expected 0",
)

# _is_vlm_hitl_message
check(
    "_is_vlm_hitl_message detects [VLM-HITL] tagged HumanMessage",
    _is_vlm_hitl_message(vlm_hitl_msg(2)) is True,
)
check(
    "_is_vlm_hitl_message rejects [VLM-AUTO] tagged HumanMessage",
    _is_vlm_hitl_message(vlm_auto_msg(2)) is False,
)
check(
    "_is_vlm_hitl_message rejects plain HumanMessage",
    _is_vlm_hitl_message(HumanMessage(content="hi")) is False,
)
check(
    "_is_vlm_hitl_message rejects ToolMessage",
    _is_vlm_hitl_message(advance_phase_tool_msg()) is False,
)

# _current_hitl_gate_start
msgs3 = [
    SystemMessage(content="sys"),
    HumanMessage(content="kickoff"),
    vlm_hitl_msg(3, "gate-1 iter-1"),
    gate_closed_msg("Approved T09_v1"),
    AIMessage(content="ok"),
    vlm_hitl_msg(2, "gate-2 iter-1"),
]
gate = _current_hitl_gate_start(msgs3, phase_start=0)
check(
    "_current_hitl_gate_start finds the most recent gate_closed marker",
    gate == 4,
    f"got {gate}, expected 4 (just after the gate_closed marker at index 3)",
)

# No gate_closed marker → falls back to phase_start
msgs4 = [
    SystemMessage(content="sys"),
    HumanMessage(content="kickoff"),
    vlm_hitl_msg(2, "first-ever gate"),
]
gate2 = _current_hitl_gate_start(msgs4, phase_start=0)
check(
    "_current_hitl_gate_start falls back to phase_start when no marker exists",
    gate2 == 0,
    f"got {gate2}, expected 0",
)

# Phase boundary trumps an older gate_closed marker
msgs5 = [
    HumanMessage(content="h0"),
    gate_closed_msg("Approved T09_v1"),
    advance_phase_tool_msg(0),
    AIMessage(content="new phase"),
    vlm_hitl_msg(2, "new phase gate"),
]
phase_start_5 = _last_phase_boundary_index(msgs5)
gate3 = _current_hitl_gate_start(msgs5, phase_start_5)
check(
    "_current_hitl_gate_start: phase boundary supersedes older gate_closed marker",
    gate3 == 3,
    f"got {gate3}, expected 3 (phase_start_5={phase_start_5})",
)

# ── Policy tests ─────────────────────────────────────────────────────────────

print("\nPolicy cases\n" + "=" * 40)

# Case 1: both modes off → strip everything
reset_vlm_modes(hitl=False, auto=False, cap=8)
messages = [
    SystemMessage(content="sys"),
    HumanMessage(content="hi"),
    vlm_auto_msg(3, "earlier"),
    vlm_hitl_msg(4, "now"),
]
out = _apply_vlm_retention_policy(messages, raw_messages=messages[1:])
check(
    "case 1: both modes off → all images stripped",
    total_image_count(out) == 0,
    f"got {total_image_count(out)} images remaining",
)

# Case 2: vlm_hitl only, in active HITL with 6 variants, cap=4 → all 6 pinned
reset_vlm_modes(hitl=True, auto=False, cap=4)
messages = [
    SystemMessage(content="sys"),
    HumanMessage(content="kickoff"),
    vlm_auto_msg(2, "stale autonomous"),  # not pinned, should be stripped
    AIMessage(content="ran something"),
    ToolMessage(content="{}", tool_call_id="t1", name="remove_gradient"),
    vlm_hitl_msg(6, "active HITL gate with 6 variants"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
check(
    "case 2: vlm_hitl only, active HITL with 6 variants > cap=4, all 6 pinned",
    total_image_count(out) == 6,
    f"got {total_image_count(out)} images remaining (expected 6)",
)

# Case 3: vlm_hitl only, NOT in active HITL → strip everything (concern 1)
reset_vlm_modes(hitl=True, auto=False, cap=8)
messages = [
    SystemMessage(content="sys"),
    vlm_hitl_msg(4, "stale gate from before"),
    gate_closed_msg("Approved T09_v1"),
    AIMessage(content="working autonomously now"),
    ToolMessage(content="{}", tool_call_id="t2", name="some_tool"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
check(
    "case 3 (concern 1): vlm_hitl only, NOT in active HITL → all images stripped",
    total_image_count(out) == 0,
    f"got {total_image_count(out)} images (expected 0 — no leakage past gate)",
)

# Case 4: vlm_autonomous on, 5 single-image messages, cap=3
reset_vlm_modes(hitl=False, auto=True, cap=3)
messages = [
    SystemMessage(content="sys"),
    vlm_auto_msg(1, "img1"),
    AIMessage(content="ok"),
    vlm_auto_msg(1, "img2"),
    AIMessage(content="ok"),
    vlm_auto_msg(1, "img3"),
    AIMessage(content="ok"),
    vlm_auto_msg(1, "img4"),
    AIMessage(content="ok"),
    vlm_auto_msg(1, "img5"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
check(
    "case 4: autonomous on, 5 single-image messages, cap=3 → newest 3 kept",
    total_image_count(out) == 3,
    f"got {total_image_count(out)} images (expected 3)",
)

# Case 5: vlm_autonomous on, single message with 5 images, cap=3
reset_vlm_modes(hitl=False, auto=True, cap=3)
messages = [
    SystemMessage(content="sys"),
    vlm_auto_msg(5, "five-pack"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
check(
    "case 5: autonomous on, single 5-image message, cap=3 → message trimmed to 3",
    total_image_count(out) == 3,
    f"got {total_image_count(out)} images (expected 3)",
)

# Case 6: vlm_autonomous on, phase boundary in middle → pre-phase stripped
reset_vlm_modes(hitl=False, auto=True, cap=8)
messages = [
    SystemMessage(content="sys"),
    vlm_auto_msg(2, "pre-phase 1"),
    vlm_auto_msg(2, "pre-phase 2"),
    advance_phase_tool_msg(0),
    AIMessage(content="new phase"),
    vlm_auto_msg(2, "current phase"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Pre-phase had 4 images; current phase has 2. Expect 2.
check(
    "case 6: autonomous on, phase boundary mid-stream → only post-boundary kept",
    total_image_count(out) == 2,
    f"got {total_image_count(out)} images (expected 2)",
)

# Case 7: gate-overflow exception (concern 2)
# 8-variant gate exceeds cap=4 → gate shown in full, no budget for older imgs.
reset_vlm_modes(hitl=True, auto=True, cap=4)
messages = [
    SystemMessage(content="sys"),
    vlm_auto_msg(1, "earlier autonomous"),  # gets stripped: 0 remaining budget
    AIMessage(content="ran tool"),
    ToolMessage(content="{}", tool_call_id="t3", name="remove_gradient"),
    vlm_hitl_msg(8, "active gate with 8 variants"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Gate overflow: pin all 8 gate images, 0 remaining for older → strip auto.
check(
    "case 7 (concern 2): 8-variant gate > cap=4 → gate shown in full, older stripped",
    total_image_count(out) == 8,
    f"got {total_image_count(out)} images (expected 8 — gate overflow, no extra budget)",
)

# Case 7b: gate fits within cap → gate counts against cap normally
# cap=8, 6-variant gate, 5 older single-image messages → newest 8 visible total
reset_vlm_modes(hitl=True, auto=True, cap=8)
messages = [
    SystemMessage(content="sys"),
    vlm_auto_msg(1, "old1"),
    vlm_auto_msg(1, "old2"),
    vlm_auto_msg(1, "old3"),
    vlm_auto_msg(1, "old4"),
    vlm_auto_msg(1, "old5"),
    AIMessage(content="ran tool"),
    ToolMessage(content="{}", tool_call_id="t3b", name="remove_gradient"),
    vlm_hitl_msg(6, "active gate with 6 variants"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Gate (6) ≤ cap (8), no overflow. Walk reverse: gate takes 6, remaining=2;
# 2 newest auto images take 2; remaining 3 auto images stripped. Total = 8.
check(
    "case 7b: gate fits cap → counts against cap, total ≤ cap",
    total_image_count(out) == 8,
    f"got {total_image_count(out)} images (expected 8 — 6 from gate + 2 newest older)",
)

# Case 8: gate boundary detection — gate_closed marker between two VLM-HITL messages
reset_vlm_modes(hitl=True, auto=True, cap=8)
messages = [
    SystemMessage(content="sys"),
    vlm_hitl_msg(3, "old gate"),     # before gate_closed → not part of current gate
    gate_closed_msg("Approved T09_v1"),
    AIMessage(content="continuing"),
    ToolMessage(content="{}", tool_call_id="t4", name="remove_gradient"),
    vlm_hitl_msg(2, "new gate"),     # after gate_closed → IS the current gate
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Current gate = new gate (2). 2 ≤ cap=8, no overflow, gate counts normally.
# Walk reverse: new gate takes 2, remaining=6; old gate (3) takes 3, remaining=3.
# Total = 5.
check(
    "case 8: gate_closed marker bounds the current gate; everything fits in cap",
    total_image_count(out) == 5,
    f"got {total_image_count(out)} images (expected 5 — 2 + 3, all within cap=8)",
)
# Tight cap: current gate = 2 fits exactly, no budget left for old gate
reset_vlm_modes(hitl=True, auto=True, cap=2)
out2 = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Current gate (2) ≤ cap (2), no overflow. Walk reverse: new gate takes 2,
# remaining=0; old gate (3) → all stripped. Total = 2.
check(
    "case 8b: tight cap filled by current gate; older gate stripped",
    total_image_count(out2) == 2,
    f"got {total_image_count(out2)} images (expected 2 — current gate fills cap)",
)

# Case 9: Approved-prefix HumanMessage WITHOUT the marker is NOT a gate close
# (validates the marker is structural, not text-based)
reset_vlm_modes(hitl=True, auto=True, cap=8)
messages = [
    SystemMessage(content="sys"),
    vlm_hitl_msg(3, "first gate"),
    HumanMessage(content="Approved this is bad"),  # plain text, no marker
    ToolMessage(content="{}", tool_call_id="t5", name="remove_gradient"),
    vlm_hitl_msg(2, "second gate"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Without marker, both gates are in the same "current gate" (no boundary).
# Both VLM-HITL messages should be pinned. Total: 5
check(
    "case 9: plain 'Approved' text without marker does NOT close the gate",
    total_image_count(out) == 5,
    f"got {total_image_count(out)} images (expected 5 — both vlm_hitl msgs are in the same current gate, all fit in cap=8)",
)

# Case 10: marker propagation — explicit gate_closed marker IS detected
reset_vlm_modes(hitl=True, auto=True, cap=8)
messages = [
    SystemMessage(content="sys"),
    vlm_hitl_msg(3, "first gate"),
    HumanMessage(
        content="Approved this is bad",
        additional_kwargs={"event": "gate_closed"},
    ),
    ToolMessage(content="{}", tool_call_id="t6", name="remove_gradient"),
    vlm_hitl_msg(2, "second gate"),
]
raw = messages[1:]
out = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Current gate = second gate (2 imgs). 2 ≤ cap=8, no overflow.
# Walk reverse: second gate takes 2, remaining=6; first gate (3) takes 3.
# Total: 5
check(
    "case 10: marker correctly closes the prior gate (loose cap keeps both)",
    total_image_count(out) == 5,
    f"got {total_image_count(out)} images (expected 5 — 2 + 3, all within cap=8)",
)
# Tight cap that triggers gate-overflow on the current gate
reset_vlm_modes(hitl=True, auto=True, cap=1)
out2 = _apply_vlm_retention_policy(messages, raw_messages=raw)
# Current gate (2 imgs) > cap=1 → gate overflow. Pin current gate, 0 remaining.
# First gate (3 imgs, NOT current) → all stripped.
# Total: 2 (gate overflow shows the full current gate, nothing else)
check(
    "case 10b: tight cap triggers gate overflow; only current gate visible",
    total_image_count(out2) == 2,
    f"got {total_image_count(out2)} images (expected 2 — gate overflow shows current gate only)",
)

# Cleanup runtime overrides so we don't pollute other test runs
reset_vlm_modes(hitl=False, auto=False, cap=8)
hitl_mod._RUNTIME_VLM_HITL = None
hitl_mod._RUNTIME_VLM_AUTONOMOUS = None
hitl_mod._RUNTIME_VLM_RETENTION_MAX = None

# ── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 40)
if _failures:
    print(f"FAILED: {len(_failures)} check(s)")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)
