#!/usr/bin/env python3
"""
Unit tests for the state-driven VLM view construction.

Run from project root:
    uv run python scripts/test_vlm_retention.py

Exit 0 = all checks pass. Exit 1 = one or more failed.

The architecture:
  - state.variant_pool   → canonical store for active HITL gate variants
  - state.visual_context → non-variant working set (present_images, phase_carry)

_select_visible_refs reads BOTH and produces the filtered VisualRef list the
agent should see this turn. _build_vlm_view base64-encodes those paths into
an ephemeral multimodal HumanMessage and appends it to the message list. No
"retention policy" walks messages — state owns visibility end to end.

Tests cover:
  1. _select_visible_refs   (pure-ish: state → filtered VisualRef list)
  2. _build_vlm_view        (state + messages → messages + ephemeral VLM msg)

Stub JPGs are written to a tempdir so _make_vlm_message has real bytes to
base64-encode.
"""

import os
import sys
import tempfile
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
from muphrid.graph.content import image_blocks, text_content
from muphrid.graph.nodes import (
    _build_vlm_view,
    _format_variant_pool_for_prompt,
    _select_visible_refs,
    _strip_vlm_images,
    build_variant_promotion_update,
)
from muphrid.graph.state import Variant, VisualRef
from muphrid.tools.utility.t31_commit_variant import commit_variant


_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "\u2713" if ok else "\u2717"
    msg = f"  {status} {name}"
    if detail:
        msg += f" \u2014 {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


# ── Stub fixtures ────────────────────────────────────────────────────────────

# Minimal valid JPG header so _make_vlm_message has something to base64-encode.
# (_make_vlm_message just reads file bytes and encodes them — content isn't
# validated by the helper itself.)
_FAKE_JPG_BYTES = b"\xff\xd8\xff\xe0fakejpegbytes\xff\xd9"


def make_stub_jpgs(n: int) -> list[str]:
    """Create n stub JPG files in a session-wide tempdir; return paths."""
    if not hasattr(make_stub_jpgs, "_dir"):
        make_stub_jpgs._dir = tempfile.mkdtemp(prefix="vlm_test_")
        make_stub_jpgs._counter = 0
    paths = []
    for _ in range(n):
        make_stub_jpgs._counter += 1
        p = Path(make_stub_jpgs._dir) / f"stub_{make_stub_jpgs._counter}.jpg"
        p.write_bytes(_FAKE_JPG_BYTES)
        paths.append(str(p))
    return paths


def vref(path: str, source: str, label: str = "stub", phase: str = "linear") -> VisualRef:
    return VisualRef(path=path, label=label, source=source, phase=phase)


def stub_variant(preview_path: str, vid: str = "T09_v1", label: str | None = None) -> Variant:
    """Build a stub Variant pointing at an existing JPG preview path."""
    return Variant(
        id=vid,
        phase="linear",
        tool_name="remove_gradient",
        label=label or f"variant {vid}",
        params={},
        file_path=preview_path,           # stub: same as preview for tests
        preview_path=preview_path,        # _resolve_variant_preview returns this if it exists
        metrics={},
        created_at="2026-04-08T00:00:00Z",
        rationale=None,
    )


def make_state(
    variant_pool: list[Variant] | None = None,
    visual_context: list[VisualRef] | None = None,
    **extra,
) -> dict:
    """Build a minimal AstroState-shaped dict for the helpers under test."""
    base = {
        "variant_pool": variant_pool or [],
        "visual_context": visual_context or [],
        "active_hitl": False,
        "phase": "linear",
        "dataset": {"working_dir": ""},
        "metrics": {"is_linear_estimate": True},
    }
    base.update(extra)
    return base


def total_image_count(messages: list) -> int:
    return sum(
        len(image_blocks(m.content)) for m in messages if hasattr(m, "content")
    )


def reset_vlm_modes(hitl: bool, auto: bool, cap: int = 8) -> None:
    """Set VLM mode flags via the runtime override globals.

    Note: vlm_hitl() now always returns True (collaboration requires visual
    access). The `hitl` parameter is retained for call-site compatibility
    with legacy test cases but has no effect — when False, the test case
    is exercising a state that is no longer reachable in production.
    """
    _ = hitl  # legacy: vlm_hitl is always on now
    hitl_mod._RUNTIME_VLM_AUTONOMOUS = auto
    hitl_mod._RUNTIME_VLM_RETENTION_MAX = cap


# ── _select_visible_refs cases ──────────────────────────────────────────────

print("_select_visible_refs cases\n" + "=" * 40)

# Case 1: vlm_hitl is always on; auto off still shows explicit state refs
reset_vlm_modes(hitl=False, auto=False, cap=8)
paths = make_stub_jpgs(2)
state = make_state(
    variant_pool=[stub_variant(paths[0], "T09_v1")],
    visual_context=[vref(paths[1], "present_images")],
)
out = _select_visible_refs(state)
check(
    "case 1: auto off still shows HITL-capable refs",
    len(out) == 2,
    f"got {len(out)}, sources={[r['source'] for r in out]}",
)

# Case 2: vlm_hitl only → variant pool plus explicit visual_context are visible
reset_vlm_modes(hitl=True, auto=False, cap=8)
paths = make_stub_jpgs(4)
state = make_state(
    variant_pool=[
        stub_variant(paths[0], "T09_v1"),
        stub_variant(paths[1], "T09_v2"),
    ],
    visual_context=[
        vref(paths[2], "present_images"),
        vref(paths[3], "phase_carry"),
    ],
)
out = _select_visible_refs(state)
check(
    "case 2: vlm_hitl only → variant_pool and visual_context visible",
    len(out) == 4
    and [r["source"] for r in out] == [
        "present_images", "phase_carry", "hitl_variant", "hitl_variant"
    ],
    f"got {len(out)}, sources={[r['source'] for r in out]}",
)

# Case 3: vlm_hitl only, empty variant_pool → explicit visual_context remains visible
reset_vlm_modes(hitl=True, auto=False, cap=8)
paths = make_stub_jpgs(2)
state = make_state(
    variant_pool=[],
    visual_context=[
        vref(paths[0], "present_images"),
        vref(paths[1], "phase_carry"),
    ],
)
out = _select_visible_refs(state)
check(
    "case 3: vlm_hitl only, empty pool → visual_context visible",
    len(out) == 2,
    f"got {len(out)}, sources={[r['source'] for r in out]}",
)

# Case 4: vlm_autonomous on, total entries ≤ cap → all visible
reset_vlm_modes(hitl=False, auto=True, cap=8)
paths = make_stub_jpgs(5)
state = make_state(
    visual_context=[vref(p, "present_images") for p in paths],
)
out = _select_visible_refs(state)
check(
    "case 4: autonomous on, 5 ≤ cap=8 → all 5 visible",
    len(out) == 5,
    f"got {len(out)}",
)

# Case 5: vlm_autonomous on, total entries > cap → newest cap visible
reset_vlm_modes(hitl=False, auto=True, cap=3)
paths = make_stub_jpgs(5)
state = make_state(
    visual_context=[
        vref(paths[0], "present_images", label="oldest"),
        vref(paths[1], "present_images"),
        vref(paths[2], "present_images"),
        vref(paths[3], "present_images"),
        vref(paths[4], "present_images", label="newest"),
    ],
)
out = _select_visible_refs(state)
check(
    "case 5: autonomous on, 5 > cap=3 → newest 3 visible",
    len(out) == 3 and out[-1]["label"] == "newest" and out[0]["path"] == paths[2],
    f"got {len(out)}, paths={[r['path'] for r in out]}",
)

# Case 6 (concern 2): gate overflow — variant_pool alone exceeds cap
reset_vlm_modes(hitl=True, auto=True, cap=4)
paths = make_stub_jpgs(9)
state = make_state(
    variant_pool=[stub_variant(paths[i + 1], f"T09_v{i+1}") for i in range(8)],
    visual_context=[vref(paths[0], "present_images")],
)
out = _select_visible_refs(state)
check(
    "case 6 (concern 2): 8 variants > cap=4 → overflow shows all 8 variants, drops other sources",
    len(out) == 8 and all(r["source"] == "hitl_variant" for r in out),
    f"got {len(out)}, sources={[r['source'] for r in out]}",
)

# Case 6b: pool fits within cap → variants + visual_context all visible
reset_vlm_modes(hitl=True, auto=True, cap=8)
paths = make_stub_jpgs(5)
state = make_state(
    variant_pool=[
        stub_variant(paths[2], "T09_v1"),
        stub_variant(paths[3], "T09_v2"),
        stub_variant(paths[4], "T09_v3"),
    ],
    visual_context=[
        vref(paths[0], "present_images"),
        vref(paths[1], "present_images"),
    ],
)
out = _select_visible_refs(state)
check(
    "case 6b: pool fits within cap → 5 entries visible (3 variants + 2 visual_context)",
    len(out) == 5,
    f"got {len(out)}",
)

# Case 6c: pool ≤ cap but total > cap → newest cap entries (variants prioritized)
reset_vlm_modes(hitl=True, auto=True, cap=3)
paths = make_stub_jpgs(4)
state = make_state(
    variant_pool=[
        stub_variant(paths[2], "T09_v1"),
        stub_variant(paths[3], "T09_v2"),
    ],
    visual_context=[
        vref(paths[0], "present_images"),
        vref(paths[1], "present_images"),
    ],
)
out = _select_visible_refs(state)
# Combined order: visual_context first, variants last (newest). Newest 3:
# present_images[1], variant_v1, variant_v2.
check(
    "case 6c: pool ≤ cap but total > cap → newest 3 visible (variants at the end)",
    len(out) == 3
    and out[-1]["source"] == "hitl_variant"
    and out[-2]["source"] == "hitl_variant"
    and out[0]["source"] == "present_images",
    f"got {len(out)}, sources={[r['source'] for r in out]}",
)

# Case 7: phase_carry + present_images, vlm_autonomous, no variants
reset_vlm_modes(hitl=False, auto=True, cap=8)
paths = make_stub_jpgs(2)
state = make_state(
    visual_context=[
        vref(paths[0], "phase_carry", label="approved from prior gate"),
        vref(paths[1], "present_images"),
    ],
)
out = _select_visible_refs(state)
check(
    "case 7: autonomous on, phase_carry + present_images visible (empty pool)",
    len(out) == 2,
    f"got {len(out)}",
)


# ── _build_vlm_view cases (with real bytes for encoding) ────────────────────

print("\n_build_vlm_view cases\n" + "=" * 40)

# Case 8: auto off but HITL visual access on → historical images stripped, state refs injected
reset_vlm_modes(hitl=False, auto=False, cap=8)
paths = make_stub_jpgs(1)
historical_msg = HumanMessage(content=[
    {"type": "text", "text": "old VLM injection from legacy state"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,XYZ"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,XYZ"}},
])
messages = [
    SystemMessage(content="sys"),
    historical_msg,
    AIMessage(content="ok"),
]
state = make_state(variant_pool=[stub_variant(paths[0], "T09_v1")])
out = _build_vlm_view(state, messages)
check(
    "case 8: historical images stripped; HITL state ref injected",
    total_image_count(out) == 1,
    f"got {total_image_count(out)} images in view",
)

# Case 9: autonomous on, refs in state → builds fresh multimodal message
reset_vlm_modes(hitl=False, auto=True, cap=8)
paths = make_stub_jpgs(3)
messages = [
    SystemMessage(content="sys"),
    HumanMessage(content="kickoff"),
    AIMessage(content="working"),
]
state = make_state(
    visual_context=[
        vref(paths[0], "present_images", label="img1"),
        vref(paths[1], "present_images", label="img2"),
        vref(paths[2], "present_images", label="img3"),
    ],
)
out = _build_vlm_view(state, messages)
check(
    "case 9: autonomous on, 3 refs in state → 3 images injected",
    total_image_count(out) == 3,
    f"got {total_image_count(out)} images",
)
check(
    "case 9: injected message is the LAST message in the view",
    isinstance(out[-1], HumanMessage) and len(image_blocks(out[-1].content)) == 3,
    "last message is not a multimodal HumanMessage with 3 images",
)

# Case 10: historical multimodal in messages + state refs → historical stripped, state wins
reset_vlm_modes(hitl=False, auto=True, cap=8)
paths = make_stub_jpgs(2)
historical_msg = HumanMessage(content=[
    {"type": "text", "text": "stale legacy VLM"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,STALE"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,STALE"}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,STALE"}},
])
messages = [
    SystemMessage(content="sys"),
    historical_msg,  # 3 stale images
    AIMessage(content="ok"),
]
state = make_state(
    variant_pool=[
        stub_variant(paths[0], "T09_v1"),
        stub_variant(paths[1], "T09_v2"),
    ],
)
out = _build_vlm_view(state, messages)
# Stale 3 stripped, fresh 2 injected → total 2
check(
    "case 10: historical images stripped; variant_pool wins",
    total_image_count(out) == 2,
    f"got {total_image_count(out)} (expected 2 — 3 stale stripped, 2 fresh added)",
)

# Case 11 (concern 2 end-to-end): cap=4, 6 variants in pool → all 6 visible
reset_vlm_modes(hitl=True, auto=True, cap=4)
paths = make_stub_jpgs(7)
state = make_state(
    variant_pool=[stub_variant(paths[i + 1], f"T09_v{i+1}") for i in range(6)],
    visual_context=[vref(paths[0], "present_images", label="earlier autonomous")],
    active_hitl=True,
)
messages = [SystemMessage(content="sys"), HumanMessage(content="review please")]
out = _build_vlm_view(state, messages)
# Gate overflow: 6 > cap=4 → show only the 6 variants, drop the autonomous one
check(
    "case 11 (concern 2): 6 variants in pool > cap=4 → all 6 visible, autonomous dropped",
    total_image_count(out) == 6,
    f"got {total_image_count(out)} (expected 6 — gate overflow)",
)

# Case 12: vlm_hitl only, empty pool → explicit visual_context is still visible
reset_vlm_modes(hitl=True, auto=False, cap=8)
paths = make_stub_jpgs(2)
state = make_state(
    variant_pool=[],
    visual_context=[
        vref(paths[0], "present_images"),
        vref(paths[1], "phase_carry"),
    ],
)
messages = [SystemMessage(content="sys"), HumanMessage(content="working")]
out = _build_vlm_view(state, messages)
check(
    "case 12: vlm_hitl only, empty pool → visual_context visible",
    total_image_count(out) == 2,
    f"got {total_image_count(out)}",
)


# ── build_variant_promotion_update + commit_variant tool ────────────────────

print("\nbuild_variant_promotion_update + commit_variant cases\n" + "=" * 40)

# Case 13: build_variant_promotion_update happy path — paths/pool updated and gate visuals cleared
paths = make_stub_jpgs(3)
state = make_state(
    variant_pool=[
        stub_variant(paths[0], "T09_v1", label="gradient pass A"),
        stub_variant(paths[1], "T09_v2", label="gradient pass B"),
        stub_variant(paths[2], "T09_v3", label="gradient pass C"),
    ],
    visual_context=[],
    paths={"current_image": "/old/path.fits"},
)
result = build_variant_promotion_update(state, "T09_v2")
check(
    "case 13: build_variant_promotion_update returns (variant, update) for valid id",
    result is not None,
    "got None for valid id",
)
variant, update = result
check(
    "case 13: paths.current_image points at the chosen variant",
    update["paths"]["current_image"] == paths[1],
    f"got {update['paths']['current_image']!r} (expected {paths[1]!r})",
)
check(
    "case 13: variant_pool is cleared",
    update["variant_pool"] == [],
    f"got {update['variant_pool']}",
)
check(
    "case 13: visual_context is cleared of gate carry entries",
    update["visual_context"] == [],
    f"got {update['visual_context']}",
)

# Case 14: invalid variant id → None
state = make_state(
    variant_pool=[stub_variant(make_stub_jpgs(1)[0], "T09_v1")],
)
result = build_variant_promotion_update(state, "T09_v999")
check(
    "case 14: build_variant_promotion_update returns None for unknown id",
    result is None,
    f"got {result}",
)

# Case 15: present_images survive promotion; stale phase_carry is cleared
paths = make_stub_jpgs(3)
existing_present = vref(paths[0], "present_images", label="prior inspection")
existing_carry = vref(paths[1], "phase_carry", label="earlier carry")
state = make_state(
    variant_pool=[stub_variant(paths[2], "T09_v1", label="new variant")],
    visual_context=[existing_present, existing_carry],
)
_, update = build_variant_promotion_update(state, "T09_v1")
new_visual = update["visual_context"]
check(
    "case 15: present_images survives and stale phase_carry is cleared",
    existing_present in new_visual and existing_carry not in new_visual,
    f"existing entries missing from {new_visual}",
)
check(
    "case 15: no new phase_carry entry is appended",
    all(r["source"] != "phase_carry" for r in new_visual),
    f"got tail {new_visual[-1]}",
)
check(
    "case 15: total visual_context length keeps only present_images",
    len(new_visual) == 1,
    f"got {len(new_visual)}",
)

# Case 16: commit_variant tool — happy path returns Command with ToolMessage
paths = make_stub_jpgs(2)
state = make_state(
    variant_pool=[
        stub_variant(paths[0], "T09_v1", label="pass A"),
        stub_variant(paths[1], "T09_v2", label="pass B"),
    ],
)
cmd = commit_variant.invoke({
    "variant_id": "T09_v2",
    "rationale": "cleaner gradient",
    "state": state,
    "tool_call_id": "test_tcid_1",
})
check(
    "case 16: commit_variant returns a Command",
    cmd is not None and hasattr(cmd, "update"),
    f"got {type(cmd).__name__}",
)
check(
    "case 16: Command.update.paths.current_image is the chosen variant's file",
    cmd.update["paths"]["current_image"] == paths[1],
    f"got {cmd.update['paths']['current_image']!r}",
)
check(
    "case 16: Command.update.variant_pool is empty after commit",
    cmd.update["variant_pool"] == [],
)
check(
    "case 16: Command.update clears carry visual_context entries",
    cmd.update["visual_context"] == [],
    f"got {cmd.update['visual_context']}",
)
tool_msgs = cmd.update["messages"]
check(
    "case 16: Command.update.messages contains exactly one ToolMessage",
    len(tool_msgs) == 1 and tool_msgs[0].__class__.__name__ == "ToolMessage",
    f"got {[type(m).__name__ for m in tool_msgs]}",
)
import json as _json
payload = _json.loads(text_content(tool_msgs[0].content))
check(
    "case 16: ToolMessage payload reports committed variant id and rationale",
    payload.get("status") == "committed"
    and payload.get("variant_id") == "T09_v2"
    and payload.get("rationale") == "cleaner gradient",
    f"got {payload}",
)
check(
    "case 16: ToolMessage payload lists the dropped variants",
    payload.get("dropped_variants") == ["T09_v1"],
    f"got {payload.get('dropped_variants')}",
)

# Case 17: commit_variant tool — invalid id returns error ToolMessage with valid_ids list
paths = make_stub_jpgs(2)
state = make_state(
    variant_pool=[
        stub_variant(paths[0], "T09_v1"),
        stub_variant(paths[1], "T09_v2"),
    ],
)
cmd = commit_variant.invoke({
    "variant_id": "T09_v99",
    "state": state,
    "tool_call_id": "test_tcid_2",
})
err_msgs = cmd.update["messages"]
err_payload = _json.loads(text_content(err_msgs[0].content))
check(
    "case 17: invalid id → error ToolMessage with status=error",
    err_payload.get("status") == "error",
    f"got {err_payload}",
)
check(
    "case 17: error payload includes the list of valid ids",
    err_payload.get("valid_ids") == ["T09_v1", "T09_v2"],
    f"got {err_payload.get('valid_ids')}",
)
check(
    "case 17: invalid commit does NOT mutate paths/pool/visual_context",
    "paths" not in cmd.update
    and "variant_pool" not in cmd.update
    and "visual_context" not in cmd.update,
    f"got update keys {list(cmd.update.keys())}",
)

# Case 18: end-to-end with VLM autonomous on — current_image auto-projection
# replaces the old phase_carry mechanism.
reset_vlm_modes(hitl=False, auto=True, cap=8)
paths = make_stub_jpgs(2)
state = make_state(
    variant_pool=[
        stub_variant(paths[0], "T09_v1"),
        stub_variant(paths[1], "T09_v2"),
    ],
)
cmd = commit_variant.invoke({
    "variant_id": "T09_v1",
    "state": state,
    "tool_call_id": "test_tcid_3",
})
preview_dir = Path(paths[0]).parent / "previews"
preview_dir.mkdir(exist_ok=True)
(preview_dir / f"preview_{Path(paths[0]).stem}.jpg").write_bytes(_FAKE_JPG_BYTES)
# Apply the Command's update onto state (simulate LangGraph's reducer)
post_state = make_state(
    variant_pool=cmd.update["variant_pool"],
    visual_context=cmd.update["visual_context"],
    paths=cmd.update["paths"],
    dataset={"working_dir": str(Path(paths[0]).parent)},
)
messages = [SystemMessage(content="sys"), HumanMessage(content="continuing")]
out = _build_vlm_view(post_state, messages)
check(
    "case 18: post-commit, _build_vlm_view shows current_image",
    total_image_count(out) == 1,
    f"got {total_image_count(out)} images (expected 1 via current_image)",
)


# ── _format_variant_pool_for_prompt cases ───────────────────────────────────
#
# The agent reads variant_pool through its own system prompt (rebuilt every
# turn by agent_node). _format_variant_pool_for_prompt is the bridge from
# state.variant_pool → text the LLM can see.

print("\n_format_variant_pool_for_prompt cases\n" + "=" * 40)

# Case 19: empty pool → empty string (no section in prompt)
out_text = _format_variant_pool_for_prompt([])
check(
    "case 19: empty pool → empty string (omits section entirely)",
    out_text == "",
    f"got {out_text!r}",
)

# Case 20: pool with three variants → markdown section listing all ids and labels
v1 = stub_variant(make_stub_jpgs(1)[0], "T09_v1", label="gradient pass A")
v2 = stub_variant(make_stub_jpgs(1)[0], "T09_v2", label="gradient pass B")
v3 = stub_variant(make_stub_jpgs(1)[0], "T09_v3", label="gradient pass C")
out_text = _format_variant_pool_for_prompt([v1, v2, v3])
check(
    "case 20: section header is present",
    "## Active variant pool" in out_text,
    f"missing header in {out_text[:100]!r}",
)
check(
    "case 20: all three variant ids appear in the section",
    "T09_v1" in out_text and "T09_v2" in out_text and "T09_v3" in out_text,
    "missing variant ids",
)
check(
    "case 20: variant labels appear in the section",
    "gradient pass A" in out_text
    and "gradient pass B" in out_text
    and "gradient pass C" in out_text,
    "missing labels",
)
check(
    "case 20: section mentions commit_variant tool",
    "commit_variant" in out_text,
    "section should reference commit_variant for autonomous mode",
)

# Case 21: variant with metrics → metrics rendered inline
v_with_metrics = Variant(
    id="T09_v1",
    phase="linear",
    tool_name="remove_gradient",
    label="gradient pass",
    params={},
    file_path="/tmp/x.fits",
    preview_path=None,
    metrics={
        "gradient_magnitude": 0.0451,
        "snr_estimate": 12.3,
        "current_fwhm": 2.8,
        "irrelevant_metric": "ignored",
    },
    created_at="2026-04-09T00:00:00Z",
    rationale=None,
)
out_text = _format_variant_pool_for_prompt([v_with_metrics])
check(
    "case 21: known metrics rendered inline (gradient_magnitude)",
    "gradient_magnitude=0.045" in out_text,
    f"missing gradient_magnitude in {out_text!r}",
)
check(
    "case 21: known metrics rendered inline (snr_estimate)",
    "snr_estimate=12.300" in out_text,
    f"missing snr_estimate in {out_text!r}",
)
check(
    "case 21: unknown metric keys are not rendered",
    "irrelevant_metric" not in out_text,
    "unknown metric leaked into prompt",
)


# Cleanup runtime overrides (vlm_hitl is no longer toggleable — always True).
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
