#!/usr/bin/env python3
"""
Focused tests for the loop-aware decision substrate:
"approval = converged, not irrevocable".

Run from project root:
    uv run python scripts/test_decision_loop.py

Exit 0 = all checks pass.

What this exercises:
  1. auto_checkpoint records a set-once step anchor (pre-step image + space).
  2. commit_variant (autonomous) archives a DecisionRecord: chosen + ALL
     candidates + pre_step; re-asserts image_space; clears the live pool.
  3. HITL approval records the same record (mode-independent substrate).
  4. revisit_decision reopens the step: restores pre-step image + image_space,
     repopulates the pool from the candidates, clears the baseline, reopens the
     gate (when enabled). The record is preserved (append-only).
  5. revisit refuses cleanly on unknown handle / missing pre-step file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

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


def _variant(vid: str, path: str) -> dict:
    return {
        "id": vid, "phase": "linear", "tool_name": "remove_gradient",
        "label": f"{vid} label", "params": {}, "file_path": path,
        "preview_path": None, "metrics": {}, "image_space": "linear",
        "created_at": "", "rationale": None,
    }


def main() -> int:
    from muphrid.graph import review as review_ctl
    from muphrid.graph.nodes import auto_checkpoint, build_variant_promotion_update, promote_variant
    from muphrid.graph.state import ProcessingPhase
    from muphrid.tools.utility.commit_variant import commit_variant
    from muphrid.tools.utility.revisit_decision import revisit_decision

    wd = tempfile.mkdtemp(prefix="decision_test_")
    pre = Path(wd) / "pre.fit"; pre.write_bytes(b"x")
    v1 = Path(wd) / "v1.fit"; v1.write_bytes(b"x")
    v2 = Path(wd) / "v2.fit"; v2.write_bytes(b"x")
    pool = [_variant("remove_gradient_v1", str(v1)), _variant("remove_gradient_v2", str(v2))]

    # ── 1. auto_checkpoint step anchor (set-once) ───────────────────────────
    print("\n[1] auto_checkpoint records a set-once step anchor")
    ac_state = {
        "phase": ProcessingPhase.LINEAR,
        "paths": {"current_image": str(pre)},
        "metadata": {"image_space": "linear"},
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "remove_gradient", "args": {}, "id": "c1", "type": "tool_call"}
        ])],
    }
    out = auto_checkpoint(ac_state)
    anchors = out.get("metadata", {}).get("step_anchors", {})
    check("anchor captured pre-step image + space",
          anchors.get("linear:remove_gradient") == {"path": str(pre), "image_space": "linear"},
          f"anchors={anchors}")
    # set-once: existing anchor (pointing elsewhere) is not overwritten
    ac_state2 = {**ac_state, "metadata": {"image_space": "linear",
                 "step_anchors": {"linear:remove_gradient": {"path": "/old", "image_space": "linear"}}}}
    out2 = auto_checkpoint(ac_state2)
    check("set-once: existing anchor not overwritten",
          "step_anchors" not in out2.get("metadata", {}),
          f"got={out2.get('metadata', {}).get('step_anchors')}")

    # ── 2. commit_variant (autonomous) archives a decision ──────────────────
    print("\n[2] commit_variant archives a DecisionRecord (autonomous)")
    base_meta = {
        "image_space": "linear",
        "step_anchors": {"linear:remove_gradient": {"path": str(pre), "image_space": "linear"}},
    }
    auto_state = {
        "phase": ProcessingPhase.LINEAR,
        "paths": {"current_image": str(v2)},
        "metadata": dict(base_meta),
        "variant_pool": list(pool),
        "review_session": None,
        "visual_context": [],
    }
    cmd = commit_variant.func(variant_id="remove_gradient_v1", rationale="best flatness",
                              state=auto_state, tool_call_id="t")
    upd = cmd.update
    rec = upd.get("metadata", {}).get("decisions", {}).get("linear:remove_gradient")
    check("decision recorded", rec is not None)
    check("chosen variant captured",
          rec and rec["chosen"]["variant_id"] == "remove_gradient_v1"
          and rec["chosen"]["path"] == str(v1) and rec["chosen"]["image_space"] == "linear")
    check("ALL candidates preserved (alternatives not destroyed)",
          rec and {c["id"] for c in rec["candidates"]} == {"remove_gradient_v1", "remove_gradient_v2"})
    check("pre_step recorded from anchor",
          rec and rec["pre_step"] == {"path": str(pre), "image_space": "linear"})
    check("mode is autonomous", rec and rec["mode"] == "autonomous")
    check("rationale captured", rec and rec["rationale"] == "best flatness")
    check("image_space re-asserted on promotion", upd.get("metadata", {}).get("image_space") == "linear")
    check("paths is a delta (current_image only)", upd.get("paths") == {"current_image": str(v1)})
    check("live pool cleared", upd.get("variant_pool") == [])

    # ── 3. HITL approval records the same record (mode-independent) ──────────
    print("\n[3] HITL approval records the decision too (mode='hitl')")
    session = review_ctl.make_review_session(
        state={"phase": ProcessingPhase.LINEAR}, hitl_key="remove_gradient",
        tool_name="remove_gradient")
    hitl_state = {**auto_state, "review_session": session}
    hupd = promote_variant(hitl_state, "remove_gradient_v2", rationale="human pick")
    hrec = hupd.get("metadata", {}).get("decisions", {}).get("linear:remove_gradient")
    check("HITL approval records decision", hrec is not None)
    check("mode is hitl", hrec and hrec["mode"] == "hitl")
    check("HITL decision keeps both candidates",
          hrec and len(hrec["candidates"]) == 2)

    # ── 4. revisit_decision reopens the step ────────────────────────────────
    print("\n[4] revisit_decision restores pre-step + pool + reopens gate")
    revisit_state = {
        "phase": ProcessingPhase.LINEAR,
        "paths": {"current_image": str(v1)},   # we 'moved on' to the chosen one
        "metadata": {"image_space": "linear", "decisions": {"linear:remove_gradient": rec}},
        "variant_pool": [],                      # live pool cleared after commit
        "regression_warnings": [{"metric": "snr"}],
        "review_session": None,
    }
    rcmd = revisit_decision.func(decision="remove_gradient", state=revisit_state, tool_call_id="t")
    rupd = rcmd.update
    check("current_image restored to pre-step", rupd.get("paths") == {"current_image": str(pre)})
    check("image_space re-asserted from pre-step", rupd.get("metadata", {}).get("image_space") == "linear")
    check("pool repopulated from recorded candidates",
          {v["id"] for v in rupd.get("variant_pool", [])} == {"remove_gradient_v1", "remove_gradient_v2"})
    check("regression baseline cleared",
          rupd.get("regression_warnings") == [] and rupd.get("metadata", {}).get("last_analysis_snapshot") is None)
    check("gate reopened (remove_gradient HITL enabled)",
          isinstance(rupd.get("review_session"), dict) and rupd.get("active_hitl") is True)
    check("decision record preserved (append-only, revisit != undo)",
          "decisions" not in rupd.get("metadata", {}))  # revisit doesn't rewrite/delete it

    # ── 5. refusals ─────────────────────────────────────────────────────────
    print("\n[5] revisit refuses cleanly on bad input")
    bad = revisit_decision.func(decision="does_not_exist", state=revisit_state, tool_call_id="t")
    bp = json.loads(bad.update["messages"][0].content)
    check("unknown handle → error with available list",
          bp.get("status") == "error" and "linear:remove_gradient" in bp.get("available_decisions", []))

    rec_no_pre = dict(rec); rec_no_pre["pre_step"] = None
    state_no_pre = {**revisit_state, "metadata": {"image_space": "linear",
                    "decisions": {"linear:remove_gradient": rec_no_pre}}}
    nopre = revisit_decision.func(decision="remove_gradient", state=state_no_pre, tool_call_id="t")
    npp = json.loads(nopre.update["messages"][0].content)
    check("missing pre-step → refuses (no fallback)", npp.get("status") == "error")

    print()
    if _failures:
        print(f"FAIL — {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f" - {f}")
        return 1
    print("All decision-loop tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
