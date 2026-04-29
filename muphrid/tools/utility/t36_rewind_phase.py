"""
T36 — rewind_phase

Cross-phase backtrack. Sets state.phase to a target phase the pipeline has
already entered and restores the working state captured at that boundary.

Snapshot source: metadata.phase_checkpoints[<target_phase>], written by
advance_phase as the pipeline transitioned into that phase. Each entry is
a PhaseSnapshot — see graph/state.py for the captured field set.

Constraints:
  - target_phase must have an entry in metadata.phase_checkpoints (i.e.
    the pipeline must have entered that phase at least once)
  - target_phase must be earlier than the current phase in pipeline order
  - metadata.phase_rewind_counts[target_phase] must be < 1; the first
    rewind to a phase is the safety net, a second is refused

Side effects on success:
  - state.phase ← target_phase
  - state.paths, state.metrics restored from the snapshot
  - state.metadata fields restored from the snapshot (excluding the
    snapshot store itself), with last_analysis_snapshot cleared and
    phase_rewind_counts[target_phase] incremented
  - state.regression_warnings, state.variant_pool, state.visual_context
    restored from the snapshot
  - state.active_hitl ← False (any in-flight HITL gate is dissolved by
    the phase change)
  - A REWIND section is appended to processing_log.md
  - A structured ToolMessage records the jump, including hitl_terminated
    so a UI client can detect a mid-conversation phase change

The narrative — messages and processing_report — is intentionally NOT
restored. The abandoned-phase reasoning stays in context so the agent can
see what was tried, and so a later reviewer can follow the decision.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState, PhaseSnapshot, ProcessingPhase, Replace

logger = logging.getLogger(__name__)


# ── Phase ordering (mirrors t30_advance_phase) ────────────────────────────────

_PHASE_ORDER = [
    ProcessingPhase.INGEST,
    ProcessingPhase.CALIBRATION,
    ProcessingPhase.REGISTRATION,
    ProcessingPhase.ANALYSIS,
    ProcessingPhase.STACKING,
    ProcessingPhase.LINEAR,
    ProcessingPhase.STRETCH,
    ProcessingPhase.NONLINEAR,
    ProcessingPhase.EXPORT,
    ProcessingPhase.COMPLETE,
]

_PHASE_INDEX = {p: i for i, p in enumerate(_PHASE_ORDER)}


# ── Pydantic input schema ──────────────────────────────────────────────────────

class RewindPhaseInput(BaseModel):
    target_phase: str = Field(
        description=(
            "ProcessingPhase value to rewind to. Must be one of: "
            "ingest, calibration, registration, analysis, stacking, linear, "
            "stretch, nonlinear, export. Must be earlier than the current "
            "phase and must have a captured snapshot in "
            "metadata.phase_checkpoints (i.e. the pipeline has entered it "
            "at least once)."
        ),
    )
    reason: str = Field(
        description=(
            "Brief explanation of why the rewind is happening. Captured into "
            "processing_log.md alongside the abandoned phase's existing entry "
            "as a durable record of what triggered the backtrack."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_target_phase(value: str) -> ProcessingPhase | None:
    """Map a string value to a ProcessingPhase, or None if invalid."""
    try:
        return ProcessingPhase(value)
    except ValueError:
        return None


def _get_rewind_reasoning(messages: list, tool_call_id: str | None) -> str:
    """
    Return the text content of the AIMessage that issued the current
    rewind_phase call. Mirrors t30's _get_advance_reasoning so the agent's
    full reasoning paragraph (not just the structured `reason` arg) is
    preserved into processing_log.md.
    """
    if not messages or not tool_call_id:
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in (msg.tool_calls or []):
            if tc.get("id") == tool_call_id and tc.get("name") == "rewind_phase":
                return str(msg.content or "").strip()
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if any(
            tc.get("name") == "rewind_phase" for tc in (msg.tool_calls or [])
        ):
            return str(msg.content or "").strip()
    return ""


def _write_rewind_log(
    working_dir: str,
    *,
    from_phase: str,
    target_phase: str,
    reason: str,
    rewind_reasoning: str,
    snapshot_captured_at: str,
    snapshot_captured_from: str,
) -> None:
    """Append a REWIND section to processing_log.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## REWIND: {from_phase.upper()} → {target_phase.upper()}",
        f"*{now}*",
        "",
        f"> **Reason:** {reason}",
        "",
        "### Snapshot Source",
        "",
        f"- Captured at: {snapshot_captured_at}",
        f"- End of phase: {snapshot_captured_from}",
        "",
    ]
    if rewind_reasoning:
        lines.append("### Rewind-Time Reasoning")
        lines.append("")
        for para in rewind_reasoning.splitlines():
            lines.append(f"> {para}" if para.strip() else ">")
        lines.append("")
    lines += ["---", ""]

    log_path = Path(working_dir) / "processing_log.md"
    if not log_path.exists():
        log_path.write_text("# Muphrid Processing Log\n\n")
    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=RewindPhaseInput)
def rewind_phase(
    target_phase: str,
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Restore working state from a captured phase snapshot and set the
    pipeline phase to `target_phase`.

    Inputs validated before any state change. Each precondition that fails
    returns a ToolMessage describing what was checked and what was found —
    no partial state mutation occurs on a rejected rewind.

    Preconditions:
      - target_phase resolves to a ProcessingPhase value
      - target_phase is earlier in pipeline order than the current phase
      - metadata.phase_checkpoints contains an entry for target_phase
      - metadata.phase_rewind_counts[target_phase] < 1

    On success the tool returns a structured ToolMessage with:
      - previous_phase, target_phase, reason
      - snapshot_captured_at, snapshot_captured_from_phase (snapshot age)
      - hitl_terminated (True when the rewind happened mid-HITL gate, so
        UI clients can detect the conversation-context shift)
      - phase_rewind_counts (post-increment counter dict)
    """
    current = state.get("phase", ProcessingPhase.INGEST)
    metadata = state.get("metadata") or {}
    phase_checkpoints: dict = metadata.get("phase_checkpoints") or {}
    rewind_counts: dict = metadata.get("phase_rewind_counts") or {}

    # ── Validation ─────────────────────────────────────────────────
    target = _resolve_target_phase(target_phase)
    if target is None:
        valid = ", ".join(p.value for p in _PHASE_ORDER if p != ProcessingPhase.COMPLETE)
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"rewind_phase: target_phase '{target_phase}' is not a "
                    f"valid ProcessingPhase. Valid values: {valid}."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if target == current:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"rewind_phase: target_phase '{target.value}' is the current "
                    f"phase. Use restore_checkpoint for within-phase rollback."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if _PHASE_INDEX.get(target, 99) >= _PHASE_INDEX.get(current, 0):
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"rewind_phase: target_phase '{target.value}' is not earlier "
                    f"than the current phase '{current.value}'. rewind_phase "
                    f"only moves backward — to advance, call advance_phase."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if target.value not in phase_checkpoints:
        captured = sorted(phase_checkpoints.keys()) or ["(none)"]
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"rewind_phase: no snapshot captured for phase "
                    f"'{target.value}'. Snapshots are written by advance_phase "
                    f"as the pipeline enters each phase, so a phase the "
                    f"pipeline never reached has no entry to restore. "
                    f"Available snapshots: {', '.join(captured)}."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if rewind_counts.get(target.value, 0) >= 1:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"rewind_phase: phase '{target.value}' has already been "
                    f"rewound to once. The first rewind is the safety net; a "
                    f"second would suggest a structural problem the framework "
                    f"can't fix on its own. Surface the situation in your next "
                    f"message — describe what's not converging — so the human "
                    f"can decide whether to flag a dataset issue or accept the "
                    f"current state."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    snapshot: PhaseSnapshot = phase_checkpoints[target.value]

    # ── Build restored state ───────────────────────────────────────
    # Deep-copy the snapshot fields so subsequent mutations don't bleed back
    # into the stored snapshot (it stays available for diagnostic inspection).
    restored_paths = copy.deepcopy(snapshot.get("paths") or {})
    restored_metrics = copy.deepcopy(snapshot.get("metrics") or {})
    restored_metadata = copy.deepcopy(snapshot.get("metadata") or {})
    restored_warnings = copy.deepcopy(snapshot.get("regression_warnings") or [])
    restored_variants = copy.deepcopy(snapshot.get("variant_pool") or [])
    restored_visuals = copy.deepcopy(snapshot.get("visual_context") or [])

    # Note on metadata reducer: AstroState declares metadata as
    # Annotated[Metadata, _merge_dicts]. The reducer deep-merges new into
    # old, so the snapshot's metadata fields overwrite current scalars and
    # nested dicts get recursive-merged. Concretely:
    #   - metadata.checkpoints (within-phase named bookmarks) deep-merges:
    #     bookmarks created in the abandoned phase will linger as pointers,
    #     but they fail gracefully on restore_checkpoint if the file is
    #     missing or remain valid if the FITS still exists on disk.
    #   - phase_checkpoints (the snapshot store) is intentionally OMITTED
    #     from the snapshot's metadata dict, so the live store survives.
    #   - last_analysis_snapshot is explicitly cleared below so the next
    #     analyze_image call establishes a fresh baseline post-rewind.
    #   - phase_rewind_counts is updated below.
    new_rewind_counts = dict(rewind_counts)
    new_rewind_counts[target.value] = new_rewind_counts.get(target.value, 0) + 1
    restored_metadata["last_analysis_snapshot"] = None
    restored_metadata["phase_rewind_counts"] = new_rewind_counts

    # ── Side effects: log + structured message ─────────────────────
    rewind_reasoning = _get_rewind_reasoning(
        state.get("messages", []), tool_call_id
    )
    working_dir = state.get("dataset", {}).get("working_dir")
    versioned_audit_reports: list[str] = []
    if working_dir:
        try:
            _write_rewind_log(
                working_dir,
                from_phase=current.value,
                target_phase=target.value,
                reason=reason,
                rewind_reasoning=rewind_reasoning,
                snapshot_captured_at=str(snapshot.get("captured_at", "unknown")),
                snapshot_captured_from=str(snapshot.get("captured_from_phase", "unknown")),
            )
        except Exception as e:
            logger.warning(f"Rewind log write failed (non-fatal): {e}")

        # Version the per-phase audit reports for the target phase and any
        # phases past it: each rename preserves the abandoned attempt as
        # NN_<phase>.v<N>.md so the new attempt can write a fresh
        # NN_<phase>.md without losing the audit trail of what didn't work.
        try:
            from muphrid.tools.utility.t30_advance_phase import (
                version_existing_audit_reports,
            )
            versioned_audit_reports = version_existing_audit_reports(
                working_dir, target.value
            )
            if versioned_audit_reports:
                logger.info(
                    f"Versioned audit reports on rewind: "
                    f"{', '.join(versioned_audit_reports)}"
                )
        except Exception as e:
            logger.warning(f"Audit report versioning failed (non-fatal): {e}")

    from muphrid.graph import review as review_ctl

    was_in_hitl = bool(review_ctl.active_review_session(state))
    result: dict = {
        "previous_phase": current.value,
        "target_phase": target.value,
        "reason": reason,
        "snapshot_captured_at": snapshot.get("captured_at"),
        "snapshot_captured_from_phase": snapshot.get("captured_from_phase"),
        "phase_rewind_counts": new_rewind_counts,
        "hitl_terminated": was_in_hitl,
        "versioned_audit_reports": versioned_audit_reports,
    }

    logger.info(
        f"rewind_phase: {current.value} → {target.value} | reason: {reason}"
        + (f" | hitl gate dissolved" if was_in_hitl else "")
    )

    return Command(update={
        "phase": target,
        # paths uses _merge_dicts (not Replace-aware): restored_paths
        # has every PathState key explicitly set, including Nones for
        # fields that should clear, so a deep-merge against current
        # state effectively replaces every documented field.
        "paths": restored_paths,
        # metrics uses _dict_merge_or_replace: wrap in Replace so post-
        # snapshot keys (e.g. analyze_image results from the abandoned
        # phases) are dropped, not merged. Without Replace, the new
        # merge reducer would leave them lingering.
        "metrics": Replace(restored_metrics),
        # metadata uses _merge_dicts: restored_metadata sets the keys
        # we want to roll back. Keys NOT in the snapshot (e.g. user
        # bookmarks) intentionally linger across rewind so the agent
        # can still address them on the next attempt.
        "metadata": restored_metadata,
        # The list-valued fields use plain replace semantics (no
        # reducer). Direct assignment fully replaces.
        "regression_warnings": restored_warnings,
        "variant_pool": restored_variants,
        "visual_context": restored_visuals,
        # Any in-flight HITL gate is dissolved by the phase change. Clear
        # the flag so the agent's next text response routes normally.
        "active_hitl": False,
        "review_session": review_ctl.close_review_session(
            state.get("review_session"),
            reason="phase_rewound",
        ),
        "messages": [ToolMessage(
            content=json.dumps(result, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    })
