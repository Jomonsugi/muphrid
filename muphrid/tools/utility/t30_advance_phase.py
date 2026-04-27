"""
T30 — advance_phase

Explicit phase advancement tool. The agent calls this when it has completed
all work for the current phase and is ready to move on. This is the ONLY
way to advance phases — text-only responses do NOT advance the phase.

This replaces the implicit "no tool_calls = advance" routing that caused
the agent to blast through phases when it was stuck and trying to
communicate with the human.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from muphrid.graph.state import AstroState, PhaseSnapshot, ProcessingPhase, RegressionWarning

logger = logging.getLogger(__name__)

# Phase ordering — same as nodes.py but needed here for the tool
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

_NEXT_PHASE = {
    _PHASE_ORDER[i]: _PHASE_ORDER[i + 1]
    for i in range(len(_PHASE_ORDER) - 1)
}


# ── Shared call scanner ────────────────────────────────────────────────────────

def _scan_phase_calls(messages: list) -> list[dict]:
    """
    Extract all tool calls made in the current phase.

    The current phase starts after the last advance_phase ToolMessage.
    advance_phase itself is excluded — it's the phase terminator, not a
    processing step. Injected parameters (tool_call_id, state) are stripped.

    Returns a list of dicts: {name, args, reasoning}
    """
    phase_start = 0
    for i, msg in enumerate(messages):
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", None) == "advance_phase"
            and "Cannot advance" not in str(msg.content)
        ):
            # Only successful advance_phase calls reset the boundary.
            # Blocked calls return "Cannot advance..." — if these reset
            # the boundary, tools called before the blocked attempt become
            # invisible to the requirement checker, forcing the agent to
            # redo work it already completed.
            phase_start = i + 1
    phase_messages = messages[phase_start:]

    calls = []
    for msg in phase_messages:
        if not (isinstance(msg, AIMessage) and msg.tool_calls):
            continue
        reasoning = str(msg.content).strip() if msg.content else ""
        for tc in msg.tool_calls:
            if tc["name"] == "advance_phase":
                continue
            args = {k: v for k, v in tc.get("args", {}).items()
                    if k not in ("tool_call_id", "state")}
            calls.append({"name": tc["name"], "args": args, "reasoning": reasoning})
    return calls


# ── Phase requirements ─────────────────────────────────────────────────────────
# Only hard requirements are listed — tools where skipping is unambiguous
# malpractice regardless of image quality or target type.
#
# Optional tools (deconvolution, noise reduction, background extraction,
# star removal, etc.) are NOT listed here. Their use is left to agent judgment
# and HITL review.

def _check_phase_requirements(
    phase: ProcessingPhase,
    calls: list[dict],
    state: AstroState,
) -> list[str]:
    """
    Return a list of human-readable violation strings.
    Empty list = all requirements met, phase may advance.
    """
    violations: list[str] = []
    called_tools = {c["name"] for c in calls}
    files = state.get("dataset", {}).get("files", {})
    meta = state.get("metadata", {})

    if phase == ProcessingPhase.CALIBRATION:
        # build_masters is required for each frame type that exists
        for frame_type, plural in (("bias", "biases"), ("dark", "darks"), ("flat", "flats")):
            if files.get(plural):
                called = any(
                    c["name"] == "build_masters"
                    and c["args"].get("file_type") == frame_type
                    for c in calls
                )
                if not called:
                    violations.append(
                        f"`build_masters(file_type='{frame_type}')` was not called, "
                        f"but {len(files[plural])} {frame_type} frame(s) are available. "
                        f"The master {frame_type} must be built before lights can be calibrated."
                    )

        # convert_sequence is required to create the FITSEQ from raw lights
        if "convert_sequence" not in called_tools:
            violations.append(
                "`convert_sequence` was not called. The raw light frames must be "
                "converted to a Siril FITSEQ sequence before calibration can be applied."
            )

        # calibrate is required if any calibration frames exist
        has_calib = any(files.get(k) for k in ("biases", "darks", "flats"))
        if has_calib and "calibrate" not in called_tools:
            violations.append(
                "`calibrate` was not called. Calibration frames exist but were never "
                "applied to the lights. Skipping calibration leaves thermal noise, "
                "bias pedestal, and vignetting uncorrected in the stacked image."
            )

    elif phase == ProcessingPhase.REGISTRATION:
        if "siril_register" not in called_tools:
            violations.append(
                "`siril_register` was not called. Frame registration is mandatory — "
                "unregistered frames will stack with misaligned stars, producing "
                "trails and smearing instead of a sharp combined image."
            )

    elif phase == ProcessingPhase.STACKING:
        if "siril_stack" not in called_tools:
            violations.append(
                "`siril_stack` was not called. Stacking is the core integration step "
                "that combines all registered frames into the master light. "
                "The linear processing phase cannot begin without a stacked image."
            )

    elif phase == ProcessingPhase.LINEAR:
        if "analyze_image" not in called_tools:
            violations.append(
                "`analyze_image` was not called in the linear phase. "
                "Baseline measurements (FWHM, noise, background level, channel balance) "
                "are required before stretching — without them you have no reference to "
                "evaluate whether the stretch result is good or whether it degraded the data."
            )
        if meta.get("is_color") and "color_calibrate" not in called_tools:
            violations.append(
                "`color_calibrate` was not called on this color image. "
                "Color calibration must happen in linear space before stretching. "
                "After a non-linear stretch, channel ratios are permanently distorted "
                "and accurate color correction is no longer possible."
            )

    elif phase == ProcessingPhase.STRETCH:
        if "stretch_image" not in called_tools:
            violations.append(
                "`stretch_image` was not called. The stretch phase exists to apply "
                "the non-linear transfer function that makes faint signal visible. "
                "Without it the image is still linear and all nonlinear tools "
                "(curves, saturation, star removal) will produce incorrect results."
            )
        if "analyze_image" not in called_tools:
            violations.append(
                "`analyze_image` was not called after stretching. "
                "You need to evaluate the stretch result — shadow clipping, highlight "
                "preservation, dynamic range, histogram shape — before committing to "
                "nonlinear processing. A poorly stretched base cannot be corrected later."
            )

    return violations


# ── Artifact post-condition gate ──────────────────────────────────────────────
# Each phase is required to LEAVE an artifact on disk. A tool call being
# present in the message history is not sufficient: the M42 Sonnet run
# called siril_register ten times, every call failed with a cfitsio error,
# yet advance_phase accepted the transition because the *call* appeared in
# the phase's message window. The agent reasoned past the failure in text
# ("Registration math complete... Siril stacking can apply transforms
# directly from the sequence metadata") and moved forward onto a sequence
# file that did not exist on disk. The artifact check below is the guard
# that turns "I claimed success" into "the file exists and is readable".
#
# Each entry in _PHASE_ARTIFACTS is a callable that takes the state and
# returns a list of {path, role, hint} dicts. role is a short label the
# error message uses to explain what's missing; hint is the remediation
# the agent should try next. If any path is missing or unreadable, the
# phase cannot advance.

def _phase_artifacts(phase: ProcessingPhase, state: AstroState) -> list[dict]:
    """
    Return the list of on-disk artifacts the given phase is required to
    produce. Each artifact is a dict:
        {"path": Path | None, "role": str, "hint": str}
    A None path means the state field that should hold the artifact name
    was never populated — equivalent to "the tool ran but left no output".
    """
    paths = state.get("paths", {}) or {}
    dataset = state.get("dataset", {}) or {}
    wdir_str = dataset.get("working_dir")
    wdir = Path(wdir_str) if wdir_str else None
    files = dataset.get("files", {}) or {}

    artifacts: list[dict] = []

    if phase == ProcessingPhase.CALIBRATION:
        # Each master requested must exist as a FITS file.
        masters = paths.get("masters", {}) or {}
        for frame_type, plural in (("bias", "biases"), ("dark", "darks"), ("flat", "flats")):
            if files.get(plural):
                mp = masters.get(frame_type)
                artifacts.append({
                    "path": Path(mp) if mp else None,
                    "role": f"master {frame_type}",
                    "hint": (
                        f"Call build_masters(file_type='{frame_type}') and verify its "
                        f"ToolMessage reports a written path. If the call errored, read "
                        f"the error output — do not retry with different parameters "
                        f"without first understanding the cause."
                    ),
                })

        # The calibrated sequence .seq file must exist (even if no masters
        # were applied, convert_sequence + calibrate always leaves one).
        calibrated = paths.get("calibrated_sequence")
        artifacts.append({
            "path": (wdir / f"{calibrated}.seq") if (wdir and calibrated) else None,
            "role": "calibrated sequence (.seq)",
            "hint": (
                "The calibrated .seq file on disk is the hand-off to registration. "
                "Call calibrate if you have not; if you called it and it errored, "
                "check the Siril output for the failure reason before retrying."
            ),
        })

    elif phase == ProcessingPhase.REGISTRATION:
        registered = paths.get("registered_sequence")
        artifacts.append({
            "path": (wdir / f"{registered}.seq") if (wdir and registered) else None,
            "role": "registered sequence (.seq)",
            "hint": (
                "siril_register must complete successfully and leave a .seq file "
                "on disk — that file is the only hand-off to stacking. If your last "
                "registration attempt returned an error (cfitsio, HDU, path-related), "
                "the .seq file was never written no matter how many times the tool "
                "was invoked. Do NOT advance_phase while the artifact is missing; "
                "fix the precondition (working dir, corrupted FITS, path encoding) "
                "and re-run registration."
            ),
        })

    elif phase == ProcessingPhase.ANALYSIS:
        # Frame-selection is a pure-metrics phase — the only durable artifact
        # is the selected_frames list on state. No disk artifact required.
        pass

    elif phase == ProcessingPhase.STACKING:
        # After stacking, current_image must point at a readable FITS.
        current = paths.get("current_image")
        artifacts.append({
            "path": Path(current) if current else None,
            "role": "stacked master light (current_image FITS)",
            "hint": (
                "siril_stack must leave a stacked FITS on disk and promote it to "
                "paths.current_image. If the stack tool errored or produced no file, "
                "the linear phase has nothing to work from. Re-run siril_stack with "
                "diagnostics turned on and inspect the failure before advancing."
            ),
        })

    elif phase in (ProcessingPhase.LINEAR, ProcessingPhase.STRETCH, ProcessingPhase.NONLINEAR):
        # Every post-stack phase has a current_image pointer that must exist.
        # This catches accidental pointer-loss (a tool setting current_image
        # to a non-existent file) across the entire second half of the pipeline.
        current = paths.get("current_image")
        artifacts.append({
            "path": Path(current) if current else None,
            "role": "current_image FITS",
            "hint": (
                "paths.current_image must point at a readable FITS for downstream "
                "tools. If it is missing, the previous phase left the pipeline in "
                "an inconsistent state — restore from a checkpoint or re-run the "
                "last tool that owned the current_image hand-off."
            ),
        })

    return artifacts


def _check_phase_artifacts(phase: ProcessingPhase, state: AstroState) -> list[str]:
    """
    Validate that each declared artifact for this phase exists and is readable.
    Returns a list of violation strings. Empty list = all artifacts present.
    """
    violations: list[str] = []
    for art in _phase_artifacts(phase, state):
        role = art["role"]
        hint = art["hint"]
        path: Path | None = art["path"]

        if path is None:
            violations.append(
                f"{role} is missing: the state field that should hold the path "
                f"was never populated (tool likely errored or was never called). "
                f"{hint}"
            )
            continue

        try:
            exists = path.exists()
        except OSError as e:
            # Non-ASCII paths, permission errors, etc. all reach here — we
            # want to report the actual OS error so the agent can act on it.
            violations.append(
                f"{role} at '{path}' could not be stat'd ({e.__class__.__name__}: {e}). "
                f"This usually indicates a malformed path (encoding issue) or a "
                f"permissions problem. {hint}"
            )
            continue

        if not exists:
            violations.append(
                f"{role} expected at '{path}' but file does not exist. {hint}"
            )
            continue

        # Basic readability check — zero-byte files and permission denials
        # fail here. This is deliberately cheap; deep validity (valid FITS
        # header, non-empty sequence) is not advance_phase's job.
        try:
            if path.is_file() and path.stat().st_size == 0:
                violations.append(
                    f"{role} at '{path}' exists but is zero bytes. "
                    f"The tool that produced it likely aborted mid-write. {hint}"
                )
        except OSError as e:
            violations.append(
                f"{role} at '{path}' could not be read ({e.__class__.__name__}: {e}). {hint}"
            )

    return violations


# ── Phase snapshot capture ─────────────────────────────────────────────────────

# Metadata fields excluded from the captured snapshot:
#   - phase_checkpoints     prevents recursive accumulation across rewinds
#   - last_analysis_snapshot transient analyze_image baseline; cleared at every
#                            phase boundary by design and re-established by
#                            the next analyze_image call
_SNAPSHOT_METADATA_EXCLUDE = {"phase_checkpoints", "last_analysis_snapshot"}


def _build_phase_snapshot(state: AstroState, captured_from_phase: str) -> PhaseSnapshot:
    """
    Capture the working-state slice rewind_phase will restore.

    See PhaseSnapshot for the captured field set and rationale. The
    snapshot is keyed in metadata.phase_checkpoints by the NEW phase's
    value (the phase being entered at this advance), so a later
    rewind_phase(target_phase=X) can read phase_checkpoints[X] to get
    the state at the start of phase X.

    Pure: deep-copies the captured fields so subsequent state mutations
    don't bleed into the snapshot.
    """
    import copy

    paths_snap = copy.deepcopy(state.get("paths") or {})
    metrics_snap = copy.deepcopy(state.get("metrics") or {})
    metadata_full = state.get("metadata") or {}
    metadata_snap = copy.deepcopy({
        k: v for k, v in metadata_full.items() if k not in _SNAPSHOT_METADATA_EXCLUDE
    })
    regression_snap = copy.deepcopy(state.get("regression_warnings") or [])
    variants_snap = copy.deepcopy(state.get("variant_pool") or [])
    visual_snap = copy.deepcopy(state.get("visual_context") or [])

    return PhaseSnapshot(
        paths=paths_snap,
        metrics=metrics_snap,
        metadata=metadata_snap,
        regression_warnings=regression_snap,
        variant_pool=variants_snap,
        visual_context=visual_snap,
        captured_at=datetime.now(timezone.utc).isoformat(),
        captured_from_phase=captured_from_phase,
    )


# ── Advance-reasoning extraction ───────────────────────────────────────────────

def _get_advance_reasoning(messages: list, tool_call_id: str | None) -> str:
    """
    Return the text content of the AIMessage that issued the current
    advance_phase call.

    This is the agent's free-form reasoning paragraph written immediately
    before deciding to advance — the passage that would (per the prompt
    contract) address any outstanding regression warnings or summarize why
    the phase's result is acceptable. Captured into the phase report so the
    rationale for each transition survives the conversation window.

    Returns an empty string if no matching AIMessage is found, which is
    the expected case for backward compatibility in tests that inject
    ToolMessages directly.
    """
    if not messages or not tool_call_id:
        return ""
    # Search from the tail — the advance_phase call is always the most
    # recent tool_call at the moment this tool executes.
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in (msg.tool_calls or []):
            if tc.get("id") == tool_call_id and tc.get("name") == "advance_phase":
                return str(msg.content or "").strip()
    # Fallback: the last AIMessage with any advance_phase tool_call.
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if any(
            tc.get("name") == "advance_phase" for tc in (msg.tool_calls or [])
        ):
            return str(msg.content or "").strip()
    return ""


# ── Phase report writer ────────────────────────────────────────────────────────

def _fmt_value(v) -> str:
    if isinstance(v, str) and len(v) > 80:
        return f"{v[:77]}..."
    return str(v)


def _fmt_args(args: dict, indent: str = "") -> list[str]:
    lines = []
    for k, v in args.items():
        if isinstance(v, dict):
            lines.append(f"{indent}- **{k}**:")
            lines.extend(_fmt_args(v, indent + "  "))
        elif isinstance(v, list):
            lines.append(f"{indent}- **{k}**:")
            for item in v:
                lines.append(f"{indent}  - {_fmt_value(item)}")
        elif v is None:
            pass  # skip null args
        else:
            lines.append(f"{indent}- **{k}**: `{_fmt_value(v)}`")
    return lines


def _write_phase_report(
    phase: str,
    calls: list[dict],
    working_dir: str,
    reason: str,
    *,
    outstanding_warnings: list[RegressionWarning] | None = None,
    advance_reasoning: str = "",
) -> None:
    """Append a human-readable phase summary to processing_log.md.

    outstanding_warnings and advance_reasoning are the Task #45 additions —
    they preserve the context at the moment of transition so later review
    can tell whether the agent knowingly accepted a regression or missed
    it. No judgment is written; the sections appear verbatim.
    """
    tool_counts = Counter(c["name"] for c in calls)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"## {phase.upper()}",
        f"*{now} · {len(calls)} tool call{'s' if len(calls) != 1 else ''} · "
        f"{len(tool_counts)} unique tool{'s' if len(tool_counts) != 1 else ''}*",
        "",
        f"> **Reason advanced:** {reason}",
        "",
    ]

    # Outstanding regression warnings at advance time.
    if outstanding_warnings:
        lines.append("### Outstanding Regression Warnings at Advance")
        lines.append("")
        for w in outstanding_warnings:
            summary = w.get("summary", "")
            origin = w.get("phase_origin", "")
            origin_str = f" (from phase: {origin})" if origin and origin != phase else ""
            lines.append(f"- {summary}{origin_str}")
        lines.append("")

    # Agent's final reasoning text — the paragraph accompanying the
    # advance_phase tool call. Captures how the agent reasoned about any
    # outstanding warnings or phase-complete judgment.
    if advance_reasoning:
        lines.append("### Advance-Time Reasoning")
        lines.append("")
        for para in advance_reasoning.splitlines():
            lines.append(f"> {para}" if para.strip() else ">")
        lines.append("")

    if tool_counts:
        lines += [
            "### Tool Summary",
            "",
            "| Tool | Calls |",
            "|------|------:|",
        ]
        for t, count in sorted(tool_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| `{t}` | {count} |")
        lines.append("")

    if calls:
        lines.append("### Call Sequence")
        lines.append("")
        for i, call in enumerate(calls, 1):
            lines.append(f"**{i}. `{call['name']}`**")
            if call["reasoning"]:
                for line in call["reasoning"].splitlines():
                    lines.append(f"> {line}" if line.strip() else ">")
            arg_lines = _fmt_args(call["args"])
            if arg_lines:
                lines.extend(arg_lines)
            else:
                lines.append("*(no arguments)*")
            lines.append("")

    lines += ["---", ""]

    log_path = Path(working_dir) / "processing_log.md"
    if not log_path.exists():
        log_path.write_text("# Muphrid Processing Log\n\n")

    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool
def advance_phase(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Advance to the next processing phase.

    Call this when you have completed ALL work for the current phase and the
    data is ready for the next stage. This is the ONLY way to move forward
    in the pipeline. Do NOT call this if you are stuck or need human input —
    just explain the situation in text and the human will respond.

    Args:
        reason: Brief explanation of why this phase is complete and the data
            is ready for the next stage. E.g. "All three master calibration
            frames built with acceptable quality diagnostics."
    """
    current = state.get("phase", ProcessingPhase.INGEST)
    next_phase = _NEXT_PHASE.get(current, ProcessingPhase.COMPLETE)

    # Guard: export_final MUST be called before the pipeline can reach COMPLETE.
    if next_phase == ProcessingPhase.COMPLETE:
        if not state.get("metadata", {}).get("export_done"):
            msg = (
                "Cannot advance to COMPLETE: export_final has not been called. "
                "You must call export_final to produce the distribution-ready output "
                "files before the pipeline can finish. Call export_final now."
            )
            logger.warning(f"advance_phase BLOCKED: {current.value} → COMPLETE — export_done not set")
            return Command(update={
                "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            })

    # Scan calls made in this phase (shared by requirement check + report writer)
    calls = _scan_phase_calls(state.get("messages", []))

    # Guard: phase-specific required tools (call-trace check)
    violations = _check_phase_requirements(current, calls, state)
    if violations:
        bullet_list = "\n".join(f"  • {v}" for v in violations)
        msg = (
            f"Cannot advance from {current.value.upper()} — the following required "
            f"steps were not completed:\n\n{bullet_list}\n\n"
            f"Complete these steps before calling advance_phase again."
        )
        logger.warning(
            f"advance_phase BLOCKED: {current.value} → {next_phase.value} — "
            f"{len(violations)} requirement(s) unmet"
        )
        return Command(update={
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        })

    # Guard: artifact post-condition check (disk-truth). The call-trace check
    # above confirms the right tools were INVOKED; this check confirms they
    # LEFT the expected artifacts on disk. Without it, an agent can call
    # siril_register ten times with every call failing and still advance the
    # phase on the strength of "siril_register was called" alone. Disk
    # artifacts are the contract between phases — if they don't exist, the
    # next phase will blow up in a harder-to-diagnose way later.
    artifact_violations = _check_phase_artifacts(current, state)
    if artifact_violations:
        bullet_list = "\n".join(f"  • {v}" for v in artifact_violations)
        msg = (
            f"Cannot advance from {current.value.upper()} — required artifacts are "
            f"not present on disk:\n\n{bullet_list}\n\n"
            f"A tool call in the message history is not sufficient — the pipeline "
            f"only moves forward when each phase has produced its durable output. "
            f"Re-run the phase's producing tool(s), read the ACTUAL error output "
            f"from the last failing attempt, and fix the precondition before "
            f"calling advance_phase again. Do not reason around a missing artifact."
        )
        logger.warning(
            f"advance_phase BLOCKED: {current.value} → {next_phase.value} — "
            f"{len(artifact_violations)} artifact(s) missing"
        )
        return Command(update={
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        })

    logger.info(f"advance_phase tool: {current.value} → {next_phase.value} | reason: {reason}")

    # Snapshot of warnings outstanding at the moment of transition. These
    # will be captured into the phase log and then cleared — baselines and
    # warnings do not carry across phases because different metrics become
    # meaningful in different phases.
    outstanding_warnings: list[RegressionWarning] = list(
        state.get("regression_warnings") or []
    )

    # Agent's natural-language reasoning paragraph that accompanied the
    # advance_phase call. Captured passively — the prompt asks the agent to
    # address outstanding warnings in text before advancing; this pulls
    # that text into the durable record without a forcing function.
    advance_reasoning = _get_advance_reasoning(
        state.get("messages", []), tool_call_id
    )

    working_dir = state.get("dataset", {}).get("working_dir")
    if working_dir:
        try:
            _write_phase_report(
                phase=current.value,
                calls=calls,
                working_dir=working_dir,
                reason=reason,
                outstanding_warnings=outstanding_warnings,
                advance_reasoning=advance_reasoning,
            )
        except Exception as e:
            logger.warning(f"Phase report write failed (non-fatal): {e}")

    result: dict = {
        "previous_phase": current.value,
        "new_phase": next_phase.value,
        "reason": reason,
    }
    # Echo the warnings carried into the transition so the ToolMessage the
    # agent sees after advancing includes a compact record of what was
    # outstanding (and thus what just got cleared from state).
    if outstanding_warnings:
        result["outstanding_regression_warnings_at_advance"] = [
            w.get("summary", "") for w in outstanding_warnings
        ]

    # Phase boundary snapshot for rewind_phase. Capture the working state
    # AS IT IS RIGHT NOW (= the end of the prior phase, equivalently the
    # start of next_phase) and key it under next_phase.value. A later
    # rewind_phase(target_phase=next_phase) reads this entry to restore
    # the agent to the start-of-phase state. Existing snapshots in
    # phase_checkpoints are preserved through the deep-merge reducer.
    snapshot = _build_phase_snapshot(state, captured_from_phase=current.value)
    new_phase_checkpoint_entry = {next_phase.value: snapshot}

    return Command(update={
        "phase": next_phase,
        # Variant pool is per-HITL-gate. Phase advance clears it so the next
        # phase starts with an empty pool. The approved variant has already
        # been promoted to current_image (either by hitl_check on variant
        # approval, or by the tool itself in autonomous mode).
        "variant_pool": [],
        # visual_context is the VLM working set. Phase advance is the natural
        # release point: prior-phase visuals are no longer decision-relevant.
        "visual_context": [],
        # Regression warnings are phase-scoped: they were either resolved
        # within the phase (auto-cleared on recovery) or acknowledged by the
        # agent as part of the advance reasoning (captured into the log above).
        # Clearing the list avoids carrying stale alerts into a new phase
        # where different metrics matter.
        "regression_warnings": [],
        # metadata uses _merge_dicts: only the fields named here are touched,
        # everything else (including any prior phase_checkpoints entries)
        # is preserved.
        "metadata": {
            "last_analysis_snapshot": None,
            "phase_checkpoints": new_phase_checkpoint_entry,
        },
        "messages": [ToolMessage(
            content=json.dumps(result, indent=2),
            tool_call_id=tool_call_id,
        )],
    })
