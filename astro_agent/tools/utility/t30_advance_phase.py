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

from astro_agent.graph.state import AstroState, ProcessingPhase

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
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "advance_phase":
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
) -> None:
    """Append a human-readable phase summary to processing_log.md."""
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
        log_path.write_text("# AstroAgent Processing Log\n\n")

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

    # Guard: phase-specific required tools
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

    logger.info(f"advance_phase tool: {current.value} → {next_phase.value} | reason: {reason}")

    working_dir = state.get("dataset", {}).get("working_dir")
    if working_dir:
        try:
            _write_phase_report(
                phase=current.value,
                calls=calls,
                working_dir=working_dir,
                reason=reason,
            )
        except Exception as e:
            logger.warning(f"Phase report write failed (non-fatal): {e}")

    result = {
        "previous_phase": current.value,
        "new_phase": next_phase.value,
        "reason": reason,
    }

    return Command(update={
        "phase": next_phase,
        "messages": [ToolMessage(
            content=json.dumps(result, indent=2),
            tool_call_id=tool_call_id,
        )],
    })
