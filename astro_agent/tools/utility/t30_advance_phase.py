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
from typing import Annotated

from langchain_core.messages import ToolMessage
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
    # The agent cannot skip or bypass the export step by calling advance_phase directly.
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

    logger.info(f"advance_phase tool: {current.value} → {next_phase.value} | reason: {reason}")

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
