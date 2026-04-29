"""
T37 — flag_dataset_issue

Agent-initiated interrupt for conditions processing tools cannot resolve.

The tool calls LangGraph's interrupt() directly so the run pauses
regardless of mode — including autonomous runs, where regular HITL
gates are skipped. This is the documented exception: the agent has
recognized that the dataset itself is the problem and that no further
processing will improve the result without human input.

Lifecycle:
  1. Agent calls flag_dataset_issue(reason=<articulated explanation>).
  2. Tool synchronously appends a FLAGGED ISSUE section to
     processing_log.md so the flag is durable even if the process is
     killed while waiting for the interrupt to be answered.
  3. Tool calls interrupt() with a payload of type="flag_dataset_issue".
     The CLI runner detects this type and behaves mode-aware:
       - autonomous CLI: prints the flag, exits with a non-zero status
         code so an unattended user returns to a clear signal and can
         open Gradio to collaborate.
       - attended CLI: prints the flag and prompts for a response.
     Gradio renders the title and reason in the chat footer; the human
     responds inline.
  4. On resume, the tool returns a Command containing a ToolMessage
     recording the flag and a HumanMessage carrying the human's
     response, so the agent's next turn picks up the conversation
     naturally.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState

logger = logging.getLogger(__name__)


# ── Pydantic input schema ──────────────────────────────────────────────────────

class FlagDatasetIssueInput(BaseModel):
    reason: str = Field(
        description=(
            "A well-articulated explanation of what was observed, what was "
            "tried, and why no further processing tool can address the "
            "condition. The reason is the only signal the human has when "
            "deciding what to do — abandon the dataset, accept the "
            "limitation and proceed in salvage mode, fix the data manually "
            "and resume — so it must be specific. A one-line summary is "
            "not sufficient."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_flag_log(
    working_dir: str,
    *,
    phase: str,
    reason: str,
    current_image: str | None,
) -> None:
    """Append a FLAGGED ISSUE section to processing_log.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"## FLAGGED ISSUE — {phase.upper()}",
        f"*{now}*",
        "",
        "### Reason",
        "",
    ]
    for para in reason.splitlines():
        lines.append(para if para.strip() else "")
    lines.append("")
    if current_image:
        lines.append(f"**Working image at flag time:** `{current_image}`")
        lines.append("")
    lines += ["---", ""]

    log_path = Path(working_dir) / "processing_log.md"
    if not log_path.exists():
        log_path.write_text("# Muphrid Processing Log\n\n")
    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def _build_payload(
    *,
    reason: str,
    phase: str,
    current_image: str | None,
    metrics: dict,
) -> dict:
    """
    Build the interrupt payload. Shape mirrors HITLPayload so the existing
    CLI / Gradio rendering infrastructure handles it without special cases:
      - type:       distinctive value the CLI runner can branch on
      - title:      headline shown in the Gradio chat footer
      - tool_name:  for diagnostic display
      - context:    structured detail the human can read

    A short, decision-relevant slice of metrics is included so the human
    can see where the run was at the moment of the flag without scrolling
    the message history.
    """
    relevant_metrics: dict = {}
    for key in (
        "snr_estimate", "current_fwhm", "current_noise",
        "background_flatness", "gradient_magnitude",
        "channel_imbalance", "star_count",
    ):
        val = metrics.get(key)
        if val is not None:
            relevant_metrics[key] = val

    return {
        "type": "flag_dataset_issue",
        "title": "Dataset issue flagged",
        "tool_name": "flag_dataset_issue",
        "phase": phase,
        "reason": reason,
        "current_image": current_image,
        "metrics_snapshot": relevant_metrics,
        # Empty image / variant lists keep the existing renderers happy.
        "images": [],
        "context": [],
        "agent_text": reason,
        "variant_pool": [],
        "proposal": [],
    }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=FlagDatasetIssueInput)
def flag_dataset_issue(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Pause the run and surface a dataset-level problem to the human.

    Calls LangGraph's interrupt() directly, regardless of mode. In
    autonomous mode the CLI runner prints the flag and exits with a
    non-zero status so the user returns to a clear signal; the dataset
    state is preserved on disk for review or resumption. In attended
    mode (CLI prompt or Gradio chat) the human responds inline and the
    response becomes a HumanMessage on the next agent turn.

    Returns a Command containing a ToolMessage that records the flag
    and a HumanMessage carrying the human's response.
    """
    state = state or {}
    phase = state.get("phase")
    phase_val = getattr(phase, "value", phase) or "unknown"
    current_image = (state.get("paths") or {}).get("current_image")
    metrics = state.get("metrics") or {}
    working_dir = (state.get("dataset") or {}).get("working_dir")

    # Durable record FIRST. If the process is killed while waiting on
    # the interrupt, the flag is still on disk for human review.
    if working_dir:
        try:
            _write_flag_log(
                working_dir,
                phase=phase_val,
                reason=reason,
                current_image=current_image,
            )
        except Exception as e:
            logger.warning(f"Flag log write failed (non-fatal): {e}")

    payload = _build_payload(
        reason=reason,
        phase=phase_val,
        current_image=current_image,
        metrics=metrics,
    )

    logger.warning(
        f"flag_dataset_issue: phase={phase_val} | reason={reason[:120]}"
        f"{'…' if len(reason) > 120 else ''}"
    )

    # Pause the graph. interrupt() returns whatever resume value the
    # presenter (CLI / Gradio) supplies. In autonomous CLI, the runner
    # exits before this returns — the tool body never resumes.
    response = interrupt(payload)
    response_text = str(response) if response is not None else ""

    # Structured tool result for the message stream.
    result = {
        "status": "flagged",
        "phase": phase_val,
        "reason": reason,
        "human_response_received": bool(response_text),
    }

    messages: list = [ToolMessage(
        content=json.dumps(result, indent=2, default=str),
        tool_call_id=tool_call_id,
    )]
    if response_text:
        # Surface the human's response as a HumanMessage so the agent's
        # next turn treats it as natural conversation context. The
        # ToolMessage above records the structured fact of the flag;
        # the HumanMessage carries the directive.
        messages.append(HumanMessage(content=response_text))

    return Command(update={"messages": messages})
