"""
LLM-driven memory extraction from HITL conversations.

Design informed by:
  - langmem (Lesson #10): Pydantic schemas constrain LLM output, preventing
    hallucinated memory structure
  - Lesson #7: contrastive learning — extract both successes AND failures
  - Lesson #1: programmatic saves — harness decides WHEN, LLM decides WHAT
  - Lesson #9: experience-following problem — v1 only extracts from HITL
    (human-validated) to avoid encoding anti-patterns

The extraction LLM is the same model running the agent (confirmed design
decision). It understands astrophotography context from the system prompt
and extraction only fires after HITL approvals (~10 times per run max).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Pydantic schemas for structured extraction (Lesson #10) ──────────────────

class ExtractedObservation(BaseModel):
    """A learning from what worked during processing."""
    content: str = Field(
        description="Natural language description of what worked and why. "
        "Include specific parameter values, metrics, and conditions."
    )
    phase: str = Field(
        description="Processing phase: ingest, calibration, registration, "
        "analysis, stacking, linear, stretch, nonlinear, export"
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Key parameters that produced this result (tool parameter names and values)"
    )
    metrics: dict = Field(
        default_factory=dict,
        description="Relevant metrics before/after (SNR, FWHM, clipping, etc.)"
    )


class ExtractedFailure(BaseModel):
    """A learning from what went wrong during processing."""
    content: str = Field(
        description="Natural language description of what failed or degraded."
    )
    phase: str = Field(
        description="Processing phase where the failure occurred"
    )
    tool: str = Field(
        description="The tool name that failed or produced a degraded result"
    )
    parameters: dict = Field(
        default_factory=dict,
        description="The parameters that caused the failure"
    )
    root_cause: str = Field(
        default="",
        description="Why it failed — the underlying reason"
    )
    resolution: str = Field(
        default="",
        description="What fixed it, if anything was tried"
    )


class ExtractedPreference(BaseModel):
    """A user aesthetic preference expressed through HITL."""
    content: str = Field(
        description="Natural language description of the user's preference. "
        "E.g. 'User prefers moderate stretch that preserves faint outer nebulosity.'"
    )
    tool: str = Field(
        description="The tool this preference relates to"
    )
    parameters: dict = Field(
        default_factory=dict,
        description="The approved parameter values"
    )


class HITLMemoryExtraction(BaseModel):
    """Complete extraction from one HITL conversation."""
    observations: list[ExtractedObservation] = Field(
        default_factory=list,
        description="What worked — successful approaches, good parameter choices"
    )
    failures: list[ExtractedFailure] = Field(
        default_factory=list,
        description="What went wrong — failed attempts, degraded results, errors"
    )
    preferences: list[ExtractedPreference] = Field(
        default_factory=list,
        description="User aesthetic preferences expressed through feedback"
    )


# ── Extraction prompt ────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are extracting long-term memories from an astrophotography processing session.

A human expert just reviewed and approved a processing step via HITL (human-in-the-loop).
The conversation below contains the agent's reasoning, tool calls, parameters, results,
and the human's feedback.

Extract THREE types of memories:

1. **Observations**: What worked and why. Include specific parameter values, metrics,
   and the conditions that made this approach successful. Be detailed — future sessions
   with similar targets/conditions should benefit from this knowledge.

2. **Failures**: What went wrong before the human approved. Include the failed parameters,
   why they failed, and what fixed it. These prevent the agent from repeating the same
   mistakes. Even small failures (wrong parameter range, unnecessary iterations) are
   worth capturing.

3. **Preferences**: The human's aesthetic choices. What did they prefer and why?
   These capture subjective judgment that data alone cannot determine — stretch
   aggressiveness, color balance, star treatment, contrast levels.

Rules:
- Only extract memories that would be useful in FUTURE sessions on DIFFERENT targets.
- Include specific numbers (parameter values, metrics) — vague memories are useless.
- For observations and failures, always include the tool name and phase.
- For preferences, describe the preference in terms the agent can act on.
- If the conversation doesn't contain useful learnings (e.g. simple approval with
  no iteration), return empty lists — don't force extraction.

## Session Context
Target: {target_name}
Target type: {target_type}
Sensor: {sensor}
Sensor type: {sensor_type}
Phase: {phase}
Tool: {tool_name}

## HITL Conversation
{conversation}
"""


# ── Extraction function ──────────────────────────────────────────────────────

def extract_hitl_memory(
    conversation: str,
    tool_name: str,
    phase: str,
    session_context: dict[str, Any],
    llm,
) -> HITLMemoryExtraction:
    """
    Extract structured memories from an HITL conversation using the agent's LLM.

    Args:
        conversation: The full HITL conversation text (agent reasoning + human feedback)
        tool_name: The tool that triggered HITL
        phase: Current processing phase
        session_context: Dict with target_name, target_type, sensor, sensor_type
        llm: The LangChain LLM instance (same model as the agent)

    Returns:
        HITLMemoryExtraction with observations, failures, and preferences
    """
    prompt = _EXTRACTION_PROMPT.format(
        target_name=session_context.get("target_name", "unknown"),
        target_type=session_context.get("target_type", "unknown"),
        sensor=session_context.get("sensor", "unknown"),
        sensor_type=session_context.get("sensor_type", "unknown"),
        phase=phase,
        tool_name=tool_name,
        conversation=conversation,
    )

    try:
        # Use structured output to constrain extraction (Lesson #10)
        structured_llm = llm.with_structured_output(HITLMemoryExtraction)
        result = structured_llm.invoke(prompt)

        if result:
            n_obs = len(result.observations)
            n_fail = len(result.failures)
            n_pref = len(result.preferences)
            logger.info(
                f"Memory extraction: {n_obs} observations, "
                f"{n_fail} failures, {n_pref} preferences "
                f"from {tool_name} HITL in {phase}"
            )
            return result

    except Exception as e:
        logger.warning(f"Memory extraction failed (non-fatal): {e}")

    return HITLMemoryExtraction()


def build_hitl_conversation_text(
    messages: list,
    tool_name: str,
    max_messages: int = 30,
) -> str:
    """
    Build a text representation of the HITL conversation from message history.

    Walks backward from the most recent messages to find the HITL conversation
    about the specified tool, including agent reasoning, tool results, and
    human feedback.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from astro_agent.graph.content import text_content

    lines = []
    count = 0
    found_tool = False

    # Walk backward through messages to find the HITL conversation
    for msg in reversed(messages):
        if count >= max_messages:
            break

        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            found_tool = True
            content = text_content(msg.content)
            # Truncate very long tool outputs
            if len(content) > 2000:
                content = content[:2000] + "... (truncated)"
            lines.append(f"[Tool Result: {msg.name}]\n{content}")
        elif isinstance(msg, AIMessage):
            ai_text = text_content(msg.content)
            if ai_text.strip():
                lines.append(f"[Agent]\n{ai_text}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.get("args", {}), indent=2, default=str)
                    lines.append(f"[Agent Tool Call: {tc['name']}]\n{args_str}")
        elif isinstance(msg, HumanMessage):
            human_text = text_content(msg.content)
            if human_text.strip():
                lines.append(f"[Human]\n{human_text}")

        count += 1

        # Stop if we've passed the relevant tool and hit an earlier HITL boundary
        if found_tool and isinstance(msg, HumanMessage) and count > 5:
            break

    # Reverse to get chronological order
    lines.reverse()
    return "\n\n".join(lines)
