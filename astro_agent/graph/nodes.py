"""
Graph nodes — phase_router, agent, action, hitl_check, phase_advance.

See graph_design.md for the architecture:

    phase_router → agent → action → hitl_check → agent  (ReAct loop)
                     │
                     └── (no tool_calls) → phase_advance → phase_router
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from astro_agent.graph.hitl import (
    images_from_tool,
    is_affirmative,
    is_enabled,
    resolve_hitl_checkpoint,
    tool_cfg,
    vlm_enabled,
)
from astro_agent.graph.prompts import PHASE_PROMPTS as _PHASE_PROMPTS
from astro_agent.graph.prompts import SYSTEM_BASE as _SYSTEM_BASE
from astro_agent.graph.registry import all_tools, tools_for_phase
from astro_agent.graph.state import AstroState, HITLPayload, ProcessingPhase

logger = logging.getLogger(__name__)


# ── phase_router ──────────────────────────────────────────────────────────────

def phase_router(state: AstroState) -> dict[str, Any]:
    """
    Read current phase, prepare phase-specific context for the agent.

    This node doesn't bind tools directly — tool binding happens in the
    agent node via the model callable pattern. This node's job is to
    ensure the agent has the right prompt context for its current phase.
    """
    phase = state.get("phase", ProcessingPhase.INGEST)
    phase_prompt = _PHASE_PROMPTS.get(phase, "")

    if phase == ProcessingPhase.COMPLETE:
        logger.info("Pipeline complete.")
    else:
        logger.info(f"Phase router: {phase.value} — routing to agent")

    return {"_phase_prompt": phase_prompt}


def route_after_phase_router(state: AstroState) -> str:
    """Route to agent or end based on current phase."""
    phase = state.get("phase", ProcessingPhase.INGEST)
    if phase == ProcessingPhase.COMPLETE:
        return "__end__"
    return "agent"


# ── agent ─────────────────────────────────────────────────────────────────────

# ── VLM image filtering ─────────────────────────────────────────────────────
# VLM images are scoped to the active HITL conversation only. Once the human
# approves and the agent moves on, ALL images are stripped from the view sent
# to the LLM. The agent goes back to data-only. State keeps everything
# (full audit trail) — this is a view filter, not a mutation.
#
# During an active HITL exchange (feedback → re-call → feedback → ...), the
# images accumulate naturally because hitl_check keeps injecting them. The
# _in_active_hitl flag checks whether the most recent messages look like an
# ongoing HITL conversation (multimodal HumanMessage as the last message).
# If so, images are preserved. If not, they're all stripped.


def _strip_vlm_images(messages: list) -> list:
    """
    Strip ALL image content blocks from multimodal HumanMessages.
    Returns a new list — does not mutate the originals.
    """
    result = []
    for msg in messages:
        if (
            isinstance(msg, HumanMessage)
            and isinstance(msg.content, list)
            and any(block.get("type") == "image_url" for block in msg.content)
        ):
            text_blocks = [b for b in msg.content if b.get("type") == "text"]
            text = " ".join(b["text"] for b in text_blocks) if text_blocks else ""
            result.append(HumanMessage(content=text))
        else:
            result.append(msg)
    return result


def _in_active_hitl(messages: list) -> bool:
    """
    Check if the conversation is in an active HITL feedback loop.

    True when the last message is a multimodal HumanMessage (VLM feedback
    just injected by hitl_check). This means the agent is about to respond
    to human feedback with the image visible — keep all images.
    """
    if not messages:
        return False
    last = messages[-1]
    return (
        isinstance(last, HumanMessage)
        and isinstance(last.content, list)
        and any(block.get("type") == "image_url" for block in last.content)
    )


def make_agent_node(model_factory):
    """
    Create the agent node with a model factory for phase-gated tool binding.

    model_factory: callable that takes (phase) and returns a model with
    the right tools bound via .bind_tools().
    """

    def agent_node(state: AstroState) -> dict[str, Any]:
        phase = state.get("phase", ProcessingPhase.INGEST)
        model = model_factory(phase)

        # Build message list with system prompt
        phase_prompt = state.get("_phase_prompt", "")
        system = f"{_SYSTEM_BASE}\n\n{phase_prompt}"

        messages = [SystemMessage(content=system)] + list(state.get("messages", []))

        # VLM scoping: images are only visible during active HITL conversations.
        # Once the human approves and we move on, strip all images from the view.
        if not _in_active_hitl(state.get("messages", [])):
            messages = _strip_vlm_images(messages)

        response = model.invoke(messages)

        logger.info(
            f"Agent response: {'tool_calls' if response.tool_calls else 'no tool_calls'}"
            + (f" ({[tc['name'] for tc in response.tool_calls]})" if response.tool_calls else "")
        )

        return {"messages": [response]}

    return agent_node


# ── action ────────────────────────────────────────────────────────────────────
# We use LangGraph's prebuilt ToolNode which handles tool execution and
# returns ToolMessages. Initialized with all_tools() — the phase_router
# controls which tools the LLM can *call* (via bind_tools), but the
# ToolNode can execute any of them.

def make_action_node():
    """Create the action node using LangGraph's prebuilt ToolNode."""
    return ToolNode(all_tools(), handle_tool_errors=True)


# ── hitl_check ────────────────────────────────────────────────────────────────


def _find_active_hitl_tool(messages: list) -> tuple[str | None, str | None]:
    """
    Walk backward through ALL messages to find the most recent HITL-triggering
    ToolMessage. Used when re-entering hitl_check during an active HITL chat
    (where the most recent messages are Human/AI chat, not ToolMessages).
    """
    from astro_agent.graph.hitl import TOOL_TO_HITL
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name in TOOL_TO_HITL:
            return TOOL_TO_HITL[msg.name], msg.name
    return None, None


def hitl_check(state: AstroState) -> dict[str, Any]:
    """
    Policy enforcement node — fires interrupt() when appropriate.

    Two entry paths:
    1. After action node: checks if the tool just executed is in hitl_config.toml
    2. After agent chat response during active HITL: re-fires interrupt() so
       the human can continue the conversation (ask more questions, approve, etc.)
    """
    messages = state.get("messages", [])
    hitl_key, tool_name = resolve_hitl_checkpoint(messages)

    if hitl_key is None:
        if not state.get("active_hitl", False):
            # No HITL mapping and no active conversation — pass through
            return {}
        # Active HITL conversation but no new tool call — the agent just
        # responded to a chat message. Re-fire interrupt so the human can
        # continue. We need to find the original tool from earlier messages.
        hitl_key, tool_name = _find_active_hitl_tool(messages)
        if hitl_key is None:
            # Shouldn't happen, but fail safe
            return {"active_hitl": False}

    # HITL disabled for this tool (or autonomous mode) — pass through
    if not is_enabled(hitl_key):
        return {}

    # Fire HITL interrupt
    cfg = tool_cfg(hitl_key)
    image_paths = images_from_tool(messages, tool_name)
    payload = HITLPayload(
        type=cfg["type"],
        title=cfg["title"],
        tool_name=tool_name,
        images=image_paths,
        context=messages[-6:],  # recent messages for continuity
    )

    # When vlm_enabled + image_review, inject the preview image BEFORE
    # interrupt() so the agent has it from the start of the HITL conversation.
    # Side effects before interrupt() must be idempotent — appending a message
    # is fine because on resume the node restarts and re-appends the same image.
    vlm_messages: list = []
    if vlm_enabled() and image_paths and cfg["type"] == "image_review":
        latest_image = Path(image_paths[-1])
        if latest_image.exists():
            mime = mimetypes.guess_type(str(latest_image))[0] or "image/jpeg"
            b64 = base64.standard_b64encode(latest_image.read_bytes()).decode()
            vlm_content: list[dict] = [
                {"type": "text", "text": f"[VLM] Current result from {tool_name}:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ]
            vlm_messages = [HumanMessage(content=vlm_content)]
            logger.info(f"VLM: injecting image {latest_image}")

    logger.info(f"HITL interrupt: {cfg['title']} (tool: {tool_name})")
    response = interrupt(payload)
    logger.info(f"HITL response: {response!r}")

    response_text = str(response)

    if is_affirmative(response_text):
        # Human approved — HITL conversation is over.
        # Return VLM image so it's in state (audit trail), but _strip_vlm_images
        # will remove it from the agent's view on the next call.
        result: dict[str, Any] = {"active_hitl": False}
        if vlm_messages:
            result["messages"] = vlm_messages
        return result

    # Human gave feedback or chat — inject VLM image + feedback text.
    # Keep active_hitl=True so the agent stays in the HITL loop even if
    # it responds without tool calls (e.g. answering a question).
    return {
        "messages": vlm_messages + [HumanMessage(content=response_text)],
        "active_hitl": True,
    }


# ── Phase ordering ────────────────────────────────────────────────────────────
# The natural progression through the pipeline. phase_advance uses this to
# determine the next phase when the agent finishes the current one.

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


# ── phase_advance ─────────────────────────────────────────────────────────────

def phase_advance(state: AstroState) -> dict[str, Any]:
    """
    Advance to the next processing phase.

    Called when the agent exits the ReAct loop (no tool_calls), meaning it
    has completed all tasks for the current phase.
    """
    current = state.get("phase", ProcessingPhase.INGEST)
    next_phase = _NEXT_PHASE.get(current, ProcessingPhase.COMPLETE)
    logger.info(f"Phase advance: {current.value} → {next_phase.value}")
    return {"phase": next_phase}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_agent(state: AstroState) -> str:
    """
    After the agent node: if there are tool_calls, go to action.
    If in an active HITL conversation (human chatting/discussing), route back
    to hitl_check so interrupt() fires again and the human can continue.
    If no tool_calls and no active HITL, advance to the next phase.
    """
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "action"

    # Active HITL conversation — agent answered a question, route back to
    # hitl_check which will re-fire interrupt() for the human to continue.
    if state.get("active_hitl", False):
        return "hitl_check"

    return "phase_advance"
