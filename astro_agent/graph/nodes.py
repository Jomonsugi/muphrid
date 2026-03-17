"""
Graph nodes — phase_router, agent, action, hitl_check, phase_advance.

See graph_design.md for the architecture:

    phase_router → agent → action → hitl_check → agent  (ReAct loop)
                     │
                     └── (no tool_calls) → phase_advance → phase_router
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from astro_agent.graph.hitl import (
    images_from_tool,
    is_affirmative,
    is_autonomous,
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

    return {}


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


# ── Stuck-loop detection ──────────────────────────────────────────────────────
# Hard fail if the agent calls the same tool N times in a row. This is a
# testing safety net — set MAX_CONSECUTIVE_SAME_TOOL in .env. 0 disables.


class StuckLoopError(RuntimeError):
    """Raised when the agent calls the same tool too many times consecutively."""


class NudgeLimitError(RuntimeError):
    """Raised when the agent refuses to call tools after repeated nudges."""


class PhaseToolLimitError(RuntimeError):
    """Raised when tool calls in a single phase exceed the configured limit."""


def _check_phase_tool_limit(messages: list, phase) -> None:
    """
    Count tool calls in the current phase (since the last advance_phase
    ToolMessage) and raise if the per-phase limit is exceeded.

    Reads MAX_TOOLS_<PHASE> from .env (e.g. MAX_TOOLS_INGEST=5).
    Falls back to MAX_TOOLS_PER_PHASE as a global default. 0 disables.
    """
    import os
    phase_key = f"MAX_TOOLS_{phase.value.upper()}"
    limit = int(os.environ.get(phase_key, os.environ.get("MAX_TOOLS_PER_PHASE", "0")))
    if limit <= 0:
        return

    # Count AIMessage tool calls since the last advance_phase boundary
    tool_call_count = 0
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "advance_phase":
            break  # reached the start of this phase
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_call_count += len(msg.tool_calls)

    if tool_call_count >= limit:
        raise PhaseToolLimitError(
            f"Agent has made {tool_call_count} tool calls in the "
            f"{phase.value.upper()} phase (limit: {limit}). This suggests the "
            f"agent is not making meaningful progress. Review the processing "
            f"log and consider whether the phase prompt or tool parameters "
            f"need adjustment. Set {phase_key} or MAX_TOOLS_PER_PHASE in "
            f".env to adjust (0 disables)."
        )


def _check_stuck_loop(messages: list) -> None:
    """
    Walk backward through recent messages to detect repeated identical tool calls.
    Raises StuckLoopError if the limit is hit. Does nothing if limit is 0.

    Only triggers when both the tool name AND arguments are identical across
    consecutive calls. Calling build_masters(file_type="bias") then
    build_masters(file_type="dark") is intentional — not a stuck loop.
    """
    import json
    import os
    limit = int(os.environ.get("MAX_CONSECUTIVE_SAME_TOOL", "0"))
    if limit <= 0:
        return

    # Collect (name, args_fingerprint) tuples for recent tool calls
    recent_calls: list[str] = []  # serialized (name, sorted_args) for comparison
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                fingerprint = json.dumps(
                    {"name": tc["name"], "args": tc.get("args", {})},
                    sort_keys=True,
                )
                recent_calls.append(fingerprint)
            if len(recent_calls) >= limit:
                break
        elif isinstance(msg, (ToolMessage,)):
            continue  # skip tool results, look at the AI calls
        elif isinstance(msg, HumanMessage):
            break  # human intervention resets the counter
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            break  # text response resets the counter

    if len(recent_calls) < limit:
        return

    check = recent_calls[:limit]
    if len(set(check)) == 1:
        # All calls are identical (same name + same args)
        call_info = json.loads(check[0])
        raise StuckLoopError(
            f"Agent called '{call_info['name']}' {limit} times with identical "
            f"arguments {call_info['args']} — aborting. "
            f"This likely means the model is stuck or the tool is broken. "
            f"Set MAX_CONSECUTIVE_SAME_TOOL=0 in .env to disable this check."
        )


# ── DeepSeek raw tool-call token rescue ───────────────────────────────────────
# DeepSeek-V3 (via Together AI) occasionally degenerates and emits its internal
# tool-call delimiters as plain text instead of populating the tool_calls field:
#
#   <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_name<｜tool▁sep｜>{...}<｜tool▁call▁end｜>
#
# When this happens, route_after_agent sees no tool_calls and routes to
# agent_chat, which nudges the model — making the loop worse (growing indent).
# This function detects the pattern and reconstructs a proper AIMessage so
# the tool actually executes.

_DEEPSEEK_TOOL_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)(?:<｜tool▁call▁end｜>|$)",
    re.DOTALL,
)


def _rescue_raw_tool_calls(response: AIMessage) -> AIMessage:
    """
    If the model emitted DeepSeek native tool-call tokens as plain text,
    return a corrective message instead of reconstructing the call.

    Reconstructing bypasses bind_tools validation and phase gating.
    Instead, tell the model what happened so it can make a proper
    structured tool call on the next turn.
    """
    content = response.content if isinstance(response.content, str) else ""
    if "<｜tool▁calls▁begin｜>" not in content:
        return response

    # Extract tool names and args for the corrective message
    matches = _DEEPSEEK_TOOL_CALL_RE.findall(content)
    parsed_calls = []
    for tool_name, args_str in matches:
        tool_name = tool_name.strip()
        args_str = args_str.strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            try:
                open_braces = args_str.count("{") - args_str.count("}")
                args = json.loads(args_str + "}" * max(open_braces, 1))
            except json.JSONDecodeError:
                args = args_str[:200]
        parsed_calls.append({"tool": tool_name, "args": args})

    logger.warning(
        f"DeepSeek rescue: detected raw tool-call tokens in response text "
        f"(tools: {[c['tool'] for c in parsed_calls] or 'unparseable'}). "
        f"Returning corrective message."
    )

    if parsed_calls:
        calls_desc = json.dumps(parsed_calls, indent=2, default=str)
        hint = (
            f"Your previous response contained raw tool-call tokens as plain "
            f"text instead of a structured tool call. The tool was NOT executed. "
            f"Here is what you were trying to call:\n\n{calls_desc}\n\n"
            f"Call the tool again using the structured tool_calls format."
        )
    else:
        hint = (
            "Your previous response contained raw tool-call tokens as plain "
            "text instead of a structured tool call. The tool was NOT executed. "
            "Call the tool again using the structured tool_calls format."
        )

    return AIMessage(content=hint, tool_calls=[])


def make_agent_node(model_factory):
    """
    Create the agent node with a model factory for phase-gated tool binding.

    model_factory: callable that takes (phase) and returns a model with
    the right tools bound via .bind_tools().
    """

    def agent_node(state: AstroState) -> dict[str, Any]:
        phase = state.get("phase", ProcessingPhase.INGEST)
        model = model_factory(phase)

        raw_messages = list(state.get("messages", []))

        # Stuck-loop detection: hard fail if the agent repeats the same tool call.
        _check_stuck_loop(raw_messages)

        # Per-phase tool call cap: count tool calls since the last advance_phase
        # and fail if the limit is exceeded.
        _check_phase_tool_limit(raw_messages, phase)

        # Build message list with system prompt — read phase prompt directly
        # from PHASE_PROMPTS so it's always in sync with the current phase
        # (advance_phase tool updates state["phase"] mid-loop).
        phase_prompt = _PHASE_PROMPTS.get(phase, "")
        system = f"{_SYSTEM_BASE}\n\n{phase_prompt}"

        messages = [SystemMessage(content=system)] + raw_messages

        # VLM scoping: images are only visible during active HITL conversations.
        # Once the human approves and we move on, strip all images from the view.
        if not _in_active_hitl(raw_messages):
            messages = _strip_vlm_images(messages)

        response = model.invoke(messages)
        response = _rescue_raw_tool_calls(response)

        # Phase gate enforcement: reject any tool call not available in the
        # current phase. This catches DeepSeek rescue reconstructing calls
        # from the full tool list in the system prompt.
        if response.tool_calls:
            allowed = {t.name for t in tools_for_phase(phase)}
            rejected = [tc for tc in response.tool_calls if tc["name"] not in allowed]
            if rejected:
                rejected_names = [tc["name"] for tc in rejected]
                allowed_names = sorted(allowed)
                logger.warning(
                    f"Phase gate rejected tool(s) {rejected_names} — "
                    f"not available in {phase.value} phase"
                )
                return {"messages": [AIMessage(
                    content=(
                        f"PHASE GATE: You are in the {phase.value.upper()} phase. "
                        f"The following tool(s) are not available in this phase: "
                        f"{', '.join(rejected_names)}.\n\n"
                        f"Tools available in {phase.value.upper()}: "
                        f"{', '.join(allowed_names)}.\n\n"
                        f"Complete this phase using the available tools, then call "
                        f"advance_phase to move to the next phase."
                    ),
                    tool_calls=[],
                )]}

        logger.info(
            f"Agent response: {'tool_calls' if response.tool_calls else 'no tool_calls'}"
            + (f" ({[tc['name'] for tc in response.tool_calls]})" if response.tool_calls else f" | text: {str(response.content)[:300]!r}")
        )

        return {"messages": [response]}

    return agent_node


# ── action ────────────────────────────────────────────────────────────────────
# We use LangGraph's prebuilt ToolNode which handles tool execution and
# returns ToolMessages. Initialized with all_tools() — the phase_router
# controls which tools the LLM can *call* (via bind_tools), but the
# ToolNode can execute any of them.


def _format_tool_error(exc: Exception) -> str:
    """
    Custom error handler that never returns a blank error message.

    LangGraph's default handler filters out validation errors for injected
    parameters (state, tool_call_id). When ALL errors are about injected
    params, the filtered list is empty and the LLM sees a blank error —
    making it impossible to self-correct. This handler catches that case
    and provides an actionable message.
    """
    from langgraph.prebuilt.tool_node import ToolInvocationError

    msg = str(exc).strip()

    if isinstance(exc, ToolInvocationError):
        # Check if the filtered errors produced an empty message
        if not msg or "with error:\n" in msg and msg.endswith("Please fix the error and try again."):
            # The error was filtered away — it's an internal injection issue
            # Give the LLM the ORIGINAL validation error so it has something to work with
            original = getattr(exc, "source", None)
            if original:
                return (
                    f"Tool '{exc.tool_name}' failed due to an internal validation error "
                    f"(not caused by your arguments). Full error: {original}\n"
                    f"Your arguments were: {exc.tool_kwargs}\n"
                    f"Try calling the tool again — if this persists, the tool may have "
                    f"a state dependency issue. Move on to the next step."
                )
            return (
                f"Tool '{exc.tool_name}' failed with an internal error that could not "
                f"be diagnosed. Your arguments {exc.tool_kwargs} appear valid. "
                f"Try calling the tool again — if this persists, move on."
            )
        return msg

    if not msg:
        return f"Tool failed with {type(exc).__name__} (no details available). Try a different approach."

    return msg


def make_action_node():
    """Create the action node using LangGraph's prebuilt ToolNode."""
    return ToolNode(all_tools(), handle_tool_errors=_format_tool_error)


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


# ── agent_chat ────────────────────────────────────────────────────────────────
# When the agent responds with text (no tool calls) and is not in an active
# HITL conversation, it means the agent wants to communicate — ask a question,
# explain a situation, report a blocker, etc. This node handles that:
#
#   HITL mode:      fire interrupt() so the human can read and respond
#   Autonomous mode: inject a nudge message telling the agent to act or advance

_AUTONOMOUS_NUDGE = (
    "Either call a tool to continue processing, or call advance_phase "
    "to move to the next phase. Do not respond with text without "
    "calling a tool."
)


def agent_chat(state: AstroState) -> dict[str, Any]:
    """
    Handle text-only agent responses.

    In HITL mode: fire interrupt() so the human can respond.
    In autonomous mode: inject a nudge message and route back to agent.
    """
    messages = state.get("messages", [])
    phase = state.get("phase", ProcessingPhase.INGEST)

    # Extract the agent's text for the interrupt payload
    agent_text = ""
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage):
            agent_text = str(last.content)[:500]

    # Count consecutive text-only AI responses (no tool calls between them).
    # This catches both autonomous nudge loops AND CLI auto-respond loops
    # where each interrupt+resume resets the recursion counter.
    import os
    max_nudges = int(os.environ.get("MAX_AUTONOMOUS_NUDGES", "2"))
    consecutive_text_only = 0
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            consecutive_text_only += 1
        elif isinstance(msg, HumanMessage):
            continue  # human/nudge responses between AI text
        else:
            break  # hit a tool call — agent was making progress

    if consecutive_text_only >= max_nudges:
        raise NudgeLimitError(
            f"Agent produced {consecutive_text_only} consecutive text-only responses "
            f"without calling any tool. Current phase: {phase.value.upper()}. "
            f"The agent must call tools to make progress or call advance_phase "
            f"to move to the next phase. "
            f"Set MAX_AUTONOMOUS_NUDGES in .env to adjust (current: {max_nudges})."
        )

    if is_autonomous():
        logger.info(f"agent_chat (autonomous): nudging agent to act or advance (phase={phase.value})")
        return {"messages": [HumanMessage(content=_AUTONOMOUS_NUDGE)]}

    # HITL mode — let the human read the agent's message and respond
    logger.info(f"agent_chat (HITL): firing interrupt for human response (phase={phase.value})")
    response = interrupt({
        "type": "agent_chat",
        "title": f"Agent message ({phase.value} phase)",
        "agent_text": agent_text,
        "phase": phase.value,
    })

    response_text = str(response)
    logger.info(f"agent_chat: human responded: {response_text!r}")

    return {"messages": [HumanMessage(content=response_text)]}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_agent(state: AstroState) -> str:
    """
    After the agent node:
    - phase is COMPLETE → end the graph
    - tool_calls → action (ReAct loop continues)
    - active HITL conversation → hitl_check (re-fire interrupt for human)
    - text only → agent_chat (human conversation or autonomous nudge)
    """
    phase = state.get("phase", ProcessingPhase.INGEST)
    if phase == ProcessingPhase.COMPLETE:
        return "__end__"

    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "action"

    # Active HITL conversation — agent answered a question, route back to
    # hitl_check which will re-fire interrupt() for the human to continue.
    if state.get("active_hitl", False):
        return "hitl_check"

    return "agent_chat"
