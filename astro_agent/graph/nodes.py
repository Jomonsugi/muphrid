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
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from astro_agent.graph.hitl import (
    TOOL_TO_HITL,
    images_from_tool,
    is_affirmative,
    is_autonomous,
    is_enabled,
    resolve_hitl_checkpoint,
    tool_cfg,
    vlm_autonomous,
    vlm_hitl,
)
from astro_agent.graph.content import image_blocks, text_content
from astro_agent.graph.prompts import PHASE_PROMPTS as _PHASE_PROMPTS
from astro_agent.graph.prompts import SYSTEM_BASE as _SYSTEM_BASE
from astro_agent.graph.registry import all_tools, tools_for_phase
from astro_agent.graph.state import AstroState, HITLPayload, ProcessingPhase

logger = logging.getLogger(__name__)


def _check_anthropic(model) -> bool:
    """Check if the model is a ChatAnthropic instance (supports cache_control)."""
    try:
        from langchain_anthropic import ChatAnthropic
        # model may be a RunnableBinding (from bind_tools), check the bound model
        bound = getattr(model, "bound", model)
        return isinstance(bound, ChatAnthropic)
    except ImportError:
        return False


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
        if isinstance(msg, HumanMessage) and image_blocks(msg.content):
            result.append(HumanMessage(content=text_content(msg.content)))
        else:
            result.append(msg)
    return result


_ANALYSIS_TOOLS = {"analyze_image", "analyze_frames"}


def _prune_phase_analysis(messages: list) -> list:
    """
    Replace analyze_image/analyze_frames ToolMessage content from completed
    phases with a short placeholder. Current phase messages are untouched.

    The model's reasoning (AIMessages) captures the conclusions from analysis
    results. The raw JSON served its purpose at decision time and is dead
    weight once the phase ends.

    Returns a new list — does not mutate state.
    """
    import os
    if os.environ.get("PRUNE_PHASE_ANALYSIS", "").lower() not in ("1", "true"):
        return messages

    # Find the last successful advance_phase boundary
    phase_boundary = 0
    for i, msg in enumerate(messages):
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", None) == "advance_phase"
            and "Cannot advance" not in str(msg.content)
        ):
            phase_boundary = i + 1

    result = []
    for i, msg in enumerate(messages):
        if (
            i < phase_boundary
            and isinstance(msg, ToolMessage)
            and getattr(msg, "name", None) in _ANALYSIS_TOOLS
        ):
            result.append(ToolMessage(
                content="[Analysis from prior phase — see reasoning above]",
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
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
    return isinstance(last, HumanMessage) and bool(image_blocks(last.content))


def _recent_present_images(messages: list) -> bool:
    """
    Check if the most recent ToolMessage is a present_images result.
    Used to decide whether to inject VLM images for autonomous inspection.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.name == "present_images"
        if isinstance(msg, AIMessage):
            continue  # Skip the AI message that made the tool call
        break
    return False


def _inject_present_images_vlm(
    messages: list, working_dir: str, is_linear: bool
) -> HumanMessage | None:
    """
    Extract image paths from the most recent present_images ToolMessage,
    convert FITS to JPG previews, and build a VLM HumanMessage.
    """
    from astro_agent.tools.utility.t22_generate_preview import generate_preview

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "present_images":
            try:
                result = json.loads(text_content(msg.content))
                if result.get("status") != "presented":
                    return None
                raw_paths = [img["path"] for img in result.get("images", []) if img.get("path")]
            except (json.JSONDecodeError, TypeError, KeyError):
                return None

            # Convert FITS → JPG previews
            preview_paths: list[str] = []
            for img in raw_paths:
                p = Path(img)
                if p.suffix.lower() in (".fit", ".fits", ".fts") and working_dir:
                    preview_dir = Path(working_dir) / "previews"
                    expected = preview_dir / f"preview_{p.stem}.jpg"
                    if expected.exists():
                        preview_paths.append(str(expected))
                    else:
                        try:
                            prev = generate_preview(
                                working_dir=working_dir,
                                fits_path=str(p),
                                format="jpg",
                                quality=95,
                                auto_stretch_linear=is_linear,
                            )
                            preview_paths.append(prev["preview_path"])
                        except Exception as e:
                            logger.warning(f"VLM preview failed for {p.name}: {e}")
                elif p.exists():
                    preview_paths.append(str(p))

            if preview_paths:
                return _make_vlm_message(preview_paths, "Visual inspection of presented images")
            return None
        break
    return None


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

        # Anthropic prompt caching: use content blocks with cache_control
        # so the stable prefix is cached across calls within a phase.
        # additional_kwargs doesn't work — must use inline content blocks.
        _is_anthropic = _check_anthropic(model)

        if _is_anthropic:
            messages = [SystemMessage(content=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ])] + raw_messages

            # Mark the last successful advance_phase as a cache boundary so
            # all completed-phase messages are cached alongside the system prompt.
            # Anthropic allows max 4 cache_control blocks, so first revert any
            # previously-marked advance_phase messages to plain strings, then
            # mark only the most recent one.
            for msg in messages:
                if (
                    isinstance(msg, ToolMessage)
                    and getattr(msg, "name", None) == "advance_phase"
                    and isinstance(msg.content, list)
                ):
                    # Revert to plain string
                    msg.content = text_content(msg.content)

            for msg in reversed(messages):
                if (
                    isinstance(msg, ToolMessage)
                    and getattr(msg, "name", None) == "advance_phase"
                    and "Cannot advance" not in str(msg.content)
                ):
                    text = str(msg.content)
                    msg.content = [
                        {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}},
                    ]
                    break
        else:
            messages = [SystemMessage(content=system)] + raw_messages

        # VLM scoping: images are only visible when relevant.
        # - During active HITL (vlm_hitl): preserve all images
        # - After present_images call outside HITL (vlm_autonomous): inject
        #   images for this one reasoning cycle
        # - Otherwise: strip all images from the view
        _active_hitl = _in_active_hitl(raw_messages)
        _has_present_images = _recent_present_images(raw_messages)

        if _active_hitl and vlm_hitl():
            pass  # Keep images — agent is in visual HITL conversation
        elif _has_present_images and vlm_autonomous() and not _active_hitl:
            # Inject present_images results as base64 for autonomous inspection
            working_dir = state.get("dataset", {}).get("working_dir", "")
            is_linear = state.get("metrics", {}).get("is_linear_estimate", True)
            vlm_msg = _inject_present_images_vlm(raw_messages, working_dir, is_linear)
            if vlm_msg:
                messages.append(vlm_msg)
                logger.info("VLM autonomous: injecting present_images for visual inspection")
        else:
            messages = _strip_vlm_images(messages)

        # Prune analysis outputs from completed phases — the model's reasoning
        # captured the conclusions; raw JSON is dead weight after phase ends.
        messages = _prune_phase_analysis(messages)

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


# ── VLM image injection ──────────────────────────────────────────────────────


def _make_vlm_message(image_paths: list[str], label: str) -> HumanMessage | None:
    """
    Build a multimodal HumanMessage with base64-encoded preview images.

    Converts FITS to JPG previews first. Returns None if no valid images.
    Images must be JPG/PNG — raw FITS cannot be base64-encoded for LLMs.
    """
    content: list[dict] = [{"type": "text", "text": f"[VLM] {label}"}]
    has_image = False

    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            continue
        # Only encode rendered formats — skip raw FITS
        if p.suffix.lower() in (".fit", ".fits", ".fts"):
            continue
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        b64 = base64.standard_b64encode(p.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
        has_image = True

    if not has_image:
        return None
    return HumanMessage(content=content)


def _collect_hitl_images(messages: list, working_dir: str, is_linear: bool) -> list[str]:
    """
    Collect all image paths relevant to the current HITL conversation:
    - Images from the HITL-triggering tool
    - Images from any present_images calls during the conversation

    Returns preview JPG paths (FITS converted via generate_preview).
    """
    from astro_agent.tools.utility.t22_generate_preview import generate_preview

    raw_paths: list[str] = []

    # Walk backward from the end to collect images in the current HITL exchange
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                result = json.loads(text_content(msg.content))
                if isinstance(result, dict):
                    # present_images results
                    if msg.name == "present_images" and result.get("status") == "presented":
                        for img in result.get("images", []):
                            if img.get("path"):
                                raw_paths.append(img["path"])
                    # HITL-triggering tool results
                    elif msg.name in TOOL_TO_HITL:
                        for key in ("output_path", "result_path", "stretched_image_path",
                                    "starless_image_path", "preview_path", "mask_path"):
                            if path := result.get(key):
                                raw_paths.append(path)
                        break  # Stop at the HITL trigger
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(msg, HumanMessage):
            # Hit a human message before finding the trigger — stop
            break

    # Convert FITS to JPG previews
    preview_paths: list[str] = []
    for img in raw_paths:
        p = Path(img)
        if p.suffix.lower() in (".fit", ".fits", ".fts") and working_dir:
            preview_dir = Path(working_dir) / "previews"
            expected = preview_dir / f"preview_{p.stem}.jpg"
            if expected.exists():
                preview_paths.append(str(expected))
            else:
                try:
                    result = generate_preview(
                        working_dir=working_dir,
                        fits_path=str(p),
                        format="jpg",
                        quality=95,
                        auto_stretch_linear=is_linear,
                    )
                    preview_paths.append(result["preview_path"])
                except Exception as e:
                    logger.warning(f"VLM preview generation failed for {p.name}: {e}")
        elif p.exists():
            preview_paths.append(str(p))

    return preview_paths


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
    Two-pass HITL checkpoint node.

    Pass 1 (tool just executed): Detects HITL-triggering tool, sets active_hitl,
    injects a prompt telling the agent to analyze and present results. Does NOT
    interrupt — lets the agent see the tool result and produce analysis first.

    Pass 2 (agent has analyzed): Fires interrupt() with the agent's analysis,
    images, and context. Human reviews, gives feedback, or approves.

    Re-run (agent re-executed HITL tool during active HITL after feedback):
    Passes through so agent can analyze the new result, then fires interrupt
    on the next text response.
    """
    messages = state.get("messages", [])
    active_hitl = state.get("active_hitl", False)
    hitl_key, tool_name = resolve_hitl_checkpoint(messages)
    last = messages[-1] if messages else None

    # ── No HITL tool found in recent messages ────────────────────────
    if hitl_key is None:
        if not active_hitl:
            return {}  # no HITL mapping, no active conversation — pass through

        # Active HITL: agent called a non-HITL tool (analyze_image, present_images)
        # Let it pass through so the agent sees the result.
        if isinstance(last, ToolMessage) and last.name not in TOOL_TO_HITL:
            return {}

        # Agent responded with text — find the original HITL trigger
        hitl_key, tool_name = _find_active_hitl_tool(messages)
        if hitl_key is None:
            return {"active_hitl": False}

    # ── HITL disabled for this tool (or autonomous mode) ─────────────
    if not is_enabled(hitl_key):
        return {}

    # ── HITL tool just executed — pass through for agent analysis ────
    # The agent hasn't seen the result yet. Let it analyze before we
    # fire the interrupt. This applies both to initial triggers AND
    # re-runs during active HITL (agent adjusted params after feedback).
    if isinstance(last, ToolMessage) and last.name in TOOL_TO_HITL:
        if not active_hitl:
            # Initial trigger — inject analysis prompt
            logger.info(f"HITL triggered for {tool_name} — routing to agent for analysis")
            return {
                "active_hitl": True,
                "messages": [HumanMessage(
                    content=(
                        f"HITL review triggered for {tool_name}. A human is now "
                        f"reviewing your work on this step.\n\n"
                        f"Analyze the result — what do the metrics show? What changed?\n"
                        f"Call present_images to show the current image.\n"
                        f"Share your assessment: what worked, what trade-offs you see.\n"
                        f"The human will give feedback or approve.\n\n"
                        f"If they give feedback, interpret it in terms of tool parameters, "
                        f"re-run the tool, and present the updated result with a comparison "
                        f"of what changed."
                    ),
                    additional_kwargs={"is_hitl_prompt": True},
                )],
            }
        else:
            # Re-run during active HITL — let agent analyze new result
            logger.info(f"HITL tool {tool_name} re-executed — letting agent analyze new result")
            return {}

    # ── Agent has analyzed — fire interrupt ──────────────────────────
    cfg = tool_cfg(hitl_key)
    image_paths = images_from_tool(messages, tool_name)

    # Extract agent's analysis text (latest AIMessage)
    _agent_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            _agent_text = text_content(msg.content)
            break

    payload = HITLPayload(
        type=cfg["type"],
        title=cfg["title"],
        tool_name=tool_name,
        images=image_paths,
        context=messages[-6:],
        agent_text=_agent_text,
    )

    # VLM during HITL: inject base64 preview images so the agent can see
    # what it's discussing with the human.
    vlm_messages: list = []
    if vlm_hitl():
        working_dir = state.get("dataset", {}).get("working_dir", "")
        is_linear = state.get("metrics", {}).get("is_linear_estimate", True)
        preview_paths = _collect_hitl_images(messages, working_dir, is_linear)
        if preview_paths:
            vlm_msg = _make_vlm_message(preview_paths, f"Current result from {tool_name}")
            if vlm_msg:
                vlm_messages = [vlm_msg]
                logger.info(f"VLM HITL: injecting {len(preview_paths)} images")

    logger.info(f"HITL interrupt: {cfg['title']} (tool: {tool_name})")
    response = interrupt(payload)
    logger.info(f"HITL response: {response!r}")

    response_text = str(response)

    if is_affirmative(response_text):
        from astro_agent.graph.hitl import extract_approval_note
        note = extract_approval_note(response_text)
        approval_text = note if note else "Approved. Continue to the next step."

        # Long-term memory extraction (v1: HITL-only)
        _extract_hitl_memories(state, messages, tool_name, note)

        return {
            "active_hitl": False,
            "messages": vlm_messages + [HumanMessage(content=approval_text)],
        }

    # Human gave feedback — keep in HITL loop
    return {
        "messages": vlm_messages + [HumanMessage(content=response_text)],
        "active_hitl": True,
    }


# ── Long-term memory extraction from HITL ────────────────────────────────────

def _extract_hitl_memories(state: AstroState, messages: list, tool_name: str, approval_note: str):
    """
    Extract and store memories after HITL approval (non-blocking, non-fatal).

    Called programmatically by the harness after every HITL approval.
    Uses the agent's LLM to extract observations, failures, and preferences
    from the HITL conversation context.

    Design: Lesson #1 (programmatic saves), #9 (HITL-only for v1),
            #10 (schema-driven extraction)
    """
    from astro_agent.graph.hitl import is_memory_enabled
    if not is_memory_enabled():
        return

    try:
        from astro_agent.memory.extraction import (
            build_hitl_conversation_text,
            extract_hitl_memory,
        )
        from astro_agent.tools.utility.t33_memory_search import _MEMORY_STORE
        from astro_agent.config import make_llm

        if _MEMORY_STORE is None:
            return

        phase = state.get("phase", ProcessingPhase.INGEST)
        phase_str = phase.value if hasattr(phase, "value") else str(phase)
        session = state.get("session", {})
        dataset = state.get("dataset", {})
        acquisition = dataset.get("acquisition_meta", {})

        session_context = {
            "target_name": session.get("target_name", "unknown"),
            "target_type": session.get("target_type", ""),
            "sensor": acquisition.get("camera_name", ""),
            "sensor_type": acquisition.get("sensor_type", ""),
        }

        # Build conversation text from message history
        conversation_text = build_hitl_conversation_text(messages, tool_name)
        if not conversation_text.strip():
            return

        # If the user added an approval note, append it
        if approval_note:
            conversation_text += f"\n\n[Human approval note]\n{approval_note}"

        # Extract memories using the agent's LLM
        llm = make_llm()
        extraction = extract_hitl_memory(
            conversation=conversation_text,
            tool_name=tool_name,
            phase=phase_str,
            session_context=session_context,
            llm=llm,
        )

        # Get thread_id for session linkage
        # (thread_id is in the config, not state — use a placeholder for now)
        session_id = None

        # Store extracted memories
        for obs in extraction.observations:
            _MEMORY_STORE.add_observation(
                content=obs.content,
                phase=obs.phase or phase_str,
                session_id=session_id,
                source="hitl",
                parameters=obs.parameters,
                metrics=obs.metrics,
            )

        for fail in extraction.failures:
            _MEMORY_STORE.add_failure(
                content=fail.content,
                phase=fail.phase or phase_str,
                tool=fail.tool or tool_name,
                session_id=session_id,
                source="hitl",
                parameters=fail.parameters,
                root_cause=fail.root_cause,
                resolution=fail.resolution,
            )

        for pref in extraction.preferences:
            _MEMORY_STORE.add_preference(
                content=pref.content,
                tool=pref.tool or tool_name,
                session_id=session_id,
                source="hitl",
                parameters=pref.parameters,
                target_type=session_context.get("target_type"),
                sensor=session_context.get("sensor"),
            )

        total = len(extraction.observations) + len(extraction.failures) + len(extraction.preferences)
        if total > 0:
            logger.info(f"Memory: stored {total} memories from {tool_name} HITL approval")

    except Exception as e:
        # Memory extraction is never fatal — the pipeline must continue
        logger.warning(f"Memory extraction failed (non-fatal): {e}")


# ── agent_chat ────────────────────────────────────────────────────────────────
# This node handles text-only agent responses (no tool calls) outside of
# active HITL. Since route_after_agent sends active_hitl=True responses to
# hitl_check instead, agent_chat is ONLY reached when the agent is working
# autonomously and responds with text instead of calling a tool.
#
# The agent should always be calling tools between HITL checkpoints.
# Text-only responses mean the model is hesitating, narrating, or stuck.
# Always nudge it to act.

_AUTONOMOUS_NUDGE = (
    "Either call a tool to continue processing, or call advance_phase "
    "to move to the next phase. Do not respond with text without "
    "calling a tool."
)


def agent_chat(state: AstroState) -> dict[str, Any]:
    """
    Handle text-only agent responses outside of active HITL.

    Always nudges the agent to call a tool. Human interaction only happens
    through hitl_check interrupts at configured checkpoints.
    """
    messages = state.get("messages", [])
    phase = state.get("phase", ProcessingPhase.INGEST)

    # Extract the agent's text for logging
    agent_text = ""
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage):
            agent_text = text_content(last.content)

    # Count consecutive text-only responses to catch loops
    import os
    max_nudges = int(os.environ.get("MAX_AUTONOMOUS_NUDGES", "2"))
    consecutive_text_only = 0
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            consecutive_text_only += 1
        elif isinstance(msg, HumanMessage) and msg.additional_kwargs.get("is_nudge"):
            continue  # skip our own nudge injections
        else:
            break  # tool results, HITL feedback, etc. reset the counter

    if consecutive_text_only >= max_nudges:
        raise NudgeLimitError(
            f"Agent produced {consecutive_text_only} consecutive text-only responses "
            f"without calling any tool. Current phase: {phase.value.upper()}. "
            f"The agent must call tools to make progress or call advance_phase "
            f"to move to the next phase. "
            f"Set MAX_AUTONOMOUS_NUDGES in .env to adjust (current: {max_nudges})."
        )

    # Always nudge — agent should be calling tools, not narrating
    logger.info(
        f"agent_chat: nudging agent (phase={phase.value}) "
        f"| agent said: {agent_text[:200]}"
    )
    return {"messages": [HumanMessage(
        content=_AUTONOMOUS_NUDGE,
        additional_kwargs={"is_nudge": True},
    )]}


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
