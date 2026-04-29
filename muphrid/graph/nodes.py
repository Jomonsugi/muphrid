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

from muphrid.graph import review as review_ctl
from muphrid.graph.hitl import (
    TOOL_TO_HITL,
    images_from_tool,
    is_enabled,
    resolve_hitl_checkpoint,
    tool_cfg,
    vlm_autonomous,
    vlm_hitl,
    vlm_phase_eligible,
    vlm_window_cap,
)
from muphrid.graph.content import image_blocks, text_content
from muphrid.graph.prompts import HITL_PARTNER_FRAGMENT as _HITL_PARTNER_FRAGMENT
from muphrid.graph.prompts import PHASE_PROMPTS as _PHASE_PROMPTS
from muphrid.graph.prompts import SYSTEM_BASE as _SYSTEM_BASE
from muphrid.graph.registry import all_tools, tools_for_phase
from muphrid.graph.state import AstroState, ProcessingPhase, Variant, VisualRef

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

# ── VLM view construction ───────────────────────────────────────────────────
# State owns visibility. state.visual_context is the live working set of
# images the agent should see; the helpers below build an ephemeral
# multimodal HumanMessage from that list at every model.invoke call.
# Messages stay text-only — they are the audit trail, not the visual context.
#
# Writers to visual_context:
#   - variant_snapshot   → mirrors variant_pool (source="hitl_variant")
#   - promote_variant    → drops hitl_variant entries, keeps approved as
#                          source="phase_carry"
#   - present_images     → replaces source="present_images" entries
#   - advance_phase      → clears the list
#
# The helpers in this section never mutate state; they read it and return
# a new message list to pass to the model.


def _strip_vlm_images(messages: list) -> list:
    """
    Strip ALL image content blocks from multimodal HumanMessages. Defensive
    against legacy state or any path that injected images directly into
    messages — the canonical source is now state.visual_context.

    Returns a new list — does not mutate the originals.
    """
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and image_blocks(msg.content):
            result.append(HumanMessage(content=text_content(msg.content)))
        else:
            result.append(msg)
    return result


def _last_phase_boundary_index(messages: list) -> int:
    """
    Return the index of the first message in the current phase, i.e. one
    past the last successful advance_phase ToolMessage. Returns 0 if no
    successful advance_phase is in the history.

    Used by _prune_phase_analysis (the only remaining message-walking helper).
    """
    boundary = 0
    for i, msg in enumerate(messages):
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", None) == "advance_phase"
            and "Cannot advance" not in str(msg.content)
        ):
            boundary = i + 1
    return boundary


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

    phase_boundary = _last_phase_boundary_index(messages)

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


def _variants_to_refs(state: AstroState) -> list[VisualRef]:
    """
    Project the current variant_pool into a list of VisualRefs the VLM can
    consume. variant_pool is the source of truth for HITL gate variants;
    this function does the on-demand FITS→JPG resolution for any variant
    whose preview_path isn't already populated. Variants whose previews
    can't be resolved are dropped silently.

    Pure-ish: only filesystem reads (preview generation is cached on disk
    via generate_preview's `expected` lookup), no state mutation.
    """
    pool = state.get("variant_pool", []) or []
    if not pool:
        return []
    working_dir = state.get("dataset", {}).get("working_dir", "")
    is_linear = state.get("metrics", {}).get("is_linear_estimate", True)
    refs: list[VisualRef] = []
    for v in pool:
        preview = _resolve_variant_preview(v, working_dir, is_linear)
        if preview is None:
            continue
        refs.append(VisualRef(
            path=preview,
            label=v.get("label") or v.get("id", "variant"),
            source="hitl_variant",
            phase=v.get("phase", ""),
        ))
    return refs


def _current_image_ref(state: AstroState) -> VisualRef | None:
    """
    Build a VisualRef for the working image's preview, if eligible.

    Auto-projection rules:
      - The current phase must be in vlm_phase_eligible() — pre-stack phases
        return None even when paths.current_image happens to point somewhere.
      - state.paths.current_image must be set; the derived agent-VLM preview
        (smaller sibling under <working_dir>/previews/) must exist on disk.
      - We prefer the agent-sized VLM preview (`preview_<stem>_vlm.jpg`) when
        present — produced at ~1024px / q=85 by t22_generate_preview alongside
        the human-facing 1920px preview. If only the human preview exists, we
        fall back to it rather than skipping (correctness over cost).

    Returns None when any precondition fails. The caller decides what to do
    in that case (typically: just don't include the working image).

    This source projects the agent's qualitative anchor — the "what does the
    image look like right now" complement to the analytical metrics. The
    agent retains explicit visual affordances (present_images) regardless of
    phase eligibility; this only governs the auto-injection.
    """
    phase = state.get("phase")
    if not vlm_phase_eligible(phase):
        return None

    paths = state.get("paths") or {}
    current_image = paths.get("current_image")
    if not current_image:
        return None

    working_dir = (state.get("dataset") or {}).get("working_dir") or ""
    stem = Path(current_image).stem
    preview_dir = Path(working_dir) / "previews" if working_dir else None

    chosen: Path | None = None
    if preview_dir is not None:
        vlm_preview = preview_dir / f"preview_{stem}_vlm.jpg"
        human_preview = preview_dir / f"preview_{stem}.jpg"
        if vlm_preview.exists():
            chosen = vlm_preview
        elif human_preview.exists():
            chosen = human_preview

    if chosen is None:
        return None

    phase_val = getattr(phase, "value", phase) or ""
    return VisualRef(
        path=str(chosen),
        label="current working image",
        source="current_image",
        phase=str(phase_val),
    )


def _select_visible_refs(state: AstroState) -> list[VisualRef]:
    """
    Pick the images the agent should see right now. Four sources:

      - state.variant_pool   → projected to VisualRefs (active decision space
                                during a HITL gate or sandwich-iteration in
                                autonomous mode). Source label "hitl_variant".
      - current_image        → auto-projected anchor produced by
                                _current_image_ref(state) when vlm_phase_eligible.
                                Source label "current_image".
      - state.visual_context → present_images and phase_carry entries the
                                agent has explicitly chosen to keep visible.

    Visibility:
      - vlm_hitl is always True (collaboration requires visual access). The
        active gate's variant pool is therefore always visible during HITL.
      - vlm_autonomous controls auto-current-image and the visual_context
        projection outside HITL. When False, the agent operates on metrics;
        present_images returns an informative ToolMessage instead of silently
        adding hidden context.

    Cap: vlm_window_cap() applies to the combined set. The gate-overflow
    exception remains — if the variant pool alone exceeds the cap, the pool
    is shown in full and everything else (including the auto-current-image)
    is dropped, so the human's referenced variant is always resolvable on
    the agent side.

    Order in the returned list (oldest → newest): visual_context entries,
    then auto-current-image (anchor), then variant pool (active decision
    space). The cap takes the newest tail when truncating.
    """
    hitl_on = vlm_hitl()
    auto_on = vlm_autonomous()
    if not hitl_on and not auto_on:
        return []

    variant_refs = _variants_to_refs(state)

    # Auto-current-image is suppressed outside autonomous mode (would otherwise
    # leak phase_carry-equivalent context across HITL gates without explicit
    # agent action). HITL-only mode shows the variant pool plus whatever the
    # gate brings in via state.visual_context.
    auto_current: VisualRef | None = (
        _current_image_ref(state) if auto_on else None
    )

    other_refs = list(state.get("visual_context", []) or [])
    # Drop any pre-existing entry that would shadow the auto-current-image —
    # phase_carry / present_images entries pointing at the same path are
    # redundant when current_image is already projected.
    if auto_current is not None:
        other_refs = [r for r in other_refs if r.get("path") != auto_current["path"]]

    # Build the combined view. Order: other (oldest) → auto-current-image →
    # variants (newest = active decision space).
    combined: list[VisualRef] = list(other_refs)
    if auto_current is not None:
        combined.append(auto_current)
    combined.extend(variant_refs)

    cap = vlm_window_cap()
    if len(variant_refs) > cap:
        # Gate overflow: pool alone exceeds cap → show the full pool, drop
        # other sources so the user can reference any variant in Gradio.
        return variant_refs
    if len(combined) <= cap:
        return combined
    # Hard cap: keep newest `cap` entries (last in list order)
    return combined[-cap:]


def _format_variant_pool_for_prompt(variant_pool: list[Variant]) -> str:
    """
    Render the current variant_pool as a markdown section to inject into the
    system prompt. Surfaces variant ids, labels, and key metrics so the agent
    can reason about what's available and call commit_variant by id without
    needing a query tool. Returns "" if the pool is empty (no section added).

    The agent's view of state goes through messages, not state directly. This
    helper is the bridge: state.variant_pool → text in the system prompt.
    """
    if not variant_pool:
        return ""

    lines = [
        "## Active variant pool",
        "",
        "You have produced the following variants in the current segment. "
        "Each is a concrete result on disk. In autonomous mode, call "
        "`commit_variant(variant_id=...)` to lock in your choice and clear "
        "the rest. In HITL mode, call `present_for_review` to deliberately "
        "share candidate(s), explain the tradeoffs, and make them available "
        "for human approval.",
        "",
    ]
    for v in variant_pool:
        vid = v.get("id", "?")
        label = v.get("label") or v.get("tool_name", "?")
        # Show a small slice of decision-relevant metrics inline
        metric_strs: list[str] = []
        metrics = v.get("metrics", {}) or {}
        for key in (
            "gradient_magnitude", "snr_estimate", "background_flatness",
            "current_fwhm", "current_noise", "star_count",
        ):
            val = metrics.get(key)
            if val is None:
                continue
            if isinstance(val, float):
                metric_strs.append(f"{key}={val:.3f}")
            else:
                metric_strs.append(f"{key}={val}")
        suffix = f"  ({', '.join(metric_strs)})" if metric_strs else ""
        lines.append(f"- **{vid}** — {label}{suffix}")
    return "\n".join(lines)


def _format_checkpoints_for_prompt(checkpoints: dict | None) -> str:
    """Render available image rollback checkpoints for the agent."""
    if not checkpoints:
        return ""

    items = [
        (str(name), str(path))
        for name, path in checkpoints.items()
        if name and path
    ]
    if not items:
        return ""

    shown = items[-20:]
    lines = [
        "## Available image checkpoints",
        "",
        "The system automatically bookmarks the current image before post-stack "
        "image-modifying tools. If a result is worse, call "
        "`restore_checkpoint(name=...)` with one of these names instead of "
        "trying to reason around a bad current_image.",
        "",
    ]
    if len(items) > len(shown):
        lines.append(f"... {len(items) - len(shown)} older checkpoint(s) hidden ...")
    for name, path in shown:
        lines.append(f"- `{name}` → `{Path(path).name}`")
    return "\n".join(lines)


def _format_review_session_for_prompt(review_session: dict | None) -> str:
    """Render explicit Review Mode state for the agent."""
    if not review_ctl.review_is_open(review_session):
        return ""
    proposal = review_session.get("proposal", {}) or {}
    candidates = proposal.get("candidates", []) or []
    last_event = review_session.get("last_human_event") or {}
    lines = [
        "## Active HITL review session",
        "",
        f"- Gate: `{review_session.get('gate_id', 'unknown')}`",
        f"- Tool: `{review_session.get('tool_name', 'unknown')}`",
        f"- Status: `{review_session.get('status', 'unknown')}`",
        f"- Turn policy: `{review_session.get('turn_policy', 'answer_visible_text_before_action')}`",
    ]
    if last_event:
        lines.append(
            f"- Last human event: `{last_event.get('type', 'feedback')}` — "
            f"{last_event.get('text', '')}"
        )
    if candidates:
        lines.append("- Presented candidates: " + ", ".join(
            str(c.get("variant_id", "?")) for c in candidates if isinstance(c, dict)
        ))
    else:
        lines.append("- Presented candidates: none yet")
    recommendation = proposal.get("recommendation")
    if recommendation:
        lines.append(f"- Current recommendation: `{recommendation}`")
    lines.append("")
    lines.append(
        "Use visible text to answer human questions. Tool calls may follow when "
        "experimentation is needed, but presentation/approval is mediated by "
        "`present_for_review`, not by `commit_variant` or `advance_phase`."
    )
    return "\n".join(lines)


def _build_vlm_view(state: AstroState, messages: list) -> list:
    """
    Construct the message list to pass to model.invoke. Strips any historical
    image blocks (defensive) and appends a single fresh multimodal HumanMessage
    built from state.visual_context (filtered by mode and cap).

    This is the only function in agent_node that decides what the VLM sees.
    """
    if not (vlm_hitl() or vlm_autonomous()):
        return _strip_vlm_images(messages)

    refs = _select_visible_refs(state)
    cleaned = _strip_vlm_images(messages)
    if not refs:
        return cleaned

    paths = [r["path"] for r in refs]
    label_summary = ", ".join(r.get("label", "") for r in refs[:6])
    if len(refs) > 6:
        label_summary += f", … ({len(refs)} total)"
    label = f"Current visual context: {label_summary}"
    vlm_msg = _make_vlm_message(paths, label)
    if vlm_msg is None:
        return cleaned

    # Estimated visual payload size, used for token-burn observability.
    # Accurate for files we just read off disk; for missing files (never
    # happens after _make_vlm_message succeeds) we fall back to 0.
    total_bytes = 0
    for p in paths:
        try:
            total_bytes += Path(p).stat().st_size
        except OSError:
            pass

    logger.debug(
        f"vlm_view: showing {len(refs)} image(s), ~{total_bytes // 1024} KB on disk "
        f"({sum(1 for r in refs if r.get('source') == 'hitl_variant')} hitl_variant, "
        f"{sum(1 for r in refs if r.get('source') == 'current_image')} current_image, "
        f"{sum(1 for r in refs if r.get('source') == 'present_images')} present_images, "
        f"{sum(1 for r in refs if r.get('source') == 'phase_carry')} phase_carry)"
    )
    return cleaned + [vlm_msg]


# ── Stuck-loop detection ──────────────────────────────────────────────────────
# Hard fail if the agent calls the same tool N times in a row. This is a
# testing safety net — set MAX_CONSECUTIVE_SAME_TOOL in .env. 0 disables.


class StuckLoopError(RuntimeError):
    """Raised when the agent calls the same tool too many times consecutively."""


class TextLoopError(RuntimeError):
    """Raised when the agent emits too many text-only responses without progress."""


def _check_text_loop(messages: list) -> None:
    """
    Hard-fail when the agent emits N consecutive text-only AIMessages with
    no HumanMessage and no tool calls in between. Catches the "agent
    talking to itself" pattern that the tool-based stuck-loop detector
    misses (M20 trace: 8 text-only responses in a row narrating the same
    image comparison, no tool calls, no human input, until the model API
    rejected the bloated payload).

    Why "no HumanMessage between" is the right signal:
      During an HITL gate the agent CAN legitimately produce many
      text-only responses while conversing — but those interleave with
      HumanMessages from the user. A run of text-only AIMessages with
      no HumanMessage interruption means the agent is running its own
      loop, not collaborating. We don't gate on similarity because the
      agent's phrasing varies even when it's stuck, and Jaccard /
      n-gram measures are too sensitive to word-choice differences.

    Tunables (env vars):
      MAX_CONSECUTIVE_TEXT_ONLY — N (default 5, 0 disables)
    """
    import os
    env_n = os.environ.get("MAX_CONSECUTIVE_TEXT_ONLY", "")
    if env_n.strip() == "0":
        return
    try:
        max_n = int(env_n) if env_n else 5
    except ValueError:
        max_n = 5
    if max_n <= 1:
        return

    # Walk the tail backward. Count text-only AIMessages until we hit:
    #   - A HumanMessage (conversation interleave — resets the run)
    #   - An AIMessage with tool_calls (agent did something — resets)
    #   - A ToolMessage (tool ran — resets)
    run_texts: list[str] = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                break
            text = str(msg.content or "").strip()
            if text:
                run_texts.append(text)
            # else: empty AIMessage doesn't count toward the run but also
            # doesn't reset it — keep walking.
        elif isinstance(msg, (HumanMessage, ToolMessage)):
            break

    if len(run_texts) < max_n:
        return

    raise TextLoopError(
        f"Agent has emitted {len(run_texts)} consecutive text-only "
        f"responses (limit {max_n}) without a human reply or a tool call. "
        f"This pattern is the narration-loop the tool-based stuck-loop "
        f"detector doesn't catch — the agent is talking to itself "
        f"instead of advancing the work. Most recent response "
        f"(first 200 chars):\n"
        f"  {run_texts[0][:200]!r}\n"
        f"To override, set MAX_CONSECUTIVE_TEXT_ONLY=0 in .env. "
        f"To raise the trigger threshold, set MAX_CONSECUTIVE_TEXT_ONLY=N."
    )


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


# Markers that identify a ToolMessage as a failure. Successful tools return
# JSON (dict/list) in their content; failing tools yield free-form error text
# from _format_tool_error or raw exception strings. We check:
#   1. content is a dict/list with "error" or "success": false
#   2. OR content is a string that matches any known failure prefix
# Substring matching is sufficient because _format_tool_error standardizes
# on these prefixes and SirilError messages contain "siril-cli exited".
_TOOL_ERROR_MARKERS: tuple[str, ...] = (
    "Tool '",                 # "Tool 'X' failed ..." from _format_tool_error
    "Error:",                 # generic error prefix
    "Error in line ",         # Siril script error
    "siril-cli exited",       # SirilError
    "Traceback (most recent", # unraised exceptions leaking through
    "validation error",       # pydantic validation
    "FileNotFoundError",      # raised from tool bodies
    "RuntimeError",           # raised from tool bodies
    "ValueError",             # raised from tool bodies
    "with an internal error", # _format_tool_error fallback
)


def _tool_message_is_error(msg, content: str) -> bool:
    """
    Classify a ToolMessage as a failure for the stuck-loop detector.

    Trusts `status='error'` when the underlying ToolNode set it (LangChain
    ≥0.2 does this for raised exceptions). Falls back to content inspection
    for older versions or custom handlers that only set the content string.

    The JSON path mirrors how successful tools report: JSON-parseable content
    is almost always success. Non-JSON content that starts with a known error
    marker is treated as failure.
    """
    status = getattr(msg, "status", None)
    if status == "error":
        return True

    if not content:
        return False

    # Successful tools return JSON. If parseable, only treat as error when the
    # JSON itself declares failure.
    stripped = content.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            if parsed.get("error"):
                return True
            if parsed.get("success") is False:
                return True
            return False
        if parsed is not None:
            return False
        # JSON-looking but unparseable — fall through to marker check.

    return any(marker in content for marker in _TOOL_ERROR_MARKERS)


def _check_stuck_loop(messages: list) -> None:
    """
    Detect repeated identical tool calls within the current segment.

    A "segment" is the run of agent activity since the most recent reset point:
    either the last successful advance_phase (new phase = clean slate) or the
    last HumanMessage (human gave new direction = clean slate). Within a
    segment, every (name, args) fingerprint is counted; if any single
    fingerprint appears `MAX_CONSECUTIVE_SAME_TOOL` or more times, the agent
    is stuck and StuckLoopError is raised.

    The previous version only flagged STRICTLY CONSECUTIVE repeats and missed
    the common stuck pattern of `[curves(X), satu(Y), curves(X), satu(Y),
    curves(X)]` — the agent thinks it's iterating but really it's re-applying
    the same parameters with no-op work in between. The counter-based check
    catches this regardless of interleaving.

    Effect-level collapsing: some tools report `"noop": true` in their result
    JSON to signal that the call produced no state change (currently
    restore_checkpoint does this when the bookmarked path already equals
    current_image). Agents in a stuck loop often try *different argument
    values* — restore_checkpoint("starless_base"), restore_checkpoint("v1"),
    restore_checkpoint("good") — all noops pointing at the same stale
    current_image. Args differ, so the args-only fingerprint never trips.
    To catch this, any tool call whose ToolMessage carries `"noop": true` is
    fingerprinted by effect (`{name, effect=noop}`) rather than by args, so
    all such noops for a given tool name share one counter.

    Error-level collapsing: the same class of stuck loop happens when a tool
    *keeps failing*. The M31 run exhibited this: siril_stack failed on every
    attempt while select_frames succeeded with varying thresholds between
    each retry, so the args-only fingerprint never saw two matching args.
    We now collapse any failing ToolMessage into `{name, effect=error}`
    regardless of args, which catches the "try three different parameter
    sets, all produce Siril errors" pattern without tripping on a single
    transient failure.

    Note: the env var name is historical — the semantic is now "max identical
    invocations within the current segment", not "max consecutive".

    Calling build_masters(file_type="bias") then build_masters(file_type="dark")
    still passes, because the args differ → distinct fingerprints → distinct
    counts. Only repeated calls with the *same* args (or the *same effect,*
    for noop-reporting tools) trip the detector.
    """
    import json
    import os
    from collections import Counter

    # Resolution order: env var override → processing.toml [limits] → hardcoded.
    # The CLI path never sets the env var (gradio_app does), so relying solely
    # on os.environ left the detector disabled for every CLI run, which is how
    # the M42 Sonnet run was able to retry siril_register a dozen times after
    # a cfitsio path error without ever tripping the guard. Falling back to
    # the toml value restores parity between the two entry points.
    env_limit = os.environ.get("MAX_CONSECUTIVE_SAME_TOOL", "")
    if env_limit:
        limit = int(env_limit)
    else:
        try:
            from muphrid.config import _pcfg
            limit = int(_pcfg("limits", "max_consecutive_same_tool", 3))
        except Exception:
            # If config loading fails for any reason, fall back to the
            # hardcoded default rather than silently disabling the detector.
            limit = 3
    if limit <= 0:
        return

    # First pass: collect tool_call_ids whose result was a no-op or an error.
    # We key by id so the AIMessage pass can recognize them without walking
    # messages twice in lockstep. The reverse walk still stops at
    # advance_phase / HumanMessage boundaries — events outside the current
    # segment must not leak in.
    noop_tool_call_ids: set[str] = set()
    error_tool_call_ids: set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if getattr(msg, "name", None) == "advance_phase":
                break
            tc_id = getattr(msg, "tool_call_id", None)
            content = msg.content if isinstance(msg.content, str) else ""
            if '"noop": true' in content or '"noop":true' in content:
                # Confirm by parsing — the substring match is a cheap prefilter.
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("noop") is True:
                    if tc_id:
                        noop_tool_call_ids.add(tc_id)
            elif tc_id and _tool_message_is_error(msg, content):
                error_tool_call_ids.add(tc_id)
            continue
        if isinstance(msg, HumanMessage):
            break

    counts: Counter[str] = Counter()
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # advance_phase marks the start of a new phase = clean slate
            if getattr(msg, "name", None) == "advance_phase":
                break
            continue  # other tool results don't affect the count
        if isinstance(msg, HumanMessage):
            # Human intervention resets — re-application after feedback is OK
            break
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id in noop_tool_call_ids:
                    # Effect-level fingerprint: collapse all noop calls to the
                    # same tool (regardless of args) into one counter. This is
                    # what catches "try a different checkpoint name each time"
                    # loops where every attempt resolves to the same file.
                    fingerprint = json.dumps(
                        {"name": tc["name"], "effect": "noop"},
                        sort_keys=True,
                    )
                elif tc_id in error_tool_call_ids:
                    # Error-level fingerprint: collapse all failing calls to
                    # the same tool into one counter. Catches oscillation like
                    # select_frames(a) → siril_stack(x, fails) → select_frames(b)
                    # → siril_stack(y, fails) where args vary on each retry but
                    # the tool keeps failing. The noop counter protects against
                    # silent no-op loops; this protects against noisy failure
                    # loops. Pairs naturally with the existing args-based path
                    # for successful calls.
                    fingerprint = json.dumps(
                        {"name": tc["name"], "effect": "error"},
                        sort_keys=True,
                    )
                else:
                    fingerprint = json.dumps(
                        {"name": tc["name"], "args": tc.get("args", {})},
                        sort_keys=True,
                    )
                counts[fingerprint] += 1
        # Text-only AIMessages are part of the segment (analysis between
        # tool calls), they don't reset the counter.

    if not counts:
        return

    # Find the fingerprint that's been called the most
    most_common, count = counts.most_common(1)[0]
    if count >= limit:
        info = json.loads(most_common)
        if info.get("effect") == "noop":
            raise StuckLoopError(
                f"Agent called '{info['name']}' {count} times with different "
                f"arguments but every result was a no-op (limit: {limit}) — "
                f"aborting. The tool keeps resolving to the same underlying "
                f"state regardless of the argument value, which means the "
                f"problem is upstream of this tool (e.g. current_image is "
                f"stale, or the checkpoints all point at the same file). "
                f"Do not retry with yet another argument — branch: inspect "
                f"paths.current_image, promote the correct file with a "
                f"different tool, or send a feedback message to reset the "
                f"segment. Set MAX_CONSECUTIVE_SAME_TOOL=0 in .env to "
                f"disable this check."
            )
        if info.get("effect") == "error":
            raise StuckLoopError(
                f"Agent called '{info['name']}' {count} times and every "
                f"result was an error (limit: {limit}) — aborting. Varying "
                f"arguments have not resolved the failure, which means the "
                f"problem is upstream of this tool's inputs (e.g. the "
                f"sequence on disk is malformed, a prior step produced "
                f"inconsistent state, or an environmental precondition is "
                f"missing). Do not retry '{info['name']}' again with yet "
                f"another parameter set — inspect the last tool output for "
                f"the actual error, fix the precondition with a different "
                f"tool, or send a feedback message to reset the segment. "
                f"Set MAX_CONSECUTIVE_SAME_TOOL=0 in .env to disable this "
                f"check."
            )
        raise StuckLoopError(
            f"Agent called '{info['name']}' {count} times with identical "
            f"arguments {info['args']} within the current segment "
            f"(limit: {limit}) — aborting. Other tool calls interleaved "
            f"between these do not unwind the count; the agent is "
            f"re-applying the same operation without meaningful change. "
            f"If you intended this re-application, send a feedback message "
            f"to reset the segment. Set MAX_CONSECUTIVE_SAME_TOOL=0 in "
            f".env to disable this check."
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


# Sentinel content used when a model returns an AIMessage with neither text
# nor tool_calls. Persisting an empty AIMessage poisons subsequent calls
# regardless of backend: OpenAI-compatible providers (Together, OpenAI,
# DeepSeek) reject with HTTP 400 "Input validation error", and Anthropic's
# Messages API rejects with "all messages must have non-empty content" — the
# rule is a wire-format invariant, not a provider quirk. Substituting a
# non-empty placeholder keeps the conversation valid for any backend while
# still signalling to the agent_chat nudger that this turn was a no-op.
_EMPTY_RESPONSE_PLACEHOLDER = (
    "(no response — I should call a tool or respond with text on the next turn)"
)


def _is_empty_ai_message(msg: AIMessage) -> bool:
    """True iff the AIMessage has no usable content AND no tool_calls."""
    content = msg.content
    if isinstance(content, str):
        text_empty = not content.strip()
    elif isinstance(content, list):
        # Multi-part content (text + image blocks): empty iff every part is
        # an empty string or a non-text part. We're permissive with non-text
        # blocks because removing those would change semantics we don't own.
        text_empty = all(
            (isinstance(part, str) and not part.strip()) or
            (isinstance(part, dict) and (
                part.get("type") == "text" and not (part.get("text") or "").strip()
            ))
            for part in content
        )
    else:
        text_empty = not content
    return text_empty and not (msg.tool_calls or [])


def _normalize_empty_response(response: AIMessage) -> AIMessage:
    """
    Replace empty AIMessage(content='', tool_calls=[]) with a placeholder
    so the conversation never contains a malformed assistant turn.

    Why: every chat completion API in use rejects conversations that
    contain an assistant turn with no content and no tool_calls. The
    OpenAI-compatible providers return HTTP 400 "Input validation error";
    Anthropic's Messages API returns "all messages must have non-empty
    content". The moment one such turn is persisted, every subsequent
    invoke on that thread fails until the message is surgically removed
    from the checkpoint. Normalizing at the source means state never
    holds a turn that would poison the next call, regardless of which
    backend is configured.

    This pairs with `_strip_empty_ai_messages` (defense-in-depth, applied
    to the messages list before invoke) so existing poisoned checkpoints
    can recover.
    """
    if not isinstance(response, AIMessage):
        return response
    if not _is_empty_ai_message(response):
        return response
    logger.warning(
        "agent_node: model returned empty response (no content, no tool_calls) — "
        "substituting placeholder to keep conversation valid for next invoke"
    )
    return AIMessage(
        content=_EMPTY_RESPONSE_PLACEHOLDER,
        tool_calls=[],
        # Preserve any other attributes the model attached (id, etc.) so
        # tracing/logging stays consistent.
        additional_kwargs=getattr(response, "additional_kwargs", None) or {},
        response_metadata=getattr(response, "response_metadata", None) or {},
    )


def _strip_empty_ai_messages(messages: list) -> list:
    """
    Remove any AIMessage with empty content AND no tool_calls from the
    messages list before sending to the model.

    This is a recovery path for checkpoints that were already poisoned
    before `_normalize_empty_response` existed (or in case it's bypassed
    by a future code path). The empty turn stays in the persisted state
    for audit; we just hide it from the model on the way out.
    """
    out = []
    dropped = 0
    for m in messages:
        if isinstance(m, AIMessage) and _is_empty_ai_message(m):
            dropped += 1
            continue
        out.append(m)
    if dropped:
        logger.info(
            f"_strip_empty_ai_messages: filtered {dropped} empty AIMessage(s) "
            f"from invoke input (state preserved; only the wire payload is "
            f"sanitized)"
        )
    return out


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

        # Text-loop detection: hard fail if the agent emits N consecutive
        # text-only responses with high pairwise similarity. Catches the
        # narration-loop pattern that the tool-based stuck-loop detector
        # misses (observed in M20 trace: 8 near-identical "Top: teal /
        # Bottom: bronze" responses without progress, eventually crashing
        # the model API on token-limit overflow).
        _check_text_loop(raw_messages)

        # Per-phase tool call cap: count tool calls since the last advance_phase
        # and fail if the limit is exceeded.
        _check_phase_tool_limit(raw_messages, phase)

        # Build message list with system prompt — read phase prompt directly
        # from PHASE_PROMPTS so it's always in sync with the current phase
        # (advance_phase tool updates state["phase"] mid-loop).
        phase_prompt = _PHASE_PROMPTS.get(phase, "")
        system = f"{_SYSTEM_BASE}\n\n{phase_prompt}"

        # Surface state.variant_pool to the agent via the system prompt.
        # The pool is dynamic (changes turn to turn), so it lives in a
        # SEPARATE content block from the cached system prompt — the cached
        # prefix stays byte-stable while the pool section refreshes each turn.
        variant_pool_section = _format_variant_pool_for_prompt(
            state.get("variant_pool", []) or []
        )
        checkpoints_section = _format_checkpoints_for_prompt(
            (state.get("metadata", {}) or {}).get("checkpoints")
        )
        review_session_section = _format_review_session_for_prompt(
            state.get("review_session")
        )

        # HITL collaboration fragment — appended only when the agent is at an
        # active HITL gate. Lives in its own dynamic block so the cached
        # prefix doesn't churn when the gate opens or closes mid-phase.
        hitl_fragment = (
            _HITL_PARTNER_FRAGMENT
            if review_ctl.active_review_session(state)
            else ""
        )

        # Anthropic prompt caching: use content blocks with cache_control
        # so the stable prefix is cached across calls within a phase.
        # additional_kwargs doesn't work — must use inline content blocks.
        _is_anthropic = _check_anthropic(model)

        if _is_anthropic:
            system_blocks: list[dict] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
            ]
            if variant_pool_section:
                # Uncached: regenerated each turn from live state
                system_blocks.append({"type": "text", "text": variant_pool_section})
            if checkpoints_section:
                # Uncached: regenerated each turn from live checkpoint state
                system_blocks.append({"type": "text", "text": checkpoints_section})
            if hitl_fragment:
                # Uncached: only present during active HITL conversations
                system_blocks.append({"type": "text", "text": hitl_fragment})
            if review_session_section:
                system_blocks.append({"type": "text", "text": review_session_section})
            messages = [SystemMessage(content=system_blocks)] + raw_messages

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
            full_system = system
            if variant_pool_section:
                full_system = f"{full_system}\n\n{variant_pool_section}"
            if checkpoints_section:
                full_system = f"{full_system}\n\n{checkpoints_section}"
            if hitl_fragment:
                full_system = f"{full_system}\n\n{hitl_fragment}"
            if review_session_section:
                full_system = f"{full_system}\n\n{review_session_section}"
            messages = [SystemMessage(content=full_system)] + raw_messages

        # VLM view: state.visual_context owns visibility. See _build_vlm_view.
        messages = _build_vlm_view(state, messages)

        # Prune analysis outputs from completed phases — the model's reasoning
        # captured the conclusions; raw JSON is dead weight after phase ends.
        messages = _prune_phase_analysis(messages)

        # Defense-in-depth: filter out any prior empty AIMessage (no content,
        # no tool_calls) before invoking the model. Both OpenAI-compatible
        # providers (Together, OpenAI, DeepSeek) and Anthropic reject
        # conversations that contain such turns. A single empty AI turn
        # poisoned by a previous agent stall would block every subsequent
        # invocation on this thread until manually surgically removed from
        # the checkpoint. Stripping at invoke time recovers gracefully.
        # Note: this only filters; the AIMessage stays in checkpoint state
        # so the audit trail is preserved. We just don't show it to the
        # model. Pair with the post-invoke normalization below — together
        # they prevent the bad message from being persisted in the first
        # place AND let us recover existing checkpoints.
        messages = _strip_empty_ai_messages(messages)

        response = model.invoke(messages)
        response = _rescue_raw_tool_calls(response)
        response = _normalize_empty_response(response)

        review_session_update: dict[str, Any] = {}
        if review_ctl.visible_response_required(state.get("review_session")):
            if response.tool_calls and not review_ctl.ai_message_has_visible_text(response):
                logger.info(
                    "HITL turn policy rejected a tool-only response after "
                    "human review feedback; routing back to agent for visible text."
                )
                return {
                    "messages": [
                        review_ctl.build_visible_response_required_prompt(
                            state.get("review_session")
                        )
                    ]
                }
            if review_ctl.ai_message_has_visible_text(response):
                review_session_update["review_session"] = review_ctl.update_review_session(
                    state.get("review_session"),
                    visible_response_required=False,
                )

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
                )], **review_session_update}

        logger.info(
            f"Agent response: {'tool_calls' if response.tool_calls else 'no tool_calls'}"
            + (f" ({[tc['name'] for tc in response.tool_calls]})" if response.tool_calls else f" | text: {str(response.content)[:300]!r}")
        )

        return {"messages": [response], **review_session_update}

    return agent_node


# ── action ────────────────────────────────────────────────────────────────────
# We use LangGraph's prebuilt ToolNode which handles tool execution and
# returns ToolMessages. Initialized with all_tools() — the phase_router
# controls which tools the LLM can *call* (via bind_tools), but the
# ToolNode can execute any of them.

_POST_STACK_CHECKPOINT_PHASES = {
    ProcessingPhase.LINEAR,
    ProcessingPhase.STRETCH,
    ProcessingPhase.NONLINEAR,
}

_IMAGE_MUTATING_TOOLS = {
    "remove_gradient",
    "color_calibrate",
    "remove_green_noise",
    "noise_reduction",
    "deconvolution",
    "stretch_image",
    "select_stretch_variant",
    "star_removal",
    "curves_adjust",
    "local_contrast_enhance",
    "saturation_adjust",
    "hsv_adjust",
    "star_restoration",
    "reduce_stars",
    "multiscale_process",
    "pixel_math",
    "masked_process",
    "hdr_composite",
}


def _checkpoint_safe_tool_label(tool_names: list[str]) -> str:
    joined = "_".join(tool_names[:3]) if tool_names else "tool"
    return re.sub(r"[^a-zA-Z0-9_]+", "_", joined).strip("_") or "tool"


def auto_checkpoint(state: AstroState) -> dict[str, Any]:
    """
    Automatically bookmark current_image before post-stack image mutations.

    This is deliberately metadata-only and runs immediately before ToolNode.
    It must not append a message, because ToolNode expects the latest message
    to be the AIMessage containing tool_calls.
    """
    phase = state.get("phase", ProcessingPhase.INGEST)
    if phase not in _POST_STACK_CHECKPOINT_PHASES:
        return {}

    messages = state.get("messages", []) or []
    if not messages or not isinstance(messages[-1], AIMessage):
        return {}

    tool_calls = messages[-1].tool_calls or []
    mutating_tools = [
        tc.get("name", "")
        for tc in tool_calls
        if tc.get("name") in _IMAGE_MUTATING_TOOLS
    ]
    if not mutating_tools:
        return {}

    current_image = (state.get("paths", {}) or {}).get("current_image")
    if not current_image:
        return {}
    path = Path(current_image)
    if not path.exists():
        logger.warning(
            f"auto_checkpoint skipped: current_image does not exist: {current_image}"
        )
        return {}

    checkpoints = (state.get("metadata", {}) or {}).get("checkpoints") or {}
    auto_count = sum(
        1
        for name in checkpoints
        if str(name).startswith("auto:") and str(name) != "auto:previous"
    )
    label = _checkpoint_safe_tool_label(mutating_tools)
    phase_value = getattr(phase, "value", str(phase))
    unique_name = f"auto:{phase_value}:before_{label}:{auto_count + 1:03d}"

    logger.info(
        f"auto_checkpoint: {unique_name} → {path.name} "
        f"before {', '.join(mutating_tools)}"
    )
    return {
        "metadata": {
            "checkpoints": {
                unique_name: str(path),
                "auto:previous": str(path),
            }
        }
    }


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

    Pure encoding helper — takes resolved JPG/PNG paths and produces a
    LangChain content-block list wrapped in a HumanMessage. Skips raw FITS
    (cannot be base64-encoded for vision models). Returns None if no valid
    images were encoded.

    Called by _build_vlm_view at every model.invoke. The label is the only
    text the agent sees alongside the images.
    """
    content: list[dict] = [{"type": "text", "text": f"[VLM] {label}"}]
    has_image = False

    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            continue
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


# ── variant_snapshot ──────────────────────────────────────────────────────────
#
# Runs after the action node, before hitl_check. Detects HITL-mapped tool
# executions and snapshots them into state.variant_pool. Each captured variant
# is a stable record of "what the agent tried" — params, output file, metrics
# at the moment of capture. The pool accumulates across iterations within a
# HITL gate and is cleared on phase advance or variant commit.
#
# Side effect per HITL-tool execution: append a Variant entry to
# state.variant_pool. The agent sees a formatted pool summary through the
# dynamic system prompt on its next turn.
#
# Behavior is mode-agnostic: the pool builds in both HITL and autonomous modes.
# In autonomous mode it's pure observability; in HITL mode it powers the UI's
# variant panel and is the surface the human approves from.
#
# File semantics: variants reference the tool's actual output_path. Most HITL
# tools (e.g. remove_gradient) auto-name outputs by parameter, so each variant
# is naturally a distinct file. For tools that overwrite, multiple pool entries
# may share a file_path — this degrades gracefully (one approvable artifact
# from the user's view) without breaking the protocol.

# Keys to probe for the variant's primary file path, in priority order. Mirrors
# _IMAGE_PATH_KEYS in hitl.py but lives here so this module is self-contained.
_VARIANT_FILE_KEYS = (
    "output_path",
    "result_path",
    "stretched_image_path",
    "starless_image_path",
    "exported_path",
    "mask_path",
)

# A small slice of metrics worth snapshotting alongside each variant. Captured
# from current state at variant creation time so the agent and the UI can
# compare variants without re-reading files. Most are populated by analyze_image
# or by the HITL tool itself; missing keys are fine.
_VARIANT_METRIC_KEYS = (
    "gradient_magnitude",
    "snr_estimate",
    "background_flatness",
    "channel_imbalance",
    "green_excess",
    "signal_coverage_pct",
    "current_fwhm",
    "current_noise",
    "current_background",
    "star_count",
    "contrast_ratio",
)


def _phase_short_code(hitl_key: str | None) -> str:
    """
    Extract the 'T09'-style prefix from a hitl_key like 'T09_gradient'.
    Falls back to 'TXX' if the key doesn't follow the convention.
    """
    if not hitl_key:
        return "TXX"
    head = hitl_key.split("_", 1)[0]
    return head if head.startswith("T") else "TXX"


def _find_ai_message_for_tool_call(messages: list, tool_call_id: str) -> AIMessage | None:
    """Walk backward through messages to find the AIMessage that issued tool_call_id."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.get("id") == tool_call_id:
                    return msg
    return None


def _extract_variant_params(tool_msg: ToolMessage, ai_msg: AIMessage | None) -> dict:
    """
    Best-effort extraction of the params used for this variant.

    Priority:
      1. Tool's own 'settings' field in the result JSON (flat, tool-curated)
      2. AIMessage tool_call args (nested but always available)
      3. Empty dict (degraded but non-fatal)
    """
    # Try parsing the tool result for a 'settings' block
    try:
        result = json.loads(text_content(tool_msg.content))
        if isinstance(result, dict) and isinstance(result.get("settings"), dict):
            return result["settings"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to the AI message tool_call args
    if ai_msg is not None:
        for tc in ai_msg.tool_calls or []:
            if tc.get("id") == tool_msg.tool_call_id:
                args = tc.get("args", {})
                return args if isinstance(args, dict) else {}
    return {}


def _extract_variant_paths(tool_msg: ToolMessage) -> tuple[str | None, str | None]:
    """
    Pull the variant's primary file path and (optional) preview path from the
    tool result JSON. Returns (file_path, preview_path).
    """
    try:
        result = json.loads(text_content(tool_msg.content))
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(result, dict):
        return None, None

    file_path: str | None = None
    for key in _VARIANT_FILE_KEYS:
        if val := result.get(key):
            file_path = str(val)
            break

    preview_path = result.get("preview_path")
    return file_path, (str(preview_path) if preview_path else None)


def _extract_variant_label(tool_msg: ToolMessage, params: dict) -> str | None:
    """
    Use the tool's own variant_label if it provides one, else None and the
    caller will synthesize one from the params.
    """
    try:
        result = json.loads(text_content(tool_msg.content))
        if isinstance(result, dict):
            label = result.get("variant_label")
            return str(label) if label else None
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _format_param_summary(params: dict, max_items: int = 4) -> str:
    """
    Render a params dict as a compact one-line string for use in labels.
    Flattens one level of nesting, drops noise like None values.
    """
    flat: list[tuple[str, Any]] = []
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                if vv is None:
                    continue
                flat.append((kk, vv))
        else:
            flat.append((k, v))
    flat = flat[:max_items]
    return ", ".join(f"{k}={v}" for k, v in flat)


def _snapshot_metrics(state: AstroState) -> dict:
    """Capture a small relevant slice of metrics from current state."""
    metrics = state.get("metrics", {}) or {}
    snapshot = {}
    for key in _VARIANT_METRIC_KEYS:
        val = metrics.get(key)
        if val is not None:
            snapshot[key] = val
    return snapshot


def _make_variant(
    tool_msg: ToolMessage,
    ai_msg: AIMessage | None,
    state: AstroState,
    pool: list[Variant],
    phase: ProcessingPhase,
) -> Variant | None:
    """
    Build a Variant entry from a HITL-tool ToolMessage. Returns None if the
    tool result lacks a usable file path (treated as a non-variant — e.g. a
    failed tool call that the action node still produced a message for).
    """
    from datetime import datetime, timezone

    hitl_key = TOOL_TO_HITL.get(tool_msg.name)
    short = _phase_short_code(hitl_key)

    file_path, preview_path = _extract_variant_paths(tool_msg)
    if not file_path:
        return None

    params = _extract_variant_params(tool_msg, ai_msg)

    # Generate stable id: T09_v1, T09_v2, ...  (counts existing entries with
    # the same prefix; pool is per-gate so this stays small)
    same_phase = [v for v in pool if v.get("id", "").startswith(f"{short}_v")]
    n = len(same_phase) + 1
    variant_id = f"{short}_v{n}"

    # Label: prefer tool's own variant_label, else synthesize from params
    tool_label = _extract_variant_label(tool_msg, params)
    if tool_label:
        label = f"{short} v{n} — {tool_label}"
    else:
        param_str = _format_param_summary(params)
        label = f"{short} v{n} — {tool_msg.name}({param_str})" if param_str else f"{short} v{n} — {tool_msg.name}"

    return Variant(
        id=variant_id,
        phase=phase.value if hasattr(phase, "value") else str(phase),
        tool_name=tool_msg.name,
        label=label,
        params=params,
        file_path=file_path,
        preview_path=preview_path,
        metrics=_snapshot_metrics(state),
        created_at=datetime.now(timezone.utc).isoformat(),
        rationale=None,
    )


def _validate_variant_pool(pool: list[Variant]) -> tuple[list[Variant], list[str]]:
    """
    Drop variants whose backing files no longer exist on disk.

    Returns (kept, dropped_ids). Used to self-correct the pool on resume —
    after a session is reloaded from checkpoint, files referenced by old
    variant entries may have been wiped (cleanup_previous_runs) or manually
    deleted. The dangling entries get pruned silently rather than confusing
    the UI with broken Approve buttons.
    """
    kept: list[Variant] = []
    dropped: list[str] = []
    for v in pool:
        path = v.get("file_path")
        if path and Path(path).exists():
            kept.append(v)
        else:
            dropped.append(v.get("id", "?"))
    return kept, dropped


def _resolve_variant_preview(
    variant: Variant, working_dir: str, is_linear: bool
) -> str | None:
    """
    Return a JPG/PNG preview path for a variant. Prefers variant.preview_path
    when present and existing on disk; falls back to generating a preview from
    the FITS file via generate_preview. Returns None if neither works.
    """
    preview = variant.get("preview_path")
    if preview and Path(preview).exists():
        return preview
    fits_path = variant.get("file_path")
    if not fits_path or not Path(fits_path).exists():
        return None
    if not working_dir:
        return None
    from muphrid.tools.utility.t22_generate_preview import generate_preview
    p = Path(fits_path)
    expected = Path(working_dir) / "previews" / f"preview_{p.stem}.jpg"
    if expected.exists():
        return str(expected)
    try:
        result = generate_preview(
            working_dir=working_dir,
            fits_path=fits_path,
            format="jpg",
            quality=95,
            auto_stretch_linear=is_linear,
        )
        return result.get("preview_path")
    except Exception as e:
        logger.warning(f"variant preview generation failed for {p.name}: {e}")
        return None


def variant_snapshot(state: AstroState) -> dict[str, Any]:
    """
    Detect HITL-mapped tool executions in the latest action result and
    snapshot them into state.variant_pool.

    Runs between action and hitl_check. It is intentionally mode-agnostic:
    HITL uses the pool as workbench history behind review candidates, while
    autonomous runs use it so the agent can compare and commit variants
    without a human gate.

    The pool grounds HITL conversations, the Gradio workbench filmstrip, and
    autonomous commit_variant decisions. The agent also sees a compact pool
    summary in the dynamic system prompt.

    Side effect: also self-corrects the pool by dropping any entries whose
    backing files no longer exist on disk. This handles resume after the
    files were cleaned up between sessions.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Walk backward through trailing ToolMessages from the most recent action.
    # Stop at the first non-ToolMessage.
    trailing: list[ToolMessage] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            trailing.append(msg)
        else:
            break
    trailing.reverse()  # restore chronological order

    if not trailing:
        return {}

    # Filter to HITL-mapped tools only
    hitl_msgs = [m for m in trailing if m.name in TOOL_TO_HITL]
    if not hitl_msgs:
        return {}

    phase = state.get("phase", ProcessingPhase.INGEST)
    raw_pool = list(state.get("variant_pool", []) or [])

    # Self-correct: drop dangling entries before processing new captures
    pool, dropped = _validate_variant_pool(raw_pool)
    if dropped:
        logger.info(
            f"variant_snapshot: dropped {len(dropped)} dangling variant(s) "
            f"with missing files: {dropped}"
        )

    pool_changed = len(dropped) > 0
    existing_paths = {v.get("file_path") for v in pool}

    new_pool = pool
    snapshotted_any = False

    for msg in hitl_msgs:
        # Dedup on file_path: if a variant with the same output file is
        # already in the pool, we've already captured this tool call.
        ai_msg = _find_ai_message_for_tool_call(messages, msg.tool_call_id)
        candidate = _make_variant(msg, ai_msg, state, new_pool, phase)

        if candidate is not None and candidate["file_path"] not in existing_paths:
            new_pool = new_pool + [candidate]
            existing_paths.add(candidate["file_path"])
            snapshotted_any = True
            logger.info(
                f"variant_snapshot: captured {candidate['id']} "
                f"({msg.name}) → {Path(candidate['file_path']).name}"
            )

    # Persist the pool whenever it changed for any reason: new captures OR
    # dangling-entry pruning. Otherwise the next invocation would re-prune
    # the same dangling entries. variant_pool is the single source of truth
    # for HITL gate variants; _select_visible_refs reads it directly when
    # building the VLM view, so there is no separate mirror to maintain.
    if snapshotted_any or pool_changed:
        return {"variant_pool": new_pool}
    return {}


# ── variant promotion ────────────────────────────────────────────────────────


def find_variant_in_pool(pool: list[Variant], variant_id: str) -> Variant | None:
    """Look up a Variant by id. Pure helper."""
    for v in pool:
        if v.get("id") == variant_id:
            return v
    return None


def build_variant_promotion_update(
    state: AstroState, variant_id: str
) -> tuple[Variant, dict[str, Any]] | None:
    """
    Compute the state mutations for promoting a variant from variant_pool.
    Pure-ish function (one filesystem read for preview resolution); does not
    construct any messages — callers add the appropriate message wrapper for
    their context (HumanMessage for HITL approval, ToolMessage for autonomous
    commit_variant).

    Returns (variant, update_dict) on success, or None if the id isn't in
    the pool. The update dict contains:
      - paths.current_image := variant.file_path
      - variant_pool := []
      - visual_context := <existing> + phase_carry entry for the approved variant
      - metadata.last_committed_variant := {id, file_path}  (race-fix signal:
        lets commit_variant detect "already promoted via HITL" when the pool
        has been cleared and return idempotent success instead of an error)

    Shared by promote_variant (HITL) and the commit_variant tool (autonomous).
    """
    pool = state.get("variant_pool", []) or []
    variant = find_variant_in_pool(pool, variant_id)
    if variant is None:
        return None

    paths = dict(state.get("paths", {}) or {})
    paths["current_image"] = variant["file_path"]

    # Drop the now-promoted variant's hitl_variant entry (variant_pool is
    # being cleared, but visual_context may have older present_images /
    # phase_carry entries we want to preserve). Drop ANY existing
    # phase_carry entries too — they were a pre-#44 mechanism for keeping
    # a chosen variant visible, but the auto-current-image source now
    # projects paths.current_image automatically. Keeping a phase_carry
    # in addition causes a stale-comparison artifact: when the next tool
    # moves current_image to a new file, phase_carry still points at the
    # old variant's preview, the VLM sees both, and the agent gets stuck
    # narrating a now-irrelevant before/after. (Was #66 in the to-do list.)
    existing_visual = [
        r for r in (state.get("visual_context", []) or [])
        if r.get("source") not in ("hitl_variant", "phase_carry")
    ]

    update: dict[str, Any] = {
        "paths": paths,
        "variant_pool": [],
        "visual_context": existing_visual,
        # metadata is merged via _merge_dicts, so partial updates are safe —
        # other metadata fields are preserved and only last_committed_variant
        # is overwritten.
        "metadata": {
            "last_committed_variant": {
                "id": variant["id"],
                "file_path": variant["file_path"],
            },
        },
    }
    return variant, update


def promote_variant(
    state: AstroState, variant_id: str, rationale: str | None = None
) -> dict[str, Any] | None:
    """
    Promote a pool variant to the current image and clear the pool. Used by
    hitl_check on human approval. Wraps build_variant_promotion_update with
    HITL-specific extras: appends a HumanMessage announcing the approval
    (so the agent's next turn sees the commit) and clears active_hitl.

    Returns the dict update for AstroState, or None if variant_id isn't in
    the pool (caller decides how to surface the error).

    Note: this is a backend state mutation that happens during HITL resume,
    not via a tool call. It's the one place where current_image can move to
    a non-most-recent file via the human-approval path. The agent doesn't
    track "last tool output" beyond what's in messages, so its next tool call
    will read the promoted file correctly.
    """
    result = build_variant_promotion_update(state, variant_id)
    if result is None:
        return None
    variant, update = result

    # The HumanMessage is both a narrative cue AND a reset point for
    # _check_stuck_loop. Make the state change explicit so the agent does not
    # then call commit_variant to "lock it in" — that would be a redundant
    # round-trip, and the variant_pool is already empty so commit_variant
    # would otherwise fall into its error path. commit_variant's
    # already_committed guard handles this idempotently too, but the clearer
    # the signal here, the fewer ticks the agent burns.
    approval_lines = [
        "HUMAN APPROVED",
        f"Approved: {variant['id']} ({variant['label']})",
        "current_image has been promoted to the approved variant.",
        "The HITL gate is closed. Continue the pipeline from this image.",
        "Do not call commit_variant for this approval; the approval itself "
        "already committed the selected variant.",
    ]
    if rationale:
        approval_lines.append(f"Rationale: {rationale}")
    approval_lines.append("Continue to the next step.")
    approval_text = "\n".join(approval_lines)

    update["active_hitl"] = False
    update["review_session"] = review_ctl.close_review_session(
        state.get("review_session"),
        reason="variant_approved",
    )
    update["messages"] = [HumanMessage(content=approval_text)]

    logger.info(
        f"promote_variant: {variant['id']} → current_image "
        f"({Path(variant['file_path']).name})"
        + (f" with rationale" if rationale else "")
    )
    return update


# ── hitl_check ────────────────────────────────────────────────────────────────


def _present_for_review_succeeded(msg: ToolMessage) -> bool:
    """True when the present_for_review tool actually updated the review set."""
    try:
        payload = json.loads(text_content(msg.content))
    except Exception:
        return True
    return isinstance(payload, dict) and payload.get("status") == "presented"


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
    review_session = review_ctl.active_review_session(state)
    review_open = review_session is not None
    hitl_key, tool_name = resolve_hitl_checkpoint(messages)
    last = messages[-1] if messages else None

    # present_for_review is the collaboration handoff, not a normal utility
    # result to keep streaming past. Once the agent deliberately presents
    # candidates during an active gate, interrupt immediately so the human
    # can review before any more autonomous work happens.
    if (
        review_open
        and isinstance(last, ToolMessage)
        and last.name == "present_for_review"
        and _present_for_review_succeeded(last)
    ):
        hitl_key, tool_name = review_ctl.active_review_tool(state)
        if hitl_key is None:
            return {}

    # ── No HITL tool found in recent messages ────────────────────────
    if hitl_key is None:
        if not review_open:
            return {}  # no HITL mapping, no active conversation — pass through

        # Active review: agent called a non-HITL tool (analyze_image, present_images)
        # Let it pass through so the agent sees the result.
        if isinstance(last, ToolMessage) and last.name not in TOOL_TO_HITL:
            return {}

        # Agent responded with text — use the explicit review session.
        hitl_key, tool_name = review_ctl.active_review_tool(state)
        if hitl_key is None:
            return {}

    # ── HITL disabled for this tool (or autonomous mode) ─────────────
    if not is_enabled(hitl_key):
        if not review_open:
            return {}

        active_key, active_tool = review_ctl.active_review_tool(state)
        if active_key is None:
            return {}
        hitl_key, tool_name = active_key, active_tool

        if isinstance(last, ToolMessage) and last.name in TOOL_TO_HITL:
            return {}

    # ── HITL tool just executed — pass through for agent analysis ────
    # The agent hasn't seen the result yet. Let it analyze before we
    # fire the interrupt. This applies both to initial triggers AND
    # re-runs during active HITL (agent adjusted params after feedback).
    if (
        isinstance(last, ToolMessage)
        and last.name in TOOL_TO_HITL
        and TOOL_TO_HITL.get(last.name) == hitl_key
    ):
        if not review_open:
            # Initial trigger — inject analysis prompt
            logger.info(f"HITL triggered for {tool_name} — routing to agent for analysis")
            return {
                "active_hitl": True,
                "review_session": review_ctl.make_review_session(
                    state=state,
                    hitl_key=hitl_key,
                    tool_name=tool_name,
                ),
                "messages": [review_ctl.build_open_review_prompt(tool_name)],
            }
        else:
            # Re-run during active HITL.
            #
            # Default behavior: pass through and let the agent narrate the
            # new result before we interrupt. This keeps the comparison
            # workflow intact (agent runs N variants, narrates them, the
            # human reviews them all in one go with the agent's analysis).
            #
            # Safety net: count HITL-mapped tools after the latest human
            # review event using explicit review_session state. When the
            # count reaches the limit, ask the agent to curate and explain
            # the current pool before it keeps experimenting.
            #
            # EXTRACTION TRIGGER (intentionally inline today; tracked in
            # CLAUDE.md "Controller modules" doctrine):
            # If you find yourself writing a `tool_runs_since_human`-shaped
            # counter or a similar "N events since last human signal"
            # backstop ANYWHERE ELSE — autonomous-mode budget, a different
            # gate type, a different policy concern — extract these two
            # lines into helpers in `graph/review.py` first:
            #     increment_tool_runs_since_human(session) -> ReviewSession
            #     silent_tool_limit_reached(session, env_var) -> bool
            # The pattern (counter + env-driven limit + curation routing
            # on trip) repeats cleanly. The inline form here is fine for
            # one writer; a second writer means it's time to extract.
            import os
            session = state.get("review_session") or {}
            tool_count = int(session.get("tool_runs_since_human", 0) or 0) + 1
            silent_limit = int(os.environ.get("MAX_SILENT_HITL_TOOLS", "3"))
            updated_session = review_ctl.update_review_session(
                session,
                status="awaiting_agent_response",
                tool_runs_since_human=tool_count,
            )
            if silent_limit > 0 and tool_count >= silent_limit:
                logger.warning(
                    f"HITL tool-run backstop tripped: {tool_count} "
                    f"HITL-mapped tools have run since the latest human "
                    f"review event (limit: {silent_limit}) — routing to "
                    "curation before more autonomous work."
                )
                curated_state = dict(state)
                curated_state["review_session"] = updated_session
                return review_ctl.curation_update(
                    state=curated_state,
                    tool_name=tool_name,
                    pool=list(state.get("variant_pool", []) or []),
                )
            logger.info(
                f"HITL tool {tool_name} re-executed "
                f"(tool_runs_since_human={tool_count}/{silent_limit}) — "
                f"letting agent analyze new result"
            )
            return {"review_session": updated_session}

    # ── Agent has analyzed — fire interrupt ──────────────────────────
    cfg = tool_cfg(hitl_key)
    image_paths = images_from_tool(messages, tool_name)

    # Extract agent's analysis text (latest AIMessage)
    _agent_text = review_ctl.extract_latest_agent_text(messages)

    # Self-correct the pool before exposing it to the UI: drop any dangling
    # entries whose files have been removed (e.g., across a session restart
    # with cleanup_previous_runs=true). The pool drives the read-only
    # "history" panel; the proposal (below) drives the Approve buttons.
    raw_pool_for_payload = list(state.get("variant_pool", []) or [])
    payload_pool, dropped_at_payload = _validate_variant_pool(raw_pool_for_payload)
    if dropped_at_payload:
        logger.info(
            f"hitl_check: dropped {len(dropped_at_payload)} dangling variant(s) "
            f"before payload: {dropped_at_payload}"
        )

    # Build the proposal from the ReviewSession artifact. Entries whose
    # variant_id no longer exists in the pool are dropped — better an
    # empty/short proposal than a broken Approve button.
    proposal = review_ctl.proposal_entries_from_session(
        state.get("review_session"), payload_pool
    )

    needs_curation = review_ctl.review_readiness(payload_pool, proposal)[0]

    if review_ctl.should_route_for_curation(
        review_session=state.get("review_session"),
        pool=payload_pool,
        proposal=proposal,
        force_interrupt=False,
    ):
        logger.info(
            f"HITL gate for {tool_name} has pool variants but no presented "
            "candidates — routing back to agent for curation before interrupt."
        )
        return review_ctl.curation_update(
            state=state,
            tool_name=tool_name,
            pool=payload_pool,
        )

    payload = review_ctl.build_interrupt_payload(
        cfg=cfg,
        tool_name=tool_name,
        image_paths=image_paths,
        messages=messages,
        agent_text=_agent_text,
        pool=payload_pool,
        proposal=proposal,
        review_session=state.get("review_session"),
    )

    # VLM visibility is state-driven. variant_snapshot has already updated
    # variant_pool, and the next agent_node call will project it (plus any
    # visual_context entries) through _select_visible_refs into a fresh
    # multimodal view. No VLM injection happens in this node — messages
    # stay text-only.
    vlm_messages: list = []

    logger.info(f"HITL interrupt: {cfg['title']} (tool: {tool_name})")
    response = interrupt(payload)
    logger.info(f"HITL response: {response!r}")

    human_event = review_ctl.parse_human_event(response)
    response_text = human_event.get("text", str(response))

    # ── Variant approval ─────────────────────────────────────────────
    # The Gradio variant panel sends a typed event naming the
    # specific variant being committed plus an optional rationale from
    # the textbox. Promote the chosen variant's file to current_image
    # and clear the pool.
    if human_event.get("type") == "approve_variant":
        variant_id = human_event.get("variant_id")
        rationale = human_event.get("rationale", "")
        if variant_id is None:
            # Malformed payload — treat as feedback rather than approval
            logger.warning(f"Malformed variant approval payload: {response!r}")
            return {
                "messages": vlm_messages + [HumanMessage(content=response_text)],
                "active_hitl": True,
                "review_session": review_ctl.update_review_session(
                    state.get("review_session"),
                    status="awaiting_agent_response",
                    last_human_event=human_event,
                    tool_runs_since_human=0,
                    visible_response_required=True,
                ),
            }

        presented_ids = review_ctl.proposal_candidate_ids(state.get("review_session"))
        if payload_pool and variant_id not in presented_ids:
            logger.warning(
                f"Variant approval for {variant_id!r} was rejected because "
                "the variant is not in review_session.proposal"
            )
            return {
                "messages": vlm_messages + [HumanMessage(
                    content=(
                        f"[System] Variant {variant_id} is in the pool but "
                        "is not in the current review proposal. Ask "
                        "the agent to compare and present candidates before "
                        "approving."
                    )
                )],
                "active_hitl": True,
                "review_session": review_ctl.update_review_session(
                    state.get("review_session"),
                    status="awaiting_human_approval",
                    last_human_event=human_event,
                    tool_runs_since_human=0,
                    visible_response_required=True,
                ),
            }

        update = promote_variant(state, variant_id, rationale=rationale or None)
        if update is None:
            # Variant id not in pool — surface as feedback so the human
            # sees something went wrong without losing the conversation
            logger.warning(
                f"Variant approval for {variant_id!r} but no matching pool entry"
            )
            return {
                "messages": vlm_messages + [HumanMessage(
                    content=(
                        f"[System] Variant {variant_id} not found in the current "
                        f"pool — it may have been cleared by a phase advance. "
                        f"Please re-review and approve again."
                    )
                )],
                "active_hitl": True,
                "review_session": review_ctl.update_review_session(
                    state.get("review_session"),
                    status="awaiting_human_approval",
                    last_human_event=human_event,
                    tool_runs_since_human=0,
                    visible_response_required=True,
                ),
            }

        # Merge vlm_messages with the promotion update's messages
        update["messages"] = vlm_messages + update["messages"]
        return update

    # ── Bare approval (CLI / fallback) ───────────────────────────────
    if human_event.get("type") == "approve_current":
        if needs_curation:
            logger.warning(
                "Bare approval rejected because the active HITL gate has "
                "pool variants but no agent-presented candidates"
            )
            return {
                "messages": vlm_messages + [HumanMessage(
                    content=(
                        "[System] Approval is not available yet. The agent "
                        "must first select candidate(s) from the pool, explain "
                        "the tradeoffs, and call present_for_review. You can "
                        "ask it what to compare or tell it to iterate."
                    )
                )],
                "active_hitl": True,
                "review_session": review_ctl.update_review_session(
                    state.get("review_session"),
                    status="awaiting_curation",
                    last_human_event=human_event,
                    tool_runs_since_human=0,
                    visible_response_required=True,
                ),
            }

        note = human_event.get("rationale", "")
        approval_lines = [
            "HUMAN APPROVED",
            "Approved without naming a specific variant.",
            "The HITL gate is closed. Continue to the next step.",
            "Do not call commit_variant for this approval.",
        ]
        if note:
            approval_lines.insert(2, f"Rationale: {note}")
        approval_text = "\n".join(approval_lines)

        # Bare approval (CLI / fallback) — drop hitl_variant entries from
        # visual_context. No specific variant was named, so nothing to carry
        # forward as phase_carry.
        existing_visual = list(state.get("visual_context", []) or [])
        non_hitl = [r for r in existing_visual if r.get("source") != "hitl_variant"]
        return {
            "active_hitl": False,
            "review_session": review_ctl.close_review_session(
                state.get("review_session"),
                reason="bare_approval",
            ),
            "variant_pool": [],
            "visual_context": non_hitl,
            "messages": vlm_messages + [HumanMessage(content=approval_text)],
        }

    # Human gave feedback — keep in HITL loop
    return review_ctl.feedback_update(
        state=state,
        event=human_event,
        tool_name=tool_name,
        pool=payload_pool,
        proposal=proposal,
        prefix_messages=vlm_messages,
    )


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
        elif (
            isinstance(msg, HumanMessage)
            and (
                msg.additional_kwargs.get("is_nudge")
                or msg.additional_kwargs.get("is_hitl_turn_policy")
            )
        ):
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
        if (
            isinstance(last, HumanMessage)
            and getattr(last, "additional_kwargs", {}).get("is_hitl_turn_policy")
        ):
            return "agent"
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "action"

    # Active HITL conversation — agent answered a question, route back to
    # hitl_check which will re-fire interrupt() for the human to continue.
    if review_ctl.active_review_session(state):
        return "hitl_check"

    return "agent_chat"
