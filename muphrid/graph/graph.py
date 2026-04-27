"""
Graph assembly — wires nodes and edges into a compiled StateGraph.

    phase_router → agent → action → hitl_check → agent  (ReAct loop)
                     │
                     └── (no tool_calls) → phase_advance → phase_router

See graph_design.md for the full architecture.
"""

from __future__ import annotations

import logging
import os

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from muphrid.config import make_llm
from muphrid.graph.nodes import (
    agent_chat,
    hitl_check,
    make_action_node,
    make_agent_node,
    phase_router,
    route_after_agent,
    route_after_phase_router,
    variant_snapshot,
)
from muphrid.graph.registry import tools_for_phase
from muphrid.graph.state import AstroState, ProcessingPhase

logger = logging.getLogger(__name__)


# ── Model factory ─────────────────────────────────────────────────────────────


def _current_model_fingerprint() -> tuple[str, str]:
    """
    Fingerprint for detecting mid-session model/provider changes.

    The Gradio app mutates LLM_MODEL / LLM_PROVIDER via _apply_ui_settings
    AFTER the graph has been built, so the factory has to notice the change
    at session start and rebuild — otherwise calls continue to target the
    originally-bound client regardless of what the UI shows.
    """
    return (
        os.environ.get("LLM_MODEL", ""),
        os.environ.get("LLM_PROVIDER", ""),
    )


def _make_model_factory(base_model=None):
    """
    Return a callable that produces a model with phase-appropriate tools bound.

    The base model is created lazily and re-created whenever the LLM_MODEL /
    LLM_PROVIDER env vars change since the last build. This is critical for
    Gradio — the app builds the graph once at startup, but the user can
    switch models mid-session via the dropdown. Without env-fingerprint
    invalidation, `base_model` would stay frozen to whatever make_llm()
    resolved at graph build time.

    When `base_model` is passed explicitly (e.g. tests), invalidation is
    disabled — the caller owns the lifecycle.
    """
    pinned = base_model  # explicit override → no invalidation
    state: dict = {
        "base_model": pinned,
        "fingerprint": _current_model_fingerprint() if pinned is None else None,
        "cache": {},
    }

    def model_for_phase(phase: ProcessingPhase):
        # If no explicit override, check whether env vars have changed since
        # we last built the base model. Rebuild + invalidate per-phase cache
        # if so. Cheap comparison on two strings each call.
        if pinned is None:
            current_fp = _current_model_fingerprint()
            if state["base_model"] is None or current_fp != state["fingerprint"]:
                if state["base_model"] is not None:
                    logger.info(
                        f"LLM rebind: {state['fingerprint']} → {current_fp}"
                    )
                state["base_model"] = make_llm()
                state["fingerprint"] = current_fp
                state["cache"] = {}

        cache = state["cache"]
        if phase not in cache:
            tools = tools_for_phase(phase)
            base = state["base_model"]
            cache[phase] = base.bind_tools(tools) if tools else base
        return cache[phase]

    return model_for_phase


# ── Graph construction ────────────────────────────────────────────────────────


def build_graph(
    checkpointer=None,
    store=None,
    base_model=None,
):
    """
    Assemble and compile the full processing graph.

    Args:
        checkpointer: LangGraph checkpointer (SqliteSaver recommended).
                      Required for HITL interrupt/resume.
        store: LangGraph BaseStore for cross-thread long-term memory.
               Optional — graph works without it, just no long-term memory.
        base_model: Override the LLM instance (useful for testing).
    """
    model_factory = _make_model_factory(base_model)
    agent_node = make_agent_node(model_factory)
    action_node = make_action_node()

    builder = StateGraph(AstroState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    builder.add_node("phase_router", phase_router)
    builder.add_node("agent", agent_node)
    builder.add_node("action", action_node)
    builder.add_node("variant_snapshot", variant_snapshot)
    builder.add_node("hitl_check", hitl_check)
    builder.add_node("agent_chat", agent_chat)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("phase_router")

    # ── Edges ─────────────────────────────────────────────────────────────

    # agent → action (tool_calls) | hitl_check (active HITL) | agent_chat (text) | END
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "action": "action",
            "hitl_check": "hitl_check",
            "agent_chat": "agent_chat",
            "__end__": END,
        },
    )

    # action → variant_snapshot → hitl_check
    # variant_snapshot captures HITL-mapped tool outputs into the variant pool
    # and enriches their ToolMessages with a pool summary for the agent. It's
    # mode-agnostic — runs in both HITL and autonomous modes.
    builder.add_edge("action", "variant_snapshot")
    builder.add_edge("variant_snapshot", "hitl_check")

    # hitl_check → agent (passes through or returns with human feedback)
    builder.add_edge("hitl_check", "agent")

    # agent_chat → agent (human responded or autonomous nudge injected)
    builder.add_edge("agent_chat", "agent")

    # phase_router → agent | END (entry point routing)
    builder.add_conditional_edges(
        "phase_router",
        route_after_phase_router,
        {"agent": "agent", "__end__": END},
    )

    # ── Compile ───────────────────────────────────────────────────────────
    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )


# ── LangGraph Studio entry point ─────────────────────────────────────────────
# Studio expects a module-level CompiledGraph or a callable.
# build_graph() with no args compiles without checkpointer — Studio injects
# its own at runtime for interrupt/resume support.

graph = build_graph()


# ── Convenience: build with SqliteSaver ───────────────────────────────────────


def build_graph_with_sqlite(
    db_path: str = "checkpoints.db",
    store=None,
    base_model=None,
):
    """Build the graph with a SqliteSaver checkpointer for durable persistence."""
    checkpointer = SqliteSaver.from_conn_string(db_path)
    return build_graph(
        checkpointer=checkpointer,
        store=store,
        base_model=base_model,
    ), checkpointer
