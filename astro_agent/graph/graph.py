"""
Graph assembly — wires nodes and edges into a compiled StateGraph.

    phase_router → agent → action → hitl_check → agent  (ReAct loop)
                     │
                     └── (no tool_calls) → phase_advance → phase_router

See graph_design.md for the full architecture.
"""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from astro_agent.config import make_llm
from astro_agent.graph.nodes import (
    hitl_check,
    make_action_node,
    make_agent_node,
    phase_advance,
    phase_router,
    route_after_agent,
    route_after_phase_router,
)
from astro_agent.graph.registry import tools_for_phase
from astro_agent.graph.state import AstroState, ProcessingPhase


# ── Model factory ─────────────────────────────────────────────────────────────


def _make_model_factory(base_model=None):
    """
    Return a callable that produces a model with phase-appropriate tools bound.

    The base model is created once. For each phase, we call .bind_tools()
    with that phase's tool list. This is the dynamic tool binding pattern
    from the LangGraph docs.
    """
    if base_model is None:
        base_model = make_llm()

    # Cache bound models per phase to avoid re-binding on every call
    _cache: dict[ProcessingPhase, object] = {}

    def model_for_phase(phase: ProcessingPhase):
        if phase not in _cache:
            tools = tools_for_phase(phase)
            if tools:
                _cache[phase] = base_model.bind_tools(tools)
            else:
                _cache[phase] = base_model
        return _cache[phase]

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
    builder.add_node("hitl_check", hitl_check)
    builder.add_node("phase_advance", phase_advance)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("phase_router")

    # ── Edges ─────────────────────────────────────────────────────────────

    # agent → action (tool_calls) | hitl_check (active HITL chat) | phase_advance
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "action": "action",
            "hitl_check": "hitl_check",
            "phase_advance": "phase_advance",
        },
    )

    # action → hitl_check (every tool execution gets checked for HITL)
    builder.add_edge("action", "hitl_check")

    # hitl_check → agent (passes through or returns with human feedback)
    builder.add_edge("hitl_check", "agent")

    # phase_advance → phase_router (advance phase, then re-enter)
    builder.add_edge("phase_advance", "phase_router")

    # phase_router → agent | END (based on current phase)
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
