"""
Muphrid LangGraph — single agent with phase-gated tool binding.

Public API:
    build_graph() — assemble the graph with a checkpointer
    build_graph_with_sqlite() — convenience: graph + SqliteSaver
    AstroState — typed graph state
    make_empty_state() — fresh state factory

Re-exports are loaded lazily so that importing a sibling submodule (e.g.
``muphrid.graph.state`` from a tool module) does not eagerly pull in
``graph.py`` → ``nodes.py`` → ``registry.py``. Registry import-time guards
still fire on first access to ``build_graph`` from a real entry point.

See graph_design.md for architecture, hitl_design.md for HITL design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from muphrid.graph.graph import build_graph, build_graph_with_sqlite
    from muphrid.graph.state import AstroState, make_empty_state

__all__ = [
    "build_graph",
    "build_graph_with_sqlite",
    "AstroState",
    "make_empty_state",
]


def __getattr__(name: str):
    if name in ("build_graph", "build_graph_with_sqlite"):
        from muphrid.graph.graph import build_graph, build_graph_with_sqlite
        return {"build_graph": build_graph, "build_graph_with_sqlite": build_graph_with_sqlite}[name]
    if name in ("AstroState", "make_empty_state"):
        from muphrid.graph.state import AstroState, make_empty_state
        return {"AstroState": AstroState, "make_empty_state": make_empty_state}[name]
    raise AttributeError(f"module 'muphrid.graph' has no attribute {name!r}")
