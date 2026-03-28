"""
Muphrid LangGraph — single agent with phase-gated tool binding.

Public API:
    build_graph()              — assemble the graph with a checkpointer
    build_graph_with_sqlite()  — convenience: graph + SqliteSaver

See graph_design.md for architecture, hitl_design.md for HITL design.
"""

from muphrid.graph.graph import build_graph, build_graph_with_sqlite
from muphrid.graph.state import AstroState, make_empty_state

__all__ = [
    "build_graph",
    "build_graph_with_sqlite",
    "AstroState",
    "make_empty_state",
]
