"""
T33 — Long-term memory search tool.

Agent-facing tool for searching past processing memories. Available in ALL
phases as a utility tool (when memory is enabled).

Design informed by:
  - Lesson #1: agentic reads — agent decides when to search
  - Lesson #8: epistemic separation — results show source provenance
  - Lesson #11: retrieval quality > write quality — this is the high-ROI piece

The system prompt encourages the agent to search memory at phase starts and
when stuck, but the agent has full autonomy over when and what to search.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from astro_agent.graph.state import AstroState

# Lazy import to avoid circular dependency — the memory store is initialized
# at startup and injected via module-level reference.
_MEMORY_STORE = None


def set_memory_store(store):
    """Called at startup to inject the active MemoryStore instance."""
    global _MEMORY_STORE
    _MEMORY_STORE = store


@tool
def memory_search(
    query: str,
    memory_type: str = "",
    limit: int = 5,
) -> str:
    """
    Search long-term memory for relevant past processing experience.

    Use this tool to recall what worked, what failed, and what the user
    preferred in past sessions. Search before making decisions on unfamiliar
    targets, when starting a new phase, or when stuck on a difficult problem.

    Examples:
      - "stretching compressed Bayer data on emission nebula"
      - "noise reduction parameters for low SNR data"
      - "what went wrong with star removal on galaxies"
      - "user preferences for saturation on nebulae"
      - "M42" or "ASI2600MC" for exact target/sensor matches

    Args:
        query: Natural language search query describing what you want to recall.
        memory_type: Optional filter — "observations", "failures", or "preferences".
                     Leave empty to search all types.
        limit: Maximum number of results to return (default 5).
    """
    if _MEMORY_STORE is None:
        return "Long-term memory is not available in this session."

    try:
        results = _MEMORY_STORE.search(
            query=query,
            limit=limit,
            memory_type=memory_type if memory_type else None,
        )
    except Exception as e:
        return f"Memory search failed: {e}"

    if not results:
        return f"No relevant memories found for: {query}"

    return _format_results(results, query)


def _format_results(results: list[dict], query: str) -> str:
    """
    Format search results as natural language the agent can reason about.

    Each result includes type, source, confidence, session context, and
    the memory content. The agent sees provenance (source=hitl vs phase_gate)
    so it can weigh reliability.
    """
    lines = [f"Found {len(results)} relevant memories for \"{query}\":\n"]

    for i, item in enumerate(results, 1):
        table = item.get("table", "unknown")
        source = item.get("source", "unknown")
        confidence = item.get("confidence", 0.5)
        content = item.get("content", "")

        # Build context line
        context_parts = []
        if item.get("session_target"):
            context_parts.append(f"target: {item['session_target']}")
        if item.get("session_sensor"):
            context_parts.append(f"sensor: {item['session_sensor']}")
        if item.get("session_date"):
            date = item["session_date"][:10]  # just the date part
            context_parts.append(f"date: {date}")
        context_str = ", ".join(context_parts) if context_parts else "no session context"

        # Type-specific label
        type_labels = {
            "observations": "Observation",
            "failures": "Failure",
            "preferences": "Preference",
        }
        type_label = type_labels.get(table, table)

        # Build entry
        header = f"{i}. [{type_label}, source: {source}, confidence: {confidence:.1f}] ({context_str})"
        lines.append(header)

        # Main content
        lines.append(f"   {content}")

        # Extra context for specific types
        if table == "failures":
            if item.get("tool"):
                lines.append(f"   Tool: {item['tool']}")
            if item.get("root_cause"):
                lines.append(f"   Root cause: {item['root_cause']}")
            if item.get("resolution"):
                lines.append(f"   Resolution: {item['resolution']}")

        if table == "preferences":
            if item.get("tool"):
                lines.append(f"   Tool: {item['tool']}")

        # Show parameters if available
        params = item.get("parameters")
        if params and isinstance(params, dict):
            params_str = json.dumps(params, indent=2, default=str)
            if len(params_str) < 300:
                lines.append(f"   Parameters: {params_str}")

        lines.append("")  # blank line between entries

    return "\n".join(lines)
