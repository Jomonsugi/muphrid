"""
Long-term memory setup — BaseStore for cross-thread parameter priors.

The store persists across thread_ids (pipeline runs). The agent reads from
it before subjective steps (T14, T17, T18, etc.) and writes to it after
HITL resolution (approved parameters + context).

Namespace convention:
    ("preferences", target_type, tool_id)  →  key: "{sensor}_{integration_tier}"

Example:
    store.put(
        namespace=("preferences", "emission_nebula", "T14_stretch"),
        key="ASI2600MC_4h",
        value={"stretch_amount": 2.8, "highlight_protection": 0.94, ...}
    )

For now this module provides a simple InMemoryStore. Production use should
swap to a SQLite-backed or Postgres-backed store for durability.

LangMem integration (automatic memory extraction from conversation history)
is a future enhancement — see hitl_design.md §Long-Term Memory.
"""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore


def make_memory_store() -> InMemoryStore:
    """
    Create the long-term memory store.

    Returns an InMemoryStore for now. Swap to a persistent backend
    (e.g. SQLite-backed store) when ready for production.
    """
    return InMemoryStore()
