"""
Long-term memory system — SQLite + sqlite-vec + FTS5.

The memory store persists across sessions at ~/.muphrid/memory.db.
It captures validated learnings from HITL conversations: observations
(what worked), failures (what went wrong), and preferences (user
aesthetic choices). The agent accesses memory via the memory_search
tool during its ReAct loop.

Architecture (informed by OpenClaw, langmem, Mem0, Zep/Graphiti, Hindsight):
  - Programmatic saves after HITL approvals (harness owns write timing)
  - Agentic reads via memory_search tool (agent owns read strategy)
  - Hybrid retrieval: sqlite-vec cosine similarity + FTS5 BM25, fused with RRF
  - Temporal validity: memories are invalidated, never deleted
  - Confidence scoring: schema-ready for v2 reinforcement logic
  - Source tagging: "hitl" (v1), "phase_gate" (v2 future), extensible

Embedding backends: Ollama (external service, large models) or FastEmbed
(in-process ONNX, no external service). Configured in processing.toml.
Init is fail-loud — if the embedder can't start, the session is blocked.

Feature flag: MEMORY_ENABLED (default: false). When off, no memory
tools are registered, no extraction fires, and no embedding connection
is attempted. Zero overhead.
"""
