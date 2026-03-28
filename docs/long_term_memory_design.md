# Long-Term Memory System: Research & Design

This document covers the research behind Muphrid's long-term memory system — what
existing solutions were studied, what lessons were extracted, and how those lessons
shaped the implementation. It serves as both a reference for the design decisions
and a survey of the agent memory landscape as of early 2026.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Survey of Existing Solutions](#survey-of-existing-solutions)
   - [OpenClaw](#openclaw)
   - [langmem](#langmem)
   - [Oracle MemoryManager](#oracle-memorymanager)
   - [Mem0](#mem0)
   - [Zep / Graphiti](#zep--graphiti)
   - [Hindsight](#hindsight)
   - [Memvid](#memvid)
   - [Cognee](#cognee)
   - [Academic Patterns](#academic-patterns)
3. [Synthesis: 12 Lessons That Shaped the Design](#synthesis-12-lessons-that-shaped-the-design)
4. [What We Built](#what-we-built)
   - [Architecture Overview](#architecture-overview)
   - [Intended Workflow](#intended-workflow-expert-teaching--autonomous-graduation)
   - [Memory Types](#memory-types)
   - [Technology Stack](#technology-stack)
   - [Hybrid Search with RRF Fusion](#hybrid-search-with-rrf-fusion)
   - [Feature Flag](#feature-flag)
   - [v2 Roadmap](#v2-roadmap)
5. [File Reference](#file-reference)

---

## The Problem

Muphrid processes raw telescope data into final astrophotography images through a
multi-phase pipeline (calibration, registration, stacking, linear processing, stretching,
non-linear processing, export). Each run takes hours. The agent uses 29 tools across 5
phase gates, with human-in-the-loop (HITL) checkpoints for subjective decisions.

The problem: **every session starts from zero.** The agent has no memory of past targets,
what parameters worked, what failed, or what the user preferred. A path to a great image
that works on one target and camera would utterly fail on another. The more sessions the
agent processes, the more experience it should accumulate — but without long-term memory,
that experience is lost after every run.

The goal is for the agent to **learn** — not by training the model, but by accessing
structured memories from past sessions. Decisions made, reasoning paths taken, parameters
that worked, parameters that failed, and user preferences expressed through HITL
conversations.

---

## Survey of Existing Solutions

### OpenClaw

**What it is:** An open-source AI agent framework that went viral in early 2026 (250K+
GitHub stars). Its memory system is the most widely studied.

**Architecture:**
- Four-tier memory hierarchy:
  1. **Bootstrap layer** (`SOUL.md`, `AGENTS.md`, `MEMORY.md`) — loaded at every session start
  2. **Daily logs** (`memory/YYYY-MM-DD.md`) — append-only working context
  3. **Session transcripts** — conversation history, indexed and searchable
  4. **Retrieval index** (SQLite-backed) — semantic search across all Markdown files
- **Core philosophy:** "If it's not written to a file, it doesn't exist." The model only
  "remembers" what gets persisted to disk.

**Key implementation details:**
- **SQLite + sqlite-vec + FTS5** for hybrid search (no external vector database)
- **Chunking:** Line-aware splitting with ~400-token targets and 80-token overlap
- **Hybrid search:** BM25 (FTS5) + cosine similarity (sqlite-vec), union not intersection,
  weighted fusion (0.7 vector + 0.3 text)
- **Embedding cache:** `SHA256(text + model)` as key, LRU eviction at 50K entries
- **Auto-compaction flush:** Before context window fills, a silent agentic turn triggers
  the model to persist important information to Markdown files
- **Agent-facing tools:** `memory_search(query)` for semantic recall, `memory_get(filepath)`
  for targeted reads
- **Graceful degradation:** Vector-only if FTS5 unavailable, keyword-only if embeddings
  unavailable

**Lessons extracted:**
- File-first philosophy keeps things simple and debuggable
- SQLite + sqlite-vec is production-viable for single-user local workloads
- Embedding cache is critical for compute-heavy models
- Hybrid search catches both semantic and exact-match queries

**Sources:**
- [Memory - OpenClaw docs](https://docs.openclaw.ai/concepts/memory)
- [Deep Dive: How OpenClaw's Memory System Works](https://snowan.gitbook.io/study-notes/ai-blogs/openclaw-memory-system-deep-dive)
- [Local-First RAG: Using SQLite for AI Agent Memory with OpenClaw](https://www.pingcap.com/blog/local-first-rag-using-sqlite-ai-agent-memory-openclaw/)

---

### langmem

**What it is:** LangChain's official long-term memory SDK for LangGraph. Built by the
LangGraph team but has **zero GitHub releases** and questionable maintenance activity
as of March 2026.

**Architecture:**
- Two-layer design:
  1. **Core API** (stateless, storage-agnostic) — functional primitives for memory extraction
  2. **Stateful Integration** (LangGraph-aware) — tools and background processors tied to BaseStore
- **Three memory types:**
  - **Semantic memory:** Facts, knowledge, static information. Stored as searchable documents
    with Pydantic schemas (e.g., `Triple` with subject/predicate/object/context).
  - **Episodic memory:** Past interactions with 4-component structure — Observation, Thoughts,
    Action, Result. Captured as few-shot examples distilled from longer raw interactions.
  - **Procedural memory:** Behavioral rules encoded as evolving system prompts. Refined
    through the prompt optimizer based on user feedback.

**Key implementation details:**
- **LLM-driven parallel tool calling:** A single LLM call handles Create, Update, and Delete
  operations simultaneously via structured tool calls
- **Schema-driven extraction:** Pydantic models constrain LLM output — the model can only
  produce memories conforming to defined schemas, preventing hallucinated structure
- **Two extraction paths:**
  - *Hot path (conscious):* Agent explicitly saves via `create_manage_memory_tool` during
    conversation. Real-time, adds latency.
  - *Background path (subconscious):* `ReflectionExecutor` processes conversations
    asynchronously after activity settles. Debounces to avoid redundant extraction.
- **Namespace-based isolation:** Memories stored with hierarchical tuple namespaces like
  `("memories", "{user_id}", "preferences")`, dynamically substituted at runtime
- **Prompt optimization as procedural memory:** Three strategies (gradient, metaprompt,
  prompt memory) that refine the system prompt based on annotated conversation trajectories

**Lessons extracted:**
- The episodic structure (Observation → Thoughts → Action → Result) maps directly to
  astrophotography tool calls and their outcomes
- Schema-driven extraction is essential — unconstrained LLM extraction produces garbage
- The hot path vs background path distinction is fundamental for write timing
- Treating the system prompt itself as a memory type is clever but adds complexity

**Sources:**
- [LangMem Documentation](https://langchain-ai.github.io/langmem/)
- [LangMem Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [LangMem SDK Launch Blog Post](https://blog.langchain.com/langmem-sdk-launch/)

---

### Oracle MemoryManager

**What it is:** Oracle's approach to agent memory using Oracle Database 23ai with native
vector support. Demonstrated in a
[developer hub notebook](https://github.com/oracle-devrel/oracle-ai-developer-hub/blob/main/notebooks/memory_context_engineering_agents.ipynb).

**Architecture:**
- Custom `MemoryManager` class managing **6 memory types** across 7 tables:
  - Conversational (SQL) — chat history by thread
  - Knowledge Base (OracleVS) — searchable documents and facts
  - Workflow (OracleVS) — learned action patterns from past executions
  - Toolbox (OracleVS) — available tool definitions with semantic matching
  - Entity (OracleVS) — extracted people, places, systems
  - Summary (OracleVS) — compressed context for lengthy conversations
  - Tool Log (SQL) — offloaded tool outputs stored separately

**Key implementation details:**
- **Programmatic context assembly BEFORE agent reasoning:** Critical baseline context
  (conversational memory, knowledge base, summaries) is loaded automatically — the
  agent never needs to remember to fetch it
- **Just-in-time expansion:** Summaries stored as brief references; the agent calls
  `expand_summary()` only when full context is needed. Lean working context with
  fidelity preserved in storage.
- **Tool output offloading:** Full tool outputs persisted to Tool Log table; context
  retains one-line references like `[Tool Log ID: 42 - calculate_orbital_elements]`
- **Semantic tool discovery:** Tool docstrings stored as vectors; agent queries match
  against them via similarity search instead of keeping the full toolbox in context
- **Context budget monitoring:** Token usage tracked; summarization and offloading
  triggered when thresholds are breached

**Lessons extracted:**
- Programmatic context loading (baseline injected automatically) vs agentic retrieval
  (agent searches on demand) is a fundamental design choice with trade-offs
- Different memory types deserve different storage strategies (sequential SQL vs
  semantic vector search)
- Tool output offloading prevents context window bloat in long-running sessions

**Sources:**
- [Agent Memory: Why Your AI Has Amnesia and How to Fix It](https://blogs.oracle.com/developers/agent-memory-why-your-ai-has-amnesia-and-how-to-fix-it)
- [Oracle AI Developer Hub notebook](https://github.com/oracle-devrel/oracle-ai-developer-hub/blob/main/notebooks/memory_context_engineering_agents.ipynb)

---

### Mem0

**What it is:** A dedicated memory layer for AI agents. Available as both a managed
service and self-hosted. Has the clearest product focus on memory infrastructure.

**Architecture:**
- **Two-phase memory management:**
  1. **Extraction:** LLM analyzes new message pairs with conversation summary context,
     extracts salient memories. Contextual, not isolated — new facts extracted relative
     to existing summaries and recent history.
  2. **Update/Consolidation:** For each extracted fact, LLM determines operation via
     function calling: ADD (new info), UPDATE (augment existing), DELETE (contradicted),
     NOOP (no change).
- **Graph memory (Mem0^g):** Structured relational representation with entities as nodes,
  relationships as edges, and semantic labels. Two-stage extraction (entity extraction →
  relationship generation) with conflict detection and temporal resolution.

**Key implementation details:**
- **Memory scoring:** `RelevanceScore = α·recency + β·importance + γ·similarity`
  with customizable criteria (joy, urgency, empathy)
- **Temporal conflict resolution:** Conflicting memories marked invalid rather than
  deleted. Old relationships preserved with validity windows for historical reasoning.
- **LangGraph integration:** Operates as contextual injection middleware —
  `memory.search()` before each turn, `memory.add()` after. No graph restructuring needed.
- **Self-hosted backend flexibility:** Supports 22+ vector stores (Qdrant, Pinecone,
  ChromaDB, pgvector), 5+ graph databases (Neo4j, Memgraph, Kuzu)
- **Performance:** Claims 26% accuracy improvement over baseline, 91% lower p95 latency,
  90%+ token cost savings through consolidation

**Lessons extracted:**
- LLM-driven consolidation (ADD/UPDATE/DELETE/NOOP) is more flexible than hardcoded
  update logic
- Graph memory enables multi-hop reasoning about relationships
- Temporal conflict resolution (mark invalid, don't delete) preserves history for
  debugging and reasoning
- Memory scoring with multiple signals (recency + importance + similarity) produces
  better retrieval than similarity alone

**Sources:**
- [Mem0 Documentation](https://docs.mem0.ai/)
- [Mem0 LangGraph Integration](https://docs.mem0.ai/integrations/langgraph)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/html/2504.19413v1)

---

### Zep / Graphiti

**What it is:** The most theoretically rigorous memory system, implementing a
**bi-temporal knowledge graph** with formal temporal semantics.

**Architecture:**
- Three-tier subgraph hierarchy:
  1. **Episode subgraph:** Raw events with original timestamps
  2. **Semantic entity subgraph:** Extracted entities and facts with vector embeddings
  3. **Community subgraph:** Inductively clustered entity communities by connectivity
- **Bi-temporal model** — tracks two independent timelines:
  - `T (Event Time)`: When facts actually occurred
  - `T' (Ingestion Time)`: When information entered the graph

**Key implementation details:**
- **Every edge carries four timestamps:** `t_valid`, `t_invalid`, `t'_created`,
  `t'_expired`. This enables temporal queries like "What was true on date X?"
- **Temporal invalidation:** New contradicting fact arrives → LLM compares against
  related edges → existing edge's `t_invalid` set to now → old fact preserved with
  validity window, NOT deleted. Agents can reason about state transitions.
- **Performance:** 90% latency reduction vs baseline, 18.5% accuracy improvement
  on LongMemEval benchmark (particularly cross-session synthesis)

**Lessons extracted:**
- **Temporal validity is fundamental:** Don't store facts as immutable records. Store
  relationships with validity windows. This enables reasoning about how states change.
- The four-timestamp model is overkill for our use case, but the core insight —
  invalidate rather than delete — is universally applicable

**Sources:**
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)
- [Graphiti: Knowledge Graph Memory for an Agentic World](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)

---

### Hindsight

**What it is:** A memory framework achieving **91.4% accuracy on LongMemEval** through
epistemic clarity — developers can distinguish what agents *know* from what they *believe*.

**Architecture:**
- **Four-network separation:**

| Network | Content | Mutability | Example |
|---------|---------|-----------|---------|
| **World (W)** | Objective facts | Immutable | "Water freezes at 0°C" |
| **Experience (B)** | Agent's biography, 1st-person actions | Append-only | "I processed image_456 on 2025-03-15" |
| **Opinion (O)** | Subjective beliefs with confidence | Mutable | ("User prefers dark mode", 0.85, timestamp) |
| **Observation (S)** | Entity summaries from facts | Synthesis | "User John: prefers Markdown, night owl" |

**Key implementation details:**
- **Confidence scoring on opinions:** Structure is `(statement, confidence ∈ [0,1], timestamp, entities)`.
  Update mechanism: reinforcing evidence adds +α, contradicting evidence subtracts -2α.
  The asymmetry prevents wild oscillation on single pieces of evidence.
- **Four-way retrieval fused with RRF (Reciprocal Rank Fusion):**
  1. Semantic similarity (vector search)
  2. Keyword matching (BM25)
  3. Graph traversal via spreading activation
  4. Temporal filtering (recency bias)
  Results merged with RRF, then reranked with cross-encoder, respecting token budget.
- **Reflect operation:** Self-update cycle where the agent loads relevant memories and
  behavioral profiles, generates updated beliefs, and forms new opinions with confidence
  scores.
- **Performance:** Multi-session questions: 79.7% (vs 21.1% baseline), temporal reasoning:
  79.7% (vs 31.6% baseline)

**Lessons extracted:**
- **Epistemic separation is powerful.** Don't mix sources of different reliability on the
  same scale — separate them categorically. This directly informed our `source` column
  design (hitl = validated beliefs, phase_gate = raw experiences).
- **Confidence with asymmetric updates** prevents the agent from becoming overconfident
  about patterns that may be coincidental
- **RRF fusion** is the right way to combine multiple retrieval methods — it's rank-based,
  needs no score normalization, and is simple to implement

**Sources:**
- [Hindsight: Agent Memory That Learns](https://arxiv.org/abs/2512.12818)
- [Building AI Agents That Actually Learn using Hindsight Memory](https://medium.com/data-science-collective/building-ai-agents-that-actually-learns-using-hindsight-memory-microsoft-agent-framework-df75aa20b3bb)

---

### Memvid

**What it is:** A "no infrastructure" approach where everything — data, embeddings,
search index, metadata — is packaged into a single immutable `.mv2` file.

**Key details:**
- Append-only, immutable sequence (inspired by video encoding)
- P50 retrieval: 0.025ms, 1,372x throughput vs standard RAG
- Cannot update or delete specific memories (rebuild entire file)
- No concurrent writes, no ACID guarantees

**Lesson extracted:**
- If your domain allows append-only writes, immutable files beat databases on speed
  and portability. Trade mutability for latency. Potentially useful for archiving
  completed session memories, though not for active use.

---

### Cognee

**What it is:** A graph-semantic hybrid that combines vector similarity with graph
structure retrieval.

**Key details:**
- Dual retrieval: vector search for semantic closeness + graph traversal for explicit
  relationships
- Living memory that evolves while remaining queryable
- Single connected graph (no siloed stores)

**Lesson extracted:**
- Vector search + graph traversal is better than either alone. For domains with rich
  entity relationships (like astrophotography targets → sensors → conditions → parameters),
  explicit graphs add value beyond pure embedding similarity.

---

### Academic Patterns

**Reflection and self-improvement:**
- Store natural language critiques after failures. Future similar tasks retrieve and apply
  the critique. Risk: confirmation bias (entrench mistakes through repeated reference).
  Mitigation: require evidence grounding, periodic expiration, external validation.

**Contrastive experience extraction:**
- Systematically contrast success vs failure trajectories. Extract discriminative rules:
  "When sky is bright, use higher stretch factor. When faint, preserve baseline."
- This is the foundation for storing both observations AND failures — not just "what worked."

**"Experience-following" problem:**
- Agents naively repeat patterns retrieved from memory, including errors. This is the core
  risk of storing autonomous agent decisions. Solution: selective addition — only store
  high-signal memories (novel + validated). This is why v1 only stores HITL-validated
  memories.

**Retrieval > write quality:**
- Research finding: "Retrieval quality has substantially larger impact on performance than
  write strategy." Perfect writes + bad retrieval = mediocre. Good retrieval + imperfect
  writes = good. This prioritizes getting hybrid search right over perfecting extraction.

**Memory bloat prevention:**
- Consolidation (merge similar facts), TTL on episodic memories, deduplication,
  tiered storage. Critical warning: unlike API errors (explicit), memory eviction that
  removes the wrong records silently degrades quality — no exception, just worse responses.

**Sources:**
- [From Storage to Experience: A Survey on the Evolution of LLM Agent Memory](https://www.preprints.org/manuscript/202601.0618/v2/download)
- [How Memory Management Impacts LLM Agents](https://arxiv.org/abs/2505.16067)
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- [Get Experience from Practice: LLM Agents with Record & Replay](https://arxiv.org/abs/2505.17716)

---

## Synthesis: 12 Lessons That Shaped the Design

Every design decision in the implementation traces back to a specific research finding:

### 1. Programmatic saves, agentic reads
**Sources:** Mem0, Zep, Hindsight, langmem

Every mature system converges on this: the **harness decides WHEN** to save, an **LLM
decides WHAT** to extract, and the **agent decides WHEN to read** via tools. This avoids
redundancy and keeps the agent focused on its primary task.

**In our design:** Harness extracts memories after HITL approvals. Agent has `memory_search`
tool to read when it chooses.

### 2. Temporal validity, not deletion
**Sources:** Zep, Hindsight

Never delete memories — mark them invalid with timestamps. "Bias subtraction destroyed
nebula signal" is still valuable context even after a fix is found.

**In our design:** `valid_until` column on all tables. `invalidate()` method sets timestamp
instead of DELETE.

### 3. Confidence scoring on beliefs
**Sources:** Hindsight

Opinions get confidence scores [0,1]. Asymmetric updates: contradictions hit harder (-2α)
than reinforcements (+α).

**In our design:** `confidence` and `reinforcement_count` columns in schema. v1 stores
default values; v2 will add update logic.

### 4. Hybrid retrieval with RRF fusion
**Sources:** Zep, Hindsight, OpenClaw

Vector similarity + keyword search, fused with Reciprocal Rank Fusion. Vector catches
semantic queries; keywords catch exact matches. Union, not intersection.

**In our design:** sqlite-vec for vector search + FTS5 for keywords + RRF fusion with k=60.

### 5. Processing log as extraction source
**Sources:** Oracle, OpenClaw

Existing structured logs are the ideal extraction source — complete and already generated.

**In our design:** `processing_log.md` (generated by `advance_phase`) is the v2 extraction
source. HITL conversation history is the v1 source.

### 6. Episodic memory structure
**Sources:** langmem

Observation → Thoughts → Action → Result maps directly to astrophotography tool calls.

**In our design:** Observations and failures capture this structure with content,
parameters, and metrics.

### 7. Contrastive learning
**Sources:** Academic research

Store both successes AND failures. The agent should ask "what went wrong?" not just
"what worked?"

**In our design:** Separate `failures` table alongside `observations`, with root_cause
and resolution fields.

### 8. Epistemic separation
**Sources:** Hindsight

Don't mix sources of different reliability. HITL memories are validated beliefs; phase gate
memories are raw experiences.

**In our design:** `source` column (free-form text: "hitl", "phase_gate", extensible).
Search results prioritize hitl source.

### 9. "Experience-following" problem
**Sources:** Academic research

Agents naively repeat patterns from memory, including errors. Fix: selective addition —
only store high-signal memories.

**In our design:** v1 only stores HITL-validated memories. Phase gate extraction deferred
until autonomous quality improves.

### 10. Schema-driven extraction
**Sources:** langmem

Pydantic models constrain LLM output, preventing hallucinated memory structure.

**In our design:** `HITLMemoryExtraction` Pydantic model with `ExtractedObservation`,
`ExtractedFailure`, `ExtractedPreference` schemas.

### 11. Retrieval quality > write quality
**Sources:** Academic research

Optimization ROI is higher for search than saves.

**In our design:** Hybrid search (RRF + FTS5 + sqlite-vec) is the most carefully designed
component. Extraction logic is simpler by comparison.

### 12. Embedding cache
**Sources:** OpenClaw

SHA256(text + model) cache key with LRU eviction.

**In our design:** `embedding_cache` table in the same memory.db, 50K entry limit.

---

## What We Built

### Architecture Overview

```
                    ┌─────────────────────────┐
                    │    HITL Conversation     │
                    │  (agent + human iterate) │
                    └───────────┬─────────────┘
                                │ human approves
                                ▼
                    ┌─────────────────────────┐
                    │   LLM Extraction        │
                    │  (same model as agent)   │
                    │  Pydantic-constrained    │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                   ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │ Observations │  │   Failures   │  │ Preferences  │
      │ (what worked)│  │(what failed) │  │(user choices)│
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                   │
             └─────────────────┼───────────────────┘
                               ▼
                    ┌─────────────────────────┐
                    │   ~/.muphrid/       │
                    │     memory.db           │
                    │  SQLite + sqlite-vec    │
                    │  + FTS5                 │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   memory_search tool    │
                    │  (agent calls on demand)│
                    │  Hybrid RRF retrieval   │
                    └─────────────────────────┘
```

### Intended Workflow: Expert Teaching → Autonomous Graduation

The memory system enables a "learning from demonstration" pattern:

1. **Expert teaches** — A skilled astrophotographer runs 5-10 HITL sessions across diverse
   targets (emission nebulae, galaxies, clusters, different sensors/conditions). Memory is
   ON. Each HITL conversation produces validated observations, failure records, and aesthetic
   preferences.

2. **Agent graduates** — Switch to autonomous mode. The agent now has a rich memory store
   to search. Before stretching a nebula, it finds "expert preferred GHS D=10 with highlight
   protection for compressed Bayer data." Before denoising, it finds "strength 0.4 worked
   well at SNR 74."

3. **Agent generalizes** — On new targets, the agent searches for relevant past experience
   and reasons about how to adapt. Memories are compositional, not fixed recipes — the agent
   combines fragments from different sessions based on current conditions.

The HITL conversations serve as reinforcement learning without model fine-tuning. The memory
store is the knowledge transfer mechanism — expert decision-making distilled into searchable
context.

### Memory Types

| Type | Table | What it stores | Written when |
|------|-------|---------------|-------------|
| **Session** | `sessions` | Full session metadata + outcome | End of run |
| **Observation** | `observations` | What worked — parameters, metrics, conditions | After HITL approval |
| **Failure** | `failures` | What went wrong — root cause, resolution | After HITL where agent iterated |
| **Preference** | `preferences` | User aesthetic choices | After HITL approval with feedback |

All tables include:
- `source` column (free-form text: "hitl", extensible to "phase_gate", "analyze", etc.)
- `confidence` and `reinforcement_count` (schema-ready for v2 logic)
- `valid_until` (NULL = active; timestamp = invalidated)
- `created_at` / `updated_at` timestamps

### Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Storage | SQLite | Single file, no server, already in the stack |
| Vector search | sqlite-vec | Local, no infrastructure, graceful degradation |
| Keyword search | FTS5 | Built into SQLite, BM25 ranking, Porter stemming |
| Embeddings | Qwen3-Embedding-8B via Ollama | Best open-source MTEB scores, runs locally on Mac |
| Retrieval fusion | RRF (k=60) | Rank-based, no score normalization, simple |
| Extraction | Agent's LLM + Pydantic schemas | Same model, astrophotography context, structured output |
| Cache | SHA256 → embedding in SQLite | Avoids re-computing identical text |

### Hybrid Search with RRF Fusion

```
Query: "stretching compressed Bayer data on emission nebula"
        │
        ├──► sqlite-vec cosine similarity ──► ranked by distance
        │    (catches: "GHS stretch produced good results
        │     for compressed data on nebula")
        │
        └──► FTS5 BM25 keyword search ──► ranked by BM25 score
             (catches: exact matches on "Bayer", "emission",
              "stretch", target names like "M42")
        │
        ▼
  RRF Fusion (k=60)
  ─────────────────
  score(item) = Σ  1/(rank_in_source + 60)

  Items appearing in BOTH sources get boosted.
  Items appearing in only ONE source still contribute.
  (union, not intersection)
        │
        ▼
  Prioritize hitl source over phase_gate
        │
        ▼
  Return top-k with full context
```

### Feature Flag

Single flag: `MEMORY_ENABLED` (default: `false`)

| Mode | Behavior |
|------|----------|
| OFF | Zero overhead. No tools registered, no Ollama connection, no extraction. `processing_log.md` still written. |
| ON + HITL active | `memory_search` available in all phases. Memories extracted after each HITL approval. |
| ON + autonomous | `memory_search` available. No new memories written (no HITL to trigger extraction in v1). |

### v2 Roadmap

The following are designed into the schema but not implemented in v1:

| Feature | Schema support | Implementation needed |
|---------|---------------|----------------------|
| **Phase gate extraction** | `source="phase_gate"` | Add extraction call in `advance_phase` after `_write_phase_report()` |
| **Confidence updates** | `confidence`, `reinforcement_count`, `valid_until` columns | Semantic similarity check between new and existing memories during extraction |
| **New memory sources** | `source` is free-form text | Add extraction hooks for `analyze_image` reasoning, specific tool calls, etc. |
| **Session records** | `sessions` table | Extract session summary at end of run (EXPORT → COMPLETE) |
| **Memory consolidation** | — | Merge redundant memories, TTL on old entries |

---

## File Reference

### New files

| File | Purpose |
|------|---------|
| `muphrid/memory/__init__.py` | Package init with architecture overview |
| `muphrid/memory/store.py` | `MemoryStore` — SQLite + sqlite-vec + FTS5, RRF hybrid search, temporal validity |
| `muphrid/memory/embeddings.py` | `OllamaEmbedder` — Qwen3-Embedding via Ollama with SHA256 cache |
| `muphrid/memory/extraction.py` | `extract_hitl_memory()` — LLM extraction with Pydantic schemas |
| `muphrid/tools/utility/t33_memory_search.py` | Agent-facing `memory_search` tool |

### Modified files

| File | Change |
|------|--------|
| `muphrid/config.py` | Added `memory_enabled`, `memory_db_path`, `memory_embedding_model` to Settings |
| `muphrid/graph/hitl.py` | Added `set_memory_enabled()` / `is_memory_enabled()` runtime flag |
| `muphrid/graph/registry.py` | Added `register_memory_tool()` for conditional tool registration |
| `muphrid/graph/nodes.py` | Added `_extract_hitl_memories()` — fires after every HITL approval |
| `muphrid/graph/prompts.py` | Added Long-Term Memory guidance section to system prompt |
| `muphrid/gradio_app.py` | Added memory checkbox in HITL Config tab |
| `muphrid/cli.py` | Added `--memory` flag |
| `pyproject.toml` | Added `sqlite-vec` and `ollama` dependencies |
