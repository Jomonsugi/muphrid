# AstroAgent — Long-Term Memory Build Plan (v2)

> **Prerequisite:** Every checkbox in `plan.md` (v1) must be checked before starting here.
> Spec: `LongTermMemory_Spec.md` — source of truth for all schemas, integration points, and contracts.
> This plan is additive. Nothing in v1 is removed or replaced.

---

## Phase 0 — Dependencies and Configuration

*Nothing else can be built until this phase is done.*

- [ ] **Add v2 dependencies to `pyproject.toml` and `requirements.txt`**
  `langmem`, `chromadb ≥ 0.5`, `sentence-transformers ≥ 3.0`. Run install and confirm no
  version conflicts with existing v1 packages. Spec: §7.

- [ ] **Add v2 `.env` keys to `.env.example`**
  `ASTRO_AGENT_MEMORY_DB`, `ASTRO_AGENT_EMBEDDING_MODEL`, `ASTRO_AGENT_MEMORY_ENABLED`.
  Update `config.py` to load these values with appropriate defaults. Spec: §7.

- [ ] **Create memory directory on first run**
  On startup, if `ASTRO_AGENT_MEMORY_ENABLED=true`, create `~/.astro_agent/memory/chroma/`
  if it does not exist. Log the path at INFO level.

---

## Phase 1 — Memory Schemas

*Import by all memory modules. Build before any store or node code.*

- [ ] **`memory/schemas.py` — all memory TypedDicts**
  `UserPreferences`, `SensorProfile`, `EpisodicRunRecord`, `ProceduralRule`, `MemoryContext`.
  Match the exact fields and types from `LongTermMemory_Spec.md` §2.3.

- [ ] **Add `memory_context` and `current_run_record` fields to `AstroState`**
  Two new optional fields in `graph/state.py`. All existing fields unchanged. Confirm that
  existing nodes that do not reference these fields still compile and run correctly.
  Spec: §5.

---

## Phase 2 — Storage Layer

- [ ] **`memory/semantic_store.py` — SQLite-backed preference and sensor store**
  Functions: `get_user_preferences() -> UserPreferences | None`,
  `update_user_preferences(new_data: dict, weight: float)`,
  `get_sensor_profile(camera_model: str) -> SensorProfile | None`,
  `upsert_sensor_profile(profile: SensorProfile)`.
  Weighted blend logic: if `confidence ≥ 0.5`, blend rather than overwrite. Spec: §3.2.

- [ ] **`memory/episodic_store.py` — ChromaDB vector store for run records**
  Functions: `add_run_record(record: EpisodicRunRecord)`,
  `search_similar_runs(query_text: str, n_results: int = 3) -> list[EpisodicRunRecord]`.
  Use `ASTRO_AGENT_EMBEDDING_MODEL` to embed both `query_text` and `record.embedding_text`.
  Spec: §2.2, §3.3.

- [ ] **`memory/procedural_store.py` — SQLite-backed rule store**
  Functions: `get_all_rules() -> list[ProceduralRule]`,
  `upsert_rule(rule: ProceduralRule)`, `seed_default_rules()`.
  `seed_default_rules()` translates the static heuristics from `AstroAgent_Spec.md` §7.3
  into `ProceduralRule` records on first run. Spec: §2.3.

- [ ] **`memory/store.py` — unified facade**
  Single import point: `MemoryStore` class that composes the three stores above.
  Instantiated once at startup and passed to nodes that need it (not a global).

---

## Phase 3 — Retrieval Node

- [ ] **`graph/nodes/memory_retrieve_node.py`**
  1. Query `MemoryStore` for `UserPreferences` and matching `SensorProfile`.
  2. Build embedding query from `state.metadata` (object type, focal length, pixel scale, filter, Bortle estimate).
  3. Call `episodic_store.search_similar_runs(query, n=3)`.
  4. Load all active `ProceduralRule` records.
  5. Assemble `MemoryContext` and write to `state.memory_context`.
  If `ASTRO_AGENT_MEMORY_ENABLED=false`, return immediately with `memory_context = None`.
  Spec: §3.1.

- [ ] **Wire `memory_retrieve_node` into the top-level graph**
  Insert between `START` and `ingest_node`. Edge: `START → memory_retrieve_node → ingest_node`.
  Spec: §6.

- [ ] **Update system prompt template to include `{memory_context}` block**
  Extend the template in `graph/nodes/planner_node.py` with the memory context section
  from `LongTermMemory_Spec.md` §4. When `memory_context is None`, omit the section entirely
  so v1 behavior is preserved. Spec: §4.

---

## Phase 4 — HITL Preference Write

- [ ] **Add preference write side effect to `hitl_interrupt_node`**
  After user input is captured and before graph resumes:
  1. Parse the HITL decision in context (which option, which question category).
  2. Call `semantic_store.update_user_preferences()` with the interpreted preference signal.
  3. Append the raw decision dict to `state.current_run_record["hitl_decisions"]`.
  No new node required — this is a side effect inside the existing node. Spec: §3.2.

---

## Phase 5 — Session-End Write Node

- [ ] **`graph/nodes/memory_write_node.py`**
  1. Compile `EpisodicRunRecord` from completed `AstroState`:
     - Target info from `state.metadata` and plate solve result
     - `successful_parameters` from `state.history` (last accepted call per tool)
     - `hitl_decisions` from `state.user_feedback` and `state.current_run_record`
     - `user_satisfaction` derived from revision cycle count
  2. Generate `embedding_text` narrative (see spec §3.3 for example format).
  3. Call `episodic_store.add_run_record(record)`.
  4. Log run ID to terminal so user can reference it.
  Spec: §3.3.

- [ ] **Wire `memory_write_node` into the top-level graph**
  Insert between `export_node` and `END`. Edge: `export_node → memory_write_node → END`.
  Spec: §6.

---

## Phase 6 — Background Rule Refinement

- [ ] **`memory/rule_refiner.py` — background helper**
  Triggered asynchronously from `memory_write_node` after record is stored.
  1. Retrieve all episodic records where `user_satisfaction` is not `"accepted"` for
     each HITL category.
  2. Identify patterns: if ≥ 70% of recent sessions for a given target type and HITL
     category show the same override direction, generate a rule update.
  3. Upsert the `ProceduralRule` into `procedural_store`.
  Run as `asyncio.create_task()` or a subprocess — must not block CLI exit. Spec: §3.4.

---

## Phase 7 — Memory CLI Commands

- [ ] **Extend `cli.py` with `memory` subcommands**
  `astro_agent memory list` — tabular list of all stored run records (run_id, timestamp, target, satisfaction).
  `astro_agent memory show <run_id>` — full record detail.
  `astro_agent memory prefs` — current `UserPreferences` with confidence score.
  `astro_agent memory rules` — all active `ProceduralRule` records.
  `astro_agent memory clear` — delete all memory stores with `y/N` confirmation prompt.
  Spec: §8.

---

## Phase 8 — Disable Flag Verification

- [ ] **Verify `ASTRO_AGENT_MEMORY_ENABLED=false` produces identical v1 behavior**
  Run the full v1 end-to-end test (from `plan.md` Phase 11) with `MEMORY_ENABLED=false`.
  Confirm: no memory files created, no performance change, system prompt unchanged,
  `memory_context` is `None` in state throughout. This is the regression gate.

---

## Phase 9 — End-to-End Validation

- [ ] **Two-session validation run**
  Session 1: Process a dataset end-to-end. Make deliberate HITL choices. Confirm that
  after exit, `astro_agent memory prefs` reflects the choices made and `memory list`
  shows the run record.

- [ ] **Session 2 on similar target: confirm memory is used**
  Process a second dataset of the same object type. Confirm:
  - System prompt contains the memory context block with Session 1's preferences.
  - Session 1 appears in `similar_runs`.
  - Agent's initial parameter suggestions are visibly influenced by the stored preferences
    (check planner's first tool call arguments against Session 1 outcomes).

- [ ] **Procedural rule refinement validation**
  After 3+ sessions where the user consistently overrides the same parameter in the same
  direction, confirm `astro_agent memory rules` shows an updated rule reflecting the pattern.
