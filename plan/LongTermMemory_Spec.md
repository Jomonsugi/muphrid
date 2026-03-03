# AstroAgent — Long-Term Memory Spec (v2 Enhancement)

> **Status:** Post-v1 enhancement. The v1 system (`AstroAgent_Spec.md`) must be fully functional
> end-to-end before this spec is implemented. Nothing in this document touches the tool layer.
> All changes are additive to the reasoning and graph layers only.

---

## 1  Motivation

The v1 system is scoped to a single session. `SqliteSaver` checkpoints preserve state within a
run, but each new dataset starts cold. The agent has no memory of:

- This photographer's aesthetic preferences (stretch style, saturation level, star prominence)
- How similar targets were processed in the past
- Which parameters caused problems (ringing from over-deconvolution, clipping from over-stretch)
- Which processing paths the user consistently overrides at HITL checkpoints

Long-term memory closes this gap. After v2, every new run benefits from every previous run.
The agent personalizes its starting parameters, retrieves relevant past experience, and
continuously refines its own procedural rules based on observed user behavior.

---

## 2  Memory Architecture

### 2.1 Three Memory Types

| Type | What it stores | Used for |
|---|---|---|
| **Semantic** | User preferences, target-type defaults, sensor characteristics | Personalize starting parameters at session start |
| **Episodic** | Records of completed processing runs and their outcomes | Retrieve analogous past runs when processing a similar target |
| **Procedural** | Rules the agent follows; updated from observed feedback patterns | Modify the system prompt and heuristics over time |

### 2.2 Storage Backend

**Library:** [LangMem](https://github.com/langchain-ai/langmem) — the LangChain-maintained
library for managing persistent agent memory.

**Vector store:** ChromaDB (local, no server required) for semantic search over episodic
memories. SQLite for structured semantic and procedural records.

**Location:** Configurable via `ASTRO_AGENT_MEMORY_DB` in `.env`. Defaults to
`~/.astro_agent/memory/`.

```
~/.astro_agent/
  checkpoints.db          ← v1 SqliteSaver (unchanged)
  memory/
    chroma/               ← ChromaDB vector store for episodic search
    semantic.db           ← SQLite for user preferences and sensor profiles
    procedural.db         ← SQLite for evolving agent rules
```

### 2.3 Memory Schemas

#### SemanticMemory — user preferences

```python
class UserPreferences(TypedDict):
    stretch_style: str          # "gentle" | "moderate" | "aggressive"
    saturation_level: str       # "conservative" | "moderate" | "rich"
    star_prominence: str        # "subtle" | "natural" | "prominent"
    noise_tolerance: str        # "clean" | "balanced" | "textured"
    deconv_aggressiveness: str  # "minimal" | "moderate" | "sharp"
    preferred_stretch_method: str  # "ght" | "asinh" | "autostretch"
    notes: str                  # free-text observations from HITL sessions
    updated_at: str             # ISO timestamp
    confidence: float           # 0.0–1.0, increases with more data points
```

#### SemanticMemory — sensor profile

```python
class SensorProfile(TypedDict):
    camera_model: str           # from FITS INSTRUME header
    scnr_amount: float          # learned preferred SCNR strength
    dark_optimization: bool     # whether dark optimization helps this sensor
    typical_bgnoise: float      # baseline background noise for this camera
    notes: str
    updated_at: str
```

#### EpisodicMemory — processing run record

```python
class ProcessingRunRecord(TypedDict):
    run_id: str                 # thread_id from v1
    timestamp: str
    target_name: str            # from plate solve or user-provided
    object_type: str            # "galaxy" | "nebula" | "cluster" | "planetary"
    focal_length_mm: float      # from FITS header
    pixel_scale_arcsec: float   # from plate solve
    bortle_estimate: int        # inferred from background sky value
    filter_type: str            # "broadband" | "narrowband" | "rgb" | "lrgb"
    frame_count: int
    final_snr: float
    processing_notes: str       # narrative summary of decisions made
    successful_parameters: dict # tool → parameter values that produced the accepted result
    hitl_decisions: list[dict]  # each HITL checkpoint: question, options, user choice
    user_satisfaction: str      # "accepted" | "revised_once" | "revised_multiple"
    embedding_text: str         # text used to generate the vector embedding for search
```

#### ProceduralMemory — agent rules

```python
class ProceduralRule(TypedDict):
    rule_id: str
    category: str               # "stretch" | "deconvolution" | "noise" | "saturation" | "general"
    rule_text: str              # natural language rule injected into the system prompt
    evidence_count: int         # number of sessions that contributed to this rule
    last_updated: str
    confidence: float
```

Default procedural rules are seeded from the spec's static heuristics (§7.3 of
`AstroAgent_Spec.md`) and then evolved from there.

---

## 3  Integration Points

There are exactly three places where memory interacts with the v1 graph. Nothing else changes.

### 3.1 Session Start — Memory Retrieval Node

**When:** Immediately after `make_initial_state()`, before the first planner invocation.

**Node name:** `memory_retrieve_node`

**What it does:**

1. Query `semantic.db` for `UserPreferences` and any `SensorProfile` matching `state.metadata.camera_model`.
2. Build a search query from the current dataset's metadata: object type, focal length, pixel scale, filter type, Bortle estimate.
3. Semantic search ChromaDB for the top-3 most similar past `ProcessingRunRecord` entries.
4. Compose a `MemoryContext` block (see §4) and inject it into `AstroState.memory_context`.
5. The planner system prompt template includes `{memory_context}` — this is how retrieved memories reach the LLM.

**If no memories exist:** `memory_context` is empty. The agent behaves identically to v1.

### 3.2 HITL Checkpoint — Preference Write

**When:** After `hitl_interrupt_node` captures user input and before the graph resumes.

**What it does:**

1. Extract the user's selection and any free-text revision notes.
2. Interpret the selection in context (e.g. "user chose 'more aggressive stretch' over 'moderate'").
3. Update `UserPreferences` in `semantic.db`. If `confidence < 0.5`, update directly. If
   `confidence ≥ 0.5`, use a weighted blend so single outlier sessions don't override a
   strong established preference.
4. Append the raw decision to the in-progress `ProcessingRunRecord` for this session.

This is non-blocking — written synchronously before graph resume.

### 3.3 Session End — Run Record Write

**When:** After `export_final` completes successfully, before CLI exits.

**Node name:** `memory_write_node`

**What it does:**

1. Compile the `ProcessingRunRecord` from the completed `AstroState`:
   - Target info from plate solve result and FITS headers
   - `successful_parameters` from `state.history` (last accepted run of each tool)
   - `hitl_decisions` from `state.user_feedback`
   - `user_satisfaction` derived from number of revision cycles at each HITL point
2. Generate `embedding_text`: a natural language summary of the dataset and processing choices
   (e.g. *"Face-on spiral galaxy, broadband RGB, 480mm focal length, Bortle 5, GHS stretch
   at intensity 1.2, moderate deconvolution 8 iterations RL, user preferred moderate saturation"*).
3. Embed `embedding_text` and store the full record in ChromaDB.
4. Run background procedural rule refinement (§3.4).

### 3.4 Background Rule Refinement

After `memory_write_node`, a background helper agent (`rule_refiner`) runs asynchronously:

1. Retrieve all `ProcessingRunRecord` entries where `user_satisfaction = "revised_once"` or
   `"revised_multiple"` at a specific HITL checkpoint type.
2. Identify patterns (e.g. "in 8 of the last 10 broadband galaxy sessions, user increased
   saturation at the final HITL checkpoint").
3. Propose a rule update (e.g. *"For broadband galaxy targets, start saturation at 0.5, not 0.3"*).
4. Write the updated `ProceduralRule` to `procedural.db`.

This runs after the session exits and has no effect on the current run.

---

## 4  Memory Context Block

The `MemoryContext` injected into `AstroState` is a structured dict:

```python
class MemoryContext(TypedDict):
    user_preferences: UserPreferences | None
    sensor_profile: SensorProfile | None
    similar_runs: list[ProcessingRunRecord]   # top-3 from episodic search
    procedural_rules: list[ProceduralRule]    # all active rules, ordered by category
```

The planner's system prompt template (§7.4 of `AstroAgent_Spec.md`) gains a new section:

```
## Memory Context

### Your knowledge of this photographer
{memory_context.user_preferences | "No preferences recorded yet."}

### Sensor characteristics
{memory_context.sensor_profile | "No sensor profile recorded yet."}

### Most similar past sessions
{memory_context.similar_runs | "No similar past sessions found."}

### Learned rules (in addition to base heuristics)
{memory_context.procedural_rules | "No learned rules yet — using base heuristics only."}
```

When `memory_context` is empty (first-ever session), the section is omitted entirely. The
agent's behavior is indistinguishable from v1.

---

## 5  AstroState Changes

Two fields are added to `AstroState` (§4.1 of `AstroAgent_Spec.md`):

```python
class AstroState(TypedDict):
    # ... all existing v1 fields unchanged ...

    # v2 additions
    memory_context: MemoryContext | None        # populated by memory_retrieve_node
    current_run_record: ProcessingRunRecord     # built incrementally, written at end
```

No existing field is modified. No tool input or output schema changes.

---

## 6  Graph Changes

The v1 `StateGraph` (§6.1 of `AstroAgent_Spec.md`) gains two nodes:

```
START
  ↓
memory_retrieve_node    ← NEW: query memory, populate state.memory_context
  ↓
ingest_node
  ↓
  ... (all v1 nodes unchanged) ...
  ↓
export_node
  ↓
memory_write_node       ← NEW: compile and store run record, trigger rule refinement
  ↓
END
```

`hitl_interrupt_node` gains a side-effect call to the preference writer (§3.2) after input
is captured. No routing logic changes.

---

## 7  New Dependencies

| Package | Purpose | Version |
|---|---|---|
| `langmem` | Memory management for LangGraph agents | latest stable |
| `chromadb` | Local vector store for episodic search | ≥ 0.5 |
| `sentence-transformers` | Embedding model for `embedding_text` | ≥ 3.0 |

Add to `pyproject.toml` and `requirements.txt`.

New `.env` keys (add to `.env.example`):

```dotenv
# Long-term memory (v2)
ASTRO_AGENT_MEMORY_DB=~/.astro_agent/memory
ASTRO_AGENT_EMBEDDING_MODEL=all-MiniLM-L6-v2   # local model, no API key required
ASTRO_AGENT_MEMORY_ENABLED=true                 # set false to disable entirely (v1 behavior)
```

When `ASTRO_AGENT_MEMORY_ENABLED=false`, `memory_retrieve_node` returns immediately with
`memory_context = None` and `memory_write_node` is a no-op. Zero behavioral change.

---

## 8  Privacy and Data Ownership

All memory is stored locally. No data leaves the machine. No cloud vector store is used.
The embedding model (`all-MiniLM-L6-v2`) runs locally via `sentence-transformers`.

A CLI command is provided for memory inspection and deletion:

```
astro_agent memory list                  # list all stored run records
astro_agent memory show <run_id>         # show a specific run record
astro_agent memory prefs                 # show current UserPreferences
astro_agent memory rules                 # show active procedural rules
astro_agent memory clear                 # delete all memory (with confirmation prompt)
```

---

## 9  What Does Not Change

| Component | Status |
|---|---|
| All tools T01–T24 | Unchanged |
| `AstroState` existing fields | Unchanged |
| Tool input/output schemas | Unchanged |
| LangGraph routing logic | Unchanged |
| HITL protocol (§8 of v1 spec) | Unchanged except side-effect write |
| `SqliteSaver` checkpointing | Unchanged |
| CLI `process` command | Unchanged |
| `.env` existing keys | Unchanged |
