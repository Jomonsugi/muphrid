# Notes For Agents Working On Muphrid

Muphrid is a LangGraph astrophotography system. Its core value is not only
that an LLM can call Siril, GraXpert, StarNet, Astropy, Photutils, and
scikit-image. Its value is that graph state gives the agent, tools, UI,
checkpoint/resume, audit reports, and prompt context a shared model of the
work.

Treat that state as the product surface.

When you solve a bug or add a feature, first decide which layer owns the
truth:

- **Tool capability** — the agent needs a new operation it can perform.
- **Diagnostic metric** — the agent needs better observations.
- **Authoritative state contract** — the system must remember a fact that
  future tools/UI/routing will depend on.
- **Controller policy** — a durable workflow concern spans nodes, UI, prompt,
  and routing.
- **Graph structure** — a control-flow step is ephemeral and does not need a
  persisted contract.
- **UI renderer** — Gradio should show existing state differently, not invent
  new truth.
- **Prompt surface** — the model needs to see an existing contract or tool
  affordance more clearly.

Most bad changes in this codebase come from putting truth in the wrong
layer: message scans instead of state, prompt instructions instead of tools,
UI deductions instead of graph contracts, or fallbacks where authoritative
state should refuse.

## State As Data, Not Deduction

If a fact matters after the current node returns, it belongs in state.

Good state contracts are:

1. **Durable** — they survive checkpoint/resume.
2. **Readable** — the UI, prompt, and graph can answer "what is happening?"
   without replaying messages.
3. **Owned** — one controller/helper layer writes transitions; readers do
   not reconstruct meaning independently.
4. **Tested** — scripts assert transitions and refusal paths directly.

Bad substitutes:

- scanning recent `ToolMessage` / `AIMessage` history to infer status,
- sentinel strings in chat,
- UI-only state that the graph cannot see,
- prompt-only instructions that the UI cannot see,
- boolean mirrors that drift from the real contract,
- fallback heuristics at read sites for state that should be authoritative.

The smell: two places answer the same question differently. If the graph,
the model prompt, the UI, a report, and checkpoint state can disagree, the
concern needs a single contract.

Messages are transcript and evidence. They are not durable workflow state.
Use them for debugging, audit narrative, and tool outputs. Do not make them
the source of truth for whether something is open, approved, current,
exported, selected, or safe to advance.

Graph position is ephemeral control flow. It is useful while execution is
inside a run, but it is not a persisted object the UI or prompt can inspect
after resume. If the UI or prompt must know it, record it in state.

## Generalize The Harness, Not The Run

Muphrid is a harness; the model is the brain. You do not know what comes next
— a nebula, a galaxy, a wide-field Milky Way pass, a target nobody has shot
before — any more than a coding agent knows its next bug. So the harness must
generalize the way a coding agent's own harness does: it ships read/edit/
search/run + version control + judgment, never "the fix," and those compose
onto any codebase. Muphrid ships tools + observations + recoverable state, and
the **agent** supplies the per-run judgment. Neither harness ships the answer;
both ship the means to find one.

The design consequence: when you add a feature to solve a concrete case you
just debugged, that case wants to ride along into the prompt as if it were the
rule. It must not — especially in always-on context. State the **capability**
and the **invariant**; let the agent map them to the situation in front of it.

Agent-facing surfaces — `SYSTEM_BASE`, phase prompts, tool docstrings, schema
field descriptions, dynamic prompt sections (`_format_*_for_prompt`) —
describe mechanics and capabilities, never the scenario that motivated them.

Allowed (these generalize):

- input-format examples, ideally spanning cases ("M42", "Andromeda Galaxy",
  "Milky Way core") — they show the field takes anything;
- mechanical descriptions of what a metric measures or a parameter does, using
  physical structure only to illustrate scale/signal ("scale 4 ≈ galaxy arms",
  "extended signal (nebulae, galaxies) inflates MAD");
- maintainer comments and path examples (not sent to the model).

Not allowed (these walk the agent down one path):

- target/scenario recipes — "for nebulae, stretch aggressively"; "a color cast
  means a gradient color-cal can't fix";
- prescriptive usage — "use X for Y", "always call A before B" (this is the
  docstring rule: docstrings say what a parameter does mechanically, never
  when to reach for the tool or what target to use it on);
- a worked example from a debugging session baked into an always-on prompt as
  if it were the general rule.

The test for any sentence in an agent-facing surface: **would it still read
correctly for a target or problem you have never seen?** If it only makes
sense for the run you just did, it belongs in a commit message or a test, not
the harness. This is the prompt-and-docstring complement to "State As Data":
truth lives in state and in general capability, not in narration that pins the
agent to one story.

## When To Add A Controller Module

Use this four-part test before inventing structure:

1. **Durable state** — must it survive checkpoint/resume?
2. **UI semantics** — does Gradio render or act on it?
3. **Prompt semantics** — does the agent need to see it in its system context?
4. **Graph routing** — does control flow branch on it?

If all four apply, use the controller pattern:

- add a typed state object in `muphrid/graph/state.py`,
- add pure-ish helpers in `muphrid/graph/<concern>.py`,
- make `nodes.py` orchestrate helpers rather than embed policy,
- document the invariant here,
- add a focused script under `scripts/test_<concern>_*.py`.

If fewer apply, keep the simpler shape:

- Routing only: graph edges or `Command(goto=...)`; no persisted state field.
- Durable + routing: typed field and reducer; usually no controller.
- Durable + UI: state field plus render mapping; usually no controller.
- Durable + prompt: state field plus prompt formatting; usually no controller.

Do not use a controller because it feels architecturally tidy. Use it when
graph position, persisted state, UI state, and prompt state would otherwise
duplicate or infer the same truth.

## Controller Recipe

`muphrid.graph.review` is the current worked example of the pattern.

A controller module should contain:

- constructors: `make_*_session(...)`,
- predicates: `*_is_open(...)`, `requires_*`, `*_limit_reached(...)`,
- transition helpers: `update_*_session(...)`, `close_*_session(...)`,
- payload/projection helpers for UI and prompt use,
- typed event parsing if the UI or CLI sends structured input.

The helpers should take state-like data and return state diffs or typed
objects. They should not perform Siril/GraXpert work, mutate files, or hide
heavy I/O side effects. Nodes call them and return their diffs through
LangGraph.

`nodes.py` can route. It should not become the policy owner.

## Existing Patterns To Reuse

### Review Session

`state.review_session` plus `muphrid.graph.review` is the pattern for a
phase-spanning, UI-visible, prompt-visible, routed policy concern.

HITL policy lives in `muphrid.graph.review`, backed by
`state.review_session`.

`review_session` is the only source of truth for:

- whether a HITL gate is open,
- what the gate is waiting on,
- which variants are approvable,
- whether visible agent text is required before more tool calls,
- how many HITL-mapped tool runs have happened since the last human event.

`variant_pool` is the workbench/history. It records what the agent tried.
It is observational. It is not an approval contract.

A variant becomes approvable only when the agent calls `present_for_review`
and updates `review_session.proposal.candidates`. Gradio approval reads
from that proposal. The model prompt reads from that proposal. Validation
reads from that proposal. There is no fallback to "last N pool entries."

Approval is structural:

- Gradio sends `{"type": "approve_variant", "variant_id": "...", ...}`.
- Chat feedback sends `{"type": "feedback", "text": "..."}`.
- The controller dispatches by event type.

Do not reintroduce:

- `presented_for_review`,
- `__APPROVE__` sentinel strings,
- regex approval parsing,
- "find active HITL by scanning old messages",
- UI-side approval rules that are not derived from `review_session`.

If a checkpoint has `active_hitl=True` but no open `review_session`, treat it
as legacy/broken state. Do not silently reconstruct a gate from messages.

### Render-State Contract

`metadata.image_space` is authoritative. It is the current example of a
state field that cannot tolerate fallbacks.

Values:

- `"linear"` — pixel values are still proportional to signal.
- `"display"` — stretch/nonlinear display transform has been applied.

Read sites:

- Gradio preview generation,
- VLM preview generation,
- `export_final` source ICC profile selection,
- checkpoint restore/export review logic.

Rules:

- Every tool that advances `paths.current_image` must also emit
  `metadata.image_space` in the same `Command.update`.
- Readers refuse if `image_space` is missing or invalid.
- Readers do not consult `metrics.is_linear_estimate`, FITS history, file
  names, or "safe defaults" as a fallback.
- `metrics.is_linear_estimate` remains diagnostic only.
- Every checkpoint entry is `{"path", "image_space"}`, built by the single
  writer `checkpoint.make_checkpoint_entry`. Both `save_checkpoint` and the
  graph's `auto_checkpoint` go through it, so it is the only place the entry
  shape is constructed and the only image_space guard. Readers (e.g.
  `restore_checkpoint`) trust the shape and do not defensively re-check it.
  There is no bare-string format to support.

Why this matters: HITL is meaningless if the preview shown for approval is
not the artifact that moves forward or exports. State authority is what
prevents hidden autostretch/export mismatches.

`muphrid.graph.registry._assert_image_space_writers` enforces the writer
half of this contract at import time.

### Export Review

Final export review is not an ordinary image-processing variant.

When `export_final` HITL is enabled:

- `export_final` stages files into a tentative export directory,
- the review candidate points at the actual rendered export JPG,
- human approval commits the tentative files to `export/`,
- `paths.current_image` remains the working FITS,
- exported JPG/TIF/JXL artifacts are not promoted into the processing chain.

The model does not decide whether final export review is tentative. The
system derives that from HITL policy. Autonomous mode can still write
directly.

### Variant Pool And Proposal

`variant_pool` is a workbench. It records concrete outputs the agent has
tried in the current segment.

`review_session.proposal` is an approval contract. It records which pool
entries the agent has deliberately presented for approval.

Keep those meanings separate:

- pool = observational history,
- proposal = actionable set,
- commit/promote = state transition.

Do not make the UI infer approvability from "last N variants" or from what
happens to be visible in a gallery.

Each `Variant` also carries `image_space` (captured from
`metadata.image_space` at snapshot time), so readers can tag a variant's
render space without re-deriving it.

### Comparing Images (compare_images)

"Which candidate is better" is a read-only, by-handle question. The agent
should never juggle several `analyze_image` firehoses or promote a variant
just to measure it.

`compare_images` takes handles the agent already sees — variant ids
(`variant_pool`), checkpoint names (`metadata.checkpoints`), or the literal
`"current"` — never a path. It resolves each handle internally, computes
metrics fresh per image, and returns a **metric-major** table (each metric
lines up across images) plus a legend tagging each image's `image_space`.

Design contracts to preserve:

- **Read-only.** It returns only a `ToolMessage`; it does not promote,
  restore, re-point `current_image`, or touch the regression baseline. It is
  correctly absent from `current_image_writer_names()` and takes no
  `tool_effects` entry.
- **Recompute, don't trust the pool slice.** `Variant.metrics` is captured
  before the agent analyzes that variant, so it is stale/offset. Comparison
  computes each image's metrics fresh.
- **No drift with analyze_image.** It reuses `analyze_image.func()` on a
  synthetic per-image state (the Synthetic State Exception) and reads the
  metrics from the returned `Command`, rather than reimplementing the metric
  core. The agent never sees the synthetic state; the real state is untouched.
- **Render-space honesty.** Each ref's `image_space` is authoritative (from the
  variant/checkpoint/current contract, never `is_linear_estimate`); mixing
  linear and display refs emits a warning rather than comparing incomparable
  background/SNR numbers.

This generalizes to every post-stack decision (linear → export): the agent
picks the metric groups that matter for the decision in front of it. It does
not apply pre-stack, where the unit is a sequence (`analyze_frames`).

### Decision Records & Revisit (Converged, Not Irrevocable)

Choosing a variant is a *commit*, not a one-way door. Approving (HITL) or
committing (`commit_variant`, autonomous) records the decision and **keeps the
alternatives**, so a converged step can be reopened later instead of being
lost when the live pool clears. This is the harness's version-control model:
the working tree is `variant_pool`, the commit is the decision, and history is
recoverable — modeled on how a coding agent relies on git, where a commit
never destroys the ability to go back.

Iteration is the LLM's job in **every** mode; HITL only adds a pause to
converse. So the substrate is mode-independent:

- `metadata.decisions["<phase>:<tool>"]` is a `DecisionRecord` — the chosen
  variant, ALL the candidates that were on the table, and the pre-step image.
- `build_variant_promotion_update` is the **single writer**, shared by HITL
  `promote_variant` and the autonomous `commit_variant`, so convergence is
  recorded identically with or without a human. It also re-asserts
  `metadata.image_space` (promotion advances `current_image`).
- `metadata.step_anchors["<phase>:<tool>"]` is the pre-step image+space,
  captured set-once by `auto_checkpoint` (the moment the correct pre-step space
  is known — it may differ from the post-tool space, e.g. stretch). It fills
  `DecisionRecord.pre_step`.
- `revisit_decision` is the loop-back, and it is an **agent** tool, not
  HITL-only: it restores the pre-step image (+image_space — it is a
  registry-verified `current_image` writer), repopulates the pool from the
  recorded candidates, clears the analysis baseline, and reopens the review
  session iff that tool's gate is enabled. It is **append-only**: the record
  is preserved; revisiting starts a fresh iteration seeded from it, it does not
  undo history. It refuses (no fallback) when the pre-step image is missing.

The disposition that makes the brain *use* this (a capability is not a
disposition) lives in the always-on `SYSTEM_BASE` ("decisions are
provisional… prefer revisiting an upstream decision over forcing forward"),
not the HITL gate prompt — because the autonomous brain needs it too. The gate
prompt stays a thin pause-and-surface overlay. Revisitable decisions are
surfaced in the prompt (`_format_decisions_for_prompt`) like checkpoints.

### Tools As Product Features

The agent's capabilities are the tools it can call. If a user asks for a
PixInsight-level operation, do not solve it with prompt wording alone.
Give the agent the right feature.

Example: expert star treatment is not a HITL feature. HITL feedback exposed
the gap, but autonomous mode needs the same capability. The correct product
surface is a set of tools:

- `star_removal` creates starless + star component/mask,
- `analyze_star_population` exposes per-source measurements,
- `selective_star_reblend` builds a population-weighted recomposition,
- `enhance_star_color` or `star_saturation_multiplier` controls chroma.

Keep tool semantics honest. A tool that operates on real image luminance can
use MAD/noise-driven thresholds. A tool that operates on a clean mask should
have mask-native behavior. Do not make synthetic fixtures drive production
fallbacks unless the fixture represents the production input type.

### Registry Guards

`muphrid.graph.registry` is allowed to refuse startup when a registered tool
violates a structural contract. This is a feature, not brittleness. Failing
at import is much cheaper than letting an agent discover schema drift or
state-authority drift halfway through a run.

Current guards include:

- schema/function drift for `@tool(args_schema=...)`,
- image-space writer drift for tools that advance `paths.current_image`.

`registry.current_image_writer_names()` is the registry-derived set of tools
that advance `paths.current_image` (same AST basis as the image-space guard,
plus resolution of `update = {...}; Command(update=update)` locals). It is the
authority for "which tools advance the working image," reused by effect
detection below. Compound tools that build `paths` via a name reference are not
detected — the documented blind spot; it only costs them no-op collapsing.

### Tool-Effect Detection (No-Op Contract)

Whether a tool call actually changed the working image is authoritative state,
not something to reconstruct from the transcript.

`state.tool_effects` maps `tool_call_id -> did this call change
paths.current_image`. It is produced by the orchestration, not self-reported
by tools:

- `auto_checkpoint` captures `state.pre_action_image` before the tool node
  runs (every action step, independent of whether a checkpoint is recorded),
- `variant_snapshot` diffs `pre_action_image` against the live
  `paths.current_image` after the tool node and records the result for each
  image-advancing call (scoped to `current_image_writer_names()`; errored
  calls excluded; multi-writer steps recorded as changed to avoid false
  no-ops).

`_check_stuck_loop` reads `tool_effects` to collapse no-op calls onto a single
fingerprint. It does NOT scan `ToolMessage` content for a `"noop"` string —
that was the message-scan anti-pattern. Message *structure* (tool_calls,
segment boundaries) is still walked; only the no-op *status* now comes from
state.

Rules:

- Do not add per-tool `"noop": true` message strings for loop detection. A new
  pointer/transformation tool that advances `paths.current_image` gets no-op
  detection for free by being in `current_image_writer_names()`.
- A tool MAY still compute no-op for its own purposes — gating a side-effect or
  narrating to the agent (see `restore_checkpoint`, which skips clearing the
  analysis baseline on a no-op restore). That computation is local; it is not
  the detector's source of truth, and the two agree by construction (a no-op
  leaves `current_image` unchanged, which the state diff also sees).
- `commit_variant` is not in the writer set (it builds `update` in a helper),
  but it carries its own pool-clearing idempotency, so it does not depend on
  this path.

## Reducer Discipline

`AstroState.paths`, `metadata`, and `metrics` have merge-aware reducers.
Tool updates must emit deltas, not whole copied fields.

Bad:

```python
return Command(update={
    "paths": {**state["paths"], "current_image": output_path}
})
```

Good:

```python
return Command(update={
    "paths": {"current_image": output_path},
    "metadata": {"image_space": incoming_image_space},
})
```

Why: modern models can call tools in parallel. If each parallel tool spreads
its stale snapshot of `state["paths"]`, later reducer order can clobber
sibling updates.

Use `Replace(...)` only for fields that intentionally need full replacement
through a replace-aware reducer, such as `metrics` clear/restore paths.

Plain replace-semantics fields (`variant_pool`, `visual_context`,
`regression_warnings`) should have a single writer per super-step that
recomputes the full list. If a second additive writer becomes necessary,
build a reducer that expresses that explicitly.

### Synthetic State Exception

When a tool calls another tool's `.func()` directly with a synthetic state
dict, that synthetic dict does not go through reducers. In that case the
inner tool expects a complete state shape, and spreading a full dict can be
correct. This applies to compound tools like `masked_process` and
`hdr_composite`.

Be deliberate: `Command.update` payloads are deltas; synthetic state passed
to direct `.func()` calls is a complete input object.

## Tool Schema Drift

Every `@tool(args_schema=Foo)` must match the underlying function signature
for non-injected args. The registry import guard `_assert_no_schema_drift`
refuses to start the system when schema and function drift.

This class of bug is otherwise brutal for agents: Pydantic fills a schema
field, Python raises `unexpected keyword`, the agent retries, and the
stuck-loop detector eventually aborts.

Also remember the langchain-core empty-schema trap: an args schema with no
fields can cause injected `state` / `tool_call_id` to be discarded. If a tool
needs no user args, add one harmless optional field (see `AnalyzeFramesInput`).

## How To Add Or Change A Feature

Use this checklist before coding:

1. **What is the user-visible capability or bug?**
   Name the behavior, not the implementation.
2. **Which layer owns truth?**
   Tool, metric, state contract, controller, graph edge, UI renderer, or prompt.
3. **Is there an existing pattern?**
   Prefer `review.py`, `metadata.image_space`, `variant_pool/proposal`,
   `registry` guards, or existing tool schemas over new shapes.
4. **What must refuse rather than fallback?**
   Any authoritative state read should fail loudly when missing.
5. **What tests assert the contract?**
   Add focused scripts that exercise state transitions and refusal paths,
   not just happy-path syntax.

Examples:

- "Agent needs to reduce star clutter but keep bright colorful stars" is a
  tool capability problem. Add or improve star-population tools.
- "Preview and export disagree" is an authoritative render-state problem.
  Add state contract and reader/writer enforcement.
- "Human approval is ambiguous" is a typed event/controller problem.
- "A step should happen before another step" may just be graph routing.
- "The UI looks wrong but state is correct" is a Gradio renderer problem.

## Debugging Runs

When a tool loop or stuck run happens, read the actual message stream before
patching.

The checkpoint DB contains the `ToolMessage.content` the agent saw. Audit
reports only exist for phases that advanced successfully, so they often miss
the failing tool output.

Quick inspection pattern:

```python
import sqlite3
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

conn = sqlite3.connect("checkpoints.db")
cur = conn.cursor()
cur.execute(
    "SELECT type, checkpoint FROM checkpoints "
    "WHERE thread_id=? ORDER BY checkpoint_id DESC LIMIT 1",
    (thread_id,),
)
type_, blob = cur.fetchone()
cp = JsonPlusSerializer().loads_typed((type_, blob))
for m in cp["channel_values"]["messages"][-15:]:
    print(type(m).__name__, getattr(m, "name", ""), str(getattr(m, "content", ""))[:400])
    if tc := getattr(m, "tool_calls", None):
        print(" tool_calls:", [
            (t["name"], {k: v for k, v in t.get("args", {}).items()
                         if k not in ("state", "tool_call_id")})
            for t in tc
        ])
```

`StuckLoopError` is a symptom. The cause is the repeated tool error message
immediately before it.

## Checkpoint DB Corruption

`sqlite3.DatabaseError: database disk image is malformed` means the SQLite
checkpoint file is physically damaged. Common causes:

1. a Python process was killed mid-write,
2. two processes wrote the same DB concurrently,
3. disk/filesystem fault.

`gradio_app._check_checkpoint_db_integrity` runs `PRAGMA integrity_check`
at startup and refuses corrupt DBs with an actionable message.

Recovery is explicit, not automatic:

```bash
uv run python scripts/recover_checkpoint_db.py [<db_path>]
```

Do not silently self-heal on launch; corruption may be evidence of a real
operator or filesystem problem.

## How To Extend The System

Before changing code, classify the concern:

- Is it a processing capability? Add or improve a tool.
- Is it a durable workflow contract? Add typed state and maybe a controller.
- Is it only control flow? Prefer graph structure.
- Is it display of existing truth? Update UI rendering, not policy.
- Is it a diagnostic observation? Put it in metrics, not authoritative state.

Then add tests at the level of the contract:

- controller transitions,
- refusal on missing authoritative state,
- schema/function drift import,
- variant/proposal approval rules,
- real artifact paths for HITL/export,
- tool behavior on representative input types.

The goal is an agentic astrophotography system that can be reasoned about in
finite time. LangGraph state is how we make that possible.
