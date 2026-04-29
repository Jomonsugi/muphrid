# Architecture

This document explains how Muphrid works, why the main components exist, and how they fit together. It is written for someone who wants to understand the system without reading every tool implementation first.

Muphrid is an agentic astrophotography processing pipeline. It combines a LangGraph state machine, a phase-gated tool registry, open-source image-processing engines, quantitative image analysis, and explicit human review sessions.

---

## System Overview

```text
Dataset folder
    |
    v
Session context
target, sky quality, equipment notes
    |
    v
LangGraph processing graph
    |
    +--> phase_router
    |        selects phase-specific tool set
    |
    +--> agent
    |        LLM with current phase tools bound
    |
    +--> auto_checkpoint
    |        bookmarks image before mutating tools
    |
    +--> action
    |        executes tool calls
    |
    +--> variant_snapshot
    |        captures reviewable image variants
    |
    +--> hitl_check
             opens or resumes human review gates

Outputs:
    runs/<thread-id>/
        calibrated/ registered/ stacked/ processed files
        variants/
        previews/
        reports/
        processing_log.md
```

The graph loops until the agent advances through all processing phases and exports the final image. Every phase exposes only the tools that make sense for that phase.

---

## Processing Phases

The pipeline is organized as explicit phases in `muphrid/graph/state.py`:

| Phase | Purpose |
|-------|---------|
| `ingest` | Resolve target and inspect dataset metadata |
| `calibration` | Build masters, convert sequences, calibrate light frames |
| `registration` | Align calibrated frames |
| `analysis` | Analyze registered frames and select acceptable inputs |
| `stacking` | Stack selected frames and crop the result |
| `linear` | Gradient removal, color calibration, noise reduction, deconvolution |
| `stretch` | Convert the linear image into a nonlinear image |
| `nonlinear` | Curves, star work, local contrast, saturation, masks, pixel math |
| `review` | Optional final review logic |
| `export` | Write final deliverables |
| `complete` | End state |

The phase router binds a different tool set for each phase through `muphrid/graph/registry.py`. This keeps the LLM from seeing tools it should not call yet, while still allowing utility tools such as analysis, preview generation, checkpoints, and review presentation where appropriate.

---

## Graph Loop

The main graph is built in `muphrid/graph/graph.py`.

```text
phase_router
    |
    v
agent
    |
    | tool calls
    v
auto_checkpoint
    |
    v
action
    |
    v
variant_snapshot
    |
    v
hitl_check
    |
    v
agent
```

If the agent produces no tool calls outside HITL, `agent_chat` nudges it back into action. This project is not a general chat app during autonomous processing; the agent is expected to call tools, inspect results, and advance phases.

Key graph nodes:

- **`phase_router`** chooses whether to continue and which tools are available.
- **`agent`** builds the prompt, attaches state-derived visual context, invokes the model, and enforces phase/tool gates.
- **`auto_checkpoint`** captures pre-call image state before post-stack mutating tools.
- **`action`** executes LangChain tools through `ToolNode`.
- **`variant_snapshot`** captures HITL-mapped tool outputs into `variant_pool`.
- **`hitl_check`** opens review sessions, interrupts for human input, validates approvals, and promotes approved variants.
- **`agent_chat`** handles text-only responses outside active review.

---

## State Model

`AstroState` is the durable contract across graph nodes, tools, UI recovery, and checkpoint resume.

Important state fields:

| Field | Purpose |
|-------|---------|
| `phase` | Current processing phase |
| `paths` | Current image, sequence names, masks, previews, masters |
| `metadata` | Dataset/equipment metadata, bookmarks, phase checkpoints |
| `metrics` | Image and frame quality measurements |
| `messages` | LangChain message history |
| `variant_pool` | Passive workbench of generated reviewable variants |
| `review_session` | Canonical HITL review state and proposal contract |
| `visual_context` | Non-variant images the model should see |
| `regression_warnings` | Metric regressions detected during analysis |

Reducers matter. Some fields are deep-merged so parallel tool calls compose safely (`paths`, `metadata`). Some are replace-aware (`metrics`). Lists such as `variant_pool` are plain replace semantics because the writer recomputes the full list.

---

## Tool Registry

Tool implementations live under `muphrid/tools/`.

```text
tools/
  preprocess/   calibration, registration, stacking
  linear/       linear-stage image processing
  nonlinear/    stretch-adjacent and nonlinear image processing
  scikit/       local Python/scikit-image processing tools
  utility/      analysis, export, checkpoints, review, masks, previews
```

Each tool is a LangChain tool with a Pydantic `args_schema`. The registry checks for schema/function drift at startup so a tool cannot silently expose parameters the function does not accept.

Most image-mutating tools return a `Command(update=...)` that writes a new `paths.current_image` and a `ToolMessage` with structured JSON. The agent reads the message; downstream tools read state.

---

## Image Processing Backends

Muphrid uses professional-grade open-source tools where possible, then fills gaps with Python libraries.

| Backend | Used for |
|---------|----------|
| Siril | Calibration, registration, stacking, background extraction |
| GraXpert | AI gradient extraction and denoising |
| StarNet2 | Star removal/restoration workflows |
| Astropy | FITS I/O, WCS, statistics |
| Photutils | Star and background measurements |
| scikit-image | Masks, morphology, local image operations |
| PyWavelets | Noise estimates and multiscale processing |

The philosophy is: use the strongest available image-processing engine for the job, and expose enough measurements for the agent to reason from data instead of vibes.

---

## Analysis and Feedback Loops

The agent is expected to inspect outcomes before moving on. `analyze_image` and `analyze_frames` produce structured metrics such as:

- background flatness
- gradient magnitude
- noise estimates
- SNR estimate
- clipped shadows/highlights
- star count and FWHM
- channel balance
- saturation
- histogram statistics

`muphrid/graph/regression.py` compares image metrics against prior analysis snapshots. When a metric worsens, the graph records a `regression_warning`. The warning does not automatically block progress; it gives the agent evidence to decide whether to accept the tradeoff, restore a checkpoint, rewind a phase, or continue.

---

## HITL Review Mode

Human-in-the-loop review is not implemented as chat parsing. It is explicit state.

`review_session` is the source of truth for:

- whether a review gate is open
- which tool opened the gate
- what the agent has proposed
- which variants are approvable
- whether the agent must answer visibly before taking tool actions
- how many HITL tool runs happened since the latest human feedback

The workflow:

```text
HITL-mapped tool runs
    |
    v
variant_snapshot captures variant_pool entry
    |
    v
hitl_check opens review_session
    |
    v
agent analyzes result
    |
    v
agent calls present_for_review
    |
    v
review_session.proposal becomes approval contract
    |
    v
Gradio/CLI interrupt waits for human feedback or approval
```

`variant_pool` is a workbench. It shows what the agent has tried. It is not an approval surface.

`review_session.proposal.candidates` is the approval contract. A human can only approve variants the agent deliberately presented through `present_for_review`.

This distinction prevents the UI and graph from disagreeing about what is actionable.

---

## Variant Pool and Proposals

At reviewable stages, tool outputs are snapshotted into `runs/<thread-id>/variants/` with stable ids such as `T09_v1` or `T14_v3`.

The agent can:

1. Run a tool once and present the result.
2. Run a small experiment with multiple parameter sets.
3. Compare metrics and visual output.
4. Present one or more candidates with rationale, tradeoffs, and a recommendation.
5. Iterate after human feedback.

Approval promotes the selected variant to `paths.current_image`, clears the pool, closes the review session, and resumes the pipeline from the approved image.

In autonomous mode, the agent uses `commit_variant` instead of human approval.

---

## Gradio UI

The Gradio app is the primary interface.

Major surfaces:

- **Processing:** chat, gallery, activity log, review state, approval controls
- **Equipment:** camera/telescope overrides for missing metadata
- **HITL Config:** review gate toggles and autonomous mode
- **Model & Limits:** model selection, recursion/tool limits, VLM retention

The UI is intentionally not the policy authority. It renders state from the graph:

- workbench filmstrip from `variant_pool`
- proposal panel from `review_session.proposal`
- approval controls from the proposal candidates
- resume/recovery state from LangGraph checkpoints

If a saved checkpoint predates Review Mode and has `active_hitl=True` without `review_session`, the UI refuses to consume a resume value as approval. That is deliberate: old implicit states are not safe approval contracts.

---

## Checkpoints, Rewind, and Audit Logs

Muphrid has several layers of recoverability:

- **LangGraph checkpoints** persist the full graph state in SQLite.
- **Automatic image checkpoints** bookmark current images before mutating post-stack tools.
- **Named checkpoints** let the agent save and restore within a phase.
- **Phase checkpoints** capture state at phase boundaries for `rewind_phase`.
- **Audit reports** summarize each phase's decisions and outcomes.

Every run writes:

```text
runs/<thread-id>/
  processing_log.md
  reports/
    01_ingest.md
    02_calibration.md
    ...
```

These logs are intended for human review and debugging. The checkpoint database is the authoritative source when diagnosing a tool failure inside a phase, because a report is only written after a phase advances successfully.

---

## VLM Context

The graph can include image previews in the model call. Visual context is state-driven rather than message-history-driven.

Sources include:

- current image preview after stacking
- active `variant_pool` previews during review
- explicit `present_images` references

`hitl_config.toml` controls autonomous visual retention. During HITL, visual access is always enabled because image collaboration without images is not meaningful.

The retention cap limits old visual references, but active review variants are preserved so the model can reason about the same images the human sees.

---

## Configuration

| File | Purpose |
|------|---------|
| `.env` | Secrets and local binary paths |
| `processing.toml` | Model, limits, tracing, behavior defaults |
| `hitl_config.toml` | Review gates, autonomous mode, VLM retention |
| `equipment.toml` | Camera/telescope metadata overrides |

Runtime Gradio settings can override many config values without rebuilding the graph. The model factory detects provider/model changes and rebinds phase tools on the next call.

---

## Key Design Decisions

**Why a phase-gated tool registry?** Astrophotography processing has a natural order. Exposing every tool at every step wastes context and invites invalid calls. Phase gating keeps the agent focused.

**Why keep `variant_pool` separate from `review_session.proposal`?** The pool is history; the proposal is intent. A human should approve what the agent deliberately presented, not whichever files happened to be most recent.

**Why typed review events instead of parsing chat?** Approval is an action, not a phrase. The UI sends typed approval events. Chat remains for questions and feedback.

**Why checkpoints and rewind?** Image processing is iterative. A good assistant needs the ability to say "that made things worse" and return to a known-good state.

**Why not fully automate every subjective choice?** Some choices are aesthetic: stretch intensity, contrast, star handling, saturation. The agent should provide data and recommendations, but the human can remain in the loop where taste matters.

**Why open-source tools?** The project is intended to be inspectable and reproducible. The north star is PixInsight-quality output, but built from tools that can be orchestrated programmatically.

---

## Known Rough Edges

This is an active project. The architecture is designed for iteration, but not every idea is finished.

- HITL flow is still mostly coordinated by one `hitl_check` node; it may eventually become a small review subgraph.
- External binaries must be installed and configured correctly.
- Model behavior varies by provider and model; safety limits are required.
- Some advanced astrophotography choices still need better evaluation metrics.
- Final image quality depends heavily on dataset quality and acquisition metadata.

The current direction is to keep cross-cutting policy concerns explicit: typed state objects, controller helpers, graph-driven transitions, and tests that assert the contract.
