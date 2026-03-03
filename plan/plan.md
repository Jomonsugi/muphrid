# AstroAgent — Build Plan

> Spec: `AstroAgent_Spec.md` — the source of truth for all contracts, schemas, and CLI commands.
> Check off each item as it is complete. The system is ready when all boxes are checked.

---

What to build	Review before proceeding

Phase 0+1 together — project scaffold, Siril engine, TypedDicts	Verify run_siril_script works against a real Siril install. TypedDicts import cleanly.
Phase 2 (T01–T08) — pre-processing tools	Run T01→T08 on real data. Stack exists on disk. This is the most linear and testable block.
Phase 3 (T09–T13) — linear tools	Run each on the stacked output from Phase 2. GraXpert integration is finicky — get it right here.
Phase 4+5+5b (T14–T27) — stretch, non-linear, scikit-image	Test each tool in isolation on real data. StarNet is the risk here.
Phase 6 milestone	Run the full manual script. Every intermediate file exists. Masked-application pattern works. Hard gate.
Phase 7 — graph wiring	Build State → Tool Executor → routing → planner → assembly in that order. Test graph compiles and routes without running end-to-end.
Phase 8+9+10 together	HITL node, system prompt, and CLI entry point are tightly coupled — the interrupt/resume loop only makes sense when all three exist.
Phase 11	Full run on real data. Final gate.

The riskiest parts to get right before moving on:
run_siril_script() (Phase 0) — everything Siril-based calls this. If output parsing is wrong here, every tool downstream is broken.
T01 ingest_dataset — if file classification is wrong (misreading IMAGETYP headers), calibration will fail silently.
T09 remove_gradient — GraXpert CLI invocation via Siril pyscript is not straightforward; get this working with real data before assuming it works.
Phase 7 tool executor post-hook — the auto-analyze + auto-HITL routing is the most architecturally novel part. Build it as a standalone unit, test the routing logic, then integrate.

Schema	Type	Why
AstroState, PathState, Metrics, Dataset, etc.	TypedDict	LangGraph state requirement
ReportEntry, HITLPayload	TypedDict	Stored in state or passed through interrupt()
ToolEntry	dataclass	Registry entry, not state
All tool input schemas (T01–T27)	Pydantic BaseModel	Runtime validation + LLM schema generation

---

## Phase 0 — Foundation

*Nothing else can be built until this phase is done.*

- [x] **Spec fix: add `working_dir: str` to every tool input schema (T01–T27)**
  All tools pass working dir to the Siril engine or use it as the FITS working directory. The spec says so in §2.5 but the input schemas in §5 are missing it. Fix the spec before writing any tool code. Note: T25–T27 (scikit-image tools) use `working_dir` for FITS I/O paths, not for Siril CLI invocation.
  T01 (`root_directory` fills the same role) and T06 (pure in-memory Python, no file I/O) intentionally skipped.

- [x] **Spec fix: clarify T07 frame-exclusion mechanism**
  `siril_stack` takes `accepted_frames` but Siril's `stack` command works on a sequence, not a file list. Exclusion requires Siril's `unselect sequencename from to` command. Documented in T07 contract (§5, T07): parse `.seq` for filename→index map, emit `unselect <seq> 0 <N-1>` to deselect all, then `select <seq> <idx> <idx>` for each accepted frame, then run `stack`.

- [x] **Project scaffold**
  Create the directory layout from §2.7. `pyproject.toml` with all dependencies from §10.2. Gitignored `.env`, committed `.env.example` with all keys from §10.1.

- [x] **`config.py` — environment loading**
  Load all values from `.env` using `python-dotenv`. Raise `ConfigError` with the key name for any missing required value. Spec: §10.1.

- [x] **`tools/_siril.py` — Siril Script Engine**
  `run_siril_script(commands: list[str], working_dir: str) -> SirilResult`. Writes temp `.ssf`, invokes `siril-cli -d <working_dir> -s <script>`, captures stdout/stderr, applies regex patterns, raises `SirilError` on failure. `SirilResult` (stdout, stderr, exit_code, script, working_dir, parsed) and `SirilError` dataclasses. Spec: §2.5.

- [x] **External dependency check**
  On startup, verify `siril-cli --version ≥ 1.4`, GraXpert binary resolves, and StarNet v2 binary resolves at `STARNET_BIN`. Also verify `import skimage` and `import pywt` succeed (scikit-image ≥ 0.22, PyWavelets ≥ 1.6). All are required — fail with an actionable install message if any are missing. Spec: §10.1.

---

## Phase 1 — Data Schemas

*Imported by every tool and graph node. Build before anything else.*

- [x] **`graph/state.py` — all TypedDicts**
  `AstroState`, `PathState`, `MasterPaths`, `Metadata`, `Metrics`, `Dataset`, `FileInventory`, `AcquisitionMeta`, `FrameMetrics`, `ReportEntry`, `HITLPayload` with correct `Annotated` reducers for `history` (append-only), `messages` (`add_messages`), and `processing_report` (append-only list of `ReportEntry`). `PathState` must include `latest_preview: str | None`. Spec: §4.

- [x] **`ProcessingPhase` enum**
  All phases from §4.3. Helper: `is_linear_phase() -> bool`. Spec: §4.3.

---

## Phase 2 — Pre-Processing Tools (T01–T08)

*Each tool is a standalone `@tool`-decorated function. No LangGraph dependency. Test each in isolation with real FITS data before moving on.*

- [ ] **T01 `ingest_dataset`**
  Walk directory, classify files by `IMAGETYP` FITS header, extract `AcquisitionMeta`. Return `Dataset` + warnings. Spec: §5 T01.

- [ ] **T02 `build_masters`**
  Build master bias, dark, or flat via Siril `convert` + `stack`. Enforce dependency order (bias → dark → flat). Return `master_path` + stats. Spec: §5 T02.

- [ ] **T03 `siril_calibrate`**
  Apply masters to lights via Siril `calibrate`. Handle CFA/OSC flag, `equalize_cfa`, cosmetic correction, dark optimization. Return calibrated sequence path. Spec: §5 T03.

- [ ] **T04 `siril_register`**
  Two-pass registration via Siril `register -2pass` + `seqapplyreg`. Support homography/affine/shift, Lanczos4 interpolation, optional drizzle, FWHM/roundness inline filtering. Spec: §5 T04.

- [ ] **T05 `analyze_frames`**
  Per-frame FWHM, eccentricity, roundness, background, noise via Siril `seqstat` + Photutils `DAOStarFinder`. Return `frame_metrics` dict + summary. Spec: §5 T05.

- [ ] **T06 `select_frames`**
  Pure Python sigma-clipping on T05 metrics. Reject frames outside FWHM, roundness, star count, and background thresholds relative to median. Safety: never return zero accepted frames. Spec: §5 T06.

- [ ] **T07 `siril_stack`**
  Mark rejected frames with Siril `unselect`, then run `stack rej` with sigma/winsorized rejection, addscale normalization, wfwhm weighting, 32-bit output. Spec: §5 T07.

- [ ] **T08 `auto_crop`**
  Load FITS with Astropy, compute bounding box on signal mask (per-pixel max across channels > threshold), apply 5px safety inset, run Siril `crop`. Spec: §5 T08.

---

## Phase 3 — Linear Processing Tools (T09–T13)

*All operate on linear data. Each must guard against being called post-stretch.*

- [ ] **T09 `remove_gradient`**
  Primary: call `GRAXPERT_BIN` directly via subprocess — do NOT use Siril's `pyscript GraXpert-AI.py` wrapper. GraXpert CLI: `graxpert <image> -cli -cmd background-extraction -output <out> -correction <type>`. Fallback: Siril `subsky -rbf` or `subsky <degree>`. Return processed image path + optional background model. Spec: §5 T09.

- [ ] **T10 `color_calibrate`**
  Internally calls `plate_solve` if WCS not present. Then Siril `pcc` or `spcc`. Return calibrated path, WCS coords, pixel scale, color coefficients. Spec: §5 T10.

- [ ] **T11 `remove_green_noise`**
  Siril `rmgreen` with average-neutral or maximum-neutral protection. Only valid on OSC/DSLR data. Spec: §5 T11.

- [ ] **T12 `noise_reduction`**
  Primary: Siril `denoise` (NL-Bayes, DA3D, SOS). Alternative: call `GRAXPERT_BIN` directly via subprocess — do NOT use Siril's `pyscript` wrapper. GraXpert CLI: `graxpert <image> -cli -cmd denoising -output <out> -strength <value>`. Measure and return `noise_before`/`noise_after` via `bgnoise`. Spec: §5 T12.

- [ ] **T13 `deconvolution`**
  Methods: Siril `rl` (Richardson-Lucy + TV regularization), Siril `wiener`, or GraXpert `pyscript -deconv_stellar` / `-deconv_obj`. PSF via Siril `makepsf -auto`. Spec: §5 T13.

---

## Phase 4 — Stretch & Non-Linear Tools (T14–T19)

- [ ] **T14 `stretch_image`**
  Siril `ght` (GHS), `asinh`, or `autostretch`. Measure `clipped_shadows_pct` and `clipped_highlights_pct` from output stats. Set `metadata.is_linear = false` in state. Spec: §5 T14.

- [ ] **T15 `star_removal`**
  Siril `starnet` *or* call StarNet binary directly via subprocess. Spec: §5 T15.

  **Note — PyTorch/MPS StarNet (starnet2-mps):** The experimental build uses a different CLI than the TensorFlow version. Uses `-i` input, `-o` starless, `-m` mask, `-w` weights path, `-u` upsampling. Siril's `starnet` command may not forward correctly to this binary. Consider calling `STARNET_BIN` directly with `STARNET_WEIGHTS` (or weights path beside binary) — produces both starless + mask in one run. Binary path requires `install_name_tool -add_rpath <lib_dir>` and `codesign --force --sign -` on first install (see StarNet setup docs).

- [ ] **T16 `curves_adjust`**
  Siril `mtf` (global) or `ght` with high B value (targeted tonal range). Per-channel support. Spec: §5 T16.

- [ ] **T17 `local_contrast_enhance`**
  Siril `clahe`, `unsharp`, or `wavelet` + `wrecons` with per-layer weights. Spec: §5 T17.

- [ ] **T18 `saturation_adjust`**
  Siril `satu` with background protection factor and optional hue index (0–6). `ght -sat` for midtone-focused saturation. Spec: §5 T18.

- [ ] **T19 `star_restoration`**
  Blend mode: Siril `pm "$starless$ + $starmask$ * <weight>"`. Synthstar mode: Siril `synthstar` on original linear image. Spec: §5 T19.

---

## Phase 5 — Utility Tools (T20–T24)

- [ ] **T20 `analyze_image`**
  Siril `stat main` + `bgnoise` + `histo` (per channel). Photutils for star count and FWHM. Compute derived metrics: `gradient_magnitude`, `green_excess`, `snr_estimate`, `is_linear_estimate`, `background_flatness_score`. Spec: §5 T20.

- [ ] **T21 `plate_solve`**
  Siril `platesolve` with RA/Dec hint, focal length, pixel size. Parse WCS result from stdout. Fallback to local astrometry.net. Spec: §5 T21.

- [ ] **T22 `generate_preview`** *(internal function — not agent-callable)*
  Copy FITS to temp, run Siril `autostretch` + `savejpg` (linear only), resize to target width. Original FITS untouched. Output to `previews/`. Called exclusively by `auto_hitl_check()` and the mandatory HITL nodes (`stretch_hitl`, `final_hitl`). Must not be registered in `PHASE_TOOLS`. Spec: §5 T22.

- [ ] **T23 `pixel_math`**
  Siril `pm "<expression>"`. Validate that all `$variable$` stems exist in working dir before invoking. Spec: §5 T23.

- [ ] **T24 `export_final`**
  Siril `icc_convert_to` + `savetif`/`savejpg`/`savepng`. Always produce 16-bit TIFF (AdobeRGB) + JPG (sRGB). Output to `export/`. Spec: §5 T24.

---

## Phase 5b — scikit-image Tools (T25–T27)

*Pure-Python tools. No Siril invocation. All I/O via `astropy.io.fits`. Each must
be tested in isolation on a real stretched FITS before wiring into the graph.*

- [ ] **T25 `create_mask`**
  Read FITS with `astropy.io.fits`. Compute luminance via `skimage.color.rgb2gray`.
  Apply threshold range to produce a binary 0/1 mask. Feather edges with
  `skimage.filters.gaussian(sigma=feather_radius)`. Optionally expand/contract
  mask with `skimage.morphology.binary_dilation` / `binary_erosion` using
  `skimage.morphology.disk`. Support mask types: `luminance`, `inverted_luminance`,
  `range`, `channel_diff`. Write single-channel float32 FITS. Return `mask_path`,
  `coverage_pct`, `mean_value`. Spec: §5 T25.

- [ ] **T26 `reduce_stars`**
  Accept FITS image and optional star mask from T15. If no mask provided,
  auto-threshold luminance at `detection_threshold` to build a star region mask.
  Apply `skimage.morphology.erosion(channel, skimage.morphology.disk(kernel_radius))`
  for `iterations` passes only within the star mask region. Optionally protect
  star cores via `binary_dilation` exclusion zone. Blend eroded result with
  original using `blend_amount` and `skimage.filters.gaussian` feathering.
  Return `reduced_image_path`, `stars_affected_count`, `mean_size_reduction_pct`.
  Spec: §5 T26.

- [ ] **T27 `multiscale_process`**
  Load FITS as float32. If `per_channel=False`, extract luminance with
  `skimage.color.rgb2gray`, process, recombine. Apply `pywt.swt2(data, wavelet='b3',
  level=num_scales)` to get wavelet coefficients per scale. For each scale apply:
  `sharpen` (multiply coefficients by weight), `denoise` (apply
  `skimage.restoration.denoise_wavelet` with `denoise_sigma`), `suppress` (zero
  coefficients), or `passthrough`. Reconstruct via `pywt.iswt2`. If `mask_path`
  supplied, blend result with original via mask. Write FITS. Return
  `processed_image_path` and `per_scale_stats` with coefficient energy before/after.
  Spec: §5 T27.

---

## Phase 6 — Tool Layer Milestone

*Prove tools work independently before building the agent.*

- [ ] **Manual pipeline integration script**
  A plain Python script (not a test, not the graph) that calls T01 → T27 sequentially on a real dataset. Every intermediate file must exist on disk. The script must demonstrate the masked-application pattern: T25 → T27 → T23 on the starless image. The script completes without error and produces a viewable export. This is the gate before Phase 7.

---

## Phase 7 — LangGraph Graph

*Wire the tools into the orchestration layer.*

- [ ] **State reducers and `make_initial_state()`**
  Confirm `history` appends (not replaces) across updates. `make_initial_state(dataset_path)` calls T01 and returns a fully populated `AstroState` with `phase = INGEST`. Spec: §4, §6.

- [ ] **`tools/control.py` — `advance_phase` and `request_hitl`**
  Both as `@tool`-decorated functions with the exact signatures and docstrings from §6.4. These are the LLM's signals for routing — not processing tools.

- [ ] **`PHASE_TOOLS` registry**
  Dict mapping each `ProcessingPhase` to the subset of tools the LLM may call during that phase. Verify: `color_calibrate` is absent from `NONLINEAR`, `stretch_image` is absent from `LINEAR`, `reduce_stars` is absent from `LINEAR`, `multiscale_process` in `LINEAR` accepts only `denoise` operations (enforced via agent notes, not code). Spec: §6.3.

- [ ] **Tool executor node**
  `ToolNode` wrapping all 27 tools (T01–T27). After execution: update
  `state.paths.current_image` and `state.history`; run `tool_executor_post_hook`
  (auto-analyze + auto-HITL check via `ToolEntry` flags); append a `ReportEntry`
  to `state.processing_report` using `build_report_entry()`. No auto-preview,
  no `state.paths.latest_preview` update in this path — preview generation
  happens only inside `auto_hitl_check()` and the mandatory HITL nodes.
  Wrap all tool errors in structured `ToolMessage` JSON. Spec: §6.5, §6.7, §9.2.

- [ ] **`parameter_selector()` — pre-processing parameter LLM**
  Single-shot LLM call that receives dataset context (frame counts, camera
  type, is_osc, bortle, prior metrics) and returns a validated parameter dict
  for the named pre-processing tool plus a `reasoning` string. Not a ReAct
  loop — one call per node, one response. Validate the returned parameters
  against the tool's input schema before use. Spec: §6.2.

- [ ] **`build_preprocess_node()` factory**
  Wraps each pre-processing tool (T02–T08) with: (1) `parameter_selector` LLM
  call, (2) tool execution, (3) `ReportEntry` append. Order enforced by graph
  edges, not LLM routing. `build_masters` node skips parameter selection and
  execution if masters already exist. Spec: §6.2.

- [ ] **Pre-processing subgraph**
  Order-deterministic `StateGraph` using `build_preprocess_node` for each of
  T02–T08. Fixed linear edges enforce the calibration → registration → analysis
  → selection → stacking → crop sequence. Compiles to a subgraph invoked from
  the top-level graph. Spec: §6.2.

- [ ] **Planner node**
  Assemble text-only prompt: `build_state_summary(state)` as a plain string `HumanMessage`. No image attached. Bind `PHASE_TOOLS[phase] + CONTROL_TOOLS`. `generate_preview` must not appear in any `PHASE_TOOLS` list. Invoke LLM. Return `{"messages": [response]}` only. Spec: §6.3.

- [ ] **`route_planner` function**
  3-way routing on last message's tool call: `call_tool` → tool executor, `request_hitl` → hitl node, `advance_phase` or no tool call → next phase node. Spec: §6.5.

- [ ] **Top-level StateGraph assembly**
  All nodes and edges from §6.1. `interrupt_before=["stretch_hitl", "final_hitl"]`. Graph compiles without error. Auto-HITL for `remove_gradient`, `deconvolution`, and `star_removal` is handled inside the tool executor post-hook via `auto_hitl_check()`, not via additional `interrupt_before` nodes.

- [ ] **Checkpointing**
  `SqliteSaver` wired to compiled graph. Thread ID format: `{dataset.id}_{timestamp}`. Implement `resume_from_checkpoint(thread_id)`. Spec: §6.8.

---

## Phase 8 — Agent Reasoning

*The system prompt is the primary intelligence layer. Phase 8 is about wiring
it correctly and ensuring all context the LLM needs reaches it at runtime.*

- [ ] **`build_state_summary()` — runtime context renderer**
  Implements the `{placeholder}` injection for the §7.4 system prompt template.
  Must populate all fields: `{phase}`, `{target_name}` (from
  `dataset.acquisition_meta.target_name`, or "unknown" if null), `{is_linear}`,
  `{is_osc}`, `{user_feedback_summary}` (human-readable serialization of the
  full `user_feedback` dict, including `notes` and all `revision_requests`),
  `{history}` (last 10 entries from `state.history`), `{metrics_json}` (latest
  `state.metrics` serialized).   The planner node uses a text-only HumanMessage — no `image_url`. Preview
  generation never happens in the planner path. Spec: §7.4, §6.5.

- [ ] **System prompt authoring and iteration**
  The §7.4 prompt is intentionally lean — it provides only what the model
  cannot know from context (physical constraints, project conventions, state).
  Domain knowledge and aesthetic judgment belong to the model. Treat §7.4 as a
  living document: start with what's written, run against real data, and add
  content only when a specific repeatable failure occurs that cannot be fixed by
  improving a tool contract. Do not add guidance preemptively. Spec: §7.4.

- [ ] **User feedback integration — `apply_user_feedback()` helper**
  Before each non-linear ReAct cycle, parse `state.user_feedback` and inject
  a summary into the prompt context via `build_state_summary()`. Implement the
  `revision_request` execution loop: map `step` → tool, revert to the relevant
  checkpoint, re-run, call `request_hitl` with the "Revision confirmed" trigger.
  This is the mechanism that makes HITL feedback actually change subsequent
  processing — without it, user input is recorded but ignored. Spec: §8.3.

- [ ] **`render_processing_report()` — markdown report generator**
  Iterates `state.processing_report`, groups entries by phase, and renders the
  format specified in §12.2. Includes: per-step tool / reasoning / parameters /
  before-after metrics / outcome, the HITL selection narrative, and a closing
  summary table showing key metrics across pipeline stages. Called by the
  `export` node after `export_final` completes; writes to
  `{working_dir}/processing_report.md`. The report is always produced — it is
  part of the deliverable alongside the exported image. Spec: §12.

---

## Phase 9 — HITL

- [ ] **`HITLPayload` TypedDict**
  Define in `graph/state.py`: `trigger`, `question`, `options`, `allow_free_text`, `preview_paths`, `preview_labels`, `context`, `checkpoint`. This is the stable contract between the graph and any presenter (CLI, Streamlit, etc.). Spec: §8.0.

- [ ] **`hitl_node` — graph node (no I/O)**
  Handles all three HITL categories. Builds an `HITLPayload` from state (preview paths from `auto_hitl_check()` or HITL node logic, question/options from the trigger). Calls `langgraph.types.interrupt(payload)` — graph pauses here. On resume, calls `apply_hitl_response()` to write the user's response into `state.user_feedback` and append a `HumanMessage`. The node contains **zero I/O** (no `print`, no `input`, no `subprocess`). Spec: §8.0.

- [ ] **`cli_presenter` — V1 I/O handler (`cli.py`)**
  Reads `HITLPayload` from the `__interrupt__` chunk in the stream loop. Opens each `preview_path` with `subprocess.Popen(["open", path])` (non-blocking, macOS). Prints question, context, and numbered options to stdout. Reads selection from `stdin`. Returns the user's string response. Calls `app.invoke(Command(resume=response), config)` to resume the graph. Spec: §8.0.

- [ ] **`--no-hitl` mode**
  When flag is set, replace `cli_presenter` with `auto_presenter`: auto-selects the middle option from `payload["options"]` and logs the decision to history. Graph still pauses/resumes via `interrupt` — only the presenter changes. Spec: §8.0.

- [ ] **`TOOL_REGISTRY` with `ToolEntry`**
  Implement `ToolEntry` dataclass with `fn`, `modifies_image`, `requires_visual_review`, `visual_review_question`, `visual_review_options` fields. Populate all 27 tools per §6.5. `requires_visual_review=True` for `remove_gradient`, `deconvolution`, `star_removal` in V1. Changing a checkpoint from required to optional in the future is a one-line boolean change. Spec: §6.5.

- [ ] **`tool_executor_post_hook()`**
  After each tool call: (1) if `entry.modifies_image`, call `analyze_image_fn` and update `state.metrics`; (2) if `entry.requires_visual_review`, call `auto_hitl_check()` which calls `generate_preview_fn` and returns a HITL payload; (3) inject payload into state routing so the graph routes to `hitl_interrupt_node` before returning to planner. `generate_preview_fn` is never called outside this hook and the mandatory HITL nodes. Spec: §6.5.

- [ ] **Feedback integration**
  User's free-text revision requests in `state.user_feedback["revision_requests"]` are read by the planner on resume and factored into parameter selection on the next iteration. Spec: §8.3.

---

## Phase 10 — Entry Point

- [ ] **`cli.py`**
  `astro_agent process <directory> [--profile conservative|balanced|aggressive] [--resume <thread_id>] [--no-hitl]`. Runs `make_initial_state` then streams the graph via `app.stream(..., stream_mode="updates")`. On each chunk, checks for `__interrupt__`: if present, calls `cli_presenter(payload)` to display images + question, reads response, and resumes with `app.invoke(Command(resume=response), config)`. If `--no-hitl`, uses `auto_presenter` instead. Print thread ID at start so the user can resume later. Spec: §2, §8.0.

---

## Phase 11 — End-to-End Validation

- [ ] **Full pipeline run on a real dataset**
  Run `astro_agent process` on 10–30 light frames with calibration sets (bias, dark, flat) of a bright target. All phases complete. Both HITL checkpoints pause and resume correctly. Final TIFF and JPG are written to `export/`. A person looking at the output judges it as a competent result.
