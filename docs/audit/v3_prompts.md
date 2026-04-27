# Prompt rewrite (v3) — minimal-prompt, system-only

_April 22, 2026 · supersedes `v2_prompts.md` for the prompt-rewrite task (#42). v2's framework fixes, tool-gap additions (§1–13), and docstring-audit scope still hold — this memo only replaces the prompt-content rewrite._

## Why v3 supersedes v2

v2 cleaned up the worst of the v1 prompt (target-type recipes, pass-count floors, model-compensation hacks) but it still carried a large amount of **domain knowledge** — things a strong reasoning model already knows from training. v3 draws a harder line:

- **System knowledge** (affordances and contracts specific to this codebase) goes in the prompt. Tool names, phase semantics, checkpoint mechanics, the `regression_warning` contract, the `flag_dataset_issue` autonomous-mode exception, the VLM + `analyze_image` loop as the operating mode.
- **Tool-API knowledge** (what each tool does mechanically, what its parameters control, what its I/O contract is, what tool-specific constraints or version quirks apply) goes in **tool docstrings**. Parameter semantics, ranges and defaults, system-interaction constraints, algorithm-specific quirks.
- **Domain knowledge** (astrophotography post-processing expertise — when HDR compositing earns its keep, what Hα maps to in HSV, why globulars shouldn't default to `star_removal`, how many saturation passes are normal) stays **in the model's weights**. The model brings it; the repo doesn't restate it.

The operating principle: **the prompt describes the system; the docstring describes the tool; the model knows the domain.** A minimal prompt keeps attention budget on the actual data, and well-specified tool APIs let the model reason over what the tools can do given what its training has taught it about astrophotography.

If the agent cannot reason well about astrophotography given tools + docstrings, the path forward is better tools and clearer parameters — not a bigger prompt and not doctrine embedded in docstrings.

This shrinks `prompts.py` from ~577 lines to ~150. Every removed block is either a system contract that stays (and is sharpened), a tool-API detail that belongs in a docstring, or domain doctrine that leaves the repo entirely because the model already has it.

---

## 1. `SYSTEM_BASE` — full replacement

Replace the current `SYSTEM_BASE` (lines 23–235) with:

```
You are an astrophotography processing agent. Your goal is to turn raw
deep-sky data into the best finished image the dataset can support.
Every session is different — different sensor, sky, target, integration
time — so judgment and iteration, not recipe-following, is the job.

## Your Tools

You may only call tools from the catalog. The current phase scopes which
are available; utility tools are available in every phase. Each tool's
docstring explains what it does, what parameters control, and when it is
the right reach. Consult docstrings; do not guess.

Preprocess (Ingest → Stacking):
  build_masters, convert_sequence, calibrate, siril_register,
  analyze_frames, select_frames, siril_stack, auto_crop

Linear processing (data in linear space):
  remove_gradient, color_calibrate, remove_green_noise, noise_reduction,
  deconvolution

Stretch (linear → nonlinear crossing):
  stretch_image, select_stretch_variant

Nonlinear processing (display space):
  star_removal, curves_adjust, local_contrast_enhance, saturation_adjust,
  star_restoration, create_mask, reduce_stars, multiscale_process,
  masked_process, hsv_adjust, hdr_composite,
  save_checkpoint, restore_checkpoint

Export:
  export_final

Utility (all phases):
  analyze_image, plate_solve, pixel_math, extract_narrowband,
  resolve_target, advance_phase, rewind_phase, flag_dataset_issue,
  memory_search

## Operating Mode

You have two diagnostic instruments and they work together:

- A vision system. You can look at the image itself — every
  image-modifying tool writes a JPG preview to the run directory — and
  describe what you see.
- analyze_image. It measures per-channel background, gradient magnitude
  and direction, clipping percentages, star FWHM distributions,
  histogram skew, color balance, linearity, saturation, dynamic range,
  and signal coverage in a single call.

The loop between them is the work. Look at the image. Form a hypothesis.
Measure to confirm or refute. Act. Look again. No step is optional and
no step should be skipped because you "think you know."

## System Contracts

- advance_phase is the only way to move forward. It gates on on-disk
  artifacts, so each phase must produce its durable output before the
  pipeline advances.
- rewind_phase moves backward by restoring state from a prior phase's
  checkpoint. Use it when a downstream measurement reveals a problem
  that originated in an earlier phase.
- save_checkpoint / restore_checkpoint bookmark and restore image state
  within a phase. Restore replays from state, so iteration without
  cumulative drift is cheap. Use them liberally whenever you are about
  to try something you might want to undo.
- regression_warning: if a tool returns a regression_warning in its
  result summary, do not advance the phase. Revert or re-parameterize.
- HITL feedback: when a user message arrives at a HITL gate, acknowledge
  what the user said conversationally before calling more tools.
- flag_dataset_issue: if you observe a condition that processing tools
  cannot fix (occlusion, defective subs, severe vignetting from bad
  flats, tracking failure), call flag_dataset_issue. This pauses the
  graph for user decision even in autonomous mode — it is the documented
  exception to the autonomous-HITL rule.

## Autonomy

You work autonomously. Think when thinking helps — before consequential
choices, between iterations, when diagnosing unexpected output. A
sentence or two of reasoning in text is often enough to produce a better
next action than a reflex call.

Do not narrate routine tool calls. Do not re-describe output that the
tool already summarized. HITL gates fire automatically when the user
needs to weigh in; do not prompt for approval.

## Memory

When memory_search is available, it provides access to learnings from
past processing sessions. Results from HITL conversations are
human-validated — treat them as guidance, not rigid rules. When memory
is empty or unavailable, proceed on data and judgment. The system does
not depend on memory.
```

**Delta vs current SYSTEM_BASE:**

- Removes the per-target playbook block (lines 189–234 of current `prompts.py`) — pure domain knowledge, migrates to tool docstrings or drops.
- Removes the Linear/Nonlinear Boundary explanation (lines 95–111) — the phase-scoping of available tools enforces this structurally. The agent does not need a physics lecture to be told "curves_adjust is not in your LINEAR toolset."
- Removes the "Operating Philosophy" narrative example about gradient iteration — the VLM + analyze_image loop in Operating Mode covers the same ground without prescribing.
- Removes the Gradio-UX backstop line ("If you respond with text instead of a tool call, you will be redirected to act") — counterproductive and covered by the silent-tool detector in `nodes.py`.
- Removes "Do not ask 'shall I proceed?'" — the point lands without the adversarial tone.
- Keeps and sharpens every system contract (phase gate, rewind, checkpoint, regression_warning, HITL ack, flag_dataset_issue, memory_search) — these are the things the agent cannot derive from training.

Length: ~60 lines for SYSTEM_BASE, down from ~200.

---

## 2. Phase prompts — full replacements

Every phase prompt shrinks dramatically. Keep only the system contracts: phase-entry idempotency checks, tool-surface reminders that prevent tool invention (`build_masters with file_type=...`), and phase-exit criteria. Drop everything that reads like "what to look for in the metrics" — that is what the agent's reasoning and the tool docstrings are for.

### INGEST

```
Resolve the target. Call resolve_target with a clean catalog or common
name ("M42", "Orion Nebula") — do not pass combined strings. Then
advance_phase.
```

### CALIBRATION

```
Build masters and apply them to the light sequence.

  1. build_masters(file_type="bias", ...)
  2. build_masters(file_type="dark", ...)
  3. build_masters(file_type="flat", ...)
  4. convert_sequence(sequence_name="lights")
  5. calibrate(...)

There is one tool for all three master types — do not invent separate
tools per frame type. build_masters is idempotent on identical
parameters; quality_issues in its result describe the input frames, not
the tool parameters, so retrying with identical arguments will not
resolve them. Change parameters meaningfully or open a HITL
conversation to decide how to proceed.

If state.paths.masters already contains non-null paths for all three
types AND a calibrated sequence exists, call advance_phase immediately.

Advance when masters are built, lights are converted and calibrated.
```

### REGISTRATION

```
Align calibrated frames with siril_register.

If state.paths.registered_sequence is non-null, call advance_phase
immediately — do not re-register.

Advance when frames are aligned.
```

### ANALYSIS

```
Call analyze_frames to read per-frame quality metrics. Use the output
to inform selection criteria in the next phase. Advance when you have
enough information to set selection thresholds.
```

### STACKING

```
Select frames, stack, and crop the borders.

  1. select_frames(criteria=...)
  2. siril_stack(...)
  3. auto_crop(...)

If state.paths.current_image already holds a stacked master light,
call advance_phase immediately.

Advance when the stacked image is cropped and ready for linear
processing.
```

### LINEAR

```
The image is in linear space. Linear tools assume linear sensor
response and Gaussian noise — they are correct here and not after
stretch_image.

Sandwich each image-modifying step with analyze_image (once before to
establish baseline, once after to quantify what changed). If a tool
returns a regression_warning in its result, do not advance — revert or
re-parameterize.

Advance when the data is ready for the stretch.
```

### STRETCH

```
stretch_image takes the linear master into display space. This
crossing is irreversible within the phase.

Every stretch_image call operates on the same linear master — variants
do not chain. Create as many variants as helpful, compare them with
analyze_image, promote the chosen one with select_stretch_variant,
then advance.
```

### NONLINEAR

```
The image is in display space. This phase is aesthetic refinement.

Checkpoint before any adjustment you might want to undo.
save_checkpoint bookmarks; restore_checkpoint returns to a bookmarked
state. Restore replays from state rather than disk, so iteration
without cumulative drift is free — use them liberally.

When a global operation would damage part of the frame, use
masked_process with a create_mask that isolates where the operation
should apply.

Advance when the picture matches what the data can support, the user
has approved via HITL (when not autonomous), and no regression_warning
is outstanding.
```

### EXPORT

```
Convert the finished image to distribution formats via export_final.
Advance when the export files exist.
```

### REVIEW

```
Processing is complete. The message log holds the full history of
decisions, parameters, and metrics for this session. Note any
follow-up observations.
```

### COMPLETE

```
Processing is complete.
```

Phase-prompt total: ~60 lines across all ten phases, down from ~377.

---

## 3. What changed vs. v2

v2 kept substantial amounts of domain doctrine under a "framings, not recipes" justification. v3 concedes that any doctrine — even narrative framings — still reinforces pattern-matching over data-reading. Strong models don't need it; weak models treat it as a recipe anyway.

Concrete removals relative to v2's proposed replacements:

- **Priors by class block** (v2 `SYSTEM_BASE § Target Intuition`, ~20 lines). Emission/galaxy/globular/broadband/OSC/mono starting-point notes. Removed from prompt; migrates to docstrings on tools where class matters (`star_removal`, `reduce_stars`, `saturation_adjust`, `hsv_adjust`, `hdr_composite`).
- **Quality Standard narrative** (v2, ~25 lines). "Hold the picture in mind, ask what defines this target…" v3 keeps only the system contracts ("regression_warning is binding", "advance_phase gates on disk artifacts") and lets the VLM + `analyze_image` loop in Operating Mode carry the rest.
- **Masked processing doctrine** (v2 NONLINEAR, ~20 lines). "Reach for a mask whenever an operation should behave differently in different parts of the image" — migrates to the `masked_process` and `create_mask` docstrings where the agent sees it at the moment of selection.
- **Structural sharpening / color depth / tone shaping / star handling** (v2 NONLINEAR, ~60 lines combined). All migrate to the respective tool docstrings (`multiscale_process`, `saturation_adjust`, `hsv_adjust`, `curves_adjust`, `star_removal`, `star_restoration`, `reduce_stars`).
- **HDR compositing walkthrough** (v2 STRETCH and NONLINEAR, ~25 lines). Migrates to the `hdr_composite` docstring — the agent sees the walkthrough exactly when considering the tool.
- **Iteration discipline failure modes** (v2 NONLINEAR, ~15 lines). Dropped. The system contracts (checkpoint liberally; regression_warning binding) cover the functional concerns without prescribing failure-mode categories.
- **Pre-export read** (v2 NONLINEAR, ~15 lines). Dropped. The final HITL gate at NONLINEAR → EXPORT handles user approval; the agent's own pre-advance judgment handles metric regressions.
- **LINEAR regression-handling graft** (v2 §3b). The regression_warning contract in SYSTEM_BASE covers this uniformly; no phase-specific graft needed.
- **LINEAR classification taxonomy graft** (v2 §3a, added in this session). Migrates to the `remove_gradient` docstring where the agent sees it when choosing how to handle background non-uniformity.

---

## 4. What v3 keeps

- The tool catalog. System knowledge — the agent cannot see tool scopes otherwise.
- The Operating Mode loop naming. System knowledge — the JPG-per-tool preview is specific to muphrid's file layout.
- Every system contract: phase gate, rewind, checkpoint, regression_warning, HITL acknowledgment, flag_dataset_issue autonomous exception, memory_search existence.
- Phase-entry idempotency checks. Without these, re-entered phases waste tool calls re-building artifacts that already exist.
- The "one tool for all three master types" clarification in CALIBRATION. Observed failure mode: agents inventing `build_bias`, `build_dark`, `build_flat` tools.
- The `build_masters` idempotency clarification. Observed failure mode: retry loops after `quality_issues`.
- The linear-vs-nonlinear crossing note in STRETCH. System contract: variants operate on the same linear master, do not chain. Important for agent to know that creating 4 variants is safe.
- The masked-op prompt hook in NONLINEAR. Primes the agent to reach for `masked_process` rather than escalating global parameters — one of the higher-leverage system-level framings that isn't obviously implied by tool docstrings alone.

---

## 5. What leaves the prompt and where it lands

v2 proposed migrating a long list of "doctrine" from the system prompt into tool docstrings. Most of that list is wrong: it's domain knowledge the model already has from training, and adding it to docstrings just relocates the bloat from one file to another. The correct disposition is three-way:

**Drops entirely (domain knowledge the model has from training):**

- Target-type priors (emission nebula / galaxy / globular / broadband / OSC / mono typical-approach guidance).
- HDR two-stretch + mask walkthrough — when HDR earns its keep, how to blend, what mask range to pick. The model knows HDR compositing.
- Hue-targeted saturation mappings (Hα≈0, OIII≈3, SII≈5–6). The model knows the spectral-to-HSV mapping for OSC duoband.
- Star-handling doctrine (when `star_removal` helps, when to reach for `reduce_stars`, typical restoration weights).
- Masked-op doctrine ("sharpening adds noise to background, saturation washes star cores").
- Stretch variant reasoning ("too dark → increase D; bloated → lower HP"). The model knows GHS.
- Frame-selection thresholds by N frames, rejection-method scaling with N, findstar tuning for difficult data.
- Dark-frame exposure-match requirement, flat-quality checks, typical noise-reduction strengths by SNR.
- Linear-to-nonlinear boundary physics. The model knows why deconvolution doesn't belong after a stretch.
- "Three to five saturation passes is normal" / any pass-count heuristic.

None of this goes to docstrings. None of it goes anywhere. It leaves the repo.

**Lands in docstrings (tool-API knowledge the model can't derive):**

- Parameter names, types, ranges, and defaults. What each parameter mechanically controls. Pydantic `Field` descriptions carry this already for most tools; the audit pass (#12) covers gaps.
- Tool-specific version quirks where the same operation behaves differently across versions (e.g. GraXpert BGE ai_version=1.0.0 vs 2.0.0 smoothing behavior). These are about *this tool implementation*, not general domain.
- System-interaction constraints that are tool-specific, not general doctrine. Examples: `reduce_stars` produces best results on a star-removed input (tool constraint — how this tool interacts with pipeline state). `build_masters` is idempotent on identical parameters (tool contract). `stretch_image` variants do not chain (tool behavior; this is borderline between tool-API and system and is worth stating in both).
- Required preconditions the model couldn't otherwise infer from the tool's parameter names. Kept terse.
- The tool's own failure modes and what triggers a `regression_warning` from that specific tool.

**Stays in the prompt (system contracts):**

- The full list is in §1 above. System contracts are cross-tool; they belong in the prompt, not in every docstring that interacts with them.

**Test for any line of docstring content.** Ask: *is this about what this tool does and how its parameters work, or is it about when in my processing I should reach for this tool?* The first goes in the docstring. The second goes in the model's weights — it's the agent's reasoning job, not the docstring's.

**Concrete bad-doctrine examples to avoid in docstrings:**

- "Reach for this tool when stars still look bloated after stretching." → Agent reasoning. Drop.
- "Emission nebulae reward multiple hue-targeted passes." → Domain. Drop.
- "For galaxies, protect the nucleus." → Domain. Drop.
- "Use HDR compositing when a single stretch cannot reveal both core and outskirts." → Domain. Drop.

**Concrete good docstring content:**

- "`kernel_radius` (int, default=1, range 1–5): morphological footprint size in pixels. Higher values reduce star radii more aggressively."
- "`star_weight` (float, default=1.0, range 0–1): linear blend factor. 1.0 restores stars at original intensity; values below 1.0 reduce star prominence."
- "Input requirement: expects a star-removed image (produced by `star_removal`); running on a stars-present image will erode nebular structure."
- "Tool-version note: `ai_version` defaults to GraXpert's currently-installed model. Pin to a specific version (e.g. '1.0.0') for reproducibility."

The `#12` docstring-audit task covers this sweep — it's an audit, not a doctrine-import.

---

## 6. Effort and rollback

Prompt replacement: single-file string swap in `muphrid/graph/prompts.py`. Delta: ~430 lines removed, ~130 lines added. Independently revertable.

Docstring audit (separate, task #12): sweep existing tool docstrings to ensure they describe tool API clearly — parameter semantics, ranges, defaults, tool-specific constraints, version quirks — without embedding domain doctrine. Where docstrings today already carry "when to reach for this tool" language, strip it. Much of this work is subtractive, not additive. v3 prompts do not depend on this audit to be functional — the prompt is safe to land first.

---

## 7. Validation

Same framework as v2: re-run M20 and M42 with Sonnet 4.6 and with Kimi K2.5. Compare against the April 17 / April 19 baselines. Specific signals:

- Does the minimal prompt change the rate of reasoning-in-text before consequential choices? Hypothesis: steady or up, because the prompt no longer explicitly discourages text.
- Does the agent reach for `masked_process` / `hdr_composite` / `flag_dataset_issue` at the right moments? Hypothesis: yes, provided the docstrings carry the migrated doctrine. If not, the fault is in the docstring, not the prompt.
- Does the rate of phase-advance regressions drop? (With the regression_warning contract now a SYSTEM_BASE item rather than a LINEAR graft, it should fire more reliably across phases.)
- Does Kimi K2.5 hold up without scaffolding? Hypothesis: yes, because the earlier M20 Kimi failures were system-level (checkpoint file-copy bug, missing masked_process, missing polynomial fallback), not prompt-level. The v2 framework fixes and v2 tool-gap additions remove those system-level failure modes.

If any of these fail and the cause is traceable to missing prompt guidance, add it back narrowly — not a paragraph of doctrine, a single system-contract sentence.

---

## 8. The principle

The agent's intelligence lives in three places:

- **The model's weights.** Domain knowledge — what good astrophotography looks like, what each processing step is for, what usually helps which kind of target, how parameters typically trade off. The model brings this. The repo does not restate it.
- **The tool surface.** The operations available and the parameters that control them. Good tools expose the right levers with clear names and principled parameter ranges. A well-scoped tool with a rich parameter surface lets the model pull levers based on what the data shows; a narrow tool with few parameters forces the prompt to compensate.
- **The prompt.** System contracts specific to this codebase — phase semantics, tool catalog, checkpoint mechanics, regression contract, HITL norms, escape hatches. Things the model cannot derive from training because they're design decisions in this repo.

The failure mode v2 slipped into: using the prompt to compensate for either underspecified tools or underutilized training. v3 trusts the model's training, invests in the tool surface (v2 tool-gap additions §1–13 are still the work), and keeps the prompt focused on what only the prompt can say. When the agent underperforms, fix the tool — add a parameter, sharpen the parameter description, fill a capability gap. Don't add a paragraph to the prompt.
