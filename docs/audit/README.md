# Muphrid audit — index and corrections

_April 19, 2026_

This folder contains a two-round audit of the Muphrid astrophotography agent. The question driving the audit: _can these open-source tools, in this agentic framing, produce PixInsight-adjacent output across the full range of deep-sky targets — not just in the hands of the current model, but in the hands of any capable agent operating in this environment?_

Round 1 (April 18) produced `AUDIT_2026-04-18.md` and four patch stubs, grounded in the Kimi K2.5 M20 run (`run-m20-20260417-165609`).

Round 2 (April 19) reworks three of those four stubs plus the prompt rewrite, after (a) a second M20 run with Sonnet 4.6 (`run-m20-20260419-005415`), (b) clearer direction from the human operator that the audit should be model-agnostic and should treat VLM + measurement as the central advantage, and (c) closer inspection that revealed two items originally flagged as tool gaps were already exposed.

## Start here

| If you want | Read |
|---|---|
| The headline answer and the one-page evidence walkthrough | `AUDIT_2026-04-18.md` §§1–3 |
| How the framework itself misbehaves and the fixes for it | `v2_framework_fixes.md` |
| What tools are missing, what's underused, and the patch sketches | `v2_tool_gaps.md` |
| The proposed new prompts (model-agnostic, de-recipe) | `v2_prompts.md` |
| Per-tool docstring revisions to close "when and why" gaps | `v2_docstrings.md` |

The remainder of this README captures what has changed since the v1 write-up: retractions, retired documents, and the updated ordering for implementation.

## Corrections and retractions

Three claims from the April 18 audit have been retracted. Two are factual corrections (misread of existing tool surface); the third is a reframing based on new evidence.

### 1. Drizzle integration is not missing

`AUDIT_2026-04-18.md` §3 flagged drizzle as a gap, claiming `siril_stack` needed a `-drizzle` flag. This was wrong. In Siril, drizzle happens at the registration stage via `seqapplyreg`, not at stacking. The correct integration path is already exposed in `muphrid/tools/preprocess/t04_register.py` (lines 332–355): `drizzle`, `drizzle_pixfrac`, `drizzle_kernel`, `drizzle_flat` are all present. The remaining gap is doctrinal — nothing in the REGISTRATION phase prompt tells the agent when drizzle is warranted. That is a prompt fix, not a tool fix. Covered in `v2_tool_gaps.md` corrections section.

### 2. `reduce_stars` is fully featured

`AUDIT_2026-04-18.md` §3 said `reduce_stars` "might be underexposed." A closer read of `muphrid/tools/scikit/t26_reduce_stars.py` shows it exposes disk / square / diamond footprints, variable kernel radius, iteration count, and mask-aware blending. It is not underexposed; it is under-called. That is a docstring + prompt problem (Fix 1 of `v2_docstrings.md`), not a tool gap.

### 3. "A stronger model would close ~15% of the gap"

`AUDIT_2026-04-18.md` §8 estimated that a stronger model on the same system would close roughly 15% of the remaining distance to pro output — i.e., that model size was a small-but-nonzero lever.

**This estimate is withdrawn.** The April 19 M20 run with Sonnet 4.6 (`run-m20-20260419-005415`) produced a substantially _worse_ result than the Kimi K2.5 run on the same data: Sonnet hit the identical checkpoint no-op loop, then got stuck across multiple HITL rounds without making forward progress, and produced a final image that the operator characterized as trash. Whatever the true cross-model picture is, the claim that "bigger model → marginally better" is not supported by the evidence; if anything, the current system's friction costs a stronger model more than a weaker one because the stronger model tries harder to engage with the friction instead of pattern-matching through it.

The reframed conclusion: **friction in the framework, tool surface, and doctrine is the binding constraint, and the constraint does not relax with a stronger model under the current design.** That is why the v2 memos prioritize framework fixes (`v2_framework_fixes.md`) and prompt/docstring rewrites (`v2_prompts.md`, `v2_docstrings.md`) as the leverage points, not model upgrades.

## Retired documents

These v1 artifacts are superseded by v2 memos. They remain in the folder for historical reference; when the v2 patches land, they can be deleted.

| v1 file | Superseded by | Why |
|---|---|---|
| `t31_checkpoint_patch.md` | `v2_framework_fixes.md` | The v1 patch added `explicit_path` to `save_checkpoint`, which makes the agent reason about file paths. The correct fix is upstream: tools that write sibling files (`star_removal`, others) must update `current_image` so the agent never needs to know which file is current. |
| `prompts_nonlinear_rewrite.md` | `v2_prompts.md` | The v1 rewrite embedded target-type playbooks that read as recipes, and included pass-count floors calibrated to a specific model's behavior. v2 replaces the playbooks with framings, drops model-specific compensations, centers the VLM + analyze_image symbiosis, and adds an HITL conversational cue. |

Two v1 artifacts remain current and are not superseded:

- `t09_gradient_patch.md` — still the right design for the polynomial fallback. v2 references it.
- `t34_masked_process.py.stub` — still the right design for the masked-process wrapper. v2 references it.

## Scope of the v2 memos

v2 covers everything flagged under the original audit's "Adequate but limited (workarounds exist)" and "Genuine missing capabilities" sections, plus the framework-level issues surfaced by the Sonnet run, plus the prompt and docstring changes that are at least as important as any tool fix.

**Framework fixes (`v2_framework_fixes.md`):**

1. `star_removal` must update `current_image` to the starless file (root cause of the checkpoint loop that trapped both Kimi and Sonnet).
2. Tool post-condition contract — explicit convention that image-modifying tools update `current_image`.
3. `restore_checkpoint` returns a `noop` flag so the agent can detect the loop.
4. Stuck-loop detector for N consecutive no-op restores.
5. `commit_variant` race fix (graceful handling when `promote_variant` has already cleared the pool).
6. `rewind_phase` tool to complement the forward-only `advance_phase`.
7. Phase gate adjustments — `create_mask` and `analyze_frames` move to utilities.
8. Removal of the "no narration" pressure; the silent-tool backstop in `nodes.py` handles the real failure mode without adversarial framing.
9. HITL conversational cue.
10. Regression gates on `advance_phase`.

**Tool-gap patches (`v2_tool_gaps.md`):**

1. GraXpert knob exposure (`ai_version`, `gpu`, `batch_size`).
2. Polynomial gradient fallback.
3. Wavelet / bilateral denoise fallback.
4. Arbitrary-points tone curves with PCHIP spline.
5. HSV-space tone curves (new `t33_hsv_adjust.py`).
6. Hue-range saturation (absorbed into HSV tool).
7. Automated HDR composite (new `t35_hdr_composite.py`).
8. Per-channel stretch extension.
9. Masked process wrapper (re-affirms the `t34` stub).
10. Prompt / docstring fixes for already-exposed tools (`reduce_stars`, drizzle, `multiscale_process`, `create_mask`).

**Prompt rewrite (`v2_prompts.md`):**

- `SYSTEM_BASE` — replace Operating Philosophy / Autonomous Operation / Quality Standard / Target-Type Strategies with a block centered on the VLM + analyze_image loop, keeping agency primary and removing model-specific compensations.
- `NONLINEAR` — replace with a framing-based version; delete target-type playbooks.
- `LINEAR` — add a regression-handling paragraph.
- `STRETCH` — expand HDR compositing section with analyze_image trigger signals.
- Long-Term Memory — tighten to be informational rather than directive when memory is sparse.

**Docstring audit (`v2_docstrings.md`):**

Eight targeted fixes, organized by leverage. The highest-leverage fixes are `reduce_stars` (absent trigger language), `local_contrast_enhance` (no redirect to `multiscale_process` for structural work), and `saturation_adjust` (no iteration doctrine). Remaining fixes refine selection logic for `curves_adjust`, `star_restoration`, `deconvolution`, `multiscale_process`, and `save_checkpoint`/`restore_checkpoint`.

## Suggested rollout

The memos are independent and can land in any order. Recommended sequence to minimize thrash:

1. **Framework fixes first** (`v2_framework_fixes.md` items 1–5). These are small, mostly one-line, and unblock everything downstream. Without them, prompt changes run into tool bugs and cannot be fairly evaluated.
2. **Prompt and docstring rewrite** (`v2_prompts.md` + `v2_docstrings.md`). Free to try; rolls back instantly.
3. **Tool-gap patches**, in inverse-effort order: `v2_tool_gaps.md §9` (masked-process wrapper) → `§2` (polynomial fallback) → `§10` (docstring fixes for already-exposed tools) → `§4` (points curves) → others.
4. **Re-run M20** on Kimi K2.5 and Sonnet 4.6, both, on the same data. The comparison against the April 17/19 baselines is the audit's validation.

## What this audit is not

- Not a runnable plan. The memos describe designs; real PRs require reproduction tests, benchmarking, and code review.
- Not exhaustive. The tool surface is larger than what the M20 runs exercised; edge cases on mosaics, multi-panel narrowband, solar-system targets, and very short integration sets are not covered.
- Not a model comparison. The Sonnet run told us the current friction hurts Sonnet; it did not tell us how Sonnet behaves in a cleaner system. Re-running after v2 patches land is the right test.

## Verification

All v2 memos were cross-read against the code at commit-time on April 19, 2026. Specific line-number references have been double-checked. Two places where the code has moved since the original audit (drizzle in `t04_register.py`, full feature set of `t26_reduce_stars.py`) have been corrected above. If the code has moved further since this audit, the memo structure still holds — only the specific patch line-counts change.
