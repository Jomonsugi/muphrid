# Docstring audit (v2)

_April 19, 2026 · a sweep across the ~25 tool docstrings with a narrow focus: does each docstring tell the agent **when** to reach for this tool and **when not to**, or does it only explain **what** the tool does and **what the parameters control**?_

## Framing

Tool docstrings in this system do double duty. They are the pydantic `Field(description=…)` strings that become the schema shown to the agent _and_ the tool-function docstring attached to the `@tool` decorator. Together they form the vocabulary for every agent decision. The prompts in `prompts.py` are the grammar; the docstrings are the dictionary.

Today, most tool docstrings describe mechanics well — what each parameter does, what values are reasonable, what the output payload contains. The gap is situational: the agent reading the docstring can tell _how_ to call the tool but often can't tell _whether_ to call it at all given the state of the current image. That gap is what this audit closes.

Scope: twelve tools that are directly implicated in the M20 Kimi and Sonnet runs or that carry the most weight in the nonlinear phase. Not exhaustive — the earliest-phase tools (`build_masters`, `convert_sequence`, `calibrate`, `siril_register`, `siril_stack`) are already well-docstringed because their triggers are deterministic (file-type, sequence-state) rather than data-dependent.

## Summary table

| Tool | When/why quality | Fix needed |
|---|---|---|
| `remove_gradient` (t09) | strong | nothing structural |
| `deconvolution` (t13) | partial | SNR gate + skip-condition |
| `stretch_image` (t14) | strong | nothing structural |
| `star_removal` (t15) | strong | nothing structural |
| `curves_adjust` (t16) | partial | ght vs mtf selection + channel use |
| `local_contrast_enhance` (t17) | partial | redirect to `multiscale_process` for structural work |
| `saturation_adjust` (t18) | partial | iteration doctrine + hue-targeting trigger |
| `star_restoration` (t19) | partial | blend vs synthstar trigger + star_weight guidance |
| `create_mask` (t25) | strong | nothing structural |
| `reduce_stars` (t26) | **absent** | when vs. `star_restoration(star_weight<1.0)` |
| `multiscale_process` (t27) | strong | minor — cross-link to mask+starless combination |
| `save_checkpoint` / `restore_checkpoint` (t31) | strong — but hinges on framework fix | adjust phrasing after framework fix from `v2_framework_fixes.md` lands |

Seven tools are already good. Four are partial. One is absent. The gaps cluster in the _conditional_ tools (should I call this or skip it?) — exactly the class of decisions the M20 runs got wrong.

---

## Fix 1 — `reduce_stars` (t26_reduce_stars.py)

**Current state.** The tool exposes morphological star reduction (erosion with disk/square/diamond footprints, variable kernel radius, iterations, mask-aware blending). The docstring describes the mechanics well. It does not describe _when_ to call this tool vs. achieving similar effect via `star_restoration(star_weight<1.0)`, and it does not describe when the tool is the right answer vs. when masked processing of the nebulosity is.

**Why this matters.** In the M20 runs (both Kimi and Sonnet), star_removal did its job but the restored image had stars that were still large and bright enough to crowd the nebulosity. Neither agent called `reduce_stars` — because nothing in the tool's surface tells them when "restored stars still feel too big" is the trigger. Similarly, for globular clusters, reduce_stars is the primary core-decongestion tool and the agent has no cue that's what it's for.

**Proposed additions.** Append to the tool docstring:

```
Reach for reduce_stars when the image *after star_restoration* still has
stars that compete visually with the nebulosity or that cluster into
luminous bloat in dense regions (common in M-class emission nebulae with
rich foreground star fields, and at the cores of globular / open clusters).

The tool shrinks stars. It does not dim them. That makes it complementary
to star_restoration(star_weight < 1.0), which dims stars without shrinking
them. Typical composition:
  - Stars are *too bright* relative to nebulosity → lower star_weight in
    star_restoration.
  - Stars are *too large* for the composition → reduce_stars.
  - Both → do star_restoration first at reduced weight, then reduce_stars.

For globular-cluster cores where star_removal is not run, call reduce_stars
directly on the post-stretch image to decongest the core before saturation
and curves.

Skip reduce_stars on sparse star fields — the morphological erosion will
round off already-small stars and create an obviously processed look.
```

Effort: 20 lines of docstring. No code changes.

---

## Fix 2 — `local_contrast_enhance` (t17_local_contrast.py)

**Current state.** The docstring explains LCE mechanics and has a correct "do not use on star fields / globular clusters" note. What it does _not_ say is that `multiscale_process` is the primary structural-detail tool and that LCE is the right call for _global_ crispness, not for scale-targeted sharpening.

**Why this matters.** In the M20 Kimi run, the agent reached for LCE three times and applied it to a starfull image (downstream of the checkpoint bug). It never called `multiscale_process`. An agent reading the current docstrings sees two sharpening tools with overlapping purpose and no guidance on which to pick — so it picks whichever is closer to its first search.

**Proposed additions.** Insert into the tool docstring, before the "Do NOT use" line:

```
local_contrast_enhance is a *single-scale* contrast boost — it sharpens the
image as a whole. For *structural* sharpening where you want to enhance
nebula filaments or galaxy arms at specific spatial scales while leaving
the noise floor alone, use multiscale_process (with a luminance mask) —
it decomposes the image into wavelet scales and lets you sharpen the
scales that carry real structure while leaving the scales that carry
noise untouched or denoised.

Use local_contrast_enhance for the final overall crispness pass *after*
multiscale_process, or for simple targets (broadband dust, galaxy
continuum) where scale-targeted sharpening is overkill.
```

Effort: 10 lines. No code changes.

---

## Fix 3 — `saturation_adjust` (t18_saturation.py)

**Current state.** Distinguishes `hue_targeted` vs `linear` vs `ght_sat`; gives hue-target hints (Hα ≈ 0, OIII ≈ 3). Does not say anything about iteration — how many passes, when one pass is enough, when to stop.

**Why this matters.** The M20 Kimi agent ran two saturation passes, one targeted and one global, and stopped. The picture was undersaturated by nearly any standard, but nothing in the docstring (or, at the time of that run, the prompt) told the agent that three-to-five hue-targeted passes is normal for an emission-line target. The v2 prompt rewrite intentionally avoids prescribing pass counts — but the docstring can still point the agent toward iterative use without fixing a number.

**Proposed additions.** Append to the tool docstring:

```
Saturation typically requires multiple passes on emission-line targets.
Each pass targets one color band (Hα, OIII, SII, broadband continuum)
at moderate amount (0.3–0.6). After each pass, look at the image and
run analyze_image — check color_balance, mean_saturation, and the per-
channel histograms. Stop when:
  - A further pass pushes one channel into clipping or creates blotchy
    posterization.
  - mean_saturation plateaus and further amount does not change it
    (diminishing returns — the data is saturated).
  - The picture reads as over-colored (subjective; compare to the
    target-type expectation in SYSTEM_BASE).

One global pass (method='ght_sat' with no hue target) is a fine starting
move for broadband targets that do not have narrow emission lines. For
emission nebulae, skip it — hue-targeted passes give more control.

Skipping saturation entirely is rarely correct on a color dataset. If
mean_saturation after stretch is below ~0.10, at least one pass is
warranted.
```

Effort: 22 lines.

---

## Fix 4 — `star_restoration` (t19_star_restoration.py)

**Current state.** Describes blend vs. synthstar modes and the star_weight mechanic. Doesn't describe when to pick synthstar over blend, or what star_weight value to start at.

**Why this matters.** Synthstar is a substantial intervention — it replaces the original stars with synthetic profiles. The right trigger is "the original stars have optical defects that processing cannot recover" (coma, trailing, severe diffraction spikes from a cheap scope, or oversized stars from poor seeing where morphology can't save them). An agent with no guidance on that trigger will default to whichever mode comes first in the schema.

**Proposed additions.** Append to the tool docstring:

```
Mode selection:
  - blend (default): blends the original star layer (saved at star_removal
    time) back onto the processed starless image. Preserves star color and
    morphology exactly. This is the right mode for the vast majority of
    datasets — stars that were fine at the start remain fine at the end.
  - synthstar: replaces original stars with synthetic profiles generated
    from detected-star photometry. Use only when the original stars have
    optical defects that processing cannot fix — coma, trailing, severe
    diffraction spikes, or asymmetric halos. Synthstar eliminates those
    defects at the cost of losing the natural color and profile variation
    of the real star field.

star_weight guidance (blend mode):
  - 1.0: full restoration (default). Stars at original intensity.
  - 0.7–0.9: dimmed. Reasonable for emission-nebula and broadband
    compositions where the nebulosity should be the subject — most
    common choice.
  - 0.5–0.7: heavily dimmed. Use for extreme nebulosity-first
    compositions, or in combination with reduce_stars when the field is
    dense.
  - Below 0.5: stars become ghostly; combine with reduce_stars for
    aesthetic control.
```

Effort: 25 lines.

---

## Fix 5 — `curves_adjust` (t16_curves.py)

**Current state.** Distinguishes `mtf` (midtone transfer), `ght` (generalized hyperbolic tone), and other methods. Explains parameters. Does not say when to pick which, when to use channel-specific vs. luminance curves, or what "points-based" curves (once they land per `v2_tool_gaps.md §4`) are for.

**Why this matters.** The M20 Kimi agent used `mtf` once and skipped curves for the rest of the nonlinear phase. Curves is a general-purpose shaping tool, not a one-shot global-contrast bump. The docstring should tell the agent what each method is good for and when to reach for a targeted (masked or channel-specific) curves call vs. a global one.

**Proposed additions.** Append to the tool docstring:

```
Method selection:
  - mtf: simple midtone transfer. Fast, predictable, good for overall
    brightness rebalancing after stretch. Symmetric around the midtone
    pivot — the whole histogram shifts together.
  - ght: generalized hyperbolic tone curve. Adds an S-curve with
    parameters to focus the contrast lift on a specific tonal range
    (D, SP, B, HP, LP — same semantics as stretch_image). Reach for ght
    when you want contrast in a specific brightness band without
    affecting the rest (for example, lifting faint-shell contrast without
    touching the core).
  - points (once available): arbitrary control points. Use when neither
    mtf nor ght fits what the data needs — for example, a targeted
    highlight pulldown combined with a shadow lift.

Channel selection:
  - Global (default): applies to all channels equally. Safe for tonal
    shaping.
  - Per-channel (r, g, b): compensates for residual color imbalances
    after color_calibrate and SCNR. Reach for it when analyze_image
    shows asymmetric per_channel histograms post-stretch.
  - Luminance (l or l_star, once available): shapes tone without
    affecting hue/saturation. Useful before a subsequent saturation
    pass, or when the picture's color is already right but the
    brightness distribution needs work.

Masked curves (via masked_process or create_mask + pixel_math) is
common. A masked curves call lets you lift faint regions while protecting
the core, or vice versa — reach for it whenever a global curves call
would damage one part of the image while helping another.

Multiple passes are normal. Checkpoint between passes so you can iterate
without cumulative drift.
```

Effort: 35 lines. Assumes the curves patch from `v2_tool_gaps.md §4` lands; otherwise drop the `points`, `l_star` references.

---

## Fix 6 — `deconvolution` (t13_deconvolution.py)

**Current state.** Good gatekeeping on linear-space only. Says "only attempt when snr_estimate > 50." Does not distinguish richardson_lucy vs. wiener, does not tell the agent when to skip deconvolution entirely even with good SNR, and does not mention the fallback path (if deconvolution is skipped, use multiscale_process in NONLINEAR as the perceptual equivalent).

**Proposed additions.** Append to the tool docstring:

```
When to skip:
  - snr_estimate < 50 from analyze_image. Deconvolution amplifies noise;
    below this SNR the amplification dominates the sharpening gain.
  - Measured FWHM is already at or below ~1.5 px (seeing-limited or
    diffraction-limited data). There is no PSF-blur headroom to recover.
  - The dataset is intended for broad-structure aesthetic processing
    (diffuse dust, smooth galaxy disks) — fine-detail sharpening does
    not add to what the target needs.

If you skip deconvolution here and later realize more detail would help,
multiscale_process with sharpen weights on wavelet scales 1–2 behind a
luminance mask is the non-linear equivalent. It cannot replicate PSF-
based sharpening exactly, but it achieves similar perceptual detail
enhancement without the noise-amplification risk.

Method selection:
  - richardson_lucy (default): iterative, forgiving of noise at moderate
    iteration counts (5–15). The usual choice.
  - wiener: single-step, aggressive. Produces sharper results but
    amplifies noise more strongly. Reserve for very high SNR datasets
    (snr_estimate > 80) where you want crispness over smoothness.
```

Effort: 25 lines.

---

## Fix 7 — `multiscale_process` (t27_multiscale.py)

**Current state.** Strong — covers phase context, mask usage, cross-references star_removal and create_mask. One small gap: it does not explicitly say that the masked+starless combination is the canonical pattern for nebula structural sharpening. An agent reading it today sees the pattern as _possible_, not as the default.

**Proposed addition.** One paragraph inserted into the tool docstring:

```
Canonical pattern for nebula structural sharpening:
    star_removal (earlier in NONLINEAR) → save_checkpoint →
    create_mask(mask_type='luminance', range_low=0.10–0.20) →
    masked_process(tool_name='multiscale_process', ...) or
    multiscale_process(use_latest_mask=True, ...)

This applies the sharpening only to the nebulosity (above the mask
threshold), leaving the background untouched and working on the starless
image so stars cannot halo. It is the standard combination for
emission-nebula, galaxy-arm, and dust-structure enhancement. Use scales
2 and 3 for typical field-of-view; move to 3 and 4 for broader structure
(large dust fields).
```

Effort: 12 lines.

---

## Fix 8 — `save_checkpoint` / `restore_checkpoint` (t31_checkpoint.py)

**Current state.** Already explains when to checkpoint well ("after star_removal produces a clean starless image; after a curves/saturation/contrast pass you are satisfied with; before any experimental adjustment"). The weakness is not the docstring — it is that the tool does not behave as the docstring promises, because upstream tools don't update `current_image`. That is a _framework_ bug (covered in `v2_framework_fixes.md`), not a docstring bug.

**Proposed change.** After the framework fix lands, remove the existing "resolved path to verify" language if present (it becomes dead weight — the agent no longer needs to verify). Keep the "use them liberally — they are cheap" framing. Add one sentence to `restore_checkpoint`:

```
If restore_checkpoint returns noop=true, the restore was effectively a
no-op — current_image already matched the bookmarked path. This usually
means the checkpoint was stale at save time; create a fresh checkpoint
from the current state before retrying the branch you were attempting.
```

(This aligns with the `noop` flag introduced in `v2_framework_fixes.md` — Issue #1, restore-response change.)

Effort: 5 lines. Depends on framework fix.

---

## Not changing (for the record)

- `remove_gradient` (t09) — already has strong when/why, especially after `v2_tool_gaps.md §1 + §2` add polynomial fallback and GraXpert knob exposure. Those patches modify Field descriptions directly; no additional docstring pass needed.
- `stretch_image` (t14) — variant independence, phase-boundary language, and parameter geometry are already well-explained. The per-channel stretch patch from `v2_tool_gaps.md §8` adds its own Field description; no broader docstring edits needed.
- `star_removal` (t15) — phase constraint and upscale logic are clear. The framework fix (updating current_image to the starless file) is invisible to the docstring; no change needed there.
- `create_mask` (t25) — three-step pattern and mask-type selector are already named as the prerequisite for targeted processing. No change.
- Early-phase tools (`build_masters`, `convert_sequence`, `calibrate`, `siril_register`, `siril_stack`, `auto_crop`, `select_frames`, `analyze_frames`) — triggers are deterministic (file-type, sequence-state); their existing docstrings describe mechanics correctly and the prompts handle sequencing. No change.
- Utility tools (`analyze_image`, `plate_solve`, `pixel_math`, `extract_narrowband`, `resolve_target`, `advance_phase`, `memory_search`, `pixel_math`) — utility semantics; their usage triggers are in the prompts, not the docstrings. No change.

---

## Effort and rollout

All eight fixes are Field and docstring-string edits in nine files. Total delta: roughly 160 lines of docstring text added across the codebase. No schema changes. Entirely backward-compatible.

The fixes can land independently. Recommended order:

1. Fix 1 (`reduce_stars`) and Fix 2 (`local_contrast_enhance`) — the two cases where the agent in M20 did the wrong thing because the docstring did not steer. Highest leverage per line.
2. Fix 3 (`saturation_adjust`) and Fix 5 (`curves_adjust`) — iteration and selection doctrine for the two most-used nonlinear shaping tools.
3. Fix 4 (`star_restoration`), Fix 6 (`deconvolution`), Fix 7 (`multiscale_process`) — refinements for correct selection in less-common cases.
4. Fix 8 (`save_checkpoint` / `restore_checkpoint`) — after `v2_framework_fixes.md` lands.

Validation: the docstring pass is below the threshold of "needs a run to verify." Code review by anyone who has seen a M20-level run is sufficient. If the prompt rewrite (`v2_prompts.md`) lands at the same time, the docstrings become the authoritative source of "when to reach for this tool" — and should be tested during the M20 re-run described in that memo's §6.
