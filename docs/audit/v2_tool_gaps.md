# Tool-gap addendum (v2)

_April 19, 2026 · patch sketches for every item under "Adequate but limited" and "Genuine missing capabilities" from the original audit, plus corrections where a gap turned out to already be exposed._

## Corrections to the original audit

Before the new patches, two items from the original audit's missing-capability list need to be withdrawn:

- **Drizzle upscaling** is _already exposed_ in `t04_register.py` (`drizzle`, `drizzle_pixfrac`, `drizzle_kernel`, `drizzle_flat`). The original audit was wrong to say `siril_stack` needs a `-drizzle` flag — in Siril, drizzle happens at the registration stage via `seqapplyreg`, not in `stack`. Tool surface is fine. The gap that remains is in the _prompts_: nothing in the LINEAR/STACKING prompt tells the agent when drizzle is warranted (under-sampled data, FWHM < 2 px, plenty of well-dithered frames). Fix is a prompt sentence in REGISTRATION.
- **Morphologically-precise star reduction** is _already exposed_ in `t26_reduce_stars.py` with disk/square/diamond footprints, variable kernel radius, iteration count, and mask-aware blending. The original audit flagged it as "might be underexposed." It is not. The gap that remains is doctrine — in the M20 run the tool was never called. Fix is a sentence in the nonlinear prompt.
- **Masked gradient removal** (from the M42 branch-occlusion scoping) is _not_ a separate tool. Once the §9 `masked_process` wrapper lands, `masked_process(tool_name="remove_gradient", tool_params=..., mask=...)` covers the use case. No dedicated `masked_remove_gradient` surface needed.

Everything else in the original list holds. Patches below.

---

## 1. GraXpert knob exposure (`t09_gradient.py`)

**Status.** Muphrid exposes 3 of GraXpert's ~7 background-extraction parameters (`correction_type`, `smoothing`, `save_background_model`). The remaining levers can change how the AI model behaves on edge cases like M20.

**Add to `GraXpertBGEOptions`:**

```python
ai_version: str | None = Field(
    default=None,
    description=(
        "GraXpert BGE AI model version (e.g. '1.0.0', '1.1.0', '2.0.0'). "
        "Different versions have different structure-preservation vs. gradient-"
        "following trade-offs. Leave None to use GraXpert's default; pin an "
        "explicit version for reproducible runs or to test whether an older/"
        "newer model behaves better on a problematic target. "
        "Versions are cached under ~/Library/Application Support/GraXpert/bge-ai-models/."
    ),
)
gpu: bool = Field(
    default=True,
    description=(
        "Enable GPU acceleration for AI inference. Set False to force CPU "
        "when GPU memory is tight or when debugging numerical-precision issues."
    ),
)
batch_size: int = Field(
    default=4, ge=1, le=16,
    description=(
        "Inference batch size. Does not affect output, only runtime/memory. "
        "Reduce to 1–2 on OOM."
    ),
)
```

**Wire into the CLI invocation:** append `-ai_version`, `-gpu`, `-batch_size` when set. Current invocation skeleton already builds a list; add three conditional appends.

**Effort.** ~20 lines. No docstring changes needed beyond the schema descriptions.

**Why this matters.** The M20 over-fit failure was diagnosed as "smoothing is crude — it is the only AI knob." Adding `ai_version` gives the agent a second degree of freedom: when smoothing=0.8 still regresses, it can try a different model version before falling back to polynomial.

---

## 2. Polynomial gradient fallback (`t09_gradient.py`)

**Still recommended.** Full stub already written at `t09_gradient_patch.md`. That patch stands; keep it. In combination with #1 above, the full set of remove_gradient levers becomes:

```
method = "graxpert"  → (correction_type, smoothing, ai_version, gpu, batch_size, save_background_model)
method = "polynomial" → (degree, samples_per_line, tolerance)
method = "siril_auto" → (no options)
```

The `regression_warning` structured field (also in that patch) is the signal the prompt should treat as binding.

---

## 3. Noise-reduction fallback — wavelet denoise (`t12_noise_reduction.py`)

**Status.** The noise-reduction tool is GraXpert-denoising only. Same category of single-backend risk as gradient. When GraXpert denoise smears faint structure or produces a plastic look on low-SNR data (it does, on some targets), there is no alternative path.

**Patch.** Extend with `method: Literal["graxpert", "wavelet", "bilateral"]`.

- `wavelet` — Siril's `denoise` command with wavelet parameters (da3d or multiscale_median). Expose strength/layer-count knobs.
- `bilateral` — edge-preserving bilateral filter via scikit-image (`skimage.restoration.denoise_bilateral`). Useful when the noise is strictly Gaussian at the pixel level.

```python
class WaveletDenoiseOptions(BaseModel):
    num_layers: int = Field(default=4, ge=2, le=6)
    threshold_sigma: float = Field(default=3.0)
    algorithm: Literal["bspline", "median"] = "bspline"
```

**Effort.** ~80 lines. Prompt update: one sentence in LINEAR to name the alternative when GraXpert denoising is too aggressive.

---

## 4. Arbitrary-points tone curves (`t16_curves.py`)

**Status.** `curves_adjust` supports `mtf` (3 control points) and `ght` (parametric). No piecewise/spline curves with user-placed anchors per channel. This is the single biggest missing operation vs. PixInsight CurvesTransformation.

**Patch.** Add a `points` method that accepts a list of `(input, output)` pairs per channel and builds a monotone cubic spline through them.

```python
class PointsCurveOptions(BaseModel):
    points: list[tuple[float, float]] = Field(
        description=(
            "List of (input, output) pairs defining the curve. Both values "
            "in [0, 1]. Must include at least the endpoints (0,0) and (1,1) "
            "unless you want to clip. Points are sorted by input and fit with "
            "a PCHIP (monotone cubic) spline — avoids the oscillation that "
            "plain cubic splines produce on non-monotone anchor sets."
        ),
    )
    channel: Literal["rgb", "r", "g", "b", "l", "l_star"] = Field(
        default="rgb",
        description=(
            "Which channel the curve applies to. 'rgb' applies to all three "
            "identically (tonal). 'r'/'g'/'b' apply to one only (color grade). "
            "'l' = perceptual luminance (Y' from Rec.709). "
            "'l_star' = CIE L* (preserves chroma; best for deep mid-tone work)."
        ),
    )
```

Implementation: numpy + `scipy.interpolate.PchipInterpolator`. Load FITS, apply per-channel, save. No Siril needed.

**Effort.** ~140 lines. Covers the 80%-use case for "I need a custom curve on just one channel."

---

## 5. HSV-space tone curves (new tool `t33_hsv_adjust.py`)

**Status.** No HSV-space operations. Saturation_adjust's 7-hue-index quantization is coarse; fine-tuning hue rotation (e.g., "pull purples toward magenta without touching reds") or desaturating a narrow hue band is impossible.

**New tool.** `hsv_adjust` converts to HSV via `skimage.color.rgb2hsv`, applies separate curves to H / S / V, converts back.

```python
class HSVAdjustInput(BaseModel):
    h_curve: list[tuple[float, float]] | None = None   # hue shift curve (0..1 cyclic)
    s_curve: list[tuple[float, float]] | None = None   # saturation curve
    v_curve: list[tuple[float, float]] | None = None   # value/brightness curve
    hue_range: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Optional hue window in degrees [start, end] (supports wrap-around, "
            "e.g. (340, 20) for reds around the wrap point). Only pixels whose "
            "hue falls inside this window are affected; others pass through. "
            "Enables narrow-band hue work impossible with saturation_adjust's "
            "7-bucket quantization."
        ),
    )
    mask_path: str | None = Field(
        default=None,
        description="Optional luminance/range mask for masked HSV work.",
    )
```

Implementation: rgb2hsv → apply curves → optional hue-window gating → hsv2rgb. Handles the `s` and `v` curves as standard, and treats `h_curve` as a cyclic mapping (add the curve output to hue, then modulo 1).

**Effort.** ~150 lines. This is the single biggest unlock for color nuance — emission-nebula palette work, star-color rescue, reflection-nebula hue shifts — none of which are practical today.

---

## 6. Hue-range saturation (extend `t18_saturation.py` or fold into `hsv_adjust`)

**Status.** `saturation_adjust(method="hue_targeted")` uses Siril's fixed 7-bucket hue quantization (indices 0–6). Fine for "boost reds" but too coarse for "boost the blue-green OIII shell without touching the cyan dust edge."

**Two paths.**

- Quick win: Extend `saturation_adjust` to accept a `hue_range_deg: tuple[float, float] | None` that supersedes `hue_target` when provided, uses rgb→hsv internally, and applies a gain to pixels in that range only. ~40 lines.
- Clean win: Make `hsv_adjust` (#5) the canonical hue-range saturation tool and deprecate the coarse `hue_target` path. Keep `hue_target` for quick pros-know-it work.

The clean path is preferred. Less redundancy, one conceptual model.

---

## 7. Automated HDR composite (new tool `t35_hdr_composite.py`)

**Status.** HDR compositing requires four hand-assembled steps: stretch_variant_A → stretch_variant_B → create_mask → pixel_math blend. In the Sonnet run, the agent executed this sequence correctly. Both the assembly and the semantic intent are tractable for a strong model — but it is friction-heavy enough that one-shot failures (wrong blend expression, wrong mask range) are common.

**New tool.** `hdr_composite` wraps the pattern into a single declarative call.

```python
class HDRCompositeInput(BaseModel):
    base_stretch: GHSOptions = Field(
        description="Stretch parameters for the faint/shadow-optimized layer.",
    )
    core_stretch: GHSOptions = Field(
        description="Stretch parameters for the bright-core-protected layer.",
    )
    mask: MaskSpec | str = Field(
        description=(
            "Either a mask spec (range mask on the bright region) or a path to "
            "an existing mask. The mask selects where the core stretch dominates; "
            "outside the mask, the base stretch dominates. Defaults to auto-"
            "generation: range mask with low=percentile(98), high=1.0, feather=20."
        ),
    )
    blend_mode: Literal["linear", "luminosity", "masked_multiscale"] = Field(
        default="linear",
        description=(
            "linear: out = core * mask + base * (1 - mask). Classic MaskedStretch. "
            "luminosity: preserves base's color channels, takes core's luminance "
            "inside the mask only. Cleaner on galaxies. "
            "masked_multiscale: multiscale blend that separates structural and "
            "tonal scales. Best on emission nebulae with both bright cores and "
            "extended faint shells."
        ),
    )
    preview: bool = Field(default=True)
```

Implementation: runs `stretch_image` twice internally (both from the same linear master, to maintain variant independence), calls `create_mask` if mask is a spec, executes the blend via pixel_math or numpy (depending on blend_mode), writes the composite as the new `current_image`.

**Effort.** ~220 lines. Comes with a prompt bump: "When the target has extreme dynamic range (galaxy cores, emission nebula bright regions, bright star + faint nebula scenes), reach for `hdr_composite` rather than a single stretch."

This supersedes the P3 "per-channel stretch + masked stretch" as the higher-value combined version.

---

## 8. Per-channel stretch (extension of `stretch_image`)

**Status.** `stretch_image` applies the same GHS to R/G/B. Per-channel stretching is a standard lever for fixing residual color casts after weak color calibration (e.g., when PCC is marginal due to sparse astrometric matches).

**Patch.** Extend `GHSOptions` with optional per-channel overrides:

```python
per_channel_D: list[float] | None = Field(
    default=None,
    description=(
        "Optional per-channel stretch_amount as [R, G, B]. When provided, "
        "overrides the scalar stretch_amount for each channel. Useful to lift "
        "a weak blue channel without over-stretching red. Leave None for "
        "standard RGB-identical stretching."
    ),
)
per_channel_SP: list[float] | None = Field(default=None)
per_channel_HP: list[float] | None = Field(default=None)
```

The tool runs the stretch three times internally (once per channel) and recombines. Slower than one pass; used only when needed.

**Effort.** ~60 lines.

---

## 9. Masked processing wrapper (`t34_masked_process.py`)

**Status.** Stub already written at `t34_masked_process.py.stub`. Holds. This is P1 — the single biggest ergonomics unlock for the nonlinear phase.

One revision to the stub: with #7's `hdr_composite` and `create_mask` becoming a utility (per the framework memo), `masked_process` should delegate to `create_mask` when passed a `MaskSpec`, not reimplement it inline. Small cleanup.

---

## 10. Prompt + docstring notes for already-exposed tools that were under-used

These are not tool gaps; they are discoverability gaps.

- **`reduce_stars` (T26).** Already full-featured. Add to NONLINEAR prompt: "if stars still look bloated after star_restoration or without star removal, call reduce_stars with kernel_radius=1 and iterations=1 as a starting point."
- **`drizzle` in `siril_register`.** Already exposed. Add to REGISTRATION prompt: "consider `drizzle=True` with `drizzle_pixfrac=0.8` when the registration reports median FWHM < 2 px and the dataset has ≥30 well-dithered frames. Drizzle doubles output resolution and helps resolve under-sampled stars."
- **`multiscale_process` (T27).** Full-featured. Prompt needs to name it as the first-choice sharpening tool (see v2 prompt rewrite).
- **`create_mask` (T25).** Full-featured. Prompt needs to name it as the first-choice selectivity tool.

None of these need code changes; they need _doctrine_ reinforcement.

---

## 11. `flag_dataset_issue` — agent-initiated HITL interrupt (new tool `t36_flag_dataset_issue.py`)

**Status.** The agent has no way to surface "this dataset has a physical acquisition problem that no processing tool will fix." The only escape valves today are HITL phase gates (which fire only on phase transitions, and only in non-autonomous modes) or silent continuation on defective data. In the M42 Sonnet run, post-stack inspection revealed "25% quadrant variation" in background — the actual cause was out-of-focus tree branches in the bottom-right quadrant, invisible on individual subs but integrated into a subtle occlusion after stacking. The agent had no primitive to say "this is an occlusion, not a gradient — please decide."

**New tool.** `flag_dataset_issue` raises a LangGraph `interrupt` with a structured diagnostic payload. The interrupt fires **even in autonomous mode** — this is the documented exception to the autonomous-mode-no-HITL rule, because "the data is broken in a way no processing tool can fix" is a human decision, not an agent decision.

**Schema:**

```python
class FlagDatasetIssueInput(BaseModel):
    severity: Literal["advisory", "blocking"] = Field(
        description=(
            "advisory: a condition that may reduce the quality ceiling; "
            "processing can continue; the user is notified and may accept, "
            "reshoot, or intervene. "
            "blocking: the data cannot produce a usable result even with "
            "the best processing choices; work halts until the user confirms "
            "whether to proceed anyway or abort."
        ),
    )
    category: Literal[
        "acquisition_defect",         # out-of-focus, frost, tracking failure
        "optical_occlusion",          # branches, wires, clouds, tarps
        "calibration_mismatch",       # dark/flat temperature or ADU mismatch
        "integration_inadequate",     # too few frames, too short exposure
        "spatial_anomaly",            # asymmetric flat, tilt, coma field
        "other",
    ] = Field(
        description=(
            "Classification of the issue. This drives how the user response "
            "and any subsequent mitigation plan are interpreted."
        ),
    )
    summary: str = Field(
        description=(
            "One or two sentences. Describe the specific thing seen in this "
            "dataset — not a restatement of the category. For example: "
            "'Bottom-right quadrant shows a dark irregular region with a "
            "sharp boundary across ~15% of the frame, consistent with an "
            "out-of-focus optical occlusion rather than a gradient.'"
        ),
    )
    evidence: list[str] = Field(
        description=(
            "Paths (relative to run directory) to preview images, "
            "analyze_image JSON output, or other artifacts supporting the "
            "flag. At minimum include one JPG/PNG preview showing the issue "
            "and one metrics snapshot showing the numerical signature "
            "(quadrant imbalance, SNR per-quadrant, FWHM distribution). "
            "These are displayed to the user alongside the summary."
        ),
    )
    recommended_action: Literal[
        "proceed_with_ceiling",         # continue; lower expected quality
        "reshoot",                      # the dataset cannot be rescued
        "masked_mitigation",            # agent has a plan to work around
        "reject_frames_and_restack",    # subset of subs is bad; restack without them
        "user_decision_required",       # agent has no preferred path
    ] = Field(
        description=(
            "The agent's best-available recommendation. The user sees this "
            "alongside the evidence and decides. This is not a commitment — "
            "the user can accept, modify, or reject."
        ),
    )
    mitigation_plan: str | None = Field(
        default=None,
        description=(
            "If recommended_action in (masked_mitigation, "
            "reject_frames_and_restack), describe the specific plan: "
            "which frames or regions, which tools, expected outcome. "
            "Omit for other actions."
        ),
    )
```

**Behavior.** The tool writes a structured diagnostic JSON into `<run_dir>/dataset_issues/<timestamp>.json`, raises a LangGraph `interrupt` with the payload, and pauses the graph. Resume semantics follow the existing HITL pattern — the user provides a response string (free text) which arrives as the `Command(resume=...)` payload. The agent reads the response, acknowledges it conversationally (per the HITL-ack fix in `v2_framework_fixes.md`), and continues along the confirmed path.

**Autonomous-mode interaction.** One of the few HITL paths that fires regardless of autonomous mode. The reasoning: HITL phase gates are workflow-level approvals ("does this look right?"); `flag_dataset_issue` is an escape hatch for data problems outside the processing tools' scope. Agents running autonomously should still be able to halt for a decision that is not in their remit. The prompt should emphasize this is for genuine acquisition-level issues — not "I'm not sure which saturation strength to pick."

**Prompt hook.** Add to `SYSTEM_BASE`'s Dataset Adequacy block:

> If you observe a condition that looks like an acquisition defect (out-of-focus regions, occluding objects, tracking smear, frost, severe vignetting) rather than a processing problem, call `flag_dataset_issue` with severity appropriate to how fatal the issue is. Include preview imagery and metrics as evidence. This tool pauses the graph for user decision even in autonomous mode — use it sparingly, only when the issue is genuinely outside the scope of processing tools to fix.

And to the LINEAR phase prompt (per the classification-taxonomy graft in `v2_prompts.md §3`):

> When background non-uniformity reads as an occlusion rather than a gradient (hard edges, asymmetric darkness that does not follow a smooth surface), `flag_dataset_issue(category="optical_occlusion")` is the correct response rather than escalating `remove_gradient` smoothing.

**Effort.** ~180 lines. New tool file, schema, LangGraph interrupt/resume wiring reusing existing HITL plumbing in `graph/nodes.py`, run-directory JSON writer. No schema changes to `AgentState`.

**Why this matters.** The M42 case had the agent guessing at mitigation strategies (stronger gradient removal, higher smoothing) for what turned out to be an optical occlusion. Processing tools cannot undo out-of-focus branches on N frames; a human must decide whether to reshoot, accept reduced quality, or restack on the unaffected subset. This tool turns "silent acceptance of bad data" into a first-class diagnostic with structured evidence.

---

## 12. `create_mask` — spatial region parameter (extension of `t25_create_mask.py`)

**Status.** `create_mask` currently supports luminance range, star, and range masks built from image statistics. It has no primitive for "select this spatial region" — rectangle, polygon, or ellipse — which is the natural representation for "that quadrant has an issue" or "that satellite trail lives here."

**Patch.** Extend `CreateMaskInput` with a `region` parameter plus a combine rule:

```python
class RegionSpec(BaseModel):
    shape: Literal["rectangle", "polygon", "ellipse"] = Field(
        description=(
            "Geometric primitive. rectangle uses axis-aligned bounds. "
            "polygon uses explicit vertices (implicitly closed). "
            "ellipse uses center + semi-axes with optional rotation."
        ),
    )
    coords: list[float] | list[tuple[float, float]] = Field(
        description=(
            "rectangle: [x0, y0, x1, y1] in pixel coordinates (origin "
            "top-left). polygon: list of (x, y) vertices in pixel "
            "coordinates. ellipse: [cx, cy, rx, ry]."
        ),
    )
    rotation_deg: float = Field(
        default=0.0,
        description="Rotation for ellipse in degrees. Ignored for rectangle/polygon.",
    )
    feather_radius: int = Field(
        default=0,
        ge=0,
        description=(
            "Optional per-region feather applied before combining with the "
            "base mask_type. Combines additively with CreateMaskInput."
            "feather_radius."
        ),
    )

class CreateMaskInput(BaseModel):
    # ...existing fields...
    region: RegionSpec | None = Field(
        default=None,
        description=(
            "Optional spatial region. When provided, mask generation is "
            "constrained per region_combine. Enables 'mitigate this "
            "quadrant only' (intersect with range mask) or 'exclude this "
            "satellite trail' (subtract) workflows that are impossible "
            "with statistic-based masks alone."
        ),
    )
    region_combine: Literal["intersect", "union", "subtract"] = Field(
        default="intersect",
        description=(
            "How the region combines with the base mask_type rule. "
            "intersect: both must be true — typical for 'mitigate this "
            "region only.' union: either — typical for 'this region plus "
            "the bright pixels.' subtract: base minus region — typical "
            "for 'protect this region from mitigation.'"
        ),
    )
```

Implementation: numpy + `skimage.draw.polygon` / `skimage.draw.ellipse` / simple rectangle slicing. The region rasterizes to a binary image, optionally feathers, then combines with the existing luminance/range/star mask per `region_combine`. Output is the familiar feathered float mask.

**Effort.** ~70 lines. Drops cleanly into existing `create_mask` plumbing.

**Why this matters.** Without spatial regions, the agent has no way to express "there specifically" in a mask. In the M42 occlusion case, the natural response is a rectangle or polygon mask over the bottom-right quadrant combined with a range mask on the dark zone — then `masked_process` (§9) applies a targeted background pedestal or local stretch to rescue that region without touching the rest of the frame. The same primitive supports "apply sharpening to the nebula core only" (polygon) or "exclude this satellite trail from saturation boosts" (subtract), both of which are workarounds today that require manual pixel_math.

---

## 13. Phase report visibility in run directory

**Status.** Each run produces a hierarchical set of artifacts under the run directory (`artifacts/`, `checkpoints/`, `logs/`, analysis outputs, previews). A per-phase human-readable report is not currently emitted. A user reading the run directory post-hoc has to grep logs and manually correlate them with tool JSON output to understand what happened in a phase. This also leaves `memory_search` without a clean corpus — the logs are fine for the agent mid-run but poor for structured retrieval after the fact.

**Patch.** On every `advance_phase` that passes the artifact gate (landed in `t30_advance_phase.py`), write a phase report to `<run_dir>/reports/<NN>_<phase_name>.md` that captures:

- **Header table.** Phase name, start/end timestamps, wall time, target, model, autonomous/HITL mode.
- **Tool Activity.** Chronological list of tool calls: tool name, condensed params (long strings truncated), tool-returned status (ok / regression_warning / error), and a short one-line summary of the tool's output when present.
- **Metrics Delta.** analyze_image snapshots at phase entry and phase exit, rendered as a before/after table per metric (mean, median, background, gradient_magnitude, SNR, FWHM stats, color_balance, clipping_pct).
- **Issues.** Any `regression_warning` payloads encountered (timestamp, origin tool, resolution status). Any `flag_dataset_issue` payloads raised during the phase (summary, category, resolution).
- **Artifacts.** Relative paths to the phase's checkpoints and preview images (last 5 plus any user-saved ones).
- **Agent Summary.** The agent's own summary text (final assistant message at phase completion) if present, lightly trimmed.

**Format.** Plain Markdown, self-contained, written atomically after phase advancement so the file is only present for completed phases. Overwrites are allowed only when a phase is re-entered via `rewind_phase` (per §43 of the framework memo); the previous report is preserved as `<NN>_<phase_name>_v<run>.md`.

**Effort.** ~120 lines. Hooks into `t30_advance_phase.py` after the artifact gate passes. Uses the existing LangGraph state for tool-call history and the run directory for file paths. No schema changes.

**Why this matters.** The artifact-gate fix landed earlier in the audit enforces disk-truth on phase transitions, but the disk currently carries raw data without narrative. A user who comes back to a completed run a week later cannot easily see which parameter caused which metric shift — they have to replay through logs. The report solves that for human review and gives `memory_search` a structured document per phase per run — far better for retrieval than raw logs. This is a foundation for the longer-horizon goal of using past runs as prior evidence on new targets.

---

## Summary and sequencing

Ordered by impact × inverse effort:

| # | Patch | Effort | Impact |
|---|---|---|---|
| 9 | `masked_process` wrapper | ~150 lines | highest — compounds with every other nonlinear tool |
| 2 | polynomial gradient fallback + regression_warning | ~70 lines | high — M20 case unblocked |
| 1 | GraXpert knob exposure | ~20 lines | medium-high — reproducibility + second degree of freedom |
| 7 | `hdr_composite` tool | ~220 lines | high — collapses a 4-step pattern |
| 4 | arbitrary-points tone curves | ~140 lines | medium-high — unlocks pro tonal shaping |
| 5 | HSV-space tone curves | ~150 lines | medium-high — unlocks color nuance |
| 10 | prompt/docstring fixes for reduce_stars, drizzle, multiscale_process, create_mask | ~20 lines | high per line — pure doctrine leverage |
| 3 | wavelet/bilateral denoise fallback | ~80 lines | medium — mitigates GraXpert denoise smear |
| 8 | per-channel stretch | ~60 lines | medium — fixes color casts after weak color_calibrate |
| 6 | hue-range saturation (absorbed into #5) | — | subsumed by HSV-space curves |
| 11 | `flag_dataset_issue` HITL interrupt | ~180 lines | medium-high — escape hatch for out-of-scope data issues |
| 12 | `create_mask` region parameter | ~70 lines | medium — unlocks spatial targeting; compounds with #9 |
| 13 | Phase report visibility | ~120 lines | medium — human debugging + `memory_search` corpus |

If forced to pick three to do first: **9, 2, 10** — they are the minimum set that would have changed the M20 outcome in either run (Kimi's or Sonnet's) without touching the agent or model. The M42 occlusion case adds **11, 12** as co-equal priorities once #9 is in — without them the agent has no way to distinguish "gradient" from "occlusion" and no way to spatially target mitigation.

## Grounding

Tool files referenced above verified in-tree at these paths (Apr 19, 2026):

- `/Users/micahshanks/Dev/muphrid/muphrid/tools/linear/t09_gradient.py`
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/linear/t12_noise_reduction.py`
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/nonlinear/t14_stretch.py`
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/nonlinear/t16_curves.py`
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/nonlinear/t18_saturation.py`
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/scikit/t25_create_mask.py` (nonlinear-phased; move to utility per framework memo)
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/scikit/t26_reduce_stars.py`
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/scikit/t27_multiscale_process.py`
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/preprocess/t04_register.py`

New tool files landing from this memo (not yet present in-tree):

- `/Users/micahshanks/Dev/muphrid/muphrid/tools/utility/t34_masked_process.py` (§9)
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/utility/t35_hdr_composite.py` (§7)
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/utility/t36_flag_dataset_issue.py` (§11)
- `/Users/micahshanks/Dev/muphrid/muphrid/tools/nonlinear/t33_hsv_adjust.py` (§5)
