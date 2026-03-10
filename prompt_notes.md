# System Prompt Notes

Things to include when writing the agent system prompt.

---

## Pipeline Overview

The pipeline has a strict linear → non-linear boundary at T14. Tools before T14
operate on linear data; tools after T14 operate on non-linear data. This boundary
must never be crossed in the wrong direction.

### Preprocessing (always run, in order)

| Tool | Name | Notes |
|------|------|-------|
| T01 | ingest_dataset | Reads RAW/FITS, extracts EXIF metadata, classifies file types |
| T29 | resolve_target | SIMBAD coordinate lookup — run immediately after T01, before any plate solve |
| T02 | build_masters | Build bias, dark, flat masters (run three times: bias → dark → flat) |
| T02b | convert_sequence | Convert raw lights to FITS sequence before calibration |
| T03 | calibrate | Apply masters, debayer. X-Trans: is_cfa=True, fix_xtrans=True, equalize_cfa=True |
| T04 | register | Align frames. framing=min for FITSEQ compatibility |
| T05 | analyze_frames | Parse registration metrics: FWHM, roundness, star count, background |
| T06 | select_frames | Reject poor frames. min_star_count = ~50% of T05 median_star_count |
| T07 | siril_stack | Stack accepted frames. output_32bit=True always. Rejection method by N: winsorized (<15), sigma_clipping (15–50), linear_fit (>50) |
| T08 | auto_crop | Crop registration borders |

### Linear processing (run on stacked FITS, before stretch)

| Tool | Name | Notes |
|------|------|-------|
| T09 | remove_gradient | GraXpert gradient removal — run before color calibration |
| T10 | color_calibrate | Plate solve + PCC/SPCC color calibration. Raises on failure — retry with adjusted params |
| T11 | remove_green_noise | OSC/DSLR only. Check green_excess from T20 first — skip if near zero |
| T12 | noise_reduction | GraXpert AI denoising. Tune strength by SNR: 0.3–0.4 (high SNR), 0.6–0.8 (low SNR) |
| T13 | deconvolution | Sharpening via PSF deconvolution. Use manual PSF with T05 median_fwhm if makepsf fails post-denoise |

### THE CROSSING — T14

| Tool | Name | Notes |
|------|------|-------|
| T14 | stretch_image | Linear → non-linear. After this, ALL linear tools are forbidden. Produce 2–3 variants (gentle/moderate/aggressive) for HITL. |

### Non-linear processing (run after stretch)

| Tool | Name | Notes |
|------|------|-------|
| T15 | star_removal | StarNet2. upscale=True only if median_fwhm < 2.0px. Produces starless + star mask |
| T16 | curves_adjust | MTF or curves tone adjustment. Light touch — avoid clipping |
| T17 | local_contrast_enhance | CLAHE-based local contrast. Good for bringing out diffuse nebulosity |
| T18 | saturation_adjust | Global for broadband OSC. Narrowband: use per-channel |
| T19 | star_restoration | Blend stars back onto processed starless image. star_weight controls star prominence |
| T24 | export_final | Convert to sRGB, save TIFF + JPG for delivery |

### Masked application pattern (T25 → T27 → T23)

The standard pattern for region-selective enhancement on the starless image:

1. **T25** — create a luminance or custom mask isolating the region to enhance
2. **T27** — apply multiscale processing (sharpen/suppress) to the full starless image
3. **T23** — blend: `$processed$ * $mask$ + $original$ * (1 - $mask$)`

This confines aggressive processing to the target region while leaving the rest unchanged.

---

## Optional Tools

These tools are not in the standard pipeline but the agent should invoke them
when the data or target warrants it.

### T20 — analyze_image
**When to use:** Before and after every image-modifying step to get quantitative
feedback. Returns per-channel stats, background noise, star metrics, clipping
percentages. Essential for deciding whether to re-run a step with different
parameters (e.g. stretch too aggressive → clipped_highlights_pct > 0.1%).
This is the agent's primary diagnostic instrument — use it liberally.

### T21 — plate_solve (standalone)
**When to use:** When WCS coordinates or pixel scale are needed independently of
color calibration — e.g. to get pixel_scale_arcsec for T13 deconvolution PSF
sizing before T10 is run, or to verify an image has valid WCS after a prior step.
T10 calls the same plate solve internally; T21 exposes it standalone.

### T26 — reduce_stars
**When to use:** When stars are visually too large/bloated after all processing
is complete (post T19), particularly in crowded fields or when tight registration
wasn't achievable. Physically shrinks star disks via morphological erosion within
masked star regions. Different from T19's star_weight (which dims stars without
changing size). Apply as a last step before T24 export.

### T28 — extract_narrowband
**When to use:** When the user captured with a duoband narrowband filter
(e.g. Optolong L-eNhance, L-Ultimate, STC Duo-Narrowband) on an OSC/DSLR.
Extracts Hα and O-III as separate grayscale FITS files from non-debayered CFA
frames. Enables the OSC dual-narrowband workflow:
- T03 with debayer=False to keep CFA data
- T28 to split Hα / O-III channels
- T04 + T07 independently for each channel
- T27 / T17 per channel
- T23 pixel_math to recombine (SHO, HOO, or custom palette)

### T11 — remove_green_noise (conditional)
**When to use:** OSC/DSLR data only. Run T20 first and check green_excess —
skip T11 entirely if the value is near zero to avoid introducing a magenta cast.
Targets with genuine green emission (e.g. [OIII]-rich narrowband) use a
reduced amount (0.5–0.7) rather than the default.

---

## HDR Stretch Compositing for High Dynamic Range Targets

Cross-tool workflow pattern — belongs here, not in T14's docstring.

**Problem:** Emission nebulae with bright cores (e.g. M42 Trapezium) have extreme
dynamic range. A single stretch aggressive enough to reveal the faint outer shell
will blow out the core; a conservative stretch protecting the core leaves the outer
regions invisible.

**Strategy:** Produce two stretch variants at different `symmetry_point` (SP) values,
then blend them using a luminance mask.

1. **T14** — faint nebula pass: low SP (0.0–0.1), high `highlight_protection` (0.95–0.98),
   aggressive `stretch_amount` (3.5–4.5). `output_suffix="faint"`.
2. **T14** — core pass: SP set to median brightness of the bright core region,
   moderate `stretch_amount` (1.5–2.5), lower `highlight_protection` (0.90–0.95).
   `output_suffix="core"`.
3. **T25** — create a luminance mask bright in the core, dark in the outer regions.
   `mask_type="luminance"`, tune `high` threshold to isolate the core.
4. **T23** — blend: `$core$ * $mask$ + $faint$ * (1 - $mask$)`.

The `output_suffix` parameter on T14 is specifically designed to allow multiple
stretch variants without overwriting each other.

---

## General Principle

Tool docstrings give the agent vocabulary (what each tool does, its parameters).
The system prompt gives the agent grammar — how to compose tools into multi-step
strategies for specific astrophotography scenarios.

Add target-type-specific strategies here as they are identified:
- Emission nebulae with bright cores → HDR compositing (above)
- Galaxy with bright nucleus + faint spiral arms → similar HDR pattern
- Globular clusters → aggressive star reduction (T26), conservative stretch
- Narrowband OSC (duoband filter) → T28 channel extraction workflow (above)
