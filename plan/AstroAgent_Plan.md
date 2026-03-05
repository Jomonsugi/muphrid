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
T01 ingest_dataset — for FITS: wrong IMAGETYP classification causes silent calibration failure. For camera RAW (current test data is Fujifilm RAF): classification relies on subdirectory names; EXIF extraction via `pyexiftool`/ExifTool must correctly parse `ExposureTime`, `FocalLength`, `ISO`, `Model`. The `input_format` field flows into T03's `is_cfa` flag.
T09 remove_gradient — GraXpert direct subprocess invocation is confirmed working (v3.0.2, CoreML acceleration, bge-ai-models/1.0.1 cached locally). Key confirmed CLI flags: `-cli -cmd background-extraction -correction Subtraction -smoothing 0.5 -ai_version 1.0.1 [-bg]`. Denoising model 2.0.0 also cached (denoise-ai-models/2.0.0). GraXpert denoising strength is not a direct CLI flag — pass a temp preferences JSON via `-preferences_file`.
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
  `AstroState`, `PathState`, `MasterPaths`, `Metadata`, `Metrics`, `Dataset`, `FileInventory`, `AcquisitionMeta`, `FrameMetrics`, `ReportEntry`, `HITLPayload`, **`SessionContext`** with correct `Annotated` reducers for `history` (append-only), `messages` (`add_messages`), and `processing_report` (append-only list of `ReportEntry`). `PathState` includes `latest_preview: str | None`. `SessionContext` holds human-provided startup context: `target_name` (required), `bortle` (required), `sqm_reading` (optional), `remove_stars`, `notes`. Injected into every planner call via `build_state_summary()`. **Phase 6:** `AcquisitionMeta` extended with sensor fields (`black_level`, `white_level`, `bit_depth`, `raw_exposure_bias`, `sensor_type`). `FrameMetrics` fields made nullable; added `laplacian_sharpness` for Laplacian-fallback path. Spec: §2.5, §4.

- [x] **`ProcessingPhase` enum**
  All phases from §4.3. Helper: `is_linear_phase() -> bool`. Spec: §4.3.

---

## Phase 2 — Pre-Processing Tools (T01–T08)

*Each tool is a standalone `@tool`-decorated function. No LangGraph dependency. Test each in isolation with real FITS data before moving on.*

- [x] **T01 `ingest_dataset`**
  Walk directory. Auto-detect format: FITS (`*.fits` etc.) → classify by `IMAGETYP` header, extract metadata via `astropy`; Camera RAW (`*.raf`, `*.cr2`, etc.) → classify by subdirectory name (`lights/`, `darks/`, `flats/`, `bias/`), extract metadata via `pyexiftool` (wraps ExifTool binary). Store `input_format` in `AcquisitionMeta` so T03 can set CFA flags. Both paths return the same `Dataset` schema. Spec: §5 T01.
  Tested against real Fujifilm RAF data — extracts camera model, ISO, exposure, focal length. FITS path is a stub (NotImplementedError). **Phase 6:** Sensor characterization via `_sensor.py` (`sensor_info_from_tags`) — black/white level, bit depth, raw exposure bias. Calibration-frame cross-validation; `summary.sensor` reports usable sensor info.

- [x] **T02b `convert_sequence`** *(new in Phase 6)*
  Convert raw light frames to Siril FITSEQ before T03. Called after T01, before T02/T03. Enables calibration and registration on CFA lights. Spec: §5 T02b.

- [x] **T02 `build_masters`**
  Build master bias, dark, or flat via Siril `convert` + `stack`. Enforce dependency order (bias → dark → flat). Takes `acquisition_meta`; returns `master_path` + `diagnostics` (with `hitl_required`, `hitl_context`, `quality_flags`). **Phase 6:** Sensor-relative HITL thresholds, frame-count HITL, bias consistency check, dark sanity check via `_sensor.py`. Spec: §5 T02.

- [x] **T03 `siril_calibrate`**
  Apply masters to lights via Siril `calibrate`. Handle CFA/OSC flag, `equalize_cfa`, cosmetic correction, dark optimization. Return calibrated sequence path. Spec: §5 T03.

- [x] **T04 `siril_register`**
  Two-pass registration via Siril `register -2pass` + `seqapplyreg`. Support homography/affine/shift, Lanczos4 interpolation, optional drizzle, FWHM/roundness inline filtering. Spec: §5 T04.

- [x] **T05 `analyze_frames`**
  **Phase 6:** Parse `.seq` file directly for per-frame metadata (no seqstat/seqpsf). Star metrics (FWHM, eccentricity, etc.) when available; Laplacian-variance sharpness fallback when star detection fails. Output: `has_star_metrics`, `background_lvl`, `number_of_stars`, `laplacian_sharpness`, nullable FWHM fields. Spec: §5 T05.

- [x] **T06 `select_frames`**
  Pure Python sigma-clipping on T05 metrics. **Phase 6:** Two paths: star-metrics (FWHM, roundness, star count, background) vs Laplacian-percentile when `has_star_metrics=false`. Inputs: `has_star_metrics`, `keep_percentile`. Output: `selection_path` ("star_metrics" or "laplacian_percentile"). Contract: non-empty input never returns zero accepted frames; empty `frame_metrics` fails loudly. Spec: §5 T06.

- [x] **T07 `siril_stack`**
  Parses .seq file to build filename→index map. Emits `unselect`/`select` preamble for accepted_frames. Runs `stack` with sigma/winsorized rejection, addscale normalization, wfwhm weighting, 32-bit output. **Phase 6:** `total_frames_hint` from T05 for acceptance-rate HITL. Returns `hitl_required`, `hitl_context` when accepted < 2 or acceptance rate < 15%. Spec: §5 T07.

- [x] **T08 `auto_crop`**
  Load FITS with Astropy, compute bounding box on signal mask (per-pixel max across channels > threshold), apply 5px safety inset, run Siril `crop`. Guard added: if inset collapses tiny signal regions, fallback to non-inset bbox so crop geometry stays valid (no negative width/height). Spec: §5 T08.

---

## Phase 3 — Linear Processing Tools (T09–T13)

*All operate on linear data. Each must guard against being called post-stretch.*

- [x] **T09 `remove_gradient`**
  Primary: call `GRAXPERT_BIN` directly via subprocess — do NOT use Siril's `pyscript GraXpert-AI.py` wrapper. GraXpert CLI: `graxpert <image> -cli -cmd background-extraction -output <out.fits> -correction <Subtraction|Division> -smoothing <float> -ai_version 1.0.1 [-bg]`. Fallback: Siril `subsky -rbf` or `subsky <degree> -samples=N -tolerance=N`. Return processed image path + optional background model. `requires_visual_review=True` by default (HITL at this checkpoint). Spec: §5 T09.
  Bugs fixed in Phase 5c: `subsky` polynomial syntax corrected to flag-based args; `dither: bool` added.

- [x] **T10 `color_calibrate`**
  Internally calls `plate_solve` if WCS not present. Then Siril `pcc` or `spcc`. Return calibrated path, WCS coords, pixel scale, color coefficients. `limitmag` and `bgtol_lower/bgtol_upper` added in Phase 5c. Spec: §5 T10.

- [x] **T11 `remove_green_noise`**
  Siril `rmgreen` with average-neutral or maximum-neutral protection. Only valid on OSC/DSLR data. Bug fixed in Phase 5c: `amount` parameter omitted for types 0/1. Spec: §5 T11.

- [x] **T12 `noise_reduction`**
  Primary: Siril `denoise` (NL-Bayes, DA3D, SOS). Alternative: call `GRAXPERT_BIN` directly via subprocess — do NOT use Siril's `pyscript` wrapper. GraXpert CLI: `graxpert <image> -cli -cmd denoising -output <out> -strength <value>`. Measure and return `noise_before`/`noise_after` via `bgnoise`. `use_vst` and `apply_cosmetic` added in Phase 5c. Spec: §5 T12.

- [x] **T13 `deconvolution`**
  Methods: Siril `rl` (Richardson-Lucy + TV regularization), Siril `wiener`. PSF via Siril `makepsf stars` (primary) or `makepsf blind` (fallback) or `makepsf manual` (Moffat/Airy — added Phase 5c). `rl_stop` and `-fh` regularization added Phase 5c. Spec: §5 T13.
  **Note:** `requires_visual_review=True` by default. Deconvolution over-sharpening (ringing, halos) is a known artefact risk — HITL is mandatory in V1. The bool flag is designed to be flipped to `False` once confidence is established.

---

## Phase 4 — Stretch & Non-Linear Tools (T14–T19)

- [x] **T14 `stretch_image`**
  Siril `ght` (GHS), `asinh`, or `autostretch`. Measure `clipped_shadows_pct` and `clipped_highlights_pct` from output stats. Set `metadata.is_linear = false` in state. Bug fixed in Phase 5c: `color_model="even"` flag now always emitted. Spec: §5 T14.

- [x] **T15 `star_removal`**
  Calls `STARNET_BIN` (starnet2-mps PyTorch/MPS build) directly via subprocess with `-i/-o/-m/-w/-u` flags. FITS→TIF→StarNet→FITS conversion chain handled internally. `requires_visual_review=True` by default. Spec: §5 T15.
  **Note — PyTorch/MPS StarNet (starnet2-mps):** Binary requires `install_name_tool -add_rpath <lib_dir>` and `codesign --force --sign -` on first install.

- [x] **T16 `curves_adjust`**
  Siril `mtf` (global) or `ght` with high B value (targeted tonal range). Per-channel support. Spec: §5 T16.

- [x] **T17 `local_contrast_enhance`**
  Siril `clahe`, `unsharp`, or `epf` (edge-preserving bilateral/guided filter — added Phase 5c). Spec: §5 T17.

- [x] **T18 `saturation_adjust`**
  Siril `satu` with background protection factor and optional hue index (0–6). `ght -sat` for midtone-focused saturation. Spec: §5 T18.

- [x] **T19 `star_restoration`**
  Blend mode: Siril `pm "$starless$ + $starmask$ * <weight>"`. Synthstar mode: Siril `synthstar` on original linear image. Spec: §5 T19.

---

## Phase 5 — Utility Tools (T20–T24)

- [x] **T20 `analyze_image`**
  Siril `bgnoise` for authoritative noise (MAD-based). Astropy+numpy for per-channel sigma-clipped stats. Photutils `IRAFStarFinder` for star count, FWHM, roundness, `fwhm_std` (PSF uniformity), `median_star_peak_ratio` (star dominance indicator). Scipy `sobel`/`gaussian_filter` for `gradient_magnitude`. Derived: `snr_estimate`, `linearity` (dual-consensus: median + histogram skewness replaces fragile `is_linear_estimate`), `flatness_score`, `per_channel_bg` (sky-only pixels per channel — color neutralization verifier), `signal_coverage_pct` (nebulosity fraction), `color` (HSV saturation stats — drives T18), `clipping`, `green_excess`. NaN/Inf input pixels are sanitized on load so tool outputs remain finite and deterministic.

- [x] **T21 `plate_solve`**
  Siril `platesolve [ra dec] -focal= -pixelsize=` (coords positional, not `-ra=/-dec=`). Reuses `_build_platesolve_cmd` helpers from T10. Added `-force` and `-localasnet` flags. Spec: §5 T21.

- [x] **T22 `generate_preview`** *(internal function — not agent-callable)*
  Siril `autostretch -linked` + `savejpg`. PIL resize + annotation. NOT a @tool — verified by verify_phase_5.py. Spec: §5 T22.

- [x] **T23 `pixel_math`**
  Siril `pm "expression"`. Validates all `$stem$` tokens exist in working_dir before invoking. Spec: §5 T23.

- [x] **T24 `export_final`**
  Siril `icc_convert_to` + `savetif`/`savetif32`/`savejpg`/`savepng`. Default: Rec2020 TIFF16 (archival, wide-gamut) + sRGB JPG (web). Note: AdobeRGB is not a Siril built-in — Rec2020 is the correct wide-gamut choice. Output to `export/`. Spec: §5 T24.

---

## Phase 5b — scikit-image Tools (T25–T27)

*Pure-Python tools. No Siril invocation. All I/O via `astropy.io.fits`.*

- [x] **T25 `create_mask`**
  Astropy load → numpy. Four mask types: `luminance`, `inverted_luminance`, `range`, `channel_diff`. Morphological `binary_dilation`/`binary_erosion` with `disk`. Gaussian feathering. Float32 FITS output. Spec: §5 T25.

- [x] **T26 `reduce_stars`**
  Accepts star mask from T15 or auto-thresholds luminance. `skimage.morphology.erosion(channel, disk(kernel_radius))` for `iterations` passes within star region. Core exclusion via `binary_dilation`. Feathered blend with `blend_amount`. Reports `stars_affected_count`, `mean_size_reduction_pct`. Spec: §5 T26.

- [x] **T27 `multiscale_process`**
  B3-spline à trous transform via `scipy.ndimage.convolve1d` with kernel `[1/16, 4/16, 6/16, 4/16, 1/16]` and stride=2^i per scale. NOT `pywt.swt2('b3')` — PyWavelets has no 'b3'. Per-scale operations: `sharpen` (weight mult), `denoise` (MAD soft-threshold), `suppress` (zero), `passthrough`. Roundtrip residual < 1e-5 verified. Supports mono and color images in luminance mode. If `mask_path` shape differs from the image, mask is auto-resized before blending.

---

## Phase 5c — Pre-Phase 6 Full Tool Review (completed)

*Full audit of all 28 tools before graph wiring. Verified against Siril 1.4 docs.*

- [x] **T09 `remove_gradient` — subsky polynomial syntax bug fixed**
  `subsky {degree} {samples} {tolerance}` (positional, WRONG) → `subsky {degree} -samples={N} -tolerance={N}` (correct Siril 1.4 flag syntax). Added `dither: bool` field for low-dynamic-range gradient suppression.

- [x] **T10 `color_calibrate` — star magnitude and background tolerance controls**
  Added `limitmag: str | None` to PCC and SPCC: `+N/-N` for relative depth, absolute value for fixed limit. Added `bgtol_lower/bgtol_upper: float | None` for background sample rejection control — critical for extended nebula frames where nebulosity fills the frame and would otherwise bias the background reference.

- [x] **T12 `noise_reduction` — VST and cosmetic correction flags**
  Added `use_vst: bool` (Anscombe variance-stabilizing transform, `-vst` flag). VST is incompatible with DA3D/SOS and is only applied for `method=standard`. Added `apply_cosmetic: bool` (default True) that maps to Siril's `-nocosmetic` flag when False — prevents double-applying cosmetic correction.

- [x] **T13 `deconvolution` — makepsf manual profiles, rl -stop=, -fh regularization**
  Added `MakePsfManualOptions` supporting `psf_source=manual`: `-moffat` (with `beta` for wing control), `-airy` (with telescope optics parameters: diameter, focal length, wavelength, obstruction), `-gaussian`, `-disc`. Added `rl_options.stop: float | None` for early stopping. Added `hessian_frobenius` as a regularization option (`-fh` flag — Frobenius norm of Hessian, better at preserving linear structure than TV). Added `symmetric: bool` for `makepsf stars -sym`. Removed stale `makepsf -auto` reference from docstring.

- [x] **T17 `local_contrast_enhance` — epf edge-preserving filter**
  Added `method=edge_preserve` mapping to Siril's `epf` command (bilateral or guided filter). New `EpfOptions` schema: `guided` (bilateral vs guided), `diameter`, `intensity_sigma` and `spatial_sigma` (normalized 0–1 for 32-bit FITS), `mod` (blend strength), `guide_image_stem` (for guided filter with separate guide). Bilateral EPF is PixInsight-tier structure-safe noise smoothing — excellent for cleaning residual noise from background and faint regions after NL-Bayes.

- [x] **T28 `extract_narrowband` — new tool: Hα/O-III/Green CFA extraction**
  New tool at `astro_agent/tools/utility/t28_extract_narrowband.py`. Wraps Siril `extract_Ha`, `extract_HaOIII`, `extract_Green`. Enables the OSC dual-narrowband workflow for Fujifilm X-T30 II (or any OSC camera) with a duoband filter (Optolong L-eNhance, L-Ultimate, STC Duo, etc.). Input must be CFA (non-debayered) FITS from T03 with `debayer=False`. `upscale_ha` and `resample` options for matching output dimensions. Returns `ha_path`, `oiii_path`, `green_path`. Next steps in output: register+stack each channel independently (T04+T07) then combine via T23 pixel_math (HOO or SHO palette). Spec: §5 T28.

---

## Future Tools (add when needed, not before)

- **T29 `fix_banding`** — Siril `fixbanding` command. Removes horizontal/vertical
  fixed-pattern noise (DSLR sensor banding). Add when banding artifacts are
  observed in real data. The agent would detect banding via a periodic noise
  metric in T20 (row/column variance analysis) and call this tool automatically.
  Not added yet — the X-T30 II may never show banding, and the tool is only
  useful if the artifact actually appears.

- **T30 `combine_channels`** — Dedicated narrowband palette mapping tool for
  multi-filter setups (Ha, OIII, SII). Would provide named palette presets
  (SHO/Hubble, HOO, HOS) with per-channel weight and bias controls, plus
  automatic star color restoration from a broadband RGB reference. Currently
  achievable via T23 `pixel_math` expressions, but a dedicated tool would give
  the agent structured parameters instead of freeform math strings — reducing
  prompt complexity and error surface for narrowband combination workflows.

---

## Phase 5d — Math Integrity Regression Gate (completed)

- [x] **Added `scripts/verify_math_integrity.py` and integrated into verification workflow**
  Focused regression checks for math-heavy Python tools (T06, T08, T20, T26, T27):
  mono/color path handling, NaN/Inf sanitation, valid crop geometry for tiny regions,
  mismatched mask resize behavior, and deterministic fail-loud preconditions.
  Current status: all checks passing.

---

## Phase 5e — Pre-Phase 6 Data Quality Audit (completed)

*Full audit of all tools focused on agent data-driven decision quality.*

- [x] **`Metrics` TypedDict expanded** to capture full T20 output
  Added 15 new fields: `dynamic_range_db`, `gradient_magnitude`, `per_channel_bg`, `channel_imbalance`, `mean_saturation`, `median_saturation`, `is_linear_estimate`, `linearity_confidence`, `histogram_skewness`, `signal_coverage_pct`, `clipped_shadows_pct`, `clipped_highlights_pct`, `star_count`, `fwhm_std`, `median_star_peak_ratio`, `contrast_ratio`. The state now carries all data the agent needs for math-based decisions without re-running T20.

- [x] **`is_osc` auto-derived from ingestion format**
  `make_empty_state()` now sets `metadata.is_osc = True` when `input_format == "raw"` (camera RAW is always CFA). Previously defaulted to False.

- [x] **T05 `analyze_frames` — enriched summary statistics**
  Added `median_eccentricity`, `median_star_count`, `median_background`, `std_background`, `median_noise`, and `frame_count` to the summary dict. The agent now has all aggregates needed to set T06 rejection thresholds without manually parsing per-frame data.

- [x] **T20 `analyze_image` — added `contrast_ratio` metric**
  `contrast_ratio = p95 - p5` of luminance. Measures usable tonal range. < 0.3 post-stretch → flat image needing T16 curves; > 0.8 → high contrast, cautious with further enhancement. Previously referenced in T16 docstring but never computed.

- [x] **T10 `color_calibrate` — `background_neutralization` description corrected**
  Siril PCC/SPCC always include background neutralization — there is no `-nobgn` CLI flag. Updated description to document this as advisory and added the fallback path: T20 `per_channel_bg` + T23 pixel_math manual equalization when plate solving fails.

- [x] **T11 `remove_green_noise` — all 4 Siril rmgreen types exposed**
  Added `maximum_mask` (type 2) and `additive_mask` (type 3) protection types, which accept the `amount` parameter (0–1). Types 0/1 (already present) do not use amount. This gives the agent proportional green removal control — critical for targets with genuine green emission (e.g., [OIII] in broadband images).

---

## Phase 6 — Tool Layer Milestone

*Prove tools work independently before building the agent.*

- [x] **Manual pipeline integration script**
  A plain Python script (not a test, not the graph) that calls T01 → T28 sequentially on a real dataset. Every intermediate file must exist on disk. The script must demonstrate the masked-application pattern: T25 → T27 → T23 on the starless image. The script completes without error and produces a viewable export. This is the gate before Phase 7.

- [x] **T02b `convert_sequence`** — raw lights → FITSEQ before calibration
  Converts camera RAW lights to Siril sequence format prior to T02/T03. Required for Fujifilm RAF and other raw workflows.

- [x] **Sensor-aware calibration** — `tools/_sensor.py`, T01/T02
  Shared helpers for black/white level, fill fraction, flat quality. T01 extracts sensor characterization; T02 uses sensor-relative thresholds for HITL (bias consistency, dark sanity, flat quality).

- [x] **`scripts/check_flat_quality.py`**
  Standalone script to validate flat/bias quality (fill fraction, clipped pixels). Not part of the main pipeline; used for dataset health checks.

---

## Phase 7 — LangGraph Graph

*Wire the tools into the orchestration layer.*

- [ ] **State reducers and `make_initial_state()`**
  Confirm `history` appends (not replaces) across updates. `make_initial_state(dataset_path)` calls T01 and returns a fully populated `AstroState` with `phase = INGEST`. Spec: §4, §6.
  **Note:** `is_osc`, `exposure_seconds`, and `frame_count` from T01 metadata
  drive the first branching decisions (calibration strategy, stacking method,
  rejection thresholds). Add an assertion that `acquisition_meta.input_format`
  is populated — if T01 can't determine this, every downstream decision is
  unreliable.

- [ ] **`tools/control.py` — `advance_phase` and `request_hitl`**
  Both as `@tool`-decorated functions with the exact signatures and docstrings from §6.4. These are the LLM's signals for routing — not processing tools.

- [ ] **`PHASE_TOOLS` registry**
  Dict mapping each `ProcessingPhase` to the subset of tools the LLM may call during that phase. Verify: `color_calibrate` is absent from `NONLINEAR`, `stretch_image` is absent from `LINEAR`, `reduce_stars` is absent from `LINEAR`, `multiscale_process` in `LINEAR` accepts only `denoise` operations (enforced via agent notes, not code). Spec: §6.3.

- [ ] **Tool executor node**
  `ToolNode` wrapping all 28 tools (T01–T28). After execution: update
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
  Implement `ToolEntry` dataclass with `fn`, `modifies_image`, `requires_visual_review`, `visual_review_question`, `visual_review_options` fields. Populate all 28 tools per §6.5. `requires_visual_review=True` for `remove_gradient`, `deconvolution`, `star_removal` in V1. Changing a checkpoint from required to optional in the future is a one-line boolean change. Spec: §6.5.

- [ ] **`tool_executor_post_hook()`**
  After each tool call: (1) if `entry.modifies_image`, call `analyze_image_fn` and update `state.metrics`; (2) if `entry.requires_visual_review`, call `auto_hitl_check()` which calls `generate_preview_fn` and returns a HITL payload; (3) inject payload into state routing so the graph routes to `hitl_interrupt_node` before returning to planner. `generate_preview_fn` is never called outside this hook and the mandatory HITL nodes. Spec: §6.5.
  **Note:** When previewing linear-stage images for HITL, the preview must
  apply an auto-stretch — raw linear FITS rendered as-is are nearly black.
  Verify T22 `generate_preview` handles this (Siril's `savejpg` auto-stretches,
  but confirm the output is human-reviewable at every pipeline stage).

- [ ] **Feedback integration**
  User's free-text revision requests in `state.user_feedback["revision_requests"]` are read by the planner on resume and factored into parameter selection on the next iteration. Spec: §8.3.

---

## Phase 10 — Entry Point

- [ ] **`cli.py`**
  ```
  astro_agent process <directory>
    --target "M42 Orion Nebula"     (REQUIRED)
    --bortle 4                      (REQUIRED)
    --sqm 20.8                      (optional)
    --remove-stars ask|yes|no       (default: ask → HITL)
    --notes "free text"             (optional)
    --profile conservative|balanced|aggressive
    --resume <thread_id>
    --no-hitl
  ```
  Collects `SessionContext` from CLI arguments (prompts interactively for required `--target` and `--bortle` if omitted). Runs `make_initial_state(dataset, session)` then streams the graph via `app.stream(..., stream_mode="updates")`. On each chunk, checks for `__interrupt__`: if present, calls `cli_presenter(payload)` to display images + question, reads response, and resumes with `app.invoke(Command(resume=response), config)`. If `--no-hitl`, uses `auto_presenter` instead. Print thread ID at start so the user can resume later. Spec: §2.5, §8.0.

---

## Phase 11 — End-to-End Validation

- [ ] **Full pipeline run on a real dataset**
  Run `astro_agent process` on 10–30 light frames with calibration sets (bias, dark, flat) of a bright target. All phases complete. Both HITL checkpoints pause and resume correctly. Final TIFF and JPG are written to `export/`. A person looking at the output judges it as a competent result.
