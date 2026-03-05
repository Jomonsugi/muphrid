# AstroAgent — Final Specification

**CLI-First | Siril 1.4+ & GraXpert CLI | LangGraph StateGraph | Metrics-Driven ReAct Agent with HITL Visual Review**

An LLM agent that fully post-processes deep-sky astrophotography images from raw
calibration frames to a finished export. The agent owns both the deterministic
pipeline (calibration, registration, stacking) and the nuanced aesthetic
decisions (gradient removal, stretching, sharpening, color grading). A
human-in-the-loop (HITL) protocol lets the photographer override subjective
choices via A/B variant selection.

---

## 0  Engineering Philosophy

**The goal is PixInsight-quality output from open source software.**
Every decision in this system — package selection, algorithm choice, tool
architecture — is made against that bar.

### Gold Standard Over Convenient

Always choose the most robust, comprehensive, and trusted tool for a given job,
even if it requires an additional system dependency or more setup.

The `pyexiftool` / ExifTool choice over `exifread` is the reference example:
ExifTool is what Adobe Lightroom, Capture One, and digiKam use under the hood.
It reads every RAW format including Fujifilm proprietary maker notes. `exifread`
failed on RAF entirely. When the easier path produces worse data, take the harder
path.

**Decision checklist for new packages/tools:**
1. Is this what professional tools (Lightroom, PixInsight, Capture One) use?
2. Does it handle the full range of formats, not just today's use case?
3. Is it actively maintained? (commits in the last 6 months)
4. Battle-tested C/C++ library wrapped in Python > pure-Python alternative
5. External binaries are a first-class pattern here (Siril, GraXpert, StarNet,
   ExifTool) — do not reject a better tool solely because it needs a binary

### Data Quality Drives Agent Quality

The agent makes decisions based on metrics and image statistics.
**Richer, more accurate data = better autonomous decisions.**

This applies everywhere:
- EXIF extraction (T01) — all maker notes, not just basic tags
- Image analysis (T20) — every meaningful statistical metric exposed
- Siril stdout parsing (`_siril.py`) — extract all parseable values

### Agentic Ethos Over Rigid Heuristics

The system gives the agent tools — the same tools a skilled human would use —
and lets it decide how and when to use them. Prescriptive checklists and
hardcoded heuristics are failures of system design, not features. The prompt
should be minimal; the tools and data should do the work.

---

## 1  Scope & Constraints


| Axis          | In Scope (v1)                                                            | Out of Scope (v2+)                                        |
| ------------- | ------------------------------------------------------------------------ | --------------------------------------------------------- |
| Targets       | Broadband deep-sky (RGB / OSC)                                           | Narrowband (SHO/HOO), mosaics, lucky imaging, planetary   |
| Cameras       | DSLR, dedicated astro OSC & mono (single-filter RGB)                     | Multi-filter mono requiring separate per-channel stacking |
| Tooling       | Siril ≥ 1.4 CLI (`siril-cli -s`), GraXpert CLI, Python/Astropy/Photutils | PixInsight, Photoshop, StarTools                          |
| Orchestration | LangGraph `StateGraph`, single-machine                                   | Distributed workers, cloud burst                          |


---

## 2  Global Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          AstroAgent                              │
│                                                                  │
│  ┌────────────┐    ┌──────────────────────────────────────────┐  │
│  │   LLM      │    │          LangGraph StateGraph            │  │
│  │  Planner   │◄──►│                                          │  │
│  │  (ReAct)   │    │  ┌──────────┐  ┌───────────────────────┐ │  │
│  └────────────┘    │  │Pre-Proc  │  │  Planner ◄─► Tool     │ │  │
│        ▲           │  │Subgraph  │─►│  Loop    ◄─► Executor │ │  │
│        │ metrics   │  │(determ.) │  │  (linear / nonlinear) │ │  │
│        │ only      │  └──────────┘  └──────────┬────────────┘ │  │
│  ┌─────┴──────┐    │                           │              │  │
│  │  Analyze   │    │  ┌──────────┐  ┌──────────▼───────────┐  │  │
│  │  (metrics) │    │  │   HITL   │◄─│  Export              │  │  │
│  └────────────┘    │  │Interrupt │  └──────────────────────┘  │  │
│                    │  └──────────┘                            │  │
│                    └──────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │               Tool Layer (Executors)                      │   │
│  │  siril-cli  │  GraXpert CLI  │  Python / Astropy / PIL    │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Component roles:**


| Component                   | Role                                                                                                                                                |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM Planner**             | Chooses next tool, selects parameters, evaluates results via quantitative metrics, decides when to advance phase or request HITL feedback.           |
| **Tool Executor**           | Translates tool-call JSON into `siril-cli` scripts, GraXpert CLI invocations, or Python routines. Auto-runs `analyze_image` after image-modifying calls; auto-triggers HITL for tools with `requires_visual_review=True`. |
| **AstroState**              | Single mutable state object threaded through every graph node. Carries paths, metrics, history, user feedback, and processing report entries.       |
| **Pre-Processing Subgraph** | Order-deterministic pipeline (calibrate → register → analyze → select → stack → crop). Parameter-agentic via `parameter_selector()` LLM calls.     |
| **Planner-Tool Loop**       | ReAct cycle for linear and non-linear processing. The LLM observes image metrics (text only), reasons, selects a tool, evaluates the result, and loops. Visual context reaches the agent only via HITL responses. |
| **HITL Interrupt**          | Hard-wired, auto-triggered, and on-demand pause points where previews are generated and the photographer provides visual assessment or feedback.    |


---

## 2.5  Session Startup — Mandatory Human Context

Before the pipeline starts, the CLI collects a minimal set of human-provided
context that shapes every subsequent decision. This information cannot be
derived from pixel data, EXIF, or plate-solve results — it is the photographer's
knowledge of the subject and session.

```
$ astro_agent process /path/to/dataset \
    --target "M42 Orion Nebula"   \    ← REQUIRED (will prompt if omitted)
    --bortle 4                    \    ← REQUIRED (will prompt if omitted)
    --sqm 20.8                    \    ← optional: SQM-L reading in mag/arcsec²
    --remove-stars ask            \    ← ask | yes | no (default: ask)
    --notes "L-eNhance duoband, 5-min subs, mediocre seeing"  ← optional
    --profile balanced                 ← conservative | balanced | aggressive
```

These five fields form the `SessionContext` stored at `state.session` and are
injected verbatim into every planner call via `build_state_summary()`. The
three-way `remove_stars` flag prevents the agent from making a star-processing
decision silently — `ask` routes to a HITL question ("Do you want to process
stars separately? This separates nebulosity from stars for independent
enhancement.") when the agent is ready to call T15.

**Why target name is required:** A frontier LLM already holds deep knowledge
about M42, M31, NGC 7000, the Milky Way core, etc. — their colour palettes,
angular extents, characteristic processing challenges, and what "good" looks
like. Providing the name unlocks that knowledge directly. Without it, the
agent can still reason from T20 metrics, but naming the target is the highest-
leverage single input.

---

## 3  Processing Pipeline — Canonical Order

The processing order below encodes the physical constraints of astrophotography
data. Tools in the linear phase operate on data whose pixel values are
proportional to photon counts; tools in the non-linear phase operate after a
non-linear stretch has been applied. Violating this order (e.g. running PCC
after stretch) produces incorrect results.

```
RAW FITS
  │
  ▼
┌─────────────────── PRE-PROCESSING (deterministic) ──────────────────┐
│ T01 ingest_dataset                                                  │
│ T02b convert_sequence  (lights → FITSEQ; raw input path only)        │
│ T02 build_masters  (bias → dark → flat)                             │
│ T03 siril_calibrate  (apply masters to lights)                      │
│ T04 siril_register   (star alignment)                               │
│ T05 analyze_frames   (per-frame FWHM, eccentricity, background)     │
│ T06 select_frames    (reject outliers)                              │
│ T07 siril_stack      (combine accepted frames)                      │
│ T08 auto_crop        (remove stacking borders)                      │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼  image is LINEAR (pixel values ∝ photon count)
┌─────────────────── LINEAR PROCESSING (agent-guided) ────────────────┐
│ T09 remove_gradient       (background extraction)                   │
│ T10 color_calibrate       (PCC / SPCC + background neutralization)  │
│ T11 remove_green_noise    (SCNR — OSC cameras only)                 │
│ T12 noise_reduction       (denoise in linear — mild, preserves SNR) │
│ T13 deconvolution         (Richardson-Lucy or Wiener — optional)    │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────── THE CROSSING ────────────────────────────────────┐
│ T14 stretch_image   (linear → non-linear: GHS, asinh, autostretch)  │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼  image is NON-LINEAR (perceptual brightness)
┌─────────────────── NON-LINEAR PROCESSING (agent + HITL) ───────────┐
│ T15 star_removal          (StarNet → starless + star mask)         │
│ T16 curves_adjust         (MTF / GHT fine-tuning on starless)      │
│ T17 local_contrast_enhance (CLAHE / wavelets / unsharp mask)       │
│ T18 saturation_adjust     (color enhancement, hue-targeted)        │
│ T19 star_restoration      (recombine or synthstar)                 │
│ T26 reduce_stars          (morphological star disk shrinking)       │
│ T27 multiscale_process    (per-scale wavelet sharpen / denoise)     │
└────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────── EXPORT ──────────────────────────────────────────┐
│ T24 export_final          (TIFF-16 / PNG / JPG + ICC profile)       │
└─────────────────────────────────────────────────────────────────────┘

Utility tools available at any phase (agent-callable):
  T20 analyze_image     T21 plate_solve     T23 pixel_math
  T24 export_final      T25 create_mask

  T22 generate_preview — internal only; not in PHASE_TOOLS; called exclusively
  by auto_hitl_check() and the mandatory HITL nodes (stretch_hitl, final_hitl).
```

---

## 4  State Schema

### 4.1 AstroState (top-level graph state)

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SessionContext(TypedDict):
    """
    Human-provided context collected before the pipeline starts.
    These five fields are the upfront knowledge that shapes the entire run.
    Injected into every planner call via build_state_summary().

    target_name   — Required. "M42 Orion Nebula", "NGC 7000 North America",
                    "M31 Andromeda", "Milky Way core". Single most impactful
                    context: a frontier LLM already knows what M42 looks like,
                    its colour palette, angular scale, and typical processing
                    decisions. Naming the target unlocks all of that knowledge.

    bortle        — Required. Bortle scale of the imaging site (1-9). Directly informs:
                      gradient removal aggressiveness (high Bortle → larger gradient)
                      noise reduction strength (high Bortle → more passes)
                      stretch conservatism (high Bortle → preserve faint structure)

    sqm_reading   — Optional SQM-L reading in mag/arcsec². If provided, this is
                    the higher-precision sky-quality signal and is used as the
                    primary metric with Bortle as coarse cross-check.

    remove_stars  — True = run T15+T19; False = skip entirely;
                    None = ask via HITL when the agent reaches that decision.

    notes         — Free-text session notes. Injected verbatim into every
                    planner prompt. Use for anything relevant but uncaptured:
                      "Shot with Optolong L-eNhance duoband filter"
                      "Very poor seeing — FWHM likely > 4px"
                      "Prioritise faint outer nebulosity over star cores"
    """
    target_name:  str          # required
    bortle:       int          # required: 1–9 Bortle scale
    sqm_reading:  float | None # optional: SQM-L reading in mag/arcsec² (e.g. 20.8)
    remove_stars: bool | None  # True/False/None=ask via HITL
    notes:        str | None   # optional free-text

class AstroState(TypedDict):
    # Human-provided upfront context — available to every node from step 1
    session: SessionContext

    # -- Core --
    dataset: Dataset
    phase: ProcessingPhase
    paths: PathState
    metadata: Metadata
    metrics: Metrics
    history: Annotated[list[str], operator.add]          # append-only processing log
    messages: Annotated[list, add_messages]               # LLM conversation
    user_feedback: dict                                   # accumulated HITL prefs
    processing_report: Annotated[list[ReportEntry], operator.add]  # narrative decision log

class PathState(TypedDict):
    current_image: str | None        # FITS path of the working image — always current
    latest_preview: str | None       # JPG path of the most recent preview — auto-updated
    starless_image: str | None       # after star removal
    star_mask: str | None            # after star removal
    masters: MasterPaths
    variants: dict[str, str]         # A/B testing: {"a": "path", "b": "path"}

class MasterPaths(TypedDict):
    bias: str | None
    dark: str | None
    flat: str | None

class Metadata(TypedDict):
    is_linear: bool
    is_color: bool                   # False for mono sensors
    is_osc: bool                     # True for one-shot-color / DSLR (CFA)
    pixel_scale: float | None        # arcsec/pixel (from plate solve)
    plate_solve_coords: dict | None  # {"ra": float, "dec": float}
    focal_length_mm: float | None
    pixel_size_um: float | None

class Metrics(TypedDict):
    frame_stats: dict[str, FrameMetrics]   # per-frame quality
    current_fwhm: float | None
    current_background: float | None
    current_noise: float | None
    snr_estimate: float | None
    channel_stats: dict | None             # per-channel mean/median/std
    background_flatness: float | None
    green_excess: float | None
```

### 4.2 Dataset

```python
class Dataset(TypedDict):
    id: str
    working_dir: str
    files: FileInventory
    acquisition_meta: AcquisitionMeta

class FileInventory(TypedDict):
    lights: list[str]
    darks: list[str]
    flats: list[str]
    biases: list[str]

class AcquisitionMeta(TypedDict):
    target_name: str | None
    focal_length_mm: float | None
    pixel_size_um: float | None
    exposure_time_s: float | None      # per-frame exposure in seconds
    iso: int | None                    # DSLR/mirrorless ISO
    gain: int | None                   # dedicated astro camera gain (ADU)
    filter: str | None                 # filter name (FITS FILTER keyword or None for RAW)
    bortle: int | None
    camera_model: str | None
    telescope: str | None
    input_format: str | None           # "fits" | "raw" — set by T01, read by T03
    # Sensor characterization — from T01, used by T02 for sensor-relative HITL thresholds
    black_level: int | None           # pedestal ADU (e.g. 1022 for Fuji X-T30 II)
    white_level: int | None            # sensor full-well ADU (e.g. 16383 for 14-bit)
    bit_depth: int | None              # 12, 14, or 16
    raw_exposure_bias: float | None    # stops (Fuji RAF:RawExposureBias; None for non-Fuji)
    sensor_type: str | None            # "bayer" | "xtrans" — affects T03 debayer kernel
```

### 4.3 ProcessingPhase

```python
from enum import Enum

class ProcessingPhase(str, Enum):
    INGEST       = "ingest"
    CALIBRATION  = "calibration"
    REGISTRATION = "registration"
    ANALYSIS     = "analysis"
    STACKING     = "stacking"
    LINEAR       = "linear"
    STRETCH      = "stretch"
    NONLINEAR    = "nonlinear"
    REVIEW       = "review"
    EXPORT       = "export"
    COMPLETE     = "complete"
```

### 4.4 FrameMetrics

```python
class FrameMetrics(TypedDict):
    fwhm: float | None
    eccentricity: float | None
    star_count: int | None          # maps to number_of_stars in T05 .seq output
    background_level: float | None  # maps to background_lvl in T05 .seq output
    noise_estimate: float | None
    roundness: float | None
    laplacian_sharpness: float | None  # when has_star_metrics=false
```

### 4.5 ReportEntry

Each entry in `state.processing_report` records one agent decision: what tool
was called, why, what parameters were chosen, and what the measurable outcome
was. The full list of entries is the source material for the human-readable
`processing_report.md` written at export time.

```python
from datetime import datetime

class ReportEntry(TypedDict):
    step: int                   # sequential step number across the full run
    phase: str                  # ProcessingPhase value at time of call
    tool: str                   # tool name as it appears in TOOL_REGISTRY
    reasoning: str              # LLM's stated reason extracted from message content
    parameters: dict            # the exact parameter dict passed to the tool
    metrics_before: dict | None # key metrics from analyze_image before the call
    metrics_after: dict | None  # key metrics from analyze_image after the call
    outcome: str                # "success" | "degraded" | "error" | "reverted"
    outcome_detail: str         # brief summary of what changed or what went wrong
    timestamp: str              # ISO 8601
```

`reasoning` is extracted from the LLM's message content immediately before the
tool call — the text in the assistant message that precedes the tool_call block.
This is the agent's own explanation of why it made the choice, in its own words.

`metrics_before` and `metrics_after` are populated only for tool calls that
modify `current_image`. For structural tools (plate_solve, create_mask,
pixel_math blends) they may be null.

`outcome` is set by the tool executor:
- `success` — tool completed and metrics did not degrade
- `degraded` — tool completed but `background_noise` increased or a key metric
  worsened; the agent may choose to revert
- `error` — tool raised an exception; see `outcome_detail` for the error
- `reverted` — this step was subsequently undone by reverting to a prior checkpoint

---

## 5  Tool Contracts

Each tool contract specifies: name, purpose, processing phase, implementation
backend, inputs, outputs, and agent decision heuristics.

All path arguments are absolute filesystem paths to FITS files unless noted.
All tools write outputs to the working directory and return paths to the new files.

---

### PRE-PROCESSING TOOLS

---

#### T01 — `ingest_dataset`

**Purpose:** Scan a directory tree, classify files by type (light/dark/flat/bias),
extract acquisition metadata, and populate the Dataset schema.

Supports two ingestion paths determined by the files found under
`root_directory`:

- **FITS path** (`*.fits`, `*.fit`, `*.fts`): classify by `IMAGETYP` /
  `FRAME` header; extract metadata from standard FITS keywords (`EXPTIME`,
  `FOCALLEN`, `XPIXSZ`, `GAIN`, `OBJECT`, `TELESCOP`, `FILTER`).
- **Camera RAW path** (`*.raf`, `*.cr2`, `*.cr3`, `*.arw`, `*.nef`, etc.):
  classify by subdirectory name (`lights/`, `darks/`, `flats/`, `bias/`);
  extract metadata via `pyexiftool` / `exiftool` binary (`ExposureTime`,
  `FocalLength`, `ISO`, `Model`). Siril reads RAW files natively and
  performs CFA debayering during calibration.

Both paths return the same `Dataset` schema. The detected format is stored in
`AcquisitionMeta.input_format` so downstream tools can set CFA flags
appropriately.

**Backend:** Python — `pathlib` directory walk; `astropy.io.fits` for FITS
headers; `pyexiftool` (wrapping the `exiftool` system binary) for camera RAW
EXIF tags. ExifTool is the gold standard for RAW metadata — it handles every
format (RAF, CR2, ARW, NEF, etc.) and reads Fujifilm proprietary maker notes
that other libraries miss. Install: `brew install exiftool`.

**Inputs:**

```json
{
  "root_directory": "str",
  "file_pattern": "str | null = null",
  "override_target_name": "str | null"
}
```

`file_pattern` is optional. When `null`, T01 auto-detects format by scanning
for known FITS and RAW extensions. Pass an explicit glob (e.g. `"*.fits"` or
`"*.RAF"`) to force a specific format.

**Outputs:**

```json
{
  "dataset": "Dataset",
  "warnings": ["str"],
  "summary": {
    "lights_count": "int",
    "darks_count": "int",
    "flats_count": "int",
    "biases_count": "int",
    "total_exposure_s": "float",
    "input_format": "fits | raw",
    "detected_extensions": ["str"]
  }
}
```

**Agent notes:** Always the first tool called. If `biases_count` or
`darks_count` is 0, adapt the calibration strategy (e.g. use synthetic bias
offset, skip dark subtraction if sensor has negligible dark current). Check
`input_format` and pass `is_cfa = true` to T03 when `input_format == "raw"`.
For RAW input, classification relies on subdirectory names — warn the user if
files are found in an unrecognized directory (not `lights/`, `darks/`,
`flats/`, or `bias/`). `override_target_name` sets `AcquisitionMeta.target_name`
when the object cannot be determined from headers or filename.

**Phase 6 additions:** T01 now populates sensor characterization in
`AcquisitionMeta`: `black_level`, `white_level`, `bit_depth`, `raw_exposure_bias`,
`sensor_type`. Cross-validates calibration frames (flat ISO vs light ISO, dark
exposure vs light, bias exposure near-zero). `summary.sensor` reports these
values for agent visibility.

---

#### T02b — `convert_sequence`

**Purpose:** Convert raw light frames to a Siril FITSEQ sequence. Required
before T03 when input is camera RAW — Siril's `calibrate` operates on
sequences, not loose files. T02 `build_masters` uses this internally for
bias/dark/flat; for lights, call T02b explicitly so the sequence exists for T03.

**Backend:** Siril CLI — `convert <name> -fitseq`. Copies raw files into
working_dir with indexed names, runs convert, moves .fit and .seq to target.
Note: single-file convert produces only .fit, not .seq; multi-file produces both.

**Inputs:**

```json
{
  "working_dir": "str",
  "input_files": ["str"],
  "sequence_name": "str",
  "debayer": "bool = false"
}
```

**Outputs:**

```json
{
  "sequence_path": "str",
  "fit_path": "str",
  "seq_path": "str"
}
```

**Agent notes:** Call after T01 for lights when `input_format == "raw"`.
Use `sequence_name = "lights_seq"`. T03 expects `lights_sequence` to match.
T02 build_masters invokes conversion internally for calibration frames.

---

#### T02 — `build_masters`

**Purpose:** Stack calibration frames into master bias, master dark, or master flat.

**Backend:** Siril CLI — `convert`, `stack` commands.

Order constraint: bias first, then dark (optionally bias-subtracted), then flat
(bias-subtracted before stacking to normalize sensor offset).

**Inputs:**

```json
{
  "working_dir": "str",
  "file_type": "bias | dark | flat",
  "input_files": ["str"],
  "master_bias_path": "str | null",
  "stack_method": "median | mean",
  "rejection_method": "sigma_clipping | winsorized | none",
  "rejection_sigma": [3.0, 3.0],
  "acquisition_meta": "dict | null"
}
```

**Outputs:**

```json
{
  "master_path": "str",
  "diagnostics": {
    "quality_flags": "dict",
    "warnings": ["str"],
    "hitl_required": "bool",
    "hitl_context": "str"
  }
}
```

**Agent notes:** Use `median` stacking for bias (lowest noise floor). Use
`sigma_clipping` rejection for darks and flats to eliminate cosmic rays and
satellite trails. Pass `master_bias_path` when building master flat to remove
bias pedestal before flat normalization. Pass `acquisition_meta` from T01
(`dataset.acquisition_meta`) to enable sensor-relative HITL thresholds — without
it, T02 reads EXIF from the first input file as fallback.

**HITL triggers (critical data issues):** `hitl_required` is true when: frame
count < 2; bias mean exceeds sensor pedestal or all bias frames saturated; dark
master too bright (wrong frame type) or darker than bias; flat median outside
sensor-relative 30–55% fill range or < 2× bias median; rejection rate > 40%.

---

#### T03 — `siril_calibrate`

**Purpose:** Apply master calibration frames to each light sub. Optionally
debayers CFA data and applies cosmetic correction (hot/cold pixel removal).

**Backend:** Siril CLI — `calibrate` command.

```
calibrate lights -bias=master_bias -dark=master_dark -flat=master_flat
         [-cfa] [-debayer] [-equalize_cfa] [-cc=dark siglo sighi] [-opt]
```

**Inputs:**

```json
{
  "working_dir": "str",
  "lights_sequence": "str",
  "master_bias": "str | null",
  "master_dark": "str | null",
  "master_flat": "str | null",
  "is_cfa": "bool",
  "debayer": "bool = true",
  "equalize_cfa": "bool = true",
  "cosmetic_correction": "bool = true",
  "cc_sigma": [3.0, 5.0],
  "optimize_dark": "bool = false"
}
```

**Outputs:**

```json
{
  "calibrated_sequence": "str",
  "calibrated_count": "int",
  "bad_pixel_count": "int"
}
```

**Agent notes:** Set `is_cfa = true` when `AcquisitionMeta.input_format == "raw"`
or when the input is an OSC/DSLR camera. For Fujifilm X-Trans sensors (e.g.
X-T30 II, X-T4), Siril handles the non-Bayer CFA pattern internally — set
`equalize_cfa = true` to prevent the color tinting that X-Trans sensors are
particularly prone to. Enable `optimize_dark` when dark frames were shot at a
slightly different temperature than lights.

---

#### T04 — `siril_register`

**Purpose:** Align calibrated lights to a common reference frame using star
matching. Supports homography, affine, and shift-only transformations with
optional drizzle for sub-pixel resolution.

**Backend:** Siril CLI — `register` (two-pass) + `seqapplyreg`.

```
register sequence -2pass [-transf=homography] [-maxstars=500]
seqapplyreg sequence [-drizzle] [-interp=lanczos4]
           [-filter-fwhm=90%] [-filter-round=90%]
```

**Inputs:**

```json
{
  "working_dir": "str",
  "calibrated_sequence": "str",
  "transformation": "homography | affine | shift",
  "interpolation": "lanczos4 | bicubic | bilinear | none",
  "max_stars": "int = 500",
  "drizzle": "bool = false",
  "drizzle_pixfrac": "float = 1.0",
  "filter_fwhm_pct": "float | null",
  "filter_round_pct": "float | null"
}
```

**Outputs:**

```json
{
  "registered_sequence": "str",
  "reference_image": "str",
  "registered_count": "int",
  "failed_count": "int",
  "metrics": {
    "avg_fwhm": "float",
    "avg_roundness": "float",
    "avg_star_count": "float"
  }
}
```

**Agent notes:** Use `homography` (default) for typical alt-az or equatorial
data. Use two-pass registration (`-2pass`) so Siril automatically picks the
best reference frame. Enable `drizzle` only when the dataset has ≥30 frames
with good dithering. Inline filtering via `filter_fwhm_pct` can replace the
separate `select_frames` step for simple rejection.

---

#### T05 — `analyze_frames`

**Purpose:** Compute per-frame quality metrics on a registered sequence. These
metrics feed the frame selection tool and give the agent statistical context.

**Backend:** Parses the `.seq` file directly (R-lines: FWHM, roundness, star
count; M-lines: mean, median, sigma, bgnoise). When star detection fails (short
exposures, sparse fields), computes Laplacian variance from the FITSEQ pixel
data as a sharpness proxy — no stars required.

**Inputs:**

```json
{
  "working_dir": "str",
  "registered_sequence": "str"
}
```

**Outputs:**

```json
{
  "frame_metrics": {
    "<frame_idx>": {
      "fwhm": "float | null",
      "weighted_fwhm": "float | null",
      "roundness": "float | null",
      "quality": "float | null",
      "background_lvl": "float | null",
      "number_of_stars": "int | null",
      "mean": "float | null",
      "median": "float | null",
      "sigma": "float | null",
      "bgnoise": "float | null",
      "laplacian_sharpness": "float | null"
    }
  },
  "summary": {
    "has_star_metrics": "bool",
    "median_fwhm": "float | null",
    "std_fwhm": "float | null",
    "median_roundness": "float | null",
    "median_laplacian_sharpness": "float | null",
    "best_frame": "str",
    "worst_frame": "str",
    "outlier_frames": ["str"]
  }
}
```

**Agent notes:** When `summary.has_star_metrics` is true, use standard
FWHM/roundness/star_count path in `select_frames`. When false (sub-second
exposures, sparse star field), pass `has_star_metrics=false` to T06 to use
Laplacian-percentile ranking instead. A high `std_fwhm` (when present) indicates
variable seeing — tighter rejection is warranted.

---

#### T06 — `select_frames`

**Purpose:** Accept or reject frames based on quality metrics. Two paths:
star-metrics (sigma-clipping on FWHM, roundness, star count, background) and
Laplacian-percentile (rank by sharpness, keep top N%) when star detection failed.

**Backend:** Python — threshold logic or percentile ranking on `analyze_frames` output.

**Inputs:**

```json
{
  "frame_metrics": "dict[str, dict]",
  "criteria": {
    "max_fwhm_sigma": "float = 2.0",
    "min_roundness": "float = 0.5",
    "min_star_count": "int = 30",
    "max_background_sigma": "float = 3.0",
    "keep_percentile": "float = 0.85"
  },
  "has_star_metrics": "bool = true"
}
```

**Outputs:**

```json
{
  "accepted_frames": ["str"],
  "rejected_frames": ["str"],
  "rejection_reasons": { "<filename>": "str" },
  "acceptance_rate": "float",
  "selection_path": "str",
  "warnings": ["str"]
}
```

**Agent notes:** Pass `summary.has_star_metrics` from T05. When false, T06 uses
Laplacian-percentile ranking (keeps best `keep_percentile` of frames by
sharpness; default 85%). When true, uses standard sigma-clipping. Metric keys
use `background_lvl` (from .seq R-lines) and `number_of_stars`. If
`acceptance_rate < 0.5`, warn and consider loosening. For small datasets
(< 15 subs), prefer `max_fwhm_sigma = 3.0`. Safety fallback: never returns
empty `accepted_frames`.

---

#### T07 — `siril_stack`

**Purpose:** Combine accepted registered frames into a single master light.
The result is a high-SNR linear image.

**Backend:** Siril CLI — `select` / `unselect` + `stack` commands.

Siril's `stack` operates on the currently *selected* frames in a sequence file
(`.seq`). The tool implementation must:

1. Parse the `.seq` file to build a `filename → frame_index` map.
2. Emit `unselect <seq> 0 <N-1>` to deselect every frame.
3. For each filename in `accepted_frames`, emit `select <seq> <idx> <idx>` to
   re-enable only those frames.
4. Run `stack` — Siril will include only the selected frames.

```
# Frame-exclusion preamble (generated per accepted_frames list)
unselect registered_lights 0 999
select registered_lights 0 0
select registered_lights 3 3
...

# Stacking command
stack registered_lights rej sigma_low sigma_high
      -norm=addscale -weight=wfwhm -out=master_light
```

**Inputs:**

```json
{
  "working_dir": "str",
  "registered_sequence": "str",
  "accepted_frames": ["str"],
  "stack_method": "mean | median",
  "rejection_method": "sigma_clipping | winsorized | linear_fit | none",
  "rejection_sigma": [3.0, 3.0],
  "normalization": "addscale | multiplicative | none",
  "weighting": "noise | wfwhm | nbstars | none",
  "output_32bit": "bool = true",
  "total_frames_hint": "int | null"
}
```

**Outputs:**

```json
{
  "master_light_path": "str",
  "stack_metrics": {
    "stacked_count": "int",
    "total_frames": "int",
    "acceptance_rate": "float",
    "total_integration_s": "float",
    "estimated_snr_gain": "float",
    "background_noise": "float"
  },
  "hitl_required": "bool",
  "hitl_context": "str"
}
```

**Agent notes:** Use `mean` + `sigma_clipping` for most datasets. Use `wfwhm`
weighting to favor sharper subs. Use `winsorized` rejection for small datasets
(< 15 frames) where sigma-clipping is unstable. Always stack in 32-bit float
to preserve dynamic range for subsequent linear processing. The `accepted_frames`
list comes directly from `select_frames` (T06); pass it unchanged so the
frame-index mapping step can correctly drive `select`/`unselect`. Pass
`total_frames_hint` from T05 `summary.frame_count` so acceptance rate is correct
for HITL. **HITL triggers:** `hitl_required` is true when accepted count < 2 or
acceptance rate < 15% with ≥ 5 total frames — entire session data may be unusable.

---

#### T08 — `auto_crop`

**Purpose:** Remove black borders introduced by registration (stacking
artifacts at image edges from frame misalignment).

**Backend:** Siril CLI — `crop` with computed geometry, or Python edge detection.

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "threshold": "float = 0.01"
}
```

**Outputs:**

```json
{
  "cropped_image_path": "str",
  "crop_geometry": { "x": "int", "y": "int", "w": "int", "h": "int" },
  "pixels_removed_pct": "float"
}
```

**Agent notes:** Always run after stacking. If `pixels_removed_pct > 15`,
registration may have poor overlap — the agent should note this in history.
If a tiny signal island causes the 5px inset to collapse geometry, the
implementation falls back to the non-inset bounding box to ensure valid crop
dimensions (`w > 0`, `h > 0`).

---

### LINEAR PROCESSING TOOLS

These tools operate on linear data (pixel values proportional to photon flux).
They MUST be applied before stretching.

---

#### T09 — `remove_gradient`

**Purpose:** Remove large-scale background gradients caused by light pollution,
moon glow, or vignetting residuals.

**Backend:** GraXpert CLI called directly via subprocess using `GRAXPERT_BIN`
(preferred — AI model needs no manual sample placement), or Siril CLI `subsky`
as fallback. Do NOT use Siril's `pyscript GraXpert-AI.py` wrapper — call
GraXpert directly so AstroAgent controls all arguments explicitly and has no
dependency on Siril's GraXpert preferences configuration.

```
# GraXpert (direct subprocess — preferred)
# Models: bge-ai-models/1.0.1/model.onnx (cached in ~/Library/Application Support/GraXpert/)
# CoreML acceleration active on Apple Silicon
$GRAXPERT_BIN <image> -cli -cmd background-extraction \
  -output <out.fits> \
  -correction <Subtraction|Division> \
  -smoothing <0.0–1.0> \
  -ai_version 1.0.1 \
  [-bg]          # also save the background model as a separate file

# Siril (fallback) — verified Siril 1.4 syntax:
subsky -rbf [-dither] -samples=25 -tolerance=1.0 -smooth=0.5
# or polynomial:
subsky <degree=4> [-dither] -samples=25 -tolerance=1.0
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "backend": "graxpert | siril",
  "graxpert_options": {
    "correction_type": "Subtraction | Division",
    "smoothing": "float = 0.5",
    "save_background_model": "bool = true",
    "ai_version": "str = 1.0.1"
  },
  "siril_options": {
    "model": "rbf | polynomial",
    "polynomial_degree": "int = 4",
    "samples_per_line": "int = 25",
    "tolerance": "float = 1.0",
    "smoothing": "float = 0.5",
    "dither": "bool = false"
  }
}
```

**Outputs:**

```json
{
  "processed_image_path": "str",
  "background_model_path": "str | null"
}
```

**Agent notes:** Prefer GraXpert AI for most cases — it handles complex
gradients without manual intervention. Use Siril `subsky` as fallback when
GraXpert is unavailable. Run `analyze_image` before and after to compare
`background_flatness` and validate improvement. Use `subtraction` correction
for additive light pollution gradients (most common). Use `division` only for
vignetting-like multiplicative gradients that survived flat correction.

---

#### T10 — `color_calibrate`

**Purpose:** Correct white balance by matching star colors to photometric
catalogs. Includes background neutralization (sets background to neutral
grey) and optionally spectrophotometric calibration.

**Backend:** Siril CLI — `pcc` or `spcc` (requires plate-solved image).

```
# PCC (coords are positional, must come before named flags)
platesolve [ra dec] -focal=<fl> -pixelsize=<px>
pcc -catalog=gaia [-limitmag=+2] [-bgtol=-2.8,2.0]

# SPCC (more accurate, models atmospheric extinction)
platesolve [ra dec] -focal=<fl> -pixelsize=<px>
spcc -oscsensor=<sensor> [-oscfilter=<filter>] [-atmos] [-limitmag=+2] [-bgtol=-2.8,2.0]
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "method": "pcc | spcc",
  "focal_length_mm": "float",
  "pixel_size_um": "float",
  "target_coords": { "ra": "float | null", "dec": "float | null" },
  "catalog": "nomad | apass | gaia | localgaia",
  "limitmag": "str | null  (e.g. '+2', '-1', '12' — adjusts star magnitude limit)",
  "bgtol_lower": "float | null  (background rejection lower sigma, default -2.8)",
  "bgtol_upper": "float | null  (background rejection upper sigma, default 2.0)",
  "spcc_options": {
    "sensor_name": "str | null",
    "filter_name": "str | null",
    "atmospheric_correction": "bool = false"
  }
}
```

**Outputs:**

```json
{
  "calibrated_image_path": "str",
  "plate_solve_success": "bool",
  "wcs_coords": { "ra": "float", "dec": "float" },
  "pixel_scale_arcsec": "float",
  "color_coefficients": { "r": "float", "g": "float", "b": "float" },
  "error_msg": "str | null"
}
```

**Agent notes:** This tool requires a plate-solved image. If plate solving
fails (no WCS solution), the agent should try: (1) provide approximate
coordinates from `acquisition_meta.target_name`, (2) increase limit magnitude,
(3) fall back to manual white balance via `pixel_math`. Prefer `spcc` over
`pcc` when the sensor model is known — it produces more accurate color. The
background neutralization included in PCC/SPCC sets all channels' background
medians equal, eliminating color casts in the sky background.

---

#### T11 — `remove_green_noise`

**Purpose:** Remove excess green signal common in one-shot-color (OSC) and DSLR
cameras due to the Bayer matrix having 2x green pixels.

**Backend:** Siril CLI — `rmgreen` (SCNR algorithm).

```
rmgreen 0 1.0    # type=average_neutral, amount=1.0, preserve lightness
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "protection_type": "average_neutral | maximum_neutral",
  "amount": "float = 1.0",
  "preserve_lightness": "bool = true"
}
```

**Outputs:**

```json
{
  "cleaned_image_path": "str"
}
```

**Agent notes:** Only apply to OSC/DSLR data (`metadata.is_osc == true`).
Check `metrics.green_excess` from `analyze_image` first — skip if green excess
is negligible. Must be applied in linear space (before stretch) for correct
color math. `amount = 1.0` is appropriate for most cases; reduce for targets
with genuinely green emission (rare in broadband).

---

#### T12 — `noise_reduction`

**Purpose:** Reduce noise while preserving signal structure. Best applied in
linear space where noise statistics are well-behaved (approximately Gaussian
after stacking). A milder second pass may optionally be applied post-stretch.

**Backend:** Siril CLI — `denoise` (NL-Bayes algorithm, optionally with DA3D
or SOS boosting), or GraXpert CLI denoising.

```
# Siril (primary) — verified Siril 1.4 syntax:
denoise [-mod=0.9] [-da3d | -sos=3] [-vst] [-indep] [-nocosmetic]

# GraXpert (alternative)
# Models: denoise-ai-models/2.0.0/model.onnx (cached in ~/Library/Application Support/GraXpert/)
# Strength is not a direct CLI flag — controlled via a temp preferences JSON
# passed with -preferences_file. Default strength = 0.5 (0.0–1.0).
$GRAXPERT_BIN <image> -cli -cmd denoising \
  -output <out.fits> \
  -ai_version 2.0.0 \
  -preferences_file <temp_prefs.json>
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "backend": "siril | graxpert",
  "siril_options": {
    "modulation": "float = 0.9",
    "method": "standard | da3d | sos",
    "sos_iterations": "int = 3",
    "independent_channels": "bool = false",
    "use_vst": "bool = false  (Anscombe variance-stabilizing transform; standard method only)",
    "apply_cosmetic": "bool = true  (false maps to -nocosmetic flag)"
  },
  "graxpert_options": {
    "strength": "float = 0.5",
    "ai_version": "str = 2.0.0"
  }
}
```

**Outputs:**

```json
{
  "denoised_image_path": "str",
  "noise_before": "float",
  "noise_after": "float",
  "noise_reduction_pct": "float"
}
```

**Agent notes:** This tool was absent from the original spec and is one of the
most critical steps in astrophotography processing. Noise reduction in linear
space prevents noise from being amplified by the stretch. Use `modulation < 1.0`
to blend denoised and original data (e.g. `0.8` retains 20% original detail).
Use `da3d` for best detail preservation at the cost of extra processing time.
After stretching, a second mild pass may be warranted on the starless image
only (see non-linear processing). Run `analyze_image` before and after to
verify noise reduction without signal loss.

---

#### T13 — `deconvolution`

**Purpose:** Sharpen the image by reversing atmospheric and optical blur
(deconvolution). Restores detail lost to seeing and diffraction.

**Backend:** Siril CLI — `rl` (Richardson-Lucy) or `wiener` (Wiener
deconvolution). PSF estimated via `makepsf` or loaded from star analysis.

```
# PSF from detected stars (preferred — most accurate for astrophotography)
makepsf stars [-sym]
rl -iters=15 -tv -alpha=3000 [-stop=1e-5]

# Blind PSF fallback (use when star count is low)
makepsf blind
rl -iters=10 -tv -alpha=3000

# Manual PSF — Moffat (accurate for atmospheric seeing, preferred over Gaussian)
makepsf manual -moffat -fwhm=2.5 -beta=3.5
rl -iters=15 -tv -alpha=3000

# Manual PSF — Airy (theoretical diffraction limit for known telescope optics)
makepsf manual -airy -dia=130 -fl=910 -wl=525 -pixelsize=3.77
rl -iters=15 -fh -alpha=3000

# Wiener alternative
makepsf stars
wiener -alpha=0.001
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "method": "richardson_lucy | wiener",
  "psf_source": "stars | blind | manual | from_file",
  "psf_file": "str | null",
  "makepsf_manual_options": {
    "profile": "moffat | gaussian | airy | disc",
    "fwhm_px": "float | null  (measure from analyze_image star metrics)",
    "moffat_beta": "float = 3.5  (wing shape; 2.0 poor seeing – 5.0 excellent)",
    "aspect_ratio": "float = 1.0  (1.0 = circular; <1.0 = elongated)",
    "angle_deg": "float = 0.0",
    "airy_diameter_mm": "float | null  (telescope aperture)",
    "airy_focal_length_mm": "float | null",
    "airy_wavelength_nm": "float = 525.0",
    "airy_obstruction_pct": "float = 0.0",
    "symmetric": "bool = false  (force circular PSF from stars: makepsf stars -sym)"
  },
  "rl_options": {
    "iterations": "int = 10",
    "regularization": "total_variation | hessian_frobenius | none",
    "alpha": "float = 3000  (lower = less regularization = sharper but more ringing)",
    "stop": "float | null  (early stopping residual threshold, e.g. 1e-5)",
    "use_multiplicative": "bool = false"
  },
  "wiener_options": {
    "alpha": "float = 0.001"
  }
}
```

**Outputs:**

```json
{
  "sharpened_image_path": "str",
  "psf_source": "str",
  "psf_profile": "str | null",
  "psf_fwhm_used": "float",
  "regularization": "str",
  "stop_criterion": "float | null"
}
```

**Agent notes:** Deconvolution is optional and should only be attempted when
SNR is adequate (`snr_estimate > 50`). Over-deconvolution creates ringing
artifacts. Richardson-Lucy with `regularization=total_variation` (`-tv`) is the
safest choice. `hessian_frobenius` (`-fh`) preserves sharp linear features
(galaxy spiral arms, nebula filaments) better than TV, but with higher ringing
risk — use only at high SNR. Use `stop=1e-5` to prevent over-iterating in
automation. The `manual` PSF source with `profile=moffat` is preferred when
the star field is sparse or when you know the seeing FWHM from `analyze_image`
star metrics. Use `profile=airy` only when telescope parameters are precisely
known and seeing is excellent. Must be done in linear space.

---

### THE CROSSING

---

#### T14 — `stretch_image`

**Purpose:** Transform the image from linear to non-linear (perceptual)
brightness space. This is the single most impactful step — it makes the
invisible signal visible.

**Backend:** Siril CLI — `ght` (Generalized Hyperbolic Stretch), `asinh`
(arcsinh), or `autostretch` (histogram-based auto-stretch).

```
# GHS
ght -D=2.5 -B=0 -LP=0 -SP=0.0 -HP=1.0 -human

# Arcsinh
asinh -human 100 0.0

# Auto-stretch
autostretch -linked -2.8 0.25
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "method": "ghs | asinh | autostretch",
  "ghs_options": {
    "stretch_amount": "float",
    "local_intensity": "float = 0.0",
    "symmetry_point": "float = 0.0",
    "shadow_protection": "float = 0.0",
    "highlight_protection": "float = 1.0",
    "color_model": "human | even | independent"
  },
  "asinh_options": {
    "stretch_factor": "float = 100",
    "black_point_offset": "float = 0.0",
    "color_model": "human | default"
  },
  "autostretch_options": {
    "shadows_clipping_sigma": "float = -2.8",
    "target_background": "float = 0.25",
    "linked": "bool = true"
  }
}
```

**Outputs:**

```json
{
  "stretched_image_path": "str",
  "statistics": {
    "median_brightness": "float",
    "mean_brightness": "float",
    "clipped_shadows_pct": "float",
    "clipped_highlights_pct": "float"
  }
}
```

**Agent notes:** This is a HITL checkpoint — the agent should generate 2-3
variants with different stretch intensities (e.g. "gentle" D=1.5, "moderate"
D=2.5, "aggressive" D=4.0) and present them for user selection. Use `human`
color model to preserve color ratios. GHS is preferred for its fine control;
`autostretch` is a safe fallback. After stretch, set `metadata.is_linear = false`.
Key metrics to monitor: `clipped_shadows_pct` should be < 0.5% (preserve faint
nebulosity), `clipped_highlights_pct` < 0.1% (avoid burned stars). Always use
linked stretch after color calibration to preserve white balance.

---

### NON-LINEAR PROCESSING TOOLS

These tools operate on stretched (non-linear) data. Many of these steps involve
subjective aesthetic decisions — the agent should leverage HITL checkpoints.

---

#### T15 — `star_removal`

**Purpose:** Separate stars from extended objects (nebulae, galaxies) using
neural network inference (StarNet). Produces a starless image and a star mask.
This enables independent processing of nebulosity and stars.

**Backend:** StarNet2 (PyTorch/MPS) called directly via subprocess using
`STARNET_BIN`. Do NOT use Siril's `starnet` wrapper — call the binary directly
so AstroAgent controls all arguments and has no dependency on Siril's StarNet
preferences configuration.

The experimental PyTorch/MPS build (`starnet2-mps`) uses named flags and
outputs both starless image and star mask in a single call:

```
$STARNET_BIN -i <input> -o <starless_out> -m <mask_out> -w <weights_path> [-u]
```

Note: binary requires `install_name_tool -add_rpath <lib_dir>` and
`codesign --force --sign -` after initial install. See plan.md T15 note.

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "upscale": "bool = false",
  "generate_star_mask": "bool = true"
}
```

**Outputs:**

```json
{
  "starless_image_path": "str",
  "star_mask_path": "str | null"
}
```

**Agent notes:** Run after initial stretch. The starless image becomes the
working canvas for curves, contrast, and saturation — preventing star bloat
during aggressive processing. The star mask is recombined later via
`star_restoration`. StarNet v2 is a required dependency; the startup check
will abort with an install message if it is missing. Use `-upscale` only if
stars are very tight (small FWHM) and being poorly detected.

---

#### T16 — `curves_adjust`

**Purpose:** Fine-tune brightness, contrast, and per-channel balance using
midtone transfer function (MTF) or targeted GHT. Replaces traditional
curves/levels tools.

**Backend:** Siril CLI — `mtf` (midtone transfer function) or `ght` with
channel-specific parameters.

```
# MTF — set black point, midtone, white point
mtf 0.0 0.35 1.0

# GHT — targeted stretch on specific brightness range
ght -D=1.5 -B=5 -SP=0.3 -LP=0.05 -HP=0.9 -human
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "method": "mtf | ght",
  "mtf_options": {
    "black_point": "float = 0.0",
    "midtone": "float = 0.5",
    "white_point": "float = 1.0",
    "channels": "all | R | G | B"
  },
  "ght_options": {
    "stretch_amount": "float",
    "local_intensity": "float = 5.0",
    "symmetry_point": "float = 0.3",
    "shadow_protection": "float = 0.05",
    "highlight_protection": "float = 0.9",
    "channels": "all | R | G | B | RG | RB | GB"
  }
}
```

**Outputs:**

```json
{
  "adjusted_image_path": "str",
  "statistics": {
    "median_brightness": "float",
    "contrast_ratio": "float"
  }
}
```

**Agent notes:** Apply to the starless image to avoid star bloat. Use MTF for
global brightness/contrast adjustment. Use GHT for targeted adjustments to
specific brightness ranges (e.g. boosting faint nebulosity without blowing out
bright cores). Set `local_intensity (B)` high (5-15) to focus the stretch
tightly around `symmetry_point`. Per-channel adjustments can correct residual
color casts. The agent may apply multiple passes with different settings.

---

#### T17 — `local_contrast_enhance`

**Purpose:** Enhance fine detail and local contrast in nebulosity and galaxy
structure without affecting global brightness. Four methods, each optimal
for different enhancement goals.

**Backend:** Siril CLI — `epf` (edge-preserving bilateral/guided filter),
`clahe` (adaptive histogram equalization), `unsharp` (unsharp mask), or
`wavelet` + `wrecons` (multi-scale wavelet sharpening).

```
# Edge-preserving bilateral filter (structure-safe noise smoothing)
# For 32-bit FITS: si/ss should be 0–1 range
epf [-guided] [-d=5] -si=0.02 -ss=0.02 -mod=0.8 [-guideimage=<stem>]

# CLAHE
clahe 2.0 8

# Unsharp mask
unsharp 2.0 0.3

# Wavelet sharpening
wavelet 5 2
wrecons 1.2 1.1 1.0 1.0 1.0 1.0
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "method": "edge_preserve | clahe | unsharp | wavelet",
  "epf_options": {
    "guided": "bool = false  (false=bilateral, true=guided filter)",
    "diameter": "int = 5  (filter kernel size; 0=auto from spatial_sigma)",
    "intensity_sigma": "float = 0.02  (0–1 range for 32-bit FITS; tonal smoothing radius)",
    "spatial_sigma": "float = 0.02  (0–1 range for 32-bit FITS; spatial smoothing radius)",
    "mod": "float = 0.8  (blend: 1.0=full filter, 0.0=no effect)",
    "guide_image_stem": "str | null  (guided filter only; null=self-guided)"
  },
  "clahe_options": {
    "clip_limit": "float = 2.0",
    "tile_size": "int = 8"
  },
  "unsharp_options": {
    "sigma": "float = 2.0",
    "amount": "float = 0.3"
  },
  "wavelet_options": {
    "num_layers": "int = 5",
    "algorithm": "linear | bspline",
    "layer_weights": ["float"]
  }
}
```

**Outputs:**

```json
{
  "enhanced_image_path": "str",
  "method": "str"
}
```

**Agent notes:** Apply to starless image only.

`edge_preserve` (epf) is the recommended first pass: the bilateral filter
smooths noise in flat areas while respecting sharp edges — this is
PixInsight-tier structure-safe processing. Excellent for cleaning residual
noise in background/faint nebula regions after NL-Bayes. Use `mod=0.6–0.9`
for blended effect. Self-guided (`guided=False`) works for most images.

CLAHE is effective for revealing faint structure but amplifies noise — always
apply after noise reduction and optionally after edge_preserve.

Wavelet sharpening gives surgical per-scale control: boost fine-scale layers
(1-2) while leaving coarse layers (3+) at 1.0. Start conservative (1.1–1.3×).

---

#### T18 — `saturation_adjust`

**Purpose:** Enhance or reduce color saturation, optionally targeting specific
hue ranges while protecting background noise from color amplification.

**Backend:** Siril CLI — `satu` (saturation with background protection and
hue targeting) or `ght -sat` (GHT applied to saturation channel).

```
# Global saturation boost with background protection
satu 0.5 1.5

# Targeted saturation (e.g., boost reds for Ha emission, index 0)
satu 0.8 1.5 0
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "method": "global | hue_targeted | ght_saturation",
  "amount": "float",
  "background_factor": "float = 1.5",
  "hue_target": "int | null",
  "ght_sat_options": {
    "stretch_amount": "float",
    "symmetry_point": "float = 0.5"
  }
}
```

Hue target indices: `0` = pink-orange (Ha), `1` = orange-yellow, `2` =
yellow-cyan, `3` = cyan (OIII), `4` = cyan-magenta, `5` = magenta-pink,
`6` = all (default).

**Outputs:**

```json
{
  "saturated_image_path": "str"
}
```

**Agent notes:** Apply to starless image to prevent star color bloat.
Use `background_factor > 1.0` to protect noise in dark sky from being
color-boosted. For emission nebulae, target hue index `0` (Ha reds) and `3`
(OIII blues/cyan) separately. Apply saturation in multiple small increments
rather than one large boost. This is a highly subjective step — always follow
with a preview for HITL evaluation. The `ght -sat` method is useful for
boosting saturation only in the midtones while protecting already-saturated
bright structures.

---

#### T19 — `star_restoration`

**Purpose:** Recombine processed starless image with stars. Offers two modes:
(1) blend original star mask back at adjustable intensity, or (2) replace
with synthetic perfect stars via Siril's `synthstar` engine.

**Backend:** Siril CLI — `pm` (pixel math for weighted blend) or `synthstar`
(synthetic star generation from PSF analysis).

```
# Pixel math recombination (stars at 80% weight)
pm "$starless$ + $starmask$ * 0.8"

# Synthetic perfect stars (run on original pre-starnet image)
synthstar
```

**Inputs:**

```json
{
  "working_dir": "str",
  "starless_image_path": "str",
  "mode": "blend | synthstar",
  "blend_options": {
    "star_mask_path": "str",
    "star_weight": "float = 1.0"
  },
  "synthstar_options": {
    "source_image_path": "str"
  }
}
```

**Outputs:**

```json
{
  "final_image_path": "str"
}
```

**Agent notes:** `blend` mode with `star_weight < 1.0` effectively reduces
star prominence — this replaces dedicated morphological star reduction.
`synthstar` mode creates perfectly round, well-profiled stars from the original
PSF measurements. Use `synthstar` when the original stars have coma, trailing,
or other optical defects. `synthstar` should be run on the linear image
BEFORE stretching for best results, so the agent must plan ahead if this mode
is desired (running starnet on the stretched image for nebulosity processing
but synthstar on the linear image for star replacement). For typical
workflows, `blend` mode with `star_weight = 0.7 - 1.0` works well.

---

### UTILITY TOOLS

Available at any processing phase.

---

#### T20 — `analyze_image`

**Purpose:** The agent's primary diagnostic instrument. Returns comprehensive
statistics, per-channel analysis, noise estimates, star metrics, and
background characterization. The agent calls this before and after processing
steps to evaluate results and decide next actions.

Input sanitation: if FITS contains NaN/Inf pixel values, they are replaced with
a robust finite fill value before metric computation so outputs remain finite
and deterministic.

**Backend:** Siril CLI — `stat`, `bgnoise`, `histo` + Python/Photutils for
star detection and advanced metrics.

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "detect_stars": "bool = true",
  "compute_histogram": "bool = true"
}
```

**Outputs:**

```json
{
  "channels": {
    "red":   { "min": "float", "max": "float", "mean": "float",
               "median": "float", "std": "float" },
    "green": { "min": "float", "max": "float", "mean": "float",
               "median": "float", "std": "float" },
    "blue":  { "min": "float", "max": "float", "mean": "float",
               "median": "float", "std": "float" }
  },
  "noise": {
    "background_noise": "float",
    "method": "siril_bgnoise_MAD"
  },
  "background": {
    "median": "float",
    "flatness_score": "float",
    "gradient_magnitude": "float",
    "per_channel_bg": {
      "red": "float", "green": "float", "blue": "float",
      "n_background_pixels": "int"
    }
  },
  "stars": {
    "count": "int",
    "median_fwhm": "float",
    "median_roundness": "float",
    "fwhm_std": "float",
    "median_star_peak_ratio": "float"
  },
  "dynamic_range_db": "float",
  "snr_estimate": "float",
  "linearity": {
    "is_linear": "bool",
    "confidence": "high | medium | low",
    "histogram_skewness": "float",
    "median_brightness": "float"
  },
  "signal_coverage_pct": "float",
  "clipping": {
    "shadows_pct": "float",
    "highlights_pct": "float"
  },
  "color_balance": {
    "green_excess": "float",
    "channel_imbalance": "float"
  },
  "color": {
    "mean_saturation": "float",
    "median_saturation": "float",
    "high_saturation_pct": "float"
  },
  "histogram": {
    "red":   { "p1": "float", "p5": "float", "p25": "float", "p50": "float",
               "p75": "float", "p95": "float", "p99": "float" },
    "green": { "..." },
    "blue":  { "..." }
  },
  "image_shape": ["int"],
  "is_color": "bool"
}
```

**Agent decision map:**

| Metric                                              | Threshold               | Implies                                                               |
| --------------------------------------------------- | ----------------------- | --------------------------------------------------------------------- |
| `background.gradient_magnitude > 0.05`              | High gradient           | Run `remove_gradient`                                                 |
| `background.flatness_score < 0.9`                   | Uneven sky              | Re-run gradient removal or increase polynomial degree                 |
| `background.per_channel_bg` — spread > 0.02         | Unbalanced sky color    | Re-run `color_calibrate` or apply manual `pixel_math` neutralization  |
| `color_balance.green_excess > 0.02`                 | Green cast              | Run `remove_green_noise`                                              |
| `snr_estimate < 30`                                 | Low SNR                 | Skip or defer `deconvolution`                                         |
| `stars.median_fwhm > 4.0 px` (pre-stretch)          | Bloated PSF             | Investigate registration / optics                                     |
| `stars.fwhm_std > 1.5 px`                           | Non-uniform PSF         | Use `makepsf manual` (Moffat) with measured FWHM for T13              |
| `stars.median_star_peak_ratio > 50`                 | Star-dominated frame    | Run `reduce_stars` (T26) post-stretch to balance star/nebula ratio    |
| `clipping.shadows_pct > 1.0`                        | Over-clipped blacks     | Reduce stretch aggressiveness; adjust `black_point`                   |
| `linearity.is_linear` (high confidence)             | Linear image            | Only linear-phase tools (T09–T13) are valid                           |
| `linearity.is_linear = False` (high confidence)     | Stretched image         | Only non-linear tools (T14–T19, T25–T27) are valid                   |
| `signal_coverage_pct < 5`                           | Sparse target           | Conservative T09 (low polynomial degree); light T17/T27 passes       |
| `signal_coverage_pct > 30`                          | Nebula/Milky Way fill   | T09 risk of nebula subtraction is higher; use GraXpert AI model       |
| `color.mean_saturation < 0.10` (post-stretch)       | Desaturated colors      | Run `saturation_adjust` (T18); increase boost significantly            |
| `color.mean_saturation > 0.40` (post-boost)         | Risk of over-saturation | Reduce T18 boost; protect background with mask                       |
| `noise.background_noise` increased after step       | Processing degraded SNR | Revert or adjust tool parameters                                      |
| Any step where global effect harms a region         | Spatial conflict        | Run `create_mask` (T25) then blend via `pixel_math` (T23)             |
| Nebula detail lacking after `curves_adjust`         | Needs targeted sharpness| Run `multiscale_process` (T27): sharpen scales 2–3 with nebula mask   |
| Noise visible in dark sky after processing          | Background noise amp    | Run `multiscale_process` (T27): denoise scales 1–2 with inv-lum mask  |


---

#### T21 — `plate_solve`

**Purpose:** Determine the precise celestial coordinates of the image center
and the pixel scale (arcsec/pixel) via astrometric plate solving. Required
before PCC/SPCC and useful for target identification.

**Backend:** Siril CLI — `platesolve`.

```
platesolve ra,dec -focal=FL -pixelsize=PX
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "approximate_coords": { "ra": "float | null", "dec": "float | null" },
  "focal_length_mm": "float | null",
  "pixel_size_um": "float | null",
  "force_resolve": "bool = false",
  "catalog": "auto | tycho2 | nomad | gaia | localgaia",
  "use_local_astrometry_net": "bool = false"
}
```

**Outputs:**

```json
{
  "success": "bool",
  "wcs_coords": { "ra": "float", "dec": "float" },
  "pixel_scale_arcsec": "float",
  "field_of_view": { "width_arcmin": "float", "height_arcmin": "float" },
  "rotation_deg": "float",
  "error_msg": "str | null"
}
```

**Agent notes:** Called automatically before `color_calibrate`. If approximate
coords are unknown, the agent can try blind solving with local astrometry.net
(`use_local_astrometry_net = true`). The resulting `pixel_scale_arcsec` is
critical for deconvolution PSF sizing.

---

#### T22 — `generate_preview`

**Purpose:** Create a JPG/PNG preview for the LLM's vision capabilities or
HITL display. For linear images, applies a temporary auto-stretch for
visualization without modifying the FITS data.

**Backend:** Siril CLI — `autostretch` + `savejpg` (on a copy), or Python PIL.

**Inputs:**

```json
{
  "working_dir": "str",
  "fits_path": "str",
  "width": "int = 1920",
  "format": "jpg | png",
  "auto_stretch_linear": "bool = true",
  "quality": "int = 95",
  "annotation": "str | null"
}
```

**Outputs:**

```json
{
  "preview_path": "str",
  "is_auto_stretched": "bool"
}
```

**Internal use only.** This is not a `@tool`-decorated function and must not
appear in any `PHASE_TOOLS` list. It is called exclusively by:
  1. `auto_hitl_check()` in the tool executor post-hook (after tools with
     `requires_visual_review=True`).
  2. The mandatory HITL nodes (`stretch_hitl`, `final_hitl`).

For linear images, always set `auto_stretch_linear = true` — otherwise the
preview will appear black. Use `annotation` to label variants for HITL
comparison (e.g. "Variant A: Gentle Stretch"). Preview generation must never
happen in the planner path — the planner's context is text-only metrics.

---

#### T23 — `pixel_math`

**Purpose:** General-purpose pixel-level mathematical operations. Used for
blending, masking, channel manipulation, and any operation not covered by
dedicated tools.

**Backend:** Siril CLI — `pm` (PixelMath).

```
pm "$image1$ * 0.7 + $image2$ * 0.3" -rescale 0 1
```

**Inputs:**

```json
{
  "working_dir": "str",
  "expression": "str",
  "rescale": "bool = false",
  "rescale_range": [0.0, 1.0]
}
```

All image variables in the expression must reference filenames (without
extension) present in the working directory, delimited by `$`.

**Outputs:**

```json
{
  "result_image_path": "str"
}
```

**Agent notes:** Pixel math is the agent's escape hatch for operations not
covered by other tools. Common uses: weighted star recombination, luminance
mask creation, channel swapping, HDR blending of multiple stretch variants.
The agent should validate the expression syntax before execution.

---

#### T24 — `export_final`

**Purpose:** Export the finished image in a distribution-ready format with
appropriate bit depth and ICC color profile.

**Backend:** Siril CLI — `savetif` / `savetif8` / `savejpg` / `savepng` +
`icc_convert_to`.

```
icc_convert_to sRGB perceptual
savetif result_16bit -deflate
savejpg result_web 95
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "formats": [
    {
      "type": "tiff16 | tiff8 | jpg | png",
      "quality": "int = 95",
      "icc_profile": "sRGB | sRGBlinear | Rec2020 | Rec2020linear | null",
      "filename_suffix": "str = ''"
    }
  ]
}
```

**Outputs:**

```json
{
  "exported_files": [
    {
      "path": "str",
      "format": "str",
      "file_size_mb": "float"
    }
  ]
}
```

**Agent notes:** Always produce at least two exports: a 16-bit TIFF (archival
master, AdobeRGB for maximum gamut) and a JPG (web sharing, sRGB for correct
display on consumer screens). Use deflate compression on TIFF to save disk
space without quality loss.

---

### SCIKIT-IMAGE TOOLS

These three tools are implemented entirely in Python using scikit-image,
PyWavelets, and Astropy. They do not invoke Siril. All I/O is via
`astropy.io.fits` so the FITS working files remain the single source of truth.
They are the primary mechanism for achieving PixInsight-parity quality in
targeted, region-specific processing.

**Masked-application pattern:** T25 produces a mask that any subsequent tool
output can be blended through via T23 `pixel_math`. This three-step sequence
is the standard way to apply any operation to a specific tonal region:

```
1. [T25 create_mask]            → mask.fits
2. [processing tool, any T]     → processed.fits  (applied globally)
3. [T23 pixel_math]  "$processed$ * $mask$ + $original$ * (1 - $mask$)"
                                → targeted_result.fits
```

The agent should default to this pattern whenever a global application would
degrade regions outside the intended target (e.g. noise reduction that also
softens stars, saturation that boosts the dark sky background, sharpening that
amplifies background noise).

---

#### T25 — `create_mask`

**Purpose:** Generate a single-channel FITS mask (pixel values 0.0–1.0) from
the working image based on luminance range, tonal window, or channel
difference. Masks confine subsequent processing to specific image regions —
protecting star cores during noise reduction, isolating faint nebulosity for
curves adjustments, shielding the dark sky from saturation boosts. This tool
is the foundational prerequisite for all targeted processing in the pipeline:
without masks, every tool applies globally and risks degrading regions it
should not touch.

**Backend:** Python — scikit-image + Astropy.

```python
# Core API calls
from astropy.io.fits import open as fits_open, writeto as fits_write
from skimage.color import rgb2gray                  # luminance: R×0.2126 + G×0.7152 + B×0.0722
from skimage.filters import gaussian                # feathering — smooths mask edges
from skimage.exposure import rescale_intensity      # normalize output to [0, 1]
from skimage.morphology import disk, binary_dilation, binary_erosion
import numpy as np
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "mask_type": "luminance | inverted_luminance | range | channel_diff",
  "luminance_options": {
    "low": "float = 0.0",
    "high": "float = 1.0"
  },
  "range_options": {
    "low": "float",
    "high": "float",
    "channel": "luminance | R | G | B"
  },
  "channel_diff_options": {
    "channel_a": "R | G | B",
    "channel_b": "R | G | B",
    "threshold": "float = 0.01"
  },
  "feather_radius": "float = 5.0",
  "expand_px": "int = 0",
  "contract_px": "int = 0",
  "invert": "bool = false",
  "output_stem": "str | null"
}
```

**Outputs:**

```json
{
  "mask_path": "str",
  "coverage_pct": "float",
  "mean_value": "float"
}
```

**Mask type guide:**

| `mask_type`           | What the mask selects          | Primary use case                              |
| --------------------- | ------------------------------ | --------------------------------------------- |
| `luminance`           | Bright pixels = 1, dark = 0    | Protect stars; target bright nebula cores     |
| `inverted_luminance`  | Dark pixels = 1, bright = 0    | Target faint nebulosity; protect sky from sat |
| `range`               | Pixels within `[low, high]`    | Target midtone nebulosity only                |
| `channel_diff`        | Where channel_a > channel_b    | Isolate Ha emission (R > B), OIII (B > R)     |

**Agent notes:**

- `feather_radius`: always use > 0 to prevent hard-edged banding in the
  blended result. 5px is conservative and appropriate for most masks. Use
  15–30px for large-scale sky/nebula boundary masks. Implemented via
  `skimage.filters.gaussian(mask, sigma=feather_radius)`.
- `expand_px` / `contract_px`: morphologically dilates or erodes the binary
  mask before feathering. Use `expand_px=5` on a star luminance mask to
  ensure halos are included. Use `contract_px=3` to tighten a mask that is
  bleeding into surrounding regions. Implemented via
  `skimage.morphology.binary_dilation(mask, disk(expand_px))`.
- `range` with `low=0.2, high=0.7` isolates midtone nebulosity — the
  most commonly targeted region for curves and saturation work.
- `luminance` with `low=0.7, high=1.0` isolates bright stars — useful before
  noise reduction to keep stars crisp while smoothing the background.
- `channel_diff` with `channel_a=R, channel_b=B, threshold=0.05` identifies
  Ha-bright emission regions for targeted red saturation.
- After generation, use the mask in T23: `"$processed$ * $mask$ + $original$ * (1 - $mask$)"`

---

#### T26 — `reduce_stars`

**Purpose:** Reduce the angular size (disk diameter) of stars in the image
using morphological erosion applied only within star regions. Unlike T19
`star_restoration` with `star_weight < 1.0` — which dims stars without
changing their size — this tool *physically shrinks* star disks, eliminating
bloom and improving the perceived sharpness of the background nebulosity.
Should be applied to the final combined (stars + starless) image after T19,
or directly before export. Not a substitute for good registration or optical
correction; intended for residual star size reduction after all other
processing is complete.

**Backend:** Python — scikit-image + Astropy.

```python
from astropy.io.fits import open as fits_open, writeto as fits_write
from skimage.morphology import disk, erosion, dilation   # morphological ops
from skimage.filters import gaussian                      # feather transition zone
import numpy as np
```

The algorithm:
1. Build a binary star mask — either from `star_mask_path` (T15 output) or
   auto-generated by thresholding the luminance channel at `detection_threshold`.
2. Optionally detect local star peaks and dilate those peak points by
   `protect_core_radius` to exclude only star cores from erosion
   (preserves core color while still shrinking outer halos).
3. Apply `skimage.morphology.erosion(channel, disk(kernel_radius))` for
   `iterations` passes, **only within the masked star region**.
4. Blend the eroded result with the original via feathered mask and
   `blend_amount` weighting.

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "star_mask_path": "str | null",
  "detection_threshold": "float = 0.6",
  "kernel_radius": "int = 1",
  "iterations": "int = 1",
  "blend_amount": "float = 1.0",
  "protect_core_radius": "int = 0",
  "feather_px": "int = 3"
}
```

**Outputs:**

```json
{
  "reduced_image_path": "str",
  "stars_affected_count": "int",
  "mean_size_reduction_pct": "float"
}
```

**Agent notes:**

- `kernel_radius=1, iterations=1` is the gentlest setting. For visibly
  bloated stars, try `kernel_radius=2` or `iterations=2`. Do not exceed
  `iterations=3` without previewing — stars become pixelated and
  structurally wrong.
- `star_mask_path`: always pass the mask from T15 when available. It is more
  precise than threshold-based detection and prevents erosion from touching
  bright nebula knots that are misidentified as stars.
- `detection_threshold`: luminance value above which a pixel is treated as a
  star. 0.6 works for typical stretched images. Reduce to 0.4 if faint stars
  are not being caught; increase to 0.75 if bright nebula cores are being
  incorrectly eroded.
- `protect_core_radius > 0`: applies `skimage.morphology.dilation` to create
  an exclusion zone around each detected star peak. Use when stars contain important
  color data (red giant cores, double star color pairs) that should not be
  eroded.
- `mean_size_reduction_pct` is computed from star-area shrinkage in the star
  region mask (pixel count before/after threshold), not from global brightness.
- `blend_amount < 1.0`: blends the eroded result with the original at the
  given weight. `0.5` gives a half-strength reduction useful when stars are
  only mildly large and you want a subtle effect.
- `feather_px`: width of the Gaussian transition between the eroded region and
  the surrounding image. Prevents visible edge rings around processed stars.
- Star reduction can be overdone and is difficult to undo. The
  `requires_visual_review` flag for this tool is `False` by default (it lacks
  the spatial-artifact signature of deconvolution or star removal), but can
  be flipped to `True` in `TOOL_REGISTRY` if real-data runs show it warrants
  a visual checkpoint.

---

#### T27 — `multiscale_process`

**Purpose:** Decompose the image into discrete spatial frequency scales via the
à trous (undecimated, shift-invariant) wavelet transform, apply independent
operations per scale — sharpen, denoise, suppress, or pass through — then
reconstruct. This is the open-source equivalent of PixInsight's Multiscale
Linear Transform (MLT). It provides surgical control that Siril's `wavelet` +
`wrecons` cannot match: boost fine nebula filaments at scale 2 while
simultaneously suppressing grain at scale 1, leave large-scale gradients at
scales 4–5 untouched, and optionally confine all of this to a masked region
via T25. The undecimated transform is used (not the standard decimated DWT)
because it preserves spatial alignment across scales, which is critical for
artifact-free astrophotography processing.

**Backend:** Python — SciPy + Astropy.

```python
from scipy.ndimage import convolve1d                         # B3-spline à trous kernel
from astropy.io.fits import open as fits_open, writeto as fits_write
import numpy as np
# B3-spline kernel: [1/16, 4/16, 6/16, 4/16, 1/16]
# Applied separably at each scale with stride=2^i (à trous = "with holes")
```

The algorithm:
1. Load FITS as float32 array. If color, operate on luminance channel only
   (unless `per_channel=True`).
2. Compute the B3-spline à trous wavelet decomposition using
   `scipy.ndimage.convolve1d` with the B3 spline kernel `[1/16, 4/16, 6/16, 4/16, 1/16]`
   applied separably at each scale with inter-scale hole-filling (stride=2^i).
   Each level yields an isotropic detail layer and a residual approximation.
   Note: PyWavelets does NOT have a `'b3'` wavelet — use the custom B3 atrous
   implementation via `scipy.ndimage.convolve1d`, not `pywt.swt2`.
3. Per scale, apply the specified operation to the detail coefficients:
   - `sharpen`: multiply coefficients by `weight` (e.g. 1.3 = 30% boost).
   - `denoise`: apply MAD-normalized soft thresholding: threshold is computed
     as `denoise_sigma * median(|coeffs|) / 0.6745`, then coefficients are
     soft-thresholded: `sign(c) * max(|c| - threshold, 0)`.
   - `suppress`: zero the coefficients (remove this scale's contribution).
   - `passthrough`: leave coefficients unchanged.
4. Reconstruct by summing all detail layers plus the final approximation residual.
5. If `mask_path` supplied: blend reconstructed result with original image
   via T25-style mask so operations are confined to the masked region.
   If mask and image dimensions differ, the mask is resized to image shape first.
6. Write result as FITS.

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str",
  "num_scales": "int = 5",
  "scale_operations": [
    {
      "scale": "int",
      "operation": "sharpen | denoise | suppress | passthrough",
      "weight": "float = 1.0",
      "denoise_sigma": "float | null"
    }
  ],
  "mask_path": "str | null",
  "per_channel": "bool = false",
  "output_stem": "str | null"
}
```

**Scale index guide (for a typical 4000×3000 stretched image):**

| Scale | Spatial frequency    | Typical content                             |
| ----- | -------------------- | ------------------------------------------- |
| 1     | 2–4 px features      | Noise, grain, hot pixel halos               |
| 2     | 4–8 px features      | Fine nebula filaments, star PSF wings       |
| 3     | 8–16 px features     | Nebula shells, dust lane edges              |
| 4     | 16–32 px features    | Galaxy arms, extended emission zones        |
| 5     | Residual (>32 px)    | Large-scale background, gradients           |

**Outputs:**

```json
{
  "processed_image_path": "str",
  "per_scale_stats": [
    {
      "scale": "int",
      "operation": "str",
      "coefficient_energy_before": "float",
      "coefficient_energy_after": "float"
    }
  ]
}
```

**Agent notes:**

- The B³-spline (starlet) is always used internally.
  PyWavelets has no `'b3'` wavelet — the implementation uses a custom à trous
  algorithm via `scipy.ndimage.convolve1d` with the `[1/16, 4/16, 6/16, 4/16, 1/16]`
  kernel applied separably with stride=2^i per scale. This is symmetric,
  produces no ringing, and is the same basis as PixInsight MLT.
- **Standard non-linear sharpening recipe** (apply on starless image with
  nebula luminance mask from T25):
  ```
  scale 1: suppress          (remove noise before sharpening)
  scale 2: sharpen weight=1.3
  scale 3: sharpen weight=1.15
  scale 4: passthrough
  scale 5: passthrough
  ```
- **Standard noise reduction recipe** (linear or post-stretch, no mask needed):
  ```
  scale 1: denoise sigma=0.5 method=BayesShrink
  scale 2: denoise sigma=0.2 method=BayesShrink
  scale 3: passthrough
  scale 4: passthrough
  scale 5: passthrough
  ```
- `mask_path`: pass a `range` or `inverted_luminance` mask from T25 when
  sharpening to confine the boost to nebulosity and avoid sharpening noise in
  the dark sky background. If mask shape mismatches image shape, the tool
  auto-resizes the mask before blending.
- `per_channel=false` (default): convert to luminance, process, recombine.
  This preserves color ratios and prevents chromatic artifacts. Set `true`
  only when independent per-channel noise characteristics require it (e.g.
  the blue channel has significantly different noise than red).
- Start conservative: denoise first, evaluate with T22, then sharpen in a
  separate call. Combining both in one call is valid but harder to debug.
- The `coefficient_energy_after / coefficient_energy_before` ratio in the
  output tells the agent whether a scale operation had meaningful effect. A
  ratio near 1.0 on a `suppress` scale means the image had little energy at
  that scale and suppression was unnecessary.
- T27 complements T17 `local_contrast_enhance` (which uses Siril's `wavelet`):
  use T17 for coarse global wavelet boosts; use T27 when you need per-scale
  control and/or mask-protected application.

---

#### T28 — `extract_narrowband`

**Purpose:** Extract narrowband signal channels (Hα, O-III, Green) from a CFA
(Bayer mosaic) image captured through a narrowband or duoband filter. Enables
the OSC dual-narrowband workflow for color cameras (e.g. Fujifilm X-T30 II +
Optolong L-eNhance or L-Ultimate) without a dedicated mono camera.

**Backend:** Siril CLI — `extract_Ha`, `extract_HaOIII`, `extract_Green`.

**IMPORTANT:** Input must be a CFA (non-debayered) FITS. T03 must be run with
`debayer=False` to produce CFA calibrated frames. The FITS header must contain
the `BAYER_PATTERN` keyword for Siril to correctly identify Bayer pixel layout.

**Dual-narrowband OSC workflow:**
1. Capture lights through a duoband filter. Keep calibrated frames as CFA.
2. Call `extract_Ha` or `extract_HaOIII` to split into Hα (red Bayer pixels)
   and O-III (blue Bayer pixels).
3. Register (T04) and stack (T07) each channel independently.
4. Process each stack with T12 (noise reduction) and T27 (multiscale detail).
5. Combine via T23 `pixel_math` using HOO or SHO palette:
   - HOO (natural nebula): `R=$Ha$ G=$OIII$ B=$OIII$`
   - SHO (Hubble): `R=$Ha$ G=0.3*$Ha$+0.7*$OIII$ B=$OIII$`

```
# Extract Hα and O-III simultaneously (recommended for duoband)
extract_HaOIII [-resample=oiii]  # -resample=oiii to equalize sizes

# Extract Hα only
extract_Ha [-upscale]

# Extract green continuum (diagnostic)
extract_Green
```

**Inputs:**

```json
{
  "working_dir": "str",
  "image_path": "str  (CFA FITS — not debayered)",
  "extraction_type": "ha | ha_oiii | green",
  "upscale_ha": "bool = false  (-upscale: 2x upsample Ha to full sensor resolution)",
  "resample": "null | 'ha' | 'oiii'  (-resample= to equalize Ha and OIII output sizes)"
}
```

**Outputs:**

```json
{
  "ha_path": "str | null",
  "oiii_path": "str | null",
  "green_path": "str | null",
  "extraction_type": "str",
  "next_steps": "str"
}
```

**Agent notes:** This tool is the entry point for all OSC narrowband workflows.
Use `extraction_type=ha_oiii` with `resample=oiii` for duoband captures so
both channels have equal dimensions after extraction. The `green` extraction is
mainly diagnostic — it isolates the pure continuum channel. After extraction,
run T04+T07 for each channel separately (two independent registration+stacking
pipelines), then converge with T23 `pixel_math` for palette combination.
The `resample` option is important: without it, the O-III image has 2× the
dimensions of the Hα image (since O-III uses 3 of 4 Bayer pixels vs 1 for Hα).

---

## 6  LangGraph Architecture

### 6.1 Top-Level StateGraph

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AstroState)

# ── Nodes ────────────────────────────────────────────────
graph.add_node("ingest",              ingest_node)
graph.add_node("preprocess",          preprocess_subgraph)   # subgraph
graph.add_node("linear_planner",      planner_node)          # LLM ReAct
graph.add_node("linear_tools",        tool_executor_node)    # tool dispatch
graph.add_node("stretch_planner",     stretch_planner_node)  # LLM + HITL
graph.add_node("stretch_tools",       tool_executor_node)
graph.add_node("stretch_hitl",        hitl_interrupt_node)   # graph interrupt
graph.add_node("nonlinear_planner",   planner_node)
graph.add_node("nonlinear_tools",     tool_executor_node)
graph.add_node("final_hitl",          hitl_interrupt_node)
graph.add_node("export",              export_node)

# ── Edges ────────────────────────────────────────────────
graph.add_edge("ingest",              "preprocess")
graph.add_edge("preprocess",          "linear_planner")

# Linear ReAct loop
graph.add_conditional_edges("linear_planner", route_planner, {
    "call_tool":    "linear_tools",
    "advance":      "stretch_planner",
})
graph.add_edge("linear_tools",        "linear_planner")

# Stretch with HITL
graph.add_conditional_edges("stretch_planner", route_planner, {
    "call_tool":    "stretch_tools",
    "hitl":         "stretch_hitl",
    "advance":      "nonlinear_planner",
})
graph.add_edge("stretch_tools",       "stretch_planner")
graph.add_edge("stretch_hitl",        "stretch_planner")

# Non-linear ReAct loop with final HITL
graph.add_conditional_edges("nonlinear_planner", route_planner, {
    "call_tool":    "nonlinear_tools",
    "hitl":         "final_hitl",
    "advance":      "export",
})
graph.add_edge("nonlinear_tools",     "nonlinear_planner")
graph.add_edge("final_hitl",          "nonlinear_planner")

graph.add_edge("export",              END)

graph.set_entry_point("ingest")
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer, interrupt_before=["stretch_hitl", "final_hitl"])
```

### 6.2 Pre-Processing Subgraph (Order-Deterministic, Parameter-Agentic)

The traversal order through pre-processing is fixed — the graph edges are
linear and unconditional. This enforces the physical dependency chain: you
cannot calibrate before building masters, register before calibrating, or
stack before selecting frames. **The order is not negotiable.**

What is agentic is **parameter selection within each node**. Hard-coded
defaults will fail on real-world data — a 10-frame dataset needs different
rejection parameters than a 60-frame one; a DSLR needs different calibration
flags than a cooled mono camera; poor seeing requires tighter frame selection
than stable conditions. Before each tool executes, a single focused LLM call
selects the parameters appropriate for the data in hand.

**`parameter_selector()`** is a lightweight, single-shot LLM call (not a
ReAct loop). It receives a structured context — the dataset characteristics
from ingest, the current metadata and metrics — and returns a parameter dict
for the next tool. It also returns a `reasoning` string that becomes the
`ReportEntry.reasoning` for that step.

```python
def parameter_selector(tool_name: str, state: AstroState) -> dict:
    """Single LLM call to select parameters for one pre-processing step.

    Returns a parameter dict ready to pass to the tool, plus a 'reasoning'
    key (str) explaining the choices. The reasoning key is stripped before
    the dict is passed to the tool.
    """
    context = {
        "tool": tool_name,
        "lights_count":  len(state["dataset"]["files"]["lights"]),
        "darks_count":   len(state["dataset"]["files"]["darks"]),
        "flats_count":   len(state["dataset"]["files"]["flats"]),
        "biases_count":  len(state["dataset"]["files"]["biases"]),
        "camera_model":  state["dataset"]["acquisition_meta"].get("camera_model"),
        "is_osc":        state["metadata"]["is_osc"],
        "bortle":        state["dataset"]["acquisition_meta"].get("bortle"),
        "frame_metrics": state["metrics"].get("frame_stats"),   # populated after T05
        "history":       state["history"][-5:],                 # last 5 steps for context
    }
    response = parameter_selector_llm.invoke(
        build_parameter_prompt(tool_name, context)
    )
    return response  # validated against tool's input schema before use


def build_preprocess_node(tool_name: str, tool_fn: callable) -> callable:
    """Factory: wraps each pre-processing tool with parameter selection + reporting."""

    def node(state: AstroState) -> dict:
        # 1. LLM selects parameters for this specific tool + dataset
        params = parameter_selector(tool_name, state)
        reasoning = params.pop("reasoning", "")

        # 2. Execute the tool
        result = tool_fn(**params, working_dir=state["dataset"]["working_dir"])

        # 3. Record decision in processing report
        entry = ReportEntry(
            step=len(state["processing_report"]) + 1,
            phase=state["phase"],
            tool=tool_name,
            reasoning=reasoning,
            parameters=params,
            metrics_before=None,   # pre-processing steps operate on sequences, not
            metrics_after=None,    # single images; no analyze_image before/after
            outcome="success",
            outcome_detail=result.get("summary", ""),
            timestamp=datetime.utcnow().isoformat(),
        )

        return {
            "history": [f"{tool_name}: {result.get('summary', 'complete')}"],
            "processing_report": [entry],
            # paths and metrics updated by each tool's specific return handling
        }

    return node
```

The subgraph graph structure remains fixed:

```python
preprocess = StateGraph(AstroState)

preprocess.add_node("build_masters", build_preprocess_node("build_masters",  build_masters_fn))
preprocess.add_node("calibrate",     build_preprocess_node("siril_calibrate", siril_calibrate_fn))
preprocess.add_node("register",      build_preprocess_node("siril_register",  siril_register_fn))
preprocess.add_node("analyze",       build_preprocess_node("analyze_frames",  analyze_frames_fn))
preprocess.add_node("select",        build_preprocess_node("select_frames",   select_frames_fn))
preprocess.add_node("stack",         build_preprocess_node("siril_stack",     siril_stack_fn))
preprocess.add_node("crop",          build_preprocess_node("auto_crop",       auto_crop_fn))

preprocess.set_entry_point("build_masters")
preprocess.add_edge("build_masters", "calibrate")
preprocess.add_edge("calibrate",     "register")
preprocess.add_edge("register",      "analyze")
preprocess.add_edge("analyze",       "select")
preprocess.add_edge("select",        "stack")
preprocess.add_edge("stack",         "crop")
preprocess.add_edge("crop",          END)

preprocess_subgraph = preprocess.compile()
```

The `build_masters` node checks if masters already exist in
`state.paths.masters` before calling the LLM — if they do, it skips both
selection and execution and logs a "masters already present, skipped" report
entry. It handles the dependency order (bias → dark → flat) in code, not
via separate graph nodes.

### 6.3 Planner Node (ReAct Loop)

The planner node is the core intelligence. It receives the full `AstroState`,
builds a prompt containing:

1. **Current phase** and **processing history** (what's been done)
2. **Latest image metrics** from `analyze_image`
3. **Latest preview** (encoded image for vision-capable LLMs)
4. **Available tools** for the current phase (tool names + docstrings)
5. **Domain heuristics** (the decision map from T20)

```python
from langchain_core.messages import SystemMessage, HumanMessage

PHASE_TOOLS = {
    ProcessingPhase.LINEAR: [
        remove_gradient, color_calibrate, remove_green_noise,
        noise_reduction, deconvolution, analyze_image,
        plate_solve, pixel_math,
        create_mask,          # T25 — masks for protected linear-phase noise reduction
        multiscale_process,   # T27 — wavelet denoise in linear space (denoise op only)
    ],
    ProcessingPhase.STRETCH: [
        stretch_image, analyze_image,
    ],
    ProcessingPhase.NONLINEAR: [
        star_removal, curves_adjust, local_contrast_enhance,
        saturation_adjust, star_restoration, noise_reduction,
        analyze_image, pixel_math,
        create_mask,          # T25 — masks for targeted curves / sat / sharpening
        reduce_stars,         # T26 — morphological star disk reduction
        multiscale_process,   # T27 — per-scale sharpen + denoise with mask support
    ],
}
# generate_preview is intentionally absent from PHASE_TOOLS in all phases.
# It is never callable by the agent. It is called exclusively inside
# auto_hitl_check() and the mandatory HITL nodes (stretch_hitl, final_hitl).

def planner_node(state: AstroState) -> dict:
    phase = state["phase"]
    tools = PHASE_TOOLS[phase]

    system = SystemMessage(content=SYSTEM_PROMPT)
    # Text-only context — no image_url. The agent makes all decisions from
    # quantitative metrics. Visual context reaches the agent only through
    # HITL: the user's response to a preview surfaces as a HumanMessage.
    context = HumanMessage(content=build_state_summary(state))

    llm_with_tools = llm.bind_tools(tools + [advance_phase, request_hitl])
    response = llm_with_tools.invoke([system, context] + state["messages"])

    return {"messages": [response]}
```

### 6.4 Routing Logic

```python
def route_planner(state: AstroState) -> str:
    last = state["messages"][-1]

    if not last.tool_calls:
        return "advance"

    tool_name = last.tool_calls[0]["name"]
    if tool_name == "advance_phase":
        return "advance"
    if tool_name == "request_hitl":
        return "hitl"
    return "call_tool"
```

### 6.5 Tool Executor Node

```python
from langgraph.prebuilt import ToolNode
from dataclasses import dataclass

@dataclass
class ToolEntry:
    fn: callable               # the tool function bound to the LLM
    modifies_image: bool       # if True, auto-analyze after call
    requires_visual_review: bool  # if True, auto-trigger HITL after call (V1 default)
    visual_review_question: str   # targeted question presented to user at HITL
    visual_review_options: list[str]  # suggested responses

TOOL_REGISTRY: dict[str, ToolEntry] = {
    "ingest_dataset":         ToolEntry(ingest_dataset_fn,         False, False, "", []),
    "build_masters":          ToolEntry(build_masters_fn,          False, False, "", []),
    "siril_calibrate":        ToolEntry(siril_calibrate_fn,        False, False, "", []),
    "register_frames":        ToolEntry(register_frames_fn,        False, False, "", []),
    "select_frames":          ToolEntry(select_frames_fn,          False, False, "", []),
    "stack_frames":           ToolEntry(stack_frames_fn,           False, False, "", []),
    "auto_crop":              ToolEntry(auto_crop_fn,              True,  False, "", []),
    "analyze_image":          ToolEntry(analyze_image_fn,          False, False, "", []),
    "remove_gradient":        ToolEntry(remove_gradient_fn,        True,  True,
        question=(
            "Gradient removal complete. Inspect for: halos or dark rings around "
            "bright nebulosity edges, over-subtraction creating artificial dark "
            "regions, or background that looks unnaturally flat."
        ),
        options=[
            "Background looks natural — continue",
            "Halos or over-subtraction present — revert and retry with adjusted samples",
            "Gradient partially corrected — re-run with different model order",
        ]),
    "color_calibrate":        ToolEntry(color_calibrate_fn,        True,  False, "", []),
    "remove_green_noise":     ToolEntry(remove_green_noise_fn,     True,  False, "", []),
    "noise_reduction":        ToolEntry(noise_reduction_fn,        True,  False, "", []),
    "deconvolution":          ToolEntry(deconvolution_fn,          True,  True,
        question=(
            "Deconvolution complete. Inspect for: ringing or halos around stars, "
            "false detail or over-sharpening in nebula cores, or amplified noise "
            "in dim regions. These artifacts may appear even when SNR metrics improve."
        ),
        options=[
            "Looks clean — continue",
            "Ringing or halos present — revert and retry with fewer iterations",
            "Over-sharpened or noisy — revert and use gentler regularization",
        ]),
    "plate_solve":            ToolEntry(plate_solve_fn,            False, False, "", []),
    "stretch_image":          ToolEntry(stretch_image_fn,          True,  False, "", []),
    "star_removal":           ToolEntry(star_removal_fn,           True,  True,
        question=(
            "Star removal complete. Inspect for: artifacts at bright star positions, "
            "erased or damaged nebulosity near dense star fields, or incomplete "
            "removal of large saturated stars."
        ),
        options=[
            "Looks clean — continue",
            "Artifacts at star positions — revert and retry",
            "Nebula damaged near stars — revert and retry with luminance-protected mask",
        ]),
    "curves_adjust":          ToolEntry(curves_adjust_fn,          True,  False, "", []),
    "local_contrast_enhance": ToolEntry(local_contrast_enhance_fn, True,  False, "", []),
    "saturation_adjust":      ToolEntry(saturation_adjust_fn,      True,  False, "", []),
    "star_restoration":       ToolEntry(star_restoration_fn,       True,  False, "", []),
    "pixel_math":             ToolEntry(pixel_math_fn,             True,  False, "", []),
    "export_final":           ToolEntry(export_final_fn,           False, False, "", []),
    "create_mask":            ToolEntry(create_mask_fn,            False, False, "", []),
    "reduce_stars":           ToolEntry(reduce_stars_fn,           True,  False, "", []),
    "multiscale_process":     ToolEntry(multiscale_process_fn,     True,  False, "", []),
}
```

Each tool function:

1. Receives the tool-call arguments
2. Translates them into a Siril script or Python operation
3. Executes via `subprocess.run(["siril-cli", "-s", script_path])`
4. Parses stdout/stderr for results
5. Updates the state (e.g. `paths.current_image`, `metrics`)
6. Returns a `ToolMessage` with structured JSON output

**Auto-analysis behavior:**

After any tool call where `entry.modifies_image` is `True`, the executor
automatically calls `analyze_image_fn` on the updated `paths.current_image`
and refreshes `state.metrics`. The planner always has current quantitative
data without spending a tool call. No preview is generated here — the agent
operates on metrics, not images.

```python
def tool_executor_post_hook(state: AstroState, tool_name: str) -> dict:
    updates = {}
    entry = TOOL_REGISTRY[tool_name]

    if entry.modifies_image:
        metrics = analyze_image_fn(fits_path=state["paths"]["current_image"])
        updates["metrics"] = metrics

    return updates
```

**Auto-HITL behavior:**

Each tool entry carries a `requires_visual_review` boolean. When `True`,
the executor generates a preview and injects a HITL payload into state
immediately after the tool call completes — before control returns to the
planner. This is a system-enforced checkpoint; the agent cannot skip or
defer it. The targeted `question` and `options` are defined in the registry
entry, not derived at runtime.

In V1, `requires_visual_review = True` for `remove_gradient`, `deconvolution`,
and `star_removal`. These are the operations known from domain knowledge to
produce spatial artifacts that aggregate metrics cannot reliably detect.
When experience with real data shows a checkpoint is consistently noise-free,
flip its flag to `False` — no other code changes required.

```python
def auto_hitl_check(state: AstroState, tool_name: str) -> dict | None:
    """Returns a HITL payload if this tool requires automatic visual review."""
    entry = TOOL_REGISTRY[tool_name]
    if not entry.requires_visual_review:
        return None

    preview = generate_preview_fn(
        fits_path=state["paths"]["current_image"],
        width=1920,
        auto_stretch_linear=state["metadata"]["is_linear"],
    )
    return {
        "trigger": f"auto_{tool_name}",
        "question": entry.visual_review_question,
        "options": entry.visual_review_options,
        "preview_path": preview["preview_path"],
    }
```

`generate_preview_fn` is called *only* inside `auto_hitl_check` and the
mandatory HITL nodes (`stretch_hitl`, `final_hitl`). It is never present
in `PHASE_TOOLS` and is never directly callable by the agent.

**Processing report behavior:**

After each LLM-driven tool call (linear, stretch, and non-linear phases), the
executor extracts the LLM's reasoning from the assistant message that preceded
the tool call and appends a `ReportEntry` to `state.processing_report`.

```python
def build_report_entry(state: AstroState, tool_call: dict, result: dict,
                       metrics_before: dict | None) -> ReportEntry:
    # Reasoning is the text content of the last assistant message
    # (the explanation the LLM wrote before the tool_call block)
    last_message = state["messages"][-1]
    reasoning = extract_text_before_tool_call(last_message)

    outcome = "success"
    outcome_detail = result.get("summary", "")
    metrics_after = result.get("metrics")

    if metrics_before and metrics_after:
        if metrics_after.get("background_noise", 0) > metrics_before.get("background_noise", 0):
            outcome = "degraded"
            outcome_detail = "background_noise increased — agent may revert"

    return ReportEntry(
        step=len(state["processing_report"]) + 1,
        phase=state["phase"],
        tool=tool_call["name"],
        reasoning=reasoning,
        parameters=tool_call["args"],
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        outcome=outcome,
        outcome_detail=outcome_detail,
        timestamp=datetime.utcnow().isoformat(),
    )
```

**Report rendering:**

At export time, `render_processing_report(state)` converts
`state.processing_report` to a formatted markdown file written to
`{working_dir}/processing_report.md`. See §12 for the report format.

### 6.6 State Transitions

```
Phase transitions managed by the planner's "advance_phase" pseudo-tool:

INGEST ──► CALIBRATION ──► REGISTRATION ──► ANALYSIS ──► STACKING
     └──────────── preprocess subgraph (automatic) ──────────────┘
                                                          │
                                                          ▼
                                                       LINEAR
                                                          │
                                                          ▼
                                                       STRETCH
                                                          │
                                                          ▼
                                                      NONLINEAR
                                                          │
                                                          ▼
                                                       REVIEW ──► EXPORT ──► COMPLETE
```

State fields updated at transitions:


| Transition          | State mutations                                   |
| ------------------- | ------------------------------------------------- |
| LINEAR → STRETCH    | `phase = STRETCH`                                 |
| STRETCH → NONLINEAR | `phase = NONLINEAR`, `metadata.is_linear = False` |
| NONLINEAR → REVIEW  | `phase = REVIEW`, variants generated              |
| REVIEW → EXPORT     | `phase = EXPORT`, user selection applied          |


### 6.7 Checkpointing & Persistence

LangGraph's `MemorySaver` (dev) or `SqliteSaver` (production) checkpoints the
full `AstroState` after every node execution. This enables:

- **Resume after crash**: pick up from the last successful step.
- **HITL pause/resume**: the graph pauses at interrupt nodes and resumes when
the user provides feedback, even hours later.
- **Branching**: the user can rewind to a checkpoint and try different
processing paths (e.g. re-stretch with different parameters).

Thread IDs should include the dataset ID for reproducibility:
`thread_id = f"{dataset.id}_{timestamp}"`.

---

## 7  Agent Reasoning Framework

### 7.1 How the Agent "Sees"

The agent perceives the image through two complementary channels:

1. **Quantitative — `analyze_image` metrics.** The primary decision engine.
   Numbers that drive all autonomous choices: noise level, SNR, gradient
   magnitude, green excess, FWHM, star count, background flatness, clipping
   percentages. `analyze_image` runs automatically after every
   image-modifying tool call via `tool_executor_post_hook` — the agent
   always operates on fresh data without spending a tool call.

2. **Qualitative — HITL + human vision.** For spatial artifacts and aesthetic
   qualities that aggregate metrics cannot reliably detect (ringing, halos,
   over-smoothing, color naturalness), the agent routes through HITL. The
   human inspects the preview and responds; that response enters the
   conversation as a `HumanMessage`, giving the agent qualitative context
   for its next decision. The agent never calls `generate_preview` directly
   and never interprets images with its own vision — all visual judgment
   belongs to the human.

This separation is intentional for V1: it forces every visual decision to
be human-verified, creating a logged record of what each HITL checkpoint
catches. Over time that record becomes the evidence base for determining
which checkpoints can be safely automated.

### 7.2 Decision Heuristics (System Prompt Reference)

These heuristics inform the content of the system prompt (§7.4). They are
**not implemented as code** — the LLM applies them using judgment. The
thresholds below are calibrated starting points for the prompt author, written
into §7.4 as guidance, not enforced programmatically.

The pre-processing subgraph (T01–T08) is fully deterministic. The agent takes
control at the linear phase. From that point forward, the decision model is:

| Phase        | Decision mode                                 |
| ------------ | --------------------------------------------- |
| Linear       | Metric-guided — thresholds inform tool calls  |
| Stretch      | Always HITL — agent generates variants, user selects |
| Non-linear   | Agent judgment — image + metrics + feedback   |
| Export       | Deterministic — fixed output formats          |

**Linear Phase — Reference heuristics for system prompt:**

These thresholds are starting points. The LLM reads `analyze_image` output and
decides whether each linear tool is warranted. The numbers below represent
typical calibrated decision boundaries; the system prompt in §7.4 encodes them
as guidance alongside the reasoning behind each.

| Tool                | Condition to apply                    | Key metric            |
| ------------------- | ------------------------------------- | --------------------- |
| `remove_gradient`   | Likely needed; confirm by measure     | `gradient_magnitude`  |
| `color_calibrate`   | Always — requires plate solve first   | n/a                   |
| `remove_green_noise`| OSC/DSLR data with measurable excess  | `green_excess`        |
| `noise_reduction`   | Always — choose modulation by SNR     | `snr_estimate`        |
| `deconvolution`     | Optional — only if SNR warrants it    | `snr_estimate`        |
| `multiscale_process`| Optional — denoise only, if needed    | `background_noise`    |

Typical threshold values (written into §7.4 as prompt guidance):
- `gradient_magnitude > 0.05` → remove_gradient warranted
- `green_excess > 0.02` + is_osc → remove_green_noise warranted
- `snr_estimate > 50` → deconvolution potentially safe; start conservative
- `snr_estimate < 30` → skip deconvolution entirely

**Non-linear Phase — Agent judgment zone:**

The non-linear phase is intentionally not a checklist. The LLM receives the
current image, its metrics, and user feedback, then decides what the image
needs. The system prompt (§7.4) provides domain knowledge, target-type
guidance, and quality criteria. The agent should:

- Read `user_feedback` for any stated preferences before calling any aesthetic tool
- Use the masked-application pattern (T25 + T23) for targeted processing
- Generate variants and call `request_hitl` for subjective choices
- Measure before and after via auto-analysis — revert if a step degraded the image

Tool availability per phase is enforced by `PHASE_TOOLS` (§6.3). The
non-linear aesthetic decisions — which tools to use, in what order, with what
parameters — belong to the LLM's judgment guided by §7.4, not to a fixed
sequence in code.

### 7.3 Parameter Selection Strategy

The agent selects tool parameters using a combination of:

1. **Metric-driven defaults** — e.g. deconvolution iterations scaled to
  `snr_estimate / 5`, stretch intensity inversely proportional to
   `background_noise`.
2. **Conservative start → iterate** — begin with gentle parameters, evaluate,
  increase if needed. This prevents over-processing.
3. **Domain knowledge in system prompt** — the planner's system prompt contains
  astrophotography processing rules (see Section 7.4).
4. **User feedback memory** — `state.user_feedback` accumulates preferences
  across sessions (e.g. "user prefers saturated colors", "user dislikes
   aggressive star reduction").

### 7.4 System Prompt

**Design principle:** The system prompt provides only what the model cannot
know from context — project-specific conventions, physical constraints that
are mathematically non-negotiable, and current runtime state. All domain
knowledge and aesthetic judgment belong to the model. This prompt is
intentionally lean; extend it only when real-data runs reveal a specific,
repeatable failure that cannot be addressed by improving a tool contract.

The `build_state_summary()` function (Phase 8) renders this template and
injects all `{placeholder}` values at runtime.

---

```
You are an expert astrophotography post-processing agent with full autonomy
over all processing and aesthetic decisions.

PHYSICAL CONSTRAINTS (mathematical — enforced by phase gating):
- Gradient removal, color calibration, SCNR, noise reduction, and
  deconvolution must complete before stretch_image. After stretch, pixel
  values are no longer proportional to photon flux — these operations are
  physically invalid on non-linear data.
- multiscale_process sharpen operations are invalid in linear space.
- star_removal and reduce_stars are non-linear phase tools only.
- remove_green_noise (SCNR) is only valid on OSC/DSLR data, not mono.

PRE-PROCESSING COMPLETE:
T01–T08 ran automatically before you took control. The stacked, cropped
linear image is your starting point. Check state.paths and state.metrics.

VISUAL INSPECTION:
You have no direct access to generate_preview. All decisions are made from
metrics. When you determine a choice requires visual judgment that data alone
cannot settle, call request_hitl — the HITL mechanism generates the preview
and surfaces the user's response back to you. Three operations also trigger
automatic HITL regardless of your next planned action (system-enforced, not
your decision): remove_gradient, deconvolution, and star_removal. You will
resume with the user's visual assessment already in context.

MANDATORY HITL CHECKPOINTS:
1. Stretch — generate 2–3 variants via stretch_image calls, then call
   request_hitl. Never auto-select the stretch; the photographer chooses.
2. Final review — after non-linear processing is complete, call request_hitl
   before export. Present processing history and key before/after metrics.
   The user confirms or requests revisions.

MASKED PROCESSING CONVENTION:
To confine any operation to a specific region: (1) create_mask (T25) to
define the target, (2) run the tool globally, (3) pixel_math (T23):
"$processed$ * $mask$ + $original$ * (1 - $mask$)"

USER FEEDBACK:
Read state.user_feedback before any aesthetic decision. It is the
highest-priority input and overrides your judgment. Apply it literally —
it is the photographer's stated preference.

CURRENT STATE:
Phase:             {phase}
Target:            {target_name}
Is linear:         {is_linear}
Is OSC/DSLR:       {is_osc}
User feedback:     {user_feedback_summary}
History (last 10): {history}
Latest metrics:    {metrics_json}
```

---

## 8  Human-in-the-Loop Protocol

### 8.0 Interrupt Architecture — Presenter Pattern

The HITL node never performs I/O directly. It builds a structured
`HITLPayload` and calls LangGraph's `interrupt(payload)`, which pauses the
graph and returns control to the calling process. The calling process
(CLI in V1, Streamlit or other UI in future) reads the payload, presents
the image(s) and question to the user, collects a response, and resumes
the graph with `Command(resume=response)`.

This separation means the graph logic never changes when the UI changes.
The `HITLPayload` is the stable contract between graph and presenter.

```python
from typing import Literal
from langgraph.types import interrupt, Command

class HITLPayload(TypedDict):
    trigger: str                    # e.g. "stretch_selection", "auto_deconvolution"
    question: str                   # question to present to the user
    options: list[str] | None       # if None, free-text response expected
    allow_free_text: bool           # True allows typed response alongside options
    preview_paths: list[str]        # one or more absolute JPG paths
    preview_labels: list[str]       # label for each preview (same length as paths)
    context: str                    # metric summary or other text context
    checkpoint: str                 # which tool or phase triggered this interrupt

def hitl_node(state: AstroState) -> dict:
    payload: HITLPayload = build_hitl_payload(state)
    user_response: str = interrupt(payload)   # graph pauses; resumes with response
    return apply_hitl_response(state, payload, user_response)
```

**V1 — CLI presenter (`cli.py`):**

```python
import subprocess, sys

def cli_presenter(payload: HITLPayload) -> str:
    # Open each preview with the OS default image viewer (macOS: 'open', Linux: 'xdg-open')
    for path in payload["preview_paths"]:
        subprocess.Popen(["open", path])   # non-blocking; images open in Preview.app

    print(f"\n{'─' * 60}")
    print(f"  HITL CHECKPOINT: {payload['checkpoint']}")
    print(f"{'─' * 60}")
    print(f"  {payload['context']}")
    print(f"\n  {payload['question']}\n")

    if payload["options"]:
        for i, opt in enumerate(payload["options"], 1):
            print(f"  [{i}] {opt}")
        if payload["allow_free_text"]:
            print(f"  [t] Type a custom response")
        choice = input("\n  Your choice: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(payload["options"]):
            return payload["options"][int(choice) - 1]
    # Free-text fallback
    return input("  Your response: ").strip()

# In the CLI run loop:
for chunk in app.stream(None, config, stream_mode="updates"):
    if "__interrupt__" in chunk:
        payload = chunk["__interrupt__"][0].value
        response = cli_presenter(payload)
        app.invoke(Command(resume=response), config)
```

**Future — Streamlit presenter (`streamlit_app.py`):**

The same `HITLPayload` schema maps directly to Streamlit widgets:
- `preview_paths` → `st.image(path, caption=label)` for each path
- `question` → `st.write(question)`
- `options` → `st.radio(question, options)` or `st.columns` with `st.button`
- `allow_free_text` → `st.text_area`
- `context` → `st.expander("Metrics")` with the context string

The Streamlit app calls `app.invoke(Command(resume=response), config)` on
form submission, just as the CLI does. No graph code changes.

### 8.1 Interrupt Points

HITL checkpoints fall into three categories:

- **Hard-wired** — `interrupt_before` nodes in the LangGraph; always pause regardless of agent state or metrics.
- **Auto-triggered** — injected by the tool executor when `entry.requires_visual_review` is `True` in `TOOL_REGISTRY`. System-enforced: the agent cannot skip or defer these. Each surfaces a targeted question about the known artifact type for that operation. Controlled by a single boolean per tool — flip to `False` when experience shows a checkpoint is consistently unnecessary.
- **On-demand** — triggered by the agent calling `request_hitl` when it determines a decision cannot be made on data alone.

All three categories call `generate_preview_fn` internally to present the image. The agent never calls `generate_preview` directly.

| Checkpoint                  | Type           | Phase     | Trigger                                                                          | Artifact focus / Question presented                                                                                     |
| --------------------------- | -------------- | --------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Stretch Selection**       | Hard-wired     | STRETCH   | Agent has called stretch_image 2–3 times for variants                            | Side-by-side previews + stretch parameters. User selects preferred result.                                              |
| **Final Review**            | Hard-wired     | REVIEW    | All non-linear processing complete, before export                                | Before/after comparison + processing history summary. User confirms or submits revision requests.                        |
| **Post-Gradient**           | Auto-triggered | LINEAR    | `remove_gradient` completes; `requires_visual_review=True`                       | Halos or dark rings around nebulosity edges; over-subtraction dark regions; unnaturally flat background areas.          |
| **Post-Deconvolution**      | Auto-triggered | LINEAR    | `deconvolution` completes; `requires_visual_review=True`                         | Ringing or halos around stars; false detail in nebula cores; amplified noise in dim regions.                            |
| **Post-Star-Removal**       | Auto-triggered | NONLINEAR | `star_removal` completes; `requires_visual_review=True`                          | Artifacts at bright star positions; damaged nebulosity near dense star fields; incomplete removal of saturated stars.   |
| **Target type unclear**     | On-demand      | NONLINEAR | `acquisition_meta.target_name` is null                                           | "Is this primarily a nebula, galaxy, or star cluster?" — sets `user_feedback.target_type`.                             |
| **Star treatment**          | On-demand      | NONLINEAR | `user_feedback.star_prominence` not set + stars compete with subject             | "Reduce star prominence, keep natural, or enhance?" — sets `user_feedback.star_prominence`.                            |
| **Borderline SNR**          | On-demand      | LINEAR    | `snr_estimate` is 40–60 (marginal deconvolution threshold)                       | "SNR is marginal. Attempt deconvolution conservatively, or skip?" + metric summary.                                    |
| **Unexpected degradation**  | On-demand      | Any       | Auto-analysis shows `background_noise` increased after a tool call               | Before/after metric diff + "This step appears to have degraded the image. Revert?"                                     |
| **Revision confirmed**      | On-demand      | NONLINEAR | Agent has executed a `revision_request` from `user_feedback.revision_requests`   | Revised result preview + "Does this address your request?"                                                              |


### 8.2 A/B Variant Strategy

```python
def generate_stretch_variants(state: AstroState) -> dict:
    variants = {}
    image = state["paths"]["current_image"]

    # Variant A: Gentle
    result_a = stretch_image(image, method="autostretch",
                             autostretch_options={"shadows_clipping_sigma": -2.5,
                                                  "target_background": 0.20,
                                                  "linked": True})
    variants["gentle"] = result_a["stretched_image_path"]

    # Variant B: Moderate (GHS)
    result_b = stretch_image(image, method="ghs",
                             ghs_options={"stretch_amount": 2.5,
                                          "color_model": "human"})
    variants["moderate"] = result_b["stretched_image_path"]

    # Variant C: Aggressive (GHS)
    result_c = stretch_image(image, method="ghs",
                             ghs_options={"stretch_amount": 4.0,
                                          "color_model": "human"})
    variants["aggressive"] = result_c["stretched_image_path"]

    # Generate previews (internal call — stretch_hitl node, not agent)
    for label, path in variants.items():
        generate_preview_fn(path, annotation=f"Stretch: {label}")

    return {"paths": {**state["paths"], "variants": variants}}
```

### 8.3 Feedback Integration

User feedback accumulates in `state.user_feedback` across all HITL sessions
for a dataset. It persists across checkpoint resumes. The agent reads it at
the start of each phase and before any aesthetic tool call.

**Schema:**

```python
user_feedback = {
    # Set by stretch HITL
    "stretch_preference": "gentle | moderate | aggressive",

    # Set by final review or on-demand star HITL
    "star_prominence": "reduced | natural | enhanced",

    # Set by saturation on-demand HITL or revision request
    "saturation_preference": "minimal | natural | high",

    # Set by target type HITL
    "target_type": "emission_nebula | reflection_nebula | galaxy | star_cluster | wide_field",

    # Free text set by user at any HITL point
    "notes": "str",

    # Accumulated list of specific revision requests from final review
    "revision_requests": [
        {
            "step": "str",       # e.g. "saturation", "sharpening", "stars"
            "direction": "str",  # "increase" | "decrease" | "change"
            "amount": "str",     # "slight" | "moderate" | "significant"
            "note": "str | null" # optional free text clarification
        }
    ]
}
```

**Field-to-tool mapping (how the agent applies each field):**

| Field                 | Value        | Agent action                                                              |
| --------------------- | ------------ | ------------------------------------------------------------------------- |
| `stretch_preference`  | `gentle`     | If re-running stretch: use autostretch with conservative sigma            |
| `stretch_preference`  | `aggressive` | If re-running stretch: use GHS D=4.0                                      |
| `star_prominence`     | `reduced`    | Call reduce_stars (T26) after T19; use star_weight=0.7–0.8 in T19        |
| `star_prominence`     | `enhanced`   | Skip reduce_stars; use star_weight=1.0–1.1 in T19                        |
| `star_prominence`     | `natural`    | Skip reduce_stars; use star_weight=0.85–0.95 in T19                      |
| `saturation_preference` | `high`     | Start T18 amount=0.5; may go to 0.7 if image supports it                 |
| `saturation_preference` | `natural`  | Cap T18 amount=0.3; always use inverted-luminance mask                    |
| `saturation_preference` | `minimal`  | T18 amount≤0.15 or skip entirely                                          |
| `target_type`         | any          | Applies target-type guidance from §7.4 for all non-linear decisions       |
| `notes`               | free text    | Standing instruction — read and apply throughout the session              |
| `revision_requests`   | list         | Execute each request in order; re-run affected tools; present result      |

**Revision request execution:**

When `revision_requests` is non-empty on phase entry or after HITL resume:

1. Read each request in order.
2. Map the `step` to the relevant tool(s) using the field-to-tool table above.
3. Revert to the checkpoint just before that step (LangGraph `SqliteSaver`).
4. Re-run from that point with adjusted parameters per `direction` and `amount`.
5. When complete, call `request_hitl` with the "Revision confirmed" trigger
   (§8.1) to show the result and confirm the user is satisfied.
6. Clear the executed request from `revision_requests` once confirmed.

This design means revisions are always backed by a checkpoint — the agent
never destructively overwrites a prior result without a recovery path.

---

## 9  Error Handling & Recovery

### 9.1 Tool-Level Errors


| Error Type                | Example                     | Recovery                                                               |
| ------------------------- | --------------------------- | ---------------------------------------------------------------------- |
| **Siril script failure**  | Non-zero exit code          | Parse stderr, log error, retry with adjusted parameters                |
| **Plate solve failure**   | No WCS solution             | Broaden search radius, try different catalog, fall back to blind solve |
| **StarNet not installed** | External dependency missing | Abort startup with install instructions — star separation is required for quality non-linear processing |
| **Insufficient frames**   | < 3 lights after rejection  | Loosen selection criteria, re-run select_frames                        |
| **Out of memory**         | Large mosaic or drizzle     | Reduce image dimensions, disable drizzle, warn user                    |


### 9.2 Graph-Level Recovery

```python
try:
    result = tool_fn(**args)
except SirilError as e:
    return ToolMessage(
        content=json.dumps({
            "error": str(e),
            "stderr": e.stderr,
            "suggestion": classify_error(e)
        }),
        tool_call_id=tool_call_id
    )
```

The planner LLM receives the error message and decides:

- **Retry** with different parameters
- **Skip** the step and proceed
- **Escalate** to the user via HITL

### 9.3 Revert Strategy

Every tool writes to a new output file (never overwrites the input). The
`state.paths.current_image` pointer advances only on success. To revert, the
agent simply points back to the previous file. The `state.history` log tracks
every operation for auditability.

---

## 10  Configuration & Defaults

### 10.1 External Dependencies


| Dependency        | Min Version | Purpose                                                | Required    |
| ----------------- | ----------- | ------------------------------------------------------ | ----------- |
| Siril             | 1.4.0       | Core processing engine                                 | Yes         |
| GraXpert          | 3.0.0       | AI gradient removal + denoising                        | Yes         |
| StarNet++         | 2.0         | Star removal neural network                            | Yes         |
| Python            | 3.11+       | Agent runtime, Astropy, Photutils                      | Yes         |
| Astropy           | 6.0+        | FITS I/O, WCS, header parsing                          | Yes         |
| ExifTool          | 12.0+       | EXIF/metadata extraction from all camera RAW formats (T01). Install: `brew install exiftool` | Yes |
| pyexiftool        | 0.5.6+      | Python wrapper for ExifTool binary (T01)               | Yes         |
| Photutils         | 2.0+        | Star detection, PSF analysis                           | Yes         |
| scikit-image      | 0.22+       | Mask generation (T25), morphological star reduction (T26), wavelet processing (T27) | Yes |
| PyWavelets (pywt) | 1.6+        | Undecimated (à trous) wavelet transform for T27 `multiscale_process` | Yes |
| LangGraph         | 0.2+        | Agent orchestration framework                          | Yes         |
| LangChain         | 0.3+        | LLM interface, tool bindings                           | Yes         |


### 10.2 Default Processing Profiles

```python
PROFILE_DEFAULTS = {
    "conservative": {
        "noise_reduction_modulation": 0.7,
        "stretch_method": "autostretch",
        "stretch_shadows_clip": -2.5,
        "saturation_amount": 0.2,
        "star_weight": 1.0,
        "deconvolution": False,
    },
    "balanced": {
        "noise_reduction_modulation": 0.85,
        "stretch_method": "ghs",
        "stretch_amount": 2.5,
        "saturation_amount": 0.4,
        "star_weight": 0.85,
        "deconvolution": True,
        "deconvolution_iterations": 10,
    },
    "aggressive": {
        "noise_reduction_modulation": 1.0,
        "stretch_method": "ghs",
        "stretch_amount": 4.0,
        "saturation_amount": 0.7,
        "star_weight": 0.6,
        "deconvolution": True,
        "deconvolution_iterations": 20,
    },
}
```

---

## 11  Future Enhancements (v2+)


| Feature                    | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| **Narrowband support**     | SHO/HOO palette mapping via per-channel stacking + `pixel_math` channel combination |
| **Mosaic assembly**        | Multi-panel stitching with overlap normalization                                    |
| **HDR composition**        | Multiple exposure blending for high-dynamic-range targets (e.g. Orion Trapezium)    |
| **Session learning**       | Persistent user preference model that improves parameter selection across datasets  |
| **Distributed processing** | Fan-out heavy operations (calibration, stacking) to multiple machines               |
| **Mono + filters**         | Full LRGB workflow with separate per-filter stacking and luminance blending         |
| **Quality report**         | Auto-generated processing report with before/after at each stage                    |


---

---

## 12  Processing Report

### 12.1 Purpose

`processing_report.md` is written to `{working_dir}/processing_report.md` at
export time. It is a human-readable narrative of every decision the agent made
from ingest through export — what tools were called, why, what parameters were
chosen, and what measurably changed. It serves three purposes:

1. **Debugging** — understand why the agent made a specific choice without
   reading raw Langsmith traces.
2. **Learning** — the report teaches astrophotography processing by showing
   which tools affect which image qualities, in what order, with what
   parameters.
3. **Prompt iteration** — patterns in the report (e.g. the agent consistently
   choosing overly aggressive deconvolution, or unnecessary tool calls) inform
   targeted improvements to tool contracts or the system prompt.

### 12.2 Rendered Format

```markdown
# AstroAgent Processing Report
**Target:** {target_name}
**Dataset:** {dataset.id}
**Completed:** {timestamp}
**Total steps:** {len(processing_report)}

---

## Pre-Processing

### Step 1 — build_masters
**Phase:** calibration
**Parameters:** `{"file_type": "bias", "stack_method": "median", ...}`
**Reasoning:** "Dataset has 20 bias frames. Using median stacking for lowest
noise floor. Sigma clipping rejection with 3σ to handle any hot frames."
**Outcome:** success — master_bias.fit written (mean=412.3, noise=2.1e)

### Step 2 — siril_calibrate
...

---

## Linear Processing

### Step 8 — remove_gradient
**Phase:** linear
**Parameters:** `{"backend": "graxpert", "graxpert_options": {"correction_type": "subtraction"}}`
**Reasoning:** "gradient_magnitude is 0.09, above the threshold that warrants
treatment. The bortle index is 6 suggesting significant light pollution. Using
GraXpert AI backend for complex gradient handling."
**Before:** gradient_magnitude=0.09, flatness_score=0.71
**After:**  gradient_magnitude=0.02, flatness_score=0.94
**Outcome:** success — gradient substantially removed

...

---

## Stretch

### Step 14 — stretch_image (HITL selection)
**User selected:** moderate (GHS D=2.5)
**User note:** "Good balance, stars not too prominent"

---

## Non-Linear Processing

### Step 15 — star_removal
**Reasoning:** "Target appears to be an emission nebula (Orion). Separating
stars will allow aggressive processing of the nebulosity without star bloat."
...

---

## Final Review
**User approved:** yes
**User notes:** "Excellent. Colors are natural, nebula detail is sharp."

---

## Summary
| Metric          | After Stack | After Linear | After Non-Linear |
|-----------------|-------------|--------------|------------------|
| background_noise| 0.0031      | 0.0018       | 0.0018           |
| snr_estimate    | 42.1        | 51.3         | 51.3             |
| gradient_mag    | 0.09        | 0.02         | 0.02             |
| fwhm (stars)    | 2.8         | 2.8          | 2.3              |
```

### 12.3 Implementation

`render_processing_report(state: AstroState) -> str` iterates
`state.processing_report`, groups entries by phase, and renders the markdown
above. It is called by the `export` node after `export_final` completes.
The file is always written alongside the exported TIFF/JPG — it is part of
the deliverable, not just a debug artifact.

---

*End of AstroAgent Final Specification*