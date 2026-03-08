"""
All state schemas for the AstroAgent LangGraph.

TypedDicts are used for LangGraph state (required by StateGraph).
Pydantic BaseModels are used for tool input/output validation (see each tool file).

Annotated reducers:
  - history:            operator.add  → append-only list of strings
  - messages:           add_messages  → LangChain message merge (handles dedup)
  - processing_report:  operator.add  → append-only list of ReportEntry
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated

from langgraph.graph.message import add_messages
from typing import TypedDict


# ── Processing phases ──────────────────────────────────────────────────────────

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


def is_linear_phase(phase: ProcessingPhase) -> bool:
    """True for phases where pixel values are still proportional to photon flux."""
    return phase in {
        ProcessingPhase.INGEST,
        ProcessingPhase.CALIBRATION,
        ProcessingPhase.REGISTRATION,
        ProcessingPhase.ANALYSIS,
        ProcessingPhase.STACKING,
        ProcessingPhase.LINEAR,
        ProcessingPhase.STRETCH,   # stretch itself is the transition point
    }


# ── Sub-schemas ────────────────────────────────────────────────────────────────

class MasterPaths(TypedDict):
    bias: str | None
    dark: str | None
    flat: str | None


class PathState(TypedDict):
    current_image:  str | None   # FITS path of the working image — always current
    latest_preview: str | None   # JPG path of the most recent preview — HITL only
    starless_image: str | None   # set after star_removal (T15)
    star_mask:      str | None   # set after star_removal (T15)
    masters:        MasterPaths
    variants:       dict[str, str]  # A/B stretch variants: {"gentle": path, ...}


class Metadata(TypedDict):
    is_linear:          bool
    is_color:           bool         # False for mono sensors
    is_osc:             bool         # True for OSC / DSLR (CFA sensor)
    pixel_scale:        float | None # arcsec/pixel (from plate solve)
    plate_solve_coords: dict | None  # {"ra": float, "dec": float}
    focal_length_mm:    float | None
    pixel_size_um:      float | None


class FrameMetrics(TypedDict, total=False):
    fwhm:             float | None
    weighted_fwhm:    float | None
    roundness:        float | None
    quality:          float | None
    background_lvl:   float | None
    number_of_stars:  int | None
    mean:             float | None
    median:           float | None
    sigma:            float | None
    bgnoise:          float | None


class Metrics(TypedDict):
    frame_stats:         dict[str, FrameMetrics]  # keyed by filename

    # Core image quality (from T20 analyze_image)
    current_fwhm:        float | None
    current_background:  float | None
    current_noise:       float | None
    snr_estimate:        float | None
    dynamic_range_db:    float | None

    # Per-channel statistics (from T20 channels)
    channel_stats:       dict | None              # per-channel mean/median/std

    # Background characterization
    background_flatness: float | None             # 0–1, 1=perfectly flat
    gradient_magnitude:  float | None             # 0–1, >0.05 = gradient present
    per_channel_bg:      dict | None              # {red, green, blue, n_background_pixels}

    # Color balance
    green_excess:        float | None
    channel_imbalance:   float | None             # max - min channel mean

    # Color saturation (HSV-based, drives T18)
    mean_saturation:     float | None
    median_saturation:   float | None

    # Linearity state (dual-consensus: median + histogram skewness)
    is_linear_estimate:  bool | None
    linearity_confidence: str | None              # "high" / "medium" / "low"
    histogram_skewness:  float | None

    # Signal coverage (fraction of frame containing nebula/galaxy signal)
    signal_coverage_pct: float | None

    # Clipping (worst channel, percentage)
    clipped_shadows_pct:    float | None
    clipped_highlights_pct: float | None

    # Star metrics (from T20 photutils)
    star_count:          int | None
    fwhm_std:            float | None             # PSF uniformity indicator
    median_star_peak_ratio: float | None          # star dominance indicator

    # Contrast (p95 - p5 luminance range)
    contrast_ratio:      float | None


class AcquisitionMeta(TypedDict):
    target_name:      str | None
    target_coords:    dict | None   # {"ra": float, "dec": float} J2000, from T29 resolve_target
    focal_length_mm:  float | None
    pixel_size_um:    float | None
    exposure_time_s:  float | None   # per-frame exposure in seconds
    iso:              int | None     # DSLR/mirrorless ISO
    gain:             int | None     # dedicated astro camera gain (ADU)
    filter:           str | None     # filter name (FITS FILTER keyword; None for camera RAW)
    bortle:           int | None
    camera_model:     str | None
    telescope:        str | None
    input_format:     str | None     # "fits" | "raw" — set by T01, read by T03

    # Sensor characterization — populated by T01, used by T02 for sensor-relative thresholds
    black_level:       int | None    # pedestal ADU (e.g. 1022 for Fuji X-T30 II)
    white_level:       int | None    # sensor full-well ADU (e.g. 16383 for 14-bit)
    bit_depth:         int | None    # raw bit depth: 12, 14, or 16
    raw_exposure_bias: float | None  # stops (Fuji RAF:RawExposureBias; None for non-Fuji)
    sensor_type:       str | None    # "bayer" | "xtrans" — affects T03 debayer kernel


class FileInventory(TypedDict):
    lights: list[str]
    darks:  list[str]
    flats:  list[str]
    biases: list[str]


class Dataset(TypedDict):
    id:               str
    working_dir:      str
    files:            FileInventory
    acquisition_meta: AcquisitionMeta


class ReportEntry(TypedDict):
    """One agent decision: tool called, why, with what parameters, and outcome."""
    step:           int          # sequential step number across the full run
    phase:          str          # ProcessingPhase value at time of call
    tool:           str          # tool name as it appears in TOOL_REGISTRY
    reasoning:      str          # LLM's stated reason, extracted from message content
    parameters:     dict         # exact parameter dict passed to the tool
    metrics_before: dict | None  # analyze_image snapshot before the call
    metrics_after:  dict | None  # analyze_image snapshot after the call
    outcome:        str          # "success" | "degraded" | "error" | "reverted"
    outcome_detail: str          # brief summary of what changed or went wrong
    timestamp:      str          # ISO 8601


# ── Session context (human-provided at startup) ────────────────────────────────

class SessionContext(TypedDict):
    """
    Human-provided context collected before the pipeline starts.

    This is the upfront knowledge that cannot be derived from pixel data or EXIF —
    it shapes the entire processing run. Collected via CLI arguments or interactive
    prompts before the agent's ReAct loop begins.

    Fields:
      target_name   — Required. The astronomical target name as the user knows it
                      ("M42 Orion Nebula", "NGC 7000 North America", "M31 Andromeda",
                       "Milky Way core"). This single piece of context informs the
                      agent's domain knowledge about the expected appearance,
                      angular size, colour palette, and appropriate processing
                      intensity for the subject. A frontier LLM already knows what
                      M42 looks like — naming it unlocks that knowledge.

      bortle        — Required. Bortle scale of the imaging site (1–9).
                      1–3 = rural dark sky, 4–5 = rural/suburban transition,
                      6–7 = suburban, 8–9 = urban. Look up at
                      lightpollutionmap.info if uncertain. Directly informs:
                       - gradient removal aggressiveness (high → larger gradient)
                       - noise reduction strength (high → more NL-Bayes passes)
                       - stretch conservatism (high → preserve faint structure)

      sqm_reading   — Optional. Sky Quality Meter reading in mag/arcsec²,
                      taken with an SQM-L at the imaging site before the session.
                      More precise than Bortle (continuous scale):
                        ≥ 21.5 → Bortle 1–2, truly dark sky
                        20.5–21.5 → Bortle 3–4, rural
                        19.5–20.5 → Bortle 5–6, suburban transition
                        < 19.5  → Bortle 7–9, suburban/urban
                      When provided, the agent uses sqm_reading as the primary
                      sky quality metric and bortle as a coarse cross-check.
                      If you forget to take a reading, leave it None — bortle
                      alone is sufficient.

      remove_stars  — User intent for star processing:
                       True  → run star_removal (T15) + star_restoration (T19)
                       False → skip T15/T19 entirely
                       None  → ask via HITL when the agent reaches that decision point
                      Default None — the agent uses HITL to ask a simple yes/no.

      notes         — Optional free-text session notes from the user. Injected into
                      the system prompt at every ReAct cycle. Use for anything that
                      affects processing but isn't captured elsewhere:
                        "Shot with Optolong L-eNhance duoband filter"
                        "Very poor seeing — FWHM likely > 4px"
                        "10-min subs, ASI2600, gain 100"
                        "Prioritise faint outer nebulosity over star quality"
    """
    target_name:  str          # required: "M42 Orion Nebula", "wide angle Milky Way core"
    bortle:       int          # required: 1–9 Bortle scale (1=darkest, 9=city)
    sqm_reading:  float | None # optional: SQM-L reading in mag/arcsec² (e.g. 20.8)
    remove_stars: bool | None  # True/False/None=ask via HITL
    notes:        str | None   # optional free-text session notes


# ── HITL payload ───────────────────────────────────────────────────────────────

class HITLPayload(TypedDict):
    """
    Stable contract between the graph (hitl_node) and any presenter
    (cli_presenter, future Streamlit presenter, etc.).

    hitl_node calls interrupt(payload: HITLPayload) and never does I/O.
    The caller reads this payload and handles all presentation.
    """
    trigger:        str           # e.g. "stretch_selection", "auto_deconvolution"
    checkpoint:     str           # which tool or phase triggered this interrupt
    question:       str           # question to present to the user
    options:        list[str] | None  # None → free-text response only
    allow_free_text: bool         # True allows typed response alongside options
    preview_paths:  list[str]     # absolute JPG paths (one per image to display)
    preview_labels: list[str]     # label for each preview (same length as paths)
    context:        str           # metric summary or other text context


# ── Top-level graph state ──────────────────────────────────────────────────────

class AstroState(TypedDict):
    # Human-provided upfront context — available to every node from the first step
    session: SessionContext

    # Core
    dataset:    Dataset
    phase:      ProcessingPhase
    paths:      PathState
    metadata:   Metadata
    metrics:    Metrics

    # Append-only logs — use operator.add reducer so updates accumulate
    history:            Annotated[list[str],         operator.add]
    processing_report:  Annotated[list[ReportEntry], operator.add]

    # LangChain message list — add_messages handles dedup and merging
    messages: Annotated[list, add_messages]

    # Accumulated HITL preferences — written by hitl_node, read by planner
    user_feedback: dict


# ── Factory ────────────────────────────────────────────────────────────────────

def make_empty_state(dataset: Dataset, session: SessionContext) -> AstroState:
    """
    Build a minimal valid AstroState for a new dataset.
    Called by make_initial_state() in cli.py after T01 ingest_dataset runs.

    session is the human-provided startup context: target_name (required),
    bortle (optional), remove_stars intent (optional), and free-text notes.
    """
    input_fmt = dataset.get("acquisition_meta", {}).get("input_format")
    is_osc = input_fmt == "raw"  # camera RAW is always OSC/CFA

    return AstroState(
        session=session,
        dataset=dataset,
        phase=ProcessingPhase.INGEST,
        paths=PathState(
            current_image=None,
            latest_preview=None,
            starless_image=None,
            star_mask=None,
            masters=MasterPaths(bias=None, dark=None, flat=None),
            variants={},
        ),
        metadata=Metadata(
            is_linear=True,
            is_color=True,
            is_osc=is_osc,
            pixel_scale=None,
            plate_solve_coords=None,
            focal_length_mm=None,
            pixel_size_um=None,
        ),
        metrics=Metrics(
            frame_stats={},
            current_fwhm=None,
            current_background=None,
            current_noise=None,
            snr_estimate=None,
            dynamic_range_db=None,
            channel_stats=None,
            background_flatness=None,
            gradient_magnitude=None,
            per_channel_bg=None,
            green_excess=None,
            channel_imbalance=None,
            mean_saturation=None,
            median_saturation=None,
            is_linear_estimate=True,
            linearity_confidence="high",
            histogram_skewness=None,
            signal_coverage_pct=None,
            clipped_shadows_pct=None,
            clipped_highlights_pct=None,
            star_count=None,
            fwhm_std=None,
            median_star_peak_ratio=None,
            contrast_ratio=None,
        ),
        history=[],
        processing_report=[],
        messages=[],
        user_feedback={},
    )
