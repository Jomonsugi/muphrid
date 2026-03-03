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


class FrameMetrics(TypedDict):
    fwhm:             float
    eccentricity:     float
    star_count:       int
    background_level: float
    noise_estimate:   float
    roundness:        float


class Metrics(TypedDict):
    frame_stats:         dict[str, FrameMetrics]  # keyed by filename
    current_fwhm:        float | None
    current_background:  float | None
    current_noise:       float | None
    snr_estimate:        float | None
    channel_stats:       dict | None              # per-channel mean/median/std
    background_flatness: float | None
    green_excess:        float | None


class AcquisitionMeta(TypedDict):
    target_name:     str | None
    focal_length_mm: float | None
    pixel_size_um:   float | None
    iso:             int | None
    gain:            int | None
    bortle:          int | None
    camera_model:    str | None
    telescope:       str | None


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

def make_empty_state(dataset: Dataset) -> AstroState:
    """
    Build a minimal valid AstroState for a new dataset.
    Called by make_initial_state() in cli.py after T01 ingest_dataset runs.
    """
    return AstroState(
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
            is_osc=False,
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
            channel_stats=None,
            background_flatness=None,
            green_excess=None,
        ),
        history=[],
        processing_report=[],
        messages=[],
        user_feedback={},
    )
