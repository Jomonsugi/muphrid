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
    current_image:      str | None   # FITS path of the working image — always current
    latest_preview:     str | None   # JPG path of the most recent preview — HITL only
    starless_image:     str | None   # set after star_removal (T15)
    star_mask:          str | None   # set after star_removal (T15)
    masters:            MasterPaths
    pre_gradient_image: str | None   # snapshot before T09 — HITL before/after comparison
    pre_decon_image:    str | None   # snapshot before T13 — HITL before/after comparison

    # Preprocessing sequence chain — each tool writes its output name here
    # so the next tool can read it via InjectedState without the LLM tracking paths.
    lights_sequence:      str | None        # sequence name after T02b → read by T03
    calibrated_sequence:  str | None        # sequence name after T03  → read by T04, T05
    registered_sequence:  str | None        # sequence name after T04  → read by T07
    selected_frames:      list[str] | None  # accepted frame keys after T06 → read by T07

    # Latest mask FITS path — written by T25, optionally read by T27
    latest_mask:          str | None


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
    frame_summary:       dict | None              # summary stats from T05 (median_fwhm, etc.)

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
    Stable contract between hitl_check and any presenter (CLI, Streamlit, etc.).

    hitl_check calls interrupt(HITLPayload) and never does I/O.
    The caller reads this payload and handles all presentation.
    """
    type:           str           # "data_review" or "image_review" (from hitl_config.toml)
    title:          str           # human-readable title (from hitl_config.toml)
    tool_name:      str           # the tool that triggered this checkpoint
    images:         list[str]     # image paths produced by the tool (for image_review)
    context:        list          # recent messages for continuity (last N)


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

    # Active HITL conversation flag — set by hitl_check when interrupt fires,
    # cleared on approval. While True, the agent routes back to hitl_check
    # (not agent_chat) even when it responds without tool calls, so the
    # human can chat freely before approving or requesting revision.
    active_hitl: bool


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
            pre_gradient_image=None,
            pre_decon_image=None,
            lights_sequence=None,
            calibrated_sequence=None,
            registered_sequence=None,
            selected_frames=None,
            latest_mask=None,
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
            frame_summary=None,
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
        active_hitl=False,
    )


def build_initial_message(dataset: Dataset, session: SessionContext, ingest_summary: dict) -> str:
    """
    Build the initial HumanMessage content with full dataset context.

    This is the agent's ONLY source of truth about the dataset it's working
    with. Without this, the agent has no idea what frames are available,
    what camera was used, or what parameters are appropriate.

    Everything a human post-processor would know when sitting down to process
    must be in this message: camera, sensor, optics, frame inventory, exposure,
    sky conditions, sensor dynamic range, equipment profile, and calibration
    strategy.
    """
    from astro_agent.equipment import load_equipment

    meta = dataset.get("acquisition_meta", {})
    files = dataset.get("files", {})
    equipment = load_equipment()
    equip_camera = equipment.get("camera", {})
    equip_optics = equipment.get("optics", {})

    lines = [
        f"Process the astrophotography dataset for **{session['target_name']}**.",
        "",
        "## Dataset",
        f"- Lights: {len(files.get('lights', []))} frames",
        f"- Darks: {len(files.get('darks', []))} frames",
        f"- Flats: {len(files.get('flats', []))} frames",
        f"- Biases: {len(files.get('biases', []))} frames",
    ]

    exp = meta.get("exposure_time_s")
    if exp:
        total = exp * len(files.get("lights", []))
        lines.append(f"- Exposure: {exp}s per frame, {total / 60:.1f} min total integration")

    # Detected file extensions from ingest
    detected_ext = ingest_summary.get("detected_extensions")
    if detected_ext:
        ext_str = ", ".join(sorted(detected_ext)) if isinstance(detected_ext, (list, set)) else str(detected_ext)
        lines.append(f"- File extensions: {ext_str}")

    lines.append("")
    lines.append("## Acquisition")

    if meta.get("camera_model"):
        lines.append(f"- Camera: {meta['camera_model']}")
    if meta.get("sensor_type"):
        lines.append(f"- Sensor: {meta['sensor_type'].upper()}")
    if meta.get("iso"):
        lines.append(f"- ISO: {meta['iso']}")
    if meta.get("gain") is not None:
        lines.append(f"- Gain: {meta['gain']}")
    if meta.get("focal_length_mm"):
        lines.append(f"- Focal length: {meta['focal_length_mm']}mm")
    if meta.get("pixel_size_um"):
        lines.append(f"- Pixel size: {meta['pixel_size_um']}μm")
    if meta.get("bit_depth"):
        lines.append(f"- Bit depth: {meta['bit_depth']}-bit")
    if meta.get("input_format"):
        lines.append(f"- Input format: {meta['input_format'].upper()}")
    if meta.get("filter"):
        lines.append(f"- Filter: {meta['filter']}")

    # Sensor characterization — dynamic range context for the agent
    black = meta.get("black_level")
    white = meta.get("white_level")
    if black is not None and white is not None:
        lines.append("")
        lines.append("## Sensor Characterization")
        lines.append(f"- Black level: {black} ADU (pedestal)")
        lines.append(f"- White level: {white} ADU (full well)")
        usable = white - black
        lines.append(f"- Usable dynamic range: ~{usable} ADU levels")
        if meta.get("raw_exposure_bias") is not None:
            lines.append(f"- Raw exposure bias: {meta['raw_exposure_bias']} stops")

    # Equipment profile — shows the agent what values came from equipment.toml
    if equip_camera or equip_optics:
        lines.append("")
        lines.append("## Equipment Profile (equipment.toml)")
        if equip_camera.get("model"):
            lines.append(f"- Camera: {equip_camera['model']}")
        if equip_camera.get("sensor_type"):
            lines.append(f"- Sensor: {equip_camera['sensor_type']}")
        if equip_camera.get("pixel_size_um"):
            lines.append(f"- Pixel size: {equip_camera['pixel_size_um']} μm")
        if equip_optics.get("focal_length_mm"):
            lines.append(f"- Focal length: {equip_optics['focal_length_mm']} mm (plate-solve-measured)")

    # Ingest sensor summary (from T01's EXIF analysis)
    sensor_summary = ingest_summary.get("sensor")
    if sensor_summary and isinstance(sensor_summary, dict):
        # Only show if it adds info beyond what's already displayed
        sensor_model = sensor_summary.get("model")
        if sensor_model and sensor_model != meta.get("camera_model"):
            lines.append(f"- Sensor model (EXIF): {sensor_model}")

    lines.append("")
    lines.append("## Session")
    lines.append(f"- Bortle: {session.get('bortle', 'unknown')}")
    if session.get("sqm_reading"):
        lines.append(f"- SQM: {session['sqm_reading']} mag/arcsec²")
    if session.get("remove_stars") is not None:
        lines.append(f"- Star removal: {'yes' if session['remove_stars'] else 'no'}")
    if session.get("notes"):
        lines.append(f"- Notes: {session['notes']}")

    # Warnings from ingest
    warnings = ingest_summary.get("warnings") or []
    cleaned = ingest_summary.get("cleaned_artifacts") or []
    if warnings:
        lines.append("")
        lines.append("## Ingest warnings")
        for w in warnings:
            lines.append(f"- {w}")
    if cleaned:
        lines.append(f"- Cleaned stale artifacts: {', '.join(cleaned)}")

    # Calibration frame availability drives the agent's strategy
    lines.append("")
    lines.append("## Calibration strategy")
    has_darks = len(files.get("darks", [])) > 0
    has_flats = len(files.get("flats", [])) > 0
    has_biases = len(files.get("biases", [])) > 0

    if has_biases and has_darks and has_flats:
        lines.append(
            "Full calibration: call build_masters(file_type='bias'), "
            "then build_masters(file_type='dark'), "
            "then build_masters(file_type='flat'), then calibrate lights."
        )
    elif has_darks and has_flats:
        lines.append(
            "No bias frames. Call build_masters(file_type='dark'), "
            "then build_masters(file_type='flat'), calibrate without bias subtraction."
        )
    elif has_darks:
        lines.append("Darks only. Call build_masters(file_type='dark'). No flats — vignetting/dust correction will be skipped.")
    elif has_flats:
        lines.append("Flats only. Call build_masters(file_type='flat'). No darks — thermal noise correction will be skipped.")
    else:
        lines.append("No calibration frames. Proceed directly to convert_sequence and registration.")

    lines.append("")
    lines.append("Begin preprocessing.")

    return "\n".join(lines)
