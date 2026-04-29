"""
All state schemas for the Muphrid LangGraph.

TypedDicts are used for LangGraph state (required by StateGraph).
Pydantic BaseModels are used for tool input/output validation (see each tool file).

Annotated reducers:
  - history:            operator.add  → append-only list of strings
  - messages:           add_messages  → LangChain message merge (handles dedup)
  - processing_report:  operator.add  → append-only list of ReportEntry
  - paths:              merge_dicts   → deep-merge so parallel tool calls don't conflict
  - metadata:           merge_dicts   → deep-merge so parallel tool calls don't conflict
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Literal

from langgraph.graph.message import add_messages

# Pydantic 2.13+ requires typing_extensions.TypedDict on Python < 3.12 —
# typing.TypedDict's runtime introspection is missing fields the schema
# generator reads, producing "all fields missing" validation errors when
# langchain auto-generates a tool args schema that includes AstroState
# (e.g. on the async Gradio path). typing_extensions.TypedDict is a
# drop-in replacement and is correct on all supported Python versions.
from typing_extensions import NotRequired, TypedDict


def _merge_dicts(old: dict, new: dict) -> dict:
    """Deep-merge two dicts. new values overwrite old; nested dicts are merged recursively."""
    merged = dict(old)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


# ── Replace-aware reducers ────────────────────────────────────────────────────
#
# Some state fields need BOTH semantics:
#   * additive composition (so parallel tool calls each contribute their delta
#     and none of them clobber the others), AND
#   * full clear / replacement (so rewind_phase can reset the field to a
#     captured snapshot, and advance_phase can wipe phase-scoped state).
#
# A plain dict-merge reducer only supports composition; a plain replace
# reducer only supports clearing. The Replace sentinel below threads the
# needle: by default the reducer composes, but when it sees Replace(value)
# it returns value as-is, throwing away whatever was there before.
#
# Use Replace from clearer/restorer code paths (rewind_phase, advance_phase
# clears, etc.). Regular tool updates should emit plain deltas — the reducer
# composes them without losing siblings under parallel execution.
#
# This pattern is documented in CLAUDE.md "Tool state updates" section so
# that future contributors don't reintroduce the {**state["X"], "k": v}
# spread idiom that silently breaks parallel writes.

# Magic key used by the Replace sentinel. Chosen to be unique enough that
# accidental collision with a real dict key is implausible. Any dict with
# this key set to True is treated as a Replace wrapper by the reducers.
_REPLACE_MAGIC = "__muphrid_replace__"


def Replace(value):  # noqa: N802 — function-style API, mirrors a constructor
    """
    Sentinel telling Replace-aware reducers to fully replace the field
    with the wrapped value rather than merging.

    Usage:
        return Command(update={"metrics": Replace(restored_metrics)})

    Implementation note: Replace is a plain dict (`{"__muphrid_replace__":
    True, "value": value}`), not a custom class. This matters because
    LangGraph's checkpointer serializes intermediate writes via ormsgpack
    BEFORE the reducer collapses them — a custom class would raise
    `Type is not msgpack serializable`. A dict packs natively. The
    reducers below recognize the magic key and unwrap.
    """
    return {_REPLACE_MAGIC: True, "value": value}


def _is_replace(x) -> bool:
    """True iff x is a Replace-wrapped value."""
    return isinstance(x, dict) and x.get(_REPLACE_MAGIC) is True


def _replace_unwrap(x):
    """If x is a Replace wrapper, return its inner value; else return x."""
    if _is_replace(x):
        return x.get("value")
    return x


def _dict_merge_or_replace(old, new):
    """
    Reducer for dict-valued fields that need parallel-safe composition AND
    explicit replacement.

      * Replace(v)            → v (full replacement)
      * dict + dict           → deep merge (additive)
      * non-dict new          → return new (replace, matches default semantics)

    Used by the `metrics` field so multiple tools can each add their own
    keys in a single parallel super-step without stomping each other, while
    rewind_phase can still clear and restore via Replace(snapshot).

    Defensive on `old`: in normal LangGraph flow, the canonical channel
    value is always already a real dict (with the Replace wrapper unwrapped
    by the prior reducer pass). The unwrap here is for robustness — if a
    sequence of updates ever produces an unexpected shape, we treat a
    Replace-wrapped old as its inner value rather than silently dropping it.
    """
    old = _replace_unwrap(old)
    if _is_replace(new):
        return new.get("value")
    if isinstance(new, dict) and isinstance(old, dict):
        return _merge_dicts(old, new)
    return new


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

    # Authoritative render state — what space the pipeline has put the
    # current image into. "linear" before stretch_image runs (stack
    # output, gradient removal, color calibration, deconvolution all
    # preserve linear); "display" after stretch_image and through every
    # nonlinear tool that follows.
    #
    # READ SITES (Gradio preview generation, export_final source-profile
    # selection, future agent prompt) consult ONLY this field. They
    # never fall back to metrics.is_linear_estimate, never infer from
    # FITS HISTORY, never default to a "safe" value when missing. A
    # missing image_space at a read site means the checkpoint predates
    # this contract — the run refuses with a clear message rather than
    # silently producing wrong renders.
    #
    # WRITERS — every tool whose Command.update writes paths.current_image
    # MUST also write metadata.image_space. The structural drift check at
    # registry import time refuses to start the system if any current_image
    # writer omits it. See `muphrid.graph.registry._assert_image_space_writers`.
    #
    # Distinct from `metrics.is_linear_estimate`, which is analyze_image's
    # diagnostic "what the data looks like." That stays as observation;
    # image_space is the contract.
    image_space:        Literal["linear", "display"]
    checkpoints:        dict[str, dict] | None  # agent-named image state bookmarks; entries are {"path": str, "image_space": Literal["linear","display"]}

    # Last variant committed to current_image — set by both HITL promote_variant
    # and the autonomous commit_variant tool. Lets commit_variant detect the
    # "already committed via HITL" race (variant_pool is empty but the variant
    # was promoted) and return idempotent success instead of a confusing error.
    # Keys: {"id": str, "file_path": str}. None before any commit.
    last_committed_variant: dict | None

    # Snapshot of the metrics dict from the most recent analyze_image call.
    # Persists across tool calls so the next analyze_image can compare
    # current measurements against it and surface regression_warnings.
    # Updated by analyze_image on every successful run; cleared by phase
    # transitions (baselines don't carry across phases — different metrics
    # become meaningful in different phases).
    last_analysis_snapshot: dict | None

    # Phase-boundary state snapshots, keyed by ProcessingPhase value.
    # Written by advance_phase as the pipeline transitions: the snapshot
    # under key X represents the working state at the moment the pipeline
    # entered phase X (= the end of the prior phase). rewind_phase reads
    # these to restore the working state when the agent decides a phase
    # mistake originated upstream. See PhaseSnapshot for the captured
    # field set.
    phase_checkpoints: dict[str, "PhaseSnapshot"] | None

    # Per-target rewind counter. rewind_phase increments the count for the
    # target phase on every successful rewind and refuses when the count
    # is already ≥ 1. The first rewind to a phase is the safety net; a
    # second would suggest a structural problem the framework can't fix
    # on its own — the agent is expected to surface that situation rather
    # than loop on rewinds.
    phase_rewind_counts: dict[str, int] | None

    # Export bookkeeping — written by t24_export and backend export approval.
    #   export_done       : True after a successful direct export OR after
    #                       commit_export promotes a tentative export.
    #   exported_files    : list of {"path", "format", "icc_profile",
    #                       "file_size_mb"} for the final-location files.
    #   tentative_export  : non-None while a HITL-gated tentative export
    #                       is staged but not yet committed. Carries the
    #                       staging dir, final dir, file list, and
    #                       preview_jpg path so commit_export and the
    #                       gate's preview consumer agree on the artifact.
    export_done:        bool | None
    exported_files:     list | None
    tentative_export:   dict | None


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
    wavelet_noise:       float | None
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

    # Structured additions from analyze_image. See T20 docstring for the
    # full definition of each. All optional so legacy callers keep working.
    pixel_coverage:        dict | None            # total / n_valid / valid_pct / etc.
    clipping_per_channel:  dict | None            # per-channel clip staircase at chosen thresholds
    mode_estimate:         dict | None            # histogram-peak per channel + luminance
    background_quadrants:  dict | None            # per-quadrant bg/noise/valid
    wavelet_noise_scales:  list | None            # noise at multiple wavelet scales
    channel_snr:           dict | None            # per-channel P95 / MAD SNR
    star_distribution:     dict | None            # FWHM/peak/roundness percentiles
    bg_box_size:           int | None             # Background2D tile size actually used
    histogram:             dict | None            # per-channel percentile summary
    analyze_params:        dict | None            # echoed call parameters for reproducibility


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


class PhaseSnapshot(TypedDict):
    """
    A frozen slice of working state captured at a phase boundary.

    Written by advance_phase into metadata.phase_checkpoints[<new_phase>]
    at the moment the pipeline transitions: the snapshot under key X
    represents "the state at the start of phase X" (equivalently, the
    end of the prior phase). rewind_phase reads these to roll the
    working state back to a known-good boundary when the agent decides
    a phase-level mistake originated upstream.

    Captured fields are everything that defines the working "scratchpad"
    of the pipeline at a moment in time. The narrative — messages,
    processing_report, processing_log.md — is intentionally NOT captured;
    it stays append-only across rewinds so the agent (and any later
    reviewer) can see what the abandoned phase tried and why it was
    abandoned. metadata.phase_checkpoints itself is also excluded from
    the metadata snapshot to avoid recursive growth across rewinds.

    Field semantics:
      paths              — full PathState dict (current_image, sequence
                            chain, masters, mask, preview pointers, etc.)
      metrics            — full Metrics dict
      metadata           — Metadata dict minus phase_checkpoints and
                            last_analysis_snapshot (which is itself a
                            transient analyze_image baseline)
      regression_warnings — pending warnings at the moment of advance
      variant_pool       — typically empty (advance_phase clears it)
                            but captured for safety
      visual_context     — typically empty (advance_phase clears it)
                            but captured for safety
      captured_at        — ISO 8601 timestamp
      captured_from_phase — the phase that just ended, for diagnostics
    """
    paths:                  dict
    metrics:                dict
    metadata:               dict
    regression_warnings:    list
    variant_pool:           list
    visual_context:         list
    captured_at:            str
    captured_from_phase:    str


class RegressionWarning(TypedDict):
    """
    An image-quality metric that worsened between two analyze_image calls.

    Written by analyze_image when a current measurement has degraded vs the
    baseline in metadata.last_analysis_snapshot. Informational — the agent
    reads these to decide whether to revert (restore_checkpoint / rewind_phase)
    or accept the tradeoff. Nothing in the framework blocks on their presence.

    Lifecycle:
      - Emitted by analyze_image when a metric breaks the configured
        deterioration threshold for its direction.
      - Cleared individually when a subsequent analyze_image shows the metric
        has returned within tolerance of the warning's own baseline value.
      - Cleared en-masse by restore_checkpoint, rewind_phase, and advance_phase
        (the current-phase warnings are finalized into processing_log.md at
        the advance moment, then state starts fresh in the new phase).
    """
    metric:           str               # canonical metric key, e.g. "clipped_highlights_pct"
    baseline:         float | int | None # value at the baseline analyze_image
    current:          float | int | None # value at the detecting analyze_image
    delta:            float              # signed change (current − baseline)
    relative_delta:   float | None       # (current − baseline) / baseline, when baseline ≠ 0
    direction:        str                # "worse" — always, since only worsening triggers a warning
    summary:          str                # "Highlight clipping rose 0.1% → 3.2% (+3.1%)"
    phase_origin:     str                # ProcessingPhase at detection time
    detected_at:      str                # ISO 8601 timestamp


class VisualRef(TypedDict):
    """
    One image the agent should see, beyond what's already in variant_pool.

    visual_context (on AstroState) holds the NON-variant working set for the
    VLM: things the agent looked at via present_images, or images carried
    forward across a HITL gate close. The active HITL gate's variants are
    NOT stored here — they live in state.variant_pool (the canonical store)
    and are projected to VisualRefs at view-build time by _select_visible_refs.

    Together with variant_pool, visual_context is what _select_visible_refs
    reads to build the agent's multimodal HumanMessage at every model.invoke
    call. Messages stay text-only; state owns visibility.

    Sources currently in visual_context:
      "present_images" — captured by the present_images tool when the agent
                         decides to inspect specific images. Replaced on
                         every present_images call.
      "phase_carry"    — written by promote_variant on HITL approval to keep
                         the chosen variant visible after the gate closes,
                         giving the agent a stable visual anchor on what it
                         carried forward.
    """
    path:   str          # JPG/PNG preview path on disk (not FITS)
    label:  str          # short human-readable label
    source: str          # "present_images" | "phase_carry"
    phase:  str          # ProcessingPhase value when added (for diagnostics)


class Variant(TypedDict):
    """
    One concrete result the agent has produced for the current HITL gate.

    A variant is created automatically when a HITL-mapped tool finishes
    executing successfully — the action node snapshots the tool's output
    and a slice of relevant metrics, and appends an entry to state.variant_pool.

    Pool semantics:
      - Pool is per-HITL-gate. It accumulates as the agent iterates within a
        phase and is cleared when the human (or autonomous mode) commits one.
      - Files are written under runs/<thread_id>/variants/<id>.fits and
        survive the active session. They are not touched by prune logic.
      - The agent reads the pool indirectly: each HITL-tool ToolMessage is
        enriched with a formatted pool summary so the agent has a stable,
        indexed view of its own attempts without re-reading prior messages.
      - On approval, the chosen variant's file is promoted to current_image
        and the pool is cleared. Promotion happens in hitl_check, not in a
        tool, so the agent's tool trajectory is unchanged.

    See variant_pool design discussion for the full rationale.
    """
    id:           str           # stable id, e.g. "T09_v3"
    phase:        str           # ProcessingPhase value when the variant was created
    tool_name:    str           # the tool that produced it (e.g. "remove_gradient")
    label:        str           # auto-generated human-readable label
    params:       dict          # exact args passed to the tool
    file_path:    str           # FITS path of the snapshotted result
    preview_path: str | None    # JPG preview path if one exists
    metrics:      dict          # snapshot of relevant metrics at capture time
    created_at:   str           # ISO 8601
    rationale:    str | None    # populated only after this variant is committed


class ReviewHumanEvent(TypedDict, total=False):
    """
    A typed human event delivered while a HITL review session is paused.

    This replaces treating every resume value as ambiguous text. Chat
    questions, revision feedback, and approval clicks remain model-visible,
    but the graph controller can apply the correct state transition without
    parsing UX intent out of prose.
    """
    type: Literal["question", "feedback", "approve_variant", "approve_current"]
    text: str
    variant_id: NotRequired[str]
    rationale: NotRequired[str]
    received_at: NotRequired[str]


class ReviewProposalCandidate(TypedDict):
    """One approvable candidate in a review proposal artifact."""
    variant_id:   str
    rationale:    str
    presented_at: str


class ReviewProposal(TypedDict, total=False):
    """
    First-class artifact rendered by clients during a HITL review.

    This artifact is the canonical UI and approval contract: which candidates
    are approvable, what the agent recommends, and what tradeoffs or metrics
    should be shown alongside the images.
    """
    candidates:        list[ReviewProposalCandidate]
    recommendation:    str | None
    rationale:         str
    tradeoffs:         list[str]
    metric_highlights: dict
    updated_at:        str


class ReviewSession(TypedDict, total=False):
    """
    Explicit HITL Review Mode state.

    This session is the source of truth for open review gates, turn policy,
    proposal artifacts, and typed human events.
    """
    gate_id:               str
    hitl_key:              str
    tool_name:             str
    phase:                 str
    title:                 str
    status:                Literal[
        "review_open",
        "awaiting_agent_response",
        "awaiting_curation",
        "awaiting_human_approval",
        "closed",
    ]
    opened_at:             str
    updated_at:            str
    closed_at:             NotRequired[str]
    close_reason:          NotRequired[str]
    last_human_event:      ReviewHumanEvent | None
    turn_policy:           str
    tool_runs_since_human: int
    visible_response_required: bool
    proposal:              ReviewProposal


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

      bortle        — Optional. Bortle scale of the imaging site (1–9).
                      1–3 = rural dark sky, 4–5 = rural/suburban transition,
                      6–7 = suburban, 8–9 = urban. When None/unknown, the agent
                      uses analyze_image metrics to assess sky quality directly
                      from the data.

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
    bortle:       int | None   # 1–9 Bortle scale. None = unknown, use analyze_image data instead.
    sqm_reading:  float | None # optional: SQM-L reading in mag/arcsec² (e.g. 20.8)
    remove_stars: bool | None  # True/False/None=ask via HITL
    notes:        str | None   # optional free-text session notes


# ── HITL payload ───────────────────────────────────────────────────────────────

class HITLPayload(TypedDict):
    """
    Stable contract between hitl_check and any presenter (CLI, Gradio, etc.).

    hitl_check calls interrupt(HITLPayload) and never does I/O.
    The caller reads this payload and handles all presentation.

    Two variant lists, two distinct UI surfaces:
      * variant_pool — every variant produced this segment. Read-only
        history; presenters render it as a "what the agent has tried"
        panel without Approve buttons.
      * proposal — the agent's curated subset from review_session.proposal.
        Each entry pairs a Variant with the rationale the agent supplied when
        surfacing it. Approve buttons render here.
    """
    type:           str             # "data_review", "image_review", or "agent_chat"
    title:          str             # human-readable title (from hitl_config.toml)
    tool_name:      str             # the tool that triggered this checkpoint
    images:         list[str]       # image paths produced by the tool (for image_review)
    context:        list            # recent messages for continuity (last N)
    agent_text:     str             # agent's response text (for multi-turn HITL display)
    variant_pool:   list[Variant]   # passive history — every variant produced this segment
    proposal:       list[dict]      # agent's curation: each item = {variant, rationale, presented_at}
    review_session: NotRequired[ReviewSession | None]  # canonical Review Mode state
    review_state:   NotRequired[str]  # "ready" or "needs_curation"
    approval_allowed: NotRequired[bool]  # false when pool is only observational


# ── Top-level graph state ──────────────────────────────────────────────────────

class AstroState(TypedDict):
    # Human-provided upfront context — available to every node from the first step
    session: SessionContext

    # Core
    dataset:    Dataset
    phase:      ProcessingPhase
    paths:      Annotated[PathState, _merge_dicts]
    metadata:   Annotated[Metadata, _merge_dicts]
    # metrics uses Replace-aware deep-merge: tools emit per-call deltas
    # and parallel writers compose without clobbering siblings;
    # rewind_phase wraps in Replace() to fully reset on backtrack.
    metrics:    Annotated[Metrics, _dict_merge_or_replace]

    # Append-only logs — use operator.add reducer so updates accumulate
    history:            Annotated[list[str],         operator.add]
    processing_report:  Annotated[list[ReportEntry], operator.add]

    # LangChain message list — add_messages handles dedup and merging
    messages: Annotated[list, add_messages]

    # Accumulated HITL preferences — written by hitl_node, read by planner
    user_feedback: dict

    # Derived HITL conversation flag for logs/legacy inspection. Review policy
    # must read review_session instead.
    active_hitl: bool

    # Canonical HITL review session. This makes review state explicit instead
    # of inferring gate identity, turn policy, and proposal eligibility from
    # recent messages.
    review_session: ReviewSession | None

    # Variant pool — passive history of every variant produced this segment.
    # Built automatically by variant_snapshot from HITL-mapped tool execution.
    # The UI shows this as a read-only "what has the agent tried" panel — it
    # is NOT the approve set. Cleared on phase advance and variant commit.
    # Default LangGraph "replace" semantics: nodes return the full new list.
    variant_pool: list[Variant]

    # Visual context — non-variant working set of images for the VLM
    # (present_images calls, phase-carry anchors). Active HITL gate variants
    # live in variant_pool, not here — _select_visible_refs reads BOTH at
    # view-build time. Messages stay text-only; state owns visibility.
    # See VisualRef for source semantics.
    visual_context: list[VisualRef]

    # Active regression warnings — metrics that worsened between the baseline
    # analyze_image and the most recent one. Informational: the agent reads
    # these to decide whether to revert or accept. The framework surfaces
    # them in advance_phase's response and writes them into processing_log.md
    # at phase transitions but never blocks on them. Managed by analyze_image
    # (emit, auto-clear-on-recovery), restore_checkpoint / rewind_phase
    # (clear on rollback), and advance_phase (clear on transition after
    # logging). Default LangGraph "replace" semantics — producers return the
    # full new list each call. See RegressionWarning for entry shape.
    regression_warnings: list[RegressionWarning]


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
            # New runs always start in linear: ingest produces linear data,
            # the entire preprocess + linear pipeline preserves it, only
            # stretch_image flips this to "display". See Metadata.image_space.
            image_space="linear",
            checkpoints=None,
            last_committed_variant=None,
            last_analysis_snapshot=None,
            phase_checkpoints=None,
            phase_rewind_counts=None,
        ),
        metrics=Metrics(
            frame_stats={},
            frame_summary=None,
            current_fwhm=None,
            current_background=None,
            current_noise=None,
            wavelet_noise=None,
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
            # Structured additions — see T20 docstring.
            pixel_coverage=None,
            clipping_per_channel=None,
            mode_estimate=None,
            background_quadrants=None,
            wavelet_noise_scales=None,
            channel_snr=None,
            star_distribution=None,
            bg_box_size=None,
            histogram=None,
            analyze_params=None,
        ),
        history=[],
        processing_report=[],
        messages=[],
        user_feedback={},
        active_hitl=False,
        review_session=None,
        variant_pool=[],
        visual_context=[],
        regression_warnings=[],
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
    from muphrid.equipment import load_equipment

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
    bortle = session.get('bortle')
    if bortle:
        lines.append(f"- Bortle: {bortle}")
    else:
        lines.append("- Bortle: unknown — rely on analyze_image metrics to assess sky quality and guide processing decisions")
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
