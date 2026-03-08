"""
T04 — siril_register

Align calibrated light frames to a common reference frame using star matching.

This tool exposes the full Siril registration pipeline:
  1. setfindstar  — configure star detection parameters (PSF fitting, thresholds)
  2. register     — detect stars, compute geometric transforms (-2pass by default)
  3. seqapplyreg  — apply transforms to produce registered output sequence,
                    with optional per-metric filtering

After registration, Siril writes per-frame metrics (FWHM, wFWHM, roundness,
quality, background, star count) as R-lines in the *input* sequence's .seq
file.  This tool returns the calibrated_sequence name so T05 can parse it.

Siril docs:
    register <seq> [-2pass] [-selected] [-prefix=] [-scale=]
        [-layer=] [-transf=] [-minpairs=] [-maxstars=] [-nostarlist] [-disto=]
        [-interp=] [-noclamp]
        [-drizzle [-pixfrac=] [-kernel=] [-flat=]]

    seqapplyreg <seq> [-prefix=] [-scale=] [-layer=] [-framing=]
        [-interp=] [-noclamp]
        [-drizzle [-pixfrac=] [-kernel=] [-flat=]]
        [-filter-fwhm=val[%|k]] [-filter-wfwhm=val[%|k]]
        [-filter-round=val[%|k]] [-filter-bkg=val[%|k]]
        [-filter-nbstars=val[%|k]] [-filter-quality=val[%|k]]
        [-filter-incl]

    setfindstar [reset] [-radius=] [-sigma=] [-roundness=] [-focal=]
        [-pixelsize=] [-convergence=] [-gaussian|-moffat] [-minbeta=]
        [-relax=on|off] [-minA=] [-maxA=] [-maxR=]
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Star detection configuration ──────────────────────────────────────────────

class SetFindStarOptions(BaseModel):
    """Star detection parameters passed to Siril's setfindstar command.

    Controls how stars are detected during registration — equivalent to the
    Dynamic PSF dialog in the Siril GUI.  Tuning these rescues registrations
    that fail with default settings (e.g. sparse fields, wide-field images,
    short exposures, non-round optics).
    """
    reset: bool = Field(
        default=False,
        description="Reset all setfindstar parameters to Siril defaults before applying.",
    )
    radius: int | None = Field(
        default=None,
        description=(
            "Radius of the initial star search box (3–50 pixels). "
            "Increase for wide-field images with large stars; decrease for "
            "narrow fields with tight PSFs."
        ),
    )
    sigma: float | None = Field(
        default=None,
        description=(
            "Detection threshold above noise (≥ 0.05). Higher values reduce "
            "false positives but may miss faint stars. Lower values detect "
            "more stars in noisy data."
        ),
    )
    roundness: float | None = Field(
        default=None,
        description=(
            "Minimum star roundness (0–0.95). Stars rounder than this are "
            "kept. Lower values accept more elongated objects."
        ),
    )
    max_roundness: float | None = Field(
        default=None,
        description=(
            "Upper bound for roundness. Use to visualize only areas with "
            "significantly elongated stars. Do not change for registration."
        ),
    )
    focal: float | None = Field(
        default=None,
        description="Telescope focal length in mm. Used for arcsec FWHM calculation.",
    )
    pixelsize: float | None = Field(
        default=None,
        description="Sensor pixel size in µm. Used for arcsec FWHM calculation.",
    )
    convergence: int | None = Field(
        default=None,
        description=(
            "PSF fitting iterations (1–3). Higher = more tolerant fitting. "
            "Increase for difficult star fields."
        ),
    )
    profile: str | None = Field(
        default=None,
        description=(
            "'gaussian' (default, faster) or 'moffat' (better for "
            "undersampled or seeing-dominated PSFs with extended wings)."
        ),
    )
    min_beta: float | None = Field(
        default=None,
        description=(
            "Moffat minimum beta (0–10). Only used when profile='moffat'. "
            "Lower values accept stars with broader wings."
        ),
    )
    relax: bool | None = Field(
        default=None,
        description=(
            "Relax star candidate checks to accept objects not shaped like "
            "stars. Useful for sparse fields where strict checks reject "
            "valid detections. Off by default."
        ),
    )
    min_amplitude: float | None = Field(
        default=None,
        description="Minimum normalized star amplitude (0–1). Filters out faint detections.",
    )
    max_amplitude: float | None = Field(
        default=None,
        description="Maximum normalized star amplitude (0–1). Filters out saturated stars.",
    )


def _build_setfindstar_cmd(opts: SetFindStarOptions) -> str | None:
    """Build a setfindstar command string, or None if no options set."""
    parts: list[str] = ["setfindstar"]

    if opts.reset:
        parts.append("reset")

    if opts.radius is not None:
        parts.append(f"-radius={opts.radius}")
    if opts.sigma is not None:
        parts.append(f"-sigma={opts.sigma}")
    if opts.roundness is not None:
        parts.append(f"-roundness={opts.roundness}")
    if opts.max_roundness is not None:
        parts.append(f"-maxR={opts.max_roundness}")
    if opts.focal is not None:
        parts.append(f"-focal={opts.focal}")
    if opts.pixelsize is not None:
        parts.append(f"-pixelsize={opts.pixelsize}")
    if opts.convergence is not None:
        parts.append(f"-convergence={opts.convergence}")
    if opts.profile == "moffat":
        parts.append("-moffat")
        if opts.min_beta is not None:
            parts.append(f"-minbeta={opts.min_beta}")
    elif opts.profile == "gaussian":
        parts.append("-gaussian")
    if opts.relax is True:
        parts.append("-relax=on")
    elif opts.relax is False:
        parts.append("-relax=off")
    if opts.min_amplitude is not None:
        parts.append(f"-minA={opts.min_amplitude}")
    if opts.max_amplitude is not None:
        parts.append(f"-maxA={opts.max_amplitude}")

    return " ".join(parts) if len(parts) > 1 else None


# ── seqapplyreg filter options ────────────────────────────────────────────────

class SeqApplyRegFilters(BaseModel):
    """Filtering options for seqapplyreg — select best frames during resampling.

    Each filter value is a string that supports three modes:
      - Absolute:    "3.5"    → reject frames worse than 3.5
      - Percentage:  "90%"    → keep the best 90% of frames
      - K-sigma:     "2.5k"   → keep frames within 2.5σ of the mean

    This mirrors the Siril UI stacking tab where you can take frames by
    percentage based on any metric.
    """
    fwhm: str | None = Field(
        default=None,
        description=(
            "Filter by FWHM. '90%' keeps best 90%; '3.5' rejects FWHM > 3.5; "
            "'2k' keeps within 2σ."
        ),
    )
    wfwhm: str | None = Field(
        default=None,
        description="Filter by weighted FWHM (same syntax as fwhm).",
    )
    roundness: str | None = Field(
        default=None,
        description=(
            "Filter by roundness. '90%' keeps best 90%; '0.6' rejects roundness < 0.6; "
            "'2k' keeps within 2σ."
        ),
    )
    background: str | None = Field(
        default=None,
        description="Filter by background level (same syntax). Catches cloudy frames.",
    )
    nbstars: str | None = Field(
        default=None,
        description="Filter by number of detected stars (same syntax).",
    )
    quality: str | None = Field(
        default=None,
        description="Filter by Siril's composite quality metric (same syntax).",
    )
    included_only: bool = Field(
        default=False,
        description="Only apply registration to manually included frames (select/unselect).",
    )


def _build_filter_flags(filters: SeqApplyRegFilters) -> list[str]:
    """Build seqapplyreg -filter-* flag strings."""
    flags: list[str] = []
    if filters.fwhm is not None:
        flags.append(f"-filter-fwhm={filters.fwhm}")
    if filters.wfwhm is not None:
        flags.append(f"-filter-wfwhm={filters.wfwhm}")
    if filters.roundness is not None:
        flags.append(f"-filter-round={filters.roundness}")
    if filters.background is not None:
        flags.append(f"-filter-bkg={filters.background}")
    if filters.nbstars is not None:
        flags.append(f"-filter-nbstars={filters.nbstars}")
    if filters.quality is not None:
        flags.append(f"-filter-quality={filters.quality}")
    if filters.included_only:
        flags.append("-filter-included")
    return flags


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SirilRegisterInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    calibrated_sequence: str = Field(
        description=(
            "Name of the calibrated sequence (without .seq extension). "
            "Typically 'pp_lights_seq' from T03."
        )
    )

    # ── register command options ──────────────────────────────────────────
    two_pass: bool = Field(
        default=True,
        description=(
            "Use -2pass registration: first pass finds the best reference image "
            "by quality and framing, second pass computes transforms. Recommended "
            "for all datasets. Disable only for manual reference image control."
        ),
    )
    transformation: str = Field(
        default="homography",
        description=(
            "Geometric transformation type. "
            "'homography' (default — handles rotation, scale, perspective), "
            "'affine' (rotation + scale, no perspective), "
            "'similarity' (rotation + uniform scale), "
            "'shift' (translation only — for stable tracking mounts)."
        ),
    )
    max_stars: int = Field(
        default=500,
        description=(
            "Maximum stars per frame for matching (100–2000). More stars = "
            "more accurate registration but slower. Increase for wide fields."
        ),
    )
    min_pairs: int | None = Field(
        default=None,
        description=(
            "Minimum star pairs a frame must share with the reference. "
            "Frames below this threshold are dropped and excluded. "
            "Increase for strict quality control; decrease for sparse fields."
        ),
    )
    layer: int | None = Field(
        default=None,
        description="Detection layer for color images (0=Red, 1=Green, 2=Blue). Default: green.",
    )
    scale: float | None = Field(
        default=None,
        description="Rescale output images (0.1–3.0). Default: no rescaling.",
    )
    selected_only: bool = Field(
        default=False,
        description="Only register frames that are currently marked as included in the sequence.",
    )
    no_starlist: bool = Field(
        default=False,
        description="Skip saving per-frame star lists to disk. Saves I/O on large datasets.",
    )
    disto: str | None = Field(
        default=None,
        description=(
            "Distortion correction from a prior platesolve solution. "
            "'image' (use loaded image's solution), "
            "'file <path>' (load from a specific FITS), "
            "'master' (auto-match per image). Requires SIP order > 1."
        ),
    )

    # ── Interpolation ─────────────────────────────────────────────────────
    interpolation: str = Field(
        default="lanczos4",
        description=(
            "Pixel interpolation for frame resampling. "
            "'lanczos4' (best quality, default), 'bicubic', 'bilinear', "
            "'linear', 'nearest', 'area', 'none' (pixel-wise shift only)."
        ),
    )
    no_clamp: bool = Field(
        default=False,
        description=(
            "Disable clamping for bicubic/lanczos4 interpolation. "
            "May produce ringing artifacts but preserves more detail."
        ),
    )

    # ── Drizzle ───────────────────────────────────────────────────────────
    drizzle: bool = Field(
        default=False,
        description=(
            "Enable drizzle integration for sub-pixel resolution. "
            "Only beneficial with ≥30 well-dithered frames. "
            "Input must be non-debayered for color cameras."
        ),
    )
    drizzle_pixfrac: float = Field(
        default=1.0,
        description="Drizzle pixel fraction (0.5–1.0). Lower = sharper but needs more frames.",
    )
    drizzle_kernel: str | None = Field(
        default=None,
        description=(
            "Drizzle kernel: 'square' (default), 'point', 'turbo', "
            "'gaussian', 'lanczos2', 'lanczos3'."
        ),
    )
    drizzle_flat: str | None = Field(
        default=None,
        description="Path to master flat for drizzle pixel weighting.",
    )

    # ── seqapplyreg options ───────────────────────────────────────────────
    framing: str = Field(
        default="min",
        description=(
            "Output framing strategy. "
            "'min' (intersection — safe for FITSEQ), "
            "'max' (union — larger canvas, may need -maximize in stack), "
            "'cog' (center-of-gravity weighted), "
            "'current' (use reference frame boundaries)."
        ),
    )
    prefix: str | None = Field(
        default=None,
        description="Override output sequence prefix (default: 'r_').",
    )

    # ── Filtering (during seqapplyreg) ────────────────────────────────────
    filters: SeqApplyRegFilters | None = Field(
        default=None,
        description=(
            "Frame filtering applied during resampling. Each filter supports "
            "absolute value, percentage (%), or k-sigma (k) modes. "
            "Example: filters.fwhm='90%' keeps the best 90% by FWHM."
        ),
    )

    # ── Star detection tuning ─────────────────────────────────────────────
    findstar: SetFindStarOptions | None = Field(
        default=None,
        description=(
            "Star detection parameters (setfindstar). Tune these when "
            "registration fails or detects too few stars. Equivalent to "
            "the Dynamic PSF dialog in Siril's GUI."
        ),
    )


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_register_output(stdout: str) -> dict:
    """Extract registration metrics from Siril stdout."""
    metrics: dict = {}

    reg_match = re.search(
        r"(\d+)\s+(?:images?\s+)?(?:frame[s]?\s+)?registered", stdout, re.IGNORECASE
    )
    if reg_match:
        metrics["registered_count"] = int(reg_match.group(1))

    fail_match = re.search(
        r"(\d+)\s+(?:images?\s+)?(?:frame[s]?\s+)?(?:failed|not registered)",
        stdout, re.IGNORECASE,
    )
    metrics["failed_count"] = int(fail_match.group(1)) if fail_match else 0

    ref_match = re.search(r"convergence.*image\s+(\d+)", stdout, re.IGNORECASE)
    if ref_match:
        metrics["reference_image_index"] = int(ref_match.group(1))

    star_matches = re.findall(r"(\d+)\s+stars?\s+(?:detected|found)", stdout, re.IGNORECASE)
    if star_matches:
        counts = [int(s) for s in star_matches]
        metrics["total_stars_detected"] = sum(counts)
        metrics["star_detection_count"] = len(counts)

    return metrics


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SirilRegisterInput)
def siril_register(
    working_dir: str,
    calibrated_sequence: str,
    two_pass: bool = True,
    transformation: str = "homography",
    max_stars: int = 500,
    min_pairs: int | None = None,
    layer: int | None = None,
    scale: float | None = None,
    selected_only: bool = False,
    no_starlist: bool = False,
    disto: str | None = None,
    interpolation: str = "lanczos4",
    no_clamp: bool = False,
    drizzle: bool = False,
    drizzle_pixfrac: float = 1.0,
    drizzle_kernel: str | None = None,
    drizzle_flat: str | None = None,
    framing: str = "min",
    prefix: str | None = None,
    filters: dict | None = None,
    findstar: dict | None = None,
) -> dict:
    """
    Register calibrated light frames using Siril star matching.

    Full registration pipeline: setfindstar (optional) → register → seqapplyreg.
    Returns the registered sequence name, the calibrated sequence name (for T05
    to parse registration data from), and registration summary metrics.

    The most important output for frame selection is the per-frame data written
    by Siril to the calibrated sequence's .seq file (R-lines). Use T05 with
    calibrated_sequence to extract it.

    Key tuning levers when registration struggles:
      - findstar.sigma: lower to detect fainter stars
      - findstar.radius: increase for wide-field / large-star images
      - findstar.convergence: increase (to 3) for difficult star fields
      - findstar.relax: True for sparse fields
      - max_stars: increase up to 2000 for dense fields
      - min_pairs: decrease for sparse fields with few stars
      - transformation: try 'shift' for well-tracked data, 'affine' for simpler geometry
    """
    commands: list[str] = []

    # ── 1. setfindstar preamble ───────────────────────────────────────────
    if findstar is not None:
        opts = findstar if isinstance(findstar, SetFindStarOptions) else SetFindStarOptions(**findstar)
        cmd = _build_setfindstar_cmd(opts)
        if cmd:
            commands.append(cmd)

    # ── 2. register command ───────────────────────────────────────────────
    transf_map = {
        "homography": "homography",
        "affine": "affine",
        "similarity": "similarity",
        "shift": "shift",
    }
    transf = transf_map.get(transformation, "homography")

    reg_parts = [f"register {calibrated_sequence}"]
    if two_pass:
        reg_parts.append("-2pass")
    reg_parts.append(f"-transf={transf}")
    reg_parts.append(f"-maxstars={max_stars}")

    if min_pairs is not None:
        reg_parts.append(f"-minpairs={min_pairs}")
    if layer is not None:
        reg_parts.append(f"-layer={layer}")
    if scale is not None:
        reg_parts.append(f"-scale={scale}")
    if selected_only:
        reg_parts.append("-selected")
    if no_starlist:
        reg_parts.append("-nostarlist")
    if disto is not None:
        reg_parts.append(f"-disto={disto}")

    if not two_pass:
        interp_map = {
            "lanczos4": "la", "bicubic": "cu", "bilinear": "li",
            "linear": "li", "nearest": "ne", "area": "ar", "none": "no",
        }
        interp = interp_map.get(interpolation, "la")
        reg_parts.append(f"-interp={interp}")
        if no_clamp:
            reg_parts.append("-noclamp")
        if drizzle:
            reg_parts.append("-drizzle")
            reg_parts.append(f"-pixfrac={drizzle_pixfrac}")
            if drizzle_kernel:
                reg_parts.append(f"-kernel={drizzle_kernel}")
            if drizzle_flat:
                reg_parts.append(f"-flat={drizzle_flat}")

    commands.append(" ".join(reg_parts))

    # ── 3. seqapplyreg command ────────────────────────────────────────────
    if two_pass:
        interp_map = {
            "lanczos4": "la", "bicubic": "cu", "bilinear": "li",
            "linear": "li", "nearest": "ne", "area": "ar", "none": "no",
        }
        interp = interp_map.get(interpolation, "la")

        framing_val = framing if framing in ("min", "max", "cog", "current") else "min"
        apply_parts = [
            f"seqapplyreg {calibrated_sequence}",
            f"-interp={interp}",
            f"-framing={framing_val}",
        ]

        if prefix is not None:
            apply_parts.append(f"-prefix={prefix}")
        if scale is not None:
            apply_parts.append(f"-scale={scale}")
        if layer is not None:
            apply_parts.append(f"-layer={layer}")
        if no_clamp:
            apply_parts.append("-noclamp")
        if drizzle:
            apply_parts.append("-drizzle")
            apply_parts.append(f"-pixfrac={drizzle_pixfrac}")
            if drizzle_kernel:
                apply_parts.append(f"-kernel={drizzle_kernel}")
            if drizzle_flat:
                apply_parts.append(f"-flat={drizzle_flat}")

        if filters is not None:
            parsed_filters = (
                filters if isinstance(filters, SeqApplyRegFilters)
                else SeqApplyRegFilters(**filters)
            )
            apply_parts.extend(_build_filter_flags(parsed_filters))

        commands.append(" ".join(apply_parts))

    # ── Execute ───────────────────────────────────────────────────────────
    result = run_siril_script(commands, working_dir=working_dir, timeout=900)
    metrics = _parse_register_output(result.stdout)

    output_prefix = prefix if prefix else "r_"
    registered_seq = f"{output_prefix}{calibrated_sequence}"

    return {
        "registered_sequence": registered_seq,
        "calibrated_sequence": calibrated_sequence,
        "registered_sequence_path": str(Path(working_dir) / f"{registered_seq}.seq"),
        "calibrated_sequence_path": str(Path(working_dir) / f"{calibrated_sequence}.seq"),
        "registered_count": metrics.get("registered_count", 0),
        "failed_count": metrics.get("failed_count", 0),
        "reference_image_index": metrics.get("reference_image_index"),
        "total_stars_detected": metrics.get("total_stars_detected"),
    }
