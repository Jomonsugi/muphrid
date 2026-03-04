"""
T04 — siril_register

Align calibrated light frames to a common reference frame using star matching.
Two-pass registration automatically selects the best reference frame.

Siril commands:
    register <seq> -2pass [-transf=<type>] [-maxstars=N]
    seqapplyreg <seq> [-interp=<method>] [-drizzle] [-framing=max]
                [-filter-fwhm=<pct>%] [-filter-round=<pct>%]
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


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
    transformation: str = Field(
        default="homography",
        description=(
            "Registration transformation type. "
            "'homography' (default — handles rotation, scale, perspective), "
            "'affine' (rotation + scale, no perspective), "
            "'shift' (translation only — for stable tracking mounts)."
        ),
    )
    interpolation: str = Field(
        default="lanczos4",
        description=(
            "Pixel interpolation for frame resampling. "
            "'lanczos4' (best quality, default), 'bicubic', 'bilinear', 'none'."
        ),
    )
    max_stars: int = Field(
        default=500,
        description="Maximum number of stars to use for registration matching.",
    )
    drizzle: bool = Field(
        default=False,
        description=(
            "Enable drizzle integration for sub-pixel resolution improvement. "
            "Only beneficial with ≥30 well-dithered frames."
        ),
    )
    drizzle_pixfrac: float = Field(
        default=1.0,
        description="Drizzle pixel fraction (0.5–1.0). Lower values sharpen but need more frames.",
    )
    filter_fwhm_pct: float | None = Field(
        default=None,
        description=(
            "Keep only frames with FWHM in the best N%. e.g. 90 keeps the 90% "
            "sharpest frames. Replaces separate T05/T06 steps for simple datasets."
        ),
    )
    filter_round_pct: float | None = Field(
        default=None,
        description=(
            "Keep only frames with roundness in the best N%. "
            "Rejects elongated-star frames from tracking errors."
        ),
    )


# ── Output parsing ─────────────────────────────────────────────────────────────

def _parse_register_output(stdout: str) -> dict:
    metrics: dict = {}

    reg_match = re.search(r"(\d+)\s+(?:frame[s]?\s+)?registered", stdout, re.IGNORECASE)
    if reg_match:
        metrics["registered_count"] = int(reg_match.group(1))

    fail_match = re.search(r"(\d+)\s+(?:frame[s]?\s+)?(?:failed|not registered)", stdout, re.IGNORECASE)
    metrics["failed_count"] = int(fail_match.group(1)) if fail_match else 0

    fwhm_match = re.search(r"FWHM[:\s]+([\d.]+)", stdout, re.IGNORECASE)
    if fwhm_match:
        metrics["avg_fwhm"] = float(fwhm_match.group(1))

    round_match = re.search(r"[Rr]oundness[:\s]+([\d.]+)", stdout)
    if round_match:
        metrics["avg_roundness"] = float(round_match.group(1))

    star_match = re.search(r"(\d+)\s+stars?\s+detected", stdout, re.IGNORECASE)
    if star_match:
        metrics["avg_star_count"] = float(star_match.group(1))

    return metrics


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SirilRegisterInput)
def siril_register(
    working_dir: str,
    calibrated_sequence: str,
    transformation: str = "homography",
    interpolation: str = "lanczos4",
    max_stars: int = 500,
    drizzle: bool = False,
    drizzle_pixfrac: float = 1.0,
    filter_fwhm_pct: float | None = None,
    filter_round_pct: float | None = None,
) -> dict:
    """
    Align calibrated light frames using two-pass star matching registration.
    Returns registered sequence name and quality metrics.

    Use homography (default) for alt-az or equatorial data with field rotation.
    Enable drizzle only with ≥30 well-dithered frames.
    """
    transf_map = {"homography": "homography", "affine": "affine", "shift": "shift"}
    transf = transf_map.get(transformation, "homography")

    interp_map = {
        "lanczos4": "lanczos4",
        "bicubic":  "bicubic",
        "bilinear": "bilinear",
        "none":     "none",
    }
    interp = interp_map.get(interpolation, "lanczos4")

    # Pass 1: register (star detection + transformation calculation)
    register_cmd = (
        f"register {calibrated_sequence} -2pass -transf={transf} -maxstars={max_stars}"
    )

    # Pass 2: seqapplyreg (actually resample all frames)
    applyreg_parts = [f"seqapplyreg {calibrated_sequence} -interp={interp} -framing=max"]
    if drizzle:
        applyreg_parts.append(f"-drizzle -pixfrac={drizzle_pixfrac}")
    if filter_fwhm_pct is not None:
        applyreg_parts.append(f"-filter-fwhm={filter_fwhm_pct}%")
    if filter_round_pct is not None:
        applyreg_parts.append(f"-filter-round={filter_round_pct}%")
    applyreg_cmd = " ".join(applyreg_parts)

    result = run_siril_script(
        [register_cmd, applyreg_cmd],
        working_dir=working_dir,
        timeout=900,
    )

    metrics = _parse_register_output(result.stdout)

    # Siril names the registered sequence r_<input_seq>
    registered_seq = f"r_{calibrated_sequence}"

    return {
        "registered_sequence": registered_seq,
        "registered_sequence_path": str(Path(working_dir) / f"{registered_seq}.seq"),
        "reference_image": result.parsed.get("output_path", ""),
        "registered_count": metrics.get("registered_count", 0),
        "failed_count": metrics.get("failed_count", 0),
        "metrics": {
            "avg_fwhm":       metrics.get("avg_fwhm"),
            "avg_roundness":  metrics.get("avg_roundness"),
            "avg_star_count": metrics.get("avg_star_count"),
        },
    }
