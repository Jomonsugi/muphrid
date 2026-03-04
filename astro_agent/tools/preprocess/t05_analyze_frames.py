"""
T05 — analyze_frames

Compute per-frame quality metrics on a registered sequence.
Metrics feed T06 (select_frames) and give the agent statistical context
for deciding rejection thresholds.

Backend: Siril CLI — seqstat (sequence statistics) and seqpsf (PSF fitting).
seqstat provides per-frame background level and noise.
seqpsf provides per-frame FWHM, eccentricity, roundness, and star count.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class AnalyzeFramesInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    registered_sequence: str = Field(
        description=(
            "Name of the registered sequence (without .seq extension). "
            "Typically 'r_pp_lights_seq' from T04."
        )
    )


# ── Siril output parsers ───────────────────────────────────────────────────────

def _parse_seqstat(stdout: str) -> dict[str, dict]:
    """
    Parse seqstat output into {filename: {background_level, noise_estimate}}.

    Siril seqstat prints lines like:
        Frame 001 (lights_00001.fit): Mean= 0.0234, Sigma= 0.0012, ...
    """
    frame_stats: dict[str, dict] = {}

    # Pattern: "Image <N> (filename): ... Mean= X ... Sigma= Y ..."
    pattern = re.compile(
        r"Image\s+\d+\s+\(([^)]+)\)[^\n]*Mean[=:\s]+([\d.e+-]+)[^\n]*"
        r"(?:Sigma|Noise)[=:\s]+([\d.e+-]+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(stdout):
        filename = Path(m.group(1)).name
        frame_stats[filename] = {
            "background_level": float(m.group(2)),
            "noise_estimate":   float(m.group(3)),
        }

    # Fallback: simpler background noise lines
    if not frame_stats:
        for line in stdout.splitlines():
            bg_match = re.search(
                r"(?:Image|Frame)[^(]*\(([^)]+)\).*(?:bg|background)[:\s]+([\d.e+-]+)",
                line,
                re.IGNORECASE,
            )
            if bg_match:
                filename = Path(bg_match.group(1)).name
                frame_stats.setdefault(filename, {})["background_level"] = float(bg_match.group(2))

    return frame_stats


def _parse_seqpsf(stdout: str) -> dict[str, dict]:
    """
    Parse seqpsf output into {filename: {fwhm, eccentricity, roundness, star_count}}.

    Siril seqpsf prints per-frame PSF lines like:
        Frame 1 (name.fit) FWHM=2.34 ecc=0.12 roundness=0.96 stars=45
    """
    psf_stats: dict[str, dict] = {}

    pattern = re.compile(
        r"Frame\s+\d+\s+\(([^)]+)\).*?"
        r"FWHM[=:\s]*([\d.]+).*?"
        r"(?:ecc(?:entricity)?)[=:\s]*([\d.]+).*?"
        r"(?:round(?:ness)?)[=:\s]*([\d.]+).*?"
        r"(?:stars?)[=:\s]*(\d+)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(stdout):
        filename = Path(m.group(1)).name
        psf_stats[filename] = {
            "fwhm":         float(m.group(2)),
            "eccentricity": float(m.group(3)),
            "roundness":    float(m.group(4)),
            "star_count":   int(m.group(5)),
        }

    # Fallback: look for FWHM lines individually
    if not psf_stats:
        fwhm_pattern = re.compile(
            r"\(([^)]+)\).*?FWHM[=:\s]*([\d.]+)", re.IGNORECASE
        )
        for m in fwhm_pattern.finditer(stdout):
            filename = Path(m.group(1)).name
            psf_stats.setdefault(filename, {})["fwhm"] = float(m.group(2))

    return psf_stats


def _merge_metrics(
    stat_data: dict[str, dict],
    psf_data: dict[str, dict],
) -> dict[str, dict]:
    """Merge seqstat and seqpsf data by filename into a unified metrics dict."""
    all_files = set(stat_data.keys()) | set(psf_data.keys())
    merged: dict[str, dict] = {}
    for fname in all_files:
        entry: dict = {
            "fwhm":             None,
            "eccentricity":     None,
            "star_count":       None,
            "background_level": None,
            "noise_estimate":   None,
            "roundness":        None,
        }
        entry.update(stat_data.get(fname, {}))
        entry.update(psf_data.get(fname, {}))
        merged[fname] = entry
    return merged


def _compute_summary(frame_metrics: dict[str, dict]) -> dict:
    fwhms      = [v["fwhm"]       for v in frame_metrics.values() if v.get("fwhm")      is not None]
    roundnesses= [v["roundness"]  for v in frame_metrics.values() if v.get("roundness") is not None]

    if not fwhms:
        return {
            "median_fwhm": None, "std_fwhm": None,
            "median_roundness": None, "best_frame": None, "worst_frame": None,
        }

    med_fwhm   = statistics.median(fwhms)
    std_fwhm   = statistics.stdev(fwhms) if len(fwhms) > 1 else 0.0
    med_round  = statistics.median(roundnesses) if roundnesses else None

    # Best = lowest FWHM, worst = highest FWHM
    sorted_by_fwhm = sorted(
        [(k, v["fwhm"]) for k, v in frame_metrics.items() if v.get("fwhm") is not None],
        key=lambda x: x[1],
    )
    best  = sorted_by_fwhm[0][0]  if sorted_by_fwhm else None
    worst = sorted_by_fwhm[-1][0] if sorted_by_fwhm else None

    return {
        "median_fwhm":      round(med_fwhm, 4),
        "std_fwhm":         round(std_fwhm, 4),
        "median_roundness": round(med_round, 4) if med_round is not None else None,
        "best_frame":       best,
        "worst_frame":      worst,
    }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=AnalyzeFramesInput)
def analyze_frames(
    working_dir: str,
    registered_sequence: str,
) -> dict:
    """
    Compute per-frame quality metrics (FWHM, roundness, background, noise,
    star count) on a registered sequence. Returns frame_metrics dict and
    summary statistics for use by select_frames (T06).

    Use summary.median_fwhm and summary.std_fwhm to set rejection thresholds.
    High std_fwhm means variable seeing — tighten rejection.
    """
    # Run seqstat for background/noise stats
    stat_result = run_siril_script(
        [f"seqstat {registered_sequence}"],
        working_dir=working_dir,
        timeout=120,
    )

    # Run seqpsf for PSF quality metrics (FWHM, eccentricity, roundness)
    psf_result = run_siril_script(
        [f"seqpsf {registered_sequence}"],
        working_dir=working_dir,
        timeout=120,
    )

    stat_data = _parse_seqstat(stat_result.stdout)
    psf_data  = _parse_seqpsf(psf_result.stdout)
    frame_metrics = _merge_metrics(stat_data, psf_data)
    summary = _compute_summary(frame_metrics)

    return {
        "frame_metrics": frame_metrics,
        "summary":       summary,
    }
