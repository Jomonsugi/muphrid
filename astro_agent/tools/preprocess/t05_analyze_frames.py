"""
T05 — analyze_frames

Compute per-frame quality metrics on a registered sequence by parsing the
.seq file directly.  Registration computes per-frame FWHM, roundness, quality,
star count, and background; these are stored in the .seq file's R-lines.
Per-frame statistics (mean, median, sigma, noise) are in the M-lines.

This replaces the old seqstat/seqpsf approach which failed in headless mode.

.seq file format (v4+, verified from Siril source: src/io/seqfile.c):
    S 'name' beg number selnum fixed refimage version ...
    L nb_layers
    I filenum incl [width,height]
    R{layer} fwhm weighted_fwhm roundness quality background_lvl nstars H h00..h22
    M{layer}-{image_idx} total ngoodpix mean median sigma avgDev mad sqrtbwmv location scale min max normValue bgnoise

## Two selection regimes

When star detection succeeds (R-lines populated), the agent uses FWHM,
roundness, and star count as primary rejection criteria in T06.

When star detection fails — typical for sub-second exposures where individual
frames don't expose enough stars — this tool falls back to Laplacian variance
as a sharpness proxy. Laplacian variance measures high-frequency image content
(focus, trailing, blur) directly from pixel data, without requiring stars.
The `has_star_metrics` flag in the summary tells T06 which path to take.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import numpy as np
from astropy.io import fits
from langchain_core.tools import tool
from pydantic import BaseModel, Field


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


# ── .seq parser ────────────────────────────────────────────────────────────────

def _parse_seq_file(seq_path: Path) -> dict:
    """
    Parse a Siril .seq file and return structured registration and stats data.

    Returns dict with keys:
        n_frames, selected_indices, regdata (dict of idx→dict), stats (dict of idx→dict)
    """
    result: dict = {
        "n_frames": 0,
        "selected_indices": [],
        "reference_image": -1,
        "regdata": {},
        "stats": {},
    }

    if not seq_path.exists():
        return result

    lines = seq_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("S "):
            s_match = re.match(
                r"S\s+'?([^']+?)'?\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)",
                stripped,
            )
            if s_match:
                result["n_frames"] = int(s_match.group(3))
                result["reference_image"] = int(s_match.group(6))

        elif stripped.startswith("I "):
            parts = stripped.split()
            if len(parts) >= 3:
                filenum = int(parts[1])
                included = int(parts[2])
                if included:
                    result["selected_indices"].append(filenum)

        elif stripped[0] == "R" and len(stripped) > 1:
            layer_char = stripped[1]
            if layer_char.isdigit() or layer_char == "*":
                layer = 0 if layer_char == "*" else int(layer_char)
                if layer != 0:
                    continue  # only parse layer 0 (green/lum)
                rest = stripped[2:].strip()
                tokens = rest.split()
                if len(tokens) >= 6:
                    frame_idx = len(result["regdata"])
                    result["regdata"][frame_idx] = {
                        "fwhm":            float(tokens[0]),
                        "weighted_fwhm":   float(tokens[1]),
                        "roundness":       float(tokens[2]),
                        "quality":         float(tokens[3]),
                        "background_lvl":  float(tokens[4]),
                        "number_of_stars": int(tokens[5]),
                    }

        elif stripped[0] == "M" and len(stripped) > 1:
            m_match = re.match(r"M([0-9*])-(\d+)\s+(.*)", stripped)
            if m_match:
                layer_char = m_match.group(1)
                if layer_char != "0" and layer_char != "*":
                    continue
                img_idx = int(m_match.group(2))
                tokens = m_match.group(3).split()
                if len(tokens) >= 15:
                    result["stats"][img_idx] = {
                        "total":     int(tokens[0]),
                        "ngoodpix":  int(tokens[1]),
                        "mean":      float(tokens[2]),
                        "median":    float(tokens[3]),
                        "sigma":     float(tokens[4]),
                        "avgDev":    float(tokens[5]),
                        "mad":       float(tokens[6]),
                        "sqrtbwmv":  float(tokens[7]),
                        "location":  float(tokens[8]),
                        "scale":     float(tokens[9]),
                        "min":       float(tokens[10]),
                        "max":       float(tokens[11]),
                        "normValue": float(tokens[12]),
                        "bgnoise":   float(tokens[13]),
                    }

    return result


# ── Laplacian sharpness (star-free sharpness proxy) ────────────────────────────

def _compute_laplacian_sharpness_fitseq(
    fitseq_path: Path,
    downsample: int = 4,
) -> dict[int, float]:
    """
    Compute Laplacian variance as a per-frame sharpness proxy from a FITSEQ.

    Laplacian variance = var(∇²I), where ∇²I is approximated as:
        top + bottom + left + right − 4 × center

    Higher value → sharper frame (more high-frequency content).
    This works without star detection and is effective for blur, trailing,
    focus drift, and atmospheric smearing — the same defects a human spots
    when blinking through subs in PixInsight.

    Downsamples by `downsample` before computing (default 4×) for speed on
    large raw frames (6000×4000px → 1500×1000px, ~16× fewer operations).

    Returns {frame_index: laplacian_variance} for all readable frames.
    """
    result: dict[int, float] = {}
    if not fitseq_path.exists():
        return result

    try:
        with fits.open(str(fitseq_path), memmap=True) as hdul:
            frame_idx = 0
            for hdu in hdul:
                if hdu.data is None or hdu.data.size == 0:
                    continue

                data = hdu.data.astype(np.float32)

                # For color FITS (NAXIS=3), extract the green channel.
                # Siril color FITSEQs store channels as (3, H, W) or (H, W, 3).
                if data.ndim == 3:
                    if data.shape[0] == 3:
                        data = data[1]          # (3, H, W) → green
                    elif data.shape[2] == 3:
                        data = data[:, :, 1]    # (H, W, 3) → green
                    else:
                        data = data[0]          # fallback: first plane

                # Subsample for speed
                if downsample > 1:
                    data = data[::downsample, ::downsample]

                h, w = data.shape
                if h < 3 or w < 3:
                    frame_idx += 1
                    continue

                # Discrete 2D Laplacian via finite differences
                lap = (
                    data[:-2, 1:-1] +   # top neighbor
                    data[2:,  1:-1] +   # bottom neighbor
                    data[1:-1, :-2] +   # left neighbor
                    data[1:-1,  2:] -   # right neighbor
                    4.0 * data[1:-1, 1:-1]
                )
                result[frame_idx] = float(np.var(lap))
                frame_idx += 1

    except Exception:
        pass

    return result


# ── Metric assembly ─────────────────────────────────────────────────────────────

def _build_frame_metrics(
    seq_data: dict,
    laplacian: dict[int, float],
) -> dict[str, dict]:
    """Merge regdata, stats, and Laplacian sharpness into per-frame dicts."""
    metrics: dict[str, dict] = {}

    all_indices = set(seq_data["regdata"].keys()) | set(seq_data["stats"].keys())
    if not all_indices:
        all_indices = set(range(seq_data["n_frames"]))

    for idx in sorted(all_indices):
        entry: dict = {
            # R-line (star-detection based)
            "fwhm":               None,
            "weighted_fwhm":      None,
            "roundness":          None,
            "quality":            None,
            "background_lvl":     None,
            "number_of_stars":    None,
            # M-line (pixel statistics, always present if Siril computed them)
            "mean":               None,
            "median":             None,
            "sigma":              None,
            "bgnoise":            None,
            # Laplacian sharpness (computed from FITSEQ pixel data)
            "laplacian_sharpness": laplacian.get(idx),
        }

        reg = seq_data["regdata"].get(idx, {})
        if reg:
            entry.update({k: reg[k] for k in entry if k in reg})

        stat = seq_data["stats"].get(idx, {})
        if stat:
            for k in ("mean", "median", "sigma", "bgnoise"):
                if k in stat:
                    entry[k] = stat[k]

        metrics[str(idx)] = entry

    return metrics


def _compute_summary(frame_metrics: dict[str, dict], seq_data: dict) -> dict:
    """
    Compute summary statistics and diagnostic flags from per-frame metrics.

    Always returns `has_star_metrics` so T06 can choose the right selection path:
      True  → FWHM/roundness/star_count available → standard sigma-clipping path
      False → No star data → Laplacian-percentile path
    """
    def _collect(key: str, positive_only: bool = True) -> list[float]:
        vals = [v[key] for v in frame_metrics.values() if v.get(key) is not None]
        return [x for x in vals if x > 0] if positive_only else vals

    fwhms       = _collect("fwhm")
    wfwhms      = _collect("weighted_fwhm")
    roundnesses = _collect("roundness")
    qualities   = _collect("quality")
    star_counts = [v["number_of_stars"] for v in frame_metrics.values()
                   if v.get("number_of_stars") is not None and v["number_of_stars"] > 0]
    backgrounds = _collect("background_lvl")
    noises      = _collect("bgnoise")
    sigmas      = _collect("sigma")
    sharpnesses = _collect("laplacian_sharpness")

    n = len(frame_metrics)
    has_star_metrics = bool(fwhms)

    # ── Star-based metrics (None when not available) ──────────────────────
    med_fwhm, std_fwhm = None, None
    seeing_stability, tracking_quality, sky_consistency = None, None, None
    outlier_frames: list[str] = []
    best_frame, worst_frame = None, None

    if has_star_metrics:
        med_fwhm = statistics.median(fwhms)
        std_fwhm = statistics.stdev(fwhms) if len(fwhms) > 1 else 0.0

        sorted_by_fwhm = sorted(
            [(k, v["fwhm"]) for k, v in frame_metrics.items()
             if v.get("fwhm") is not None and v["fwhm"] > 0],
            key=lambda x: x[1],
        )
        best_frame  = sorted_by_fwhm[0][0]  if sorted_by_fwhm else None
        worst_frame = sorted_by_fwhm[-1][0] if sorted_by_fwhm else None

        seeing_stability = round(std_fwhm / med_fwhm, 4) if med_fwhm else None
        tracking_quality = round(statistics.median(roundnesses), 4) if roundnesses else None

        if len(backgrounds) > 1:
            med_bg = statistics.median(backgrounds)
            if med_bg > 0:
                sky_consistency = round(statistics.stdev(backgrounds) / med_bg, 4)

        if std_fwhm and std_fwhm > 0:
            threshold = med_fwhm + 2 * std_fwhm
            outlier_frames = [
                k for k, v in frame_metrics.items()
                if v.get("fwhm") is not None and v["fwhm"] > threshold
            ]

    elif sharpnesses:
        # No FWHM — rank by Laplacian sharpness for best/worst
        sorted_by_sharp = sorted(
            [(k, v["laplacian_sharpness"]) for k, v in frame_metrics.items()
             if v.get("laplacian_sharpness") is not None],
            key=lambda x: x[1],
            reverse=True,   # higher = sharper
        )
        best_frame  = sorted_by_sharp[0][0]  if sorted_by_sharp else None
        worst_frame = sorted_by_sharp[-1][0] if sorted_by_sharp else None

    return {
        # Selection regime flag — T06 reads this to pick the right path
        "has_star_metrics": has_star_metrics,

        "frame_count":      n,
        "reference_image":  seq_data.get("reference_image", -1),
        "selected_count":   len(seq_data.get("selected_indices", [])),

        # Star-based (None when has_star_metrics=False)
        "median_fwhm":         round(med_fwhm, 4) if med_fwhm is not None else None,
        "std_fwhm":            round(std_fwhm, 4) if std_fwhm is not None else None,
        "median_weighted_fwhm": round(statistics.median(wfwhms), 4) if wfwhms else None,
        "median_roundness":    round(statistics.median(roundnesses), 4) if roundnesses else None,
        "median_quality":      round(statistics.median(qualities), 6) if qualities else None,
        "median_star_count":   int(statistics.median(star_counts)) if star_counts else None,
        "seeing_stability":    seeing_stability,
        "tracking_quality":    tracking_quality,
        "sky_consistency":     sky_consistency,
        "outlier_frames":      outlier_frames,

        # Stats-based (from M-lines, available regardless of star detection)
        "median_background":   round(statistics.median(backgrounds), 6) if backgrounds else None,
        "std_background":      round(statistics.stdev(backgrounds), 6) if len(backgrounds) > 1 else None,
        "median_noise":        round(statistics.median(noises), 6) if noises else None,
        "median_sigma":        round(statistics.median(sigmas), 6) if sigmas else None,

        # Laplacian sharpness (from FITSEQ pixel data, always attempted)
        "median_laplacian_sharpness": round(statistics.median(sharpnesses), 4) if sharpnesses else None,
        "std_laplacian_sharpness":    round(statistics.stdev(sharpnesses), 4) if len(sharpnesses) > 1 else None,

        # Best/worst frame key
        "best_frame":  best_frame,
        "worst_frame": worst_frame,
    }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=AnalyzeFramesInput)
def analyze_frames(
    working_dir: str,
    registered_sequence: str,
) -> dict:
    """
    Compute per-frame quality metrics on a registered sequence.

    Primary path: parses FWHM, roundness, star count, background from .seq
    R-lines (written by Siril's star-detection during registration).

    Fallback path: when star detection failed (short exposures, sparse fields),
    computes Laplacian variance sharpness directly from the FITSEQ pixel data.
    This measures blur, trailing, and focus drift without needing stars.

    Always returns `summary.has_star_metrics` (bool) so the agent can tell T06
    which selection strategy to use:
      True  → pass frame_metrics to select_frames with standard criteria
      False → pass frame_metrics to select_frames with has_star_metrics=False
               and a keep_percentile (default 0.85) to rank by sharpness

    Use summary.median_fwhm and summary.std_fwhm to set rejection thresholds
    when has_star_metrics=True.
    High seeing_stability (CV > 0.3) means variable seeing — tighten rejection.
    Low tracking_quality (< 0.7) means elongation — investigate mount issues.
    """
    wdir = Path(working_dir)
    seq_path = wdir / f"{registered_sequence}.seq"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Sequence file not found: {seq_path}. "
            f"Ensure registration (T04) completed successfully."
        )

    seq_data = _parse_seq_file(seq_path)

    # Compute Laplacian sharpness from the FITSEQ pixel data.
    # Try both .fit and .fits extensions.
    fitseq_path = wdir / f"{registered_sequence}.fit"
    if not fitseq_path.exists():
        fitseq_path = wdir / f"{registered_sequence}.fits"
    laplacian = _compute_laplacian_sharpness_fitseq(fitseq_path)

    frame_metrics = _build_frame_metrics(seq_data, laplacian)
    summary = _compute_summary(frame_metrics, seq_data)

    return {
        "frame_metrics": frame_metrics,
        "summary":       summary,
    }
