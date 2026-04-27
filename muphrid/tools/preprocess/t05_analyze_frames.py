"""
T05 — analyze_frames

Extract and present per-frame registration data from a Siril .seq file.

This is the agent's equivalent of Siril's "Plot" tab.  After registration
(T04), Siril writes per-frame metrics to the *calibrated* sequence's .seq
file as R-lines:

    R{layer} fwhm weighted_fwhm roundness quality background_lvl nstars H h00..h22

And per-frame pixel statistics as M-lines:

    M{layer}-{idx} total ngoodpix mean median sigma avgDev mad sqrtbwmv location scale min max normValue bgnoise

This tool parses both, computes summary statistics, and returns everything
the agent needs to reason about frame quality and decide what to keep.

The returned data mirrors what a human sees in Siril's registration output /
Plot tab — FWHM, wFWHM, roundness, star count, background, quality — enabling
the agent to make the same data-driven decisions a human would.

IMPORTANT: Pass the *calibrated_sequence* name (the input to registration),
NOT the registered output sequence.  Siril writes R-lines to the input .seq
during `register`, not to the output `r_<seq>.seq` from `seqapplyreg`.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from muphrid.graph.state import AstroState


# ── .seq parser ────────────────────────────────────────────────────────────────

def _parse_seq_file(seq_path: Path) -> dict:
    """
    Parse a Siril .seq file and return structured registration and stats data.

    .seq file format (v4+, verified from Siril source: src/io/seqfile.c):
        S 'name' beg number selnum fixed refimage version ...
        L nb_layers
        I filenum incl [width,height]
        R{layer} fwhm weighted_fwhm roundness quality background_lvl nstars H h00..h22
        M{layer}-{idx} total ngoodpix mean median sigma avgDev mad sqrtbwmv
                        location scale min max normValue bgnoise

    Siril writes seq files in two layouts depending on sequence type:
      - Interleaved: each I-line is immediately followed by its R-line(s).
      - Block (FITSEQ): all I-lines first, then all R-lines in frame order.

    Both are handled by tracking which layers have been seen for the current
    R-frame group; a repeated layer signals a new frame.

    Returns dict with keys:
        n_frames, selected_indices, reference_image, regdata, stats
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

    # frame_order: I-line frame indices in file order, used to map R-lines in
    # block-format seq files where all I-lines precede all R-lines.
    frame_order: list[int] = []
    r_frame_counter: int = 0         # index into frame_order for current R-frame
    r_seen_layers: set[int] = set()  # layers already recorded for this R-frame

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
                frame_idx = int(parts[1])
                frame_order.append(frame_idx)
                included = int(parts[2])
                if included:
                    result["selected_indices"].append(frame_idx)

        elif stripped[0] == "R" and len(stripped) > 1:
            layer_char = stripped[1]
            if not (layer_char.isdigit() or layer_char == "*"):
                continue

            layer = int(layer_char) if layer_char.isdigit() else 0

            # A repeated layer means we've moved to the next frame's R-lines.
            if layer in r_seen_layers:
                r_frame_counter += 1
                r_seen_layers = {layer}
            else:
                r_seen_layers.add(layer)

            if r_frame_counter >= len(frame_order):
                continue

            actual_idx = frame_order[r_frame_counter]
            if actual_idx in result["regdata"]:
                continue  # already captured a layer for this frame

            rest = stripped[2:].strip()
            tokens = rest.split()
            if len(tokens) >= 6:
                result["regdata"][actual_idx] = {
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
                img_idx = int(m_match.group(2))
                # Accept any layer but only record the first M-line per frame
                # (avoids overwriting with a lower-quality channel's stats).
                if img_idx in result["stats"]:
                    continue
                tokens = m_match.group(3).split()
                if len(tokens) >= 14:
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


# ── Metric assembly ─────────────────────────────────────────────────────────────

def _build_frame_metrics(seq_data: dict) -> dict[str, dict]:
    """Build per-frame metric dicts from parsed .seq R-lines and M-lines."""
    metrics: dict[str, dict] = {}

    all_indices = set(seq_data["regdata"].keys()) | set(seq_data["stats"].keys())
    if not all_indices:
        all_indices = set(range(seq_data["n_frames"]))

    for idx in sorted(all_indices):
        entry: dict = {
            "fwhm":               None,
            "weighted_fwhm":      None,
            "roundness":          None,
            "quality":            None,
            "background_lvl":     None,
            "number_of_stars":    None,
            "mean":               None,
            "median":             None,
            "sigma":              None,
            "bgnoise":            None,
        }

        reg = seq_data["regdata"].get(idx, {})
        if reg:
            for k in ("fwhm", "weighted_fwhm", "roundness", "quality",
                       "background_lvl", "number_of_stars"):
                if k in reg:
                    entry[k] = reg[k]

        stat = seq_data["stats"].get(idx, {})
        if stat:
            for k in ("mean", "median", "sigma", "bgnoise"):
                if k in stat:
                    entry[k] = stat[k]

        metrics[str(idx)] = entry

    return metrics


def _compute_summary(frame_metrics: dict[str, dict], seq_data: dict) -> dict:
    """
    Compute summary statistics from per-frame metrics.

    Returns a rich summary dict analogous to what a human reads from Siril's
    Plot tab: medians, standard deviations, outlier detection, diagnostic flags.
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

    n = len(frame_metrics)
    has_registration_data = bool(fwhms)

    med_fwhm = std_fwhm = None
    seeing_stability = tracking_quality = sky_consistency = None
    outlier_frames: list[str] = []
    best_frame = worst_frame = None

    if has_registration_data:
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

    return {
        "has_registration_data": has_registration_data,

        "frame_count":      n,
        "reference_image":  seq_data.get("reference_image", -1),
        "selected_count":   len(seq_data.get("selected_indices", [])),

        "median_fwhm":          round(med_fwhm, 4) if med_fwhm is not None else None,
        "std_fwhm":             round(std_fwhm, 4) if std_fwhm is not None else None,
        "median_weighted_fwhm": round(statistics.median(wfwhms), 4) if wfwhms else None,
        "median_roundness":     round(statistics.median(roundnesses), 4) if roundnesses else None,
        "median_quality":       round(statistics.median(qualities), 6) if qualities else None,
        "median_star_count":    int(statistics.median(star_counts)) if star_counts else None,
        "seeing_stability":     seeing_stability,
        "tracking_quality":     tracking_quality,
        "sky_consistency":      sky_consistency,
        "outlier_frames":       outlier_frames,

        "median_background":    round(statistics.median(backgrounds), 6) if backgrounds else None,
        "std_background":       round(statistics.stdev(backgrounds), 6) if len(backgrounds) > 1 else None,
        "median_noise":         round(statistics.median(noises), 6) if noises else None,
        "median_sigma":         round(statistics.median(sigmas), 6) if sigmas else None,

        "best_frame":  best_frame,
        "worst_frame": worst_frame,
    }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool
def analyze_frames(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AstroState, InjectedState],
) -> Command:
    """
    Extract per-frame registration data from a Siril .seq file.

    Reads the calibrated sequence from state. Parses per-frame metrics that
    Siril computed during registration (FWHM, wFWHM, roundness, quality,
    background, star count) and writes them to state.

    The returned data enables frame selection decisions:
      - median_fwhm + std_fwhm → set FWHM rejection thresholds
      - tracking_quality (median roundness) → detect mount/tracking issues
      - seeing_stability (FWHM CV) → assess atmospheric consistency
      - sky_consistency (background CV) → detect clouds / moon interference
      - outlier_frames → frames > 2σ above median FWHM

    If has_registration_data is False, re-run T04 with tuned findstar
    parameters (lower sigma, increased radius, relax=True).
    """
    working_dir = state["dataset"]["working_dir"]
    calibrated_sequence = state["paths"]["calibrated_sequence"]
    if not calibrated_sequence:
        raise ValueError("calibrated_sequence not found in state. Run calibrate (T03) and register (T04) first.")

    wdir = Path(working_dir)
    seq_path = wdir / f"{calibrated_sequence}.seq"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Sequence file not found: {seq_path}. "
            f"Ensure calibration (T03) and registration (T04) completed."
        )

    # Artifact-based validation (mirrors t03_calibrate / t04_register):
    # the .seq file existing is necessary but not sufficient. A zero-byte
    # or malformed .seq silently yields an all-null summary; surface that
    # explicitly so the agent has a clear signal instead of stale metrics.
    seq_size = seq_path.stat().st_size
    if seq_size == 0:
        raise RuntimeError(
            f"analyze_frames: Sequence file {seq_path} is empty (0 bytes). "
            f"Calibration (T03) or registration (T04) did not write frame "
            f"data. Re-run the upstream preprocessing step and verify the "
            f"input frames are intact."
        )

    seq_data = _parse_seq_file(seq_path)

    # The S-line defines the sequence header (n_frames). If it is missing or
    # malformed, the .seq file is not a real Siril sequence — likely the
    # wrong path or a corrupted write. Disk existence already confirmed.
    if seq_data["n_frames"] == 0:
        head = seq_path.read_text(encoding="utf-8", errors="replace")[:400]
        raise RuntimeError(
            f"analyze_frames: Sequence file {seq_path} has no valid header "
            f"(S-line with n_frames). File may be corrupted or truncated. "
            f"Re-run calibration/registration. First 400 chars:\n{head}"
        )

    frame_metrics = _build_frame_metrics(seq_data)
    summary = _compute_summary(frame_metrics, seq_data)

    # Attach an actionable hint when registration data is absent so the
    # agent has a concrete next step rather than a null-valued blob. This
    # is the expected "re-run T04 with tuned findstar" path, not a hard
    # fail — the tool completes successfully and the agent decides.
    if not summary["has_registration_data"]:
        summary = {
            **summary,
            "hint": (
                "No registration data in the .seq file — Siril's findstar "
                "likely failed to detect stars during T04 registration. "
                "Re-run siril_register with tuned parameters: lower threshold "
                "(e.g. threshold=0.8), increased radius (e.g. radius=10), "
                "and relax=True. Star-sparse fields may also need minstars "
                "lowered (e.g. minstars=20). The .seq header parsed fine "
                f"({summary['frame_count']} frames), so the sequence itself "
                f"is valid — only the registration pass needs rerunning."
            ),
        }

    import json
    return Command(update={
        "metrics": {**state["metrics"], "frame_stats": frame_metrics, "frame_summary": summary},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2), tool_call_id=tool_call_id)],
    })
