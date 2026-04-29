"""
T40 — analyze_star_population

Per-source star detection on the working image's luminance, returning a
ranked table of measurements plus aggregate distribution stats. Sibling
of analyze_image — analyze_image returns aggregates only, this returns
the per-source table.

Detection: photutils IRAFStarFinder over MAD-noise threshold (mirrors the
detection scheme in t20_analyze._detect_stars_full).

Per-source measurements:
  - x, y, peak, fwhm, roundness from IRAFStarFinder
  - chroma sampled from RGB at/near the source. Two sampling modes:
      "peak":    saturation at the peak pixel (fast; biased toward 0 on
                 cores that clip to white).
      "annulus": mean saturation in an annulus from `fwhm` to `2*fwhm`
                 around the source (avoids the clipped-core bias).

Score: composed per `score_mode` from peak and chroma; only used to rank
the truncation cap and to populate score percentile bins.

Read-only — does not modify paths.current_image or any state field.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from astropy.io import fits as astropy_fits
from astropy.stats import mad_std
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState


# ── Pydantic input schema ──────────────────────────────────────────────────────


class AnalyzeStarPopulationInput(BaseModel):
    threshold_sigma: float = Field(
        default=5.0,
        description=(
            "Detection threshold in MAD-noise units above background. "
            "5.0: standard. 3.5: catch fainter sources. 8.0: brightest only."
        ),
    )
    fwhm_guess: float = Field(
        default=3.0,
        description=(
            "Initial FWHM guess in pixels passed to IRAFStarFinder. "
            "3.0: typical undersampled DSO image. 2.0: well-sampled at small "
            "pixel scale. 5.0: heavily binned or poor seeing."
        ),
    )
    min_separation_fwhm: float = Field(
        default=2.0,
        description=(
            "Minimum allowed source separation in units of fwhm_guess. "
            "Sources closer than this are merged. 2.0: standard."
        ),
    )
    max_sources: int = Field(
        default=5000,
        description=(
            "Hard cap on returned source count. When detection exceeds the "
            "cap, sources are kept by descending score (see score_mode); the "
            "lowest-scoring sources are dropped. Caps runtime on dense fields "
            "where photutils returns 10^4+ sources."
        ),
    )
    chroma_sample_mode: Literal["peak", "annulus"] = Field(
        default="annulus",
        description=(
            "How chroma (HSV saturation) is sampled per source. "
            "'peak': saturation at the peak pixel. Fastest but biased toward "
            "0 when bright cores clip to white. "
            "'annulus': mean saturation in an annulus from fwhm to 2*fwhm "
            "around the source. Robust to clipped cores."
        ),
    )
    score_mode: Literal["brightness_priority", "color_priority", "balanced"] = Field(
        default="brightness_priority",
        description=(
            "Composition rule for the per-source score used for ranking. "
            "'brightness_priority': score = peak_norm * (1 + 0.3 * chroma_norm). "
            "'color_priority':      score = chroma_norm * (1 + 0.3 * peak_norm). "
            "'balanced':            score = 0.5 * (peak_rank + chroma_rank), "
            "where _rank is the source's percentile rank in [0,1]."
        ),
    )
    return_table_rows: int = Field(
        default=50,
        description=(
            "Maximum per-source rows included in the ToolMessage payload, "
            "ranked by score. The aggregate counts and percentile triples "
            "always reflect the full detected population (post-cap)."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_fits(image_path: Path) -> tuple[np.ndarray, bool]:
    """Load FITS, return ((C,H,W) float32, is_color). Mono → (1,H,W)."""
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)
    if data.max() > 1.0:
        data = data / data.max()
    if data.ndim == 3 and data.shape[0] == 3:
        return data, True
    if data.ndim == 3 and data.shape[2] == 3:
        return np.moveaxis(data, -1, 0), True
    return data.squeeze()[np.newaxis, :, :], False


def _luminance(data: np.ndarray) -> np.ndarray:
    if data.shape[0] == 1:
        return data[0]
    return 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]


def _saturation_at(rgb: np.ndarray, y: int, x: int) -> float:
    """HSV saturation at pixel (y, x). rgb is (3, H, W)."""
    r = float(rgb[0, y, x])
    g = float(rgb[1, y, x])
    b = float(rgb[2, y, x])
    mx = max(r, g, b)
    if mx <= 0.0:
        return 0.0
    mn = min(r, g, b)
    return (mx - mn) / mx


def _annulus_saturation(
    rgb: np.ndarray,
    y: int,
    x: int,
    r_inner: float,
    r_outer: float,
) -> float:
    """Mean HSV saturation over the annulus r_inner..r_outer around (y,x)."""
    H, W = rgb.shape[1], rgb.shape[2]
    r_outer_i = int(np.ceil(r_outer))
    y_lo = max(0, y - r_outer_i)
    y_hi = min(H, y + r_outer_i + 1)
    x_lo = max(0, x - r_outer_i)
    x_hi = min(W, x + r_outer_i + 1)
    if y_hi <= y_lo or x_hi <= x_lo:
        return 0.0
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    rr2 = (yy - y) ** 2 + (xx - x) ** 2
    in_annulus = (rr2 >= r_inner ** 2) & (rr2 <= r_outer ** 2)
    if not np.any(in_annulus):
        return 0.0
    patch = rgb[:, y_lo:y_hi, x_lo:x_hi]
    mx = patch.max(axis=0)
    mn = patch.min(axis=0)
    sat = np.where(mx > 0.0, (mx - mn) / np.where(mx > 0.0, mx, 1.0), 0.0)
    return float(np.mean(sat[in_annulus]))


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return the percentile rank of each entry in [0, 1]. Stable on ties."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values))
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return ranks / (len(values) - 1)


def _compose_score(
    peak: np.ndarray,
    chroma: np.ndarray,
    mode: str,
) -> np.ndarray:
    peak_norm = peak / (peak.max() + 1e-9)
    chroma_norm = chroma / (chroma.max() + 1e-9) if chroma.max() > 0 else np.zeros_like(chroma)
    if mode == "brightness_priority":
        return peak_norm * (1.0 + 0.3 * chroma_norm)
    if mode == "color_priority":
        return chroma_norm * (1.0 + 0.3 * peak_norm)
    if mode == "balanced":
        return 0.5 * (_percentile_rank(peak) + _percentile_rank(chroma))
    raise ValueError(f"Unknown score_mode: {mode}")


# ── LangChain tool ─────────────────────────────────────────────────────────────


@tool(args_schema=AnalyzeStarPopulationInput)
def analyze_star_population(
    threshold_sigma: float = 5.0,
    fwhm_guess: float = 3.0,
    min_separation_fwhm: float = 2.0,
    max_sources: int = 5000,
    chroma_sample_mode: str = "annulus",
    score_mode: str = "brightness_priority",
    return_table_rows: int = 50,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Detect stars in the working image and return per-source measurements
    and aggregate distribution stats.

    Per-source: x, y, peak, fwhm, roundness, chroma. Aggregate: count,
    percentile triples (p10/p50/p90) for peak/fwhm/chroma/score, score
    histogram bins.

    Detection runs on luminance with a MAD-based noise floor and a
    threshold_sigma threshold via photutils IRAFStarFinder. Chroma is
    sampled from RGB by chroma_sample_mode. The full detected population
    is truncated to max_sources (kept by descending score) before stats
    are computed; the ToolMessage payload includes the top
    return_table_rows by score.

    Read-only: does not modify paths or metadata.
    """
    image_path = state["paths"]["current_image"]
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    data, is_color = _load_fits(img_path)
    lum = _luminance(data)

    # MAD-based noise floor; matches t20's detection scheme.
    bg_noise = float(mad_std(lum))
    if bg_noise <= 0:
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps({
                    "count": 0,
                    "note": "MAD noise <= 0; image appears constant.",
                }, indent=2),
                tool_call_id=tool_call_id,
            )]
        })

    bg_level = float(np.median(lum))
    threshold = threshold_sigma * bg_noise

    try:
        from photutils.detection import IRAFStarFinder
    except ImportError as e:
        raise ImportError(
            "analyze_star_population requires photutils. "
            "Install with: uv pip install photutils"
        ) from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder = IRAFStarFinder(
            threshold=threshold,
            fwhm=fwhm_guess,
            minsep_fwhm=min_separation_fwhm,
        )
        sources = finder(lum - bg_level)

    if sources is None or len(sources) == 0:
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps({
                    "count": 0,
                    "threshold_sigma": threshold_sigma,
                    "fwhm_guess": fwhm_guess,
                    "bg_noise_mad": bg_noise,
                    "note": "No sources detected at the given threshold.",
                }, indent=2),
                tool_call_id=tool_call_id,
            )]
        })

    xs = np.array(sources["xcentroid"], dtype=np.float32)
    ys = np.array(sources["ycentroid"], dtype=np.float32)
    peaks = np.array(sources["peak"], dtype=np.float32)
    fwhms = np.array(sources["fwhm"], dtype=np.float32)
    roundness = np.abs(np.array(sources["roundness"], dtype=np.float32))

    # Chroma per source.
    if not is_color:
        chroma = np.zeros_like(peaks)
    else:
        chroma_list: list[float] = []
        for i in range(len(peaks)):
            yi = int(round(float(ys[i])))
            xi = int(round(float(xs[i])))
            yi = max(0, min(data.shape[1] - 1, yi))
            xi = max(0, min(data.shape[2] - 1, xi))
            if chroma_sample_mode == "peak":
                chroma_list.append(_saturation_at(data, yi, xi))
            else:  # annulus
                fi = float(fwhms[i])
                chroma_list.append(
                    _annulus_saturation(data, yi, xi, fi, 2.0 * fi)
                )
        chroma = np.array(chroma_list, dtype=np.float32)

    score = _compose_score(peaks, chroma, score_mode)

    # Cap by score.
    if len(score) > max_sources:
        keep_idx = np.argsort(score)[-max_sources:][::-1]
        xs, ys = xs[keep_idx], ys[keep_idx]
        peaks, fwhms = peaks[keep_idx], fwhms[keep_idx]
        roundness, chroma = roundness[keep_idx], chroma[keep_idx]
        score = score[keep_idx]

    # Order by score desc for the truncated table.
    order = np.argsort(score)[::-1]

    def _pct(arr: np.ndarray) -> dict[str, float]:
        return {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    table_n = min(return_table_rows, len(score))
    table = [
        {
            "x": float(xs[order[i]]),
            "y": float(ys[order[i]]),
            "peak": float(peaks[order[i]]),
            "fwhm": float(fwhms[order[i]]),
            "roundness": float(roundness[order[i]]),
            "chroma": float(chroma[order[i]]),
            "score": float(score[order[i]]),
        }
        for i in range(table_n)
    ]

    payload = {
        "count": int(len(score)),
        "max_sources_cap": int(max_sources),
        "score_mode": score_mode,
        "chroma_sample_mode": chroma_sample_mode,
        "bg_noise_mad": bg_noise,
        "bg_level": bg_level,
        "threshold_sigma": threshold_sigma,
        "is_color": bool(is_color),
        "peak":      _pct(peaks),
        "fwhm":      _pct(fwhms),
        "roundness": _pct(roundness),
        "chroma":    _pct(chroma),
        "score":     _pct(score),
        "score_histogram": [
            int(c) for c in np.histogram(score, bins=10, range=(0.0, 1.0))[0]
        ],
        "top_sources_by_score": table,
    }

    return Command(update={
        "messages": [ToolMessage(
            content=json.dumps(payload, indent=2, default=str),
            tool_call_id=tool_call_id,
        )]
    })
