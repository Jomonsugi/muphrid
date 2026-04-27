"""
T20 — analyze_image

The agent's primary diagnostic instrument. Returns comprehensive statistics,
per-channel analysis, noise estimates, star metrics, and background
characterization. Called before and after every image-modifying tool to
evaluate results and guide next decisions.

Architecture (PixInsight-aligned):
- Zero/saturated pixel rejection: registration borders and clipped pixels
  excluded before any computation (matches PixInsight Statistics behavior)
- Background: photutils Background2D (2D median tiling with MAD RMS)
- Noise: dual estimator — MAD (general) + wavelet MRS (signal-excluded)
- Per-channel stats: raw percentiles + astropy mad_std
- Star metrics: photutils IRAFStarFinder with MAD-based threshold
- All statistics use MAD as the robust scale estimator — works correctly
  on both linear data (near-zero mode) and stretched data
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Annotated

import numpy as np
from astropy.io import fits as astropy_fits
from astropy.stats import mad_std
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter, sobel

from muphrid.graph.regression import (
    detect_regressions,
    filter_resolved,
    format_warnings,
    merge_warnings,
)

logger = logging.getLogger(__name__)


# ── Pydantic input schema ──────────────────────────────────────────────────────

class AnalyzeImageInput(BaseModel):
    detect_stars: bool = Field(
        default=True,
        description=(
            "Run star detection via photutils IRAFStarFinder. "
            "Returns count, median_fwhm, median_roundness, and (when "
            "star_distribution=True) FWHM/peak/roundness percentiles. "
            "Set False for speed when star metrics are not needed."
        ),
    )
    compute_histogram: bool = Field(
        default=True,
        description=(
            "Compute per-channel histogram summary stats (percentiles p1-p99). "
            "Used to detect clipping and assess stretch completeness."
        ),
    )
    shadow_thresholds: list[float] | None = Field(
        default=None,
        description=(
            "List of shadow clip thresholds (in [0,1]) at which to report the "
            "percentage of pixels at or below that value, per channel. "
            "None uses the defaults [0.0, 0.001, 0.01, 0.05]. "
            "Examples of useful custom lists: "
            "[0.0] for pure hard-black count; "
            "[0.0, 0.002, 0.01, 0.05, 0.1] for a fine staircase when you "
            "suspect black-point placement is the cause of an SNR anomaly; "
            "thresholds derived from `current_background` or `current_noise` "
            "(e.g., bg_level ± k·noise) to characterize how much of the frame "
            "sits inside the noise floor. No warnings are issued — the agent "
            "reads the staircase and reasons about it."
        ),
    )
    highlight_thresholds: list[float] | None = Field(
        default=None,
        description=(
            "List of highlight clip thresholds (in [0,1]) at which to report the "
            "percentage of pixels at or above that value, per channel. "
            "None uses the defaults [0.95, 0.99, 0.999, 1.0]. "
            "Useful custom lists: [0.999, 1.0] for a strict burned-star check; "
            "[0.8, 0.9, 0.95, 0.99, 1.0] to assess how aggressive a stretch was; "
            "per-channel imbalance shows up when one channel saturates before others."
        ),
    )
    valid_pixel_min: float = Field(
        default=0.0,
        description=(
            "Lower bound (exclusive) for the 'valid pixel' mask used in noise, "
            "background, and SNR computations. Pixels with lum <= this value "
            "are excluded. Default 0.0 rejects zero-padded registration borders "
            "and calibration-clipped pixels (PixInsight-style zero rejection). "
            "Raise slightly (e.g. 0.0001) to exclude the very bottom of the "
            "noise floor when analyzing stacks with visible pedestal noise; "
            "lower to -0.001 to include genuine negative residuals from "
            "background subtraction."
        ),
    )
    valid_pixel_max: float = Field(
        default=0.999,
        description=(
            "Upper bound (exclusive) for the valid pixel mask. Pixels with "
            "lum >= this value are excluded as saturated. Default 0.999 "
            "matches PixInsight. Lower to 0.95 when you want to exclude the "
            "knee of a strong stretch from noise/background measurements; "
            "raise to 1.001 to include everything (rarely useful)."
        ),
    )
    quadrant_analysis: bool = Field(
        default=True,
        description=(
            "Compute per-quadrant background median, MAD noise, and valid-"
            "pixel share. Four quadrants: top_left, top_right, bottom_left, "
            "bottom_right. Exposes asymmetric gradient / vignetting / light "
            "pollution — if one quadrant's median is notably higher or "
            "noisier than the others, flat-fielding or background removal "
            "is likely incomplete. Cheap; disable only for minimum-cost runs."
        ),
    )
    compute_mode: bool = Field(
        default=True,
        description=(
            "Estimate the mode (histogram peak) of each channel and the "
            "luminance. On linear data the mode is effectively the black "
            "point — where most pixels are sitting. Divergence between "
            "mode and median indicates distribution shape (heavy-tailed "
            "from stars/nebula versus symmetric noise). After stretching "
            "the mode tracks where the midtone was placed."
        ),
    )
    star_distribution: bool = Field(
        default=True,
        description=(
            "When detect_stars is True, also report FWHM, peak, and roundness "
            "percentiles (p10/p50/p90) across detected stars rather than only "
            "the median. Surfaces PSF field uniformity: wide FWHM spread → "
            "position-dependent aberrations; high roundness spread → tracking "
            "errors or mount flex. Low extra cost once stars are detected."
        ),
    )
    wavelet_scales: int = Field(
        default=3,
        description=(
            "Number of wavelet scales for multi-scale noise estimation. "
            "Scale 1 = finest (pixel-scale shot/read noise, same as the "
            "existing `wavelet_noise` scalar). Each successive scale is "
            "~2x coarser. 3 scales is enough to distinguish pixel-scale "
            "noise from low-frequency pattern noise (walking, banding, "
            "residual gradient). Set 1 to match the legacy single-scale "
            "output; 5 for very detailed structured-noise profiling."
        ),
        ge=1,
        le=6,
    )
    background_box_size: int | None = Field(
        default=None,
        description=(
            "Override the auto-selected Background2D tile size (pixels). "
            "None uses max(32, min(256, min(H,W)//10)). Smaller boxes "
            "capture finer-scale gradients but can over-fit stars/nebula; "
            "larger boxes produce smoother background models. Useful when "
            "the default gives a visibly wrong background level on a "
            "specific image."
        ),
    )


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_fits_float32(image_path: Path) -> tuple[np.ndarray, dict]:
    """Load a FITS file as float32 in [0,1] range."""
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = dict(hdul[0].header)

    if not np.isfinite(data).all():
        finite_vals = data[np.isfinite(data)]
        fill = float(np.median(finite_vals)) if finite_vals.size else 0.0
        data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)

    if data.max() > 1.0:
        bit_depth = header.get("BITPIX", 16)
        if bit_depth > 0:
            data = data / (2 ** bit_depth - 1)
        else:
            data = data / data.max()

    return data, header


def _trim_zero_borders(data: np.ndarray) -> np.ndarray:
    """
    Remove zero-padded registration borders from the numpy array.

    Registration with framing=min fills edges with exact zeros. X-Trans
    debayering + Siril stacking can also produce scattered zeros. This
    function trims the rectangular bounding box of non-zero data.

    PixInsight-aligned: Statistics implicitly rejects zero and one pixels.
    The original FITS file is untouched — this is analysis-only.
    """
    if data.ndim == 3:
        # Any channel nonzero means the pixel has data
        lum_check = np.max(np.abs(data), axis=0)
    else:
        lum_check = np.abs(data)

    row_has_data = np.any(lum_check > 0, axis=1)
    col_has_data = np.any(lum_check > 0, axis=0)

    if not np.any(row_has_data) or not np.any(col_has_data):
        return data  # All zeros — return as-is

    r0 = int(np.argmax(row_has_data))
    r1 = len(row_has_data) - int(np.argmax(row_has_data[::-1]))
    c0 = int(np.argmax(col_has_data))
    c1 = len(col_has_data) - int(np.argmax(col_has_data[::-1]))

    trimmed = data[:, r0:r1, c0:c1] if data.ndim == 3 else data[r0:r1, c0:c1]
    logger.info(
        f"analyze_image: trimmed zero borders "
        f"({data.shape} → {trimmed.shape}, "
        f"{(1 - trimmed.size/data.size)*100:.1f}% removed)"
    )
    return trimmed


# ── Core statistical helpers ─────────────────────────────────────────────────

def _robust_stats(channel: np.ndarray) -> dict:
    """
    Per-channel statistics using MAD-based noise estimation.

    MAD (Median Absolute Deviation) is a robust scale estimator that works
    correctly on any distribution shape — unlike sigma-clipped stats which
    collapse to zero on near-zero linear astronomical data.
    """
    return {
        "min": float(np.min(channel)),
        "max": float(np.max(channel)),
        "mean": float(np.mean(channel)),
        "median": float(np.median(channel)),
        "std": float(mad_std(channel)),
    }


def _background_estimate(lum: np.ndarray, box_size: int | None = None) -> dict:
    """
    Estimate 2D background level and noise using photutils Background2D.

    Background2D tiles the image, computes the median per tile (robust
    to stars/nebula), and interpolates a smooth 2D background model.
    The RMS map uses MAD-based estimation.

    sigma_clip=None prevents the convergence-to-zero bug that affects
    sigma-clipped stats on near-zero linear data.

    Returns bg_level, bg_noise, the chosen box_size, and the 2D bg_map
    for gradient analysis. If box_size is provided it is used verbatim
    (clamped to [16, min(H,W)//2]); otherwise the auto heuristic is used.
    """
    try:
        from photutils.background import (
            Background2D,
            MADStdBackgroundRMS,
            MedianBackground,
        )

        h, w = lum.shape
        if box_size is None:
            box_size = max(32, min(256, min(h, w) // 10))
        else:
            # Clamp to a sane range; Background2D crashes if box > image.
            box_size = max(16, min(box_size, max(16, min(h, w) // 2)))

        bkg = Background2D(
            lum,
            box_size=(box_size, box_size),
            bkg_estimator=MedianBackground(),
            bkgrms_estimator=MADStdBackgroundRMS(),
            sigma_clip=None,
        )
        bg_level = float(bkg.background_median)
        bg_noise = float(bkg.background_rms_median)
        bg_map = bkg.background
        used_box = int(box_size)
    except Exception as e:
        logger.warning(f"Background2D failed ({e}), using MAD fallback")
        bg_level = float(np.median(lum))
        bg_noise = float(mad_std(lum))
        bg_map = None
        used_box = 0

    # Fallback: if Background2D returned zero noise (shouldn't happen
    # with trimmed data, but safety net)
    if bg_noise <= 0:
        bg_noise = float(mad_std(lum))
    if bg_noise <= 0 and np.max(lum) > 0:
        # IQR-based fallback
        bg_noise = float(
            np.percentile(lum, 75) - np.percentile(lum, 25)
        ) * 0.7413

    return {
        "bg_level": bg_level,
        "bg_noise": bg_noise,
        "bg_map": bg_map,
        "box_size": used_box,
    }


def _wavelet_noise(lum: np.ndarray) -> float:
    """
    MRS-style noise estimation using wavelet decomposition.

    PixInsight's gold standard: the starlet (à trous) wavelet isolates
    noise at the finest spatial scale. The MAD of the first wavelet layer
    gives the true noise floor with extended signal (nebulae, galaxies)
    excluded — because those structures occupy coarser scales.

    This provides a signal-excluded noise estimate that complements the
    MAD-based bg_noise from Background2D. When they agree, high confidence.
    When wavelet_noise << bg_noise, extended signal is inflating MAD.
    """
    try:
        import pywt

        # Pad to even dimensions (required for SWT)
        h, w = lum.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            lum_padded = np.pad(lum, ((0, pad_h), (0, pad_w)), mode="reflect")
        else:
            lum_padded = lum

        # Stationary wavelet transform — first level detail coefficients.
        # With trim_approx=True, pywt returns:
        #   [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
        # so coeffs[0] is the approximation and coeffs[-1] is the finest
        # (scale-1) detail tuple we want here.
        coeffs = pywt.swt2(lum_padded, wavelet="bior1.3", level=1, trim_approx=True)
        detail_h, detail_v, detail_d = coeffs[-1]

        # Combine all detail orientations for robust noise estimate
        all_details = np.concatenate([
            detail_h.ravel(), detail_v.ravel(), detail_d.ravel()
        ])

        # MAD of detail coefficients, scaled to Gaussian sigma
        noise = float(np.median(np.abs(all_details)) * 1.4826)
        return noise
    except Exception as e:
        logger.warning(f"Wavelet noise estimation failed ({e})")
        return 0.0


# ── Derived metrics ──────────────────────────────────────────────────────────

def _gradient_magnitude(
    lum: np.ndarray, bg_map: np.ndarray | None = None
) -> float:
    """
    Large-scale gradient magnitude, normalized by data range (P99-P1).

    Two modes:
    1. bg_map provided: the background model IS the gradient. Its range
       relative to the data range gives the gradient magnitude directly.
    2. Fallback: Sobel on heavily smoothed luminance.

    Normalization by data range (not median) ensures meaningful values
    on linear data where the median is near zero.
    """
    data_range = float(np.percentile(lum, 99) - np.percentile(lum, 1))
    if data_range < 1e-9:
        return 0.0

    if bg_map is not None:
        bg_range = float(np.max(bg_map) - np.min(bg_map))
        return min(bg_range / data_range, 1.0)

    smooth = gaussian_filter(lum, sigma=max(lum.shape[0] // 20, 10))
    gx = sobel(smooth, axis=0)
    gy = sobel(smooth, axis=1)
    mean_grad = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    return min(mean_grad / data_range, 1.0)


def _flatness_score(
    lum: np.ndarray,
    bg_map: np.ndarray | None = None,
    block_size: int = 128,
) -> float:
    """
    Background flatness: 1.0 = perfectly flat, lower = gradient/vignetting.

    Block-median spread normalized by data range (not mean), avoiding
    degenerate results on near-zero linear data.
    """
    data_range = float(np.percentile(lum, 99) - np.percentile(lum, 1))
    if data_range < 1e-9:
        return 1.0

    source = bg_map if bg_map is not None else lum
    h, w = source.shape
    block_vals = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block_vals.append(
                np.median(source[i : i + block_size, j : j + block_size])
            )

    if not block_vals:
        return 1.0

    bv = np.array(block_vals)
    spread = float(np.std(bv)) / data_range
    return float(max(0.0, 1.0 - spread / 0.05))


def _clipping(channel: np.ndarray) -> tuple[float, float]:
    """Return (shadows_pct, highlights_pct) clipping percentages at 0.001/0.999.

    Preserved for the legacy flat `clipped_shadows_pct` / `clipped_highlights_pct`
    fields. For richer reporting use `_clipping_at_thresholds` below.
    """
    total = channel.size
    shadows_pct = float(np.sum(channel <= 0.001) / total * 100)
    highlights_pct = float(np.sum(channel >= 0.999) / total * 100)
    return shadows_pct, highlights_pct


def _clipping_at_thresholds(
    channel: np.ndarray,
    shadow_thresholds: list[float],
    highlight_thresholds: list[float],
) -> dict:
    """
    Percentage of pixels at or below each shadow threshold and at or above
    each highlight threshold.

    Returns a dict like:
      {
        "shadow_le_0.0":   x,    # pct at or below 0.0 (hard black)
        "shadow_le_0.001": x,
        "shadow_le_0.01":  x,
        "highlight_ge_0.95":  x,
        "highlight_ge_0.99":  x,
        "highlight_ge_1.0":   x, # pct at or above 1.0 (hard white)
      }

    No judgment — the agent interprets these values in context. A short Fuji
    frame legitimately has 50%+ shadow clip at the noise floor; a stretched
    M31 stack should have <1% — both use the same data, the agent decides.
    """
    total = channel.size
    result: dict = {}
    for t in shadow_thresholds:
        key = f"shadow_le_{t:g}"
        result[key] = float(np.sum(channel <= t) / total * 100)
    for t in highlight_thresholds:
        key = f"highlight_ge_{t:g}"
        result[key] = float(np.sum(channel >= t) / total * 100)
    return result


def _mode_estimate(channel: np.ndarray, bins: int = 4096) -> float:
    """
    Estimate the mode of the channel distribution — the effective black-point
    location for linear data, or the peak of the midtone for stretched data.

    Uses a histogram with 4096 bins over the observed min..max range. The
    returned value is the midpoint of the bin with the highest count. Robust
    to outliers; no smoothing (caller can reason about bin noise via percentiles).
    """
    c_min = float(np.min(channel))
    c_max = float(np.max(channel))
    if c_max <= c_min:
        return c_min
    counts, edges = np.histogram(channel, bins=bins, range=(c_min, c_max))
    peak_idx = int(np.argmax(counts))
    return float((edges[peak_idx] + edges[peak_idx + 1]) / 2)


def _quadrant_background(
    lum: np.ndarray, valid_mask: np.ndarray | None = None
) -> dict:
    """
    Per-quadrant median luminance, MAD noise, and valid-pixel share.

    Splits the frame into 2×2 quadrants — top_left, top_right, bottom_left,
    bottom_right — and reports each quadrant's background median (P25 of
    valid pixels within the quadrant), MAD noise, and valid-pixel fraction.
    Asymmetric values indicate gradient, vignetting, or asymmetric light
    pollution. Symmetric values indicate a flat field.

    All stats are computed over valid pixels only (or all pixels if no mask).
    """
    h, w = lum.shape
    mid_h = h // 2
    mid_w = w // 2
    regions = {
        "top_left":     (slice(0, mid_h),     slice(0, mid_w)),
        "top_right":    (slice(0, mid_h),     slice(mid_w, w)),
        "bottom_left":  (slice(mid_h, h),     slice(0, mid_w)),
        "bottom_right": (slice(mid_h, h),     slice(mid_w, w)),
    }
    out: dict = {}
    for name, (ry, rx) in regions.items():
        sub = lum[ry, rx]
        if valid_mask is not None:
            sub_mask = valid_mask[ry, rx]
            sub_valid = sub[sub_mask]
            valid_share = float(np.sum(sub_mask) / sub.size) if sub.size else 0.0
        else:
            sub_valid = sub
            valid_share = 1.0

        if sub_valid.size >= 16:
            out[name] = {
                "bg_p25":    float(np.percentile(sub_valid, 25)),
                "median":    float(np.median(sub_valid)),
                "mad_std":   float(mad_std(sub_valid)) if sub_valid.size >= 2 else 0.0,
                "valid_pct": round(valid_share * 100, 3),
            }
        else:
            out[name] = {
                "bg_p25": None, "median": None, "mad_std": None,
                "valid_pct": round(valid_share * 100, 3),
            }
    return out


def _wavelet_noise_multiscale(lum: np.ndarray, n_scales: int = 3) -> list[float]:
    """
    Wavelet noise estimate at multiple scales (scales 1..n).

    Scale 1 isolates fine-grained pixel noise (same as `_wavelet_noise`).
    Successive scales capture progressively coarser variation — scale 2 is
    ~2x pixel separation, scale 3 is ~4x, etc.

    Interpretation: if noise[1] >> noise[2], the image is dominated by
    shot noise / read noise (typical of well-calibrated single frames).
    If noise[2] and noise[3] are comparable to noise[1], there is
    structured noise at those spatial scales — walking, banding, pattern
    noise, or residual gradients.
    """
    try:
        import pywt

        out: list[float] = []
        h, w = lum.shape
        pad_h = (2 ** n_scales) - (h % (2 ** n_scales))
        pad_w = (2 ** n_scales) - (w % (2 ** n_scales))
        pad_h = 0 if pad_h == (2 ** n_scales) else pad_h
        pad_w = 0 if pad_w == (2 ** n_scales) else pad_w
        lum_p = (np.pad(lum, ((0, pad_h), (0, pad_w)), mode="reflect")
                 if (pad_h or pad_w) else lum)

        coeffs = pywt.swt2(
            lum_p, wavelet="bior1.3", level=n_scales, trim_approx=True
        )
        # With trim_approx=True, pywt returns:
        #   [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
        # i.e. coeffs[0] is the coarsest approximation (not a tuple), and
        # coeffs[1..n_scales] are detail tuples ordered coarsest → finest.
        # Iterate in reverse so out[0] = scale 1 (finest, pixel-scale noise).
        for detail in reversed(coeffs[1:]):
            cH, cV, cD = detail
            all_details = np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
            out.append(float(np.median(np.abs(all_details)) * 1.4826))
        return out
    except Exception as e:
        logger.warning(f"Multi-scale wavelet noise failed ({e})")
        return []


def _channel_snr_estimates(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, valid_mask: np.ndarray | None
) -> dict:
    """
    Per-channel SNR: P95 of valid pixels / MAD of valid pixels.

    Gives R/G/B an independent SNR number so the agent can see when one
    channel is much noisier than another (common with light pollution
    filters or bandpass imaging). Complements global `snr_estimate`.
    """
    def one(ch: np.ndarray) -> float:
        vals = ch[valid_mask] if valid_mask is not None else ch
        if vals.size < 2:
            return 0.0
        noise = float(mad_std(vals))
        if noise <= 0:
            return 0.0
        return float(np.percentile(vals, 95) / noise)
    return {"red": one(r), "green": one(g), "blue": one(b)}


def _histogram_percentiles(channel: np.ndarray) -> dict:
    """Percentile summary of the channel distribution."""
    return {
        "p1": float(np.percentile(channel, 1)),
        "p5": float(np.percentile(channel, 5)),
        "p25": float(np.percentile(channel, 25)),
        "p50": float(np.percentile(channel, 50)),
        "p75": float(np.percentile(channel, 75)),
        "p95": float(np.percentile(channel, 95)),
        "p99": float(np.percentile(channel, 99)),
    }


def _background_channel_medians(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    lum: np.ndarray,
    bg_noise: float,
) -> dict:
    """
    Per-channel median of sky background pixels only.

    Background mask: lum < P25 + 3 * bg_noise. This isolates the darkest
    quarter of the frame plus a noise margin — predominantly sky background
    for astronomical data.
    """
    p25 = float(np.percentile(lum, 25))
    threshold = p25 + 3.0 * bg_noise
    bg_mask = lum < threshold
    n_bg = int(np.sum(bg_mask))

    result: dict = {}
    for name, ch in (("red", r), ("green", g), ("blue", b)):
        if n_bg >= 100:
            result[name] = float(np.median(ch[bg_mask]))
        else:
            result[name] = float(np.median(ch))
    result["n_background_pixels"] = n_bg
    return result


def _color_saturation(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> dict:
    """HSV saturation statistics."""
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    with np.errstate(invalid="ignore", divide="ignore"):
        sat = np.where(cmax > 1e-6, delta / cmax, 0.0).astype(np.float32)
    return {
        "mean_saturation": float(np.mean(sat)),
        "median_saturation": float(np.median(sat)),
        "high_saturation_pct": float(np.sum(sat > 0.30) / sat.size * 100),
    }


def _signal_coverage(
    lum: np.ndarray, bg_level: float, bg_noise: float
) -> float:
    """
    Fraction of frame containing significant signal above sky background (%).

    Threshold: bg_level + 5 * bg_noise (5σ with MAD noise).
    """
    threshold = bg_level + 5.0 * bg_noise
    if threshold <= 0 or bg_noise <= 0:
        threshold = float(np.percentile(lum, 90))
    return float(np.sum(lum > threshold) / lum.size * 100)


def _linearity_analysis(lum: np.ndarray, overall_median: float) -> dict:
    """Dual consensus linearity detection: median threshold + skewness."""
    sample = lum.flatten()[::16]
    centered = sample - np.mean(sample)
    m2 = float(np.mean(centered ** 2))
    if m2 < 1e-12:
        skewness = 0.0
    else:
        m3 = float(np.mean(centered ** 3))
        skewness = m3 / (m2 ** 1.5)
        if not np.isfinite(skewness):
            skewness = 0.0

    median_says_linear = overall_median < 0.15
    skew_says_linear = skewness > 2.0
    if median_says_linear and skew_says_linear:
        is_linear, confidence = True, "high"
    elif median_says_linear or skew_says_linear:
        is_linear = median_says_linear
        confidence = "medium"
    else:
        is_linear, confidence = False, "high"

    return {
        "is_linear": is_linear,
        "confidence": confidence,
        "histogram_skewness": round(skewness, 3),
        "median_brightness": round(overall_median, 5),
    }


def _empty_star_result() -> dict:
    return {
        "count": 0,
        "median_fwhm": None,
        "median_roundness": None,
        "fwhm_std": None,
        "median_star_peak_ratio": None,
    }


def _detect_stars_full(
    lum: np.ndarray,
    bg_level: float,
    bg_noise: float,
    include_distribution: bool = True,
) -> dict:
    """
    Star detection using MAD-based threshold.

    Uses bg_level for background subtraction and bg_noise for the detection
    threshold, avoiding sigma-clipped stats that collapse on linear data.

    When include_distribution is True (default), also returns FWHM / peak /
    roundness percentile triples (p10, p50, p90) over all detected stars,
    and their std. This exposes PSF uniformity across the field — tight
    percentile spread = uniform optics, wide spread = position-dependent
    aberrations, coma, or tracking drift.
    """
    try:
        from photutils.detection import IRAFStarFinder

        if bg_noise <= 0:
            bg_noise = float(mad_std(lum))
        if bg_noise <= 0:
            return _empty_star_result()

        threshold = 5.0 * bg_noise

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = IRAFStarFinder(
                threshold=threshold, fwhm=3.0, minsep_fwhm=2.0
            )
            sources = finder(lum - bg_level)

        if sources is None or len(sources) == 0:
            return _empty_star_result()

        fwhm_arr = np.array(sources["fwhm"])
        peak_arr = np.array(sources["peak"])
        round_arr = np.abs(np.array(sources["roundness"]))

        result = {
            "count": len(sources),
            "median_fwhm": float(np.median(fwhm_arr)),
            "median_roundness": float(np.median(round_arr)),
            "fwhm_std": float(np.std(fwhm_arr)),
            "median_star_peak_ratio": float(np.median(peak_arr))
            / (bg_noise + 1e-9),
        }
        if include_distribution:
            result["star_distribution"] = {
                "fwhm_p10": float(np.percentile(fwhm_arr, 10)),
                "fwhm_p50": float(np.percentile(fwhm_arr, 50)),
                "fwhm_p90": float(np.percentile(fwhm_arr, 90)),
                "peak_p10": float(np.percentile(peak_arr, 10)),
                "peak_p50": float(np.percentile(peak_arr, 50)),
                "peak_p90": float(np.percentile(peak_arr, 90)),
                "roundness_p10": float(np.percentile(round_arr, 10)),
                "roundness_p50": float(np.percentile(round_arr, 50)),
                "roundness_p90": float(np.percentile(round_arr, 90)),
                "roundness_std": float(np.std(round_arr)),
                "peak_over_noise_p10": float(np.percentile(peak_arr, 10))
                / (bg_noise + 1e-9),
                "peak_over_noise_p90": float(np.percentile(peak_arr, 90))
                / (bg_noise + 1e-9),
            }
        return result
    except Exception as e:
        return {
            **_empty_star_result(),
            "detection_error": (
                f"Star detection failed ({type(e).__name__}: {e}). "
                "count=0 may not reflect actual star count."
            ),
        }


# ── LangChain tool ─────────────────────────────────────────────────────────────

_DEFAULT_SHADOW_THRESHOLDS = [0.0, 0.001, 0.01, 0.05]
_DEFAULT_HIGHLIGHT_THRESHOLDS = [0.95, 0.99, 0.999, 1.0]


@tool(args_schema=AnalyzeImageInput)
def analyze_image(
    detect_stars: bool = True,
    compute_histogram: bool = True,
    shadow_thresholds: list[float] | None = None,
    highlight_thresholds: list[float] | None = None,
    valid_pixel_min: float = 0.0,
    valid_pixel_max: float = 0.999,
    quadrant_analysis: bool = True,
    compute_mode: bool = True,
    star_distribution: bool = True,
    wavelet_scales: int = 3,
    background_box_size: int | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Comprehensive image analysis — the agent's primary diagnostic instrument.

    Returns per-channel statistics, dual noise estimation, 2D background
    modeling, star metrics, and derived quality indicators. Zero-padded
    registration borders and saturated pixels are automatically excluded
    from all statistics (PixInsight-aligned zero/one rejection).

    ## Metric Definitions

    **Background & Noise**
    - current_background: median sky brightness of valid pixels. Near zero
      in linear space (correct after calibration); rises after stretch.
    - current_noise: MAD-based noise floor from Background2D spatial model.
      Measures pixel-to-pixel variation in the sky background.
    - wavelet_noise: noise estimated from the first wavelet scale (MRS method).
      Isolates fine-grained noise from extended signal (nebulae, galaxies).
      When wavelet_noise << current_noise, extended signal is inflating MAD.
    - snr_estimate: 95th percentile of valid pixel brightness / noise.
      Measures how far the bright signal rises above the noise floor.
    - dynamic_range_db: 20*log10(max_signal / noise). Total usable range in dB.

    **Spatial Quality**
    - gradient_magnitude: large-scale brightness variation normalized by data
      range (P99-P1). 0.0 = perfectly uniform; higher = gradient present.
    - background_flatness: uniformity of the background model. 1.0 = flat;
      lower values indicate gradients or vignetting.
    - signal_coverage_pct: percentage of the frame containing signal above
      5σ of the background noise. Indicates how much of the frame the target
      fills — compact objects have low coverage, extended nebulae have high.

    **Color**
    - per_channel_bg: per-channel (R/G/B) median of sky background pixels.
      After successful color calibration, all three should converge.
    - green_excess: green channel mean minus average of red and blue means.
      Positive values indicate green bias (common in Bayer/X-Trans sensors).
    - channel_imbalance: spread between the brightest and dimmest channel means.
    - mean_saturation / median_saturation: HSV color saturation across the image.
    - high_saturation_pct: percentage of pixels with saturation > 0.30.

    **Stars**
    - star_count: number of point sources detected above 5σ threshold.
    - current_fwhm: median Full Width at Half Maximum of detected stars (pixels).
      Measures the seeing / optical quality.
    - fwhm_std: standard deviation of FWHM across detected stars. Low = uniform
      PSF across the field; high = position-dependent aberrations.
    - median_star_peak_ratio: median star peak brightness / background noise.
      Measures how dominant stars are relative to the background.

    **Linearity**
    - is_linear_estimate: True if the data appears to be in linear (unstretched)
      space. Based on dual consensus: median brightness < 0.15 AND histogram
      skewness > 2.0.
    - linearity_confidence: "high" when both indicators agree, "medium" when
      they disagree, "low" when both indicate non-linear.
    - histogram_skewness: third moment of the luminance distribution. High
      positive skewness is characteristic of linear data (most pixels near
      zero with a long tail from stars/nebula).

    **Clipping (flat, legacy)**
    - clipped_shadows_pct: worst-channel percentage at or below 0.001.
      High values in linear data indicate most pixels are at the noise floor —
      expected for short exposures or uncropped frames. For fine-grained
      reasoning, prefer `clipping_per_channel` below.
    - clipped_highlights_pct: worst-channel percentage at or above 0.999.
      Indicates star cores or bright regions are saturated.

    **Clipping (structured)**
    - clipping_per_channel: nested dict with four keys (red, green, blue,
      luminance), each a staircase of percentages at the agent-chosen
      shadow_thresholds and highlight_thresholds. No thresholds are
      editorialized — a short Fuji frame legitimately has >50% at the
      noise floor, a stretched M31 stack should have <1%; both use the
      same fields and the agent decides what the numbers mean in context.
      Also includes the exact threshold list used.

    **Pixel Coverage**
    - pixel_coverage: total pixel count, number valid, percentage valid,
      number excluded by valid_pixel_min (zero/below-floor) and by
      valid_pixel_max (saturated), and the exact bounds used. Use this
      to reason about whether a noise/background number is trustworthy
      — if valid_pct is low the frame is mostly borders or clipped and
      spatial stats will be degraded.

    **Mode (per-channel histogram peak)**
    - mode_estimate: peak of the 4096-bin histogram per channel, plus
      luminance. On linear data this is effectively the black-point
      location; on stretched data it tracks midtone placement. When
      mode ≪ median, the distribution is heavy-tailed from bright
      structure (stars/nebula). None if compute_mode=False.

    **Background Quadrants**
    - background_quadrants: per-quadrant P25, median, MAD noise, and
      valid-pixel share for {top_left, top_right, bottom_left,
      bottom_right}. Diverging quadrant values indicate gradient,
      vignetting, or asymmetric light pollution. None if
      quadrant_analysis=False.

    **Multi-scale Noise**
    - wavelet_noise_scales: list of noise estimates at successive
      wavelet scales (scale 1 = pixel-scale, scale N ≈ 2^(N-1) pixels).
      noise[1] >> noise[2] → dominated by shot/read noise (good). If
      higher scales are comparable to scale 1, there's structured
      noise (walking, banding, residual gradient).

    **Per-channel SNR**
    - channel_snr: {red, green, blue} P95 / MAD std, computed over
      valid pixels per channel. Surfaces asymmetric noise — useful
      with narrowband, bandpass filters, or uncalibrated color data.

    **Star Distribution (when detect_stars=True and stars found)**
    - star_distribution: FWHM / peak / roundness percentile triples
      (p10, p50, p90) plus std, and peak-over-noise percentiles.
      Wide FWHM spread = position-dependent aberrations; wide
      roundness spread = tracking/mount issues. Missing when
      detect_stars=False or star_distribution=False.

    **Contrast**
    - contrast_ratio: P95 - P5 of valid pixel luminance. Measures the usable
      dynamic range between shadows and highlights in the current image state.

    **Per-Channel Stats** (channel_stats.red/green/blue)
    - min, max: absolute extreme values
    - mean, median: central tendency of valid pixels
    - std: MAD-based robust standard deviation

    **Histogram** (when compute_histogram=True)
    - histogram: per-channel percentile summary (p1, p5, p25, p50,
      p75, p95, p99) over the full channel (not just valid pixels).

    **Reproducibility**
    - bg_box_size: tile size Background2D actually used, echoing the
      auto-selection or the caller's background_box_size override.
    - analyze_params: the exact parameters this call ran with, so later
      reasoning can tell whether two snapshots are comparable.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"].get("current_image")

    if not image_path:
        import json
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps({
                    "error": "No image available to analyze. current_image is not set in state. "
                    "This is expected before stacking — the pipeline is working with a "
                    "multi-frame sequence, not a single image. analyze_image will produce "
                    "results after siril_stack creates the integrated master light."
                }),
                tool_call_id=tool_call_id,
            )],
        })

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Load, trim borders, extract channels ──
    data, header = _load_fits_float32(img_path)
    data = _trim_zero_borders(data)

    if data.ndim == 3 and data.shape[0] == 3:
        r, g, b = data[0], data[1], data[2]
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        is_color = True
    elif data.ndim == 3 and data.shape[2] == 3:
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        is_color = True
    else:
        lum = data.squeeze().astype(np.float32)
        r = g = b = lum
        is_color = False

    # ── PixInsight-aligned: reject zero and saturated pixels ──
    # Zero pixels come from registration borders, calibration clipping,
    # and X-Trans debayering artifacts. Saturated pixels (≥0.999) are
    # clipped and uninformative. Both are excluded from noise / background
    # computations by default, matching PixInsight's implicit zero/one
    # rejection. The bounds are agent-configurable via valid_pixel_min /
    # valid_pixel_max so short-exposure / heavy-stretch data can be
    # analyzed with a different notion of "valid".
    valid_mask = (lum > valid_pixel_min) & (lum < valid_pixel_max)
    n_valid = int(np.sum(valid_mask))
    valid_pct = float(n_valid / lum.size * 100)
    n_zero = int(np.sum(lum <= valid_pixel_min))
    n_saturated = int(np.sum(lum >= valid_pixel_max))
    logger.info(
        f"analyze_image: {valid_pct:.2f}% valid pixels "
        f"({n_valid:,} / {lum.size:,}, min={valid_pixel_min}, max={valid_pixel_max})"
    )

    # If very few valid pixels, the image may be empty or uncropped.
    # Still compute what we can — the agent needs to see the state.
    has_valid = n_valid >= 100

    # ── Background estimation on valid pixels ──
    if has_valid:
        if valid_pct > 50:
            # Majority of pixels are valid — Background2D works well
            bg_est = _background_estimate(lum, box_size=background_box_size)
            bg_level = bg_est["bg_level"]
            bg_noise = bg_est["bg_noise"]
            bg_map = bg_est["bg_map"]
        else:
            # Too many zeros for Background2D (can hang or produce garbage).
            # Use valid-pixel statistics directly.
            bg_level = 0.0
            bg_noise = 0.0
            bg_map = None
            bg_est = {"bg_level": 0.0, "bg_noise": 0.0, "bg_map": None, "box_size": 0}

        # If Background2D returned zeros or was skipped, use valid pixels
        if bg_noise <= 0 or bg_level <= 0:
            valid_lum = lum[valid_mask]
            bg_level = float(np.median(valid_lum))
            bg_noise = float(mad_std(valid_lum))
    else:
        bg_level = 0.0
        bg_noise = 0.0
        bg_map = None
        bg_est = {"bg_level": 0.0, "bg_noise": 0.0, "bg_map": None, "box_size": 0}

    # ── Wavelet noise (MRS-style, signal-excluded) ──
    w_noise = _wavelet_noise(lum) if has_valid else 0.0
    # If wavelet also returns zero (dominated by zero pixels), use valid MAD
    if w_noise <= 0 and has_valid:
        w_noise = float(mad_std(lum[valid_mask]))

    # ── Per-channel statistics (MAD-based, on valid pixels) ──
    channels = {
        "red": _robust_stats(r[valid_mask]) if has_valid else _robust_stats(r),
        "green": _robust_stats(g[valid_mask]) if has_valid else _robust_stats(g),
        "blue": _robust_stats(b[valid_mask]) if has_valid else _robust_stats(b),
    }

    # ── Derived metrics ──
    if has_valid:
        valid_lum = lum[valid_mask]
        overall_median = float(np.median(valid_lum))
        overall_max = float(np.max(valid_lum))
    else:
        overall_median = float(np.median(lum))
        overall_max = float(np.max(lum))

    # For gradient, flatness, and per-channel background: if most pixels
    # are zero (subsecond exposures), these functions produce degenerate
    # results on the full array. Use valid-pixel data when available.
    if has_valid and valid_pct < 50:
        # Too few valid pixels for spatial analysis (gradient, flatness).
        # Report what we can from valid pixels only.
        gradient_mag = _gradient_magnitude(lum[valid_mask].reshape(-1, 1))
        flatness = 0.0  # Can't assess spatial flatness without a full 2D field
        # Per-channel bg: use valid pixels directly
        valid_r = r[valid_mask]
        valid_g = g[valid_mask]
        valid_b = b[valid_mask]
        p25 = float(np.percentile(valid_lum, 25))
        bg_threshold = p25 + 3.0 * bg_noise
        bg_mask_v = valid_lum < bg_threshold
        n_bg = int(np.sum(bg_mask_v))
        per_channel_bg = {}
        for name, ch_v in (("red", valid_r), ("green", valid_g), ("blue", valid_b)):
            per_channel_bg[name] = float(np.median(ch_v[bg_mask_v])) if n_bg >= 10 else float(np.median(ch_v))
        per_channel_bg["n_background_pixels"] = n_bg
    else:
        gradient_mag = _gradient_magnitude(lum, bg_map=bg_map)
        flatness = _flatness_score(lum, bg_map=bg_map)
        per_channel_bg = _background_channel_medians(
            r, g, b, lum, bg_noise=bg_noise
        )

    # SNR: P95 of valid pixels / noise
    if has_valid:
        signal_level = float(np.percentile(lum[valid_mask], 95))
    else:
        signal_level = float(np.percentile(lum, 95))
    snr_estimate = float(signal_level / (bg_noise + 1e-9)) if bg_noise > 0 else 0.0

    # Dynamic range in dB
    dynamic_range_db = (
        float(20 * np.log10((overall_max + 1e-9) / (bg_noise + 1e-9)))
        if bg_noise > 0
        else 0.0
    )

    # Clipping — worst channel
    r_shadows, r_highlights = _clipping(r)
    g_shadows, g_highlights = _clipping(g)
    b_shadows, b_highlights = _clipping(b)
    shadows_pct = max(r_shadows, g_shadows, b_shadows)
    highlights_pct = max(r_highlights, g_highlights, b_highlights)

    # Color balance
    green_excess = float(
        channels["green"]["mean"]
        - (channels["red"]["mean"] + channels["blue"]["mean"]) / 2
    )
    channel_imbalance = float(
        max(channels["red"]["mean"], channels["green"]["mean"], channels["blue"]["mean"])
        - min(channels["red"]["mean"], channels["green"]["mean"], channels["blue"]["mean"])
    )

    # HSV saturation
    color_saturation = (
        _color_saturation(r, g, b)
        if is_color
        else {"mean_saturation": None, "median_saturation": None, "high_saturation_pct": None}
    )

    # Signal coverage
    signal_cov = _signal_coverage(lum, bg_level=bg_level, bg_noise=bg_noise)

    # Linearity
    linearity = _linearity_analysis(lum, overall_median)

    # Star detection
    star_metrics: dict = _empty_star_result()
    if detect_stars:
        star_metrics = _detect_stars_full(
            lum,
            bg_level=bg_level,
            bg_noise=bg_noise,
            include_distribution=star_distribution,
        )

    # Histogram percentiles
    histogram: dict = {}
    if compute_histogram:
        histogram = {
            "red": _histogram_percentiles(r),
            "green": _histogram_percentiles(g),
            "blue": _histogram_percentiles(b),
        }

    # Contrast ratio (on valid pixels to avoid zero-dominated percentiles)
    if has_valid:
        vl = lum[valid_mask]
        contrast_ratio = float(np.percentile(vl, 95) - np.percentile(vl, 5))
    else:
        contrast_ratio = float(np.percentile(lum, 95) - np.percentile(lum, 5))

    # ── New structured fields ───────────────────────────────────────────────
    # These augment the flat scalars above with richer per-channel and
    # spatial breakdowns. The agent reads whatever is useful for the
    # decision at hand; nothing is filtered, no thresholds are applied.

    # Pixel coverage — how much of the frame participated in the analysis.
    pixel_coverage = {
        "total":         int(lum.size),
        "n_valid":       n_valid,
        "valid_pct":     round(valid_pct, 3),
        "n_at_or_below_min":   n_zero,
        "pct_at_or_below_min": round(n_zero / lum.size * 100, 3),
        "n_at_or_above_max":   n_saturated,
        "pct_at_or_above_max": round(n_saturated / lum.size * 100, 3),
        "valid_min": float(valid_pixel_min),
        "valid_max": float(valid_pixel_max),
    }

    # Per-channel clipping at user-chosen thresholds.
    shadow_ts = shadow_thresholds if shadow_thresholds is not None else _DEFAULT_SHADOW_THRESHOLDS
    highlight_ts = highlight_thresholds if highlight_thresholds is not None else _DEFAULT_HIGHLIGHT_THRESHOLDS
    clipping_per_channel = {
        "red":       _clipping_at_thresholds(r, shadow_ts, highlight_ts),
        "green":     _clipping_at_thresholds(g, shadow_ts, highlight_ts),
        "blue":      _clipping_at_thresholds(b, shadow_ts, highlight_ts),
        "luminance": _clipping_at_thresholds(lum, shadow_ts, highlight_ts),
        "thresholds": {
            "shadow": list(shadow_ts),
            "highlight": list(highlight_ts),
        },
    }

    # Mode estimate per channel (histogram peak).
    mode_estimate = None
    if compute_mode:
        mode_estimate = {
            "red":       _mode_estimate(r),
            "green":     _mode_estimate(g),
            "blue":      _mode_estimate(b),
            "luminance": _mode_estimate(lum),
        }

    # Per-quadrant background.
    background_quadrants = None
    if quadrant_analysis:
        background_quadrants = _quadrant_background(
            lum, valid_mask=valid_mask if has_valid else None
        )

    # Multi-scale wavelet noise.
    wavelet_noise_scales: list[float] = []
    if wavelet_scales >= 1 and has_valid:
        wavelet_noise_scales = _wavelet_noise_multiscale(lum, n_scales=wavelet_scales)

    # Per-channel SNR.
    channel_snr = _channel_snr_estimates(
        r, g, b, valid_mask=valid_mask if has_valid else None
    )

    # Background estimator box size actually used (for reproducibility).
    bg_box_size = int(bg_est.get("box_size", 0))

    metrics_update = {
        "current_fwhm": star_metrics.get("median_fwhm"),
        "current_background": bg_level,
        "current_noise": bg_noise,
        "wavelet_noise": w_noise,
        "snr_estimate": snr_estimate,
        "dynamic_range_db": dynamic_range_db,
        "channel_stats": channels,
        "background_flatness": flatness,
        "gradient_magnitude": gradient_mag,
        "per_channel_bg": per_channel_bg,
        "green_excess": green_excess,
        "channel_imbalance": channel_imbalance,
        "mean_saturation": color_saturation.get("mean_saturation"),
        "median_saturation": color_saturation.get("median_saturation"),
        "is_linear_estimate": linearity.get("is_linear"),
        "linearity_confidence": linearity.get("confidence"),
        "histogram_skewness": linearity.get("histogram_skewness"),
        "signal_coverage_pct": signal_cov,
        "clipped_shadows_pct": shadows_pct,
        "clipped_highlights_pct": highlights_pct,
        "star_count": star_metrics.get("count"),
        "fwhm_std": star_metrics.get("fwhm_std"),
        "median_star_peak_ratio": star_metrics.get("median_star_peak_ratio"),
        "contrast_ratio": round(contrast_ratio, 5),

        # ── Structured additions (nested dicts / lists) ──
        "pixel_coverage":        pixel_coverage,
        "clipping_per_channel":  clipping_per_channel,
        "mode_estimate":         mode_estimate,
        "background_quadrants":  background_quadrants,
        "wavelet_noise_scales":  wavelet_noise_scales,
        "channel_snr":           channel_snr,
        "star_distribution":     star_metrics.get("star_distribution"),
        "bg_box_size":           bg_box_size,
        "histogram":             histogram if compute_histogram else None,

        # Echo of the parameters the agent chose, so reasoning over prior
        # tool calls can see what thresholds produced this snapshot.
        "analyze_params": {
            "detect_stars":          detect_stars,
            "compute_histogram":     compute_histogram,
            "shadow_thresholds":     list(shadow_ts),
            "highlight_thresholds":  list(highlight_ts),
            "valid_pixel_min":       float(valid_pixel_min),
            "valid_pixel_max":       float(valid_pixel_max),
            "quadrant_analysis":     quadrant_analysis,
            "compute_mode":          compute_mode,
            "star_distribution":     star_distribution,
            "wavelet_scales":        int(wavelet_scales),
            "background_box_size":   background_box_size,
        },
    }

    # ── Regression detection ────────────────────────────────────────────────
    # Compare this snapshot against metadata.last_analysis_snapshot and carry
    # any existing pending warnings forward. filter_resolved drops warnings
    # whose metric has recovered within tolerance of its known-good baseline.
    # merge_warnings preserves each metric's original baseline when a
    # regression persists across multiple tool calls, while refreshing the
    # current/delta fields so summaries reflect the latest measurement.
    # Everything here is informational — the agent decides what to do.
    existing_warnings = state.get("regression_warnings") or []
    baseline_snapshot = (state.get("metadata") or {}).get("last_analysis_snapshot")
    phase_val = getattr(state.get("phase"), "value", state.get("phase")) or ""

    kept_warnings = filter_resolved(existing_warnings, metrics_update)
    new_warnings = detect_regressions(
        metrics_update, baseline_snapshot, phase_val
    )
    regression_warnings_next = merge_warnings(
        kept_warnings, new_warnings, metrics_update
    )

    import json

    # ToolMessage payload: metrics first, then any outstanding regression
    # warnings as a structured list AND the prose summary. The agent reads
    # both — the structured list is precise, the summary is scannable.
    message_payload: dict = {"image": img_path.name, **metrics_update}
    if regression_warnings_next:
        message_payload["regression_warnings"] = regression_warnings_next
        message_payload["regression_summary"] = format_warnings(
            regression_warnings_next
        )

    return Command(
        update={
            "metrics": {**state["metrics"], **metrics_update},
            # Replace semantics on the top-level list: pass the full new list.
            "regression_warnings": regression_warnings_next,
            # metadata uses _merge_dicts reducer — this only touches the
            # snapshot field, other metadata values are preserved.
            "metadata": {"last_analysis_snapshot": metrics_update},
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        message_payload,
                        indent=2,
                        default=str,
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
