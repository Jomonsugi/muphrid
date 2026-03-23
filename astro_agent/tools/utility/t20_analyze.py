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

logger = logging.getLogger(__name__)


# ── Pydantic input schema ──────────────────────────────────────────────────────

class AnalyzeImageInput(BaseModel):
    detect_stars: bool = Field(
        default=True,
        description=(
            "Run star detection via photutils IRAFStarFinder. "
            "Returns count, median_fwhm, median_roundness. "
            "Set False for speed when star metrics are not needed."
        ),
    )
    compute_histogram: bool = Field(
        default=True,
        description=(
            "Compute per-channel histogram summary stats (percentiles). "
            "Used to detect clipping and assess stretch completeness."
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


def _background_estimate(lum: np.ndarray) -> dict:
    """
    Estimate 2D background level and noise using photutils Background2D.

    Background2D tiles the image, computes the median per tile (robust
    to stars/nebula), and interpolates a smooth 2D background model.
    The RMS map uses MAD-based estimation.

    sigma_clip=None prevents the convergence-to-zero bug that affects
    sigma-clipped stats on near-zero linear data.

    Returns bg_level, bg_noise, and the 2D bg_map for gradient analysis.
    """
    try:
        from photutils.background import (
            Background2D,
            MADStdBackgroundRMS,
            MedianBackground,
        )

        h, w = lum.shape
        box_size = max(32, min(256, min(h, w) // 10))

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
    except Exception as e:
        logger.warning(f"Background2D failed ({e}), using MAD fallback")
        bg_level = float(np.median(lum))
        bg_noise = float(mad_std(lum))
        bg_map = None

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

        # Stationary wavelet transform — first level detail coefficients
        coeffs = pywt.swt2(lum_padded, wavelet="bior1.3", level=1, trim_approx=True)
        # coeffs[0] = (cH, cV, cD) — horizontal, vertical, diagonal details
        detail_h = coeffs[0][0]
        detail_v = coeffs[0][1]
        detail_d = coeffs[0][2]

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
    """Return (shadows_pct, highlights_pct) clipping percentages."""
    total = channel.size
    shadows_pct = float(np.sum(channel <= 0.001) / total * 100)
    highlights_pct = float(np.sum(channel >= 0.999) / total * 100)
    return shadows_pct, highlights_pct


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
    lum: np.ndarray, bg_level: float, bg_noise: float
) -> dict:
    """
    Star detection using MAD-based threshold.

    Uses bg_level for background subtraction and bg_noise for the detection
    threshold, avoiding sigma-clipped stats that collapse on linear data.
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

        return {
            "count": len(sources),
            "median_fwhm": float(np.median(fwhm_arr)),
            "median_roundness": float(
                np.median(np.abs(sources["roundness"]))
            ),
            "fwhm_std": float(np.std(fwhm_arr)),
            "median_star_peak_ratio": float(np.median(peak_arr))
            / (bg_noise + 1e-9),
        }
    except Exception as e:
        return {
            **_empty_star_result(),
            "detection_error": (
                f"Star detection failed ({type(e).__name__}: {e}). "
                "count=0 may not reflect actual star count."
            ),
        }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=AnalyzeImageInput)
def analyze_image(
    detect_stars: bool = True,
    compute_histogram: bool = True,
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

    **Clipping**
    - clipped_shadows_pct: percentage of pixels at or below 0.001 (black clip).
      High values in linear data indicate most pixels are at the noise floor —
      expected for short exposures or uncropped images with borders.
    - clipped_highlights_pct: percentage of pixels at or above 0.999 (white clip).
      Indicates star cores or bright regions are saturated.

    **Contrast**
    - contrast_ratio: P95 - P5 of valid pixel luminance. Measures the usable
      dynamic range between shadows and highlights in the current image state.

    **Per-Channel Stats** (channel_stats.red/green/blue)
    - min, max: absolute extreme values
    - mean, median: central tendency of valid pixels
    - std: MAD-based robust standard deviation
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
    # clipped and uninformative. Both are excluded from all statistics,
    # matching PixInsight's implicit zero/one rejection.
    valid_mask = (lum > 0) & (lum < 0.999)
    valid_pct = float(np.sum(valid_mask) / lum.size * 100)
    logger.info(
        f"analyze_image: {valid_pct:.2f}% valid pixels "
        f"({np.sum(valid_mask):,} / {lum.size:,})"
    )

    # If very few valid pixels, the image may be empty or uncropped.
    # Still compute what we can — the agent needs to see the state.
    has_valid = np.sum(valid_mask) >= 100

    # ── Background estimation on valid pixels ──
    if has_valid:
        if valid_pct > 50:
            # Majority of pixels are valid — Background2D works well
            bg_est = _background_estimate(lum)
            bg_level = bg_est["bg_level"]
            bg_noise = bg_est["bg_noise"]
            bg_map = bg_est["bg_map"]
        else:
            # Too many zeros for Background2D (can hang or produce garbage).
            # Use valid-pixel statistics directly.
            bg_level = 0.0
            bg_noise = 0.0
            bg_map = None

        # If Background2D returned zeros or was skipped, use valid pixels
        if bg_noise <= 0 or bg_level <= 0:
            valid_lum = lum[valid_mask]
            bg_level = float(np.median(valid_lum))
            bg_noise = float(mad_std(valid_lum))
    else:
        bg_level = 0.0
        bg_noise = 0.0
        bg_map = None
        bg_est = {"bg_level": 0.0, "bg_noise": 0.0, "bg_map": None}

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
            lum, bg_level=bg_level, bg_noise=bg_noise
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
    }

    import json

    return Command(
        update={
            "metrics": {**state["metrics"], **metrics_update},
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {"image": img_path.name, **metrics_update},
                        indent=2,
                        default=str,
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
