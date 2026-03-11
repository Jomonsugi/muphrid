"""
T20 — analyze_image

The agent's primary diagnostic instrument. Returns comprehensive statistics,
per-channel analysis, noise estimates, star metrics, and background
characterization. Called before and after every image-modifying tool to
evaluate results and guide next decisions.

Architecture:
- Per-channel stats: astropy sigma-clipped statistics (robust to star outliers)
- Background noise: Siril `bgnoise` command (authoritative MAD-based estimate)
- Star metrics: photutils IRAFStarFinder (consistent with registration quality)
- Derived metrics: computed from loaded numpy array

Siril commands:
    load <stem>
    bgnoise
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits as astropy_fits
from astropy.stats import sigma_clipped_stats
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter, sobel

from astro_agent.tools._siril import SirilError, run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class AnalyzeImageInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description="Absolute path to the FITS image to analyze."
    )
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_fits_float32(image_path: Path) -> tuple[np.ndarray, dict]:
    """
    Load a FITS file and return a float32 array in (3, H, W) or (H, W) shape
    plus a dict of header metadata.

    Siril saves color FITS as (NAXIS3=3, NAXIS2=H, NAXIS1=W), which astropy
    reads as (3, H, W). Mono images come through as (H, W).
    """
    with astropy_fits.open(image_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = dict(hdul[0].header)

    # Sanitize NaN/Inf so metrics remain finite and deterministic.
    if not np.isfinite(data).all():
        finite_vals = data[np.isfinite(data)]
        fill = float(np.median(finite_vals)) if finite_vals.size else 0.0
        data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)

    # Normalize pixel range to [0, 1] if values exceed 1 (16-bit stored as int)
    if data.max() > 1.0:
        bit_depth = header.get("BITPIX", 16)
        if bit_depth > 0:
            data = data / (2 ** bit_depth - 1)
        else:
            data = data / data.max()

    return data, header


def _channel_stats(channel: np.ndarray) -> dict:
    """Return sigma-clipped stats for a single channel."""
    mean, median, std = sigma_clipped_stats(channel, sigma=3.0, maxiters=3)
    return {
        "min": float(np.min(channel)),
        "max": float(np.max(channel)),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
    }


def _siril_bgnoise(stem: str, working_dir: str) -> float | None:
    """Run Siril bgnoise on the loaded image. Returns the noise level or None."""
    try:
        result = run_siril_script(
            [f"load {stem}", "bgnoise"],
            working_dir=working_dir,
            timeout=30,
        )
        m = re.search(
            r"background\s+noise[^\d\-]*([\d.e+\-]+)",
            result.stdout,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))
        # Fallback: look for any float after "bgnoise" or "noise level"
        m = re.search(r"[\d.e+\-]+", result.stdout.split("bgnoise")[-1])
        if m:
            return float(m.group(0))
    except (SirilError, ValueError):
        pass
    return None



def _gradient_magnitude(lum: np.ndarray) -> float:
    """
    Estimate large-scale gradient magnitude.
    Smooth heavily, compute Sobel gradient, normalize by image median.
    """
    overall_median = float(np.median(lum))
    if overall_median < 1e-6:
        return 0.0
    smooth = gaussian_filter(lum, sigma=max(lum.shape[0] // 20, 10))
    gx = sobel(smooth, axis=0)
    gy = sobel(smooth, axis=1)
    mean_grad = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    return min(mean_grad / (overall_median + 1e-9), 1.0)


def _flatness_score(lum: np.ndarray, block_size: int = 128) -> float:
    """
    Background flatness: 1.0 = perfectly flat, lower = gradient or vignetting.
    Computed as 1 - (CV of block medians / 0.15), clamped to [0, 1].
    """
    h, w = lum.shape
    block_medians = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = lum[i : i + block_size, j : j + block_size]
            block_medians.append(np.median(block))

    if not block_medians:
        return 1.0
    bm = np.array(block_medians)
    cv = float(np.std(bm) / (np.mean(bm) + 1e-9))
    return float(max(0.0, 1.0 - cv / 0.15))


def _clipping(channel: np.ndarray) -> tuple[float, float]:
    """Return (shadows_pct, highlights_pct) clipping percentages."""
    total = channel.size
    shadows_pct = float(np.sum(channel <= 0.001) / total * 100)
    highlights_pct = float(np.sum(channel >= 0.999) / total * 100)
    return shadows_pct, highlights_pct


def _histogram_percentiles(channel: np.ndarray) -> dict:
    """Return percentile summary of the channel distribution."""
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
    r: np.ndarray, g: np.ndarray, b: np.ndarray, lum: np.ndarray
) -> dict:
    """
    Per-channel median of true sky background pixels only (signal-free region).

    Strategy: build a background mask from luminance — pixels below
    (sigma-clipped median + 2*sigma) are classified as background sky.
    Applying the same mask per channel gives the actual sky color, which
    directly measures color neutralization success/failure:
    - After good PCC/SPCC: red_bg ≈ green_bg ≈ blue_bg
    - After failed neutralization: one channel will be elevated

    Falls back to overall channel median when no background pixels are found
    (fully-filled nebula frames or very bright sky backgrounds).
    """
    _, bg_level, bg_std = sigma_clipped_stats(lum, sigma=2.5, maxiters=3)
    bg_mask = lum < (bg_level + 2.0 * bg_std)
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
    """
    HSV saturation statistics across the image.

    HSV saturation = (cmax - cmin) / cmax, with 0 where cmax == 0.
    Ranges [0, 1]; 0 = neutral gray, 1 = fully saturated.

    Drives T18 saturation_adjust decisions:
    - mean_saturation < 0.10 after stretch: colors likely need significant boost
    - mean_saturation > 0.35 after boost: risk of over-saturation / noise colour
    - high_saturation_pct: fraction of pixels already vivid
    """
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


def _signal_coverage(lum: np.ndarray) -> float:
    """
    Fraction of frame containing significant signal above the sky background (%).

    A 3-sigma threshold above the sigma-clipped background separates signal from sky.
    - < 5 % → sparse star field or compact galaxy; very little nebulosity
    - 5–30 % → galaxy with extended halo or small nebula
    - > 30 % → emission/reflection nebula or Milky Way; signal fills the frame

    Used by the agent to scale T17 (local contrast), T27 (multiscale), and to
    decide whether gradient removal risk of nebula subtraction is high.
    """
    _, bg_level, bg_std = sigma_clipped_stats(lum, sigma=2.5, maxiters=3)
    threshold = bg_level + 3.0 * bg_std
    return float(np.sum(lum > threshold) / lum.size * 100)


def _linearity_analysis(lum: np.ndarray, overall_median: float) -> dict:
    """
    Quantify how linear (un-stretched) the image is.

    A linear FITS stack has:
    1. Very low median (most pixels near sky background)
    2. Heavy right-skewness (long tail from stars/nebula, bulk at zero)

    A stretched image has:
    1. Median typically > 0.15 (for aggressive stretches, > 0.3)
    2. Skewness drops toward 0 or negative as the histogram broadens

    Returns:
    - is_linear: consensus from median threshold AND skewness
    - histogram_skewness: moment-3; > 3.0 almost certainly linear
    - median_brightness: overall luminance median (same as background.median)
    - confidence: "high" / "medium" / "low" when median and skew disagree
    """
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
        is_linear = median_says_linear  # median is the primary gate
        confidence = "medium"
    else:
        is_linear, confidence = False, "high"
    return {
        "is_linear": is_linear,
        "confidence": confidence,
        "histogram_skewness": round(skewness, 3),
        "median_brightness": round(overall_median, 5),
    }


def _detect_stars_full(lum: np.ndarray) -> dict:
    """
    Extended star detection: adds fwhm_std and median_star_peak_ratio.

    - fwhm_std: spread of FWHM across detected stars.
      Small std → uniform PSF → use `makepsf stars` with default profile.
      Large std → position-dependent PSF, consider manual Moffat measurement.
    - median_star_peak_ratio: median(star peak) / background_median.
      High ratio → stars dominate the frame → star reduction (T26) may be warranted.
      Low ratio → stars are faint relative to nebula → leave stars alone.
    """
    try:
        from photutils.detection import IRAFStarFinder

        _, background, std = sigma_clipped_stats(lum, sigma=3.0)
        if std <= 0:
            return {
                "count": 0,
                "median_fwhm": None,
                "median_roundness": None,
                "fwhm_std": None,
                "median_star_peak_ratio": None,
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            finder = IRAFStarFinder(threshold=5.0 * std, fwhm=3.0, minsep_fwhm=2.0)
            sources = finder(lum - background)

        if sources is None or len(sources) == 0:
            return {
                "count": 0,
                "median_fwhm": None,
                "median_roundness": None,
                "fwhm_std": None,
                "median_star_peak_ratio": None,
            }

        fwhm_arr = np.array(sources["fwhm"])
        peak_arr = np.array(sources["peak"])
        bg_med = float(np.median(lum))

        return {
            "count": len(sources),
            "median_fwhm": float(np.median(fwhm_arr)),
            "median_roundness": float(np.median(np.abs(sources["roundness"]))),
            "fwhm_std": float(np.std(fwhm_arr)),
            "median_star_peak_ratio": float(np.median(peak_arr)) / (bg_med + 1e-9),
        }
    except Exception:
        return {
            "count": 0,
            "median_fwhm": None,
            "median_roundness": None,
            "fwhm_std": None,
            "median_star_peak_ratio": None,
        }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=AnalyzeImageInput)
def analyze_image(
    working_dir: str,
    image_path: str,
    detect_stars: bool = True,
    compute_histogram: bool = True,
) -> dict:
    """
    Comprehensive image analysis tool — the agent's primary decision instrument.

    Returns per-channel statistics, background noise, star metrics, and derived
    quality indicators.

    Key metrics and their decision thresholds:
    - background.gradient_magnitude > 0.05: run remove_gradient
    - background.per_channel_bg — after color calibration, red/green/blue should converge;
      channel spread > 0.02 indicates incomplete neutralization
    - color_balance.green_excess > 0.02: run remove_green_noise
    - snr_estimate < 30: skip deconvolution (insufficient SNR)
    - background.flatness_score < 0.9: re-run gradient removal
    - stars.median_fwhm > 4.0 px (pre-stretch): investigate registration quality
    - stars.fwhm_std > 1.5 px: position-dependent PSF, use manual Moffat for T13
    - stars.median_star_peak_ratio > 50: bright stars dominate; T26 star reduction advised
    - clipping.shadows_pct > 1.0: reduce stretch aggressiveness
    - linearity.is_linear + linearity.confidence: phase gate for linear/non-linear tools
    - signal_coverage_pct < 5: sparse target, be conservative with T09 / T17 / T27
    - color.mean_saturation < 0.10 post-stretch: T18 saturation boost likely needed
    - contrast_ratio < 0.3 post-stretch: image is flat, T16 curves adjustment needed
    - contrast_ratio > 0.8: very contrasty, be cautious with further local contrast

    All statistics use sigma-clipping (3σ, 3 iterations) for robustness
    against star outliers. Background noise uses Siril's bgnoise (MAD-based).
    Linearity uses dual consensus: median threshold + histogram skewness.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and normalize
    data, header = _load_fits_float32(img_path)

    # Determine channel arrays — Siril stores (3, H, W) for color
    if data.ndim == 3 and data.shape[0] == 3:
        r, g, b = data[0], data[1], data[2]
        rgb_hwc = np.moveaxis(data, 0, -1)  # (H, W, 3)
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        is_color = True
    elif data.ndim == 3 and data.shape[2] == 3:
        rgb_hwc = data
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        is_color = True
    else:
        # Mono: squeeze to (H, W)
        lum = data.squeeze().astype(np.float32)
        r = g = b = lum
        is_color = False

    # Per-channel statistics
    channels = {
        "red":   _channel_stats(r),
        "green": _channel_stats(g),
        "blue":  _channel_stats(b),
    }

    # Background noise via Siril bgnoise (authoritative)
    bgnoise = _siril_bgnoise(img_path.stem, working_dir)
    if bgnoise is None:
        # Fallback: robust MAD estimate from luminance background region
        _, lum_median, lum_std = sigma_clipped_stats(lum, sigma=3.0)
        bgnoise = float(lum_std)

    # Derived scalar metrics
    overall_median = float(np.median(lum))
    overall_mean = float(np.mean(lum))
    overall_max = float(np.max(lum))

    gradient_mag = _gradient_magnitude(lum)
    flatness = _flatness_score(lum)

    # Per-channel background sky medians (mask out signal pixels)
    per_channel_bg = _background_channel_medians(r, g, b, lum)

    snr_estimate = float(overall_median / (bgnoise + 1e-9)) if bgnoise > 0 else 0.0

    # Dynamic range in dB
    dynamic_range_db = float(
        20 * np.log10((overall_max + 1e-9) / (bgnoise + 1e-9))
    ) if bgnoise > 0 else 0.0

    # Clipping — report worst channel
    r_shadows, r_highlights = _clipping(r)
    g_shadows, g_highlights = _clipping(g)
    b_shadows, b_highlights = _clipping(b)
    shadows_pct = max(r_shadows, g_shadows, b_shadows)
    highlights_pct = max(r_highlights, g_highlights, b_highlights)

    # Color balance (green excess — key DSLR quality indicator)
    green_excess = float(channels["green"]["mean"] - (channels["red"]["mean"] + channels["blue"]["mean"]) / 2)
    channel_imbalance = float(
        max(channels["red"]["mean"], channels["green"]["mean"], channels["blue"]["mean"])
        - min(channels["red"]["mean"], channels["green"]["mean"], channels["blue"]["mean"])
    )

    # HSV saturation — drives T18 saturation_adjust decisions
    color_saturation = _color_saturation(r, g, b) if is_color else {
        "mean_saturation": None,
        "median_saturation": None,
        "high_saturation_pct": None,
    }

    # Signal coverage — fraction of frame containing nebula/galaxy signal
    signal_cov = _signal_coverage(lum)

    # Linearity analysis — dual consensus replaces the fragile median < 0.15 heuristic
    linearity = _linearity_analysis(lum, overall_median)

    # Star detection (extended — includes fwhm_std, median_star_peak_ratio)
    star_metrics: dict = {
        "count": 0,
        "median_fwhm": None,
        "median_roundness": None,
        "fwhm_std": None,
        "median_star_peak_ratio": None,
    }
    if detect_stars:
        star_metrics = _detect_stars_full(lum)

    # Histogram percentiles (per channel)
    histogram: dict = {}
    if compute_histogram:
        histogram = {
            "red":   _histogram_percentiles(r),
            "green": _histogram_percentiles(g),
            "blue":  _histogram_percentiles(b),
        }

    # Contrast ratio: usable dynamic range expressed as p95 - p5 of luminance.
    # Measures the tonal spread the image actually occupies post-stretch.
    # < 0.3 after stretch → image is flat, needs curves adjustment.
    # 0.4–0.7 → good tonal distribution. > 0.8 → very contrasty.
    contrast_ratio = float(np.percentile(lum, 95) - np.percentile(lum, 5))

    return {
        "channels": channels,
        "noise": {
            "background_noise": bgnoise,
            "method": "siril_bgnoise_MAD",
        },
        "background": {
            "median": overall_median,
            "flatness_score": flatness,
            "gradient_magnitude": gradient_mag,
            "per_channel_bg": per_channel_bg,
        },
        "stars": star_metrics,
        "dynamic_range_db": dynamic_range_db,
        "snr_estimate": snr_estimate,
        "linearity": linearity,
        "signal_coverage_pct": signal_cov,
        "clipping": {
            "shadows_pct": shadows_pct,
            "highlights_pct": highlights_pct,
        },
        "color_balance": {
            "green_excess": green_excess,
            "channel_imbalance": channel_imbalance,
        },
        "color": color_saturation,
        "contrast_ratio": round(contrast_ratio, 5),
        "histogram": histogram,
        "image_shape": list(data.shape),
        "is_color": is_color,
    }
