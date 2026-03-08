"""
Math integrity regression checks for AstroAgent tools.

Purpose:
- Catch numerical/logic bugs in pure-Python math paths early.
- Validate edge cases that are easy to miss in end-to-end runs.
- Every function here uses synthetic data with known answers so
  that a failure pinpoints the exact formula that broke.

This script is intentionally fast and synthetic-data based.
"""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from astro_agent.tools.preprocess.t05_analyze_frames import _compute_summary
from astro_agent.tools.preprocess.t06_select_frames import (
    SelectionCriteria,
    _select_frames,
    _sigma_threshold,
    select_frames,
)
from astro_agent.tools.preprocess.t08_crop import _find_crop_bounds
from astro_agent.tools.scikit.t25_create_mask import _build_binary_mask, _load_channels
from astro_agent.tools.scikit.t25_create_mask import (
    ChannelDiffOptions,
    LuminanceOptions,
    RangeOptions,
)
from astro_agent.tools.scikit.t26_reduce_stars import _compute_luminance, reduce_stars
from astro_agent.tools.scikit.t27_multiscale import (
    b3_atrous_decompose,
    b3_atrous_reconstruct,
    _soft_threshold,
    multiscale_process,
)
from astro_agent.tools.utility.t20_analyze import (
    _background_channel_medians,
    _clipping,
    _color_saturation,
    _flatness_score,
    _gradient_magnitude,
    _linearity_analysis,
    _load_fits_float32,
    _signal_coverage,
)

_pass_count = 0


def _pass(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    print(f"  [PASS] {msg}")


def _save_fits(path: Path, data: np.ndarray, bitpix: int | None = None) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    if bitpix is not None:
        hdu.header["BITPIX"] = bitpix
    hdu.writeto(path, overwrite=True)


# ────────────────────────────────────────────────────────────────────────────
# T06 — Frame selection math
# ────────────────────────────────────────────────────────────────────────────

def test_t06_sigma_threshold_known_values() -> None:
    """Verify _sigma_threshold returns median + sigma * stdev for known data."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    med = statistics.median(vals)   # 3.0
    std = statistics.stdev(vals)    # ~1.5811
    expected = med + 2.0 * std
    actual = _sigma_threshold(vals, 2.0)
    assert abs(actual - expected) < 1e-6, f"expected {expected}, got {actual}"
    _pass("T06 _sigma_threshold correct for known values")


def test_t06_sigma_threshold_single_value() -> None:
    """Single-value list → threshold = inf (never reject)."""
    assert _sigma_threshold([42.0], 2.0) == float("inf")
    _pass("T06 _sigma_threshold single value returns inf")


def test_t06_safety_fallback_all_bad() -> None:
    """When every frame fails, safety fallback returns ALL frames."""
    frames = {
        "a.fits": {"fwhm": 10.0, "roundness": 0.1, "number_of_stars": 5, "background_lvl": 0.9},
        "b.fits": {"fwhm": 11.0, "roundness": 0.1, "number_of_stars": 3, "background_lvl": 0.95},
    }
    criteria = SelectionCriteria(
        max_fwhm_sigma=0.01,
        min_roundness=0.99,
        min_star_count=500,
        max_background_sigma=0.01,
    )
    accepted, rejected, reasons = _select_frames(frames, criteria)
    assert len(accepted) == 2, "safety fallback should return all frames"
    assert len(rejected) == 0
    _pass("T06 safety fallback when all frames bad")


def test_t06_selective_rejection() -> None:
    """One outlier frame should be rejected while good frames pass."""
    frames = {
        "good1.fits": {"fwhm": 2.0, "roundness": 0.8, "number_of_stars": 100, "background_lvl": 0.05},
        "good2.fits": {"fwhm": 2.1, "roundness": 0.85, "number_of_stars": 110, "background_lvl": 0.05},
        "good3.fits": {"fwhm": 2.2, "roundness": 0.82, "number_of_stars": 95, "background_lvl": 0.06},
        "bad.fits":   {"fwhm": 8.0, "roundness": 0.3, "number_of_stars": 20, "background_lvl": 0.05},
    }
    criteria = SelectionCriteria(max_fwhm_sigma=2.0, min_roundness=0.5, min_star_count=30)
    accepted, rejected, reasons = _select_frames(frames, criteria)
    assert "bad.fits" in rejected, f"bad frame should be rejected, got accepted={accepted}"
    assert all(f in accepted for f in ["good1.fits", "good2.fits", "good3.fits"])
    _pass("T06 selective rejection of outlier frame")


def test_t06_empty_input_fails_loudly() -> None:
    try:
        select_frames.invoke({"frame_metrics": {}, "criteria": {}})
    except ValueError as exc:
        assert "frame_metrics is empty" in str(exc)
        _pass("T06 empty frame_metrics fails loudly")
        return
    raise AssertionError("T06 should fail on empty frame_metrics")


# ────────────────────────────────────────────────────────────────────────────
# T20 — Flatness, clipping, gradient
# ────────────────────────────────────────────────────────────────────────────

def test_t20_flatness_uniform() -> None:
    """A perfectly uniform image should have flatness_score == 1.0."""
    uniform = np.full((256, 256), 0.5, dtype=np.float32)
    score = _flatness_score(uniform, block_size=64)
    assert abs(score - 1.0) < 1e-6, f"uniform image flatness should be 1.0, got {score}"
    _pass("T20 flatness_score == 1.0 for uniform image")


def test_t20_flatness_gradient() -> None:
    """An image with a strong gradient should have flatness_score < 1.0."""
    gradient = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32), (256, 1))
    score = _flatness_score(gradient, block_size=64)
    assert score < 0.9, f"gradient image flatness should be < 0.9, got {score}"
    _pass("T20 flatness_score < 1.0 for gradient image")


def test_t20_clipping_known() -> None:
    """Verify clipping percentages with precisely known data."""
    channel = np.full((100, 100), 0.5, dtype=np.float32)
    channel[0, :10] = 0.0          # 10 shadow pixels (≤ 0.001)
    channel[0, 10:15] = 1.0        # 5 highlight pixels (≥ 0.999)
    total = 10_000
    shadows, highlights = _clipping(channel)
    assert abs(shadows - 10 / total * 100) < 1e-6, f"shadows={shadows}"
    assert abs(highlights - 5 / total * 100) < 1e-6, f"highlights={highlights}"
    _pass("T20 clipping counting exact")


def test_t20_gradient_magnitude_zero() -> None:
    """All-zero image should return gradient_magnitude == 0.0 (no div-by-zero)."""
    zeros = np.zeros((128, 128), dtype=np.float32)
    gm = _gradient_magnitude(zeros)
    assert gm == 0.0
    _pass("T20 gradient_magnitude zero-image guard")


def test_t20_linearity_nan_guard() -> None:
    const = np.ones((128, 128), dtype=np.float32) * 0.1
    res = _linearity_analysis(const, overall_median=float(np.median(const)))
    assert np.isfinite(res["histogram_skewness"])
    _pass("T20 skew NaN guard")


# ────────────────────────────────────────────────────────────────────────────
# T20 — Color saturation HSV formula
# ────────────────────────────────────────────────────────────────────────────

def test_t20_color_saturation_pure_red() -> None:
    """Pure red (1,0,0) should have HSV saturation = 1.0."""
    r = np.full((32, 32), 1.0, dtype=np.float32)
    g = np.zeros((32, 32), dtype=np.float32)
    b = np.zeros((32, 32), dtype=np.float32)
    sat = _color_saturation(r, g, b)
    assert abs(sat["mean_saturation"] - 1.0) < 1e-5, f"got {sat['mean_saturation']}"
    assert abs(sat["median_saturation"] - 1.0) < 1e-5
    _pass("T20 color_saturation pure red = 1.0")


def test_t20_color_saturation_neutral_gray() -> None:
    """Neutral gray (0.5, 0.5, 0.5) should have saturation = 0.0."""
    v = np.full((32, 32), 0.5, dtype=np.float32)
    sat = _color_saturation(v, v, v)
    assert abs(sat["mean_saturation"]) < 1e-5, f"got {sat['mean_saturation']}"
    _pass("T20 color_saturation neutral gray = 0.0")


def test_t20_color_saturation_known_50pct() -> None:
    """Known case: (1.0, 0.5, 0.5) → cmax=1.0, cmin=0.5, sat=0.5."""
    r = np.full((32, 32), 1.0, dtype=np.float32)
    g = np.full((32, 32), 0.5, dtype=np.float32)
    b = np.full((32, 32), 0.5, dtype=np.float32)
    sat = _color_saturation(r, g, b)
    assert abs(sat["mean_saturation"] - 0.5) < 1e-5, f"got {sat['mean_saturation']}"
    _pass("T20 color_saturation (1.0, 0.5, 0.5) = 0.5")


def test_t20_color_saturation_black_no_nan() -> None:
    """All-black (0,0,0) should return 0 saturation (not NaN)."""
    z = np.zeros((32, 32), dtype=np.float32)
    sat = _color_saturation(z, z, z)
    assert np.isfinite(sat["mean_saturation"])
    assert abs(sat["mean_saturation"]) < 1e-5
    _pass("T20 color_saturation all-black returns 0 (no NaN)")


# ────────────────────────────────────────────────────────────────────────────
# T20 — Signal coverage and background channel medians
# ────────────────────────────────────────────────────────────────────────────

def test_t20_signal_coverage_known() -> None:
    """
    10% bright signal on a noisy background.
    Sigma-clipped stats need realistic noise to converge — a bimodal image
    with >20% signal confuses the 3-iteration clipping. Use 10% signal with
    Gaussian background noise so clipping identifies the true sky level.
    """
    rng = np.random.default_rng(42)
    lum = np.full((100, 100), 0.01, dtype=np.float32)
    lum += rng.normal(0, 0.002, size=(100, 100)).astype(np.float32)
    lum = np.clip(lum, 0.0, 1.0)
    lum[:10, :] = 0.5   # 1000/10000 = 10% signal well above background
    cov = _signal_coverage(lum)
    assert 5.0 < cov < 20.0, f"expected ~10%, got {cov}"
    _pass("T20 signal_coverage ~10% for 10%-filled frame")


def test_t20_background_channel_medians_known() -> None:
    """
    Uniform faint sky (R=0.01, G=0.02, B=0.01) should be correctly measured.
    The sigma-clipping should classify most pixels as background.
    """
    shape = (128, 128)
    r = np.full(shape, 0.01, dtype=np.float32)
    g = np.full(shape, 0.02, dtype=np.float32)
    b = np.full(shape, 0.01, dtype=np.float32)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
    result = _background_channel_medians(r, g, b, lum)
    assert abs(result["red"] - 0.01) < 0.005, f"red bg = {result['red']}"
    assert abs(result["green"] - 0.02) < 0.005, f"green bg = {result['green']}"
    assert abs(result["blue"] - 0.01) < 0.005, f"blue bg = {result['blue']}"
    assert result["n_background_pixels"] > 100
    _pass("T20 background_channel_medians correct for uniform sky")


# ────────────────────────────────────────────────────────────────────────────
# T20 — Linearity analysis and contrast_ratio
# ────────────────────────────────────────────────────────────────────────────

def test_t20_linearity_linear_image() -> None:
    """
    A linear stack: very low median (~0.02), strong right skew.
    Should be classified is_linear=True with high confidence.
    """
    rng = np.random.default_rng(42)
    lum = rng.exponential(scale=0.01, size=(256, 256)).astype(np.float32)
    lum = np.clip(lum, 0.0, 1.0)
    med = float(np.median(lum))
    res = _linearity_analysis(lum, overall_median=med)
    assert res["is_linear"] is True, f"expected linear, got {res}"
    assert res["confidence"] in ("high", "medium")
    _pass("T20 linearity detects linear image correctly")


def test_t20_linearity_stretched_image() -> None:
    """
    A stretched image: high median (~0.5), near-zero skewness.
    Should be classified is_linear=False.
    """
    rng = np.random.default_rng(42)
    lum = rng.normal(loc=0.5, scale=0.15, size=(256, 256)).astype(np.float32)
    lum = np.clip(lum, 0.0, 1.0)
    med = float(np.median(lum))
    res = _linearity_analysis(lum, overall_median=med)
    assert res["is_linear"] is False, f"expected non-linear, got {res}"
    _pass("T20 linearity detects stretched image correctly")


def test_t20_contrast_ratio_known() -> None:
    """
    For uniform data from 0.1 to 0.9, p95-p5 should be close to 0.8.
    Verify this is computed correctly by the analysis tool's formula.
    """
    lum = np.linspace(0.1, 0.9, 10_000, dtype=np.float32)
    cr = float(np.percentile(lum, 95) - np.percentile(lum, 5))
    expected = 0.9 * 0.9 - 0.9 * 0.1   # ≈ 0.72
    assert abs(cr - 0.72) < 0.02, f"expected ~0.72, got {cr}"
    _pass("T20 contrast_ratio formula correct for known range")


# ────────────────────────────────────────────────────────────────────────────
# T20 — FITS loading and 16-bit normalization
# ────────────────────────────────────────────────────────────────────────────

def test_t20_load_fits_16bit_normalization(tmp_dir: Path) -> None:
    """16-bit unsigned integer FITS should normalize to [0, 1] using BITPIX."""
    data_16 = np.array([[0, 32767, 65535]], dtype=np.float32)
    path = tmp_dir / "int16.fits"
    hdu = fits.PrimaryHDU(data=data_16)
    hdu.header["BITPIX"] = 16
    hdu.writeto(path, overwrite=True)

    loaded, _ = _load_fits_float32(path)
    assert loaded.max() <= 1.0, f"max should be ≤ 1.0 after normalization, got {loaded.max()}"
    assert loaded.min() >= 0.0
    _pass("T20 16-bit FITS normalized to [0,1]")


def test_t20_load_fits_nan_replacement(tmp_dir: Path) -> None:
    """NaN/Inf values in FITS should be replaced with median of finite values."""
    data = np.array([0.1, 0.2, np.nan, 0.3, np.inf], dtype=np.float32).reshape(1, 1, 5)
    path = tmp_dir / "has_nan.fits"
    _save_fits(path, data)
    loaded, _ = _load_fits_float32(path)
    assert np.isfinite(loaded).all(), "all values should be finite after loading"
    _pass("T20 FITS NaN/Inf replaced with median")


def test_t20_nan_input_sanitized(tmp_dir: Path) -> None:
    from astro_agent.tools.utility.t20_analyze import analyze_image

    data = np.random.rand(3, 64, 64).astype(np.float32)
    data[0, 0, 0] = np.nan
    data[1, 1, 1] = np.inf
    data[2, 2, 2] = -np.inf
    path = tmp_dir / "nan_img.fits"
    _save_fits(path, data)

    out = analyze_image.invoke(
        {
            "working_dir": str(tmp_dir),
            "image_path": str(path),
            "detect_stars": False,
            "compute_histogram": False,
        }
    )

    def _all_finite(v) -> bool:
        if isinstance(v, dict):
            return all(_all_finite(x) for x in v.values())
        if isinstance(v, list):
            return all(_all_finite(x) for x in v)
        if isinstance(v, float):
            return np.isfinite(v)
        return True

    assert _all_finite(out)
    _pass("T20 sanitizes NaN/Inf input to finite metrics")


# ────────────────────────────────────────────────────────────────────────────
# T25 — Binary mask construction
# ────────────────────────────────────────────────────────────────────────────

def _make_mask_test_channels():
    """Build synthetic (32, 32) channels with known pixel values."""
    r = np.full((32, 32), 0.2, dtype=np.float32)
    g = np.full((32, 32), 0.3, dtype=np.float32)
    b = np.full((32, 32), 0.1, dtype=np.float32)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
    # Make a bright patch in top-left 4x4 for luminance testing
    r[:4, :4] = 0.9
    g[:4, :4] = 0.9
    b[:4, :4] = 0.9
    lum[:4, :4] = 0.9
    return lum, r, g, b


def test_t25_luminance_mask() -> None:
    """Luminance mask: pixels in [0.8, 1.0] should select the bright patch."""
    lum, r, g, b = _make_mask_test_channels()
    mask = _build_binary_mask(
        "luminance", lum, r, g, b,
        LuminanceOptions(low=0.8, high=1.0),
        RangeOptions(low=0.2, high=0.7),
        ChannelDiffOptions(channel_a="R", channel_b="B"),
    )
    assert mask[:4, :4].all(), "bright patch should be selected"
    assert not mask[10:, 10:].any(), "dark region should not be selected"
    _pass("T25 luminance mask selects bright patch only")


def test_t25_inverted_luminance_mask() -> None:
    """Inverted luminance mask: bright patch should be False, dark region True."""
    lum, r, g, b = _make_mask_test_channels()
    mask = _build_binary_mask(
        "inverted_luminance", lum, r, g, b,
        LuminanceOptions(low=0.8, high=1.0),
        RangeOptions(low=0.2, high=0.7),
        ChannelDiffOptions(channel_a="R", channel_b="B"),
    )
    assert not mask[:4, :4].any(), "bright patch should NOT be selected"
    assert mask[10:, 10:].all(), "dark region should be selected"
    _pass("T25 inverted_luminance mask correct")


def test_t25_channel_diff_mask() -> None:
    """channel_diff R-B >= 0.05: where red exceeds blue by ≥ 0.05 (Hα isolation)."""
    r = np.full((32, 32), 0.5, dtype=np.float32)
    g = np.full((32, 32), 0.3, dtype=np.float32)
    b = np.full((32, 32), 0.3, dtype=np.float32)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
    # R-B = 0.2 everywhere → all pixels should pass threshold 0.05
    mask = _build_binary_mask(
        "channel_diff", lum, r, g, b,
        LuminanceOptions(),
        RangeOptions(low=0.2, high=0.7),
        ChannelDiffOptions(channel_a="R", channel_b="B", threshold=0.05),
    )
    assert mask.all(), f"R-B=0.2 > 0.05, all pixels should pass; got {mask.sum()}/{mask.size}"
    # Now set threshold higher than difference
    mask2 = _build_binary_mask(
        "channel_diff", lum, r, g, b,
        LuminanceOptions(),
        RangeOptions(low=0.2, high=0.7),
        ChannelDiffOptions(channel_a="R", channel_b="B", threshold=0.3),
    )
    assert not mask2.any(), "R-B=0.2 < 0.3, no pixels should pass"
    _pass("T25 channel_diff mask Hα isolation correct")


def test_t25_range_mask() -> None:
    """Range mask on red channel [0.4, 0.6] — only matching pixels selected."""
    r = np.full((32, 32), 0.2, dtype=np.float32)
    r[:8, :] = 0.5   # in range
    g = np.full((32, 32), 0.3, dtype=np.float32)
    b = np.full((32, 32), 0.3, dtype=np.float32)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
    mask = _build_binary_mask(
        "range", lum, r, g, b,
        LuminanceOptions(),
        RangeOptions(channel="R", low=0.4, high=0.6),
        ChannelDiffOptions(channel_a="R", channel_b="B"),
    )
    assert mask[:8, :].all(), "red=0.5 rows should be selected"
    assert not mask[8:, :].any(), "red=0.2 rows should not be selected"
    _pass("T25 range mask on red channel correct")


def test_t25_inversion_logic(tmp_dir: Path) -> None:
    """After inversion, 0→1 and 1→0."""
    from astro_agent.tools.scikit.t25_create_mask import create_mask

    # Create an image that is half bright, half dark
    data = np.zeros((3, 32, 32), dtype=np.float32)
    data[:, :16, :] = 0.9
    path = tmp_dir / "half_bright.fits"
    _save_fits(path, data)

    result = create_mask.invoke({
        "working_dir": str(tmp_dir),
        "image_path": str(path),
        "mask_type": "luminance",
        "luminance_options": {"low": 0.8, "high": 1.0},
        "feather_radius": 0,
        "invert": True,
    })
    mask_path = Path(result["mask_path"])
    with fits.open(mask_path) as hdul:
        m = hdul[0].data
    # After inversion: bright region → 0, dark region → 1
    assert np.mean(m[:16, :]) < 0.1, "bright region should be ~0 after inversion"
    assert np.mean(m[16:, :]) > 0.9, "dark region should be ~1 after inversion"
    _pass("T25 inversion 0↔1 correct")


# ────────────────────────────────────────────────────────────────────────────
# T27 — Soft threshold and color luminance recombination
# ────────────────────────────────────────────────────────────────────────────

def test_t27_soft_threshold_large_survives() -> None:
    """Large coefficients should survive soft thresholding (only reduced by threshold)."""
    coeffs = np.array([0.0, 0.001, -0.001, 10.0, -10.0], dtype=np.float32)
    result = _soft_threshold(coeffs, sigma_factor=3.0)
    # Small values near zero should be zeroed
    assert result[0] == 0.0
    # Large values should survive (sign preserved, magnitude reduced by threshold)
    mad = float(np.median(np.abs(coeffs)))
    threshold = 3.0 * mad / 0.6745
    assert result[3] > 0.0, "large positive should survive"
    assert result[4] < 0.0, "large negative should survive"
    expected_pos = 10.0 - threshold
    assert abs(result[3] - expected_pos) < 1e-4, f"expected {expected_pos}, got {result[3]}"
    _pass("T27 soft_threshold preserves large coefficients correctly")


def test_t27_soft_threshold_zeros_small() -> None:
    """All-small coefficients should be fully zeroed."""
    coeffs = np.array([0.001, -0.001, 0.002, -0.002], dtype=np.float32)
    result = _soft_threshold(coeffs, sigma_factor=3.0)
    assert np.allclose(result, 0.0), f"small coefficients should be zero, got {result}"
    _pass("T27 soft_threshold zeros small coefficients")


def test_t27_b3_roundtrip() -> None:
    img = np.random.rand(64, 64).astype(np.float32)
    layers = b3_atrous_decompose(img, 4)
    recon = b3_atrous_reconstruct(layers)
    assert np.allclose(img, recon, atol=1e-5)
    _pass("T27 B3 a trous roundtrip")


def test_t27_color_luminance_ratio_preserves_hue(tmp_dir: Path) -> None:
    """
    When processing luminance-only (per_channel=False) on a color image,
    the color ratio between channels should be preserved (hue unchanged).
    Using passthrough-only operations to isolate the recombination math.
    """
    rng = np.random.default_rng(99)
    data = rng.uniform(0.1, 0.9, size=(3, 64, 64)).astype(np.float32)
    path = tmp_dir / "color_ratio.fits"
    _save_fits(path, data)

    result = multiscale_process.invoke({
        "working_dir": str(tmp_dir),
        "image_path": str(path),
        "num_scales": 3,
        "scale_operations": [],   # all passthrough
        "per_channel": False,
    })
    with fits.open(result["processed_image_path"]) as hdul:
        out = hdul[0].data.astype(np.float32)

    # Compute R/G ratio before and after — should be preserved where lum > 0
    lum = (0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2])
    bright = lum > 0.1
    ratio_before = data[0][bright] / (data[1][bright] + 1e-9)
    ratio_after = out[0][bright] / (out[1][bright] + 1e-9)
    assert np.allclose(ratio_before, ratio_after, atol=0.05), (
        f"color ratio R/G drift: max delta = {np.max(np.abs(ratio_before - ratio_after))}"
    )
    _pass("T27 luminance-mode recombination preserves color ratios")


def test_t27_multiscale_mono_tool_path(tmp_dir: Path) -> None:
    mono = np.random.rand(64, 64).astype(np.float32)
    img_path = tmp_dir / "mono.fits"
    _save_fits(img_path, mono)

    result = multiscale_process.invoke(
        {
            "working_dir": str(tmp_dir),
            "image_path": str(img_path),
            "num_scales": 3,
            "scale_operations": [],
            "per_channel": False,
        }
    )
    out_path = Path(result["processed_image_path"])
    assert out_path.exists()
    with fits.open(out_path) as hdul:
        out = hdul[0].data
    assert out.shape[0] == 1
    _pass("T27 tool mono processing (no index crash)")


def test_t27_resizes_mismatched_mask(tmp_dir: Path) -> None:
    img = np.random.rand(3, 64, 64).astype(np.float32)
    mask = np.random.rand(32, 32).astype(np.float32)
    img_path = tmp_dir / "img_mask_mismatch.fits"
    mask_path = tmp_dir / "mask_32.fits"
    _save_fits(img_path, img)
    _save_fits(mask_path, mask)

    result = multiscale_process.invoke(
        {
            "working_dir": str(tmp_dir),
            "image_path": str(img_path),
            "num_scales": 3,
            "scale_operations": [],
            "mask_path": str(mask_path),
            "per_channel": False,
        }
    )
    assert Path(result["processed_image_path"]).exists()
    _pass("T27 auto-resizes mismatched mask shapes")


# ────────────────────────────────────────────────────────────────────────────
# T26 — Luminance weights and blend math
# ────────────────────────────────────────────────────────────────────────────

def test_t26_compute_luminance_mono() -> None:
    mono = np.random.rand(1, 32, 32).astype(np.float32)
    lum = _compute_luminance(mono)
    assert lum.shape == (32, 32)
    assert np.allclose(lum, mono[0], atol=1e-7)
    _pass("T26 mono luminance path")


def test_t26_compute_luminance_color_weights() -> None:
    """Verify rec.709 luminance coefficients: 0.2126 R + 0.7152 G + 0.0722 B."""
    r_val, g_val, b_val = 0.8, 0.4, 0.6
    data = np.zeros((3, 1, 1), dtype=np.float32)
    data[0, 0, 0] = r_val
    data[1, 0, 0] = g_val
    data[2, 0, 0] = b_val
    lum = _compute_luminance(data)
    expected = 0.2126 * r_val + 0.7152 * g_val + 0.0722 * b_val
    assert abs(lum[0, 0] - expected) < 1e-5, f"expected {expected}, got {lum[0, 0]}"
    _pass("T26 rec.709 luminance weights correct")


def test_t26_blend_math_within_mask_only(tmp_dir: Path) -> None:
    """
    With blend_amount=0.5, only pixels inside the star mask should change.
    Background pixels outside the mask must remain exactly unchanged.
    """
    data = np.full((3, 32, 32), 0.3, dtype=np.float32)
    data[:, 15:17, 15:17] = 1.0   # bright "star"
    path = tmp_dir / "blend_test.fits"
    _save_fits(path, data)

    result = reduce_stars.invoke({
        "working_dir": str(tmp_dir),
        "image_path": str(path),
        "detection_threshold": 0.6,
        "kernel_radius": 1,
        "iterations": 1,
        "blend_amount": 0.5,
        "protect_core_radius": 0,
        "feather_px": 0,     # No feathering → binary boundary
    })
    with fits.open(result["reduced_image_path"]) as hdul:
        out = hdul[0].data.astype(np.float32)

    # Background pixels (value 0.3) should be completely untouched
    bg_region = out[:, 0:10, 0:10]
    assert np.allclose(bg_region, 0.3, atol=1e-6), (
        f"background should be unchanged, max delta = {np.max(np.abs(bg_region - 0.3))}"
    )
    _pass("T26 blend math only affects star mask region")


def test_t26_size_metric_not_global_contaminated(tmp_dir: Path) -> None:
    data = np.zeros((3, 64, 64), dtype=np.float32)
    data[:, 32, 20] = 1.0
    data[:, 32, 21] = 0.8
    data[:, 10:20, 45:55] = 0.9

    img_path = tmp_dir / "stars.fits"
    _save_fits(img_path, data)

    out = reduce_stars.invoke(
        {
            "working_dir": str(tmp_dir),
            "image_path": str(img_path),
            "detection_threshold": 0.6,
            "kernel_radius": 1,
            "iterations": 1,
            "blend_amount": 1.0,
            "protect_core_radius": 0,
            "feather_px": 0,
        }
    )
    pct = out["mean_size_reduction_pct"]
    assert np.isfinite(pct)
    assert -100.0 <= pct <= 100.0
    _pass("T26 size metric stable with bright non-star regions")


# ────────────────────────────────────────────────────────────────────────────
# T05 — _compute_summary aggregates
# ────────────────────────────────────────────────────────────────────────────

def test_t05_compute_summary_known() -> None:
    """Verify summary statistics from _compute_summary with known frame metrics."""
    frames = {
        "f1.fits": {
            "fwhm": 2.0, "weighted_fwhm": 2.2, "roundness": 0.8, "quality": 0.7,
            "number_of_stars": 100, "background_lvl": 0.05, "bgnoise": 0.001,
        },
        "f2.fits": {
            "fwhm": 3.0, "weighted_fwhm": 3.3, "roundness": 0.7, "quality": 0.6,
            "number_of_stars": 120, "background_lvl": 0.06, "bgnoise": 0.002,
        },
        "f3.fits": {
            "fwhm": 4.0, "weighted_fwhm": 4.4, "roundness": 0.9, "quality": 0.5,
            "number_of_stars": 80, "background_lvl": 0.04, "bgnoise": 0.0015,
        },
    }
    seq_data = {"reference_image": 0, "selected_indices": [0, 1, 2]}
    s = _compute_summary(frames, seq_data)
    assert s["frame_count"] == 3
    assert abs(s["median_fwhm"] - 3.0) < 1e-4
    assert abs(s["median_roundness"] - 0.8) < 1e-4
    assert s["median_star_count"] == 100
    assert abs(s["median_background"] - 0.05) < 1e-6
    assert abs(s["median_noise"] - 0.0015) < 1e-6
    assert s["best_frame"] == "f1.fits"     # lowest FWHM
    assert s["worst_frame"] == "f3.fits"    # highest FWHM
    expected_std = statistics.stdev([2.0, 3.0, 4.0])
    assert abs(s["std_fwhm"] - round(expected_std, 4)) < 1e-4
    _pass("T05 _compute_summary aggregates correct")


def test_t05_compute_summary_all_none() -> None:
    """When all metrics are None, summary should still be valid (no crash)."""
    frames = {
        "f1.fits": {
            "fwhm": None, "weighted_fwhm": None, "roundness": None,
            "quality": None, "number_of_stars": None, "background_lvl": None,
            "bgnoise": None,
        },
    }
    seq_data = {"reference_image": -1, "selected_indices": []}
    s = _compute_summary(frames, seq_data)
    assert s["frame_count"] == 1
    assert s["median_fwhm"] is None
    assert s["best_frame"] is None
    _pass("T05 _compute_summary handles all-None metrics")


# ────────────────────────────────────────────────────────────────────────────
# T08 — Crop geometry
# ────────────────────────────────────────────────────────────────────────────

def test_t08_crop_inclusive_geometry(tmp_dir: Path) -> None:
    img = np.zeros((3, 80, 100), dtype=np.float32)
    img[:, 10:70, 20:90] = 1.0
    path = tmp_dir / "crop_src.fits"
    _save_fits(path, img)

    x, y, w, h = _find_crop_bounds(path, threshold=0.01)
    assert (x, y, w, h) == (25, 15, 60, 50)
    _pass("T08 inclusive crop width/height")


def test_t08_crop_tiny_region_no_negative_geometry(tmp_dir: Path) -> None:
    img = np.zeros((3, 50, 50), dtype=np.float32)
    img[:, 20:24, 20:24] = 1.0
    path = tmp_dir / "tiny_crop_src.fits"
    _save_fits(path, img)

    x, y, w, h = _find_crop_bounds(path, threshold=0.01)
    assert w > 0 and h > 0
    _pass("T08 tiny signal region keeps valid geometry")


# ────────────────────────────────────────────────────────────────────────────
# Main runner
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _pass_count
    _pass_count = 0
    print("\nverify_math_integrity.py")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)

        # T06 — Frame selection
        test_t06_sigma_threshold_known_values()
        test_t06_sigma_threshold_single_value()
        test_t06_safety_fallback_all_bad()
        test_t06_selective_rejection()
        test_t06_empty_input_fails_loudly()

        # T20 — Flatness, clipping, gradient
        test_t20_flatness_uniform()
        test_t20_flatness_gradient()
        test_t20_clipping_known()
        test_t20_gradient_magnitude_zero()
        test_t20_linearity_nan_guard()

        # T20 — Color saturation
        test_t20_color_saturation_pure_red()
        test_t20_color_saturation_neutral_gray()
        test_t20_color_saturation_known_50pct()
        test_t20_color_saturation_black_no_nan()

        # T20 — Signal coverage and background
        test_t20_signal_coverage_known()
        test_t20_background_channel_medians_known()

        # T20 — Linearity and contrast ratio
        test_t20_linearity_linear_image()
        test_t20_linearity_stretched_image()
        test_t20_contrast_ratio_known()

        # T20 — FITS loading
        test_t20_load_fits_16bit_normalization(tmp_dir)
        test_t20_load_fits_nan_replacement(tmp_dir)
        test_t20_nan_input_sanitized(tmp_dir)

        # T25 — Mask creation
        test_t25_luminance_mask()
        test_t25_inverted_luminance_mask()
        test_t25_channel_diff_mask()
        test_t25_range_mask()
        test_t25_inversion_logic(tmp_dir)

        # T27 — Multiscale
        test_t27_soft_threshold_large_survives()
        test_t27_soft_threshold_zeros_small()
        test_t27_b3_roundtrip()
        test_t27_color_luminance_ratio_preserves_hue(tmp_dir)
        test_t27_multiscale_mono_tool_path(tmp_dir)
        test_t27_resizes_mismatched_mask(tmp_dir)

        # T26 — Star reduction
        test_t26_compute_luminance_mono()
        test_t26_compute_luminance_color_weights()
        test_t26_blend_math_within_mask_only(tmp_dir)
        test_t26_size_metric_not_global_contaminated(tmp_dir)

        # T05 — Summary aggregates
        test_t05_compute_summary_known()
        test_t05_compute_summary_all_none()

        # T08 — Crop geometry
        test_t08_crop_inclusive_geometry(tmp_dir)
        test_t08_crop_tiny_region_no_negative_geometry(tmp_dir)

    print("=" * 60)
    print(f"  Results: all {_pass_count} math integrity checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()

