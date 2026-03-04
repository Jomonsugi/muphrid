"""
Math integrity regression checks for AstroAgent tools.

Purpose:
- Catch numerical/logic bugs in pure-Python math paths early.
- Validate edge cases that are easy to miss in end-to-end runs.

This script is intentionally fast and synthetic-data based.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from astro_agent.tools.preprocess.t06_select_frames import select_frames
from astro_agent.tools.preprocess.t08_crop import _find_crop_bounds
from astro_agent.tools.scikit.t26_reduce_stars import _compute_luminance, reduce_stars
from astro_agent.tools.scikit.t27_multiscale import b3_atrous_decompose, b3_atrous_reconstruct, multiscale_process
from astro_agent.tools.utility.t20_analyze import _linearity_analysis


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _save_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data.astype(np.float32)).writeto(path, overwrite=True)


def test_t26_compute_luminance_mono() -> None:
    mono = np.random.rand(1, 32, 32).astype(np.float32)
    lum = _compute_luminance(mono)
    assert lum.shape == (32, 32)
    assert np.allclose(lum, mono[0], atol=1e-7)
    _pass("T26 mono luminance path")


def test_t27_b3_roundtrip() -> None:
    img = np.random.rand(64, 64).astype(np.float32)
    layers = b3_atrous_decompose(img, 4)
    recon = b3_atrous_reconstruct(layers)
    assert np.allclose(img, recon, atol=1e-5)
    _pass("T27 B3 a trous roundtrip")


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


def test_t20_linearity_nan_guard() -> None:
    const = np.ones((128, 128), dtype=np.float32) * 0.1
    res = _linearity_analysis(const, overall_median=float(np.median(const)))
    assert np.isfinite(res["histogram_skewness"])
    _pass("T20 skew NaN guard")


def test_t08_crop_inclusive_geometry(tmp_dir: Path) -> None:
    img = np.zeros((3, 80, 100), dtype=np.float32)
    img[:, 10:70, 20:90] = 1.0
    path = tmp_dir / "crop_src.fits"
    _save_fits(path, img)

    x, y, w, h = _find_crop_bounds(path, threshold=0.01)
    # Expected bounds with 5px inset:
    # rows 10..69 -> y=15, y2=64 => h=50
    # cols 20..89 -> x=25, x2=84 => w=60
    assert (x, y, w, h) == (25, 15, 60, 50)
    _pass("T08 inclusive crop width/height")


def test_t08_crop_tiny_region_no_negative_geometry(tmp_dir: Path) -> None:
    img = np.zeros((3, 50, 50), dtype=np.float32)
    img[:, 20:24, 20:24] = 1.0  # tiny signal island
    path = tmp_dir / "tiny_crop_src.fits"
    _save_fits(path, img)

    x, y, w, h = _find_crop_bounds(path, threshold=0.01)
    assert w > 0 and h > 0
    _pass("T08 tiny signal region keeps valid geometry")


def test_t26_size_metric_not_global_contaminated(tmp_dir: Path) -> None:
    # Build a toy image with one bright star + bright nebulosity patch elsewhere.
    data = np.zeros((3, 64, 64), dtype=np.float32)
    data[:, 32, 20] = 1.0  # star
    data[:, 32, 21] = 0.8
    data[:, 10:20, 45:55] = 0.9  # unrelated bright patch

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
    # Metric should remain finite and bounded.
    pct = out["mean_size_reduction_pct"]
    assert np.isfinite(pct)
    assert -100.0 <= pct <= 100.0
    _pass("T26 size metric stable with bright non-star regions")


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


def test_t06_empty_input_fails_loudly() -> None:
    try:
        select_frames.invoke({"frame_metrics": {}, "criteria": {}})
    except ValueError as exc:
        assert "frame_metrics is empty" in str(exc)
        _pass("T06 empty frame_metrics fails loudly")
        return
    raise AssertionError("T06 should fail on empty frame_metrics")


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


def main() -> None:
    print("\nverify_math_integrity.py")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        test_t26_compute_luminance_mono()
        test_t27_b3_roundtrip()
        test_t27_multiscale_mono_tool_path(tmp_dir)
        test_t20_linearity_nan_guard()
        test_t08_crop_inclusive_geometry(tmp_dir)
        test_t08_crop_tiny_region_no_negative_geometry(tmp_dir)
        test_t26_size_metric_not_global_contaminated(tmp_dir)
        test_t27_resizes_mismatched_mask(tmp_dir)
        test_t06_empty_input_fails_loudly()
        test_t20_nan_input_sanitized(tmp_dir)
    print("=" * 60)
    print("  Results: all math integrity checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()

