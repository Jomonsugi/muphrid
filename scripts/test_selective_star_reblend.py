#!/usr/bin/env python3
"""
Smoke tests for the expert star-treatment tools:

  - t26 reduce_stars regression: star_mask_path field/param wired.
  - t40 analyze_star_population: detection + ranking on synthetic stars.
  - t41 selective_star_reblend: tiered keep/suppress; region zoning.
  - t42 enhance_star_color: HSV saturation boost on the star contribution.

Run from project root:
    uv run python scripts/test_selective_star_reblend.py

Exit 0 = all checks pass. Tests synthesize tiny RGB FITS files in a temp
working_dir; no real astronomy data required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
from astropy.io import fits as astropy_fits
from skimage.color import rgb2hsv

# Import the graph package first to trigger full registry init in the
# correct order — the tool modules below all import from muphrid.graph.state,
# which transitively loads registry, which imports the tool modules. Direct
# tool imports without this priming hit a circular ImportError.
import muphrid.graph  # noqa: F401

from muphrid.tools.utility.t40_analyze_star_population import (
    analyze_star_population,
)
from muphrid.tools.scikit.t26_reduce_stars import reduce_stars
from muphrid.tools.scikit.t41_selective_star_reblend import (
    selective_star_reblend,
)
from muphrid.tools.scikit.t42_enhance_star_color import enhance_star_color


_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    msg = f"  {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


# ── Synthetic fixtures ─────────────────────────────────────────────────────────


def _gauss(h: int, w: int, cy: int, cx: int, sigma: float, peak: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def make_two_peak_fixture(workdir: Path) -> dict:
    """
    Two stars on a flat dark sky, in a 256×256 RGB image.

      Peak A at (60,60):  bright, very red (high R, near-zero G/B)
      Peak B at (180,180): bright, neutral (white — equal R/G/B)

    Saved as starless (flat dark sky), star_mask (just the peaks), and an
    "original" reconstruction (starless + mask) for completeness.
    """
    h, w = 256, 256
    sigma = 3.0
    peak_brightness = 0.9
    # Real astrophotography images always have read noise / shot noise;
    # MAD-based detection short-circuits on a noiseless background.
    rng = np.random.default_rng(seed=42)
    starless = (
        np.full((3, h, w), 0.02, dtype=np.float32)
        + rng.normal(scale=0.005, size=(3, h, w)).astype(np.float32)
    )

    # Peak A — saturated red.
    pa_r = _gauss(h, w, 60, 60, sigma, peak_brightness)
    pa_g = _gauss(h, w, 60, 60, sigma, 0.05 * peak_brightness)
    pa_b = _gauss(h, w, 60, 60, sigma, 0.05 * peak_brightness)
    # Peak B — neutral white (slightly higher peak so brightness_priority
    # can prefer it).
    pb_r = _gauss(h, w, 180, 180, sigma, peak_brightness * 1.05)
    pb_g = _gauss(h, w, 180, 180, sigma, peak_brightness * 1.05)
    pb_b = _gauss(h, w, 180, 180, sigma, peak_brightness * 1.05)

    mask = np.stack([pa_r + pb_r, pa_g + pb_g, pa_b + pb_b], axis=0).astype(np.float32)
    # StarNet masks are not pixel-perfectly zero outside stars; MAD-based
    # detection requires a non-zero noise floor.
    mask = mask + rng.normal(scale=0.002, size=mask.shape).astype(np.float32)
    mask = np.clip(mask, 0.0, None)
    original = (starless + mask).astype(np.float32)

    starless_p = workdir / "fixture_starless.fits"
    mask_p     = workdir / "fixture_starmask.fits"
    original_p = workdir / "fixture_original.fits"
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(data=starless)]).writeto(starless_p, overwrite=True)
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(data=mask)]).writeto(mask_p, overwrite=True)
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(data=original)]).writeto(original_p, overwrite=True)
    return {
        "starless": str(starless_p),
        "star_mask": str(mask_p),
        "original": str(original_p),
        "peak_a": (60, 60),
        "peak_b": (180, 180),
        "shape": (h, w),
    }


def make_clean_star_mask_fixture(workdir: Path) -> dict:
    """
    Same two-star geometry as make_two_peak_fixture, but the star mask has an
    exactly zero background. This models a clean component/mask image rather
    than real sky luminance, and should exercise t41's mask-native fallback.
    """
    h, w = 256, 256
    sigma = 3.0
    starless = np.full((3, h, w), 0.02, dtype=np.float32)

    pa_r = _gauss(h, w, 60, 60, sigma, 0.9)
    pa_g = _gauss(h, w, 60, 60, sigma, 0.05 * 0.9)
    pa_b = _gauss(h, w, 60, 60, sigma, 0.05 * 0.9)
    pb_r = _gauss(h, w, 180, 180, sigma, 0.9)
    pb_g = _gauss(h, w, 180, 180, sigma, 0.9)
    pb_b = _gauss(h, w, 180, 180, sigma, 0.9)

    mask = np.stack([pa_r + pb_r, pa_g + pb_g, pa_b + pb_b], axis=0).astype(np.float32)
    original = (starless + mask).astype(np.float32)

    starless_p = workdir / "clean_starless.fits"
    mask_p = workdir / "clean_starmask.fits"
    original_p = workdir / "clean_original.fits"
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(data=starless)]).writeto(starless_p, overwrite=True)
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(data=mask)]).writeto(mask_p, overwrite=True)
    astropy_fits.HDUList([astropy_fits.PrimaryHDU(data=original)]).writeto(original_p, overwrite=True)
    return {
        "starless": str(starless_p),
        "star_mask": str(mask_p),
        "original": str(original_p),
        "peak_a": (60, 60),
        "peak_b": (180, 180),
        "shape": (h, w),
    }


def make_state(workdir: Path, fx: dict, latest_mask: str | None = None) -> dict:
    return {
        "dataset": {"working_dir": str(workdir)},
        "paths": {
            "current_image": fx["original"],
            "previous_image": None,
            "starless_image": fx["starless"],
            "star_mask": fx["star_mask"],
            "latest_mask": latest_mask,
        },
        "metadata": {"image_space": "display"},
    }


def _read_fits(path: str) -> np.ndarray:
    with astropy_fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32)


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_t26_star_mask_path_wired():
    """reduce_stars with star_mask_path must not raise NameError (t26 fix)."""
    workdir = Path(tempfile.mkdtemp(prefix="t26_fix_"))
    fx = make_two_peak_fixture(workdir)
    state = make_state(workdir, fx)

    # Pass an explicit star_mask_path (the fix) and a small kernel.
    try:
        result = reduce_stars.func(
            detection_threshold=0.5,
            kernel_radius=1,
            structuring_element="disk",
            erosion_mode="reflect",
            iterations=1,
            blend_amount=1.0,
            protect_core_radius=0,
            peak_threshold_abs=None,
            peak_num_peaks=None,
            peak_exclude_border=True,
            label_connectivity=2,
            feather_px=2,
            output_stem="t26_test",
            star_mask_path=fx["star_mask"],
            tool_call_id="t26-tcid",
            state=state,
        )
        ran = True
        update = getattr(result, "update", None) or {}
        out_paths = update.get("paths", {})
        wrote_image = "current_image" in out_paths
    except NameError as e:
        ran = False
        wrote_image = False
        check("t26_star_mask_path_wired", False, f"NameError: {e}")
        return
    except Exception as e:
        ran = False
        wrote_image = False
        check("t26_star_mask_path_wired", False, f"unexpected: {type(e).__name__}: {e}")
        return

    check("t26_star_mask_path_wired", ran and wrote_image)


def test_analyze_star_population_finds_both_peaks():
    workdir = Path(tempfile.mkdtemp(prefix="t40_"))
    fx = make_two_peak_fixture(workdir)
    state = make_state(workdir, fx)

    result = analyze_star_population.func(
        threshold_sigma=3.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        chroma_sample_mode="annulus",
        score_mode="brightness_priority",
        return_table_rows=10,
        tool_call_id="t40-tcid",
        state=state,
    )
    msg_payload = json.loads(result.update["messages"][0].content)
    detected = msg_payload.get("count", 0)
    check(
        "t40_detects_both_peaks",
        detected >= 2,
        f"detected count={detected}",
    )
    check(
        "t40_writes_sidecar_catalog",
        Path(msg_payload.get("source_catalog_path", "")).exists(),
        str(msg_payload.get("source_catalog_path")),
    )

    # color_priority should rank peak A (red) above peak B (white) on chroma.
    result_color = analyze_star_population.func(
        threshold_sigma=3.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        chroma_sample_mode="annulus",
        score_mode="color_priority",
        return_table_rows=10,
        tool_call_id="t40-color",
        state=state,
    )
    payload_c = json.loads(result_color.update["messages"][0].content)
    top = payload_c["top_sources_by_score"][:2]
    if len(top) >= 2:
        # Distance to peak A == (60,60).
        d_to_a = lambda r: (r["x"] - 60) ** 2 + (r["y"] - 60) ** 2
        ranked_first_is_a = d_to_a(top[0]) < d_to_a(top[1])
        check(
            "t40_color_priority_ranks_red_first",
            ranked_first_is_a,
            f"top 2 = {[(round(t['x']),round(t['y']),round(t['chroma'],2)) for t in top]}",
        )
    else:
        check("t40_color_priority_ranks_red_first", False, "fewer than 2 sources")


def test_selective_reblend_clean_mask_fallback_and_edges():
    workdir = Path(tempfile.mkdtemp(prefix="t41_clean_"))
    fx = make_clean_star_mask_fixture(workdir)
    state = make_state(workdir, fx)

    result_full = selective_star_reblend.func(
        mode="brightness_priority",
        keep_fraction=1.0,
        suppress_strength=0.0,
        core_radius_factor=1.5,
        feather_sigma_px=0.0,
        mask_dilation_px=0,
        confine_to_region_mask=False,
        threshold_sigma=5.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        output_stem="t41_clean_full",
        tool_call_id="t41-clean-full",
        state=state,
    )
    full = _read_fits(result_full.update["paths"]["current_image"])
    original = _read_fits(fx["original"])
    check(
        "t41_clean_mask_keep_all_full_restore",
        np.allclose(full, np.clip(original, 0.0, 1.0), atol=1e-5),
    )

    result_none = selective_star_reblend.func(
        mode="brightness_priority",
        keep_fraction=0.0,
        suppress_strength=0.0,
        core_radius_factor=1.5,
        feather_sigma_px=0.0,
        mask_dilation_px=0,
        confine_to_region_mask=False,
        threshold_sigma=5.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        output_stem="t41_clean_none",
        tool_call_id="t41-clean-none",
        state=state,
    )
    none = _read_fits(result_none.update["paths"]["current_image"])
    starless = _read_fits(fx["starless"])
    check(
        "t41_clean_mask_keep_none_suppresses_all",
        np.allclose(none, starless, atol=1e-5),
    )


def test_selective_reblend_color_priority_keeps_red():
    workdir = Path(tempfile.mkdtemp(prefix="t41_color_"))
    fx = make_two_peak_fixture(workdir)
    state = make_state(workdir, fx)

    result = selective_star_reblend.func(
        mode="color_priority",
        keep_fraction=0.5,         # keep 1 of 2
        suppress_strength=0.0,     # remove the other entirely
        core_radius_factor=2.0,
        feather_sigma_px=1.0,
        mask_dilation_px=0,
        confine_to_region_mask=False,
        threshold_sigma=15.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        output_stem="t41_color",
        tool_call_id="t41-color",
        state=state,
    )
    out = result.update["paths"]["current_image"]
    img = _read_fits(out)

    # Sample at peak A (red, kept) vs peak B (white, suppressed).
    val_a = float(img[0, 60, 60])  # red channel near peak A
    val_b = float(img[0, 180, 180])  # red channel near peak B
    # Peak A should retain most of its bright red contribution; peak B
    # should be suppressed close to the starless background (~0.02).
    check(
        "t41_color_priority_kept_red",
        val_a > 0.5 and val_b < 0.2,
        f"R@A={val_a:.3f} R@B={val_b:.3f}",
    )

    # image_space delta passed through.
    md = result.update.get("metadata", {})
    check("t41_emits_image_space", md.get("image_space") == "display")


def test_selective_reblend_brightness_priority_keeps_brighter():
    workdir = Path(tempfile.mkdtemp(prefix="t41_bright_"))
    fx = make_two_peak_fixture(workdir)
    state = make_state(workdir, fx)

    # Peak B is slightly brighter (1.05x) — brightness_priority should
    # prefer it over the colorful peak A.
    result = selective_star_reblend.func(
        mode="brightness_priority",
        keep_fraction=0.5,
        suppress_strength=0.0,
        core_radius_factor=2.0,
        feather_sigma_px=1.0,
        mask_dilation_px=0,
        confine_to_region_mask=False,
        threshold_sigma=15.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        output_stem="t41_bright",
        tool_call_id="t41-bright",
        state=state,
    )
    out = result.update["paths"]["current_image"]
    img = _read_fits(out)

    # Peak B kept (high all channels); peak A's red should be suppressed
    # (still present from starless ~0.02 only).
    val_b_g = float(img[1, 180, 180])  # green at B (B is white)
    val_a_r = float(img[0, 60, 60])
    check(
        "t41_brightness_priority_kept_white",
        val_b_g > 0.5 and val_a_r < 0.2,
        f"G@B={val_b_g:.3f} R@A={val_a_r:.3f}",
    )


def test_selective_reblend_confine_to_region():
    """region_mask covering only the right half: left half stars suppressed,
    right half restored to W=1.0 regardless of keep_fraction/suppress_strength."""
    workdir = Path(tempfile.mkdtemp(prefix="t41_region_"))
    fx = make_two_peak_fixture(workdir)
    h, w = fx["shape"]

    # Build a region mask covering only the right half of the image.
    region = np.zeros((h, w), dtype=np.float32)
    region[:, w // 2:] = 1.0
    region_p = workdir / "region.fits"
    astropy_fits.HDUList(
        [astropy_fits.PrimaryHDU(data=region)]
    ).writeto(region_p, overwrite=True)

    state = make_state(workdir, fx, latest_mask=str(region_p))
    # Peak A is at (60, 60) — left half (outside region) → restore W=1.
    # Peak B is at (180, 180) — right half (inside region) → suppress=0.
    result = selective_star_reblend.func(
        mode="brightness_priority",
        keep_fraction=0.0,         # nothing in keep set inside region
        suppress_strength=0.0,     # inside region: full removal
        core_radius_factor=2.0,
        feather_sigma_px=1.0,
        mask_dilation_px=0,
        confine_to_region_mask=True,
        threshold_sigma=15.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        output_stem="t41_region",
        tool_call_id="t41-region",
        state=state,
    )
    out = result.update["paths"]["current_image"]
    img = _read_fits(out)

    val_a_r = float(img[0, 60, 60])    # outside region: full restore
    val_b_g = float(img[1, 180, 180])  # inside region: suppressed
    check(
        "t41_confine_to_region_outside_full",
        val_a_r > 0.5,
        f"R@A={val_a_r:.3f} (expected > 0.5)",
    )
    check(
        "t41_confine_to_region_inside_suppressed",
        val_b_g < 0.2,
        f"G@B={val_b_g:.3f} (expected < 0.2)",
    )


def test_selective_reblend_composes_color_boost():
    workdir = Path(tempfile.mkdtemp(prefix="t41_color_boost_"))
    fx = make_two_peak_fixture(workdir)
    state = make_state(workdir, fx)

    result = selective_star_reblend.func(
        mode="brightness_priority",
        keep_fraction=1.0,
        suppress_strength=0.0,
        core_radius_factor=2.0,
        feather_sigma_px=0.0,
        mask_dilation_px=0,
        star_saturation_multiplier=2.0,
        confine_to_region_mask=False,
        threshold_sigma=3.0,
        fwhm_guess=3.0,
        min_separation_fwhm=2.0,
        max_sources=100,
        output_stem="t41_color_boost",
        tool_call_id="t41-color-boost",
        state=state,
    )
    boosted = _read_fits(result.update["paths"]["current_image"])
    original = _read_fits(fx["original"])
    o_hsv_at_a = rgb2hsv(np.moveaxis(original[:, 55:65, 55:65], 0, -1))
    b_hsv_at_a = rgb2hsv(np.moveaxis(boosted[:, 55:65, 55:65], 0, -1))
    check(
        "t41_color_boost_composes_with_reblend",
        float(np.mean(b_hsv_at_a[..., 1])) > float(np.mean(o_hsv_at_a[..., 1])),
    )


def test_enhance_star_color_increases_saturation():
    workdir = Path(tempfile.mkdtemp(prefix="t42_"))
    fx = make_two_peak_fixture(workdir)
    state = make_state(workdir, fx)

    result = enhance_star_color.func(
        saturation_multiplier=2.0,
        confine_to_region_mask=False,
        output_stem="t42_test",
        tool_call_id="t42-tcid",
        state=state,
    )
    out = result.update["paths"]["current_image"]
    boosted = _read_fits(out)

    # Compare HSV saturation at the originally-saturated peak A (red star)
    # vs the input (starless + mask = original). Peak A should retain or
    # amplify saturation; peak B (already neutral) should remain low.
    original = _read_fits(fx["original"])
    o_hsv_at_a = rgb2hsv(np.moveaxis(original[:, 55:65, 55:65], 0, -1))
    b_hsv_at_a = rgb2hsv(np.moveaxis(boosted[:, 55:65, 55:65],  0, -1))
    o_sat = float(np.mean(o_hsv_at_a[..., 1]))
    b_sat = float(np.mean(b_hsv_at_a[..., 1]))
    check(
        "t42_saturation_boost_at_red_peak",
        b_sat > o_sat,
        f"saturation: original={o_sat:.3f} → boosted={b_sat:.3f}",
    )

    md = result.update.get("metadata", {})
    check("t42_emits_image_space", md.get("image_space") == "display")


def test_selective_reblend_refuses_missing_starless():
    """When paths.starless_image is missing, selective_star_reblend must
    raise a clear error pointing the agent at star_removal — no fallback."""
    workdir = Path(tempfile.mkdtemp(prefix="t41_missing_"))
    fx = make_two_peak_fixture(workdir)
    state = make_state(workdir, fx)
    state["paths"]["starless_image"] = None

    raised = False
    detail = ""
    try:
        selective_star_reblend.func(
            mode="brightness_priority",
            keep_fraction=0.1,
            suppress_strength=0.3,
            core_radius_factor=1.5,
            feather_sigma_px=2.0,
            mask_dilation_px=0,
            confine_to_region_mask=False,
            threshold_sigma=3.0,
            fwhm_guess=3.0,
            min_separation_fwhm=2.0,
            max_sources=100,
            output_stem=None,
            tool_call_id="t41-missing",
            state=state,
        )
    except FileNotFoundError as e:
        raised = True
        detail = str(e)
    except Exception as e:
        detail = f"wrong type: {type(e).__name__}: {e}"

    check(
        "t41_refuses_missing_starless",
        raised and "star_removal" in detail.lower(),
        detail[:120],
    )


def test_star_tools_are_hitl_mapped():
    from muphrid.graph.hitl import TOOL_TO_HITL

    check(
        "t41_hitl_mapped",
        TOOL_TO_HITL.get("selective_star_reblend") == "T41_selective_star_reblend",
    )
    check(
        "t42_hitl_mapped",
        TOOL_TO_HITL.get("enhance_star_color") == "T42_enhance_star_color",
    )


# ── Runner ─────────────────────────────────────────────────────────────────────


def main() -> int:
    print("== expert star treatment tests ==")
    test_t26_star_mask_path_wired()
    test_analyze_star_population_finds_both_peaks()
    test_selective_reblend_clean_mask_fallback_and_edges()
    test_selective_reblend_color_priority_keeps_red()
    test_selective_reblend_brightness_priority_keeps_brighter()
    test_selective_reblend_confine_to_region()
    test_selective_reblend_composes_color_boost()
    test_enhance_star_color_increases_saturation()
    test_selective_reblend_refuses_missing_starless()
    test_star_tools_are_hitl_mapped()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)}")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
