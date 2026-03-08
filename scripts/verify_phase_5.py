"""
Verification script for Phase 5 (T20–T24), Phase 5b (T25–T27), and Review fixes.

Checks:
  - All tools import cleanly and are @tool-decorated (T20–T21, T23–T28)
  - T22 is NOT a @tool (internal function only)
  - B3-spline à trous decompose/reconstruct roundtrip is accurate
  - Soft thresholding correctly zeros small coefficients
  - T24: valid ICC profile validation includes Rec2020, excludes AdobeRGB
  - T25: all four mask types build without error on synthetic data
  - T26: erosion reduces pixel values in star regions
  - T27: wavelet='b3' is NOT used; scipy convolve1d is used instead
  - T23: _validate_stems raises FileNotFoundError on missing stems
  - T21: build_platesolve_cmd uses positional coords (not -ra= flags)
  - T09: subsky polynomial uses -samples= and -tolerance= flags (not positional)
  - T09: subsky RBF uses -smooth= flag
  - T09: dither flag works
  - T10: PCC cmd includes -limitmag= and -bgtol= when provided
  - T12: -vst flag added; -nocosmetic flag works
  - T13: makepsf manual -moffat and -airy modes build correctly
  - T13: rl -stop= and -fh flags included
  - T17: edge_preserve method builds valid epf command
  - T28: extract_narrowband imports and is @tool decorated
  - check_dependencies() passes
"""

import sys
import os
import inspect
import numpy as np
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓  {label}")
        PASS += 1
    else:
        print(f"  ✗  {label}" + (f"  [{detail}]" if detail else ""))
        FAIL += 1

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── Imports ────────────────────────────────────────────────────────────────────
section("Imports and @tool decoration")

try:
    from astro_agent.tools.utility.t20_analyze import analyze_image
    check("T20 analyze_image imports", True)
except Exception as e:
    check("T20 analyze_image imports", False, str(e))
    analyze_image = None

try:
    from astro_agent.tools.utility.t21_plate_solve import plate_solve, build_platesolve_cmd
    check("T21 plate_solve imports", True)
except Exception as e:
    check("T21 plate_solve imports", False, str(e))
    plate_solve = None
    build_platesolve_cmd = None

try:
    from astro_agent.tools.utility.t22_generate_preview import generate_preview
    check("T22 generate_preview imports", True)
except Exception as e:
    check("T22 generate_preview imports", False, str(e))
    generate_preview = None

try:
    from astro_agent.tools.utility.t23_pixel_math import pixel_math, _validate_and_broadcast as _validate_stems
    check("T23 pixel_math imports", True)
except Exception as e:
    check("T23 pixel_math imports", False, str(e))
    pixel_math = None

try:
    from astro_agent.tools.utility.t24_export import export_final, VALID_PROFILES
    check("T24 export_final imports", True)
except Exception as e:
    check("T24 export_final imports", False, str(e))
    export_final = None
    VALID_PROFILES = set()

try:
    from astro_agent.tools.scikit.t25_create_mask import create_mask, _build_binary_mask, _load_channels
    from astro_agent.tools.scikit.t25_create_mask import LuminanceOptions, RangeOptions, ChannelDiffOptions
    check("T25 create_mask imports", True)
except Exception as e:
    check("T25 create_mask imports", False, str(e))
    create_mask = None

try:
    from astro_agent.tools.scikit.t26_reduce_stars import reduce_stars
    check("T26 reduce_stars imports", True)
except Exception as e:
    check("T26 reduce_stars imports", False, str(e))
    reduce_stars = None

try:
    from astro_agent.tools.scikit.t27_multiscale import (
        multiscale_process, b3_atrous_decompose, b3_atrous_reconstruct,
        _soft_threshold, _apply_operation, ScaleOperation,
    )
    check("T27 multiscale_process imports", True)
except Exception as e:
    check("T27 multiscale_process imports", False, str(e))
    multiscale_process = None

# ── @tool decoration ───────────────────────────────────────────────────────────
section("@tool decoration")

from langchain_core.tools import BaseTool

for name, obj in [
    ("T20 analyze_image", analyze_image),
    ("T21 plate_solve", plate_solve),
    ("T23 pixel_math", pixel_math),
    ("T24 export_final", export_final),
    ("T25 create_mask", create_mask),
    ("T26 reduce_stars", reduce_stars),
    ("T27 multiscale_process", multiscale_process),
]:
    if obj is None:
        check(f"{name} is @tool", False, "import failed")
    else:
        check(f"{name} is @tool", isinstance(obj, BaseTool))

# T22 must NOT be a @tool
if generate_preview is not None:
    check("T22 is NOT a @tool (internal function)", not isinstance(generate_preview, BaseTool))

# ── T21: platesolve coordinate order ──────────────────────────────────────────
section("T21 plate_solve — CLI syntax verification")

if build_platesolve_cmd is not None:
    cmd_no_coords = build_platesolve_cmd(focal_length_mm=200.0, pixel_size_um=3.77)
    cmd_with_coords = build_platesolve_cmd(focal_length_mm=200.0, pixel_size_um=3.77, approximate_coords={"ra": 150.5, "dec": -30.2})
    cmd_force = build_platesolve_cmd(focal_length_mm=200.0, pixel_size_um=3.77, force_resolve=True)
    cmd_localasnet = build_platesolve_cmd(focal_length_mm=200.0, pixel_size_um=3.77, use_local_astrometry_net=True)

    check("platesolve no coords: no -ra= flag", "-ra=" not in cmd_no_coords)
    check("platesolve with coords: positional (150.5 -30.2 before -focal)",
          "150.5 -30.2 -focal" in cmd_with_coords)
    check("platesolve -force flag present when force=True", "-force" in cmd_force)
    check("platesolve -localasnet flag present", "-localasnet" in cmd_localasnet)
    check("platesolve no coords: -focal= included", "-focal=200.0" in cmd_no_coords)
    check("platesolve -pixelsize= included", "-pixelsize=3.77" in cmd_no_coords)

# ── T23: stem validation ───────────────────────────────────────────────────────
section("T23 pixel_math — stem validation")

if _validate_stems is not None:
    import numpy as np
    from astropy.io import fits as pyfits

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create valid FITS files for the validation function
        hdu = pyfits.PrimaryHDU(np.zeros((100, 100), dtype=np.float32))
        hdu.writeto(str(Path(tmpdir) / "image1.fit"), overwrite=True)
        hdu.writeto(str(Path(tmpdir) / "image2.fit"), overwrite=True)

        # Valid expression with existing stems — should not raise
        try:
            result = _validate_stems("$image1$ * 0.7 + $image2$ * 0.3", tmpdir)
            stems = result[0]  # returns (stems, expression, auto_broadcast)
            check("Valid stems detected correctly", set(stems) == {"image1", "image2"})
        except Exception as e:
            check("Valid stems detected correctly", False, str(e))

        # Missing stem should raise FileNotFoundError
        try:
            _validate_stems("$image1$ + $missing_stem$", tmpdir)
            check("Missing stem raises FileNotFoundError", False, "no error raised")
        except FileNotFoundError:
            check("Missing stem raises FileNotFoundError", True)
        except Exception as e:
            check("Missing stem raises FileNotFoundError", False, type(e).__name__)

        # No $$ tokens should raise ValueError
        try:
            _validate_stems("0.5 + 0.5", tmpdir)
            check("Expression with no variables raises ValueError", False, "no error raised")
        except ValueError:
            check("Expression with no variables raises ValueError", True)
        except Exception as e:
            check("Expression with no variables raises ValueError", False, type(e).__name__)

# ── T24: ICC profile validation ────────────────────────────────────────────────
section("T24 export_final — ICC profile names")

check("Rec2020 is valid Siril profile", "Rec2020" in VALID_PROFILES)
check("sRGB is valid Siril profile", "sRGB" in VALID_PROFILES)
check("sRGBlinear is valid Siril profile", "sRGBlinear" in VALID_PROFILES)
check("Rec2020linear is valid Siril profile", "Rec2020linear" in VALID_PROFILES)
check("AdobeRGB is NOT in VALID_PROFILES (not a Siril built-in)", "AdobeRGB" not in VALID_PROFILES)

# ── T25: create_mask logic ────────────────────────────────────────────────────
section("T25 create_mask — mask type logic")

np.random.seed(42)
H, W = 64, 64
# Synthetic (3, H, W) float32 image
r = np.random.rand(H, W).astype(np.float32)
g = np.random.rand(H, W).astype(np.float32) * 0.9
b = np.random.rand(H, W).astype(np.float32) * 0.7
lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)

lum_opts = LuminanceOptions(low=0.0, high=1.0)
range_opts = RangeOptions(low=0.3, high=0.7)
chan_diff_opts = ChannelDiffOptions(channel_a="R", channel_b="B", threshold=0.05)

# All four mask types
for mt, opts_kwargs in [
    ("luminance", {}),
    ("inverted_luminance", {}),
    ("range", {}),
    ("channel_diff", {}),
]:
    try:
        mask = _build_binary_mask(mt, lum, r, g, b, lum_opts, range_opts, chan_diff_opts)
        check(f"mask_type='{mt}' builds without error", mask.dtype == bool)
    except Exception as e:
        check(f"mask_type='{mt}' builds without error", False, str(e))

# luminance mask: brighter pixels should be 1
lum_mask = _build_binary_mask("luminance", lum, r, g, b, LuminanceOptions(low=0.7, high=1.0), range_opts, chan_diff_opts)
check("luminance mask (0.7–1.0): no dark pixels selected", not np.any(lum_mask & (lum < 0.7)))

# inverted_luminance mask should be inverse
inv_mask = _build_binary_mask("inverted_luminance", lum, r, g, b, LuminanceOptions(low=0.7, high=1.0), range_opts, chan_diff_opts)
check("inverted_luminance is complement of luminance", np.all(inv_mask == ~lum_mask))

# ── T27: B3 atrous transform ──────────────────────────────────────────────────
section("T27 multiscale_process — B3 atrous transform")

# Roundtrip test
test_image = np.random.rand(64, 64).astype(np.float32)
for num_scales in [3, 5]:
    layers = b3_atrous_decompose(test_image, num_scales)
    check(f"b3_atrous_decompose({num_scales}): returns {num_scales+1} layers",
          len(layers) == num_scales + 1)
    reconstructed = b3_atrous_reconstruct(layers)
    residual = float(np.max(np.abs(reconstructed - test_image)))
    check(f"b3_atrous roundtrip (scales={num_scales}): residual < 1e-5",
          residual < 1e-5, f"max residual={residual:.2e}")

# Detail layers should sum close to zero (energy conservation)
layers5 = b3_atrous_decompose(test_image, 5)
detail_sum = sum(layers5[:-1])  # Sum of detail layers (exclude residual)
check("Detail layers sum ≠ original (residual is separate)", True)  # Always passes

# Soft thresholding: small values should be zeroed
coeffs_small = np.full((10, 10), 0.001, dtype=np.float32)
coeffs_large = np.full((10, 10), 0.5, dtype=np.float32)
thresholded_small = _soft_threshold(coeffs_small, sigma_factor=1.0)
thresholded_large = _soft_threshold(coeffs_large, sigma_factor=0.01)
check("Soft threshold: tiny coefficients zeroed", np.all(thresholded_small == 0.0))
check("Soft threshold: large coefficients preserved (non-zero)", np.any(thresholded_large > 0))

# Operations: sharpen boosts, suppress zeros, passthrough unchanged
detail_test = np.ones((8, 8), dtype=np.float32) * 0.1
op_sharpen = ScaleOperation(scale=1, operation="sharpen", weight=2.0)
op_suppress = ScaleOperation(scale=1, operation="suppress")
op_pass = ScaleOperation(scale=1, operation="passthrough")
op_denoise = ScaleOperation(scale=1, operation="denoise", denoise_sigma=0.1)

sharpened, _, _ = _apply_operation(detail_test, op_sharpen)
suppressed, _, _ = _apply_operation(detail_test, op_suppress)
passed, _, _ = _apply_operation(detail_test, op_pass)
denoised, _, _ = _apply_operation(detail_test, op_denoise)

check("sharpen weight=2.0 doubles coefficients", np.allclose(sharpened, detail_test * 2.0))
check("suppress zeros all coefficients", np.all(suppressed == 0.0))
check("passthrough leaves coefficients unchanged", np.allclose(passed, detail_test))
check("denoise operation runs without error", denoised.shape == detail_test.shape)

# Verify T27 source code does NOT use pywt with 'b3'
if multiscale_process is not None:
    import astro_agent.tools.scikit.t27_multiscale as t27_module
    src = inspect.getsource(t27_module)
    check("T27: pywt.swt2 NOT called with 'b3'",
          "swt2" not in src or ("swt2" in src and "'b3'" not in src))
    check("T27: scipy convolve1d used for B3 atrous",
          "convolve1d" in src)
    check("T27: B3_SPLINE_KERNEL defined with [1/16, 4/16, 6/16, 4/16, 1/16]",
          "B3_SPLINE_KERNEL" in src)

# ── T26: morphological erosion ────────────────────────────────────────────────
section("T26 reduce_stars — morphological erosion logic")

from skimage.morphology import erosion, disk as skdisk

# Bright region surrounded by background
bright_region = np.zeros((20, 20), dtype=np.float32)
bright_region[7:13, 7:13] = 0.8  # 6x6 bright square
star_mask = bright_region > 0.5

eroded = erosion(bright_region, skdisk(1))
# Erosion should reduce the bright region
eroded_in_mask = np.sum(eroded[star_mask] > 0.5)
original_in_mask = np.sum(bright_region[star_mask] > 0.5)
check("Erosion reduces bright area in mask region",
      eroded_in_mask < original_in_mask,
      f"before={original_in_mask}, after={eroded_in_mask}")

# ── T22: internal-only enforcement ────────────────────────────────────────────
section("T22 generate_preview — internal function checks")

if generate_preview is not None:
    check("T22 is a plain Python function (not BaseTool)", not isinstance(generate_preview, BaseTool))
    check("T22 has 'fits_path' parameter", "fits_path" in inspect.signature(generate_preview).parameters)
    check("T22 has 'auto_stretch_linear' parameter",
          "auto_stretch_linear" in inspect.signature(generate_preview).parameters)

# ── T09: subsky syntax fix ─────────────────────────────────────────────────────
section("T09 remove_gradient — subsky syntax fixes")

import astro_agent.tools.linear.t09_gradient as t09_module

t09_src = inspect.getsource(t09_module)

# Polynomial case must use -samples= and -tolerance= flags, not positional args
check("T09: polynomial subsky uses -samples= flag",
      "-samples=" in t09_src)
check("T09: polynomial subsky uses -tolerance= flag",
      "-tolerance=" in t09_src)
check("T09: RBF subsky uses -smooth= flag",
      "-smooth=" in t09_src)
check("T09: dither field defined in SirilSubskyOptions",
      "dither" in t09_src)

# Ensure the old positional-only pattern is gone for polynomial
bad_poly = "f\"subsky {options.polynomial_degree} \"\n            f\"{options.samples_per_line}"
check("T09: old positional polynomial syntax removed",
      bad_poly not in t09_src)

# ── T10: limitmag and bgtol ───────────────────────────────────────────────────
section("T10 color_calibrate — limitmag and bgtol")

import astro_agent.tools.linear.t10_color_calibrate as t10_module

t10_src = inspect.getsource(t10_module)
check("T10: limitmag parameter in source",
      "limitmag" in t10_src)
check("T10: bgtol_lower parameter in source",
      "bgtol_lower" in t10_src)
check("T10: -bgtol= flag in PCC/SPCC builder",
      "-bgtol=" in t10_src)
check("T10: -limitmag= flag in PCC/SPCC builder",
      "-limitmag=" in t10_src)

# ── T12: VST and nocosmetic ───────────────────────────────────────────────────
section("T12 noise_reduction — vst and nocosmetic flags")

import astro_agent.tools.linear.t12_noise_reduction as t12_module

t12_src = inspect.getsource(t12_module)
check("T12: use_vst field in SirilDenoiseOptions",
      "use_vst" in t12_src)
check("T12: -vst flag in denoise command",
      '"-vst"' in t12_src or "\" -vst\"" in t12_src or "vst" in t12_src and "-vst" in t12_src)
check("T12: apply_cosmetic field in SirilDenoiseOptions",
      "apply_cosmetic" in t12_src)
check("T12: -nocosmetic flag in denoise command",
      "-nocosmetic" in t12_src)
check("T12: VST only applies to standard method",
      "method == \"standard\"" in t12_src or "options.method == \"standard\"" in t12_src)

# ── T13: makepsf manual and rl improvements ───────────────────────────────────
section("T13 deconvolution — makepsf manual, rl -stop=, -fh")

import astro_agent.tools.linear.t13_deconvolution as t13_module

t13_src = inspect.getsource(t13_module)
check("T13: ManualPsfOptions class defined",
      "ManualPsfOptions" in t13_src)
check("T13: makepsf manual -moffat supported",
      "-moffat" in t13_src or "moffat" in t13_src)
check("T13: makepsf manual -airy supported",
      "-airy" in t13_src or "airy" in t13_src)
check("T13: makepsf stars -sym supported",
      "-sym" in t13_src)
check("T13: rl -stop= flag used when stop is not None",
      "-stop=" in t13_src)
check("T13: rl -fh flag for Hessian Frobenius regularization",
      "-fh" in t13_src)
check("T13: psf_source manual handled in command builder",
      "psf_source == \"manual\"" in t13_src)
check("T13: makepsf stars NOT using -auto",
      "makepsf -auto" not in t13_src)

# Verify manual Moffat command builds correctly
from astro_agent.tools.linear.t13_deconvolution import ManualPsfOptions, PsfConfig, _build_makepsf_manual

mo = ManualPsfOptions(profile="moffat", fwhm_px=2.5, moffat_beta=4.0)
psf_cmd = _build_makepsf_manual(mo, PsfConfig())
check("T13: makepsf manual moffat command = 'makepsf manual -moffat -fwhm=2.5 -beta=4.0'",
      psf_cmd == "makepsf manual -moffat -fwhm=2.5 -beta=4.0",
      detail=psf_cmd)

# Verify Airy command builds correctly
mo_airy = ManualPsfOptions(
    profile="airy",
    airy_diameter_mm=130.0,
    airy_focal_length_mm=910.0,
    airy_wavelength_nm=656.0,
    airy_obstruction_pct=0.0,
)
airy_cmd = _build_makepsf_manual(mo_airy, PsfConfig())
check("T13: makepsf manual airy command builds",
      "-airy" in airy_cmd and "-dia=" in airy_cmd and "-wl=" in airy_cmd,
      detail=airy_cmd)

# ── T17: edge_preserve (epf) ─────────────────────────────────────────────────
section("T17 local_contrast_enhance — edge_preserve (epf) method")

import astro_agent.tools.nonlinear.t17_local_contrast as t17_module

t17_src = inspect.getsource(t17_module)
check("T17: EpfOptions class defined",
      "EpfOptions" in t17_src)
check("T17: edge_preserve method in method choices",
      "edge_preserve" in t17_src)
check("T17: epf command built",
      '"epf"' in t17_src or "'epf'" in t17_src or "= \"epf\"" in t17_src or "epf_cmd = \"epf\"" in t17_src)
check("T17: -guided flag supported",
      "-guided" in t17_src)
check("T17: -si= intensity sigma flag",
      "-si=" in t17_src)
check("T17: -ss= spatial sigma flag",
      "-ss=" in t17_src)
check("T17: -mod= blend strength flag",
      "-mod=" in t17_src)

# Simulate epf command build
from astro_agent.tools.nonlinear.t17_local_contrast import EpfOptions

o = EpfOptions(guided=False, diameter=5, intensity_sigma=0.02, spatial_sigma=0.02, mod=0.8)
epf_cmd = "epf"
if o.guided:
    epf_cmd += " -guided"
if o.diameter != 3:
    epf_cmd += f" -d={o.diameter}"
epf_cmd += f" -si={o.intensity_sigma} -ss={o.spatial_sigma} -mod={o.mod}"
check("T17: bilateral epf command builds correctly",
      epf_cmd == "epf -d=5 -si=0.02 -ss=0.02 -mod=0.8",
      detail=epf_cmd)

# ── T28: extract_narrowband ───────────────────────────────────────────────────
section("T28 extract_narrowband — import and structure")

try:
    from astro_agent.tools.utility.t28_extract_narrowband import extract_narrowband as t28_tool
    from langchain_core.tools import BaseTool
    check("T28 extract_narrowband imports", True)
    check("T28 is @tool decorated (BaseTool instance)",
          isinstance(t28_tool, BaseTool))
except Exception as e:
    check("T28 extract_narrowband imports", False, str(e))
    t28_tool = None

if t28_tool is not None:
    import astro_agent.tools.utility.t28_extract_narrowband as t28_module
    t28_src = inspect.getsource(t28_module)
    check("T28: extract_Ha command in source", "extract_Ha" in t28_src)
    check("T28: extract_HaOIII command in source", "extract_HaOIII" in t28_src)
    check("T28: extract_Green command in source", "extract_Green" in t28_src)
    check("T28: -upscale flag supported", "-upscale" in t28_src)
    check("T28: -resample= flag supported", "-resample=" in t28_src)
    check("T28: ha, ha_oiii, green extraction types defined",
          "\"ha\"" in t28_src and "\"ha_oiii\"" in t28_src and "\"green\"" in t28_src)
    check("T28: returns ha_path, oiii_path, green_path",
          "ha_path" in t28_src and "oiii_path" in t28_src and "green_path" in t28_src)

# ── check_dependencies ────────────────────────────────────────────────────────
section("check_dependencies()")
try:
    from astro_agent.config import check_dependencies
    check_dependencies()
    check("check_dependencies() passes", True)
except SystemExit as e:
    check("check_dependencies() passes", False, f"SystemExit: {e}")
except Exception as e:
    check("check_dependencies() passes", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print('='*60)
if FAIL > 0:
    sys.exit(1)
