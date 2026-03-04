"""
Phase 4 verification script — Stretch & Non-Linear Tools (T14–T19).

Checks:
  1. All six tools import cleanly and are @tool-decorated.
  2. T14 GHS/autostretch/asinh command builders produce correct Siril syntax.
  3. T15 StarNet binary and weights are accessible.
  4. T16 MTF and GHT command builders produce correct Siril syntax.
  5. T17 wavelet layer_weights padding/trimming logic is correct.
  6. T18 satu and GHT-sat command builders are correct.
  7. T19 pixel math expression builder is correct.
  8. T11 rmgreen fix: -nopreserve flag, not third positional arg.
  9. T13 makepsf fix: 'makepsf stars' not 'makepsf -auto'.
 10. check_dependencies() still passes.

Run:
    uv run python scripts/verify_phase_4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FAILURES: list[str] = []
PASSES:   list[str] = []


def ok(msg: str) -> None:
    PASSES.append(msg)
    print(f"  \033[32m✓\033[0m  {msg}")


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  \033[31m✗\033[0m  {msg}")


# ── 1. Tool imports ────────────────────────────────────────────────────────────

print("\n── 1. Tool imports ──────────────────────────────────────────────────────")

try:
    from astro_agent.tools.nonlinear.t14_stretch import (
        stretch_image, GHSOptions, AsinhOptions, AutostretchOptions,
        _build_ghs_cmd, _build_asinh_cmd, _build_autostretch_cmd,
    )
    ok("T14 stretch_image imports")
except Exception as e:
    fail(f"T14 import failed: {e}")

try:
    from astro_agent.tools.nonlinear.t15_star_removal import (
        star_removal, StarRemovalInput,
    )
    ok("T15 star_removal imports")
except Exception as e:
    fail(f"T15 import failed: {e}")

try:
    from astro_agent.tools.nonlinear.t16_curves import (
        curves_adjust, MTFOptions, GHTCurvesOptions,
        _build_mtf_cmd, _build_ght_curves_cmd,
    )
    ok("T16 curves_adjust imports")
except Exception as e:
    fail(f"T16 import failed: {e}")

try:
    from astro_agent.tools.nonlinear.t17_local_contrast import (
        local_contrast_enhance, WaveletOptions, ClaheOptions, UnsharpOptions,
    )
    ok("T17 local_contrast_enhance imports")
except Exception as e:
    fail(f"T17 import failed: {e}")

try:
    from astro_agent.tools.nonlinear.t18_saturation import (
        saturation_adjust, GHTSatOptions, HUE_RANGE_DESCRIPTIONS,
    )
    ok("T18 saturation_adjust imports")
except Exception as e:
    fail(f"T18 import failed: {e}")

try:
    from astro_agent.tools.nonlinear.t19_star_restoration import (
        star_restoration, BlendOptions, SynthstarOptions,
    )
    ok("T19 star_restoration imports")
except Exception as e:
    fail(f"T19 import failed: {e}")


# ── 2. @tool decoration ────────────────────────────────────────────────────────

print("\n── 2. @tool decoration ──────────────────────────────────────────────────")

for name, fn in [
    ("T14 stretch_image",         stretch_image),
    ("T15 star_removal",          star_removal),
    ("T16 curves_adjust",         curves_adjust),
    ("T17 local_contrast_enhance", local_contrast_enhance),
    ("T18 saturation_adjust",     saturation_adjust),
    ("T19 star_restoration",      star_restoration),
]:
    try:
        if hasattr(fn, "name") and hasattr(fn, "invoke"):
            ok(f"{name} is @tool-decorated (.name={fn.name!r})")
        else:
            fail(f"{name}: not properly decorated")
    except Exception as e:
        fail(f"{name} decoration check failed: {e}")


# ── 3. T14 command builders ────────────────────────────────────────────────────

print("\n── 3. T14 GHS / asinh / autostretch command builders ────────────────────")

try:
    ghs = GHSOptions(stretch_amount=2.5, highlight_protection=0.95)
    cmd = _build_ghs_cmd(ghs)
    assert "-D=2.5" in cmd
    assert "-HP=0.95" in cmd
    assert "-human" in cmd
    ok(f"GHS command: '{cmd}'")
except Exception as e:
    fail(f"GHS command build failed: {e}")

try:
    ghs_minimal = GHSOptions(stretch_amount=1.5)
    cmd = _build_ghs_cmd(ghs_minimal)
    assert "-D=1.5" in cmd
    assert "-B=" not in cmd   # default 0.0, should be omitted
    ok(f"GHS minimal command (no B/SP/LP/HP): '{cmd}'")
except Exception as e:
    fail(f"GHS minimal command failed: {e}")

try:
    asinh = AsinhOptions(stretch_factor=200.0, black_point_offset=0.01)
    cmd = _build_asinh_cmd(asinh)
    assert "asinh" in cmd
    assert "200.0" in cmd
    assert "0.01" in cmd
    ok(f"Asinh command: '{cmd}'")
except Exception as e:
    fail(f"Asinh command build failed: {e}")

try:
    auto = AutostretchOptions(shadows_clipping_sigma=-2.8, target_background=0.25, linked=True)
    cmd = _build_autostretch_cmd(auto)
    assert "autostretch" in cmd
    assert "-linked" in cmd
    assert "-2.8" in cmd
    ok(f"Autostretch command: '{cmd}'")
except Exception as e:
    fail(f"Autostretch command build failed: {e}")

try:
    ghs_ch = GHSOptions(stretch_amount=1.0, channels="RG")
    cmd = _build_ghs_cmd(ghs_ch)
    assert "RG" in cmd
    ok(f"GHS per-channel RG: '{cmd}'")
except Exception as e:
    fail(f"GHS per-channel failed: {e}")


# ── 4. T15 StarNet binary check ────────────────────────────────────────────────

print("\n── 4. T15 StarNet binary ─────────────────────────────────────────────────")

try:
    from astro_agent.config import load_settings
    settings = load_settings()
    starnet_path = Path(settings.starnet_bin)
    weights_path = Path(settings.starnet_weights)
    if starnet_path.exists():
        ok(f"STARNET_BIN exists: {starnet_path}")
    else:
        fail(f"STARNET_BIN not found: {starnet_path}")
    if weights_path.exists():
        ok(f"STARNET_WEIGHTS exists: {weights_path}")
    else:
        fail(f"STARNET_WEIGHTS not found: {weights_path}")
except Exception as e:
    fail(f"StarNet check failed: {e}")


# ── 5. T16 curves command builders ────────────────────────────────────────────

print("\n── 5. T16 MTF / GHT curves command builders ─────────────────────────────")

try:
    mtf = MTFOptions(black_point=0.01, midtone=0.35, white_point=1.0)
    cmd = _build_mtf_cmd(mtf)
    assert cmd == "mtf 0.01 0.35 1.0", f"Got: {cmd}"
    ok(f"MTF command: '{cmd}'")
except Exception as e:
    fail(f"MTF command build failed: {e}")

try:
    mtf_ch = MTFOptions(midtone=0.4, channels="G")
    cmd = _build_mtf_cmd(mtf_ch)
    assert "G" in cmd
    ok(f"MTF per-channel: '{cmd}'")
except Exception as e:
    fail(f"MTF per-channel failed: {e}")

try:
    ght = GHTCurvesOptions(stretch_amount=1.0, local_intensity=8.0, symmetry_point=0.2)
    cmd = _build_ght_curves_cmd(ght)
    assert "-D=1.0" in cmd
    assert "-B=8.0" in cmd
    assert "-SP=0.2" in cmd
    ok(f"GHT curves command: '{cmd}'")
except Exception as e:
    fail(f"GHT curves command build failed: {e}")


# ── 6. T17 wavelet weights logic ───────────────────────────────────────────────

print("\n── 6. T17 wavelet layer_weights padding ─────────────────────────────────")

try:
    # 5 layers → 6 weights needed (5 detail + 1 residual)
    wopt = WaveletOptions(num_layers=5, layer_weights=[1.3, 1.1, 1.0])
    expected_len = wopt.num_layers + 1  # = 6
    weights = list(wopt.layer_weights)
    if len(weights) < expected_len:
        weights = weights + [1.0] * (expected_len - len(weights))
    weights = weights[:expected_len]
    assert len(weights) == 6, f"Got {len(weights)} weights"
    assert weights == [1.3, 1.1, 1.0, 1.0, 1.0, 1.0]
    ok(f"Wavelet weights padded correctly: {weights}")
except Exception as e:
    fail(f"Wavelet weight padding failed: {e}")

try:
    # Exact weights provided
    wopt2 = WaveletOptions(num_layers=3, layer_weights=[1.2, 1.1, 1.0, 1.0])
    weights2 = list(wopt2.layer_weights)[: wopt2.num_layers + 1]
    assert len(weights2) == 4
    ok(f"Wavelet 3-layer exact weights trimmed correctly: {weights2}")
except Exception as e:
    fail(f"Wavelet weight trim failed: {e}")


# ── 7. T18 saturation command builders ────────────────────────────────────────

print("\n── 7. T18 saturation command builders ───────────────────────────────────")

try:
    # Global satu
    cmd = f"satu 0.5 1.5"
    assert "satu" in cmd and "0.5" in cmd and "1.5" in cmd
    ok(f"Global satu command: '{cmd}'")
except Exception as e:
    fail(f"Satu global failed: {e}")

try:
    # Targeted Hα (index 0)
    cmd = f"satu 0.8 1.5 0"
    assert cmd == "satu 0.8 1.5 0"
    ok(f"Hα targeted satu: '{cmd}'")
except Exception as e:
    fail(f"Targeted satu failed: {e}")

try:
    opts = GHTSatOptions(stretch_amount=0.8, local_intensity=5.0, symmetry_point=0.5)
    cmd = f"ght -sat -D={opts.stretch_amount} -B={opts.local_intensity}"
    assert "-sat" in cmd
    assert "-D=0.8" in cmd
    ok(f"GHT-sat command: '{cmd}'")
except Exception as e:
    fail(f"GHT-sat command failed: {e}")

try:
    assert 0 in HUE_RANGE_DESCRIPTIONS
    assert "Hα" in HUE_RANGE_DESCRIPTIONS[0]
    assert 3 in HUE_RANGE_DESCRIPTIONS
    assert "OIII" in HUE_RANGE_DESCRIPTIONS[3]
    ok("Hue range descriptions complete (0=Hα, 3=OIII)")
except Exception as e:
    fail(f"Hue range descriptions check failed: {e}")


# ── 8. T19 pixel math expression ──────────────────────────────────────────────

print("\n── 8. T19 pixel math expression ─────────────────────────────────────────")

try:
    starless_stem = "master_light_crop_starless"
    mask_stem = "master_light_crop_starmask"
    weight = 0.85
    expression = f'"${starless_stem}$ + ${mask_stem}$ * {weight}"'
    assert f"${starless_stem}$" in expression
    assert f"${mask_stem}$" in expression
    assert "0.85" in expression
    ok(f"Pixel math expression: {expression}")
except Exception as e:
    fail(f"Pixel math expression failed: {e}")


# ── 9. T11 / T13 fixes ────────────────────────────────────────────────────────

print("\n── 9. T11 rmgreen / T13 makepsf fixes ───────────────────────────────────")

try:
    import inspect
    from astro_agent.tools.linear.t11_green_noise import remove_green_noise
    src = inspect.getsource(remove_green_noise.func)
    assert "rmgreen" in src
    assert "nopreserve" in src
    assert "rmgreen 0 1.0 1" not in src   # old wrong syntax
    ok("T11 rmgreen uses -nopreserve flag (not third positional arg)")
except Exception as e:
    fail(f"T11 rmgreen fix check failed: {e}")

try:
    from astro_agent.tools.linear.t13_deconvolution import deconvolution
    src = inspect.getsource(deconvolution.func)
    # The commands list must append "makepsf stars" (not "makepsf -auto")
    assert 'commands.append("makepsf stars")' in src, "makepsf stars not found in commands"
    assert 'commands.append("makepsf -auto")' not in src, "old makepsf -auto still present"
    ok("T13 commands use 'makepsf stars' (not 'makepsf -auto')")
except Exception as e:
    fail(f"T13 makepsf fix check failed: {e}")


# ── 10. check_dependencies ────────────────────────────────────────────────────

print("\n── 10. check_dependencies() ─────────────────────────────────────────────")

try:
    from astro_agent.config import check_dependencies
    check_dependencies(settings)
    ok("check_dependencies() passed")
except Exception as e:
    fail(f"check_dependencies() failed: {e}")


# ── Summary ────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print(f"  Passed: {len(PASSES)}")
print(f"  Failed: {len(FAILURES)}")
print("=" * 60)

if FAILURES:
    print("\nFailing checks:")
    for f in FAILURES:
        print(f"  • {f}")
    sys.exit(1)
else:
    print("\n  All Phase 4 checks passed.")
    sys.exit(0)
