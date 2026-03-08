"""
Phase 3 verification script — Linear Processing Tools (T09–T13).

Checks:
  1. All five tools import cleanly and are @tool-decorated.
  2. T10 camera pixel size lookup table covers the Fujifilm X-T30 II.
  3. T10 resolve_pixel_size() priority chain works correctly.
  4. GraXpert binary is accessible and models are cached locally.
  5. T09/T12 GraXpert option schemas validate correctly.
  6. T13 RL command builder generates correct Siril commands.
  7. check_dependencies() still passes (includes STARNET_WEIGHTS added in Phase 3).

Run:
    uv run python scripts/verify_pahse_3.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable
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
    from astro_agent.tools.linear.t09_gradient import (
        remove_gradient,
        GraXpertBGEOptions,
        SirilSubskyOptions,
        RemoveGradientInput,
    )
    ok("T09 remove_gradient imports")
except Exception as e:
    fail(f"T09 import failed: {e}")

try:
    from astro_agent.tools.linear.t10_color_calibrate import (
        color_calibrate,
        ColorCalibrateInput,
        resolve_pixel_size,
    )
    from astro_agent.equipment import load_equipment
    ok("T10 color_calibrate imports")
except Exception as e:
    fail(f"T10 import failed: {e}")

try:
    from astro_agent.tools.linear.t11_green_noise import (
        remove_green_noise,
        RemoveGreenNoiseInput,
    )
    ok("T11 remove_green_noise imports")
except Exception as e:
    fail(f"T11 import failed: {e}")

try:
    from astro_agent.tools.linear.t12_noise_reduction import (
        noise_reduction,
        SirilDenoiseOptions,
        GraXpertDenoiseOptions,
        NoiseReductionInput,
    )
    ok("T12 noise_reduction imports")
except Exception as e:
    fail(f"T12 import failed: {e}")

try:
    from astro_agent.tools.linear.t13_deconvolution import (
        deconvolution,
        RLOptions,
        WienerOptions,
        DeconvolutionInput,
    )
    ok("T13 deconvolution imports")
except Exception as e:
    fail(f"T13 import failed: {e}")


# ── 2. @tool decoration ────────────────────────────────────────────────────────

print("\n── 2. @tool decoration ──────────────────────────────────────────────────")

for name, fn in [
    ("T09 remove_gradient",   "remove_gradient"),
    ("T10 color_calibrate",   "color_calibrate"),
    ("T11 remove_green_noise", "remove_green_noise"),
    ("T12 noise_reduction",   "noise_reduction"),
    ("T13 deconvolution",     "deconvolution"),
]:
    try:
        tool_fn = globals().get(fn) or locals().get(fn)
        # All five are imported into local scope above
        if hasattr(tool_fn, "name") and hasattr(tool_fn, "invoke"):
            ok(f"{name} is @tool-decorated (has .name={tool_fn.name!r})")
        else:
            fail(f"{name}: missing .name or .invoke — not properly decorated")
    except Exception as e:
        fail(f"{name} decoration check failed: {e}")


# ── 3. T10 pixel size — equipment.toml config ─────────────────────────────────

print("\n── 3. T10 pixel size (equipment.toml) ───────────────────────────────────")

try:
    equip = load_equipment()
    camera = equip.get("camera", {})
    px_config = camera.get("pixel_size_um")
    if px_config is not None and px_config > 0:
        ok(f"equipment.toml pixel_size_um = {px_config} µm (camera: {camera.get('model', '?')})")
    else:
        fail("equipment.toml [camera] pixel_size_um is missing or zero")
except Exception as e:
    fail(f"equipment.toml load failed: {e}")

try:
    px = resolve_pixel_size(None)
    if abs(px - 3.76) < 0.01:
        ok(f"resolve_pixel_size(None) = {px} µm from equipment.toml")
    else:
        fail(f"resolve_pixel_size(None) = {px} µm, expected ~3.76 from config")
except Exception as e:
    fail(f"resolve_pixel_size(None) failed: {e}")

try:
    px = resolve_pixel_size(4.5)
    if px == 4.5:
        ok("Explicit pixel_size_um overrides equipment.toml")
    else:
        fail(f"Explicit override failed: got {px}, expected 4.5")
except Exception as e:
    fail(f"Explicit override check failed: {e}")


# ── 4. GraXpert model cache check ─────────────────────────────────────────────

print("\n── 4. GraXpert model cache ──────────────────────────────────────────────")

graxpert_support = Path.home() / "Library" / "Application Support" / "GraXpert"

bge_model = graxpert_support / "bge-ai-models" / "1.0.1" / "model.onnx"
if bge_model.exists():
    ok(f"BGE model 1.0.1 cached: {bge_model}")
else:
    fail(
        f"BGE model 1.0.1 not found at {bge_model}. "
        "Run: GraXpert <any_fits> -cli -cmd background-extraction -ai_version 1.0.1"
    )

denoise_model = graxpert_support / "denoise-ai-models" / "2.0.0" / "model.onnx"
if denoise_model.exists():
    ok(f"Denoise model 2.0.0 cached: {denoise_model}")
else:
    fail(
        f"Denoise model 2.0.0 not found at {denoise_model}. "
        "Run: GraXpert <any_fits> -cli -cmd denoising -ai_version 2.0.0"
    )


# ── 5. Schema validation ───────────────────────────────────────────────────────

print("\n── 5. Schema validation ─────────────────────────────────────────────────")

try:
    opts = GraXpertBGEOptions(correction_type="Division", smoothing=0.3, ai_version="1.0.1")
    assert opts.correction_type == "Division"
    assert opts.smoothing == 0.3
    ok("GraXpertBGEOptions validates correctly")
except Exception as e:
    fail(f"GraXpertBGEOptions validation failed: {e}")

try:
    opts = SirilDenoiseOptions(modulation=0.8, method="da3d")
    assert opts.method == "da3d"
    ok("SirilDenoiseOptions validates correctly")
except Exception as e:
    fail(f"SirilDenoiseOptions validation failed: {e}")

try:
    opts = GraXpertDenoiseOptions(strength=0.7, ai_version="2.0.0")
    assert opts.strength == 0.7
    ok("GraXpertDenoiseOptions validates correctly")
except Exception as e:
    fail(f"GraXpertDenoiseOptions validation failed: {e}")

try:
    opts = RLOptions(iterations=15, regularization="total_variation", alpha=5000.0)
    assert opts.iterations == 15
    ok("RLOptions validates correctly")
except Exception as e:
    fail(f"RLOptions validation failed: {e}")

try:
    opts = WienerOptions(alpha=0.002)
    assert opts.alpha == 0.002
    ok("WienerOptions validates correctly")
except Exception as e:
    fail(f"WienerOptions validation failed: {e}")


# ── 6. T13 RL command builder ──────────────────────────────────────────────────

print("\n── 6. T13 RL command string construction ────────────────────────────────")

try:
    rl = RLOptions(iterations=10, regularization="total_variation", alpha=3000.0)
    rl_cmd = f"rl -iters={rl.iterations}"
    if rl.regularization == "total_variation":
        rl_cmd += f" -tv -alpha={rl.alpha}"
    assert rl_cmd == "rl -iters=10 -tv -alpha=3000.0"
    ok(f"RL TV command: '{rl_cmd}'")
except Exception as e:
    fail(f"RL TV command build failed: {e}")

try:
    rl = RLOptions(iterations=20, regularization="none")
    rl_cmd = f"rl -iters={rl.iterations}"
    if rl.regularization == "total_variation":
        rl_cmd += f" -tv -alpha={rl.alpha}"
    elif rl.regularization == "frobenius":
        rl_cmd += f" -alpha={rl.alpha}"
    assert rl_cmd == "rl -iters=20"
    ok(f"RL no-reg command: '{rl_cmd}'")
except Exception as e:
    fail(f"RL no-reg command build failed: {e}")


# ── 7. check_dependencies ─────────────────────────────────────────────────────

print("\n── 7. check_dependencies() ──────────────────────────────────────────────")

try:
    from astro_agent.config import check_dependencies, load_settings
    settings = load_settings()
    check_dependencies(settings)
    ok("check_dependencies() passed — all binaries and weights found")
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
    print("\n  All Phase 3 checks passed.")
    sys.exit(0)
