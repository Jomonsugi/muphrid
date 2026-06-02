#!/usr/bin/env python3
"""
Focused execution test for the Siril `subsky` background-extraction backends.

Run from project root:
    uv run python scripts/test_gradient_subsky.py

Exit 0 = all checks pass (or Siril is unavailable and the test is skipped).

Why this exists: remove_gradient's polynomial backend once emitted the
command `background -gradient ...`, which does not exist in any Siril
version. Every static/import check passed, yet the tool failed at runtime
the moment an agent reached for it. The only test that catches that class
of bug is one that actually invokes siril-cli. This script runs both
deterministic backends (polynomial and rbf) against a synthetic FITS and
asserts the command is accepted (exit 0) and produces output.

Requires siril-cli (SIRIL_BIN or 'siril-cli' on PATH) and astropy. If
Siril is not installed, the test skips rather than fails.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    msg = f" {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


def _siril_available() -> bool:
    from muphrid.config import load_settings

    try:
        binary = load_settings().siril_bin
    except Exception:
        binary = os.environ.get("SIRIL_BIN", "siril-cli")
    return shutil.which(binary) is not None or Path(binary).is_file()


def _make_synthetic_fits(path: Path) -> float:
    """Write a 3-layer FITS with a smooth additive gradient + faint signal.

    Returns the mean pixel value before background removal.
    """
    import numpy as np
    from astropy.io import fits

    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    gradient = 0.1 + 0.3 * (xx / w) + 0.2 * (yy / h)
    rng = np.random.default_rng(0)
    chans = []
    for _ in range(3):
        sig = np.zeros((h, w), "float32")
        sig[100:120, 100:140] = 0.4  # a compact 'nebula'
        chans.append(
            (gradient + sig + rng.normal(0, 0.002, (h, w))).clip(0, 1).astype("float32")
        )
    data = np.stack(chans)
    fits.writeto(path, data, overwrite=True)
    return float(data.mean())


def test_subsky_backends_run() -> None:
    """Both subsky backends are accepted by Siril and produce valid output."""
    print("\n[1] subsky polynomial + rbf backends run against a real FITS")

    if not _siril_available():
        print(" SKIP - siril-cli not found (set SIRIL_BIN); cannot exercise subsky")
        return

    from astropy.io import fits

    from muphrid.tools.linear.gradient import (
        PolynomialBGEOptions,
        RBFBGEOptions,
        _run_polynomial_bge,
        _run_rbf_bge,
    )

    wd = tempfile.mkdtemp(prefix="subsky_test_")
    src = Path(wd) / "src.fit"
    before_mean = _make_synthetic_fits(src)

    # Polynomial path (with dither) — the backend that regressed.
    try:
        out, bg = _run_polynomial_bge(
            src, PolynomialBGEOptions(degree=2, dither=True), "src_poly", wd
        )
        out_data = fits.getdata(out)
        check("polynomial subsky exits 0 and writes output", Path(out).exists())
        check("polynomial returns no separate bg-model file", bg is None)
        check(
            "polynomial reduced the background",
            float(out_data.mean()) < before_mean,
            f"{before_mean:.4f} -> {float(out_data.mean()):.4f}",
        )
    except Exception as e:
        check("polynomial subsky exits 0 and writes output", False, f"{type(e).__name__}: {e}")

    # RBF path.
    try:
        out, bg = _run_rbf_bge(
            src, RBFBGEOptions(smoothing=0.5, dither=False), "src_rbf", wd
        )
        out_data = fits.getdata(out)
        check("rbf subsky exits 0 and writes output", Path(out).exists())
        check("rbf returns no separate bg-model file", bg is None)
        check(
            "rbf reduced the background",
            float(out_data.mean()) < before_mean,
            f"{before_mean:.4f} -> {float(out_data.mean()):.4f}",
        )
    except Exception as e:
        check("rbf subsky exits 0 and writes output", False, f"{type(e).__name__}: {e}")


def main() -> int:
    test_subsky_backends_run()
    print()
    if _failures:
        print(f"FAIL — {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f" - {f}")
        return 1
    print("All subsky backend tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
