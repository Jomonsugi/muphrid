#!/usr/bin/env python3
"""
Diagnostic: which calibration approach preserves the nebula?

Each test calibrates the full lights sequence with different settings,
registers, stacks, and saves a single master light FITS. Intermediate
sequences (100GB+) are deleted between tests to conserve disk space.

After running, open each result in Siril → autostretch → check for nebula.

Usage:
    uv run python scripts/diagnose_calibration.py
"""

import subprocess
import sys
import time
from pathlib import Path

SIRIL = "/Applications/Siril.app/Contents/MacOS/siril-cli"
WD = Path("/Users/micahshanks/Desktop/orion_nebula/runs/run-m42-20260319-221651")
OUTPUT_DIR = WD / "diag"


def run_siril(script: str, label: str) -> bool:
    """Run a Siril script. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    start = time.time()

    proc = subprocess.run(
        [SIRIL, "-s", "-"],
        input=f"requires 1.2.0\n{script}",
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min max per test
    )

    elapsed = time.time() - start
    ok = proc.returncode == 0

    # Check for errors in stdout
    has_error = False
    for line in proc.stdout.split("\n"):
        if "error" in line.lower() and "cosmetic" not in line.lower():
            has_error = True
            print(f"    {line.strip()}")

    if not ok or has_error:
        print(f"  FAILED ({elapsed:.0f}s)")
        if not ok:
            for line in proc.stdout.split("\n")[-10:]:
                if line.strip():
                    print(f"    {line.strip()}")
        return False

    print(f"  OK ({elapsed:.0f}s)")
    return True


def cleanup(prefix: str):
    """Remove intermediate calibrated/registered sequences to free disk."""
    for pattern in [f"{prefix}lights.*", f"r_{prefix}lights.*"]:
        for f in WD.glob(pattern):
            size_gb = f.stat().st_size / (1024**3)
            f.unlink()
            print(f"    Cleaned {f.name} ({size_gb:.1f}GB)")


def run_test(name: str, description: str, calibrate_flags: str) -> bool:
    """Run one full test: calibrate → register → stack → save result."""
    prefix = f"d{name}_"
    output = OUTPUT_DIR / f"stack_{name}.fit"

    if output.exists():
        print(f"\n  Skipping {name} — {output.name} already exists")
        return True

    # Step 1: Calibrate
    cal_script = f"""cd {WD}
calibrate lights {calibrate_flags} -prefix={prefix}
"""
    if not run_siril(cal_script, f"Test {name} — CALIBRATE: {description}"):
        cleanup(prefix)
        return False

    # Step 2: Register (compute transforms)
    reg_script = f"""cd {WD}
register {prefix}lights -2pass -transf=homography -maxstars=500
"""
    if not run_siril(reg_script, f"Test {name} — REGISTER"):
        cleanup(prefix)
        return False

    # Step 2b: Apply registration (creates r_ sequence with interpolated frames)
    apply_script = f"""cd {WD}
seqapplyreg {prefix}lights -framing=min -interp=lanczos4
"""
    if not run_siril(apply_script, f"Test {name} — APPLY REG"):
        cleanup(prefix)
        return False

    # Step 3: Stack — output to working directory, then move to diag/
    out_name = f"stack_{name}"
    stack_script = f"""cd {WD}
stack r_{prefix}lights rej s 3 3 -norm=addscale -output_norm -weight=wfwhm -32b -out={out_name}
"""
    if not run_siril(stack_script, f"Test {name} — STACK"):
        cleanup(prefix)
        return False

    # Move result to diag/
    src = WD / f"{out_name}.fit"
    if not src.exists():
        src = WD / f"{out_name}.fits"
    if src.exists():
        import shutil
        shutil.move(str(src), str(output))

    # Step 4: Cleanup intermediate sequences
    print(f"  Cleaning intermediates...")
    cleanup(prefix)

    if output.exists():
        size_mb = output.stat().st_size / (1024**2)
        print(f"  Result: {output.name} ({size_mb:.0f}MB)")
        return True
    else:
        print(f"  ERROR: {output.name} not created")
        return False


def main():
    if not WD.exists():
        print(f"ERROR: {WD} not found")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("CALIBRATION DIAGNOSTIC")
    print("Which calibration preserves the M42 nebula?")
    print("=" * 60)
    print(f"Working directory: {WD}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print()

    # Define tests: (name, description, calibrate flags)
    tests = [
        (
            "B_bias_only",
            "Bias only — no dark, no flat",
            "-bias=master_bias -cfa -debayer",
        ),
        (
            "C_bias_dark",
            "Bias + Dark — no flat",
            "-bias=master_bias -dark=master_dark -cfa -debayer",
        ),
        (
            "D_bias_flat",
            "Bias + Flat — no dark (tests flat correction alone)",
            "-bias=master_bias -flat=master_flat -cfa -debayer",
        ),
        (
            "E_bias_flat_eq",
            "Bias + Flat + equalize_cfa — no dark",
            "-bias=master_bias -flat=master_flat -cfa -debayer -equalize_cfa",
        ),
        (
            "F_full_no_xtrans",
            "Bias + Dark + Flat — no X-Trans flags",
            "-bias=master_bias -dark=master_dark -flat=master_flat -cfa -debayer",
        ),
        (
            "G_full_all",
            "Full calibration — all flags (current agent behavior)",
            "-bias=master_bias -dark=master_dark -flat=master_flat -cfa -debayer -equalize_cfa -fix_xtrans",
        ),
    ]

    results = []
    for name, desc, flags in tests:
        ok = run_test(name, desc, flags)
        results.append((name, desc, ok))

    # Print summary
    print("\n\n")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    for name, desc, ok in results:
        mark = "✓" if ok else "✗"
        exists = (OUTPUT_DIR / f"stack_{name}.fit").exists()
        print(f"  {mark} {name}: {desc}")
        if exists:
            print(f"      File: diag/stack_{name}.fit")

    print()
    print("=" * 60)
    print("HOW TO CHECK EACH RESULT IN SIRIL")
    print("=" * 60)
    print()
    print(f"1. Open Siril")
    print(f"2. In console, type:")
    print(f"   cd {WD}")
    print()

    for name, desc, ok in results:
        if (OUTPUT_DIR / f"stack_{name}.fit").exists():
            print(f"   Test {name} — {desc}:")
            print(f"   > load diag/stack_{name}")
            print(f"   > autostretch")
            print(f"   → Nebula visible? (yes / no)")
            print()

    print("=" * 60)
    print("Report which tests show the nebula.")
    print("This tells us exactly what calibration approach works.")
    print("=" * 60)


if __name__ == "__main__":
    main()
