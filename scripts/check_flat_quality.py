#!/usr/bin/env python3
"""
Flat and bias quality checker.

Per-frame flat evaluation uses sensor-native fill fraction:
  fill% = (median_adu - black_level) / (white_level - black_level) * 100

  This is camera-agnostic: a 14-bit Fuji and a 16-bit cooled sensor both
  target the same 30–55% fill of their usable ADU range.

Usage:
  uv run python scripts/check_flat_quality.py \\
    --flats test_images/flats_test --biases test_images/bias

  uv run python scripts/check_flat_quality.py \\
    --flats test_images/flats_test --biases test_images/bias --skip-folder

  uv run python scripts/check_flat_quality.py \\
    --bias-test test_images/bias_test

Target states (per frame):
  USABLE    — fill 30–55%, good signal, not clipped
  UNDER     — fill < 30%, increase exposure or light brightness
  OVER      — fill > 55%, decrease exposure or light brightness
  SATURATED — fill ≥ 97% OR near-zero variance at high ADU (clipped sensor well)
  UNKNOWN   — could not read frame
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv

load_dotenv(project_root / ".env")
load_dotenv(project_root / "muphrid" / ".env")

from muphrid.tools._sensor import (
    TARGET_FILL_CENTER,
    TARGET_FILL_MAX,
    TARGET_FILL_MIN,
    compute_fill,
    flat_adu_range,
    flat_fill_state,
    flat_siril_norm_thresholds,
    infer_white_level,
    read_frame_exif,
)
from muphrid.tools._siril import run_siril_script
from muphrid.tools.preprocess.t02_masters import build_masters


RAW_EXTS = {".raf", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef"}
FITS_EXTS = {".fit", ".fits", ".fts"}
IMAGE_EXTS = RAW_EXTS | FITS_EXTS


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class FlatEval:
    name: str
    state: str                          # USABLE / UNDER / OVER / SATURATED / UNKNOWN
    fill_pct: float | None              # primary metric: fraction of usable sensor range
    median_adu: int | None
    std_adu: float | None
    black_level: int | None
    white_level: int | None
    distance_to_target: float | None    # |fill - TARGET_FILL_CENTER|
    exposure_time: float | None = None  # shutter speed in seconds from EXIF
    siril_norm_median: float | None = None   # populated by folder aggregate (T02)
    siril_norm_min: float | None = None      # sensor-relative T02 threshold min
    siril_norm_max: float | None = None      # sensor-relative T02 threshold max
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BiasEval:
    name: str
    median_adu: float
    std_adu: float
    hot_pixel_pct: float
    rank: int = 0
    error: str | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _fmt_exposure(exp: float | None) -> str:
    """Format exposure time as a human-readable shutter speed string."""
    if exp is None:
        return "n/a"
    if exp >= 1.0:
        return f"{exp:.1f}s"
    denom = round(1.0 / exp)
    return f"1/{denom}s"


def _fill_target_label(fill: float | None) -> str:
    if fill is None:
        return "n/a"
    if fill < TARGET_FILL_MIN:
        return "LOW"
    if fill > TARGET_FILL_MAX:
        return "HIGH"
    return "OK"


def _read_adu_from_fitseq(fitseq_path: Path) -> tuple[float, float, float, int]:
    """Returns (median, std, hot_pixel_pct, data_max)."""
    with fits.open(str(fitseq_path)) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = hdu.data.astype(np.float64)
            median = float(np.median(data))
            std = float(np.std(data))
            data_max = int(data.max())
            hot_pct = (100.0 * np.sum(data > median + 5 * std) / data.size
                       if std > 0 else 0.0)
            return median, std, hot_pct, data_max
    return 0.0, 0.0, 0.0, 0


def _convert_single_to_fitseq(working_dir: Path, file_path: Path, seq_name: str) -> Path:
    """Convert one raw file to FITSEQ via Siril. Returns path to .fit."""
    wdir = Path(working_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    for stale in list(wdir.glob(f"{seq_name}*")) + list(wdir.glob(f"{seq_name}_*")):
        if stale.is_file():
            stale.unlink()
    dest = wdir / f"{seq_name}_0000{file_path.suffix}"
    shutil.copy2(file_path, dest)
    run_siril_script([f"convert {seq_name} -fitseq"], working_dir=str(wdir), timeout=120)
    for ext in (".fit", ".fits"):
        p = wdir / f"{seq_name}{ext}"
        if p.exists():
            return p
    present = ", ".join(sorted(p.name for p in wdir.iterdir() if p.is_file()))
    raise FileNotFoundError(
        f"Siril convert did not produce {seq_name}.fit/.fits. "
        f"Files present: {present or '(none)'}"
    )


def _fmt_float(v: float | None, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if v is not None else "n/a"


# ── Per-frame flat evaluation ──────────────────────────────────────────────────

def _run_single_flat_eval(working_dir: Path, flat_file: Path, label: str) -> FlatEval:
    """Evaluate one flat frame using sensor-native fill fraction."""
    adu_dir = Path(working_dir) / "_adu"
    adu_dir.mkdir(parents=True, exist_ok=True)

    median_adu: int | None = None
    std_adu: float | None = None
    white_level: int | None = None

    try:
        fit_path = _convert_single_to_fitseq(adu_dir, flat_file, "flat_adu")
        median, std, _, data_max = _read_adu_from_fitseq(fit_path)
        median_adu = int(round(median))
        std_adu = float(std)
        white_level = infer_white_level(data_max)
    except Exception as e:
        return FlatEval(
            name=label, state="UNKNOWN", fill_pct=None,
            median_adu=None, std_adu=None,
            black_level=None, white_level=None,
            distance_to_target=None,
            error=str(e),
        )

    # Single ExifTool call via _sensor.py for black_level + exposure_time
    frame = read_frame_exif(flat_file)
    black_level = frame.sensor.black_level
    exposure_time = frame.exposure_time

    fill = compute_fill(median_adu, black_level, white_level)
    state = flat_fill_state(fill, median_adu, std_adu, white_level)
    dist = abs(fill - TARGET_FILL_CENTER)

    return FlatEval(
        name=label,
        state=state,
        fill_pct=fill,
        median_adu=median_adu,
        std_adu=std_adu,
        black_level=black_level,
        white_level=white_level,
        distance_to_target=dist,
        exposure_time=exposure_time,
    )


# ── Bias test evaluation ───────────────────────────────────────────────────────

def _run_bias_test(working_dir: Path, bias_files: list[Path]) -> list[BiasEval]:
    """Convert each bias, read ADU, rank by quality (lowest std = best)."""
    results: list[BiasEval] = []
    wd = working_dir / "bias_test"
    wd.mkdir(parents=True, exist_ok=True)

    for bf in bias_files:
        out_dir = wd / bf.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            fit_path = _convert_single_to_fitseq(out_dir, bf, "bias")
            median, std, hot_pct, _ = _read_adu_from_fitseq(fit_path)
            results.append(BiasEval(name=bf.name, median_adu=median, std_adu=std,
                                    hot_pixel_pct=hot_pct))
        except Exception as e:
            results.append(BiasEval(name=bf.name, median_adu=0.0, std_adu=0.0,
                                    hot_pixel_pct=0.0, error=str(e)))

    valid = [r for r in results if r.error is None]
    if valid:
        for rank, r in enumerate(sorted(valid, key=lambda r: (r.std_adu, r.hot_pixel_pct)), 1):
            r.rank = rank
    return results


# ── Folder aggregate (T02) ─────────────────────────────────────────────────────

def _run_t02_bias(working_dir: Path, bias_files: list[Path]) -> tuple[str, float | None]:
    result = build_masters.invoke({
        "working_dir": str(working_dir),
        "file_type": "bias",
        "input_files": [str(p.resolve()) for p in bias_files],
    })
    return result["master_path"], result["diagnostics"]["quality_flags"].get("median")


def _run_t02_folder_flat(
    working_dir: Path,
    flat_files: list[Path],
    master_bias_path: str,
    label: str,
) -> FlatEval:
    """Run T02 on a group of flats. Reports Siril-normalized median + sensor thresholds."""
    try:
        result = build_masters.invoke({
            "working_dir": str(working_dir),
            "file_type": "flat",
            "input_files": [str(p.resolve()) for p in flat_files],
            "master_bias_path": master_bias_path,
        })
        diag = result["diagnostics"]
        qf = diag["quality_flags"]
        siril_norm = qf.get("flat_median_normalized")
        norm_min = qf.get("flat_norm_threshold_min")
        norm_max = qf.get("flat_norm_threshold_max")
        return FlatEval(
            name=label,
            state="n/a",
            fill_pct=None,
            median_adu=None,
            std_adu=None,
            black_level=qf.get("sensor_black"),
            white_level=qf.get("sensor_white"),
            distance_to_target=None,
            siril_norm_median=float(siril_norm) if siril_norm is not None else None,
            siril_norm_min=float(norm_min) if norm_min is not None else None,
            siril_norm_max=float(norm_max) if norm_max is not None else None,
            warnings=list(diag["warnings"]),
        )
    except Exception as e:
        return FlatEval(
            name=label, state="ERROR", fill_pct=None,
            median_adu=None, std_adu=None, black_level=None, white_level=None,
            distance_to_target=None, error=str(e),
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Flat and bias quality checker")
    parser.add_argument("--flats", type=Path, default=None,
                        help="Single flat file or folder of flats to evaluate.")
    parser.add_argument("--biases", type=Path,
                        default=project_root / "test_images" / "bias",
                        help="Bias file or folder used to build master bias.")
    parser.add_argument("--bias-test", type=Path, default=None,
                        help="Folder of bias files to evaluate individually.")
    parser.add_argument("--working-dir", type=Path,
                        default=project_root / "manual_pipeline_output" / "flat_quality_check",
                        help="Working directory for Siril outputs.")
    parser.add_argument("--skip-individual", action="store_true",
                        help="Skip per-flat individual checks.")
    parser.add_argument("--skip-folder", action="store_true",
                        help="Skip aggregate folder check.")
    args = parser.parse_args()

    wd = args.working_dir.resolve()
    wd.mkdir(parents=True, exist_ok=True)

    # ── Bias test mode ─────────────────────────────────────────────────────────
    if args.bias_test is not None:
        bias_files = _collect_images(args.bias_test.resolve())
        if not bias_files:
            print(f"No bias files found at: {args.bias_test}")
            return 1

        print("=" * 72)
        print("Bias Quality Check (per-frame ADU)")
        print("=" * 72)
        print(f"Bias folder: {args.bias_test} ({len(bias_files)} file(s))")
        print(f"Work:        {wd}")
        print("-" * 72)
        print("  Good bias: low median (near sensor black level), low std (read noise)")
        print("  Bad  bias: high or maxed-out median, zero std (not a real bias frame)")
        print("-" * 72)

        results = _run_bias_test(wd, bias_files)
        print(f"{'Bias':20} {'Median ADU':>10} {'Std ADU':>10} {'Hot %':>8} {'Rank':>6}")
        print("-" * 72)
        for r in results:
            if r.error:
                print(f"{r.name[:20]:20}   error: {r.error}")
            else:
                rank_str = f"#{r.rank}" if r.rank else "-"
                print(f"{r.name[:20]:20} {r.median_adu:10.1f} {r.std_adu:10.2f} "
                      f"{r.hot_pixel_pct:8.4f} {rank_str:>6}")

        best = next((r for r in results if r.rank == 1), None)
        if best:
            print("-" * 72)
            print(f"Best: {best.name} (lowest read noise std={best.std_adu:.2f})")
        print("\nDone.")
        return 0

    # ── Flat mode ──────────────────────────────────────────────────────────────
    if args.flats is None:
        print("Error: provide --flats or --bias-test")
        return 1

    flats = _collect_images(args.flats.resolve())
    biases = _collect_images(args.biases.resolve())

    if not flats:
        print(f"No flat files found at: {args.flats}")
        return 1
    if not biases:
        print(f"No bias files found at: {args.biases}")
        return 1

    print("=" * 72)
    print("Flat Quality Check")
    print("=" * 72)
    print(f"Flats:  {args.flats} ({len(flats)} file(s))")
    print(f"Biases: {args.biases} ({len(biases)} file(s))")
    print(f"Work:   {wd}")
    print("-" * 72)
    print(f"Target fill: {TARGET_FILL_MIN*100:.0f}–{TARGET_FILL_MAX*100:.0f}% of usable sensor range")
    print(f"  (fill% = (median_adu - black_level) / (white_level - black_level))")
    print(f"  States: USABLE={TARGET_FILL_MIN*100:.0f}–{TARGET_FILL_MAX*100:.0f}%  "
          f"UNDER=<{TARGET_FILL_MIN*100:.0f}%  OVER=>{TARGET_FILL_MAX*100:.0f}%  "
          f"SATURATED=clipped")

    # Build master bias (needed for folder aggregate check)
    bias_master_dir = wd / "bias_master"
    bias_master_dir.mkdir(parents=True, exist_ok=True)
    master_bias_path, _ = _run_t02_bias(bias_master_dir, biases)

    # ── Individual flat checks ─────────────────────────────────────────────────
    if not args.skip_individual:
        print("\n[Individual Flat Checks]")
        results: list[FlatEval] = []
        for f in flats:
            single_dir = wd / "single" / f.stem
            single_dir.mkdir(parents=True, exist_ok=True)
            ev = _run_single_flat_eval(single_dir, f, f.name)
            results.append(ev)

        # Rank by closeness to target center fill
        ranked = sorted(
            results,
            key=lambda r: (r.distance_to_target if r.distance_to_target is not None else 999.0),
        )
        rank_map = {r.name: i + 1 for i, r in enumerate(ranked)}

        # Show ADU target range derived from the first valid frame's sensor levels
        first_valid = next((r for r in results if r.white_level is not None), None)
        if first_valid and first_valid.black_level is not None:
            bl = first_valid.black_level
            wl = first_valid.white_level
            adu_lo, adu_hi, adu_ideal = flat_adu_range(bl, wl)
            print(f"Sensor:  black={bl}  white={wl}  usable={wl - bl} ADU")
            print(f"Target ADU range: {adu_lo}–{adu_hi}  (ideal center ~{adu_ideal})")

        print()
        print(f"{'Flat':24} {'Shutter':8} {'State':10} {'Fill%':6} {'Target':6} "
              f"{'Median ADU':>10} {'Std ADU':>9} {'Rank':>5}")
        print("-" * 80)
        for ev in results:
            fill_str = f"{ev.fill_pct*100:.1f}%" if ev.fill_pct is not None else "n/a"
            target_lbl = _fill_target_label(ev.fill_pct)
            adu_str = str(ev.median_adu) if ev.median_adu is not None else "n/a"
            std_str = _fmt_float(ev.std_adu, 1)
            rank = rank_map.get(ev.name, 0)
            exp_str = _fmt_exposure(ev.exposure_time)
            print(f"{ev.name[:24]:24} {exp_str:>8} {ev.state:10} {fill_str:>6} {target_lbl:6} "
                  f"{adu_str:>10} {std_str:>9} {rank:>5}")
            if ev.error:
                print(f"  error: {ev.error}")
            for w in ev.warnings:
                print(f"  warn: {w}")

        usable = [r for r in results if r.state == "USABLE"]
        print("-" * 80)
        print(f"Usable: {len(usable)}/{len(results)}")

        best = ranked[0] if ranked else None
        if best is not None:
            fill_str = f"{best.fill_pct*100:.1f}%" if best.fill_pct is not None else "n/a"
            print(f"Closest to target ({TARGET_FILL_CENTER*100:.1f}%): "
                  f"{best.name}  fill={fill_str}  state={best.state}  ADU={best.median_adu}")

    # ── Folder aggregate check (T02) ───────────────────────────────────────────
    if not args.skip_folder and len(flats) > 1:
        print("\n[Folder Aggregate — Siril master flat]")
        folder_dir = wd / "folder_aggregate"
        folder_dir.mkdir(parents=True, exist_ok=True)
        ev = _run_t02_folder_flat(folder_dir, flats, master_bias_path, "ALL_FLATS")
        print(f"Siril normalized median: {_fmt_float(ev.siril_norm_median)}")
        if ev.siril_norm_min is not None:
            print(f"Sensor-relative target:  [{ev.siril_norm_min:.4f}, {ev.siril_norm_max:.4f}]"
                  f"  (= 30–55% fill, sensor black={ev.black_level} white={ev.white_level})")
            if ev.siril_norm_median is not None:
                in_range = ev.siril_norm_min <= ev.siril_norm_median <= ev.siril_norm_max
                print(f"T02 HITL triggered: {'NO — within range' if in_range else 'YES — outside range'}")
        if ev.error:
            print(f"Error: {ev.error}")
        for w in ev.warnings:
            print(f"Warn: {w}")
    elif not args.skip_folder:
        print("\n[Folder Aggregate] Skipped: need >1 flat file.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
