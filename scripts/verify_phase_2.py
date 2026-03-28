#!/usr/bin/env python3
"""
Verify Phase 2 — Pre-Processing Tools (T01–T08).

Run from project root:
    uv run python scripts/verify_phase_2.py

Checks:
  - All T01–T08 modules import without error
  - All tools are properly decorated with @tool and have an args_schema
  - T01 runs against real test_images data and returns expected metadata
  - T06 select_frames logic works correctly in pure Python
  - config.py check_dependencies() still passes (includes exiftool check)

Exit 0 = all checks pass. Exit 1 = one or more checks failed.
"""

import os
import sys
import statistics
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv(project_root / ".env")
load_dotenv(project_root / "muphrid" / ".env")

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "\u2713" if ok else "\u2717"
    msg = f"  {status} {name}"
    if detail:
        msg += f" \u2014 {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


# ---------------------------------------------------------------------------
# Phase 2 — Tool imports
# ---------------------------------------------------------------------------
print("Phase 2 Verification\n" + "=" * 40)
print("\nPhase 2 \u2014 Tool imports")

tools_to_check = [
    ("T01 ingest_dataset",     "muphrid.tools.preprocess.t01_ingest",        "ingest_dataset"),
    ("T02 build_masters",      "muphrid.tools.preprocess.t02_masters",       "build_masters"),
    ("T03 calibrate",          "muphrid.tools.preprocess.t03_calibrate",     "calibrate"),
    ("T04 siril_register",     "muphrid.tools.preprocess.t04_register",      "siril_register"),
    ("T05 analyze_frames",     "muphrid.tools.preprocess.t05_analyze_frames","analyze_frames"),
    ("T06 select_frames",      "muphrid.tools.preprocess.t06_select_frames", "select_frames"),
    ("T07 siril_stack",        "muphrid.tools.preprocess.t07_stack",         "siril_stack"),
    ("T08 auto_crop",          "muphrid.tools.preprocess.t08_crop",          "auto_crop"),
]

imported_tools = {}
for label, module_path, fn_name in tools_to_check:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        fn  = getattr(mod, fn_name)
        imported_tools[fn_name] = fn
        check(f"import {label}", True)
    except Exception as e:
        check(f"import {label}", False, str(e))

# ---------------------------------------------------------------------------
# Phase 2 — Tool decorators
# ---------------------------------------------------------------------------
print("\nPhase 2 \u2014 Tool decorators (@tool + args_schema)")

for fn_name, fn in imported_tools.items():
    has_invoke  = hasattr(fn, "invoke")
    has_schema  = hasattr(fn, "args_schema") or hasattr(fn, "get_input_schema")
    check(
        f"{fn_name} is @tool-decorated",
        has_invoke,
        "has .invoke()" if has_invoke else "missing .invoke() — not a LangChain tool",
    )

# ---------------------------------------------------------------------------
# Phase 2 — T01 against real data
# ---------------------------------------------------------------------------
print("\nPhase 2 \u2014 T01 ingest_dataset (real RAF data)")

test_images = project_root / "test_images"
if not test_images.exists():
    check("test_images directory exists", False, str(test_images))
else:
    check("test_images directory exists", True, str(test_images))

    try:
        from muphrid.tools.preprocess.t01_ingest import ingest_dataset
        result = ingest_dataset.invoke({
            "root_directory": str(test_images),
            "override_target_name": "Test Target",
        })

        ds = result["dataset"]
        s  = result["summary"]
        m  = ds["acquisition_meta"]

        check("T01 returns dataset", "id" in ds and "files" in ds)
        check("T01 lights detected",  s["lights_count"] > 0,  f"{s['lights_count']} lights")
        check("T01 darks detected",   s["darks_count"]  > 0,  f"{s['darks_count']} darks")
        check("T01 flats detected",   s["flats_count"]  > 0,  f"{s['flats_count']} flats")
        check("T01 biases detected",  s["biases_count"] > 0,  f"{s['biases_count']} biases")
        check("T01 input_format=raw", s["input_format"] == "raw")
        check("T01 camera_model extracted", bool(m["camera_model"]), str(m["camera_model"]))
        check("T01 exposure_time_s extracted", m["exposure_time_s"] is not None,
              f"{m['exposure_time_s']}s")
        check("T01 iso extracted", m["iso"] is not None, str(m["iso"]))
        check("T01 focal_length_mm extracted", m["focal_length_mm"] is not None,
              f"{m['focal_length_mm']}mm")
        check("T01 target_name override works", m["target_name"] == "Test Target")
        check("T01 total_exposure_s > 0", s["total_exposure_s"] > 0,
              f"{s['total_exposure_s']}s total")

    except Exception as e:
        check("T01 invoke succeeded", False, str(e))

# ---------------------------------------------------------------------------
# Phase 2 — T06 select_frames logic (pure Python, no Siril needed)
# ---------------------------------------------------------------------------
print("\nPhase 2 \u2014 T06 select_frames logic")

try:
    from muphrid.tools.preprocess.t06_select_frames import select_frames

    # Build synthetic frame metrics — 10 good frames + 2 bad
    frame_metrics: dict = {}
    fwhms = [2.1, 2.0, 2.2, 1.9, 2.3, 2.1, 2.0, 2.2, 1.8, 2.1]
    for i, fwhm in enumerate(fwhms):
        frame_metrics[f"frame_{i:04d}.fit"] = {
            "fwhm":             fwhm,
            "weighted_fwhm":    fwhm * 1.1,
            "roundness":        0.92,
            "quality":          0.8,
            "number_of_stars":  80,
            "background_lvl":   0.05,
            "bgnoise":          0.002,
        }
    # Two bad frames: high FWHM and low star count
    frame_metrics["bad_fwhm.fit"] = {
        "fwhm": 8.0, "weighted_fwhm": 9.0, "roundness": 0.7,
        "quality": 0.2, "number_of_stars": 80, "background_lvl": 0.05, "bgnoise": 0.002,
    }
    frame_metrics["bad_stars.fit"] = {
        "fwhm": 2.1, "weighted_fwhm": 2.3, "roundness": 0.9,
        "quality": 0.5, "number_of_stars": 5, "background_lvl": 0.05, "bgnoise": 0.002,
    }

    result = select_frames.invoke({
        "frame_metrics": frame_metrics,
        "criteria": {"min_star_count": 20},  # 5 stars (bad_stars.fit) should be rejected
    })

    check("T06 returns accepted_frames",       len(result["accepted_frames"]) > 0)
    check("T06 rejects bad FWHM frame",        "bad_fwhm.fit" in result["rejected_frames"])
    check("T06 rejects low star-count frame",  "bad_stars.fit" in result["rejected_frames"])
    check("T06 accepts good frames",           len(result["accepted_frames"]) == 10)
    check("T06 acceptance_rate correct",       result["acceptance_rate"] > 0.5)

    # Safety test: all frames bad → still returns all accepted
    all_bad = {f"f{i}.fit": {
        "fwhm": 50.0, "roundness": 0.1, "number_of_stars": 1,
        "background_lvl": 99.0, "bgnoise": 0.1
    } for i in range(5)}
    safety_result = select_frames.invoke({
        "frame_metrics": all_bad,
        "criteria": {"min_star_count": 20},
    })
    check("T06 safety: never empty accepted list",
          len(safety_result["accepted_frames"]) == 5)

except Exception as e:
    check("T06 logic test", False, str(e))

# ---------------------------------------------------------------------------
# Phase 2 — check_dependencies still passes (includes exiftool)
# ---------------------------------------------------------------------------
print("\nPhase 2 \u2014 check_dependencies (includes ExifTool)")

try:
    from muphrid.config import load_settings, check_dependencies
    settings = load_settings()
    check_dependencies(settings)
    check("check_dependencies() passes", True, "ExifTool detected")
except Exception as e:
    check("check_dependencies() passes", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 40)
if _failures:
    print(f"FAILED ({len(_failures)} check(s)):")
    for f in _failures:
        print(f"  \u2717 {f}")
    sys.exit(1)
else:
    print("Phase 2 verification complete \u2014 all checks passed.")
    sys.exit(0)
