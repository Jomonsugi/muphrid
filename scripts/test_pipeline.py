#!/usr/bin/env python3
"""
Test the preprocessing pipeline tool-by-tool WITHOUT the LLM.

Calls the underlying Python functions directly (bypassing LangChain tool
wrappers and InjectedState) to isolate tool bugs from model issues.

Usage:
    uv run python scripts/test_pipeline.py --dataset /path/to/dataset --target "M20"
    uv run python scripts/test_pipeline.py --dataset /path/to/dataset --target "M42" --bortle 6
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Test pipeline tool-by-tool")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--bortle", type=int, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    print(f"{'='*60}")
    print(f"PIPELINE TEST: {args.target}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}")

    # ── Step 1: Ingest ──
    print(f"\n--- INGEST ---")
    from astro_agent.tools.preprocess.t01_ingest import ingest_dataset
    ingest_result = ingest_dataset.invoke({
        "root_directory": str(dataset_path),
        "thread_id": "test-pipeline",
    })
    dataset = ingest_result["dataset"]
    summary = ingest_result["summary"]
    meta = dataset.get("acquisition_meta", {})

    print(f"  Format: {summary.get('input_format')}")
    print(f"  Lights: {summary.get('lights_count')}, Darks: {summary.get('darks_count')}")
    print(f"  Flats: {summary.get('flats_count')}, Biases: {summary.get('biases_count')}")
    print(f"  Camera: {meta.get('camera_model')}")
    print(f"  Sensor: {meta.get('sensor_type')}, Bit depth: {meta.get('bit_depth')}")
    print(f"  Exposure: {meta.get('exposure_time_s')}s, Gain: {meta.get('gain')}")
    for w in ingest_result.get("warnings", []):
        print(f"  WARN: {w}")
    print(f"  ✓ Ingest OK")

    # Build state
    from astro_agent.graph.state import make_empty_state
    session = {
        "target_name": args.target,
        "bortle": args.bortle,
        "sqm_reading": None,
        "remove_stars": None,
        "notes": None,
    }
    state = make_empty_state(dataset=dataset, session=session)

    working_dir = dataset["working_dir"]

    # ── Step 2: Convert sequence ──
    print(f"\n--- CONVERT SEQUENCE ---")
    from astro_agent.tools.preprocess.t02b_convert_sequence import _convert_to_sequence
    try:
        conv_result = _convert_to_sequence(
            working_dir=working_dir,
            input_files=dataset["files"]["lights"],
            sequence_name="lights",
            debayer=False,
        )
        state["paths"]["lights_sequence"] = conv_result["sequence_name"]
        print(f"  Sequence: {conv_result['sequence_name']}")
        print(f"  Frames: {conv_result['frame_count']}")
        print(f"  FITSEQ: {conv_result.get('fitseq_path')}")
        print(f"  ✓ Convert OK")
    except Exception as e:
        print(f"  ✗ Convert FAILED: {e}")
        return 1

    # ── Step 3: Calibrate ──
    print(f"\n--- CALIBRATE ---")
    from astro_agent.tools._siril import run_siril_script

    sensor_type = meta.get("sensor_type", "bayer")
    is_xtrans = sensor_type == "xtrans"

    cal_parts = [f"calibrate lights"]
    masters = state["paths"]["masters"]
    if masters.get("bias"):
        cal_parts.append(f"-bias={Path(masters['bias']).stem}")
    if masters.get("dark"):
        cal_parts.append(f"-dark={Path(masters['dark']).stem}")
    if masters.get("flat"):
        cal_parts.append(f"-flat={Path(masters['flat']).stem}")
    cal_parts.append("-cfa -debayer")
    if is_xtrans:
        cal_parts.append("-equalize_cfa -fix_xtrans")

    cal_cmd = " ".join(cal_parts)
    print(f"  Command: {cal_cmd}")

    try:
        result = run_siril_script([cal_cmd], working_dir=working_dir, timeout=600)
        state["paths"]["calibrated_sequence"] = "pp_lights"
        print(f"  ✓ Calibrate OK")
    except Exception as e:
        print(f"  ✗ Calibrate FAILED: {e}")
        return 1

    # ── Step 4: Register ──
    print(f"\n--- REGISTER ---")
    reg_cmd = "register pp_lights -2pass -transf=homography -maxstars=500"
    apply_cmd = "seqapplyreg pp_lights -framing=min -interp=lanczos4"

    try:
        run_siril_script([reg_cmd], working_dir=working_dir, timeout=900)
        run_siril_script([apply_cmd], working_dir=working_dir, timeout=900)
        state["paths"]["registered_sequence"] = "r_pp_lights"
        print(f"  ✓ Register OK")
    except Exception as e:
        print(f"  ✗ Register FAILED: {e}")
        return 1

    # ── Step 5: Stack ──
    print(f"\n--- STACK ---")
    stack_cmd = "stack r_pp_lights rej s 3 3 -norm=addscale -output_norm -weight=wfwhm -32b -out=master_light"

    try:
        run_siril_script([stack_cmd], working_dir=working_dir, timeout=1200)
        master = Path(working_dir) / "master_light.fit"
        if not master.exists():
            master = Path(working_dir) / "master_light.fits"
        if master.exists():
            state["paths"]["current_image"] = str(master)
            size_mb = master.stat().st_size / (1024**2)
            print(f"  Master light: {master.name} ({size_mb:.0f}MB)")
            print(f"  ✓ Stack OK")
        else:
            print(f"  ✗ Stack produced no output file")
            return 1
    except Exception as e:
        print(f"  ✗ Stack FAILED: {e}")
        return 1

    # ── Step 6: Analyze ──
    print(f"\n--- ANALYZE IMAGE ---")
    from astro_agent.tools.utility.t20_analyze import (
        _load_fits_float32, _trim_zero_borders, _background_estimate,
        _robust_stats, _detect_stars_full,
    )
    import numpy as np

    try:
        data, _ = _load_fits_float32(master)
        data = _trim_zero_borders(data)
        if data.ndim == 3 and data.shape[0] == 3:
            lum = (0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]).astype(np.float32)
        else:
            lum = data.squeeze().astype(np.float32)

        valid = (lum > 0) & (lum < 0.999)
        valid_pct = np.sum(valid) / lum.size * 100
        bg = _background_estimate(lum)
        stars = _detect_stars_full(lum, bg["bg_level"], bg["bg_noise"])

        print(f"  Valid pixels: {valid_pct:.1f}%")
        print(f"  Background: {bg['bg_level']:.6f}")
        print(f"  Noise: {bg['bg_noise']:.6f}")
        print(f"  Stars: {stars['count']}, FWHM: {stars.get('median_fwhm')}")
        print(f"  ✓ Analyze OK")
    except Exception as e:
        print(f"  ✗ Analyze FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"  ALL STEPS PASSED")
    print(f"  Master light: {state['paths']['current_image']}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
