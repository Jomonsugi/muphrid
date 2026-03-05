#!/usr/bin/env python3
"""
Phase 6 — Manual Pipeline Integration Script

A plain Python script that calls T01 → T24 sequentially on a real dataset.
Every intermediate file exists on disk. Demonstrates the masked-application
pattern: T25 → T27 → T23 on the starless image.

Run from project root:
    uv run python scripts/run_manual_pipeline.py [--dataset PATH] [--output DIR]

Requirements:
  - test_images/ (or --dataset) with lights/, darks/, flats/, bias/
  - Siril, GraXpert, StarNet, ExifTool installed and configured
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv

load_dotenv(project_root / ".env")
load_dotenv(project_root / "astro_agent" / ".env")

from astro_agent.tools.linear.t09_gradient import remove_gradient
from astro_agent.tools.linear.t10_color_calibrate import color_calibrate
from astro_agent.tools.linear.t11_green_noise import remove_green_noise
from astro_agent.tools.linear.t12_noise_reduction import noise_reduction
from astro_agent.tools.linear.t13_deconvolution import deconvolution
from astro_agent.tools.nonlinear.t14_stretch import stretch_image
from astro_agent.tools.nonlinear.t15_star_removal import star_removal
from astro_agent.tools.nonlinear.t16_curves import curves_adjust
from astro_agent.tools.nonlinear.t17_local_contrast import local_contrast_enhance
from astro_agent.tools.nonlinear.t18_saturation import saturation_adjust
from astro_agent.tools.nonlinear.t19_star_restoration import star_restoration
from astro_agent.tools.preprocess.t01_ingest import ingest_dataset
from astro_agent.tools.preprocess.t02_masters import build_masters
from astro_agent.tools.preprocess.t02b_convert_sequence import convert_sequence
from astro_agent.tools.preprocess.t03_calibrate import siril_calibrate
from astro_agent.tools.preprocess.t04_register import siril_register
from astro_agent.tools.preprocess.t05_analyze_frames import analyze_frames
from astro_agent.tools.preprocess.t06_select_frames import select_frames
from astro_agent.tools.preprocess.t07_stack import siril_stack
from astro_agent.tools.preprocess.t08_crop import auto_crop
from astro_agent.tools.scikit.t25_create_mask import create_mask
from astro_agent.tools.scikit.t27_multiscale import multiscale_process
from astro_agent.tools.utility.t23_pixel_math import pixel_math
from astro_agent.tools.utility.t24_export import export_final


def _step(name: str, fn, **kwargs):
    print(f"  [{name}] ...")
    result = fn.invoke(kwargs)
    print(f"  [{name}] ok")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 manual pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "test_images",
        help="Dataset root with lights/, darks/, flats/, bias/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "manual_pipeline_output",
        help="Working directory for pipeline outputs",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    working_dir = args.output.resolve()

    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}")
        return 1

    print("\n" + "=" * 60)
    print("Phase 6 — Manual Pipeline")
    print("=" * 60)
    print(f"Dataset:      {dataset_path}")
    print(f"Working dir:  {working_dir}")
    print("=" * 60)

    working_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # T01 — Ingest
    # -------------------------------------------------------------------------
    r01 = _step(
        "T01 ingest",
        ingest_dataset,
        root_directory=str(dataset_path),
        override_target_name="ManualPipeline",
    )
    dataset = r01["dataset"]
    files = dataset["files"]
    meta = dataset.get("acquisition_meta", {})
    focal_mm = meta.get("focal_length_mm") or 0.0
    camera_model = meta.get("camera_model")

    # -------------------------------------------------------------------------
    # T02b — Convert lights to sequence (needed before T03)
    # -------------------------------------------------------------------------
    r02b = _step(
        "T02b convert_lights",
        convert_sequence,
        working_dir=str(working_dir),
        input_files=files["lights"],
        sequence_name="lights_seq",
        debayer=False,
    )

    # -------------------------------------------------------------------------
    # T02 — Build masters (bias, dark, flat)
    # -------------------------------------------------------------------------
    r02_bias = _step(
        "T02 bias",
        build_masters,
        working_dir=str(working_dir),
        file_type="bias",
        input_files=files.get("biases", []),
    )
    master_bias = r02_bias["master_path"]
    if r02_bias["diagnostics"]["hitl_required"]:
        print(f"  [T02 bias] HITL: {r02_bias['diagnostics']['hitl_context']}")

    r02_dark = _step(
        "T02 dark",
        build_masters,
        working_dir=str(working_dir),
        file_type="dark",
        input_files=files.get("darks", []),
    )
    master_dark = r02_dark["master_path"]
    if r02_dark["diagnostics"]["hitl_required"]:
        print(f"  [T02 dark] HITL: {r02_dark['diagnostics']['hitl_context']}")

    r02_flat = _step(
        "T02 flat",
        build_masters,
        working_dir=str(working_dir),
        file_type="flat",
        input_files=files.get("flats", []),
        master_bias_path=master_bias,
    )
    master_flat = r02_flat["master_path"]
    if r02_flat["diagnostics"]["hitl_required"]:
        print(f"  [T02 flat] HITL: {r02_flat['diagnostics']['hitl_context']}")

    # -------------------------------------------------------------------------
    # T03 — Calibrate
    # -------------------------------------------------------------------------
    r03 = _step(
        "T03 calibrate",
        siril_calibrate,
        working_dir=str(working_dir),
        lights_sequence="lights_seq",
        master_bias=master_bias,
        master_dark=master_dark,
        master_flat=master_flat,
        is_cfa=True,
        debayer=True,
        equalize_cfa=True,
    )

    # -------------------------------------------------------------------------
    # T04 — Register (framing=min for FITSEQ compatibility)
    # -------------------------------------------------------------------------
    r04 = _step(
        "T04 register",
        siril_register,
        working_dir=str(working_dir),
        calibrated_sequence=r03["calibrated_sequence"],
        framing="min",
    )

    # -------------------------------------------------------------------------
    # T05 — Analyze frames (parses .seq file directly)
    # -------------------------------------------------------------------------
    r05 = _step(
        "T05 analyze_frames",
        analyze_frames,
        working_dir=str(working_dir),
        registered_sequence=r04["registered_sequence"],
    )
    summary = r05["summary"]
    print(f"       FWHM: median={summary.get('median_fwhm')}, "
          f"std={summary.get('std_fwhm')}, "
          f"seeing_stability={summary.get('seeing_stability')}")
    print(f"       Tracking: roundness={summary.get('tracking_quality')}, "
          f"stars={summary.get('median_star_count')}")
    if summary.get("outlier_frames"):
        print(f"       Outlier frames: {summary['outlier_frames']}")

    # -------------------------------------------------------------------------
    # T06 — Select frames
    # -------------------------------------------------------------------------
    r06 = _step(
        "T06 select_frames",
        select_frames,
        frame_metrics=r05["frame_metrics"],
        criteria={},
    )
    accepted = r06["accepted_frames"]

    # -------------------------------------------------------------------------
    # T07 — Stack (1-based select/unselect, fixed FITSEQ parser)
    # -------------------------------------------------------------------------
    r07 = _step(
        "T07 stack",
        siril_stack,
        working_dir=str(working_dir),
        registered_sequence=r04["registered_sequence"],
        accepted_frames=accepted,
    )
    current_image = r07["master_light_path"]

    # -------------------------------------------------------------------------
    # T08 — Crop
    # -------------------------------------------------------------------------
    r08 = _step(
        "T08 crop",
        auto_crop,
        working_dir=str(working_dir),
        image_path=current_image,
    )
    current_image = r08["cropped_image_path"]

    # -------------------------------------------------------------------------
    # T09 — Gradient removal (GraXpert, fixed output path + casing + ai_version)
    # -------------------------------------------------------------------------
    r09 = _step(
        "T09 gradient",
        remove_gradient,
        working_dir=str(working_dir),
        image_path=current_image,
        backend="graxpert",
    )
    current_image = r09["processed_image_path"]

    assert Path(current_image).parent.resolve() == working_dir.resolve(), (
        f"T09 output must be in working_dir for T10: {current_image}"
    )

    # -------------------------------------------------------------------------
    # T10 — Color calibrate
    # -------------------------------------------------------------------------
    r10 = _step(
        "T10 color_calibrate",
        color_calibrate,
        working_dir=str(working_dir),
        image_path=current_image,
        focal_length_mm=focal_mm,
        camera_model=camera_model,
    )
    if not r10.get("plate_solve_success"):
        print("  [T10] WARNING: plate solve failed, continuing without color calibration")
        if r10.get("calibrated_image_path"):
            current_image = r10["calibrated_image_path"]
    else:
        current_image = r10["calibrated_image_path"]

    # -------------------------------------------------------------------------
    # T11 — Green noise (error classifier fixed — no more spurious SirilError)
    # -------------------------------------------------------------------------
    r11 = _step(
        "T11 green_noise",
        remove_green_noise,
        working_dir=str(working_dir),
        image_path=current_image,
    )
    current_image = r11["cleaned_image_path"]

    # -------------------------------------------------------------------------
    # T12 — Noise reduction (error classifier fixed)
    # -------------------------------------------------------------------------
    r12 = _step(
        "T12 noise_reduction",
        noise_reduction,
        working_dir=str(working_dir),
        image_path=current_image,
    )
    current_image = r12["denoised_image_path"]

    # -------------------------------------------------------------------------
    # T13 — Deconvolution
    # -------------------------------------------------------------------------
    r13 = _step(
        "T13 deconvolution",
        deconvolution,
        working_dir=str(working_dir),
        image_path=current_image,
    )
    current_image = r13["processed_image_path"]

    # -------------------------------------------------------------------------
    # T14 — Stretch
    # -------------------------------------------------------------------------
    r14 = _step(
        "T14 stretch",
        stretch_image,
        working_dir=str(working_dir),
        image_path=current_image,
        method="ghs",
        ghs_options={
            "stretch_amount": 2.5,
            "highlight_protection": 0.95,
        },
    )
    current_image = r14["stretched_image_path"]

    # -------------------------------------------------------------------------
    # T15 — Star removal
    # -------------------------------------------------------------------------
    r15 = _step(
        "T15 star_removal",
        star_removal,
        working_dir=str(working_dir),
        image_path=current_image,
        generate_star_mask=True,
    )
    starless_image = r15["starless_image_path"]
    star_mask_path = r15["star_mask_path"]

    # -------------------------------------------------------------------------
    # Masked-application pattern: T25 → T27 → T23
    # T23 now auto-broadcasts 1-layer mask to 3-layer RGB via rgbcomp
    # -------------------------------------------------------------------------
    starless_stem = Path(starless_image).stem

    r25 = _step(
        "T25 create_mask",
        create_mask,
        working_dir=str(working_dir),
        image_path=starless_image,
        mask_type="luminance",
        luminance_options={"low": 0.15, "high": 0.85},
        feather_radius=8.0,
    )
    mask_path = r25["mask_path"]
    mask_stem = Path(mask_path).stem

    r27 = _step(
        "T27 multiscale_process",
        multiscale_process,
        working_dir=str(working_dir),
        image_path=starless_image,
        num_scales=5,
        scale_operations=[
            {"scale": 1, "operation": "suppress"},
            {"scale": 2, "operation": "sharpen", "weight": 1.25},
            {"scale": 3, "operation": "sharpen", "weight": 1.15},
        ],
    )
    processed_starless_path = r27["processed_image_path"]
    processed_stem = Path(processed_starless_path).stem

    r23 = _step(
        "T23 pixel_math",
        pixel_math,
        working_dir=str(working_dir),
        expression=f"${processed_stem}$ * ${mask_stem}$ + ${starless_stem}$ * (1 - ${mask_stem}$)",
        output_stem=f"{starless_stem}_masked_sharp",
    )
    processed_starless_image = r23["result_image_path"]
    if r23.get("auto_broadcast"):
        print("  [T23] auto-broadcast: 1-layer mask expanded to 3-layer RGB")

    # -------------------------------------------------------------------------
    # T16 — Curves (light touch)
    # -------------------------------------------------------------------------
    r16 = _step(
        "T16 curves",
        curves_adjust,
        working_dir=str(working_dir),
        image_path=processed_starless_image,
        method="mtf",
        mtf_options={"black_point": 0.0, "midtone": 0.48, "white_point": 1.0},
    )
    current_image = r16["adjusted_image_path"]

    # -------------------------------------------------------------------------
    # T17 — Local contrast
    # -------------------------------------------------------------------------
    r17 = _step(
        "T17 local_contrast",
        local_contrast_enhance,
        working_dir=str(working_dir),
        image_path=current_image,
    )
    current_image = r17["enhanced_image_path"]

    # -------------------------------------------------------------------------
    # T18 — Saturation (amount now has a default; uses processed_image_path)
    # -------------------------------------------------------------------------
    r18 = _step(
        "T18 saturation",
        saturation_adjust,
        working_dir=str(working_dir),
        image_path=current_image,
        amount=0.5,
    )
    current_image = r18["processed_image_path"]

    # -------------------------------------------------------------------------
    # T19 — Star restoration
    # -------------------------------------------------------------------------
    r19 = _step(
        "T19 star_restoration",
        star_restoration,
        working_dir=str(working_dir),
        starless_image_path=current_image,
        mode="blend",
        blend_options={"star_mask_path": star_mask_path, "star_weight": 1.0},
    )
    current_image = r19["final_image_path"]

    # -------------------------------------------------------------------------
    # T24 — Export (icc_assign now inserted automatically before icc_convert_to)
    # -------------------------------------------------------------------------
    r24 = _step(
        "T24 export",
        export_final,
        working_dir=str(working_dir),
        image_path=current_image,
        source_profile="sRGBlinear",
    )
    exported = r24["exported_files"]

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("Exported files:")
    for f in exported:
        print(f"  {f.get('path', f)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
