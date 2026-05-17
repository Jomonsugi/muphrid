#!/usr/bin/env python3
"""
Focused tests for acquisition metadata as state.

Run from project root:
    uv run python scripts/test_acquisition_state_contract.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from muphrid.graph.state import _merge_dicts, build_initial_message, make_empty_state
from muphrid.tools.preprocess import ingest as ingest_mod


_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    msg = f" {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


class _FakeExifToolHelper:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get_metadata(self, _path: str):
        return [{
            "FITS:Exptime": 120.0,
            "FITS:Gain": 100,
            "FITS:Instrument": "ASI2600MC",
            "FITS:Focallen": 500.0,
            "FITS:Xpixsz": 3.76,
        }]


def test_fits_input_format_is_preserved() -> None:
    original_helper = ingest_mod.exiftool.ExifToolHelper
    ingest_mod.exiftool.ExifToolHelper = _FakeExifToolHelper
    try:
        light = Path(tempfile.mkdtemp(prefix="acq_state_")) / "light.fit"
        light.write_text("placeholder")
        meta = ingest_mod._extract_acquisition_meta(  # noqa: SLF001 - contract test
            [light],
            override_target_name="M42",
            input_format="fits",
        )
    finally:
        ingest_mod.exiftool.ExifToolHelper = original_helper

    check("FITS metadata keeps input_format=fits", meta.get("input_format") == "fits")
    dataset = {
        "id": "test-dataset",
        "working_dir": str(light.parent),
        "files": {"lights": [str(light)], "darks": [], "flats": [], "biases": []},
        "acquisition_meta": meta,
    }
    state = make_empty_state(
        dataset=dataset,
        session={
            "target_name": "M42",
            "bortle": None,
            "sqm_reading": None,
            "remove_stars": None,
            "notes": None,
        },
    )
    check("FITS runs are not marked OSC by default", state["metadata"]["is_osc"] is False)


def test_fits_ingest_writes_fits_format() -> None:
    from astropy.io import fits as astropy_fits

    root = Path(tempfile.mkdtemp(prefix="acq_fits_ingest_"))
    lights = root / "lights"
    lights.mkdir()
    light = lights / "light.fit"
    hdu = astropy_fits.PrimaryHDU()
    hdu.header["IMAGETYP"] = "LIGHT"
    hdu.writeto(light)

    original_helper = ingest_mod.exiftool.ExifToolHelper
    ingest_mod.exiftool.ExifToolHelper = _FakeExifToolHelper
    try:
        dataset, _warnings, summary = ingest_mod._ingest_fits(  # noqa: SLF001 - contract test
            root,
            file_pattern=None,
            override_target_name="M42",
            thread_id="fits-contract",
        )
    finally:
        ingest_mod.exiftool.ExifToolHelper = original_helper

    meta = dataset["acquisition_meta"]
    check("FITS ingest stores acquisition_meta.input_format", meta.get("input_format") == "fits")
    check("FITS ingest summary reports fits", summary.get("input_format") == "fits")


def test_initial_message_uses_acquisition_state_only() -> None:
    dataset = {
        "id": "test-dataset",
        "working_dir": tempfile.mkdtemp(prefix="acq_prompt_"),
        "files": {"lights": ["l1.fit"], "darks": [], "flats": [], "biases": []},
        "acquisition_meta": {
            "target_name": "M42",
            "target_coords": None,
            "focal_length_mm": 500.0,
            "pixel_size_um": 3.76,
            "exposure_time_s": 120.0,
            "iso": None,
            "gain": 100,
            "filter": None,
            "bortle": None,
            "camera_model": "ASI2600MC",
            "telescope": None,
            "input_format": "fits",
            "black_level": 0,
            "white_level": 65535,
            "bit_depth": 16,
            "raw_exposure_bias": None,
            "sensor_type": None,
        },
    }
    message = build_initial_message(
        dataset=dataset,
        session={
            "target_name": "M42",
            "bortle": None,
            "sqm_reading": None,
            "remove_stars": None,
            "notes": None,
        },
        ingest_summary={"detected_extensions": [".fit"]},
    )
    check("prompt includes state focal length", "- Focal length: 500.0mm" in message)
    check("prompt includes state input format", "- Input format: FITS" in message)
    check("prompt does not reread equipment.toml", "Equipment Profile" not in message)


def test_dataset_acquisition_deltas_compose() -> None:
    base = {
        "id": "test-dataset",
        "working_dir": "/tmp/muphrid",
        "files": {"lights": [], "darks": [], "flats": [], "biases": []},
        "acquisition_meta": {
            "target_name": "M42",
            "target_coords": None,
            "focal_length_mm": None,
            "pixel_size_um": 3.76,
        },
    }
    with_coords = _merge_dicts(
        base,
        {"acquisition_meta": {"target_coords": {"ra": 83.8, "dec": -5.4}}},
    )
    with_focal_length = _merge_dicts(
        with_coords,
        {"acquisition_meta": {"focal_length_mm": 500.0}},
    )
    meta = with_focal_length["acquisition_meta"]
    check("dataset reducer keeps target coords delta", meta.get("target_coords", {}).get("ra") == 83.8)
    check("dataset reducer keeps focal length delta", meta.get("focal_length_mm") == 500.0)
    check("dataset reducer preserves sibling pixel size", meta.get("pixel_size_um") == 3.76)


def main() -> int:
    print("Acquisition state contract tests")
    test_fits_input_format_is_preserved()
    test_fits_ingest_writes_fits_format()
    test_initial_message_uses_acquisition_state_only()
    test_dataset_acquisition_deltas_compose()
    if _failures:
        print(f"\n{len(_failures)} failure(s): {', '.join(_failures)}")
        return 1
    print("\nAll acquisition state contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
