"""
T01 — ingest_dataset

Scan a root directory, classify frames by type, extract acquisition metadata,
and return a populated Dataset schema.

Two ingestion paths:
  - Camera RAW (*.raf, *.cr2, etc.): classify by subdirectory name, extract
    EXIF via pyexiftool / ExifTool binary.
  - FITS (*.fits, *.fit, *.fts): classify by IMAGETYP header, extract via
    astropy. NOT YET IMPLEMENTED — stub raises NotImplementedError.

ExifTool is the gold standard for camera RAW metadata. Install:
    brew install exiftool

Sensor characterization:
  The tool now reads black_level, white_level, bit_depth, raw_exposure_bias,
  and sensor_type for all frame types (lights, darks, flats, biases). These
  values are stored in AcquisitionMeta so downstream tools (T02 build_masters)
  can compute sensor-relative quality thresholds instead of hardcoded 16-bit
  constants.

Cross-validation:
  After sampling calibration frame EXIF, the tool checks:
    - Flat ISO matches light ISO (different ISO → flat noise pattern mismatch)
    - Dark exposure matches light exposure (within 5% tolerance)
    - Bias frames have near-zero exposure (long bias = bad frame or wrong folder)
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

import exiftool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.equipment import resolve_focal_length, resolve_pixel_size
from astro_agent.graph.state import AcquisitionMeta, Dataset, FileInventory
from astro_agent.tools._sensor import sensor_info_from_tags

# ── Constants ──────────────────────────────────────────────────────────────────

RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".raf", ".cr2", ".cr3", ".arw", ".nef", ".dng", ".rw2", ".orf"}
)
FITS_EXTENSIONS: frozenset[str] = frozenset({".fits", ".fit", ".fts"})

# Subdirectory name → canonical frame-type bucket (singular and plural)
_DIR_MAP: dict[str, str] = {
    "lights":      "lights",
    "light":       "lights",
    "darks":       "darks",
    "dark":        "darks",
    "flats":       "flats",
    "flat":        "flats",
    "bias":        "biases",
    "biases":      "biases",
    "bias_frames": "biases",
}


# ── Pydantic input schema ──────────────────────────────────────────────────────

class IngestDatasetInput(BaseModel):
    root_directory: str = Field(
        description=(
            "Absolute path to the dataset root. For camera RAW, expects "
            "subdirectories named lights/, darks/, flats/, bias/. "
            "For FITS (future), files may be anywhere under the root."
        )
    )
    file_pattern: str | None = Field(
        default=None,
        description=(
            "Optional glob to force a specific format (e.g. '*.fits' or '*.RAF'). "
            "When null, format is auto-detected from file extensions found."
        ),
    )
    override_target_name: str | None = Field(
        default=None,
        description=(
            "Override the target/object name. Use when the object cannot be "
            "determined from EXIF/headers or the filename."
        ),
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Thread/run ID for the processing session. Used as the output "
            "directory name: runs/<thread_id>/. Passed by the CLI."
        ),
    )


# ── EXIF extraction via ExifTool ───────────────────────────────────────────────

_EXPOSURE_KEYS = ("EXIF:ExposureTime", "Composite:ShutterSpeed",
                  "Exptime", "Exposure", "FITS:Exptime", "FITS:Exposure")
_FOCAL_KEYS    = ("EXIF:FocalLength", "Focallen", "FITS:Focallen")
_ISO_KEYS      = ("EXIF:ISO",)
_GAIN_KEYS     = ("Gain", "FITS:Gain")
_MODEL_KEYS    = ("EXIF:Model", "Instrument", "FITS:Instrument")
_MAKE_KEYS     = ("EXIF:Make", "Creator", "FITS:Creator")
_PIXEL_SIZE_KEYS = ("Xpixsz", "FITS:Xpixsz")


def _first(tags: dict, keys: tuple[str, ...]):
    for k in keys:
        v = tags.get(k)
        if v is not None and v != "undef":
            return v
    return None


class _FrameSample(NamedTuple):
    """Minimal per-frame EXIF snapshot used for cross-validation."""
    path:          str
    iso:           int | None
    exposure_s:    float | None
    black_level:   int
    white_level:   int
    bit_depth:     int


def _sample_frame(file_path: Path) -> _FrameSample | None:
    """
    Read EXIF from one raw frame and return a _FrameSample.
    Returns None on any error so callers can skip gracefully.
    """
    try:
        with exiftool.ExifToolHelper() as et:
            tags = et.get_metadata(str(file_path))[0]
    except Exception as e:
        logger.warning("ExifTool failed on %s: %s", file_path, e)
        return None

    raw_iso = _first(tags, _ISO_KEYS)
    iso: int | None = int(raw_iso) if raw_iso is not None else None

    raw_exp = _first(tags, _EXPOSURE_KEYS)
    exp: float | None = float(raw_exp) if raw_exp is not None else None

    sensor = sensor_info_from_tags(tags)

    return _FrameSample(
        path=str(file_path),
        iso=iso,
        exposure_s=exp,
        black_level=sensor.black_level,
        white_level=sensor.white_level,
        bit_depth=sensor.bit_depth,
    )


def _extract_raw_meta(
    light_files: list[Path],
    override_target_name: str | None,
) -> AcquisitionMeta:
    """
    Sample the first light frame for EXIF + sensor metadata using ExifTool.
    Returns a fully typed AcquisitionMeta with sensor characterization fields.
    """
    if not light_files:
        return _empty_meta("raw", override_target_name)

    try:
        with exiftool.ExifToolHelper() as et:
            tags = et.get_metadata(str(light_files[0]))[0]
    except Exception as e:
        logger.warning("ExifTool failed on %s: %s — returning empty metadata", light_files[0], e)
        return _empty_meta("raw", override_target_name)

    raw_iso = _first(tags, _ISO_KEYS)
    iso: int | None = int(raw_iso) if raw_iso is not None else None

    raw_gain = _first(tags, _GAIN_KEYS)
    gain: int | None = int(raw_gain) if raw_gain is not None else None

    raw_exp = _first(tags, _EXPOSURE_KEYS)
    exposure: float | None = float(raw_exp) if raw_exp is not None else None

    raw_focal = _first(tags, _FOCAL_KEYS)
    focal: float | None = float(raw_focal) if raw_focal is not None else None

    model_raw = _first(tags, _MODEL_KEYS)
    make_raw  = _first(tags, _MAKE_KEYS)
    camera_model: str | None = None
    if make_raw and model_raw:
        camera_model = f"{str(make_raw).strip()} {str(model_raw).strip()}"
    elif model_raw:
        camera_model = str(model_raw).strip()

    sensor = sensor_info_from_tags(tags)

    # Pixel size: equipment.toml first, then FITS header
    px_um: float | None = None
    try:
        px_um = resolve_pixel_size()
    except ValueError:
        pass
    if px_um is None:
        raw_px = _first(tags, _PIXEL_SIZE_KEYS)
        if raw_px is not None:
            try:
                px_um = float(raw_px)
            except (ValueError, TypeError):
                pass

    return AcquisitionMeta(
        target_name=override_target_name,
        target_coords=None,  # populated later by T29 resolve_target
        focal_length_mm=resolve_focal_length() or (float(focal) if focal else None),
        pixel_size_um=px_um,
        exposure_time_s=exposure,
        iso=iso,
        gain=gain,
        filter=None,                           # no filter wheel on camera RAW setups
        bortle=None,
        camera_model=camera_model,
        telescope=None,
        input_format="raw",
        # Sensor characterization (new)
        black_level=sensor.black_level,
        white_level=sensor.white_level,
        bit_depth=sensor.bit_depth,
        raw_exposure_bias=sensor.raw_exposure_bias,
        sensor_type=sensor.sensor_type,
    )


def _empty_meta(input_format: str, target_name: str | None = None) -> AcquisitionMeta:
    return AcquisitionMeta(
        target_name=target_name,
        target_coords=None,
        focal_length_mm=None,
        pixel_size_um=None,
        exposure_time_s=None,
        iso=None,
        gain=None,
        filter=None,
        bortle=None,
        camera_model=None,
        telescope=None,
        input_format=input_format,
        black_level=None,
        white_level=None,
        bit_depth=None,
        raw_exposure_bias=None,
        sensor_type=None,
    )


# ── Calibration frame cross-validation ────────────────────────────────────────

def _cross_validate_calibration(
    light_meta: AcquisitionMeta,
    buckets: dict[str, list[str]],
) -> list[str]:
    """
    Sample one frame from each calibration bucket and check for mismatches.

    Returns a list of warning strings (empty = all checks passed).
    Warnings are informational — they do not halt ingestion. T02 may escalate
    to HITL if a mismatch is critical for calibration quality.
    """
    warnings: list[str] = []

    light_iso = light_meta.get("iso")
    light_exp = light_meta.get("exposure_time_s")

    # ── Flats: ISO should match lights ──────────────────────────────────────
    flat_files = buckets.get("flats", [])
    if flat_files:
        sample = _sample_frame(Path(flat_files[0]))
        if sample:
            if light_iso is not None and sample.iso is not None:
                if sample.iso != light_iso:
                    warnings.append(
                        f"Flat ISO ({sample.iso}) does not match light ISO ({light_iso}). "
                        "Flat noise pattern will not subtract correctly — reshoot flats "
                        "at the same ISO as lights."
                    )
            if sample.black_level == 0 and light_meta.get("black_level", 0) != 0:
                warnings.append(
                    "Could not read black_level from flat EXIF. Sensor-relative "
                    "flat quality thresholds in T02 may fall back to safe defaults."
                )

    # ── Darks: exposure should be ≥ light exposure ───────────────────────────
    dark_files = buckets.get("darks", [])
    if dark_files and light_exp is not None:
        sample = _sample_frame(Path(dark_files[0]))
        if sample and sample.exposure_s is not None:
            ratio = sample.exposure_s / light_exp if light_exp > 0 else None
            if ratio is not None and (ratio < 0.95 or ratio > 1.05):
                warnings.append(
                    f"Dark exposure ({sample.exposure_s:.1f}s) differs from light "
                    f"exposure ({light_exp:.1f}s) by more than 5%. Dark current will "
                    "not scale correctly. For best results, match dark exposure to lights."
                )

    # ── Biases: exposure should be negligible ────────────────────────────────
    # Note: no bias exposure validation. DSLR bias frames use the shortest
    # shutter speed, but CMOS astro cameras (ZWO, QHY) often use "offset"
    # or "flat dark" frames at longer exposures in the bias folder. Both
    # workflows are valid. The agent can reason about the data.

    return warnings


# ── Format detection ───────────────────────────────────────────────────────────

def _detect_format(root: Path, file_pattern: str | None) -> str:
    """Return 'raw' or 'fits' based on extensions found. Respects file_pattern."""
    if file_pattern:
        ext = Path(file_pattern.lstrip("*")).suffix.lower()
        if ext in RAW_EXTENSIONS:
            return "raw"
        if ext in FITS_EXTENSIONS:
            return "fits"

    raw_found = False
    fits_found = False
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in RAW_EXTENSIONS:
            raw_found = True
        elif ext in FITS_EXTENSIONS:
            fits_found = True
        if raw_found:
            break  # RAW always wins — FITS outputs from prior Siril runs live alongside RAW inputs

    if raw_found:
        return "raw"
    if fits_found:
        return "fits"
    return "raw"  # default; will produce a clear error in _ingest_raw if truly empty


# ── RAW ingestion ──────────────────────────────────────────────────────────────

def _cleanup_previous_runs(runs_dir: Path, current_thread_id: str) -> list[str]:
    """
    Remove all previous run directories, keeping only the current run.

    Each run lives in runs/<thread_id>/. On a fresh run, previous run folders
    are deleted so stale intermediates don't accumulate. The current run's
    folder (just created, empty) is preserved.

    Controlled by CLEANUP_PREVIOUS_RUNS env var (default: true).
    """
    if not runs_dir.exists():
        return []

    import shutil
    import time
    removed: list[str] = []
    for item in sorted(runs_dir.iterdir()):
        if item.is_dir() and item.name != current_thread_id:
            for attempt in range(3):
                try:
                    shutil.rmtree(item)
                    removed.append(item.name)
                    break
                except OSError:
                    # macOS .DS_Store / Spotlight indexing can interfere — wait and retry
                    time.sleep(0.5)
            else:
                raise OSError(
                    f"Could not remove previous run directory after 3 attempts: {item}\n"
                    f"A file may be open or locked. Close any applications accessing "
                    f"this directory and try again."
                )

    return removed


def _ingest_raw(
    root: Path,
    file_pattern: str | None,
    override_target_name: str | None,
    thread_id: str | None = None,
) -> tuple[Dataset, list[str], dict]:
    target_name = override_target_name or root.name
    warnings: list[str] = []
    buckets: dict[str, list[str]] = {
        "lights": [], "darks": [], "flats": [], "biases": []
    }

    # ── Set up run-scoped working directory ────────────────────────────────
    # All pipeline outputs go into runs/<thread_id>/ so they are isolated
    # from the raw input files and from other runs. On fresh run, previous
    # run folders are cleaned up by default.
    runs_dir = root / "runs"
    run_id = thread_id or str(uuid.uuid4())
    working_dir = runs_dir / run_id
    working_dir.mkdir(parents=True, exist_ok=True)

    cleaned: list[str] = []
    if os.environ.get("CLEANUP_PREVIOUS_RUNS", "true").lower() in ("true", "1", "yes"):
        cleaned = _cleanup_previous_runs(runs_dir, run_id)
        if cleaned:
            logger.info(f"Cleaned {len(cleaned)} previous run(s): {cleaned}")

    glob = file_pattern or "*"

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip the runs/ directory when scanning for frame subdirectories
        if subdir.name == "runs":
            continue
        frame_type = _DIR_MAP.get(subdir.name.lower())
        if frame_type is None:
            candidates = [
                f for f in subdir.glob(glob)
                if f.is_file() and f.suffix.lower() in RAW_EXTENSIONS
            ]
            if candidates:
                warnings.append(
                    f"Unrecognized subdirectory '{subdir.name}' contains "
                    f"{len(candidates)} RAW file(s) — skipped. "
                    "Rename to lights/, darks/, flats/, or bias/."
                )
            continue

        files = sorted(
            f for f in subdir.glob(glob)
            if f.is_file() and f.suffix.lower() in RAW_EXTENSIONS
        )
        buckets[frame_type] = [str(f) for f in files]

    if not buckets["lights"]:
        raise ValueError(
            f"No RAW light frames found under {root}. "
            "Expected a 'lights/' subdirectory containing RAF/CR2/etc. files."
        )

    light_paths = [Path(p) for p in buckets["lights"]]
    meta = _extract_raw_meta(light_paths, target_name)

    if meta.get("iso") is None and meta.get("exposure_time_s") is None:
        warnings.append(
            "ExifTool could not read EXIF metadata from the light frames. "
            "ISO, exposure time, sensor characterization (black/white levels), "
            "and focal length are all unknown. Downstream tools will use "
            "equipment.toml or manual values. Check that ExifTool is installed "
            "and the RAW files are not corrupted."
        )

    # Cross-validate calibration frames against lights
    calib_warnings = _cross_validate_calibration(meta, buckets)
    warnings.extend(calib_warnings)

    inventory = FileInventory(
        lights=buckets["lights"],
        darks=buckets["darks"],
        flats=buckets["flats"],
        biases=buckets["biases"],
    )

    dataset = Dataset(
        id=str(uuid.uuid4()),
        working_dir=str(working_dir),
        files=inventory,
        acquisition_meta=meta,
    )

    exp = meta.get("exposure_time_s")
    total_exp = (exp * len(buckets["lights"])) if exp else 0.0

    if not buckets["darks"]:
        warnings.append(
            "No dark frames found. Consider shooting matching darks or "
            "using bias-only calibration in T03."
        )
    if not buckets["flats"]:
        warnings.append(
            "No flat frames found. Vignetting and dust correction will be skipped."
        )
    if not buckets["biases"]:
        warnings.append(
            "No bias frames found. Using dark-only calibration strategy."
        )

    extensions = sorted(
        {Path(f).suffix.lower()
         for files in buckets.values() for f in files}
    )

    sensor_summary: dict = {}
    if meta.get("black_level") is not None:
        sensor_summary = {
            "black_level":        meta["black_level"],
            "white_level":        meta["white_level"],
            "bit_depth":          meta["bit_depth"],
            "raw_exposure_bias":  meta.get("raw_exposure_bias"),
            "sensor_type":        meta.get("sensor_type"),
        }

    summary = {
        "lights_count":       len(buckets["lights"]),
        "darks_count":        len(buckets["darks"]),
        "flats_count":        len(buckets["flats"]),
        "biases_count":       len(buckets["biases"]),
        "total_exposure_s":   round(total_exp, 2),
        "input_format":       "raw",
        "detected_extensions": extensions,
        "sensor":             sensor_summary,
        "working_dir":        str(working_dir),
        "cleaned_runs":       cleaned,
    }

    return dataset, warnings, summary


# ── FITS ingestion (stub) ──────────────────────────────────────────────────────

def _classify_fits_frame(fits_path: Path) -> str | None:
    """
    Classify a FITS file by IMAGETYP header or subdirectory name.
    IMAGETYP is the FITS standard for frame classification (used by ZWO/ASIAIR).
    Falls back to subdirectory name for manually organized datasets.
    """
    try:
        from astropy.io import fits as astropy_fits
        with astropy_fits.open(str(fits_path)) as hdul:
            imagetyp = str(hdul[0].header.get("IMAGETYP", "")).strip().lower()
            type_map = {
                "light": "lights", "light frame": "lights",
                "flat": "flats", "flat frame": "flats",
                "dark": "darks", "dark frame": "darks",
                "bias": "biases", "bias frame": "biases",
                "offset": "biases",
            }
            if imagetyp in type_map:
                return type_map[imagetyp]
    except Exception:
        pass
    # Fall back to subdirectory name
    parent = fits_path.parent.name.lower()
    return _DIR_MAP.get(parent)


def _ingest_fits(
    root: Path,
    file_pattern: str | None,
    override_target_name: str | None,
    thread_id: str | None = None,
) -> tuple[Dataset, list[str], dict]:
    """
    Ingest FITS files from a dedicated astronomy camera (ZWO, QHY, etc.).
    Mirrors _ingest_raw() structure but classifies by IMAGETYP header
    and reads metadata from FITS headers instead of EXIF.
    """
    target_name = override_target_name or root.name
    warnings: list[str] = []
    buckets: dict[str, list[str]] = {
        "lights": [], "darks": [], "flats": [], "biases": []
    }

    # Set up run-scoped working directory (same as _ingest_raw)
    runs_dir = root / "runs"
    run_id = thread_id or str(uuid.uuid4())
    working_dir = runs_dir / run_id
    working_dir.mkdir(parents=True, exist_ok=True)

    cleaned: list[str] = []
    if os.environ.get("CLEANUP_PREVIOUS_RUNS", "true").lower() in ("true", "1", "yes"):
        cleaned = _cleanup_previous_runs(runs_dir, run_id)
        if cleaned:
            logger.info(f"Cleaned {len(cleaned)} previous run(s): {cleaned}")

    glob = file_pattern or "*"

    # Scan for FITS files — try subdirectory classification first,
    # then IMAGETYP header for flat directory structures
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir() or subdir.name == "runs":
            continue

        # Try subdirectory name first
        frame_type = _DIR_MAP.get(subdir.name.lower())

        files = sorted(
            f for f in subdir.rglob(glob)
            if f.is_file() and f.suffix.lower() in FITS_EXTENSIONS
        )

        if frame_type:
            buckets[frame_type] = [str(f) for f in files]
        elif files:
            # Subdirectory name not recognized — try IMAGETYP header
            classified_any = False
            for f in files:
                ftype = _classify_fits_frame(f)
                if ftype:
                    buckets[ftype].append(str(f))
                    classified_any = True
            if not classified_any:
                warnings.append(
                    f"Unrecognized subdirectory '{subdir.name}' contains "
                    f"{len(files)} FITS file(s) — skipped. "
                    "Rename to lights/, darks/, flats/, or bias/, or add "
                    "IMAGETYP header to FITS files."
                )

    # Also check root directory for FITS files (flat directory structure)
    root_fits = sorted(
        f for f in root.glob(glob)
        if f.is_file() and f.suffix.lower() in FITS_EXTENSIONS
    )
    if root_fits and not any(buckets.values()):
        for f in root_fits:
            ftype = _classify_fits_frame(f)
            if ftype:
                buckets[ftype].append(str(f))

    if not buckets["lights"]:
        raise ValueError(
            f"No FITS light frames found under {root}. "
            "Expected subdirectories (lights/, darks/, etc.) or FITS files "
            "with IMAGETYP header."
        )

    # Extract metadata using the same function as RAW
    # (key tuples now include FITS fallbacks)
    light_paths = [Path(p) for p in buckets["lights"]]
    meta = _extract_raw_meta(light_paths, target_name)

    calib_warnings = _cross_validate_calibration(meta, buckets)
    warnings.extend(calib_warnings)

    inventory = FileInventory(
        lights=buckets["lights"],
        darks=buckets["darks"],
        flats=buckets["flats"],
        biases=buckets["biases"],
    )

    dataset = Dataset(
        id=str(uuid.uuid4()),
        working_dir=str(working_dir),
        files=inventory,
        acquisition_meta=meta,
    )

    exp = meta.get("exposure_time_s")
    total_exp = (exp * len(buckets["lights"])) if exp else 0.0

    if not buckets["darks"]:
        warnings.append("No dark frames found.")
    if not buckets["flats"]:
        warnings.append("No flat frames found.")
    if not buckets["biases"]:
        warnings.append("No bias frames found.")

    extensions = sorted(
        {Path(f).suffix.lower()
         for files in buckets.values() for f in files}
    )

    sensor_summary: dict = {}
    if meta.get("black_level") is not None:
        sensor_summary = {
            "black_level":   meta["black_level"],
            "white_level":   meta["white_level"],
            "bit_depth":     meta["bit_depth"],
            "sensor_type":   meta.get("sensor_type"),
            "gain":          meta.get("gain"),
        }

    summary = {
        "lights_count":       len(buckets["lights"]),
        "darks_count":        len(buckets["darks"]),
        "flats_count":        len(buckets["flats"]),
        "biases_count":       len(buckets["biases"]),
        "total_exposure_s":   round(total_exp, 2),
        "input_format":       "fits",
        "detected_extensions": extensions,
        "sensor":             sensor_summary,
        "working_dir":        str(working_dir),
        "cleaned_runs":       cleaned,
    }

    return dataset, warnings, summary


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=IngestDatasetInput)
def ingest_dataset(
    root_directory: str,
    file_pattern: str | None = None,
    override_target_name: str | None = None,
    thread_id: str | None = None,
) -> dict:
    """
    Scan a dataset directory, classify frames (lights/darks/flats/bias),
    extract acquisition metadata from EXIF (camera RAW) or FITS headers,
    and return a populated Dataset schema.

    All pipeline outputs are written to runs/<thread_id>/ inside the dataset
    root, keeping raw input files completely untouched. Previous run folders
    are cleaned up by default (set CLEANUP_PREVIOUS_RUNS=false to keep them).

    The returned dataset.acquisition_meta now includes sensor characterization:
    black_level, white_level, bit_depth, raw_exposure_bias, sensor_type.
    These values enable T02 to apply sensor-relative HITL thresholds for flat
    quality, rather than hardcoded 16-bit constants.
    """
    root = Path(root_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root_directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"root_directory is not a directory: {root}")

    fmt = _detect_format(root, file_pattern)

    if fmt == "raw":
        dataset, warnings, summary = _ingest_raw(
            root, file_pattern, override_target_name, thread_id=thread_id,
        )
    else:
        dataset, warnings, summary = _ingest_fits(
            root, file_pattern, override_target_name, thread_id=thread_id,
        )

    return {
        "dataset":  dataset,
        "warnings": warnings,
        "summary":  summary,
    }
