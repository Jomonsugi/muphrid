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
"""

from __future__ import annotations

import uuid
from pathlib import Path

import exiftool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.graph.state import AcquisitionMeta, Dataset, FileInventory

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


# ── EXIF extraction via ExifTool ───────────────────────────────────────────────

# ExifTool tag names we care about (group-prefixed, default pyexiftool format)
_EXPOSURE_KEYS   = ("EXIF:ExposureTime", "Composite:ShutterSpeed")
_FOCAL_KEYS      = ("EXIF:FocalLength",)
_ISO_KEYS        = ("EXIF:ISO",)
_MODEL_KEYS      = ("EXIF:Model",)
_MAKE_KEYS       = ("EXIF:Make",)


def _first(tags: dict, keys: tuple[str, ...]):
    for k in keys:
        v = tags.get(k)
        if v is not None and v != "undef":
            return v
    return None


def _extract_raw_meta(light_files: list[Path], override_target_name: str | None) -> AcquisitionMeta:
    """
    Sample the first light frame for EXIF metadata using ExifTool.
    Returns a fully typed AcquisitionMeta.
    """
    if not light_files:
        return _empty_meta("raw", override_target_name)

    try:
        with exiftool.ExifToolHelper() as et:
            tags = et.get_metadata(str(light_files[0]))[0]
    except Exception:
        return _empty_meta("raw", override_target_name)

    raw_iso = _first(tags, _ISO_KEYS)
    iso: int | None = int(raw_iso) if raw_iso is not None else None

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

    return AcquisitionMeta(
        target_name=override_target_name,
        focal_length_mm=focal,
        pixel_size_um=None,      # not in EXIF; known per sensor model
        exposure_time_s=exposure,
        iso=iso,
        gain=None,               # DSLR/mirrorless has no ADU gain concept
        filter=None,             # no filter wheel on camera RAW setups
        bortle=None,
        camera_model=camera_model,
        telescope=None,
        input_format="raw",
    )


def _empty_meta(input_format: str, target_name: str | None = None) -> AcquisitionMeta:
    return AcquisitionMeta(
        target_name=target_name,
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
    )


# ── Format detection ───────────────────────────────────────────────────────────

def _detect_format(root: Path, file_pattern: str | None) -> str:
    """Return 'raw' or 'fits' based on extensions found. Respects file_pattern."""
    if file_pattern:
        ext = Path(file_pattern.lstrip("*")).suffix.lower()
        if ext in RAW_EXTENSIONS:
            return "raw"
        if ext in FITS_EXTENSIONS:
            return "fits"

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() in RAW_EXTENSIONS:
            return "raw"
        if f.suffix.lower() in FITS_EXTENSIONS:
            return "fits"

    return "raw"  # default; will produce a clear error in _ingest_raw if truly empty


# ── RAW ingestion ──────────────────────────────────────────────────────────────

def _ingest_raw(
    root: Path,
    file_pattern: str | None,
    override_target_name: str | None,
) -> tuple[Dataset, list[str], dict]:
    # Use explicit override; fall back to root directory name (e.g. "M42_OrionNebula_2025")
    target_name = override_target_name or root.name
    warnings: list[str] = []
    buckets: dict[str, list[str]] = {
        "lights": [], "darks": [], "flats": [], "biases": []
    }

    glob = file_pattern or "*"

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
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

    inventory = FileInventory(
        lights=buckets["lights"],
        darks=buckets["darks"],
        flats=buckets["flats"],
        biases=buckets["biases"],
    )

    dataset = Dataset(
        id=str(uuid.uuid4()),
        working_dir=str(root),
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

    summary = {
        "lights_count":       len(buckets["lights"]),
        "darks_count":        len(buckets["darks"]),
        "flats_count":        len(buckets["flats"]),
        "biases_count":       len(buckets["biases"]),
        "total_exposure_s":   round(total_exp, 2),
        "input_format":       "raw",
        "detected_extensions": extensions,
    }

    return dataset, warnings, summary


# ── FITS ingestion (stub) ──────────────────────────────────────────────────────

def _ingest_fits(
    root: Path,
    file_pattern: str | None,
    override_target_name: str | None,
) -> tuple[Dataset, list[str], dict]:
    raise NotImplementedError(
        "FITS ingestion is not yet implemented. "
        "The current pipeline is built for camera RAW (RAF/CR2/etc.) input. "
        "Implement _ingest_fits() when a FITS-native camera (e.g. ZWO ASI) is used."
    )


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=IngestDatasetInput)
def ingest_dataset(
    root_directory: str,
    file_pattern: str | None = None,
    override_target_name: str | None = None,
) -> dict:
    """
    Scan a dataset directory, classify frames (lights/darks/flats/bias),
    extract acquisition metadata from EXIF (camera RAW) or FITS headers,
    and return a populated Dataset schema.

    Always call this tool first. The returned dataset flows into all
    subsequent tools. Check summary.input_format — if 'raw', pass
    is_cfa=true to siril_calibrate (T03).
    """
    root = Path(root_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root_directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"root_directory is not a directory: {root}")

    fmt = _detect_format(root, file_pattern)

    if fmt == "raw":
        dataset, warnings, summary = _ingest_raw(root, file_pattern, override_target_name)
    else:
        dataset, warnings, summary = _ingest_fits(root, file_pattern, override_target_name)

    return {
        "dataset":  dataset,
        "warnings": warnings,
        "summary":  summary,
    }
