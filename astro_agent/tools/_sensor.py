"""
Sensor-level utilities: black/white level extraction, fill fraction computation,
and flat quality state classification.

These helpers are camera-agnostic — they work for any raw format by reading
sensor characterization data from EXIF/maker notes. Used by:
  - T01 (ingest_dataset)         — populate AcquisitionMeta sensor fields
  - T02 (build_masters)          — sensor-relative HITL thresholds
  - scripts/check_flat_quality.py — per-frame flat quality assessment

## Why fill fraction instead of Siril-normalized value

Siril normalizes float FITS using a 65535 divisor (16-bit scale). For a 14-bit
sensor with white level 16383 and black level ~1022, the maximum achievable
Siril-normalized value is:

    (16383 - 1022) / 65535 ≈ 0.234

The old hardcoded HITL threshold of 0.25 is literally unreachable on any
14-bit DSLR or mirrorless camera (Fuji, Canon, Nikon, Sony). Fill fraction
avoids this: it expresses signal as a percentage of the sensor's own usable
ADU range and is equally valid on any sensor.

Target: 30–55% fill = well-exposed flat on any camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import exiftool

# ── Fill fraction targets (sensor-agnostic) ────────────────────────────────────

TARGET_FILL_MIN    = 0.30   # 30% of usable ADU range — minimum acceptable flat
TARGET_FILL_MAX    = 0.55   # 55% — above this, risk of non-linearity near clipping
TARGET_FILL_CENTER = 0.425  # ideal center for ranking

SATURATED_FILL     = 0.97   # fill ≥ this → sensor well is clipped
SATURATED_STD_MAX  = 2.0    # also SATURATED if std is near-zero at high ADU


# ── Known X-Trans sensor models (Fuji) ────────────────────────────────────────
# Bayer vs X-Trans matters for T03 debayering. All other major brands are Bayer.

_XTRANS_MODELS: frozenset[str] = frozenset({
    "X-T5", "X-T4", "X-T3", "X-T30 II", "X-T30", "X-T20", "X-T10",
    "X-S20", "X-S10", "X-Pro3", "X-Pro2", "X-H2", "X-H2S", "X-H1",
    "X-E4", "X-E3", "X100VI", "X100V", "GFX 100S", "GFX 50S II", "GFX 50R",
})

# Dedicated astronomy camera makes that output 16-bit data
_ASTRO_CAM_MAKES: frozenset[str] = frozenset({
    "ZWO", "QHY", "SBIG", "MORAVIAN", "ATIK", "STARLIGHT XPRESS",
    "PLAYER ONE", "TOUPTEK",
})

# EXIF keys to try for black level (in priority order)
_BLACK_KEYS = (
    "RAF:BlackLevel",
    "MakerNotes:BlackLevel",
    "EXIF:BlackLevel",
    "MakerNotes:BlackLevels",
    "MakerNotes:BlackLevel2",
    "MakerNotes:NormalBlackLevel",
)

# EXIF keys to try for white level
_WHITE_KEYS = (
    "RAF:WhiteLevel",
    "MakerNotes:WhiteLevel",
    "EXIF:WhiteLevel",
    "MakerNotes:LinearityUpperMargin",
    "MakerNotes:NormalWhiteLevel",
)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class SensorInfo:
    """Sensor characterization extracted from EXIF."""
    black_level:       int           # pedestal ADU
    white_level:       int           # sensor max ADU
    bit_depth:         int           # 12, 14, or 16
    raw_exposure_bias: float | None  # stops (Fuji: -0.72 typical, may vary)
    sensor_type:       str | None    # "bayer" | "xtrans" | None


@dataclass
class FrameExif:
    """Combined EXIF read for a single frame: sensor info + acquisition info."""
    sensor:        SensorInfo
    exposure_time: float | None   # shutter speed in seconds
    iso:           int | None
    make:          str | None
    model:         str | None


# ── Internal helpers ───────────────────────────────────────────────────────────

def _parse_first_int(val) -> int | None:
    """Parse the first integer from a space-separated value or scalar."""
    if val is None:
        return None
    try:
        return int(float(str(val).split()[0]))
    except (ValueError, TypeError):
        return None


def _parse_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def infer_white_level(data_max: int) -> int:
    """
    Map a raw data maximum to the nearest 2^n − 1 sensor white level.
    Used as a fallback when EXIF does not contain white level.
    """
    for bits in (8, 10, 12, 14, 16):
        candidate = (1 << bits) - 1
        if data_max <= candidate:
            return candidate
    return 65535


def infer_bit_depth(white_level: int) -> int:
    """Return the bit depth implied by a known white level."""
    for bits in (8, 10, 12, 14, 16):
        if white_level <= (1 << bits) - 1:
            return bits
    return 16


def sensor_info_from_tags(tags: dict) -> SensorInfo:
    """
    Build a SensorInfo from an already-loaded ExifTool tags dict.
    Called by T01 which loads tags once and reuses them.

    Supports both camera RAW (EXIF metadata) and FITS (FITS headers).
    EXIF keys are checked first; FITS-native keys serve as fallbacks.
    """
    # Camera identification — EXIF first, then FITS headers
    # ExifTool Python API prefixes FITS keys with "FITS:", CLI does not
    make = str(
        tags.get("EXIF:Make", "")
        or tags.get("Make", "")
        or tags.get("FITS:Creator", "")
    ).upper()
    model = str(
        tags.get("EXIF:Model", "")
        or tags.get("Instrument", "")
        or tags.get("FITS:Instrument", "")
        or tags.get("Model", "")
    )

    # If make is empty but model contains a known astro cam brand, extract it
    if not make and model:
        model_upper = model.upper()
        for brand in _ASTRO_CAM_MAKES:
            if brand in model_upper:
                make = brand
                break

    # Black level — EXIF keys first, then leave as 0 for FITS
    # (FITS cameras don't store black level in headers; T02 detects from data)
    black = 0
    for key in _BLACK_KEYS:
        v = _parse_first_int(tags.get(key))
        if v is not None and v > 0:
            black = v
            break

    # White level from EXIF
    white_exif: int | None = None
    for key in _WHITE_KEYS:
        v = _parse_first_int(tags.get(key))
        if v is not None and v > black:
            white_exif = v
            break

    # Bit depth + white level fallback
    if white_exif is not None:
        bit_depth = infer_bit_depth(white_exif)
        white = white_exif
    else:
        # Check FITS BITPIX header (CLI: "Bitpix", Python API: "FITS:Bitpix")
        bitpix = tags.get("Bitpix") or tags.get("FITS:Bitpix")
        if bitpix is not None:
            bp = abs(int(bitpix))
            if bp in (8, 16, 32):
                bit_depth = bp
                white = (1 << min(bp, 16)) - 1
            else:
                bit_depth = 16
                white = 65535
        else:
            is_astro = any(s in make for s in _ASTRO_CAM_MAKES)
            bit_depth = 16 if is_astro else 14
            white = (1 << bit_depth) - 1

    # Raw exposure bias (Fuji-specific; harmless elsewhere)
    raw_exp_bias: float | None = None
    for key in ("RAF:RawExposureBias", "MakerNotes:RawExposureBias"):
        v = _parse_float(tags.get(key))
        if v is not None:
            raw_exp_bias = v
            break

    # Sensor type — EXIF make first, then FITS Bayerpat header
    sensor_type: str | None = None
    if "FUJIFILM" in make:
        sensor_type = "xtrans" if model.strip() in _XTRANS_MODELS else "bayer"
    elif make:
        sensor_type = "bayer"

    # FITS fallback: Bayerpat header indicates Bayer CFA
    if sensor_type is None:
        bayerpat = str(tags.get("Bayerpat", "") or tags.get("FITS:Bayerpat", "")).strip()
        if bayerpat:
            sensor_type = "bayer"
        elif any(s in make for s in _ASTRO_CAM_MAKES):
            # Known astro cam without Bayer pattern → mono
            sensor_type = "mono"

    return SensorInfo(
        black_level=black,
        white_level=white,
        bit_depth=bit_depth,
        raw_exposure_bias=raw_exp_bias,
        sensor_type=sensor_type,
    )


def read_frame_exif(file_path: Path) -> FrameExif:
    """
    Read sensor + acquisition EXIF from a raw file in a single ExifTool call.
    Returns a FrameExif with best-effort values; falls back gracefully on error.
    """
    try:
        with exiftool.ExifToolHelper() as et:
            tags = et.get_metadata(str(file_path))[0]
    except Exception:
        return FrameExif(
            sensor=SensorInfo(0, 65535, 16, None, None),
            exposure_time=None,
            iso=None,
            make=None,
            model=None,
        )

    sensor = sensor_info_from_tags(tags)

    exp: float | None = None
    for key in ("EXIF:ExposureTime", "Composite:ShutterSpeed",
                "Exptime", "Exposure", "FITS:Exptime", "FITS:Exposure"):
        v = _parse_float(tags.get(key))
        if v is not None:
            exp = v
            break

    iso: int | None = None
    v = _parse_first_int(tags.get("EXIF:ISO"))
    if v is not None:
        iso = v

    make  = tags.get("EXIF:Make") or tags.get("Make") or tags.get("FITS:Creator")
    model = tags.get("EXIF:Model") or tags.get("Instrument") or tags.get("FITS:Instrument")

    return FrameExif(
        sensor=sensor,
        exposure_time=exp,
        iso=iso,
        make=str(make).strip() if make else None,
        model=str(model).strip() if model else None,
    )


def compute_fill(median_adu: int, black: int, white: int) -> float:
    """
    Fraction of usable sensor range filled by the signal.
    0.0 = signal is at the black level (no light captured).
    1.0 = signal is at the white level (sensor well full / saturated).
    """
    usable = white - black
    if usable <= 0:
        return 0.0
    return max(0.0, min(1.0, (float(median_adu) - black) / usable))


def flat_fill_state(
    fill: float | None,
    median_adu: int | None = None,
    std_adu: float | None = None,
    white_level: int | None = None,
) -> str:
    """
    Classify a flat's fill fraction into a usability state.

    Returns one of:
      USABLE    — fill 30–55%, good signal, not clipped
      UNDER     — fill < 30%, increase exposure or light source brightness
      OVER      — fill > 55%, decrease exposure or light source brightness
      SATURATED — fill ≥ 97%, or near-zero variance at high ADU (clipped wells)
      UNKNOWN   — could not compute fill
    """
    if fill is None:
        return "UNKNOWN"
    if fill >= SATURATED_FILL:
        return "SATURATED"
    if (std_adu is not None and std_adu <= SATURATED_STD_MAX
            and median_adu is not None and white_level is not None
            and median_adu >= int(white_level * 0.95)):
        return "SATURATED"
    if fill < TARGET_FILL_MIN:
        return "UNDER"
    if fill > TARGET_FILL_MAX:
        return "OVER"
    return "USABLE"


def flat_siril_norm_thresholds(black: int, white: int) -> tuple[float, float]:
    """
    Compute sensor-relative Siril-normalized flat thresholds.

    Siril normalizes calibrated float FITS using a 65535 divisor regardless
    of the sensor's native bit depth. For a 14-bit camera (white = 16383,
    black = 1022), the maximum achievable normalized value is:
        (16383 - 1022) / 65535 ≈ 0.234

    This function returns the (min, max) Siril-normalized values that
    correspond to TARGET_FILL_MIN and TARGET_FILL_MAX on this sensor,
    so downstream HITL thresholds are physically meaningful.
    """
    usable = white - black
    norm_min = (usable * TARGET_FILL_MIN) / 65535.0
    norm_max = (usable * TARGET_FILL_MAX) / 65535.0
    return norm_min, norm_max


def flat_adu_range(black: int, white: int) -> tuple[int, int, int]:
    """
    Return (min_adu, max_adu, ideal_adu) for a well-exposed flat on this sensor.
    Corresponds to TARGET_FILL_MIN, TARGET_FILL_MAX, TARGET_FILL_CENTER.
    """
    usable = white - black
    return (
        int(black + TARGET_FILL_MIN   * usable),
        int(black + TARGET_FILL_MAX   * usable),
        int(black + TARGET_FILL_CENTER * usable),
    )
