"""
Equipment configuration loader.

Reads equipment.toml from the project root and provides equipment data to
any tool that needs it. This replaces the hardcoded camera lookup table
that was in T10.

Resolution order for pixel_size_um:
  1. Explicit argument passed to the tool
  2. equipment.toml [camera] pixel_size_um
  3. PIXEL_SIZE_UM environment variable (backward compat)

Resolution order for focal_length_mm:
  1. Explicit argument passed to the tool
  2. equipment.toml [optics] focal_length_mm
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11 fallback


def _find_config_path() -> Path | None:
    """Walk up from this file's directory looking for equipment.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "equipment.toml"
        if candidate.exists():
            return candidate
        current = current.parent
    env_path = os.environ.get("EQUIPMENT_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return None


@lru_cache(maxsize=1)
def load_equipment() -> dict[str, Any]:
    """Load and cache the equipment configuration."""
    config_path = _find_config_path()
    if config_path is None:
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def reload_equipment() -> dict[str, Any]:
    """Force reload the equipment config (clears cache)."""
    load_equipment.cache_clear()
    return load_equipment()


def get_camera() -> dict[str, Any]:
    return load_equipment().get("camera", {})


def get_optics() -> dict[str, Any]:
    return load_equipment().get("optics", {})


def get_location() -> dict[str, Any]:
    return load_equipment().get("location", {})


def resolve_pixel_size(explicit_value: float | None = None) -> float:
    """
    Resolve pixel size in microns.
      1. Explicit value (from tool argument)
      2. equipment.toml [camera] pixel_size_um
      3. PIXEL_SIZE_UM env var
    Raises ValueError if none can provide a value.
    """
    if explicit_value is not None and explicit_value > 0:
        return explicit_value

    camera = get_camera()
    config_val = camera.get("pixel_size_um")
    if config_val is not None and config_val > 0:
        return float(config_val)

    env_val = os.environ.get("PIXEL_SIZE_UM", "").strip()
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass

    raise ValueError(
        "Cannot resolve pixel_size_um. Options:\n"
        "  1. Pass pixel_size_um explicitly to the tool.\n"
        "  2. Set pixel_size_um in equipment.toml [camera].\n"
        "  3. Set PIXEL_SIZE_UM in .env."
    )


def resolve_focal_length(explicit_value: float | None = None) -> float | None:
    """
    Resolve focal length in mm.
      1. Explicit value (from tool argument)
      2. equipment.toml [optics] focal_length_mm
    Returns None if unavailable (some tools require it, others don't).
    """
    if explicit_value is not None and explicit_value > 0:
        return explicit_value

    optics = get_optics()
    config_val = optics.get("focal_length_mm")
    if config_val is not None and config_val > 0:
        return float(config_val)

    return None


def resolve_target_coords(target_name: str) -> dict[str, float] | None:
    """
    Resolve an astronomical target name to {'ra': float, 'dec': float} in decimal
    degrees (J2000) using astropy's SIMBAD/NED name resolver.

    Returns None on failure (network error, unknown target, etc.) — callers
    should treat a None result as a soft failure and proceed without coordinates.
    """
    try:
        from astropy.coordinates import SkyCoord
        coord = SkyCoord.from_name(target_name)
        return {"ra": float(coord.ra.deg), "dec": float(coord.dec.deg)}
    except Exception:
        return None
