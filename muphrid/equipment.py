"""
Equipment configuration loader.

equipment.toml is the durable config file. It seeds state.acquisition_meta
at ingest_dataset time. After ingest, state is canonical — downstream tools
read state, not this module.

Two resolution layers, kept apart:

  At ingest time (called only by ingest_dataset):
    `resolve_focal_length_for_ingest()` / `resolve_pixel_size_for_ingest()`
    return the equipment.toml value (or None / ValueError). Ingest combines
    them with explicit override kwargs and FITS/EXIF header values.

  At downstream tool runtime:
    `equipment_from_state(state, require=(…,))` reads
    state.dataset.acquisition_meta and refuses cleanly when required fields
    are null, naming the state path the caller should resolve.

The env-var hop (FOCAL_LENGTH_MM / PIXEL_SIZE_UM / SENSOR_TYPE_OVERRIDE)
that the Gradio app used to write is gone. Gradio now passes override
kwargs to ingest_dataset directly; state is the contract from there.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib # Python < 3.11 fallback


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


def resolve_pixel_size() -> float | None:
    """
    Read pixel size in microns from equipment.toml. Returns None if unset.
    Called only by ingest_dataset as one tier of its EXIF/FITS/toml chain.
    Downstream tools should read state.dataset.acquisition_meta.pixel_size_um.
    """
    camera = get_camera()
    config_val = camera.get("pixel_size_um")
    if config_val is not None and config_val > 0:
        return float(config_val)
    return None


def resolve_focal_length() -> float | None:
    """
    Read focal length in mm from equipment.toml. Returns None if unset.
    Called only by ingest_dataset as one tier of its EXIF/FITS/toml chain.
    Downstream tools should read state.dataset.acquisition_meta.focal_length_mm.
    """
    optics = get_optics()
    config_val = optics.get("focal_length_mm")
    if config_val is not None and config_val > 0:
        return float(config_val)
    return None


def equipment_from_state(
    state: dict,
    *,
    require: tuple[str, ...] = (),
) -> dict:
    """
    Downstream-tool resolver: return state.dataset.acquisition_meta.

    `state` is the canonical source after ingest_dataset. Tools that need
    hardware facts call this and never re-resolve from equipment.toml or
    env vars; that's the ingest layer's job exactly once per run.

    If any name in `require` is null/missing, raise ValueError naming the
    state path so the agent (or user) knows where to set the value.

    Returns the acquisition_meta dict (empty dict when none is set).
    """
    meta = ((state or {}).get("dataset") or {}).get("acquisition_meta") or {}
    missing = [name for name in require if not meta.get(name)]
    if missing:
        joined = ", ".join(f"acquisition_meta.{n}" for n in missing)
        raise ValueError(
            f"Required equipment state is null: {joined}. "
            f"Set the value in equipment.toml (for CLI/autonomous runs) or "
            f"the Gradio Equipment tab (for UI runs) and re-ingest the dataset. "
            f"State is the contract — re-running this tool without populated "
            f"state will keep failing."
        )
    return meta


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
