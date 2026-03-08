"""
T29 — resolve_target

Resolve an astronomical target name to RA/DEC coordinates (J2000, decimal degrees)
via SIMBAD. Returns coordinates that are passed as target_coords / approximate_coords
to any tool that performs plate solving (T10 color_calibrate, T21 plate_solve).

This is a mandatory early step in every pipeline run. The target name is required
input — the pipeline cannot proceed to plate-dependent operations (T10, T21) without
known coordinates, since blind wide-field plate solving is unreliable without a
position hint, especially for manual lenses without EXIF focal length metadata.

Call this immediately after T01 ingest. Store ra/dec in agent state and pass to
T10 and T21.

Requires internet access (queries the CDS SIMBAD name resolver).
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.equipment import resolve_target_coords


class ResolveTargetInput(BaseModel):
    target_name: str = Field(
        description=(
            "Astronomical target name as recognized by SIMBAD. "
            "Examples: 'M42', 'Orion Nebula', 'NGC 1977', 'Andromeda Galaxy', "
            "'Horsehead Nebula', 'Pleiades', 'M31'. "
            "Common catalog prefixes: M (Messier), NGC, IC, Sh2- (Sharpless). "
            "This is required — the pipeline cannot perform plate solving without it."
        )
    )


@tool(args_schema=ResolveTargetInput)
def resolve_target(target_name: str) -> dict:
    """
    Resolve an astronomical target name to RA/DEC coordinates via SIMBAD.

    MANDATORY: Call this immediately after T01 ingest with the user-provided target
    name. Store ra/dec and pass to color_calibrate (T10) as target_coords and to
    plate_solve (T21) as approximate_coords. Plate solving will fail or be
    unreliable without a position hint.

    Returns ra/dec in decimal degrees (J2000).

    Example:
        result = resolve_target("M42")
        # result["ra"] = 83.82, result["dec"] = -5.39
        color_calibrate(..., target_coords={"ra": result["ra"], "dec": result["dec"]})

    If SIMBAD cannot resolve the name, try an alternative spelling before
    falling back to manual coordinates.
    """
    coords = resolve_target_coords(target_name)
    if coords is None:
        return {
            "success": False,
            "target_name": target_name,
            "ra": None,
            "dec": None,
            "error_msg": (
                f"Could not resolve '{target_name}' via SIMBAD. "
                "Try an alternative name (e.g. 'M42' instead of 'Orion Nebula'), "
                "or provide target_coords manually."
            ),
        }
    return {
        "success": True,
        "target_name": target_name,
        "ra": coords["ra"],
        "dec": coords["dec"],
        "error_msg": None,
    }
