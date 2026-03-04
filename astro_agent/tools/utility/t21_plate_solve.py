"""
T21 — plate_solve

Determine precise celestial coordinates and pixel scale via astrometric plate
solving. This is a standalone tool exposing the plate-solve capability
independently of color calibration. T10 calls this internally; T21 is for
cases where the agent needs WCS data without immediately running color
calibration (e.g., to get pixel_scale_arcsec for deconvolution PSF sizing).

Reuses _build_platesolve_cmd and _parse_plate_solve_result from T10.

Siril commands (verified against Siril 1.4 CLI docs):
    load <stem>
    platesolve [ra dec] [-focal=<fl>] [-pixelsize=<px>] [-force] [-localasnet]
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import SirilError, run_siril_script
from astro_agent.tools.linear.t10_color_calibrate import (
    _parse_plate_solve_result,
    resolve_pixel_size,
)


# ── Pydantic input schema ──────────────────────────────────────────────────────

class PlateSolveInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description="Absolute path to the FITS image to plate solve."
    )
    approximate_coords: dict | None = Field(
        default=None,
        description=(
            "Hint coordinates for the image center: {'ra': float, 'dec': float} "
            "in decimal degrees (J2000). Not required but significantly improves "
            "success rate and speed. Obtain from acquisition metadata or target name lookup."
        ),
    )
    focal_length_mm: float | None = Field(
        default=None,
        description=(
            "Imaging focal length in mm. Required for scale-constrained solving. "
            "From acquisition_meta.focal_length_mm."
        ),
    )
    pixel_size_um: float | None = Field(
        default=None,
        description=(
            "Pixel size in microns. If null, resolved from PIXEL_SIZE_UM env var "
            "or camera model lookup table."
        ),
    )
    camera_model: str | None = Field(
        default=None,
        description="Camera model string for pixel size lookup when pixel_size_um is null.",
    )
    force_resolve: bool = Field(
        default=False,
        description=(
            "Force plate solving even if WCS is already present in the FITS header. "
            "Use when WCS data is suspected to be inaccurate."
        ),
    )
    use_local_astrometry_net: bool = Field(
        default=False,
        description=(
            "Use local Astrometry.net installation for blind solving. "
            "More powerful for fields with missing coordinate hints but requires "
            "local index files installed."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_cmd(
    focal_length_mm: float | None,
    pixel_size_um: float | None,
    approximate_coords: dict | None,
    force_resolve: bool,
    use_local_astrometry_net: bool,
) -> str:
    """Build the Siril platesolve command string."""
    parts = ["platesolve"]

    # Positional coords come first (verified against Siril 1.4 docs)
    if approximate_coords:
        ra = approximate_coords.get("ra")
        dec = approximate_coords.get("dec")
        if ra is not None and dec is not None:
            parts.append(f"{ra} {dec}")

    if focal_length_mm is not None:
        parts.append(f"-focal={focal_length_mm}")
    if pixel_size_um is not None:
        parts.append(f"-pixelsize={pixel_size_um}")
    if force_resolve:
        parts.append("-force")
    if use_local_astrometry_net:
        parts.append("-localasnet")

    return " ".join(parts)


def _parse_field_of_view(stdout: str) -> dict | None:
    """Parse field of view dimensions from Siril plate solve output."""
    m_w = re.search(r"field.*?width[^\d]*([\d.]+)\s*['\"]", stdout, re.IGNORECASE)
    m_h = re.search(r"field.*?height[^\d]*([\d.]+)\s*['\"]", stdout, re.IGNORECASE)
    if m_w and m_h:
        return {
            "width_arcmin": float(m_w.group(1)),
            "height_arcmin": float(m_h.group(1)),
        }
    return None


def _parse_rotation(stdout: str) -> float | None:
    """Parse image rotation from Siril plate solve output."""
    m = re.search(r"rotation[^\d\-]*([\-\d.]+)\s*deg", stdout, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=PlateSolveInput)
def plate_solve(
    working_dir: str,
    image_path: str,
    approximate_coords: dict | None = None,
    focal_length_mm: float | None = None,
    pixel_size_um: float | None = None,
    camera_model: str | None = None,
    force_resolve: bool = False,
    use_local_astrometry_net: bool = False,
) -> dict:
    """
    Astrometric plate solving — determines the celestial coordinates and
    pixel scale (arcsec/pixel) of the image.

    Called automatically by color_calibrate (T10) before PCC/SPCC. Call T21
    directly when WCS data is needed without color calibration — most commonly
    to get pixel_scale_arcsec for deconvolution PSF sizing in T13.

    If approximate_coords are unknown, try use_local_astrometry_net=True for
    blind solving (requires local index files). If solving fails, check that
    focal_length_mm and pixel_size_um are correct — wrong scale is the most
    common cause of failure.

    Returns pixel_scale_arcsec in the output — this value is critical input
    for T13 deconvolution (determines PSF star size in pixels).
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resolve pixel size
    resolved_px_um: float | None = None
    try:
        resolved_px_um = resolve_pixel_size(pixel_size_um, camera_model)
    except ValueError:
        pass  # Proceed without pixel size if unavailable

    cmd = _build_cmd(
        focal_length_mm=focal_length_mm,
        pixel_size_um=resolved_px_um,
        approximate_coords=approximate_coords,
        force_resolve=force_resolve,
        use_local_astrometry_net=use_local_astrometry_net,
    )

    try:
        result = run_siril_script(
            [f"load {img_path.stem}", cmd],
            working_dir=working_dir,
            timeout=120,
        )
        wcs_info = _parse_plate_solve_result(result)
        fov = _parse_field_of_view(result.stdout)
        rotation = _parse_rotation(result.stdout)

        return {
            "success": True,
            "wcs_coords": {
                "ra": wcs_info.get("ra"),
                "dec": wcs_info.get("dec"),
            },
            "pixel_scale_arcsec": wcs_info.get("pixel_scale_arcsec"),
            "field_of_view": fov,
            "rotation_deg": rotation,
            "error_msg": None,
        }

    except SirilError as exc:
        stdout_lower = exc.result.stdout.lower() + exc.result.stderr.lower()
        if any(kw in stdout_lower for kw in ("plate", "wcs", "astrometry", "solve", "match")):
            return {
                "success": False,
                "wcs_coords": None,
                "pixel_scale_arcsec": None,
                "field_of_view": None,
                "rotation_deg": None,
                "error_msg": (
                    f"Plate solving failed. Suggestions: provide approximate_coords, "
                    f"verify focal_length_mm and pixel_size_um, or try "
                    f"use_local_astrometry_net=True. "
                    f"Siril: {exc.result.stderr[:300]}"
                ),
            }
        raise
