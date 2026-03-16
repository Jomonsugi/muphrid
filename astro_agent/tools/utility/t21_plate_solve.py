"""
T21 — plate_solve

Determine precise celestial coordinates and pixel scale via astrometric plate
solving. This is a standalone tool exposing the plate-solve capability
independently of color calibration. T10 calls the shared _build_platesolve_cmd
helper; T21 is for cases where the agent needs WCS data without immediately
running color calibration (e.g. pixel_scale_arcsec for deconvolution PSF sizing).

Siril commands (verified against Siril 1.4 CLI docs):
    platesolve [-force] [image_center_coords] [-focal=] [-pixelsize=]
               [-noflip] [-downscale] [-order=] [-radius=] [-disto=]
               [-limitmag=[+-]] [-catalog=] [-nocrop]
               [-localasnet [-blindpos] [-blindres]]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from astro_agent.equipment import resolve_pixel_size, resolve_target_coords
from astro_agent.graph.state import AstroState
from astro_agent.tools._siril import SirilError, run_siril_script
from astro_agent.tools.linear.t10_color_calibrate import _parse_plate_solve_result


# ── Pydantic input schema ──────────────────────────────────────────────────────

class PlateSolveInput(BaseModel):
    target_name: str | None = Field(
        default=None,
        description=(
            "Astronomical target name resolved via SIMBAD to RA/DEC (e.g. 'M42'). "
            "Used as position hint when approximate_coords is not provided. "
            "Prefer calling T29 resolve_target first and passing approximate_coords explicitly."
        ),
    )
    approximate_coords: dict | None = Field(
        default=None,
        description=(
            "Hint coordinates for the image center: {'ra': float, 'dec': float} "
            "in decimal degrees (J2000). Significantly improves success rate and "
            "speed. Takes precedence over target_name."
        ),
    )
    focal_length_mm: float | None = Field(
        default=None,
        description="Imaging focal length in mm. Overrides image/settings values.",
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
        description="Force a new solve even if WCS already present in the FITS header.",
    )
    no_flip: bool = Field(
        default=False,
        description=(
            "Do not auto-flip the image if detected as upside-down. "
            "Use when you know the orientation is correct."
        ),
    )
    downscale: bool = Field(
        default=False,
        description=(
            "Downsample the image for faster star detection. "
            "Use for large images (> 6000px wide) where solving is slow."
        ),
    )
    sip_order: int | None = Field(
        default=None,
        description=(
            "SIP distortion polynomial order (1–5). Higher orders model more "
            "distortion but need more stars. Default from Siril preferences."
        ),
    )
    search_radius: float | None = Field(
        default=None,
        description=(
            "Cone search radius in degrees for near-search when initial solve fails. "
            "0 disables near search. Default from Siril preferences."
        ),
    )
    save_disto: str | None = Field(
        default=None,
        description="Save the plate solve solution as a distortion file at this path.",
    )
    limitmag: str | None = Field(
        default=None,
        description=(
            "Override automatic star magnitude limit. "
            "'+2': deeper (more stars). '-2': shallower (brighter only). "
            "'12': absolute magnitude limit."
        ),
    )
    catalog: str | None = Field(
        default=None,
        description=(
            "Force a specific star catalog: 'tycho2', 'nomad', 'localgaia', "
            "'gaia', 'ppmxl', 'brightstars', 'apass'. "
            "Default: auto-selected based on FOV and magnitude."
        ),
    )
    no_crop: bool = Field(
        default=False,
        description=(
            "Disable center crop for wide-field images (FOV > 5°). "
            "Without this, Siril crops to center for star detection."
        ),
    )
    use_local_astrometry_net: bool = Field(
        default=False,
        description=(
            "Use local Astrometry.net solve-field for solving. More powerful "
            "for unknown fields but requires local index files installed."
        ),
    )
    blind_pos: bool = Field(
        default=False,
        description=(
            "Solve blindly for position (with -localasnet). "
            "Use when image location is completely unknown."
        ),
    )
    blind_res: bool = Field(
        default=False,
        description=(
            "Solve blindly for resolution (with -localasnet). "
            "Use when image sampling/scale is completely unknown."
        ),
    )
    findstar: dict | None = Field(
        default=None,
        description=(
            "setfindstar options to apply before platesolve. Accepts the same fields "
            "as T04's findstar parameter: sigma, relax, radius, roundness, convergence, "
            "profile, focal, pixelsize, reset. "
            "Example: {'sigma': 0.5, 'relax': True}. "
            "Use when platesolve reports 'not enough stars detected'."
        ),
    )


# ── Command builder ────────────────────────────────────────────────────────────

def build_platesolve_cmd(
    focal_length_mm: float | None = None,
    pixel_size_um: float | None = None,
    approximate_coords: dict | None = None,
    force_resolve: bool = False,
    no_flip: bool = False,
    downscale: bool = False,
    sip_order: int | None = None,
    search_radius: float | None = None,
    save_disto: str | None = None,
    limitmag: str | None = None,
    catalog: str | None = None,
    no_crop: bool = False,
    use_local_astrometry_net: bool = False,
    blind_pos: bool = False,
    blind_res: bool = False,
) -> str:
    """Build the Siril platesolve command string with all available options."""
    parts = ["platesolve"]

    # Siril's parser checks for coordinates FIRST (word[1] not starting with '-'
    # or negative number), then processes flags. -force must come AFTER coords.
    # Source: src/core/command.c — coords parsed before the flags while-loop.
    # The official docs show -force first, but that contradicts the source.
    if approximate_coords:
        ra = approximate_coords.get("ra")
        dec = approximate_coords.get("dec")
        if ra is not None and dec is not None:
            parts.append(f"{ra} {dec}")

    if force_resolve:
        parts.append("-force")

    if focal_length_mm is not None and focal_length_mm > 0:
        parts.append(f"-focal={focal_length_mm}")
    if pixel_size_um is not None and pixel_size_um > 0:
        parts.append(f"-pixelsize={pixel_size_um}")
    if no_flip:
        parts.append("-noflip")
    if downscale:
        parts.append("-downscale")
    if sip_order is not None:
        parts.append(f"-order={sip_order}")
    if search_radius is not None:
        parts.append(f"-radius={search_radius}")
    if save_disto is not None:
        parts.append(f"-disto={save_disto}")
    if limitmag is not None:
        parts.append(f"-limitmag={limitmag}")
    if catalog is not None:
        parts.append(f"-catalog={catalog}")
    if no_crop:
        parts.append("-nocrop")
    if use_local_astrometry_net:
        parts.append("-localasnet")
        if blind_pos:
            parts.append("-blindpos")
        if blind_res:
            parts.append("-blindres")

    return " ".join(parts)


def _parse_field_of_view(stdout: str) -> dict | None:
    m_w = re.search(r"field.*?width[^\d]*([\d.]+)\s*['\"]", stdout, re.IGNORECASE)
    m_h = re.search(r"field.*?height[^\d]*([\d.]+)\s*['\"]", stdout, re.IGNORECASE)
    if m_w and m_h:
        return {
            "width_arcmin": float(m_w.group(1)),
            "height_arcmin": float(m_h.group(1)),
        }
    return None


def _parse_rotation(stdout: str) -> float | None:
    m = re.search(r"rotation[^\d\-]*([\-\d.]+)\s*deg", stdout, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=PlateSolveInput)
def plate_solve(
    target_name: str | None = None,
    approximate_coords: dict | None = None,
    focal_length_mm: float | None = None,
    pixel_size_um: float | None = None,
    camera_model: str | None = None,
    force_resolve: bool = False,
    no_flip: bool = False,
    downscale: bool = False,
    sip_order: int | None = None,
    search_radius: float | None = None,
    save_disto: str | None = None,
    limitmag: str | None = None,
    catalog: str | None = None,
    no_crop: bool = False,
    use_local_astrometry_net: bool = False,
    blind_pos: bool = False,
    blind_res: bool = False,
    findstar: dict | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Astrometric plate solving — determines celestial coordinates and pixel
    scale (arcsec/pixel) of the image.

    Returned measured_focal_length_mm is derived from the measured plate scale and
    the known pixel size (focal_mm = pixel_um × 206.265 / plate_scale_arcsec). This
    is more accurate than the manufacturer's nominal focal length.

    Troubleshooting failed solves:
      - Provide approximate_coords (even rough RA/DEC helps enormously)
      - Verify focal_length_mm and pixel_size_um (wrong scale = #1 failure cause)
      - Try downscale=True for large images (> 6000px)
      - Try a different catalog (gaia, nomad, tycho2)
      - Increase limitmag ('+2' or '+3') for sparse fields
      - Try use_local_astrometry_net=True with blind_pos/blind_res for unknown fields
      - Set search_radius to widen the cone search
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    resolved_px_um: float | None = None
    try:
        resolved_px_um = resolve_pixel_size(pixel_size_um)
    except ValueError:
        pass

    # Resolve position hint: explicit coords > target_name SIMBAD lookup
    resolved_coords = approximate_coords
    if resolved_coords is None and target_name:
        resolved_coords = resolve_target_coords(target_name)

    cmd = build_platesolve_cmd(
        focal_length_mm=focal_length_mm,
        pixel_size_um=resolved_px_um,
        approximate_coords=resolved_coords,
        force_resolve=force_resolve,
        no_flip=no_flip,
        downscale=downscale,
        sip_order=sip_order,
        search_radius=search_radius,
        save_disto=save_disto,
        limitmag=limitmag,
        catalog=catalog,
        no_crop=no_crop,
        use_local_astrometry_net=use_local_astrometry_net,
        blind_pos=blind_pos,
        blind_res=blind_res,
    )

    script: list[str] = [f"load {img_path.stem}"]
    if findstar is not None:
        from astro_agent.tools.preprocess.t04_register import (
            SetFindStarOptions,
            _build_setfindstar_cmd,
        )
        fs_cmd = _build_setfindstar_cmd(SetFindStarOptions(**findstar))
        if fs_cmd:
            script.append(fs_cmd)
    script.append(cmd)

    try:
        result = run_siril_script(script, working_dir=working_dir, timeout=120)
        wcs_info = _parse_plate_solve_result(result)
        fov = _parse_field_of_view(result.stdout)
        rotation = _parse_rotation(result.stdout)

        measured_fl = wcs_info.get("measured_focal_length_mm")
        coords = {"ra": wcs_info.get("ra"), "dec": wcs_info.get("dec")}
        pixel_scale = wcs_info.get("pixel_scale_arcsec")

        summary = {
            "status": "solved",
            "ra": coords.get("ra"),
            "dec": coords.get("dec"),
            "pixel_scale_arcsec": pixel_scale,
            "measured_focal_length_mm": measured_fl,
            "field_of_view": fov,
            "rotation_deg": rotation,
            "input_focal_length_mm": focal_length_mm,
            "input_pixel_size_um": resolved_px_um,
            "resolved_coords_hint": resolved_coords,
        }

        return Command(update={
            "metadata": {
                **state["metadata"],
                "plate_solve_coords": coords,
                "pixel_scale": pixel_scale,
            },
            "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
        })

    except SirilError as exc:
        stdout_lower = exc.result.stdout.lower() + exc.result.stderr.lower()
        if any(kw in stdout_lower for kw in ("plate", "wcs", "astrometry", "solve", "match")):
            failure_summary = {
                "status": "failed",
                "error": "plate solving failed — WCS not found",
                "input_focal_length_mm": focal_length_mm,
                "input_pixel_size_um": resolved_px_um,
                "resolved_coords_hint": resolved_coords,
                "siril_stdout_tail": exc.result.stdout[-500:] if exc.result.stdout else None,
            }
            return Command(update={
                "metadata": {
                    **state["metadata"],
                    "plate_solve_coords": None,
                    "pixel_scale": None,
                },
                "messages": [ToolMessage(content=json.dumps(failure_summary, indent=2, default=str), tool_call_id=tool_call_id)],
            })
        raise
