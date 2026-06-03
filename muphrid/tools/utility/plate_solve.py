"""
plate_solve

Determine precise celestial coordinates and pixel scale via astrometric plate
solving. This is a standalone tool exposing the plate-solve capability
independently of color calibration. color_calibrate calls the shared _build_platesolve_cmd
helper; plate_solve is for cases where the agent needs WCS data without immediately
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

from muphrid.equipment import equipment_from_state
from muphrid.graph.state import AstroState
from muphrid.tools._siril import SirilError, run_siril_script, siril_script_path
from muphrid.tools.linear.color_calibrate import _read_wcs_from_fits


# ── Pydantic input schema ──────────────────────────────────────────────────────

class PlateSolveInput(BaseModel):
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
            "as siril_register's findstar parameter: sigma, relax, radius, roundness, convergence, "
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
        # save_disto is expected to be pre-sanitized by the caller via
        # siril_script_path() so it contains no whitespace. Builder stays
        # pure (no working_dir dependency).
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

    State contract:
      Required (refused when null):
        acquisition_meta.focal_length_mm — needed to constrain the search.
        acquisition_meta.pixel_size_um   — needed to translate plate scale
          to focal length and to convert star sizes to angular size.
      Optional:
        acquisition_meta.target_coords   — used as a position hint that
          greatly speeds and stabilizes the solve. Call resolve_target first
          if null.

    On success, plate_solve writes the measured focal length back to
    acquisition_meta.focal_length_mm (single field — plate-solve wins over
    equipment.toml). The ToolMessage announces the mutation so the agent's
    reasoning stays current.

    Troubleshooting failed solves:
      - First ensure acquisition_meta.target_coords is populated (run
        resolve_target). A rough RA/DEC hint helps enormously.
      - Try downscale=True for large images (> 6000px).
      - Try a different catalog (gaia, nomad, tycho2).
      - Increase limitmag ('+2' or '+3') for sparse fields.
      - Try use_local_astrometry_net=True with blind_pos/blind_res for
        unknown fields.
      - Set search_radius to widen the cone search.
      - Verify acquisition_meta.pixel_size_um and acquisition_meta.focal_length_mm
        (wrong scale = #1 failure cause). Correct in equipment.toml.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # State-canonical equipment reads. Refuses cleanly when null.
    acq = equipment_from_state(
        state, require=("focal_length_mm", "pixel_size_um"),
    )
    focal_length_mm = float(acq["focal_length_mm"])
    resolved_px_um = float(acq["pixel_size_um"])
    resolved_coords = acq.get("target_coords")

    # Rewrite save_disto to a whitespace-free reference before building the
    # command. Siril's tokenizer splits on whitespace and would truncate
    # any unsanitized path that lives under a dataset root with spaces.
    safe_save_disto = (
        siril_script_path(save_disto, working_dir) if save_disto else None
    )

    cmd = build_platesolve_cmd(
        focal_length_mm=focal_length_mm,
        pixel_size_um=resolved_px_um,
        approximate_coords=resolved_coords,
        force_resolve=force_resolve,
        no_flip=no_flip,
        downscale=downscale,
        sip_order=sip_order,
        search_radius=search_radius,
        save_disto=safe_save_disto,
        limitmag=limitmag,
        catalog=catalog,
        no_crop=no_crop,
        use_local_astrometry_net=use_local_astrometry_net,
        blind_pos=blind_pos,
        blind_res=blind_res,
    )

    # plate_solve produces NO durable output by default — Siril solves
    # in memory and discards the WCS when the script exits. We need to
    # force a save so we can read the WCS keywords (CRVAL1/CRVAL2/CD…)
    # straight out of the FITS header via astropy. Using a unique stem
    # under the working dir keeps the original image untouched (this is
    # a metadata-read tool; we don't silently mutate the user's input).
    import uuid
    temp_stem = f"_platesolve_{uuid.uuid4().hex[:8]}"
    temp_fits = Path(working_dir) / f"{temp_stem}.fit"

    script: list[str] = [f"load {img_path.stem}"]
    if findstar is not None:
        from muphrid.tools.preprocess.register import (
            SetFindStarOptions,
            _build_setfindstar_cmd,
        )
        fs_cmd = _build_setfindstar_cmd(SetFindStarOptions(**findstar))
        if fs_cmd:
            script.append(fs_cmd)
    script.append(cmd)
    script.append(f"save {temp_stem}")

    try:
        result = run_siril_script(script, working_dir=working_dir, timeout=120)

        # The .fit suffix is Siril's default; .fits is also possible if
        # FITS_EXTENSION is configured otherwise. Probe both.
        if not temp_fits.exists():
            alt = Path(working_dir) / f"{temp_stem}.fits"
            if alt.exists():
                temp_fits = alt

        # Read the canonical WCS keywords Siril wrote to the FITS header.
        wcs_info = _read_wcs_from_fits(temp_fits, pixel_size_um=resolved_px_um)
        fov = _parse_field_of_view(result.stdout)
        rotation = _parse_rotation(result.stdout)

        if wcs_info.get("ra") is None or wcs_info.get("dec") is None:
            # WCS read genuinely failed — surface the specific reason
            # (FITS missing, no WCS keys present, malformed header) so
            # the agent can see what actually went wrong.
            failure_summary = {
                "status": "failed",
                "error": "plate solving completed but WCS could not be read from FITS header",
                "wcs_read_error": wcs_info.get("wcs_read_error"),
                "pltsolvd": wcs_info.get("pltsolvd"),
                "input_focal_length_mm": focal_length_mm,
                "input_pixel_size_um": resolved_px_um,
                "resolved_coords_hint": resolved_coords,
                "siril_stdout_tail": (result.stdout or "")[-500:],
            }
            return Command(update={
                # Delta-only emit: paths/metadata reducer composes with
                # parallel siblings; never spread the existing dict.
                "metadata": {
                    "plate_solve_coords": None,
                    "pixel_scale": None,
                },
                "messages": [ToolMessage(
                    content=json.dumps(failure_summary, indent=2, default=str),
                    tool_call_id=tool_call_id,
                )],
            })

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
            "wcs_source": wcs_info.get("wcs_source"),
        }

        # Authoritative state mutation: plate solution is the most accurate
        # focal length measurement we have. Write it back to acquisition_meta
        # so subsequent scale-sensitive tools read the corrected value.
        # Single field — plate-solve wins; equipment.toml is just the
        # starting config the user typed.
        dataset_delta: dict = {}
        if measured_fl and measured_fl > 0:
            dataset_delta["acquisition_meta"] = {"focal_length_mm": float(measured_fl)}
            if focal_length_mm and abs(measured_fl - focal_length_mm) > 1.0:
                summary["state_update"] = (
                    f"Updated acquisition_meta.focal_length_mm: "
                    f"{focal_length_mm:.1f} → {measured_fl:.1f} "
                    f"(measured from plate solution; subsequent scale-sensitive "
                    f"tools will use {measured_fl:.1f})."
                )
            else:
                summary["state_update"] = (
                    f"Wrote acquisition_meta.focal_length_mm = {measured_fl:.1f} "
                    f"(measured from plate solution)."
                )

        update: dict = {
            # Delta-only emit so parallel-update composition works.
            "metadata": {
                "plate_solve_coords": coords,
                "pixel_scale": pixel_scale,
            },
            "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
        }
        if dataset_delta:
            update["dataset"] = dataset_delta
        return Command(update=update)

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
                # Delta-only emit so parallel-update composition works.
                "metadata": {
                    "plate_solve_coords": None,
                    "pixel_scale": None,
                },
                "messages": [ToolMessage(content=json.dumps(failure_summary, indent=2, default=str), tool_call_id=tool_call_id)],
            })
        raise
    finally:
        # Best-effort cleanup of the temp FITS we wrote purely for WCS
        # extraction. Failures here are non-fatal — leaving a stale
        # _platesolve_*.fit in the working dir is mildly untidy but
        # better than aborting the tool over an unlink permission error.
        try:
            if temp_fits.exists():
                temp_fits.unlink()
        except OSError:
            pass
