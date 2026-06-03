"""
color_calibrate

Correct white balance by matching star colors to photometric catalogs.
Siril's PCC/SPCC always include background neutralization as part of the
operation — it cannot be disabled.

Plate solving is attempted internally if WCS is not already present in the
FITS header. The tool reads imaging focal length, pixel size, and target
coordinates from state.dataset.acquisition_meta. equipment.toml seeds those
fields at ingest_dataset time (CLI) or the Gradio Equipment tab passes
overrides at ingest time (UI). Mid-run hardware editing is not supported.

State contract:
  Required (refused when null):
    acquisition_meta.focal_length_mm — set by ingest from equipment.toml /
      FITS-EXIF; refined by plate_solve when it runs first.
    acquisition_meta.pixel_size_um  — set by ingest from equipment.toml /
      FITS-EXIF.
  Optional:
    acquisition_meta.target_coords  — set by resolve_target. Used as a
      position hint for blind plate solving when present.
"""

from __future__ import annotations

import json
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
from muphrid.tools._siril import SirilError, SirilResult, run_siril_script


# ── Pydantic input schemas ────────────────────────────────────────────────────

class SpccNarrowbandOptions(BaseModel):
    """Narrowband filter parameters for SPCC -narrowband mode."""
    r_wavelength: float | None = Field(
        default=None,
        description="Red filter center wavelength in nm (-rwl=).",
    )
    r_bandwidth: float | None = Field(
        default=None,
        description="Red filter bandwidth in nm (-rbw=).",
    )
    g_wavelength: float | None = Field(
        default=None,
        description="Green filter center wavelength in nm (-gwl=).",
    )
    g_bandwidth: float | None = Field(
        default=None,
        description="Green filter bandwidth in nm (-gbw=).",
    )
    b_wavelength: float | None = Field(
        default=None,
        description="Blue filter center wavelength in nm (-bwl=).",
    )
    b_bandwidth: float | None = Field(
        default=None,
        description="Blue filter bandwidth in nm (-bbw=).",
    )


class SpccAtmosphericOptions(BaseModel):
    """Atmospheric extinction correction params for SPCC -atmos."""
    obs_height: float | None = Field(
        default=None,
        description="Observer height above sea level in metres (-obsheight=). Default: 10.",
    )
    pressure: float | None = Field(
        default=None,
        description=(
            "Local atmospheric pressure at observing site in hPa (-pressure=). "
            "Mutually exclusive with sea_level_pressure."
        ),
    )
    sea_level_pressure: float | None = Field(
        default=None,
        description=(
            "Sea-level atmospheric pressure in hPa (-slp=). Default: 1013.25. "
            "Mutually exclusive with pressure."
        ),
    )


class SpccOptions(BaseModel):
    """Configuration for Spectrophotometric Color Calibration (SPCC).

    OSC-only. Mono camera support is on the backlog (see BACKLOG.md
    item 12); the SPCC database entries for mono sensors and per-channel
    filters are not currently wired through state and equipment.toml.
    """
    # OSC mode
    osc_sensor_name: str | None = Field(
        default=None,
        description=(
            "OSC sensor/camera name as in Siril's SPCC database. "
            "Use spcc_list oscsensor to see available names."
        ),
    )
    osc_filter_name: str | None = Field(
        default=None,
        description="OSC filter name as in Siril's SPCC database.",
    )
    osc_lpf: str | None = Field(
        default=None,
        description="OSC light pollution filter name.",
    )

    white_reference: str | None = Field(
        default=None,
        description=(
            "White reference target for SPCC (-whiteref=). "
            "Example: 'Average Spiral Galaxy'. Use spcc_list whiteref for available names."
        ),
    )

    narrowband: SpccNarrowbandOptions | None = Field(
        default=None,
        description=(
            "Narrowband filter mode. When set, sensor/filter options are ignored "
            "and filter wavelengths/bandwidths are used instead."
        ),
    )

    atmospheric: SpccAtmosphericOptions | None = Field(
        default=None,
        description=(
            "Atmospheric extinction correction. When set, -atmos is passed "
            "along with optional height and pressure parameters."
        ),
    )


class PlateSolveOptions(BaseModel):
    """Plate solving options passed through to platesolve command."""
    no_flip: bool = Field(
        default=False,
        description="Do not auto-flip the image if detected as upside-down.",
    )
    downscale: bool = Field(
        default=False,
        description="Downsample for faster star detection on large images.",
    )
    sip_order: int | None = Field(
        default=None,
        description="SIP distortion polynomial order (1–5).",
    )
    search_radius: float | None = Field(
        default=None,
        description="Cone search radius in degrees for near-search fallback.",
    )
    ps_limitmag: str | None = Field(
        default=None,
        description="Override plate solve magnitude limit (separate from PCC/SPCC limitmag).",
    )
    ps_catalog: str | None = Field(
        default=None,
        description=(
            "Force plate solve catalog: tycho2, nomad, localgaia, gaia, ppmxl, "
            "brightstars, apass."
        ),
    )
    no_crop: bool = Field(
        default=False,
        description="Disable center crop for wide-field images (FOV > 5°).",
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
    use_local_astrometry_net: bool = Field(
        default=False,
        description="Use local Astrometry.net solve-field.",
    )
    blind_pos: bool = Field(
        default=False,
        description="Blind position solve (with localasnet).",
    )
    blind_res: bool = Field(
        default=False,
        description="Blind resolution solve (with localasnet).",
    )


class ColorCalibrateInput(BaseModel):
    method: str = Field(
        default="pcc",
        description=(
            "'pcc': Photometric Color Calibration using catalog stars. "
            "Reliable for all OSC/DSLR images. "
            "'spcc': Spectrophotometric Color Calibration — more accurate, "
            "models the sensor's spectral response. Configure via spcc_options."
        ),
    )
    catalog: str = Field(
        default="gaia",
        description="Star catalog for PCC: gaia, nomad, apass, localgaia.",
    )
    limitmag: str | None = Field(
        default=None,
        description=(
            "Override auto star magnitude limit for PCC/SPCC catalog matching. "
            "'+2': deeper. '-2': shallower. '12': absolute limit."
        ),
    )
    bgtol_lower: float | None = Field(
        default=None,
        description=(
            "Background sample lower rejection tolerance in sigma units. "
            "Default: -2.8. Less negative = more aggressive background exclusion."
        ),
    )
    bgtol_upper: float | None = Field(
        default=None,
        description=(
            "Background sample upper rejection tolerance in sigma units. "
            "Default: 2.0."
        ),
    )
    spcc_options: SpccOptions = Field(default_factory=SpccOptions)
    platesolve_options: PlateSolveOptions = Field(default_factory=PlateSolveOptions)


# ── Command builders ──────────────────────────────────────────────────────────

def _build_platesolve_cmds_for_color_calibrate(
    focal_length_mm: float,
    pixel_size_um: float,
    target_coords: dict | None,
    ps_opts: PlateSolveOptions,
) -> list[str]:
    """Build [optional setfindstar, platesolve] commands for color_calibrate."""
    from muphrid.tools.preprocess.register import (
        SetFindStarOptions,
        _build_setfindstar_cmd,
    )
    from muphrid.tools.utility.plate_solve import build_platesolve_cmd

    cmds: list[str] = []

    if ps_opts.findstar is not None:
        fs_opts = SetFindStarOptions(**ps_opts.findstar)
        fs_cmd = _build_setfindstar_cmd(fs_opts)
        if fs_cmd:
            cmds.append(fs_cmd)

    cmds.append(build_platesolve_cmd(
        focal_length_mm=focal_length_mm,
        pixel_size_um=pixel_size_um,
        approximate_coords=target_coords,
        no_flip=ps_opts.no_flip,
        downscale=ps_opts.downscale,
        sip_order=ps_opts.sip_order,
        search_radius=ps_opts.search_radius,
        limitmag=ps_opts.ps_limitmag,
        catalog=ps_opts.ps_catalog,
        no_crop=ps_opts.no_crop,
        use_local_astrometry_net=ps_opts.use_local_astrometry_net,
        blind_pos=ps_opts.blind_pos,
        blind_res=ps_opts.blind_res,
    ))
    return cmds


def _build_pcc_cmd(
    catalog: str,
    limitmag: str | None = None,
    bgtol_lower: float | None = None,
    bgtol_upper: float | None = None,
) -> str:
    cmd = f"pcc -catalog={catalog}"
    if limitmag is not None:
        cmd += f" -limitmag={limitmag}"
    if bgtol_lower is not None or bgtol_upper is not None:
        lo = bgtol_lower if bgtol_lower is not None else -2.8
        hi = bgtol_upper if bgtol_upper is not None else 2.0
        cmd += f" -bgtol={lo},{hi}"
    return cmd


def _build_spcc_cmd(
    opts: SpccOptions,
    limitmag: str | None = None,
    bgtol_lower: float | None = None,
    bgtol_upper: float | None = None,
) -> str:
    cmd = "spcc"

    # OSC sensor/filter configuration. Mono is on the backlog (item 12).
    if opts.osc_sensor_name:
        cmd += f' "-oscsensor={opts.osc_sensor_name}"'
        if opts.osc_filter_name:
            cmd += f' "-oscfilter={opts.osc_filter_name}"'
        if opts.osc_lpf:
            cmd += f' "-osclpf={opts.osc_lpf}"'

    if opts.white_reference:
        cmd += f' "-whiteref={opts.white_reference}"'

    # Narrowband mode
    if opts.narrowband:
        cmd += " -narrowband"
        nb = opts.narrowband
        if nb.r_wavelength is not None:
            cmd += f" -rwl={nb.r_wavelength}"
        if nb.r_bandwidth is not None:
            cmd += f" -rbw={nb.r_bandwidth}"
        if nb.g_wavelength is not None:
            cmd += f" -gwl={nb.g_wavelength}"
        if nb.g_bandwidth is not None:
            cmd += f" -gbw={nb.g_bandwidth}"
        if nb.b_wavelength is not None:
            cmd += f" -bwl={nb.b_wavelength}"
        if nb.b_bandwidth is not None:
            cmd += f" -bbw={nb.b_bandwidth}"

    if limitmag is not None:
        cmd += f" -limitmag={limitmag}"

    if bgtol_lower is not None or bgtol_upper is not None:
        lo = bgtol_lower if bgtol_lower is not None else -2.8
        hi = bgtol_upper if bgtol_upper is not None else 2.0
        cmd += f" -bgtol={lo},{hi}"

    # Atmospheric correction
    if opts.atmospheric:
        cmd += " -atmos"
        atm = opts.atmospheric
        if atm.obs_height is not None:
            cmd += f" -obsheight={atm.obs_height}"
        if atm.pressure is not None:
            cmd += f" -pressure={atm.pressure}"
        elif atm.sea_level_pressure is not None:
            cmd += f" -slp={atm.sea_level_pressure}"

    return cmd


def _read_wcs_from_fits(fits_path: Path, pixel_size_um: float | None = None) -> dict:
    """
    Canonical post-platesolve WCS reader: pull the standard FITS keywords
    Siril writes via WCSLIB straight out of the file.

    After `platesolve` succeeds, Siril writes the standard FITS WCS
    keywords — CRVAL1/CRVAL2 (RA/Dec at the reference pixel, in decimal
    degrees), the CD matrix (CD1_1/CD1_2/CD2_1/CD2_2 in degrees per pixel),
    and PLTSOLVD = T as the success flag — directly to the saved FITS
    header. Reading them via `astropy.wcs.WCS` is locale-, version-, and
    log-format-agnostic. This is THE interface for consuming a Siril
    plate-solve; there is no regex fallback because falling back to
    scraping stdout was the bug we just retired.

    On a real failure (FITS missing, header malformed, no WCS keys
    present, or astropy import broken) the returned dict carries
    `wcs_read_error` and all-None values. That's a real condition the
    caller surfaces to the agent — it is not papered over.

    Sources:
      * https://siril.readthedocs.io/en/stable/astrometry/platesolving.html
      * https://siril.readthedocs.io/en/latest/FITS-header.html
    """
    import math

    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError:
        return {
            "ra": None, "dec": None,
            "pixel_scale_arcsec": None,
            "measured_focal_length_mm": None,
            "wcs_read_error": "astropy not installed",
        }

    try:
        with fits.open(str(fits_path)) as hdul:
            header = hdul[0].header
            pltsolvd = bool(header.get("PLTSOLVD", False))

            # Siril can write solved WCS keywords onto RGB FITS files where
            # the data/header still has NAXIS=3 (color plane + two spatial
            # axes). Astropy rejects SIP/distortion tables on a full 3D WCS
            # with "distortions only work in 2 dimensions", even though the
            # celestial solution itself is valid. Ask explicitly for the
            # 2D celestial WCS so plate_solve/color_calibrate work on RGB
            # products as well as mono/2D FITS.
            wcs = WCS(header, naxis=2)
            crval = wcs.wcs.crval if wcs.has_celestial else None

            ra = float(crval[0]) if crval is not None and len(crval) >= 1 else None
            dec = float(crval[1]) if crval is not None and len(crval) >= 2 else None

            # Pixel scale: |det(CD)|^0.5 × 3600 gives arcsec/pixel and is
            # invariant under image rotation. astropy's pixel_scale_matrix
            # handles both CD- and PC+CDELT-form headers transparently.
            pixel_scale = None
            try:
                psm = wcs.pixel_scale_matrix
                # psm is in degrees/pixel; take the geometric mean of the
                # two axis scales to handle anisotropic pixels gracefully.
                if psm is not None and psm.shape == (2, 2):
                    det = abs(psm[0][0] * psm[1][1] - psm[0][1] * psm[1][0])
                    pixel_scale = math.sqrt(det) * 3600.0
            except Exception:
                pass

            # Measured focal length: derived from pixel scale and the
            # known sensor pixel size via the standard small-angle formula
            # f_mm = (px_um × 206.265) / scale_arcsec_per_pixel
            # Siril doesn't write a FOCALLEN keyword on platesolve (the
            # info is implicit in the CD matrix), so we compute it.
            measured_focal_length_mm = None
            if pixel_scale and pixel_size_um:
                measured_focal_length_mm = (pixel_size_um * 206.265) / pixel_scale

            return {
                "ra": ra,
                "dec": dec,
                "pixel_scale_arcsec": pixel_scale,
                "measured_focal_length_mm": measured_focal_length_mm,
                "pltsolvd": pltsolvd,
                "wcs_source": "fits_header",
            }
    except Exception as e:
        return {
            "ra": None, "dec": None,
            "pixel_scale_arcsec": None,
            "measured_focal_length_mm": None,
            "wcs_read_error": f"{type(e).__name__}: {e}",
        }


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=ColorCalibrateInput)
def color_calibrate(
    method: str = "pcc",
    catalog: str = "gaia",
    limitmag: str | None = None,
    bgtol_lower: float | None = None,
    bgtol_upper: float | None = None,
    spcc_options: SpccOptions | None = None,
    platesolve_options: PlateSolveOptions | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Correct white balance by matching star colors to photometric catalogs.
    Performs background neutralization and photometric or spectrophotometric
    color calibration.

    Plate solving is attempted internally when WCS is not already present in
    the FITS header. Imaging focal length, pixel size, and (optional) target
    coordinates are read from state.dataset.acquisition_meta — they are
    populated by ingest_dataset and (for focal length) potentially refined
    by an earlier plate_solve call.

    Method guidance:
      pcc — reliable for all OSC/DSLR images, uses Gaia catalog by default.
      spcc — more accurate, models sensor spectral response. OSC only;
        mono support is on the backlog.

    SPCC modes (configured via spcc_options):
      OSC broadband: set osc_sensor_name, optionally osc_filter_name + osc_lpf.
      Narrowband: set narrowband with wavelengths/bandwidths per channel.
      Atmospheric: set atmospheric for extinction correction at low elevations.
      White ref: set white_reference to target a specific reference spectrum.

    Plate solve options (configured via platesolve_options):
      downscale, sip_order, search_radius, catalog, limitmag, no_crop,
      local astrometry.net with blind_pos / blind_res.

    Raises ValueError when required state is null
    (acquisition_meta.focal_length_mm or acquisition_meta.pixel_size_um).
    Raises RuntimeError when plate solving fails on the input image; retry
    with adjusted platesolve_options (lower sigma, relax=True) or first
    call resolve_target so acquisition_meta.target_coords carries a hint.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    if spcc_options is None:
        spcc_options = SpccOptions()
    if platesolve_options is None:
        platesolve_options = PlateSolveOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # State-canonical equipment reads. equipment_from_state refuses cleanly
    # with a state-naming message when required fields are null.
    acq = equipment_from_state(
        state, require=("focal_length_mm", "pixel_size_um"),
    )
    resolved_fl = float(acq["focal_length_mm"])
    px_size = float(acq["pixel_size_um"])
    resolved_coords = acq.get("target_coords")

    stem = img_path.stem
    output_stem = f"{stem}_cc"

    platesolve_cmds = _build_platesolve_cmds_for_color_calibrate(
        resolved_fl, px_size, resolved_coords, platesolve_options
    )
    platesolve_cmd = platesolve_cmds[-1] # for diagnostics

    if method == "spcc":
        cal_cmd = _build_spcc_cmd(spcc_options, limitmag, bgtol_lower, bgtol_upper)
    else:
        cal_cmd = _build_pcc_cmd(catalog, limitmag, bgtol_lower, bgtol_upper)

    commands = [f"load {stem}", *platesolve_cmds, cal_cmd, f"save {output_stem}"]

    wcs_info: dict = {}

    try:
        result = run_siril_script(commands, working_dir=working_dir, timeout=180)
    except SirilError as exc:
        stdout_lower = exc.result.stdout.lower() + exc.result.stderr.lower()
        if "plate" in stdout_lower or "wcs" in stdout_lower or "astrometry" in stdout_lower:
            raise RuntimeError(
                f"Plate solving failed — cannot color calibrate.\n"
                f"platesolve_cmd={platesolve_cmd!r}\n"
                f"stdout={exc.result.stdout[-600:]!r}\n"
                f"stderr={exc.result.stderr[-400:]!r}\n"
                "Retry with adjusted platesolve_options (lower sigma, relax=True). "
                "If acquisition_meta.target_coords is null, call resolve_target "
                "to populate a position hint, or correct "
                "acquisition_meta.pixel_size_um / acquisition_meta.focal_length_mm "
                "in equipment.toml."
            ) from exc
        raise

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(
            f"Color calibration did not produce: {output_path}"
        )

    # Read WCS straight from the FITS header Siril just wrote. This is
    # the canonical post-platesolve interface (CRVAL1/CRVAL2, CD matrix,
    # PLTSOLVD = T) — we never scrape stdout. If the read fails, the
    # returned dict carries `wcs_read_error` and the agent sees the real
    # failure rather than a stale-regex-derived value.
    wcs_info = _read_wcs_from_fits(output_path, pixel_size_um=px_size)

    summary = {
        "output_path": str(output_path),
        "method": method.upper(),
        "catalog": catalog if method == "pcc" else None,
        "wcs_info": wcs_info,
        "resolved_focal_length_mm": resolved_fl,
        "resolved_pixel_size_um": px_size,
        "resolved_coords": resolved_coords,
        "limitmag": limitmag,
    }

    # Discrepancy reporting: inform if measured focal length differs from user input
    measured_fl = wcs_info.get("measured_focal_length_mm")
    if measured_fl and resolved_fl and abs(measured_fl - resolved_fl) > 1.0:
        summary["focal_length_note"] = (
            f"Plate-solve measured {measured_fl:.1f}mm, which differs from the "
            f"provided {resolved_fl:.1f}mm. The measured value is more accurate. "
            f"Consider updating the Equipment tab for future runs."
        )

    return Command(update={
        "paths": {"current_image": str(output_path)},
        # PCC/SPCC apply per-channel linear scaling (white balance) and
        # background neutralization. Both operations are linear in pixel
        # value space and produce linear output. See Metadata.image_space.
        "metadata": {"image_space": "linear"},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
