"""
T10 — color_calibrate

Correct white balance by matching star colors to photometric catalogs.
Includes background neutralization (sets background to neutral grey) and
optionally spectrophotometric calibration for more accurate color.

This tool requires a plate-solved image. Plate solving is attempted internally
if WCS is not already present in the FITS header.

Pixel size is resolved in this order:
  1. pixel_size_um argument (explicit override)
  2. PIXEL_SIZE_UM environment variable
  3. Built-in camera model lookup table (keyed on AcquisitionMeta.camera_model)
  4. Fail with a clear error message — plate solving cannot proceed without it

Siril commands (verified against Siril 1.4 CLI docs):
    platesolve — see T21 for full options
    pcc [-limitmag=[+-]] [-catalog=] [-bgtol=lower,upper]
    spcc [-limitmag=[+-]]
         [ { -monosensor= [-rfilter=] [-gfilter=] [-bfilter=]
           | -oscsensor= [-oscfilter=] [-osclpf=] } ]
         [-whiteref=] [-narrowband [-rwl=] [-gwl=] [-bwl=] [-rbw=] [-gbw=] [-bbw=]]
         [-bgtol=lower,upper] [-atmos [-obsheight=] { [-pressure=] | [-slp=] }]
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.equipment import resolve_focal_length as _resolve_fl
from astro_agent.equipment import resolve_pixel_size as _resolve_px
from astro_agent.equipment import resolve_target_coords as _resolve_target
from astro_agent.tools._siril import SirilError, SirilResult, run_siril_script


def resolve_pixel_size(
    pixel_size_um_arg: float | None,
    camera_model: str | None = None,
) -> float:
    """
    Resolve pixel size in microns. Priority:
      1. Explicit argument
      2. equipment.toml [camera] pixel_size_um
      3. PIXEL_SIZE_UM env var

    camera_model is accepted for backward compatibility but is no longer
    used for lookup. Maintain pixel_size_um in equipment.toml instead.
    """
    return _resolve_px(pixel_size_um_arg)


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
    """Configuration for Spectrophotometric Color Calibration (SPCC)."""
    # OSC mode
    osc_sensor_name: str | None = Field(
        default=None,
        description=(
            "OSC sensor/camera name as in Siril's SPCC database (-oscsensor=). "
            "Use spcc_list oscsensor to see available names."
        ),
    )
    osc_filter_name: str | None = Field(
        default=None,
        description="OSC filter name as in Siril's SPCC database (-oscfilter=).",
    )
    osc_lpf: str | None = Field(
        default=None,
        description="OSC light pollution filter name (-osclpf=).",
    )

    # Mono mode
    mono_sensor_name: str | None = Field(
        default=None,
        description=(
            "Mono sensor name as in Siril's SPCC database (-monosensor=). "
            "Use when imaging with a mono camera + separate RGB filters."
        ),
    )
    r_filter: str | None = Field(
        default=None,
        description="Red channel filter name for mono mode (-rfilter=).",
    )
    g_filter: str | None = Field(
        default=None,
        description="Green channel filter name for mono mode (-gfilter=).",
    )
    b_filter: str | None = Field(
        default=None,
        description="Blue channel filter name for mono mode (-bfilter=).",
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
            "as T04's findstar parameter: sigma, relax, radius, roundness, convergence, "
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
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description="Absolute path to the linear FITS image (post gradient removal)."
    )
    method: str = Field(
        default="pcc",
        description=(
            "'pcc': Photometric Color Calibration using catalog stars. "
            "Reliable for all OSC/DSLR images. "
            "'spcc': Spectrophotometric Color Calibration — more accurate, "
            "models the sensor's spectral response. Configure via spcc_options."
        ),
    )
    focal_length_mm: float = Field(
        default=0.0,
        description=(
            "Imaging focal length in mm. If 0 or omitted, resolved from "
            "equipment.toml [optics] focal_length_mm."
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
    target_name: str | None = Field(
        default=None,
        description=(
            "Astronomical target name resolved via SIMBAD to RA/DEC (e.g. 'M42'). "
            "Used as plate solve position hint when target_coords is not provided. "
            "Prefer calling T29 resolve_target first and passing target_coords explicitly."
        ),
    )
    target_coords: dict | None = Field(
        default=None,
        description=(
            "Explicit hint coordinates for plate solving: {'ra': float, 'dec': float} "
            "in decimal degrees. Takes precedence over target_name."
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
    background_neutralization: bool = Field(
        default=True,
        description=(
            "Advisory flag: Siril PCC/SPCC always include background neutralization. "
            "No CLI flag to disable. If bg neutralization without PCC/SPCC is needed "
            "(e.g. plate solve fails), use T20 + T23 pixel_math to equalize channels."
        ),
    )


# ── Command builders ──────────────────────────────────────────────────────────

def _build_platesolve_cmds_for_t10(
    focal_length_mm: float,
    pixel_size_um: float,
    target_coords: dict | None,
    ps_opts: PlateSolveOptions,
) -> list[str]:
    """Build [optional setfindstar, platesolve] commands for T10."""
    from astro_agent.tools.preprocess.t04_register import (
        SetFindStarOptions,
        _build_setfindstar_cmd,
    )
    from astro_agent.tools.utility.t21_plate_solve import build_platesolve_cmd

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

    # Mono vs OSC sensor/filter configuration
    if opts.mono_sensor_name:
        cmd += f' "-monosensor={opts.mono_sensor_name}"'
        if opts.r_filter:
            cmd += f' "-rfilter={opts.r_filter}"'
        if opts.g_filter:
            cmd += f' "-gfilter={opts.g_filter}"'
        if opts.b_filter:
            cmd += f' "-bfilter={opts.b_filter}"'
    elif opts.osc_sensor_name:
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


def _parse_plate_solve_result(result: SirilResult) -> dict:
    """Extract WCS coords and pixel scale from Siril platesolve stdout."""
    import re
    ra, dec, pixel_scale = None, None, None

    m = re.search(r"RA\s*[=:]\s*([\d.]+)", result.stdout, re.IGNORECASE)
    if m:
        ra = float(m.group(1))
    m = re.search(r"DEC?\s*[=:]\s*(-?[\d.]+)", result.stdout, re.IGNORECASE)
    if m:
        dec = float(m.group(1))
    m = re.search(r"pixel\s+scale[^=]*[=:]\s*([\d.]+)", result.stdout, re.IGNORECASE)
    if m:
        pixel_scale = float(m.group(1))

    return {"ra": ra, "dec": dec, "pixel_scale_arcsec": pixel_scale}


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=ColorCalibrateInput)
def color_calibrate(
    working_dir: str,
    image_path: str,
    method: str = "pcc",
    focal_length_mm: float = 0.0,
    pixel_size_um: float | None = None,
    camera_model: str | None = None,
    target_name: str | None = None,
    target_coords: dict | None = None,
    catalog: str = "gaia",
    limitmag: str | None = None,
    bgtol_lower: float | None = None,
    bgtol_upper: float | None = None,
    spcc_options: SpccOptions | None = None,
    platesolve_options: PlateSolveOptions | None = None,
    background_neutralization: bool = True,
) -> dict:
    """
    Correct white balance by matching star colors to photometric catalogs.
    Performs background neutralization and photometric or spectrophotometric
    color calibration.

    Plate solving runs internally first — requires focal_length_mm and
    pixel_size_um. pixel_size_um resolves automatically from the camera model
    lookup table when not explicitly provided.

    Method guidance:
      pcc  — reliable for all OSC/DSLR images, uses Gaia catalog by default.
      spcc — more accurate, models sensor spectral response.

    SPCC modes (configured via spcc_options):
      OSC broadband: set osc_sensor_name, optionally osc_filter_name + osc_lpf.
      Mono broadband: set mono_sensor_name + r_filter / g_filter / b_filter.
      Narrowband: set narrowband with wavelengths/bandwidths per channel.
      Atmospheric: set atmospheric for extinction correction at low elevations.
      White ref: set white_reference to target a specific reference spectrum.

    Plate solve options (configured via platesolve_options):
      downscale, sip_order, search_radius, catalog, limitmag, no_crop,
      local astrometry.net with blind_pos / blind_res.

    If plate solving fails, returns plate_solve_success=False with diagnostic
    error_msg. The agent should try providing target_coords, adjusting
    platesolve_options, or fall back to pixel_math for manual neutralization.

    Run analyze_image after to verify color_coefficients are physically
    plausible (r, g, b all near 1.0 +/- 30%).
    """
    if spcc_options is None:
        spcc_options = SpccOptions()
    if platesolve_options is None:
        platesolve_options = PlateSolveOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        px_size = resolve_pixel_size(pixel_size_um, camera_model)
    except ValueError as e:
        return {
            "calibrated_image_path": None,
            "plate_solve_success": False,
            "wcs_coords": None,
            "pixel_scale_arcsec": None,
            "color_coefficients": None,
            "error_msg": str(e),
        }

    # Resolve focal length: explicit arg → equipment.toml → None (Siril uses header/prefs)
    resolved_fl = _resolve_fl(focal_length_mm if focal_length_mm and focal_length_mm > 0 else None)

    # Resolve position hint: explicit coords > target_name SIMBAD lookup
    resolved_coords = target_coords
    if resolved_coords is None and target_name:
        resolved_coords = _resolve_target(target_name)

    stem = img_path.stem
    output_stem = f"{stem}_cc"

    platesolve_cmds = _build_platesolve_cmds_for_t10(
        resolved_fl, px_size, resolved_coords, platesolve_options
    )
    platesolve_cmd = platesolve_cmds[-1]  # for diagnostics

    if method == "spcc":
        cal_cmd = _build_spcc_cmd(spcc_options, limitmag, bgtol_lower, bgtol_upper)
    else:
        cal_cmd = _build_pcc_cmd(catalog, limitmag, bgtol_lower, bgtol_upper)

    commands = [f"load {stem}", *platesolve_cmds, cal_cmd, f"save {output_stem}"]

    plate_solve_success = True
    wcs_info: dict = {}
    error_msg: str | None = None

    try:
        result = run_siril_script(commands, working_dir=working_dir, timeout=180)
        wcs_info = _parse_plate_solve_result(result)
    except SirilError as exc:
        stdout_lower = exc.result.stdout.lower() + exc.result.stderr.lower()
        if "plate" in stdout_lower or "wcs" in stdout_lower or "astrometry" in stdout_lower:
            plate_solve_success = False
            error_msg = (
                f"Plate solving failed. platesolve_cmds={platesolve_cmds!r} "
                f"stdout={exc.result.stdout[-600:]!r} "
                f"stderr={exc.result.stderr[-400:]!r}"
            )
        else:
            raise

    if not plate_solve_success:
        return {
            "calibrated_image_path": None,
            "plate_solve_success": False,
            "wcs_coords": None,
            "pixel_scale_arcsec": None,
            "color_coefficients": None,
            "error_msg": error_msg,
        }

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(
            f"Color calibration did not produce: {output_path}"
        )

    return {
        "calibrated_image_path": str(output_path),
        "plate_solve_success": True,
        "wcs_coords": {
            "ra": wcs_info.get("ra"),
            "dec": wcs_info.get("dec"),
        },
        "pixel_scale_arcsec": wcs_info.get("pixel_scale_arcsec"),
        "pixel_size_um_used": px_size,
        "color_coefficients": None,
        "error_msg": None,
    }
