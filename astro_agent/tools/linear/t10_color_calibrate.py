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

Camera lookup table includes all currently supported cameras. Add new entries
to CAMERA_PIXEL_SIZE_UM as new cameras are added to the workflow.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import SirilError, SirilResult, run_siril_script


# ── Camera pixel size lookup table ─────────────────────────────────────────────
# Keys are lowercase, stripped camera model strings as reported by ExifTool.
# Values are pixel pitch in microns (µm).
# Calculation: sensor_width_mm / sensor_width_px  (or height equivalent)

CAMERA_PIXEL_SIZE_UM: dict[str, float] = {
    # Fujifilm X-Trans / Bayer APS-C
    "fujifilm x-t30 ii":      3.77,   # 23.5mm / 6240px
    "fujifilm x-t30ii":       3.77,
    "fujifilm x-t30":         3.76,
    "fujifilm x-t4":          3.76,
    "fujifilm x-t3":          3.76,
    "fujifilm x-t2":          3.96,
    "fujifilm x-s10":         3.76,
    "fujifilm x-e4":          3.76,
    "fujifilm x100v":         3.76,
    # Canon EOS (DSLR / Mirrorless)
    "canon eos ra":            6.45,   # Full-frame astro variant
    "canon eos r":             5.36,
    "canon eos r5":            4.39,
    "canon eos r6":            6.45,
    "canon eos 6d":            6.54,
    "canon eos 6d mark ii":   5.72,
    "canon eos 90d":           3.23,
    # Nikon Z / D series
    "nikon z6":                5.94,
    "nikon z6_2":              5.94,
    "nikon z6 ii":             5.94,
    "nikon z6ii":              5.94,
    "nikon d810":              4.88,
    "nikon d810a":             4.88,
    "nikon d850":              4.35,
    # Sony Alpha full-frame
    "sony ilce-7m3":           5.94,   # A7 III
    "sony ilce-7m4":           5.26,   # A7 IV
    "sony ilce-7sm3":          9.38,   # A7S III (high-sensitivity)
    "sony ilce-7rm4":          3.76,   # A7R IV
    # ZWO dedicated astro cameras (common pairings)
    "zwo asi294mc pro":        4.63,
    "zwo asi2600mc pro":       3.76,
    "zwo asi533mc pro":        3.76,
    "zwo asi183mc":            2.40,
    "zwo asi183mc pro":        2.40,
    "zwo asi071mc pro":        4.78,
    "zwo asi120mc-s":          3.75,
    # QHY
    "qhy268m":                 3.76,
    "qhy294m":                 4.63,
}


def resolve_pixel_size(
    pixel_size_um_arg: float | None,
    camera_model: str | None,
) -> float:
    """
    Resolve pixel size in microns from the priority chain:
    1. Explicit argument
    2. PIXEL_SIZE_UM env var
    3. Camera model lookup table
    Raises ValueError if none of the three sources can provide a value.
    """
    if pixel_size_um_arg is not None and pixel_size_um_arg > 0:
        return pixel_size_um_arg

    env_val = os.environ.get("PIXEL_SIZE_UM", "").strip()
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass

    if camera_model:
        key = camera_model.lower().strip()
        if key in CAMERA_PIXEL_SIZE_UM:
            return CAMERA_PIXEL_SIZE_UM[key]
        # Partial match: try contains
        for table_key, px in CAMERA_PIXEL_SIZE_UM.items():
            if table_key in key or key in table_key:
                return px

    raise ValueError(
        f"Cannot resolve pixel_size_um for camera '{camera_model}'. "
        "Options:\n"
        "  1. Pass pixel_size_um explicitly to color_calibrate.\n"
        "  2. Set PIXEL_SIZE_UM=<value> in .env.\n"
        "  3. Add your camera to CAMERA_PIXEL_SIZE_UM in t10_color_calibrate.py."
    )


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SpccOptions(BaseModel):
    sensor_name: str | None = Field(
        default=None,
        description=(
            "Sensor/camera model name as recognized by Siril's SPCC database. "
            "Examples: 'ZWO ASI294MC Pro', 'Canon EOS Ra'. "
            "Leave null for PCC or when sensor is not in Siril's catalog."
        ),
    )
    filter_name: str | None = Field(
        default=None,
        description=(
            "OSC filter name as recognized by Siril's SPCC database (-oscfilter=). "
            "Null for broadband (no filter) or when using PCC."
        ),
    )
    atmospheric_correction: bool = Field(
        default=False,
        description=(
            "Apply atmospheric extinction correction in SPCC (-atmos). "
            "Improves accuracy at lower elevations but requires location data."
        ),
    )


class ColorCalibrateInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the linear FITS image. "
            "Gradient should already be removed (post T09)."
        )
    )
    method: str = Field(
        default="pcc",
        description=(
            "pcc: Photometric Color Calibration using NOMAD/APASS/Gaia catalog stars. "
            "Reliable for all OSC/DSLR images. "
            "spcc: Spectrophotometric Color Calibration — more accurate, models the "
            "sensor's spectral response. Requires sensor_name to be set."
        ),
    )
    focal_length_mm: float = Field(
        description=(
            "Imaging focal length in mm. From acquisition_meta.focal_length_mm. "
            "Required for plate solving."
        )
    )
    pixel_size_um: float | None = Field(
        default=None,
        description=(
            "Pixel size in microns. If null, resolved from PIXEL_SIZE_UM env var or "
            "camera model lookup table (Fujifilm X-T30 II = 3.77 µm)."
        ),
    )
    camera_model: str | None = Field(
        default=None,
        description=(
            "Camera model string from AcquisitionMeta (used for pixel size lookup "
            "when pixel_size_um is not set explicitly)."
        ),
    )
    target_coords: dict | None = Field(
        default=None,
        description=(
            "Hint coordinates for plate solving: {'ra': float, 'dec': float} in degrees. "
            "Not required but speeds up solving and improves success rate for crowded fields."
        ),
    )
    catalog: str = Field(
        default="gaia",
        description=(
            "Star catalog for PCC: gaia (most complete, preferred), "
            "nomad, apass, localgaia (offline). SPCC always uses Gaia."
        ),
    )
    limitmag: str | None = Field(
        default=None,
        description=(
            "Override the automatic star magnitude limit for catalog matching. "
            "'+2': use 2 magnitudes deeper (more stars, slower). "
            "'-2': use 2 magnitudes shallower (brighter stars only). "
            "'12': use absolute limit magnitude of 12. "
            "Null = automatic (recommended for most cases). "
            "Increase depth (+N) for sparse star fields; reduce (-N) when "
            "catalog is crowded or PCC is fitting to galaxy cores."
        ),
    )
    bgtol_lower: float | None = Field(
        default=None,
        description=(
            "Background sample lower rejection tolerance in sigma units. "
            "Default: -2.8. Use a less negative value (e.g. -2.0) to be more "
            "aggressive about excluding background samples — useful when faint "
            "nebulosity fills much of the frame and would otherwise skew the "
            "background reference."
        ),
    )
    bgtol_upper: float | None = Field(
        default=None,
        description=(
            "Background sample upper rejection tolerance in sigma units. "
            "Default: 2.0. Lower values exclude more bright outliers from the "
            "background reference."
        ),
    )
    spcc_options: SpccOptions = Field(default_factory=SpccOptions)
    background_neutralization: bool = Field(
        default=True,
        description=(
            "Advisory flag: Siril PCC/SPCC always include background neutralization "
            "(sets channel background medians equal, removing sky color casts). "
            "There is no Siril CLI flag to disable this. This field exists so the "
            "agent can record intent. If background neutralization without PCC/SPCC "
            "is needed (e.g., plate solve fails), use T20 per_channel_bg to measure "
            "channel imbalance, then T23 pixel_math to equalize manually: "
            "'$image$ * (1, green_bg/red_bg, green_bg/blue_bg)'."
        ),
    )


# ── Plate-solve + calibrate helpers ───────────────────────────────────────────

def _build_platesolve_cmd(
    focal_length_mm: float,
    pixel_size_um: float,
    target_coords: dict | None,
) -> str:
    # Positional coords come first (RA DEC in decimal degrees), then named flags.
    # Siril syntax: platesolve [ra dec] -focal= -pixelsize=
    coord_prefix = ""
    if target_coords:
        ra  = target_coords.get("ra")
        dec = target_coords.get("dec")
        if ra is not None and dec is not None:
            coord_prefix = f"{ra} {dec} "
    return f"platesolve {coord_prefix}-focal={focal_length_mm} -pixelsize={pixel_size_um}"


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
    if opts.sensor_name:
        cmd += f' -oscsensor="{opts.sensor_name}"'
    if opts.filter_name:
        cmd += f' -oscfilter="{opts.filter_name}"'
    if opts.atmospheric_correction:
        cmd += " -atmos"
    if limitmag is not None:
        cmd += f" -limitmag={limitmag}"
    if bgtol_lower is not None or bgtol_upper is not None:
        lo = bgtol_lower if bgtol_lower is not None else -2.8
        hi = bgtol_upper if bgtol_upper is not None else 2.0
        cmd += f" -bgtol={lo},{hi}"
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
    target_coords: dict | None = None,
    catalog: str = "gaia",
    limitmag: str | None = None,
    bgtol_lower: float | None = None,
    bgtol_upper: float | None = None,
    spcc_options: SpccOptions | None = None,
    background_neutralization: bool = True,
) -> dict:
    """
    Correct white balance by matching star colors to photometric catalogs.
    Performs background neutralization (removes color cast from sky background)
    and photometric or spectrophotometric calibration.

    Plate solving is performed internally first — it requires focal_length_mm
    and pixel_size_um. If pixel_size_um is null, the camera model lookup table
    resolves it (Fujifilm X-T30 II = 3.77 µm).

    Method guidance:
    - pcc: reliable for all OSC/DSLR images, uses Gaia catalog by default.
    - spcc: more accurate, requires sensor_name to be set in spcc_options.

    If plate solving fails, the tool returns plate_solve_success=False and a
    descriptive error_msg. The agent should then try providing target_coords
    from acquisition_meta, or fall back to pixel_math for manual neutralization.

    This tool modifies pixel values — run analyze_image after to verify
    color_coefficients are physically plausible (r, g, b all near 1.0 ± 30%).
    """
    if spcc_options is None:
        spcc_options = SpccOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resolve pixel size
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

    stem = img_path.stem
    output_stem = f"{stem}_cc"

    # Step 1: plate solve (required for PCC and SPCC)
    platesolve_cmd = _build_platesolve_cmd(focal_length_mm, px_size, target_coords)

    # Step 2: color calibration
    if method == "spcc":
        cal_cmd = _build_spcc_cmd(spcc_options, limitmag, bgtol_lower, bgtol_upper)
    else:
        cal_cmd = _build_pcc_cmd(catalog, limitmag, bgtol_lower, bgtol_upper)

    commands = [
        f"load {stem}",
        platesolve_cmd,
        cal_cmd,
        f"save {output_stem}",
    ]

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
                f"Plate solving failed. Try providing target_coords or "
                f"increasing limit magnitude. Siril output: {exc.result.stderr[:400]}"
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
        "color_coefficients": None,  # populated from Siril stdout in Phase 7 post-hook
        "error_msg": None,
    }
