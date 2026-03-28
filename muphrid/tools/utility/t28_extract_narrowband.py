"""
T28 — extract_narrowband

Extract narrowband signal channels (Hα, O-III, Green) from CFA (Bayer mosaic)
images captured through narrowband or dual-narrowband (duoband) filters.

This tool enables the OSC dual-narrowband workflow — the most accessible path
to narrowband astrophotography without a dedicated mono camera:

  1. Capture with a duoband filter (e.g., Optolong L-eNhance, L-Ultimate,
     STC Duo-Narrowband, IDAS NB1) mounted on an OSC/DSLR camera like the
     Fujifilm X-T30 II.
  2. Calibrate (T02-T06) keeping frames in CFA format (T03 debayer=False).
  3. Call this tool to extract Hα (red pixels) and O-III (blue pixels) into
     separate grayscale FITS files.
  4. Register and stack each channel independently (T04+T07 twice).
  5. Process and enhance each channel with T27 multiscale, T17 local_contrast.
  6. Recombine via T23 pixel_math using SHO, HOO, or custom palettes.

IMPORTANT: This tool operates on CFA (non-debayered) FITS images only.
  - Input must be a calibrated sequence or single FITS with Bayer mosaic data.
  - Siril reads the BAYER_PATTERN keyword from the FITS header.
  - Fujifilm X-T30 II with standard Bayer pattern: RGGB.
  - Output images are half-sized (half the original width and height) unless
    upscaling is requested.

Supported Siril commands (verified against Siril 1.4):
  extract_Ha [-upscale]
  extract_HaOIII [-resample={ha|oiii}]
  extract_Green

This tool is also useful for purely diagnostic purposes — extracting the green
channel from a stacked OSC image provides a noise-only reference, since the
green channel in broadband captures mostly continuum light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class ExtractNarrowbandInput(BaseModel):
    extraction_type: str = Field(
        default="ha_oiii",
        description=(
            "ha: Extract Hα channel only (red Bayer pixels). "
            "Produces a half-sized grayscale FITS prefixed with 'Ha_'. "
            "ha_oiii: Extract both Hα and O-III simultaneously (recommended for "
            "duoband workflows). Produces 'Ha_' and 'OIII_' prefixed FITS files. "
            "The O-III output includes interpolated data for red pixels. "
            "green: Extract the averaged green channel. Produces a half-sized "
            "grayscale FITS prefixed with 'Green_'. Useful for pure continuum "
            "reference or diagnostics."
        ),
    )
    upscale_ha: bool = Field(
        default=False,
        description=(
            "Upscale the Hα output 2× to match the full sensor resolution (-upscale). "
            "Use when you want the Hα image to have the same dimensions as a "
            "full-resolution debayered image for direct combination. "
            "Only applies to extraction_type=ha or ha_oiii."
        ),
    )
    resample: str | None = Field(
        default=None,
        description=(
            "For ha_oiii: control output sizes so both Hα and O-III have equal "
            "dimensions (-resample=). "
            "'ha': upsample the Hα image to match the full-size O-III. "
            "'oiii': downsample the O-III image to match the half-size Hα. "
            "Null (default): no resampling — Hα is half the size of O-III, "
            "since O-III is produced from 3 of 4 Bayer pixels vs only 1 for Hα."
        ),
    )


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=ExtractNarrowbandInput)
def extract_narrowband(
    extraction_type: str = "ha_oiii",
    upscale_ha: bool = False,
    resample: str | None = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> dict:
    """
    Extract narrowband channels (Hα, O-III, Green) from a CFA (Bayer mosaic) image.

    Use when the image was captured through a duoband filter (L-eNhance,
    L-Ultimate, STC Duo-Narrowband, etc.) on a color OSC/DSLR camera to
    separate Hα (red) and O-III (blue) signals for independent processing
    and custom palette combination (HOO, SHO, etc.).

    Prerequisites:
    - image_path must be a CFA FITS (not debayered). Calibration must have
      been run with debayer=False, or this must be a single calibrated CFA frame.
    - The FITS header must contain the BAYER_PATTERN keyword for Siril to
      perform correct extraction.

    Palette combination examples via pixel_math:
      HOO palette: R=$Ha$ G=$OIII$ B=$OIII$
      SHO palette: R=$Ha$ G=$Ha$*0.3+$OIII$*0.7 B=$OIII$
      Custom: any linear combination appropriate to the target

    Returns paths to extracted channel FITS files.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = img_path.stem
    commands: list[str] = [f"load {stem}"]

    output_paths: dict[str, str | None] = {"ha": None, "oiii": None, "green": None}

    if extraction_type == "ha":
        ha_cmd = "extract_Ha"
        if upscale_ha:
            ha_cmd += " -upscale"
        commands.append(ha_cmd)
        # Output stem: Siril prepends "Ha_" to the original filename
        ha_stem = f"Ha_{stem}"
        output_paths["ha"] = str(Path(working_dir) / f"{ha_stem}.fits")

    elif extraction_type == "ha_oiii":
        haoiii_cmd = "extract_HaOIII"
        if resample in ("ha", "oiii"):
            haoiii_cmd += f" -resample={resample}"
        commands.append(haoiii_cmd)
        ha_stem = f"Ha_{stem}"
        oiii_stem = f"OIII_{stem}"
        output_paths["ha"] = str(Path(working_dir) / f"{ha_stem}.fits")
        output_paths["oiii"] = str(Path(working_dir) / f"{oiii_stem}.fits")

    elif extraction_type == "green":
        commands.append("extract_Green")
        green_stem = f"Green_{stem}"
        output_paths["green"] = str(Path(working_dir) / f"{green_stem}.fits")

    else:
        raise ValueError(
            f"Unknown extraction_type '{extraction_type}'. "
            "Choose from: ha, ha_oiii, green."
        )

    run_siril_script(commands, working_dir=working_dir, timeout=120)

    # Resolve actual output paths — Siril may write .fit or .fits
    resolved: dict[str, str | None] = {}
    for channel, expected_path in output_paths.items():
        if expected_path is None:
            resolved[channel] = None
            continue
        p = Path(expected_path)
        if p.exists():
            resolved[channel] = str(p)
        else:
            # Try .fit extension
            alt = p.with_suffix(".fit")
            resolved[channel] = str(alt) if alt.exists() else None

    missing = [ch for ch, p in resolved.items() if p is None and output_paths[ch] is not None]
    if missing:
        raise FileNotFoundError(
            f"Siril did not produce expected narrowband outputs for channels: {missing}. "
            f"Ensure the input is a valid CFA FITS with BAYER_PATTERN keyword set."
        )

    return {
        "ha_path": resolved["ha"],
        "oiii_path": resolved["oiii"],
        "green_path": resolved["green"],
        "extraction_type": extraction_type,
        "upscale_ha": upscale_ha,
        "resample": resample,
        "next_steps": (
            "Register and stack each extracted channel independently (T04+T07). "
            "Then process channels with T12 and T27 before combining via T23 pixel_math."
        ),
    }
