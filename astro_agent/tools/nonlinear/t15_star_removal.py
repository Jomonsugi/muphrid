"""
T15 — star_removal

Separate stars from extended objects (nebulae, galaxies) using StarNet v2
neural network inference. Produces a starless image and a star mask, enabling
independent processing of nebulosity vs stars.

Backend: StarNet2 MPS build (PyTorch with Apple Silicon CoreML acceleration)
called directly via subprocess using STARNET_BIN and STARNET_WEIGHTS.
Do NOT use Siril's `starnet` wrapper — call the binary directly for full
argument control and to avoid dependency on Siril's internal StarNet config.

Format pipeline:
  StarNet2 does not accept FITS. The conversion chain is:
    1. Siril: load FITS → savetif → 16-bit TIF
    2. StarNet2: TIF in → starless TIF + mask TIF
    3. Siril: load TIF → save FITS (×2, for starless + mask)

HITL: requires_visual_review=True by default.
Star removal quality is a known subjective checkpoint — the threshold between
"stars cleanly removed" and "nebula structure damaged" requires visual judgment.

Input requirements:
  - Must be a stretched (non-linear) image (post T14).
  - Image should be color (RGB). Mono support is available but unusual.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.config import load_settings
from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class StarRemovalInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the stretched (non-linear) FITS image. "
            "Must be post-stretch (T14). StarNet processes non-linear images."
        )
    )
    upscale: bool = Field(
        default=False,
        description=(
            "Apply 2× intermediate upsampling before star removal (-u flag). "
            "Use when stars are very small (tight FWHM < 2px) and StarNet is "
            "partially missing them. Increases processing time significantly."
        ),
    )
    generate_star_mask: bool = Field(
        default=True,
        description=(
            "Generate a star mask alongside the starless image. "
            "The mask is required for T19 star_restoration blend mode. "
            "Always set True unless you will use synthstar mode in T19."
        ),
    )


# ── StarNet subprocess ─────────────────────────────────────────────────────────

def _run_starnet(
    tif_input: Path,
    starless_out: Path,
    mask_out: Path | None,
    upscale: bool,
) -> None:
    """Call StarNet2 MPS binary directly via subprocess."""
    settings = load_settings()

    cmd: list[str] = [
        settings.starnet_bin,
        "-i", str(tif_input),
        "-o", str(starless_out),
        "-w", settings.starnet_weights,
    ]
    if mask_out is not None:
        cmd += ["-m", str(mask_out)]
    if upscale:
        cmd.append("-u")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(
            f"StarNet2 failed (exit {result.returncode}):\n"
            f"{result.stderr or result.stdout}"
        )

    if not starless_out.exists():
        raise FileNotFoundError(
            f"StarNet2 did not produce starless output: {starless_out}\n"
            f"stdout: {result.stdout}"
        )


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=StarRemovalInput)
def star_removal(
    working_dir: str,
    image_path: str,
    upscale: bool = False,
    generate_star_mask: bool = True,
) -> dict:
    """
    Remove stars from the stretched image using StarNet v2 neural network.

    Produces a starless image and optionally a star mask. The starless image
    becomes the working canvas for all non-linear processing (T16–T18, T25–T27).
    The star mask is recombined with the processed starless image in T19.

    Processing pipeline (internal):
      1. Convert FITS → 16-bit TIF (Siril savetif)
      2. Run StarNet2 MPS on TIF → starless TIF + mask TIF
      3. Convert both TIFs back → FITS (Siril save)

    Use upscale=True only if stars are very small (tight PSF) and StarNet is
    visibly missing them — it doubles processing time.

    HITL visual review is triggered automatically after this tool (V1).
    Check: nebula structure intact, no dark halos around removed stars.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = img_path.stem
    tif_stem = f"{stem}_for_starnet"
    starless_stem = f"{stem}_starless"
    mask_stem = f"{stem}_starmask"

    tif_path      = Path(working_dir) / f"{tif_stem}.tif"
    starless_tif  = Path(working_dir) / f"{starless_stem}.tif"
    mask_tif      = Path(working_dir) / f"{mask_stem}.tif" if generate_star_mask else None

    # Step 1: FITS → 16-bit TIF
    siril_export = [
        f"load {stem}",
        f"savetif {tif_stem}",
    ]
    run_siril_script(siril_export, working_dir=working_dir, timeout=60)

    if not tif_path.exists():
        raise FileNotFoundError(
            f"Siril did not produce TIF at {tif_path}. "
            "Check that the image was loaded correctly."
        )

    # Step 2: StarNet2 inference
    _run_starnet(tif_path, starless_tif, mask_tif, upscale)

    # Step 3: TIF → FITS (starless + mask in one Siril run)
    siril_import: list[str] = [
        f"load {starless_stem}.tif",
        f"save {starless_stem}",
    ]
    if generate_star_mask and mask_tif and mask_tif.exists():
        siril_import += [
            f"load {mask_stem}.tif",
            f"save {mask_stem}",
        ]
    run_siril_script(siril_import, working_dir=working_dir, timeout=60)

    # Locate output FITS files
    starless_fits = Path(working_dir) / f"{starless_stem}.fit"
    if not starless_fits.exists():
        starless_fits = Path(working_dir) / f"{starless_stem}.fits"
    if not starless_fits.exists():
        raise FileNotFoundError(f"Starless FITS not found: {starless_fits}")

    mask_fits_path: str | None = None
    if generate_star_mask:
        mf = Path(working_dir) / f"{mask_stem}.fit"
        if not mf.exists():
            mf = Path(working_dir) / f"{mask_stem}.fits"
        if mf.exists():
            mask_fits_path = str(mf)

    return {
        "starless_image_path": str(starless_fits),
        "star_mask_path": mask_fits_path,
        "tif_intermediate": str(tif_path),
    }
