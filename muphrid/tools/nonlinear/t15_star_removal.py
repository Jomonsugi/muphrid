"""
T15 — star_removal

Separate stars from extended objects (nebulae, galaxies) using StarNet v2
neural network inference. Produces a color starless image and a star mask,
enabling independent processing of nebulosity vs stars.

Backend: StarNet2 MPS build (PyTorch with Apple Silicon CoreML acceleration)
called directly via subprocess using STARNET_BIN and STARNET_WEIGHTS.
Do NOT use Siril's `starnet` wrapper — call the binary directly for full
argument control and to avoid dependency on Siril's internal StarNet config.

Format pipeline:
  StarNet2 does not accept FITS. The conversion chain is:
    1. Siril: load FITS → savetif → 16-bit TIF (Siril's savetif is 16-bit
       by default; savetif8 would cause StarNet to fall back to mono output)
    2. StarNet2: TIF in → starless TIF + mask TIF (both color if input is color)
    3. Siril: load TIF → save FITS (×2, for starless + mask)

Input requirements:
  - Must be a stretched (non-linear) image (post T14).
  - Image should be color (RGB). Mono support is available but unusual.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.config import load_settings
from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class StarRemovalInput(BaseModel):
    upscale: bool = Field(
        description=(
            "Apply 2× intermediate upsampling before StarNet inference (-u "
            "flag). True roughly doubles processing time. StarNet detects "
            "stars less reliably at small FWHM (below ~2px) without upscaling."
        ),
    )
    generate_star_mask: bool = Field(
        default=True,
        description=(
            "Generate a star mask alongside the starless image. Required "
            "input for star_restoration blend mode."
        ),
    )


# ── StarNet subprocess ─────────────────────────────────────────────────────────

def _run_starnet(
    tif_input: Path,
    starless_out: Path,
    mask_out: Path | None,
    upscale: bool,
) -> str:
    """Call StarNet2 MPS binary directly via subprocess. Returns stdout."""
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

    return result.stdout


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=StarRemovalInput)
def star_removal(
    upscale: bool,
    generate_star_mask: bool = True,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Remove stars from a non-linear image using StarNet v2.

    Produces a color starless image and optionally a star mask.

    Processing pipeline (internal):
      1. Convert FITS → 16-bit TIF (Siril savetif)
      2. Run StarNet2 MPS on TIF → starless TIF + mask TIF
      3. Convert TIFs back → FITS (Siril save)

    upscale=True applies 2× intermediate upsampling — roughly doubles
    processing time; StarNet's detection reliability drops at FWHM < ~2px
    without it.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

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

    # Step 1: FITS → 16-bit TIF.
    # Siril's `savetif` saves 16-bit TIFF by default. `savetif8` would produce
    # 8-bit output, which causes StarNet to fall back to grayscale processing.
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
    starnet_stdout = _run_starnet(tif_path, starless_tif, mask_tif, upscale)
    color_detected = "Color image" in starnet_stdout
    if not color_detected:
        raise RuntimeError(
            "StarNet2 processed the image as monochrome (grayscale), not color. "
            "This means the TIF passed to StarNet was not recognized as RGB. "
            "Possible causes:\n"
            "  1. The FITS input to T15 is already mono (1-channel) — check T14 stretch output.\n"
            "  2. T03 calibrate was run without debayer=True for a color camera.\n"
            "  3. T15 was intentionally called on a luminance-only image (unusual).\n"
            f"StarNet stdout: {starnet_stdout.strip()!r}"
        )

    # Step 3: TIF → FITS (starless + mask)
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

    # Post-condition contract: image-modifying tools update paths.current_image
    # to their primary output. star_removal's primary output is the starless
    # image — that is the file downstream nonlinear tools should process. The
    # pre-starless image is preserved at paths.previous_image so the agent can
    # refer back to it if needed, and the starless image is ALSO kept at
    # paths.starless_image for explicit reference (star_restoration needs it
    # to know what to blend stars back onto).
    #
    # Prior to this fix, current_image was left pointing at the pre-starless
    # input, which caused save_checkpoint("starless_base") to bookmark the
    # starred file by mistake — restore_checkpoint then became a no-op relative
    # to any subsequent starless work, and the agent could not recover.
    prev_current = state["paths"].get("current_image")
    summary = {
        "current_image": str(starless_fits),
        "previous_image": prev_current,
        "starless_image_path": str(starless_fits),
        "star_mask_path": mask_fits_path,
        "upscale": upscale,
        "generate_star_mask": generate_star_mask,
        "color_detected": color_detected,
        "starnet_stdout": starnet_stdout.strip(),
        "note": (
            "current_image now points at the starless FITS. Subsequent tools "
            "(curves, saturation, multiscale_process, local_contrast_enhance) "
            "will read the starless image. The pre-starless file is still on "
            "disk at previous_image. Call star_restoration before export to "
            "blend the stars back."
        ),
    }

    # Star removal preserves image_space: the operation splits an image into
    # (starless, mask) but each output is in the same value space as the
    # input. The documented contract is that t15 runs post-stretch, so the
    # input should be display-space — but state is the authoritative
    # contract, so pass through whatever the input claims to be. State
    # missing image_space is a legacy/broken checkpoint; refuse rather
    # than guess. See Metadata.image_space.
    incoming_image_space = state["metadata"].get("image_space")
    if incoming_image_space not in ("linear", "display"):
        raise RuntimeError(
            "star_removal: state.metadata.image_space is missing or invalid "
            f"(got {incoming_image_space!r}). Every writer of paths.current_image "
            "must also write metadata.image_space; this looks like a legacy "
            "checkpoint or a writer that skipped its bookkeeping. Refusing to "
            "guess — restart from a fresh checkpoint."
        )

    # Delta-only paths emit; the spread used here previously was the
    # parallel-update anti-pattern (CLAUDE.md). Each key is only named if
    # this tool changes it.
    return Command(update={
        "paths": {
            "current_image": str(starless_fits),
            "previous_image": prev_current,
            "starless_image": str(starless_fits),
            "star_mask": mask_fits_path,
        },
        "metadata": {"image_space": incoming_image_space},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
