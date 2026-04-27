"""
T23 — pixel_math

General-purpose pixel-level mathematical operations via Siril's PixelMath
engine. Primary use case is the masked-application pattern:
  "$processed$ * $mask$ + $original$ * (1 - $mask$)"

All image variables in expressions must reference FITS file stems (without
extension) present in the working directory, delimited by `$`.

Siril commands (verified against Siril 1.4 CLI docs):
    pm "expression" [-rescale [low] [high]] [-nosum]
    save <output_stem>

Siril pm notes:
- Variables are image stem names from the working directory: $stem$
- Result is left in memory; must be saved explicitly with `save`
- Maximum 10 image variables per expression
- -rescale normalizes output to [low, high] range (default 0–1 if no values given)
- -nosum prevents Siril from accumulating LIVETIME/STACKCNT in the output header.
  Always use for blending operations — pixel math here is not a stack, and writing
  those header values would corrupt downstream metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import numpy as np
from astropy.io import fits
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class PixelMathInput(BaseModel):
    expression: str = Field(
        description=(
            "PixelMath expression. Image variable stems are wrapped in $: "
            "'$image1$ * 0.7 + $image2$ * 0.3'. "
            "All referenced stems must exist as FITS files in working_dir. "
            "Max 10 variables. Standard arithmetic operators supported."
        )
    )
    output_stem: str | None = Field(
        default=None,
        description=(
            "Output filename stem (no extension). If null, auto-generated "
            "from 'pm_result'. Output is saved to working_dir."
        ),
    )
    rescale: bool = Field(
        default=False,
        description=(
            "Rescale output to [rescale_low, rescale_high] range. "
            "Use when the expression can produce values outside [0, 1], "
            "e.g. additions without normalization."
        ),
    )
    rescale_low: float = Field(
        default=0.0,
        description="Lower bound for output rescaling (used only if rescale=True).",
    )
    rescale_high: float = Field(
        default=1.0,
        description="Upper bound for output rescaling (used only if rescale=True).",
    )
    nosum: bool = Field(
        default=True,
        description=(
            "Prevent Siril from accumulating LIVETIME and STACKCNT in the output "
            "FITS header (-nosum). Default True — pixel math in this pipeline is "
            "used for blending, not stacking. Set False only if you are performing "
            "a genuine image sum and want the header updated accordingly."
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_fits(wd: Path, stem: str) -> Path | None:
    """Locate a FITS file by stem in the working directory."""
    for ext in (".fit", ".fits", ".FIT", ".FITS"):
        p = wd / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _get_nlayers(fits_path: Path) -> int:
    """
    Return the number of image channels: 1 for mono, 3 for RGB.

    FITS stores axes in Fortran order, so a numpy array of shape (C, H, W)
    is written as NAXIS1=W, NAXIS2=H, NAXIS3=C. A mono image may be stored
    as (H, W) → NAXIS=2, or as (1, H, W) → NAXIS=3 with NAXIS3=1.
    Both are 1-channel; NAXIS alone does not distinguish them from RGB (NAXIS3=3).
    """
    with fits.open(str(fits_path)) as hdul:
        hdr = hdul[0].header
        naxis = int(hdr.get("NAXIS", 0))
        if naxis < 3:
            return 1
        # NAXIS3 is the channel count for (C, H, W) arrays written by astropy/Siril.
        return int(hdr.get("NAXIS3", 1))


def _validate_and_broadcast(expression: str, working_dir: str) -> tuple[list[str], str, bool]:
    """
    Extract $stem$ variables from the expression, verify each FITS file
    exists, and auto-broadcast mono (1-layer) images to 3-layer RGB when
    mixed with 3-layer images.  Returns (stems, possibly-rewritten expression,
    auto_broadcast flag).

    Siril pm requires all input images to have the same number of layers.
    Mono images may be stored as (H, W) NAXIS=2 OR as (1, H, W) NAXIS=3 —
    both are 1-channel and both trigger broadcast when mixed with 3-channel RGB.
    """
    stems = re.findall(r"\$([^$]+)\$", expression)
    if not stems:
        raise ValueError(
            f"No $variable$ tokens found in expression: {expression!r}. "
            "Image variables must be wrapped in $ signs."
        )
    if len(stems) > 10:
        raise ValueError(
            f"Expression uses {len(stems)} variables; Siril pm supports max 10."
        )

    wd = Path(working_dir)
    missing = []
    stem_layers: dict[str, int] = {}
    for stem in set(stems):
        fpath = _find_fits(wd, stem)
        if fpath is None:
            missing.append(stem)
        else:
            stem_layers[stem] = _get_nlayers(fpath)

    if missing:
        raise FileNotFoundError(
            f"PixelMath expression references missing FITS stems: {missing}. "
            f"Working dir: {working_dir}"
        )

    layer_values = set(stem_layers.values())
    auto_broadcast = False

    if len(layer_values) > 1:
        # Mix of channel counts — broadcast all 1-layer images to 3-layer RGB.
        auto_broadcast = True
        mono_stems = [s for s, n in stem_layers.items() if n == 1]
        for mono_stem in mono_stems:
            rgb_stem = f"{mono_stem}_rgb3"
            if _find_fits(wd, rgb_stem) is None:
                # Broadcast mono→RGB in Python to guarantee pixel-exact dimensions.
                # Siril's rgbcomp can lose 1 pixel due to internal crop/pad behavior.
                mono_path = _find_fits(wd, mono_stem)
                with fits.open(str(mono_path)) as hdul:
                    mono_data = hdul[0].data.astype(np.float32)
                mono_2d = mono_data.squeeze()
                rgb_data = np.stack([mono_2d, mono_2d, mono_2d], axis=0)
                hdu = fits.PrimaryHDU(data=rgb_data)
                hdu.writeto(str(wd / f"{rgb_stem}.fits"), overwrite=True)
            expression = expression.replace(f"${mono_stem}$", f"${rgb_stem}$")

    return list(set(re.findall(r"\$([^$]+)\$", expression))), expression, auto_broadcast


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=PixelMathInput)
def pixel_math(
    expression: str,
    output_stem: str | None = None,
    rescale: bool = False,
    rescale_low: float = 0.0,
    rescale_high: float = 1.0,
    nosum: bool = True,
    state: Annotated[AstroState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    General-purpose pixel math using Siril's PixelMath engine.

    Common uses:
    - Masked application: "$processed$ * $mask$ + $original$ * (1 - $mask$)"
    - Weighted star recombination: "$starless$ + $starmask$ * 0.8"
    - Channel extraction: "$image$[R]" (where Siril channels are indexed)
    - HDR blending: "$bright$ * 0.3 + $mid$ * 0.7"

    All image variable stems must exist as FITS files in working_dir.
    The result image is saved to working_dir/{output_stem}.fit.

    Validate expression syntax before calling — the agent should check that
    all $variable$ names correspond to known image files.

    Post-condition: paths.current_image is updated to the result file, and
    paths.previous_image records what current_image was before this call.
    This matches the post-condition contract for every other image-modifying
    tool. If you intended to produce an intermediate file without promoting
    it to current_image, call save_checkpoint BEFORE pixel_math and
    restore_checkpoint AFTER — the result file stays on disk either way.
    """
    working_dir = state["dataset"]["working_dir"]
    _stems, expression, auto_broadcast = _validate_and_broadcast(expression, working_dir)

    out_stem = output_stem or "pm_result"

    pm_cmd = f'pm "{expression}"'
    if rescale:
        pm_cmd += f" -rescale {rescale_low} {rescale_high}"
    if nosum:
        pm_cmd += " -nosum"

    commands = [pm_cmd, f"save {out_stem}"]
    run_siril_script(commands, working_dir=working_dir, timeout=120)

    wd = Path(working_dir)
    output_path = wd / f"{out_stem}.fit"
    if not output_path.exists():
        output_path = wd / f"{out_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(
            f"pixel_math: Siril did not produce expected output: {wd / out_stem}.fit"
        )

    prev_current = state["paths"].get("current_image")
    summary = {
        "result_image_path": str(output_path),
        "current_image": str(output_path),
        "previous_image": prev_current,
        "expression": expression,
        "auto_broadcast": auto_broadcast,
    }
    return Command(update={
        "paths": {
            **state["paths"],
            "current_image": str(output_path),
            "previous_image": prev_current,
        },
        "messages": [ToolMessage(
            content=json.dumps(summary, indent=2, default=str),
            tool_call_id=tool_call_id,
        )],
    })
