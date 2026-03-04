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
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class PixelMathInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_stems(expression: str, working_dir: str) -> list[str]:
    """
    Extract $stem$ variables from the expression and verify each FITS file
    exists in working_dir. Returns list of found stems.
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
    for stem in stems:
        fits_exists = any((wd / f"{stem}{ext}").exists() for ext in (".fit", ".fits", ".FIT", ".FITS"))
        if not fits_exists:
            missing.append(stem)

    if missing:
        raise FileNotFoundError(
            f"PixelMath expression references missing FITS stems: {missing}. "
            f"Working dir: {working_dir}"
        )

    return stems


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=PixelMathInput)
def pixel_math(
    working_dir: str,
    expression: str,
    output_stem: str | None = None,
    rescale: bool = False,
    rescale_low: float = 0.0,
    rescale_high: float = 1.0,
) -> dict:
    """
    General-purpose pixel math using Siril's PixelMath engine.

    The primary use case in this pipeline is the masked-application pattern —
    combining a globally-processed image with the original using a mask to
    confine the effect to a tonal region:
        "$processed$ * $mask$ + $original$ * (1 - $mask$)"

    Other common uses:
    - Weighted star recombination: "$starless$ + $starmask$ * 0.8"
    - Channel extraction: "$image$[R]" (where Siril channels are indexed)
    - HDR blending: "$bright$ * 0.3 + "$mid$ * 0.7"

    All image variable stems must exist as FITS files in working_dir.
    The result image is saved to working_dir/{output_stem}.fit.

    Validate expression syntax before calling — the agent should check that
    all $variable$ names correspond to known pipeline outputs.
    """
    # Validate all stems exist before running Siril
    _validate_stems(expression, working_dir)

    out_stem = output_stem or "pm_result"

    pm_cmd = f'pm "{expression}"'
    if rescale:
        pm_cmd += f" -rescale {rescale_low} {rescale_high}"

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

    return {"result_image_path": str(output_path)}
