"""
T18 — saturation_adjust

Enhance or reduce color saturation, with optional targeting of specific hue
ranges and background noise protection.

Backend: Siril CLI — `satu` (saturation with background protection and hue
targeting) or `ght -sat` (GHT applied to the HSL saturation channel for
midtone-focused saturation boost).

Apply to the starless image to prevent star color bloat. Use background_factor
to protect noise in the dark sky from being color-amplified. For emission
nebulae, target specific hue indices separately (Ha = 0, OIII = 3).

Apply in small increments — multiple passes with moderate amounts produce
better results than one large boost.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# Hue range index reference (documented for agent and user)
# 0 = pink-orange (Hα emission, pinkish reds)
# 1 = orange-yellow
# 2 = yellow-cyan
# 3 = cyan (OIII emission, blue-green)
# 4 = cyan-magenta
# 5 = magenta-pink
# 6 = all channels (default)

HUE_RANGE_DESCRIPTIONS = {
    0: "pink-orange (Hα emission reds)",
    1: "orange-yellow",
    2: "yellow-cyan",
    3: "cyan (OIII emission blue-green)",
    4: "cyan-magenta",
    5: "magenta-pink",
    6: "all hue ranges (global)",
}


class GHTSatOptions(BaseModel):
    stretch_amount: float = Field(
        description=(
            "D: stretch strength applied to the saturation channel (0–10). "
            "Use small values (0.3–1.5) for post-stretch saturation boost. "
            "Higher values risk over-saturation and neon artefacts."
        ),
    )
    local_intensity: float = Field(
        default=0.0,
        description=(
            "B: focus the saturation boost around the symmetry point. "
            "0 = standard exponential. Higher (5–10) = targets mid-saturation "
            "pixels, protects already-saturated colors."
        ),
    )
    symmetry_point: float = Field(
        default=0.5,
        description=(
            "SP: saturation value (0–1) at which the boost is most intense. "
            "0.5 targets mid-saturation pixels — good default. "
            "Lower SP for images with mostly unsaturated colors."
        ),
    )
    clip_mode: str = Field(
        default="rgbblend",
        description=(
            "How out-of-range values are handled. "
            "'clip', 'rescale', 'rgbblend' (default), 'globalrescale'."
        ),
    )


class SaturationAdjustInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the non-linear starless FITS image. "
            "Apply after local contrast enhancement (T17)."
        )
    )
    method: str = Field(
        default="global",
        description=(
            "global: uniform saturation adjustment across all hues. "
            "hue_targeted: boost or reduce a specific hue range (set hue_target). "
            "ght_saturation: GHT applied to HSL saturation channel — targets "
            "mid-saturation pixels, protects already-saturated stars."
        ),
    )
    amount: float = Field(
        default=0.5,
        description=(
            "Saturation adjustment multiplier. "
            "> 0: increase saturation (0.3–1.0 typical). "
            "< 0: decrease saturation. "
            "0: no change. "
            "Apply in steps of 0.3–0.5 and iterate rather than one large boost."
        ),
    )
    background_factor: float = Field(
        default=1.5,
        description=(
            "Background protection threshold factor. "
            "Pixels below (median + background_factor * sigma) are not saturated. "
            "Protects background noise from color amplification. "
            "1.5 is a good default. 0 = disable protection (not recommended). "
            "Increase to 2.0–3.0 for noisy images."
        ),
    )
    hue_target: int | None = Field(
        default=None,
        description=(
            "Hue range index for targeted saturation (only used in hue_targeted mode). "
            "0=pink-orange (Hα), 1=orange-yellow, 2=yellow-cyan, "
            "3=cyan (OIII), 4=cyan-magenta, 5=magenta-pink, 6=all (same as global)."
        ),
    )
    ght_sat_options: GHTSatOptions | None = Field(
        default=None,
        description="GHT saturation parameters. Required when method=ght_saturation.",
    )


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SaturationAdjustInput)
def saturation_adjust(
    working_dir: str,
    image_path: str,
    method: str = "global",
    amount: float = 0.5,
    background_factor: float = 1.5,
    hue_target: int | None = None,
    ght_sat_options: GHTSatOptions | None = None,
) -> dict:
    """
    Adjust color saturation of the starless non-linear image.

    Apply to the starless image only — saturation on a star-containing image
    causes rainbow fringing and color bloat around bright stars.

    Emission nebula targeting:
    - For Hα (reds): method=hue_targeted, hue_target=0
    - For OIII (blue-green): method=hue_targeted, hue_target=3
    - Multiple targeted passes (one per emission line) give better control
      than one global boost.

    GHT saturation (ght_saturation) is recommended over global for images
    where the sky background is already marginally saturated — it targets
    mid-saturation pixels and avoids over-saturating already vivid colors.

    Always use background_factor > 0 to protect sky background noise from
    being color-boosted into a colorful noise pattern.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if method == "ght_saturation" and ght_sat_options is None:
        raise ValueError(
            "ght_sat_options.stretch_amount is required for method=ght_saturation."
        )

    stem = img_path.stem
    output_stem = f"{stem}_satu"

    if method == "ght_saturation":
        opts = ght_sat_options  # type: ignore[assignment]
        cmd = f"ght -sat -D={opts.stretch_amount}"
        if opts.local_intensity != 0.0:
            cmd += f" -B={opts.local_intensity}"
        if opts.symmetry_point != 0.5:
            cmd += f" -SP={opts.symmetry_point}"
        if opts.clip_mode != "rgbblend":
            cmd += f" -clipmode={opts.clip_mode}"
    elif method == "hue_targeted" and hue_target is not None:
        cmd = f"satu {amount} {background_factor} {hue_target}"
    else:
        cmd = f"satu {amount} {background_factor}"

    commands = [
        f"load {stem}",
        cmd,
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=60)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"saturation_adjust did not produce: {output_path}")

    hue_desc = HUE_RANGE_DESCRIPTIONS.get(hue_target, "all") if hue_target is not None else "all"

    return {
        "processed_image_path": str(output_path),
        "saturated_image_path": str(output_path),  # backward-compat alias
        "method": method,
        "amount": amount,
        "hue_target": hue_target,
        "hue_description": hue_desc,
    }
