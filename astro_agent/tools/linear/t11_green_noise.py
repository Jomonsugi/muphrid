"""
T11 — remove_green_noise

Remove excess green signal common in one-shot-color (OSC) and DSLR cameras.
The Bayer / X-Trans CFA has 2× more green pixels than red or blue, which
causes a systematic green cast in the stacked image after demosaicing.

Backend: Siril SCNR (Selective Color Noise Reduction) — rmgreen command.

Important constraints:
  - ONLY valid on OSC/DSLR data (metadata.is_osc == True).
  - Must be applied in linear space (before stretch).
  - Check metrics.green_excess from analyze_image first — skip if the value
    is near zero. Over-applying can introduce a magenta cast.
  - Targets with genuine green emission (rare in broadband; e.g. [OIII] can
    appear green-shifted in RGB) should use a reduced amount (0.5–0.7).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class RemoveGreenNoiseInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the linear OSC/DSLR FITS to process. "
            "Must be color (is_osc=True) and in linear space."
        )
    )
    protection_type: str = Field(
        default="average_neutral",
        description=(
            "average_neutral (type 0): Reduces green relative to the average of "
            "red and blue — the standard choice for most images. "
            "maximum_neutral (type 1): Reduces green relative to the maximum of "
            "red and blue — more aggressive; use when average_neutral is insufficient."
        ),
    )
    amount: float = Field(
        default=1.0,
        description=(
            "Reduction strength 0.0–1.0. "
            "1.0: full green neutralization (appropriate when green_excess is > 0.05). "
            "0.5–0.8: partial reduction for targets with some genuine green content "
            "(e.g. broadband images with strong [OIII])."
        ),
    )
    preserve_lightness: bool = Field(
        default=True,
        description=(
            "Preserve overall luminance when reducing green. "
            "Almost always desirable — disabling can dim the image."
        ),
    )


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=RemoveGreenNoiseInput)
def remove_green_noise(
    working_dir: str,
    image_path: str,
    protection_type: str = "average_neutral",
    amount: float = 1.0,
    preserve_lightness: bool = True,
) -> dict:
    """
    Remove systematic green cast from OSC/DSLR images using Siril SCNR.

    Only apply to OSC/DSLR data (metadata.is_osc=True). Check green_excess
    from analyze_image first — if it is near zero, skip this tool to avoid
    introducing a magenta cast. Use amount < 1.0 for targets with genuine
    green emission content.

    Must be called in linear space, typically after color_calibrate (T10).
    Run analyze_image after to confirm green_excess is reduced.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = img_path.stem
    output_stem = f"{stem}_rmg"

    type_int = 0 if protection_type == "average_neutral" else 1
    # -nopreserve disables lightness preservation; default IS to preserve
    nopreserve_flag = " -nopreserve" if not preserve_lightness else ""

    # Verified Siril 1.4 rmgreen syntax:
    #   rmgreen [-nopreserve] [type] [amount]
    # The `amount` argument is ONLY valid for types 2 (maximum_mask) and
    # 3 (additive_mask). Types 0 and 1 (the only types exposed by this tool)
    # do not accept an amount argument. Passing it would cause a Siril error.
    commands = [
        f"load {stem}",
        f"rmgreen{nopreserve_flag} {type_int}",
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=60)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"rmgreen did not produce: {output_path}")

    return {
        "cleaned_image_path": str(output_path),
        "protection_type": protection_type,
        "amount": amount,
        "preserve_lightness": preserve_lightness,
    }
