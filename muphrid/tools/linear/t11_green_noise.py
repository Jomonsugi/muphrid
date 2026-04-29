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

import json
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState
from muphrid.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class RemoveGreenNoiseInput(BaseModel):
    protection_type: str = Field(
        default="average_neutral",
        description=(
            "average_neutral (type 0): Reduces green relative to the average of "
            "red and blue — the standard choice for most images. No amount control. "
            "maximum_neutral (type 1): Reduces green relative to the maximum of "
            "red and blue — more aggressive than average_neutral. No amount control. "
            "maximum_mask (type 2): Creates a green-excess mask using max(R,B) as "
            "reference. Accepts amount parameter (0–1) for proportional reduction. "
            "Use when you want partial green removal with fine control. "
            "additive_mask (type 3): Additive mask approach — subtracts a fraction "
            "of the green excess. Accepts amount parameter (0–1). Gentlest method; "
            "good for targets with genuine green content (e.g. [OIII] emission)."
        ),
    )
    amount: float = Field(
        default=1.0,
        description=(
            "Reduction strength 0.0–1.0. Only used by maximum_mask (type 2) and "
            "additive_mask (type 3). Types 0/1 ignore this parameter. "
            "1.0: full green neutralization. "
            "0.5–0.8: partial reduction for targets with genuine green emission."
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
    protection_type: str = "average_neutral",
    amount: float = 1.0,
    preserve_lightness: bool = True,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Remove systematic green cast from OSC/DSLR images using Siril SCNR.

    The Bayer/X-Trans CFA has 2× more green pixels than red or blue, which
    causes a systematic green cast in the stacked image after demosaicing.
    Only applicable to OSC/DSLR color data. If green_excess is near zero,
    applying this tool may introduce a magenta cast.

    Protection type guide:
    - average_neutral: standard, reliable. Full green neutralization with no
      amount control. Best first choice for most broadband images.
    - maximum_neutral: more aggressive than average_neutral. Use when
      average_neutral leaves visible green.
    - maximum_mask: accepts amount (0–1) for proportional control. Use when
      you want partial removal (e.g., amount=0.7 for targets with genuine
      green [OIII] emission).
    - additive_mask: gentlest, accepts amount (0–1). Best for narrowband
      or targets where preserving subtle green tones matters.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = img_path.stem
    output_stem = f"{stem}_rmg"

    _TYPE_MAP = {
        "average_neutral": 0,
        "maximum_neutral": 1,
        "maximum_mask": 2,
        "additive_mask": 3,
    }
    type_int = _TYPE_MAP.get(protection_type, 0)
    nopreserve_flag = " -nopreserve" if not preserve_lightness else ""

    # Siril 1.4 rmgreen syntax: rmgreen [-nopreserve] [type] [amount]
    # amount is ONLY valid for types 2 and 3.
    rmgreen_cmd = f"rmgreen{nopreserve_flag} {type_int}"
    if type_int >= 2:
        rmgreen_cmd += f" {amount}"

    commands = [
        f"load {stem}",
        rmgreen_cmd,
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=60)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"rmgreen did not produce: {output_path}")

    summary = {
        "output_path": str(output_path),
        "settings": {
            "protection_type": protection_type,
            "siril_type_int": type_int,
            "amount": amount if type_int >= 2 else None,
            "preserve_lightness": preserve_lightness,
        },
        "siril_command": rmgreen_cmd,
    }

    return Command(update={
        "paths": {"current_image": str(output_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
