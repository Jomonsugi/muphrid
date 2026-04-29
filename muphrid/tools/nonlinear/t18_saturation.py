"""
T18 — saturation_adjust

Increase or decrease color saturation, optionally targeting a specific hue
range, with background-noise protection.

Backend: Siril CLI — `satu` (saturation with background threshold and hue
targeting) or `ght -sat` (GHT applied to the HSL saturation channel).
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
from muphrid.tools._siril import fits_nlayers, run_siril_script


# Hue range index → color band. These are Siril `satu` command values.
HUE_RANGE_DESCRIPTIONS = {
    0: "pink-orange",
    1: "orange-yellow",
    2: "yellow-cyan",
    3: "cyan",
    4: "cyan-magenta",
    5: "magenta-pink",
    6: "all hue ranges (global)",
}


class GHTSatOptions(BaseModel):
    stretch_amount: float = Field(
        description=(
            "D: stretch strength applied to the saturation channel (0–10)."
        ),
    )
    local_intensity: float = Field(
        default=0.0,
        description=(
            "B: focus the saturation boost around the symmetry point "
            "(-5 to 15). 0 = standard exponential. Higher values tighten "
            "the boost around SP."
        ),
    )
    symmetry_point: float = Field(
        default=0.5,
        description=(
            "SP: saturation value (0–1) at which the boost peaks."
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
    method: str = Field(
        description=(
            "global: uniform saturation across all hues (Siril `satu`).\n"
            "hue_targeted: saturation applied only to the hue range set by "
            "hue_target (Siril `satu` with hue index).\n"
            "ght_saturation: Generalized Hyperbolic Stretch applied to the "
            "HSL saturation channel (Siril `ght -sat`). Non-uniform in "
            "saturation — boost peaks at symmetry_point."
        ),
    )
    amount: float = Field(
        description=(
            "Saturation adjustment multiplier for Siril `satu`. "
            "> 0 increases saturation, < 0 decreases, 0 no change."
        ),
    )
    background_factor: float = Field(
        default=1.5,
        description=(
            "Background protection threshold factor. Pixels below "
            "(median + background_factor * sigma) are excluded from the "
            "saturation adjustment. 0 disables protection."
        ),
    )
    hue_target: int | None = Field(
        default=None,
        description=(
            "Hue range index (only used when method=hue_targeted). "
            "0=pink-orange, 1=orange-yellow, 2=yellow-cyan, 3=cyan, "
            "4=cyan-magenta, 5=magenta-pink, 6=all (equivalent to global)."
        ),
    )
    ght_sat_options: GHTSatOptions | None = Field(
        default=None,
        description="GHT saturation parameters. Required when method=ght_saturation.",
    )


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SaturationAdjustInput)
def saturation_adjust(
    method: str,
    amount: float,
    background_factor: float = 1.5,
    hue_target: int | None = None,
    ght_sat_options: GHTSatOptions | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Adjust color saturation of the non-linear image.

    Requires a 3-channel RGB input (fails on mono). background_factor > 0
    excludes pixels below (median + background_factor * sigma) from the
    saturation operation.

    method=ght_saturation requires ght_sat_options.stretch_amount.
    method=hue_targeted uses hue_target to select one of the 6 Siril hue
    bands.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if fits_nlayers(img_path) < 3:
        raise ValueError(
            f"saturation_adjust requires a color (3-channel RGB) image, but "
            f"{img_path.name} is monochrome. "
            "Saturation adjustment has no effect on mono images. "
            "Trace back to T15 star_removal — if the starless image is mono, "
            "StarNet received a mono or 8-bit input. Ensure T14 stretch output "
            "is RGB and that T15 uses savetif16 for the TIF conversion."
        )

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

    summary: dict = {
        "output_path": str(output_path),
        "method": method,
        "amount": amount,
        "background_factor": background_factor,
        "siril_command": cmd,
    }
    if method == "hue_targeted" and hue_target is not None:
        summary["hue_target"] = hue_target
        summary["hue_target_description"] = HUE_RANGE_DESCRIPTIONS.get(hue_target, "unknown")
    if method == "ght_saturation" and ght_sat_options is not None:
        summary["ght_sat_parameters"] = {
            "stretch_amount": ght_sat_options.stretch_amount,
            "local_intensity": ght_sat_options.local_intensity,
            "symmetry_point": ght_sat_options.symmetry_point,
            "clip_mode": ght_sat_options.clip_mode,
        }

    return Command(update={
        "paths": {"current_image": str(output_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
