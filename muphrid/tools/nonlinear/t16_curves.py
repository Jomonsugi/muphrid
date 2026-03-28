"""
T16 — curves_adjust

Fine-tune brightness, contrast, and per-channel tonal balance using Siril's
Midtone Transfer Function (MTF) or Generalized Hyperbolic Stretch (GHT).
The primary tool for post-stretch tonal refinement.

Backend: Siril CLI — `mtf` (midtone transfer function) or `ght` with targeted
B/SP parameters for specific tonal ranges.

Apply to the starless image (post T15) to avoid star color bloat.
The agent may apply multiple passes with progressively refined parameters.
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

class MTFOptions(BaseModel):
    black_point: float = Field(
        default=0.0,
        description=(
            "Low (shadows) clipping point (0–1). "
            "0.0 = no black point lift. Small values (0.01–0.05) remove "
            "faint residual sky glow."
        ),
    )
    midtone: float = Field(
        default=0.5,
        description=(
            "Midtone transfer point (0–1). The core brightness control. "
            "0.5 = linear (no change). "
            "< 0.5 = brightens midtones. > 0.5 = darkens midtones. "
            "Typical range for brightening: 0.2–0.4."
        ),
    )
    white_point: float = Field(
        default=1.0,
        description=(
            "High (highlights) clipping point (0–1). "
            "1.0 = no highlight clip. Reduce only if bright stars are blown out."
        ),
    )
    channels: str = Field(
        default="all",
        description=(
            "Which channels to apply MTF to. "
            "all: all channels equally (preserves color balance). "
            "R, G, B: single channel for targeted color correction. "
            "RG, RB, GB: two-channel pairs."
        ),
    )


class GHTCurvesOptions(BaseModel):
    stretch_amount: float = Field(
        description=(
            "D: stretch strength 0–10. The primary control. "
            "Use small values (0.3–1.5) for post-stretch refinement — "
            "not a first-stretch tool."
        ),
    )
    local_intensity: float = Field(
        default=5.0,
        description=(
            "B: tonal focus (-5 to 15). Higher values tightly focus the "
            "adjustment around symmetry_point. 5–13 for targeted mid-tone boost."
        ),
    )
    symmetry_point: float = Field(
        default=0.3,
        description=(
            "SP: the brightness value to target (0–1). "
            "Set to the median value of the region you want to enhance — "
            "e.g., the faint outer nebula at 0.1–0.2, bright core at 0.5–0.7."
        ),
    )
    shadow_protection: float = Field(
        default=0.05,
        description="LP: shadow protection floor (0–SP).",
    )
    highlight_protection: float = Field(
        default=0.9,
        description="HP: highlight protection ceiling (HP–1).",
    )
    channels: str = Field(
        default="all",
        description="Which channels to apply to. all, R, G, B, RG, RB, GB.",
    )
    clip_mode: str = Field(
        default="rgbblend",
        description=(
            "How out-of-range values are handled. "
            "'clip', 'rescale', 'rgbblend' (default), 'globalrescale'."
        ),
    )


class CurvesAdjustInput(BaseModel):
    method: str = Field(
        default="mtf",
        description=(
            "mtf: Midtone Transfer Function — global brightness and contrast "
            "adjustment with black/white point control. Simple and reliable. "
            "ght: Generalized Hyperbolic Stretch applied post-stretch — targeted "
            "adjustment of a specific brightness range using B and SP parameters."
        ),
    )
    mtf_options: MTFOptions = Field(default_factory=MTFOptions)
    ght_options: GHTCurvesOptions | None = Field(
        default=None,
        description="GHT parameters. Required when method=ght.",
    )


# ── Command builders ───────────────────────────────────────────────────────────

def _build_mtf_cmd(opts: MTFOptions) -> str:
    cmd = f"mtf {opts.black_point} {opts.midtone} {opts.white_point}"
    if opts.channels != "all":
        cmd += f" {opts.channels}"
    return cmd


def _build_ght_curves_cmd(opts: GHTCurvesOptions) -> str:
    cmd = f"ght -D={opts.stretch_amount} -B={opts.local_intensity}"
    cmd += f" -SP={opts.symmetry_point}"
    if opts.shadow_protection != 0.0:
        cmd += f" -LP={opts.shadow_protection}"
    if opts.highlight_protection != 1.0:
        cmd += f" -HP={opts.highlight_protection}"
    if opts.clip_mode != "rgbblend":
        cmd += f" -clipmode={opts.clip_mode}"
    cmd += " -human"
    if opts.channels != "all":
        cmd += f" {opts.channels}"
    return cmd


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=CurvesAdjustInput)
def curves_adjust(
    method: str = "mtf",
    mtf_options: MTFOptions | None = None,
    ght_options: GHTCurvesOptions | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Fine-tune brightness, contrast, and per-channel tonal balance.

    Method guidance:
    - mtf: simple and reliable. Set midtone < 0.5 to brighten, > 0.5 to darken.
      Use channels=R/G/B to correct residual color casts.
    - ght: targeted tonal stretch. Use high local_intensity (B=8–13) and set
      symmetry_point to the median of the region you want to enhance.
      Effective for boosting faint outer nebula without saturating bright core.

    Multiple passes with small adjustments are better than one large adjustment.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

    if mtf_options is None:
        mtf_options = MTFOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if method == "ght" and ght_options is None:
        raise ValueError(
            "ght_options.stretch_amount is required for method=ght."
        )

    stem = img_path.stem
    output_stem = f"{stem}_curves"

    if method == "ght":
        adjust_cmd = _build_ght_curves_cmd(ght_options)  # type: ignore[arg-type]
    else:
        adjust_cmd = _build_mtf_cmd(mtf_options)

    commands = [
        f"load {stem}",
        adjust_cmd,
        f"save {output_stem}",
    ]
    run_siril_script(commands, working_dir=working_dir, timeout=60)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"curves_adjust did not produce: {output_path}")

    summary: dict = {
        "output_path": str(output_path),
        "method": method,
        "siril_command": adjust_cmd,
    }
    if method == "mtf":
        summary["mtf_parameters"] = {
            "black_point": mtf_options.black_point,
            "midtone": mtf_options.midtone,
            "white_point": mtf_options.white_point,
            "channels": mtf_options.channels,
        }
    elif method == "ght" and ght_options is not None:
        summary["ght_parameters"] = {
            "stretch_amount": ght_options.stretch_amount,
            "local_intensity": ght_options.local_intensity,
            "symmetry_point": ght_options.symmetry_point,
            "shadow_protection": ght_options.shadow_protection,
            "highlight_protection": ght_options.highlight_protection,
            "channels": ght_options.channels,
            "clip_mode": ght_options.clip_mode,
        }

    return Command(update={
        "paths": {**state["paths"], "current_image": str(output_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
