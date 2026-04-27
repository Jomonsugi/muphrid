"""
T14 — stretch_image

Apply a non-linear transfer function that transforms the image from linear
(sensor-space) to non-linear (display-space) brightness.

Backend: Siril CLI. Methods: GHS (Generalized Hyperbolic Stretch), arcsinh,
autostretch.

Produces variants identified by output_suffix. Each call stretches from the
same linear master — variants are independent and do not chain. After the
stretch, the image is non-linear; linear-space tools are no longer valid.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from muphrid.graph.state import AstroState
from muphrid.tools._siril import fits_has_nan, run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class GHSOptions(BaseModel):
    stretch_amount: float = Field(
        description=(
            "D: stretch intensity (0–10). Primary control — determines how "
            "aggressively the non-linear transfer function redistributes pixel "
            "values. Higher D moves the histogram further from linear."
        ),
    )
    local_intensity: float = Field(
        default=0.0,
        description=(
            "B: focus of the stretch around the symmetry point (-5 to 15). "
            "0 = standard exponential stretch. Positive values tighten the "
            "stretch around SP. Negative values de-focus it."
        ),
    )
    symmetry_point: float = Field(
        default=0.0,
        description=(
            "SP: the brightness level (0–1) at which the stretch adds the "
            "most contrast. 0.0 targets the darkest values."
        ),
    )
    shadow_protection: float = Field(
        default=0.0,
        description=(
            "LP: low-end linear region. Pixels below LP are stretched linearly. "
            "Protects shadow detail from clipping. MUST be <= symmetry_point (SP). "
            "0.0 = no protection (default)."
        ),
    )
    highlight_protection: float = Field(
        default=1.0,
        description=(
            "HP: pixels above this value are stretched linearly instead of "
            "non-linearly (0–1). 1.0 = no protection. Lower HP stretches "
            "less of the bright end non-linearly."
        ),
    )
    color_model: str = Field(
        default="human",
        description=(
            "human: L*a*b* lightness-weighted (preserves perceived color). "
            "even: equal weights across channels. "
            "independent: each channel processed independently (can shift "
            "white balance)."
        ),
    )
    channels: str = Field(
        default="all",
        description=(
            "Which channels receive the stretch. "
            "all: all channels with the same parameters (linked stretch). "
            "R, G, B: single channel. RG, RB, GB: channel pairs."
        ),
    )
    clip_mode: str = Field(
        default="rgbblend",
        description=(
            "How out-of-range values are handled after stretch. "
            "'clip': hard clip to [0,1]. "
            "'rescale': rescale to fit. "
            "'rgbblend': blend RGB to preserve color. "
            "'globalrescale': rescale uniformly across all channels."
        ),
    )


class AsinhOptions(BaseModel):
    stretch_factor: float = Field(
        default=100.0,
        description=(
            "Stretch strength (1–1000). Higher values produce a stronger "
            "non-linear transfer response."
        ),
    )
    black_point_offset: float = Field(
        default=0.0,
        description=(
            "Normalized offset (0–1) applied to the black point before "
            "stretching."
        ),
    )
    color_model: str = Field(
        default="human",
        description=(
            "human: L*a*b* lightness-weighted. "
            "default: simple channel mean for luminance calculation."
        ),
    )
    clip_mode: str = Field(
        default="rgbblend",
        description=(
            "How out-of-range values are handled. "
            "'clip', 'rescale', 'rgbblend' (default), 'globalrescale'."
        ),
    )


class AutostretchOptions(BaseModel):
    shadows_clipping_sigma: float = Field(
        default=-2.8,
        description=(
            "Shadows clipping point in sigma units from the histogram peak. "
            "-2.8 is the Siril default. More negative = more shadow detail preserved. "
            "Less negative (e.g., -1.5) = blacker background."
        ),
    )
    target_background: float = Field(
        default=0.25,
        description=(
            "Target background brightness (0–1). 0.25 is the Siril default. "
            "Lower = darker image (more contrast). Higher = brighter, flatter result."
        ),
    )
    linked: bool = Field(
        default=True,
        description=(
            "True: apply identical parameters to all channels (linked "
            "stretch — preserves channel ratios). False: each channel "
            "computed independently (can shift white balance)."
        ),
    )


class StretchImageInput(BaseModel):
    method: str = Field(
        default="ghs",
        description=(
            "ghs: Generalized Hyperbolic Stretch — parametric transfer with "
            "five controls (see GHSOptions). "
            "asinh: Arcsinh transfer function (see AsinhOptions). "
            "autostretch: Siril automatic histogram-based stretch (see "
            "AutostretchOptions)."
        ),
    )
    ghs_options: GHSOptions | None = Field(
        default=None,
        description="GHS parameters. Required when method=ghs.",
    )
    asinh_options: AsinhOptions = Field(default_factory=AsinhOptions)
    autostretch_options: AutostretchOptions = Field(default_factory=AutostretchOptions)
    output_suffix: str = Field(
        default="stretch",
        description=(
            "Suffix appended to the output filename stem. Each distinct "
            "suffix yields an independent variant file; identical suffix "
            "with identical parameters is rejected as a duplicate."
        ),
    )


# ── Command builders ───────────────────────────────────────────────────────────

def _build_ghs_cmd(opts: GHSOptions) -> str:
    # Siril constraint: LP must be <= SP. Clamp if the agent violated this.
    if opts.shadow_protection > opts.symmetry_point:
        opts.shadow_protection = opts.symmetry_point

    cmd = f"ght -D={opts.stretch_amount}"
    if opts.local_intensity != 0.0:
        cmd += f" -B={opts.local_intensity}"
    if opts.symmetry_point != 0.0:
        cmd += f" -SP={opts.symmetry_point}"
    if opts.shadow_protection != 0.0:
        cmd += f" -LP={opts.shadow_protection}"
    if opts.highlight_protection != 1.0:
        cmd += f" -HP={opts.highlight_protection}"
    if opts.clip_mode != "rgbblend":
        cmd += f" -clipmode={opts.clip_mode}"
    cmd += f" -{opts.color_model}"
    if opts.channels != "all":
        cmd += f" {opts.channels}"
    return cmd


def _build_asinh_cmd(opts: AsinhOptions) -> str:
    cmd = "asinh"
    if opts.color_model == "human":
        cmd += " -human"
    cmd += f" {opts.stretch_factor}"
    if opts.black_point_offset != 0.0:
        cmd += f" {opts.black_point_offset}"
    if opts.clip_mode != "rgbblend":
        cmd += f" -clipmode={opts.clip_mode}"
    return cmd


def _build_autostretch_cmd(opts: AutostretchOptions) -> str:
    cmd = "autostretch"
    if opts.linked:
        cmd += " -linked"
    cmd += f" {opts.shadows_clipping_sigma} {opts.target_background}"
    return cmd


def _parse_stretch_stats(stdout: str) -> dict:
    """Parse median/mean brightness from Siril stretch output if available."""
    stats: dict = {}
    m = re.search(r"median[:\s]+([\d.e+-]+)", stdout, re.IGNORECASE)
    if m:
        stats["median_brightness"] = float(m.group(1))
    m = re.search(r"mean[:\s]+([\d.e+-]+)", stdout, re.IGNORECASE)
    if m:
        stats["mean_brightness"] = float(m.group(1))
    return stats


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=StretchImageInput)
def stretch_image(
    method: str = "ghs",
    ghs_options: GHSOptions | None = None,
    asinh_options: AsinhOptions | None = None,
    autostretch_options: AutostretchOptions | None = None,
    output_suffix: str = "stretch",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Apply a non-linear stretch to the linear FITS image.

    Transforms linear (sensor-space) data into non-linear (display-space)
    brightness. All variants stretch from the same linear master — they are
    independent, not chained.

    Each call produces a variant with an auto-generated name from the method
    and parameters (e.g. ghs_d25_b0_sp009) unless output_suffix is set. The
    result includes the variant name and file path. Use select_stretch_variant
    to promote a variant to the active image.

    Methods:
    - ghs: Generalized Hyperbolic Stretch. Five parameters shape the transfer
      function (see GHSOptions).
    - asinh: arcsinh transfer function (see AsinhOptions).
    - autostretch: Siril automatic histogram-based stretch (see
      AutostretchOptions).
    """
    working_dir = state["dataset"]["working_dir"]
    metadata = state.get("metadata", {})

    # Always stretch from the linear master, not a previous variant.
    # First call saves the linear image as pre_stretch_image in metadata.
    # Subsequent calls load from that, so all variants are independent.
    pre_stretch = metadata.get("pre_stretch_image")
    image_path = pre_stretch if pre_stretch else state["paths"]["current_image"]

    if asinh_options is None:
        asinh_options = AsinhOptions()
    if autostretch_options is None:
        autostretch_options = AutostretchOptions()

    # Duplicate-call guard: reject identical (output_suffix, args) combinations.
    # This catches the model repeating the same stretch without making progress.
    # Different args with the same suffix (HITL-driven re-stretch) are allowed —
    # they overwrite the file intentionally with new parameters.
    _args_fingerprint = json.dumps({
        "method": method,
        "ghs": ghs_options.model_dump() if ghs_options else None,
        "asinh": asinh_options.model_dump() if asinh_options else None,
        "autostretch": autostretch_options.model_dump() if autostretch_options else None,
    }, sort_keys=True)
    _stretch_variants: dict = state.get("metadata", {}).get("stretch_variants", {})
    if _stretch_variants.get(output_suffix) == _args_fingerprint:
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"DUPLICATE STRETCH REJECTED — output_suffix={output_suffix!r} with these "
                    f"exact parameters was already applied. The variant already exists "
                    f"and re-running with identical parameters would produce an identical "
                    f"result.\n\n"
                    f"What to do:\n"
                    f"  - Call select_stretch_variant(variant='{output_suffix}') to switch "
                    f"to this variant, then call analyze_image to evaluate it.\n"
                    f"  - To create a genuinely different variant: use a new output_suffix "
                    f"AND change the parameters (e.g. lower stretch_amount for 'gentle', "
                    f"higher for 'aggressive', different highlight_protection).\n"
                    f"  - If the existing variants are good enough: call "
                    f"select_stretch_variant to pick the best one, then advance_phase."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if fits_has_nan(img_path):
        raise ValueError(
            f"Input image {img_path.name} contains NaN or Inf values. "
            "Stretching NaN data will produce an all-white or corrupted image. "
            "Trace back to the previous tool that produced this file — most likely "
            "T13 deconvolution diverged. Re-run T13 with lower iterations or higher "
            "regularization, or skip deconvolution if SNR is insufficient."
        )

    if method == "ghs" and ghs_options is None:
        raise ValueError(
            "ghs_options.stretch_amount is required for method=ghs."
        )

    stem = img_path.stem

    # Auto-generate suffix from parameters if not provided
    if not output_suffix or output_suffix == "stretch":
        if method == "ghs" and ghs_options is not None:
            d = int(ghs_options.stretch_amount * 10) if ghs_options.stretch_amount else 0
            b = int(ghs_options.local_intensity) if ghs_options.local_intensity else 0
            sp = int(ghs_options.symmetry_point * 1000) if ghs_options.symmetry_point else 0
            output_suffix = f"ghs_d{d}_b{b}_sp{sp:03d}"
        elif method == "asinh":
            sf = int(asinh_options.stretch_factor * 10) if asinh_options.stretch_factor else 0
            output_suffix = f"asinh_sf{sf}"
        elif method == "autostretch":
            bg = int(autostretch_options.target_background * 100) if autostretch_options.target_background else 25
            output_suffix = f"auto_bg{bg:03d}"

    output_stem = f"{stem}_{output_suffix}"

    if method == "ghs":
        stretch_cmd = _build_ghs_cmd(ghs_options)  # type: ignore[arg-type]
    elif method == "asinh":
        stretch_cmd = _build_asinh_cmd(asinh_options)
    else:
        stretch_cmd = _build_autostretch_cmd(autostretch_options)

    commands = [
        f"load {stem}",
        stretch_cmd,
        f"save {output_stem}",
    ]
    result = run_siril_script(commands, working_dir=working_dir, timeout=120)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"Stretch did not produce: {output_path}")

    stretch_stats = _parse_stretch_stats(result.stdout)

    summary: dict = {
        "output_path": str(output_path),
        "method": method,
        "output_suffix": output_suffix,
        "siril_command": stretch_cmd,
    }
    if method == "ghs" and ghs_options is not None:
        summary["ghs_parameters"] = {
            "stretch_amount": ghs_options.stretch_amount,
            "local_intensity": ghs_options.local_intensity,
            "symmetry_point": ghs_options.symmetry_point,
            "shadow_protection": ghs_options.shadow_protection,
            "highlight_protection": ghs_options.highlight_protection,
            "color_model": ghs_options.color_model,
            "channels": ghs_options.channels,
            "clip_mode": ghs_options.clip_mode,
        }
    elif method == "asinh":
        summary["asinh_parameters"] = {
            "stretch_factor": asinh_options.stretch_factor,
            "black_point_offset": asinh_options.black_point_offset,
        }
    elif method == "autostretch":
        summary["autostretch_parameters"] = {
            "shadows_clipping_sigma": autostretch_options.shadows_clipping_sigma,
            "target_background": autostretch_options.target_background,
            "linked": autostretch_options.linked,
        }
    if stretch_stats:
        summary["stretch_stats"] = stretch_stats

    return Command(update={
        "paths": {**state["paths"], "current_image": str(output_path)},
        "metadata": {
            **state["metadata"],
            "pre_stretch_image": metadata.get("pre_stretch_image", state["paths"]["current_image"]),
            "stretch_variants": {**_stretch_variants, output_suffix: _args_fingerprint},
        },
        # Stretch output is always non-linear — mark it so downstream
        # consumers (e.g. export_final) use the correct ICC source profile.
        "metrics": {**state.get("metrics", {}), "is_linear_estimate": False},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })


# ── Select variant tool ──────────────────────────────────────────────────────

@tool
def select_stretch_variant(
    variant: Annotated[str, Field(
        description=(
            "The output_suffix of the variant to select (e.g. 'aggressive', "
            "'moderate', 'gentle'). Must match a suffix used in a previous "
            "stretch_image call."
        ),
    )],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Select a previously created stretch variant as the active image.

    After creating multiple stretch variants with stretch_image and analyzing
    each with analyze_image, call this tool to set the best variant as the
    current image before advancing to the next phase. You can also select a
    variant to analyze it further, then create additional variants informed
    by those findings — all variants are independent stretches of the same
    linear master.
    """
    metadata = state.get("metadata", {})
    stretch_variants = metadata.get("stretch_variants", {})
    pre_stretch = metadata.get("pre_stretch_image")

    if not stretch_variants:
        return Command(update={
            "messages": [ToolMessage(
                content="No stretch variants exist yet. Call stretch_image first.",
                tool_call_id=tool_call_id,
            )],
        })

    if variant not in stretch_variants:
        available = ", ".join(f"'{k}'" for k in stretch_variants)
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"Variant '{variant}' does not exist. "
                    f"Available variants: {available}"
                ),
                tool_call_id=tool_call_id,
            )],
        })

    if not pre_stretch:
        return Command(update={
            "messages": [ToolMessage(
                content="Internal error: pre_stretch_image not set in metadata.",
                tool_call_id=tool_call_id,
            )],
        })

    # Build the variant file path from the linear master stem + suffix
    linear_stem = Path(pre_stretch).stem
    working_dir = state["dataset"]["working_dir"]
    variant_path = Path(working_dir) / f"{linear_stem}_{variant}.fit"
    if not variant_path.exists():
        variant_path = variant_path.with_suffix(".fits")
    if not variant_path.exists():
        return Command(update={
            "messages": [ToolMessage(
                content=(
                    f"Variant '{variant}' was recorded but the file is missing. "
                    f"Call stretch_image with output_suffix='{variant}' to recreate it."
                ),
                tool_call_id=tool_call_id,
            )],
        })

    # Parse the stored fingerprint to report which params this variant used
    try:
        params = json.loads(stretch_variants[variant])
    except (json.JSONDecodeError, TypeError):
        params = {}

    return Command(update={
        "paths": {**state["paths"], "current_image": str(variant_path)},
        "messages": [ToolMessage(
            content=json.dumps({
                "selected_variant": variant,
                "method": params.get("method", "unknown"),
                "status": "active",
                "hint": (
                    "This variant is now the active image. Call analyze_image "
                    "to evaluate it. You can create additional variants with "
                    "stretch_image — all variants stretch from the same linear "
                    "master, so they are independent."
                ),
            }, indent=2),
            tool_call_id=tool_call_id,
        )],
    })
