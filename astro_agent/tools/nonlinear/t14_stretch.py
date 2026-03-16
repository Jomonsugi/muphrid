"""
T14 — stretch_image

Transform the image from linear to non-linear (perceptual) brightness space.
This is the single most impactful step — it makes the invisible faint signal
visible by applying a non-linear transfer function.

Backend: Siril CLI — GHS (Generalized Hyperbolic Stretch), arcsinh, or
autostretch. GHS offers the most control and is preferred.

Produce multiple variants using distinct output_suffix values (e.g. 'gentle',
'moderate', 'aggressive') so the best result can be selected. After stretch,
the image is non-linear — subsequent linear tools (T09–T13) must not be
applied.

Key metrics to monitor (from analyze_image after stretch):
  - clipped_shadows_pct < 0.5% — preserve faint nebulosity
  - clipped_highlights_pct < 0.1% — avoid burned star cores
Always use linked stretch (same transfer function across channels) after color
calibration to preserve white balance.
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

from astro_agent.graph.state import AstroState
from astro_agent.tools._siril import fits_has_nan, run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class GHSOptions(BaseModel):
    stretch_amount: float = Field(
        description=(
            "D: stretch strength 0–10. The primary control. "
            "Typical range: 1.5 (gentle, preserves faint detail) to 4.0 (aggressive). "
            "Start with 2.5 for a moderate first pass."
        ),
    )
    local_intensity: float = Field(
        default=0.0,
        description=(
            "B: focuses the stretch around the symmetry point (-5 to 15). "
            "0 = standard exponential stretch. "
            "Higher values (5–13) create a very targeted stretch around SP — "
            "useful for brightening specific tonal ranges without affecting others. "
            "Negative values create a de-focused stretch."
        ),
    )
    symmetry_point: float = Field(
        default=0.0,
        description=(
            "SP: the pixel value at which the stretch is most intense (0–1). "
            "0.0 = stretch darkest values most (standard first stretch). "
            "Set to the median of faint nebulosity for targeted enhancement."
        ),
    )
    shadow_protection: float = Field(
        default=0.0,
        description=(
            "LP: low-end linear region (0–SP). Pixels below LP are stretched linearly. "
            "Protects shadow detail from clipping. 0.0 = no protection (default)."
        ),
    )
    highlight_protection: float = Field(
        default=1.0,
        description=(
            "HP: high-end linear region (HP–1). Pixels above HP are stretched linearly. "
            "Protects bright star cores from bloating. 1.0 = no protection. "
            "0.9–0.95 prevents star bloat in most cases."
        ),
    )
    color_model: str = Field(
        default="human",
        description=(
            "human: preserves human color perception (L*a*b* lightness) — recommended "
            "for color images after color calibration. "
            "even: equal weights across channels. "
            "independent: each channel processed independently — use only if you "
            "intentionally want to alter white balance."
        ),
    )
    channels: str = Field(
        default="all",
        description=(
            "Which channels to apply the stretch to. "
            "all: all channels (default, use for linked stretch). "
            "R, G, B: single channel. RG, RB, GB: channel pairs."
        ),
    )
    clip_mode: str = Field(
        default="rgbblend",
        description=(
            "How out-of-range values are handled after stretch. "
            "'clip': hard clip to [0,1]. "
            "'rescale': rescale to fit. "
            "'rgbblend': blend RGB to preserve color (default, recommended). "
            "'globalrescale': rescale uniformly across all channels."
        ),
    )


class AsinhOptions(BaseModel):
    stretch_factor: float = Field(
        default=100.0,
        description=(
            "Stretch strength, typically 1–1000. "
            "Higher values = stronger stretch. 100 is a reasonable starting point."
        ),
    )
    black_point_offset: float = Field(
        default=0.0,
        description=(
            "Normalized offset (0–1) applied to the black point before stretching. "
            "Small positive values (0.0–0.05) lift the black level slightly."
        ),
    )
    color_model: str = Field(
        default="human",
        description=(
            "human: preserves L*a*b* lightness (recommended for color). "
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
            "Linked stretch: apply same parameters to all channels. "
            "Always use True after color calibration to preserve white balance."
        ),
    )


class StretchImageInput(BaseModel):
    method: str = Field(
        default="ghs",
        description=(
            "ghs: Generalized Hyperbolic Stretch — preferred for fine control. "
            "asinh: Arcsinh stretch — simpler, smooth shadow lifting. "
            "autostretch: Automatic histogram-based stretch — safe fallback / "
            "quick first look. Use linked=True to preserve color balance."
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
            "Suffix appended to the output filename stem. "
            "Use descriptive names when producing variants: 'gentle', 'moderate', "
            "'aggressive'. Allows the agent to produce multiple variants without "
            "overwriting each other."
        ),
    )


# ── Command builders ───────────────────────────────────────────────────────────

def _build_ghs_cmd(opts: GHSOptions) -> str:
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

    Transforms linear (un-stretched) data into perceptual brightness space,
    making faint signal visible. Use distinct output_suffix values (e.g.
    'gentle', 'moderate', 'aggressive') to produce multiple variants for
    comparison without overwriting each other.

    Method guidance:
    - ghs: best control, recommended. Use highlight_protection=0.92-0.98 to
      prevent star core saturation.
    - autostretch: reliable fallback, useful for a quick first look.
    - asinh: good for smooth shadow lifting without highlight crushing.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

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
                    f"exact parameters was already successfully applied in this session.\n\n"
                    f"Why this is a problem: stretch_image always operates on the current "
                    f"image in state, which is now the already-stretched output from the "
                    f"previous call. Applying the same transfer function again would "
                    f"double-stretch that result — shadows crushed to black, highlights "
                    f"blown out, stars bloated. The output would be corrupted and unusable.\n\n"
                    f"What to do:\n"
                    f"  - Call analyze_image to evaluate the existing {output_suffix!r} "
                    f"variant before deciding whether a new one is needed.\n"
                    f"  - To create a genuinely different variant: use a new output_suffix "
                    f"AND change the parameters (e.g. lower stretch_amount for 'gentle', "
                    f"higher for 'aggressive', different highlight_protection).\n"
                    f"  - If the existing variants are good enough: call advance_phase."
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
            "ghs_options.stretch_amount is required for method=ghs. "
            "Provide a stretch_amount (e.g., 2.5 for a moderate first stretch)."
        )

    stem = img_path.stem
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
            "stretch_variants": {**_stretch_variants, output_suffix: _args_fingerprint},
        },
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
