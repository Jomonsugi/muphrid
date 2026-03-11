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

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the linear FITS image to stretch. "
            "This is the calibrated, registered, stacked, and optionally "
            "gradient-removed, color-calibrated, and noise-reduced linear image."
        )
    )
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
    working_dir: str,
    image_path: str,
    method: str = "ghs",
    ghs_options: GHSOptions | None = None,
    asinh_options: AsinhOptions | None = None,
    autostretch_options: AutostretchOptions | None = None,
    output_suffix: str = "stretch",
) -> dict:
    """
    Apply a non-linear stretch to the linear FITS image.

    This is THE CROSSING — after this tool, the image is non-linear.
    ALL subsequent tools (T15–T19, T25–T27) must only be called post-stretch.
    ALL linear tools (T09–T13) must NOT be called after stretch.

    Call this tool multiple times with different parameters to produce variants
    for comparison, using distinct output_suffix values:
      - 'gentle': ghs_options.stretch_amount=1.5, highlight_protection=0.98
      - 'moderate': ghs_options.stretch_amount=2.5 (default)
      - 'aggressive': ghs_options.stretch_amount=4.0, highlight_protection=0.92

    Method guidance:
    - ghs: best control, recommended. Use highlight_protection=0.92-0.98 to
      prevent star core saturation.
    - autostretch: reliable fallback, useful for a quick first look.
    - asinh: good for smooth shadow lifting without highlight crushing.

    After stretching, always run analyze_image and check:
    - clipped_shadows_pct < 0.5% (faint nebulosity preserved)
    - clipped_highlights_pct < 0.1% (star cores not burned)
    """
    if asinh_options is None:
        asinh_options = AsinhOptions()
    if autostretch_options is None:
        autostretch_options = AutostretchOptions()

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

    stats = _parse_stretch_stats(result.stdout)

    return {
        "stretched_image_path": str(output_path),
        "method": method,
        "output_suffix": output_suffix,
        "is_linear": False,
        "statistics": {
            "median_brightness": stats.get("median_brightness"),
            "mean_brightness": stats.get("mean_brightness"),
            "clipped_shadows_pct": None,   # use analyze_image for full stats
            "clipped_highlights_pct": None,
        },
    }
