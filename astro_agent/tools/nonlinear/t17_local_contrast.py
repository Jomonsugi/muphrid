"""
T17 — local_contrast_enhance

Enhance fine detail and local contrast in nebulosity and galaxy structure
without affecting global brightness. Three complementary methods: CLAHE
(adaptive histogram equalization), unsharp mask, or wavelet sharpening.

Backend: Siril CLI — `clahe`, `unsharp`, or `wavelet` + `wrecons`.

Apply to the starless image only (post T15). Stars are added back later via
T19 — processing on a starless image prevents star-halo artefacts from
local contrast enhancement.

Wavelet sharpening offers the most surgical control: boost fine-scale detail
layers (1–2) while leaving coarse structure layers (3–5) and the residual
untouched. CLAHE is effective but can amplify noise — always apply after
noise reduction.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class ClaheOptions(BaseModel):
    clip_limit: float = Field(
        default=2.0,
        description=(
            "Contrast limiting threshold. Higher = stronger enhancement but "
            "more noise amplification. "
            "1.5–2.5 for most images. Reduce to 1.0–1.5 for noisy data."
        ),
    )
    tile_size: int = Field(
        default=8,
        description=(
            "Size of the histogram equalization tiles (in pixels). "
            "Smaller = more local adaptation (good for fine structure). "
            "Larger = more global. 8–16 is the typical range."
        ),
    )


class UnsharpOptions(BaseModel):
    sigma: float = Field(
        default=2.0,
        description=(
            "Gaussian blur sigma for the unsharp mask. Controls which spatial "
            "frequencies are enhanced. "
            "1.0–2.0: fine detail (star halos, nebula filaments). "
            "3.0–5.0: medium structure. "
            "Higher values affect larger-scale structure."
        ),
    )
    amount: float = Field(
        default=0.3,
        description=(
            "Blend factor for the sharpening effect. "
            "out = in * (1 + amount) + blurred * (-amount). "
            "0.1–0.3: gentle. 0.4–0.8: moderate. > 1.0: aggressive (ringing risk)."
        ),
    )


class WaveletOptions(BaseModel):
    num_layers: int = Field(
        default=5,
        description=(
            "Number of wavelet decomposition layers. "
            "Each layer represents a finer spatial scale. "
            "5 is standard; 4 for smaller images, 6 for very large images."
        ),
    )
    algorithm: str = Field(
        default="bspline",
        description=(
            "Decomposition algorithm. "
            "bspline (type=2): smoother, better for nebula structure. "
            "linear (type=1): à trous algorithm, sharper but more ringing risk."
        ),
    )
    layer_weights: list[float] = Field(
        default=[1.2, 1.1, 1.0, 1.0, 1.0, 1.0],
        description=(
            "Reconstruction weights for each layer (num_layers + 1 values). "
            "Layers ordered finest to coarsest: [layer1, layer2, ..., residual]. "
            "Weight > 1.0 = sharpen (boost that frequency). "
            "Weight = 1.0 = passthrough (no change). "
            "Weight < 1.0 = suppress (reduce that frequency). "
            "Example: [1.3, 1.1, 1.0, 1.0, 1.0, 1.0] sharpens finest two layers "
            "while leaving coarse structure and residual untouched."
        ),
    )


class EpfOptions(BaseModel):
    guided: bool = Field(
        default=False,
        description=(
            "False: bilateral filter — edge-aware smoothing using -si= and -ss=. "
            "True: guided filter — uses -sc= parameter only (si/ss are ignored). "
            "Guided is excellent for color denoising guided by luminance."
        ),
    )
    diameter: int = Field(
        default=5,
        description=(
            "Filter kernel diameter in pixels (-d=). "
            "Bilateral: 3–7 for noise smoothing; 9–15 for aggressive. "
            "Values > 20 are computationally expensive. "
            "0 = auto-compute from spatial_sigma."
        ),
    )
    intensity_sigma: float = Field(
        default=0.02,
        description=(
            "Bilateral only (-si=). Intensity sigma for 32-bit images (0.0–1.0). "
            "Controls tonal range over which the filter smooths. "
            "0.01–0.02: tight edge preservation. 0.05–0.1: stronger smoothing. "
            "Ignored when guided=True."
        ),
    )
    spatial_sigma: float = Field(
        default=0.02,
        description=(
            "Bilateral only (-ss=). Spatial sigma for 32-bit images (0.0–1.0). "
            "Controls spatial extent of filter influence. "
            "Ignored when guided=True."
        ),
    )
    guided_sigma: float = Field(
        default=0.04,
        description=(
            "Guided filter only (-sc=). Controls the guided filter's smoothing "
            "strength. For 32-bit images: 0.0–1.0 range. "
            "0.01–0.03: subtle smoothing. 0.04–0.1: moderate. 0.1+: aggressive. "
            "Ignored when guided=False."
        ),
    )
    mod: float = Field(
        default=0.8,
        description=(
            "Blend strength of the filtered result (-mod=, 0.0–1.0). "
            "1.0: full filter. 0.0: no effect."
        ),
    )
    guide_image_stem: str | None = Field(
        default=None,
        description=(
            "FITS stem of guide image for guided filter (-guideimage=). "
            "Guide must have same dimensions as input. "
            "Null = self-guided (most common)."
        ),
    )


class LocalContrastInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    image_path: str = Field(
        description=(
            "Absolute path to the non-linear starless FITS image. "
            "Must be post-star-removal (T15). "
            "Apply before saturation adjustment (T18)."
        )
    )
    method: str = Field(
        default="wavelet",
        description=(
            "wavelet: per-scale sharpening — best control, recommended for nebulae "
            "and galaxy structure. Surgically targets specific spatial scales. "
            "edge_preserve: edge-preserving bilateral/guided filter (epf) — "
            "structure-safe noise smoothing, excellent for cleaning residual noise "
            "in background and faint regions without blurring nebula edges. "
            "clahe: adaptive histogram equalization — effective for revealing faint "
            "nebula structure but amplifies noise; apply after edge_preserve. "
            "unsharp: Gaussian unsharp mask — gentle, fast, good starting point "
            "or combined with edge_preserve for a sharpen-then-smooth workflow."
        ),
    )
    clahe_options: ClaheOptions = Field(default_factory=ClaheOptions)
    unsharp_options: UnsharpOptions = Field(default_factory=UnsharpOptions)
    wavelet_options: WaveletOptions = Field(default_factory=WaveletOptions)
    epf_options: EpfOptions = Field(default_factory=EpfOptions)


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=LocalContrastInput)
def local_contrast_enhance(
    working_dir: str,
    image_path: str,
    method: str = "wavelet",
    clahe_options: ClaheOptions | None = None,
    unsharp_options: UnsharpOptions | None = None,
    wavelet_options: WaveletOptions | None = None,
    epf_options: EpfOptions | None = None,
) -> dict:
    """
    Enhance local contrast and fine detail in nebulosity/galaxy structure.

    Apply to the starless image only — doing this with stars present creates
    halos around bright stars from CLAHE and ringing from wavelet sharpening.

    Method guidance:
    - wavelet: surgical control per spatial scale. Use layer_weights > 1.0 only
      on fine layers (1–2); leave coarse layers (3+) at 1.0 to avoid global
      brightness shift. Start conservative (1.1–1.3) and iterate.
    - clahe: effective for revealing faint nebula structure but noisy. Always
      apply after noise reduction (T12). Use clip_limit 1.5–2.0.
    - unsharp: simplest and gentlest. Good for mild enhancement before more
      aggressive wavelet processing.

    Run analyze_image before and after to confirm the detail improvement
    without significant noise amplification.
    """
    if clahe_options is None:
        clahe_options = ClaheOptions()
    if unsharp_options is None:
        unsharp_options = UnsharpOptions()
    if wavelet_options is None:
        wavelet_options = WaveletOptions()
    if epf_options is None:
        epf_options = EpfOptions()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = img_path.stem
    output_stem = f"{stem}_lce"

    if method == "clahe":
        enhance_cmds: list[str] = [
            f"clahe {clahe_options.clip_limit} {clahe_options.tile_size}",
        ]
    elif method == "unsharp":
        enhance_cmds = [
            f"unsharp {unsharp_options.sigma} {unsharp_options.amount}",
        ]
    elif method == "edge_preserve":
        # Siril epf syntax (verified Siril 1.4):
        # Bilateral: epf [-d=] [-si=] [-ss=] [-mod=]
        # Guided:    epf -guided [-d=] [-sc=] [-mod=] [-guideimage=]
        o = epf_options
        epf_cmd = "epf"
        if o.guided:
            epf_cmd += " -guided"
        if o.diameter != 3:
            epf_cmd += f" -d={o.diameter}"
        if o.guided:
            epf_cmd += f" -sc={o.guided_sigma}"
        else:
            epf_cmd += f" -si={o.intensity_sigma} -ss={o.spatial_sigma}"
        epf_cmd += f" -mod={o.mod}"
        if o.guided and o.guide_image_stem:
            epf_cmd += f" -guideimage={o.guide_image_stem}"
        enhance_cmds = [epf_cmd]
    else:
        # wavelet: decompose then reconstruct with per-layer weights
        algo_int = 2 if wavelet_options.algorithm == "bspline" else 1
        weights = wavelet_options.layer_weights
        expected = wavelet_options.num_layers + 1
        # Pad or trim weights to match num_layers + 1 (layers + residual)
        if len(weights) < expected:
            weights = list(weights) + [1.0] * (expected - len(weights))
        weights = weights[:expected]
        weights_str = " ".join(str(w) for w in weights)
        enhance_cmds = [
            f"wavelet {wavelet_options.num_layers} {algo_int}",
            f"wrecons {weights_str}",
        ]

    commands = [f"load {stem}"] + enhance_cmds + [f"save {output_stem}"]
    run_siril_script(commands, working_dir=working_dir, timeout=120)

    output_path = Path(working_dir) / f"{output_stem}.fit"
    if not output_path.exists():
        output_path = Path(working_dir) / f"{output_stem}.fits"
    if not output_path.exists():
        raise FileNotFoundError(f"local_contrast_enhance did not produce: {output_path}")

    return {
        "enhanced_image_path": str(output_path),
        "method": method,
    }
