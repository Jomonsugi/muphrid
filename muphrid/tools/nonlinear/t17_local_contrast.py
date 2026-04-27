"""
T17 — local_contrast_enhance

Local-contrast and fine-detail enhancement. Four methods: CLAHE (adaptive
histogram equalization), unsharp mask, wavelet (per-scale reconstruction),
and edge-preserving filter (bilateral / guided).

Backend: Siril CLI — `clahe`, `unsharp`, `wavelet` + `wrecons`, `epf`.
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

class ClaheOptions(BaseModel):
    clip_limit: float = Field(
        default=2.0,
        description=(
            "Contrast-limiting threshold. Higher values permit stronger "
            "local contrast enhancement and also amplify noise proportionally."
        ),
    )
    tile_size: int = Field(
        default=8,
        description=(
            "Edge length of the histogram-equalization tile grid in pixels. "
            "Smaller = more local adaptation. Larger = closer to global "
            "equalization."
        ),
    )


class UnsharpOptions(BaseModel):
    sigma: float = Field(
        default=2.0,
        description=(
            "Gaussian blur sigma for the unsharp mask. Controls which "
            "spatial frequencies the mask enhances — smaller sigma "
            "targets finer-scale detail."
        ),
    )
    amount: float = Field(
        default=0.3,
        description=(
            "Blend factor for the sharpening: "
            "out = in * (1 + amount) + blurred * (-amount). "
            "Values above ~1.0 introduce visible ringing at edges."
        ),
    )


class WaveletOptions(BaseModel):
    num_layers: int = Field(
        default=5,
        description=(
            "Number of wavelet decomposition layers. Each layer represents "
            "a finer spatial scale."
        ),
    )
    algorithm: str = Field(
        default="bspline",
        description=(
            "Decomposition algorithm. "
            "bspline (type=2): B-spline wavelets — smoother response. "
            "linear (type=1): à trous algorithm — sharper response, more "
            "ringing."
        ),
    )
    layer_weights: list[float] = Field(
        description=(
            "Reconstruction weights for each layer (num_layers + 1 values). "
            "Layers ordered finest to coarsest: [layer1, layer2, ..., residual]. "
            "1.0 = unchanged, > 1.0 = boost that scale, < 1.0 = suppress. "
            "The residual (last value) is the smooth background."
        ),
    )


class EpfOptions(BaseModel):
    guided: bool = Field(
        default=False,
        description=(
            "False: bilateral filter — uses -si= and -ss=. "
            "True: guided filter — uses -sc= only (si/ss ignored). "
            "guide_image_stem may supply an external guide."
        ),
    )
    diameter: int = Field(
        default=5,
        description=(
            "Filter kernel diameter in pixels (-d=). Larger values are "
            "progressively more expensive; values > 20 become very slow. "
            "0 = auto-compute from spatial_sigma."
        ),
    )
    intensity_sigma: float = Field(
        default=0.02,
        description=(
            "Bilateral only (-si=). Intensity sigma for 32-bit images "
            "(0.0–1.0). Tonal-range scale over which the filter smooths. "
            "Ignored when guided=True."
        ),
    )
    spatial_sigma: float = Field(
        default=0.02,
        description=(
            "Bilateral only (-ss=). Spatial sigma for 32-bit images "
            "(0.0–1.0). Spatial extent of filter influence. "
            "Ignored when guided=True."
        ),
    )
    guided_sigma: float = Field(
        default=0.04,
        description=(
            "Guided filter only (-sc=). Smoothing strength for 32-bit "
            "images (0.0–1.0 range). Ignored when guided=False."
        ),
    )
    mod: float = Field(
        default=0.8,
        description=(
            "Blend fraction of the filtered result with input (-mod=, "
            "0.0–1.0). 1.0 = full filter output, 0.0 = no effect."
        ),
    )
    guide_image_stem: str | None = Field(
        default=None,
        description=(
            "FITS stem of an external guide image for the guided filter "
            "(-guideimage=). Must share dimensions with the input. "
            "None = self-guided."
        ),
    )


class LocalContrastInput(BaseModel):
    method: str = Field(
        default="wavelet",
        description=(
            "wavelet: per-scale reconstruction. Independent weights per spatial "
            "scale via layer_weights. "
            "edge_preserve: edge-preserving bilateral or guided filter (epf). "
            "Smooths while preserving edges. "
            "clahe: Contrast-Limited Adaptive Histogram Equalization. Per-tile "
            "local contrast boost; amplifies noise as a side-effect. "
            "unsharp: Gaussian unsharp mask."
        ),
    )
    clahe_options: ClaheOptions = Field(default_factory=ClaheOptions)
    unsharp_options: UnsharpOptions = Field(default_factory=UnsharpOptions)
    wavelet_options: WaveletOptions = Field(default_factory=WaveletOptions)
    epf_options: EpfOptions = Field(default_factory=EpfOptions)


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=LocalContrastInput)
def local_contrast_enhance(
    method: str = "wavelet",
    clahe_options: ClaheOptions | None = None,
    unsharp_options: UnsharpOptions | None = None,
    wavelet_options: WaveletOptions | None = None,
    epf_options: EpfOptions | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Enhance local contrast and fine detail.

    Methods:
    - wavelet: decomposes the image into num_layers scales and reconstructs
      with per-layer weights. layer_weights of all 1.0 is a passthrough.
    - edge_preserve: bilateral (guided=False) or guided (guided=True) filter
      via Siril `epf`. Smooths while preserving edges.
    - clahe: Siril `clahe`. Per-tile adaptive histogram equalization.
    - unsharp: Siril `unsharp`. Gaussian unsharp mask.

    All methods operate on paths.current_image and write a _lce suffixed
    output.
    """
    working_dir = state["dataset"]["working_dir"]
    image_path = state["paths"]["current_image"]

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

    summary: dict = {
        "output_path": str(output_path),
        "method": method,
    }
    if method == "clahe":
        summary["clahe_parameters"] = {
            "clip_limit": clahe_options.clip_limit,
            "tile_size": clahe_options.tile_size,
        }
    elif method == "unsharp":
        summary["unsharp_parameters"] = {
            "sigma": unsharp_options.sigma,
            "amount": unsharp_options.amount,
        }
    elif method == "edge_preserve":
        summary["epf_parameters"] = {
            "guided": epf_options.guided,
            "diameter": epf_options.diameter,
            "mod": epf_options.mod,
        }
        if epf_options.guided:
            summary["epf_parameters"]["guided_sigma"] = epf_options.guided_sigma
        else:
            summary["epf_parameters"]["intensity_sigma"] = epf_options.intensity_sigma
            summary["epf_parameters"]["spatial_sigma"] = epf_options.spatial_sigma
    elif method == "wavelet":
        summary["wavelet_parameters"] = {
            "num_layers": wavelet_options.num_layers,
            "algorithm": wavelet_options.algorithm,
            "layer_weights": wavelet_options.layer_weights,
        }

    return Command(update={
        "paths": {**state["paths"], "current_image": str(output_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
