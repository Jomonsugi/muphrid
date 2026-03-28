"""
T07 — siril_stack

Combine accepted registered frames into a single high-SNR master light.

Frame exclusion mechanism:
    Siril's stack command operates on the *selected* frames in a .seq file.
    This tool:
      1. Parses the .seq file to build a filename→index map.
      2. Emits `unselect <seq> 1 <N>` to deselect all frames (1-based).
      3. Emits `select <seq> <idx+1> <idx+1>` for each accepted frame.
      4. Runs `stack` — only selected frames are included.

    IMPORTANT: Siril 1.4 select/unselect use 1-based indexing at runtime,
    verified by ground-truth testing against Siril 1.4.2.

Siril docs:
    stack seqname { sum | min | max } [-output_norm] [-out=] [-maximize] [-upscale] [-32b]
    stack seqname { med | median } [-nonorm, -norm=] [-fastnorm] [-rgb_equal]
        [-output_norm] [-out=] [-32b]
    stack seqname { rej | mean } [rejection_type] [sigma_low sigma_high]
        [-rejmap[s]] [-nonorm, -norm=] [-fastnorm] [-overlap_norm]
        [-weight={noise|wfwhm|nbstars|nbstack}] [-feather=]
        [-rgb_equal] [-output_norm] [-out=] [-maximize] [-upscale] [-32b]

    Rejection types: n[one], p[ercentile], s[igma], m[edian], w[insorized],
                     l[inear], g[eneralized], [m]a[d]

    Normalization: -norm=add, -norm=mul, -norm=addscale, -norm=mulscale, -nonorm
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
from muphrid.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SirilStackInput(BaseModel):
    # ── Stack method ──────────────────────────────────────────────────────
    stack_method: str = Field(
        default="mean",
        description=(
            "Stacking method. "
            "'mean' (default — average with rejection, best SNR), "
            "'median' (robust to outliers, no rejection needed, lower SNR), "
            "'sum' (additive — for photometry), "
            "'min' (minimum — for dark/hot pixel maps), "
            "'max' (maximum — for satellite/meteor detection)."
        ),
    )

    # ── Rejection (only for mean/rej stacking) ────────────────────────────
    rejection_method: str = Field(
        description=(
            "Pixel rejection algorithm (only for stack_method='mean'). "
            "Must be chosen based on accepted frame count from T06:\n"
            "  'winsorized': < 15 frames — replaces outliers without removing them, "
            "statistically valid for small N.\n"
            "  'sigma_clipping': 15–50 frames — standard sigma clip, "
            "requires sufficient N to estimate distribution.\n"
            "  'linear_fit': > 50 frames — most accurate for large datasets.\n"
            "  'generalized': mixed populations (e.g. dithered mosaic frames).\n"
            "  'mad': asymmetric noise distributions.\n"
            "  'percentile': fixed percentile bounds regardless of distribution.\n"
            "  'none': no rejection (use only when frame count is too low for any "
            "statistical rejection, e.g. < 5 frames)."
        ),
    )
    rejection_sigma: list[float] = Field(
        default=[3.0, 3.0],
        description=(
            "[low_sigma, high_sigma] for rejection algorithms. "
            "Tighter values (2.0) reject more aggressively; "
            "looser values (4.0) preserve more but risk artifacts."
        ),
    )

    # ── Normalization ─────────────────────────────────────────────────────
    normalization: str = Field(
        default="addscale",
        description=(
            "Input normalization before stacking. "
            "'addscale' (default — additive with scale, best general choice), "
            "'mulscale' (multiplicative with scale), "
            "'add' (additive only), "
            "'mul' (multiplicative only), "
            "'none' (no normalization)."
        ),
    )
    fast_norm: bool = Field(
        default=False,
        description=(
            "Use faster normalization estimators instead of IKSS. "
            "Faster but slightly less accurate. Use for large datasets."
        ),
    )
    overlap_norm: bool = Field(
        default=False,
        description=(
            "Compute normalization on image overlaps instead of whole images. "
            "Only valid with -maximize framing. Useful for mosaic stacking."
        ),
    )
    output_norm: bool = Field(
        default=False,
        description="Rescale the stacked result to [0, 1] range.",
    )

    # ── Weighting ─────────────────────────────────────────────────────────
    weighting: str = Field(
        default="wfwhm",
        description=(
            "Frame weighting strategy (only for mean/rej stacking). "
            "'wfwhm' (weight by weighted FWHM — favors sharper frames), "
            "'noise' (weight by inverse background noise), "
            "'nbstars' (weight by star count), "
            "'nbstack' (weight by prior stacking count — for live stacking), "
            "'none' (equal weight)."
        ),
    )

    # ── Output options ────────────────────────────────────────────────────
    output_32bit: bool = Field(
        default=True,
        description="Stack in 32-bit float. Always recommended for astrophotography.",
    )
    rgb_equal: bool = Field(
        default=False,
        description=(
            "Equalize RGB channel backgrounds during stacking. "
            "Useful when PCC/SPCC or unlinked autostretch will not be used."
        ),
    )
    output_name: str = Field(
        default="master_light",
        description="Output filename (without extension).",
    )

    # ── Framing ───────────────────────────────────────────────────────────
    maximize: bool = Field(
        default=False,
        description=(
            "Create full bounding-box output encompassing all frames. "
            "Larger canvas with borders — use with framing='max' in T04."
        ),
    )
    upscale: bool = Field(
        default=False,
        description="Upscale by 2× before stacking using registration data.",
    )
    feather: int | None = Field(
        default=None,
        description=(
            "Apply feathering mask on image borders over this distance (pixels). "
            "Smooths edge transitions for mosaic stacking."
        ),
    )

    # ── Rejection maps ────────────────────────────────────────────────────
    rejection_maps: str | None = Field(
        default=None,
        description=(
            "Generate rejection maps. "
            "'single' = one map showing all rejections, "
            "'split' = separate low/high rejection maps."
        ),
    )

    total_frames_hint: int | None = Field(
        default=None,
        description=(
            "Total registered frames from T05 summary.frame_count. "
            "Used to compute acceptance rate."
        ),
    )


# ── .seq parser ────────────────────────────────────────────────────────────────

def _parse_seq_file(seq_path: Path) -> tuple[dict[str, int], int]:
    """
    Parse a Siril .seq file and return ({frame_key: frame_index}, n_frames).
    """
    name_to_index: dict[str, int] = {}
    n_frames = 0

    try:
        lines = seq_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return name_to_index, n_frames

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("S "):
            s_match = re.match(
                r"S\s+'?([^']+?)'?\s+(\d+)\s+(\d+)",
                stripped,
            )
            if s_match:
                n_frames = int(s_match.group(3))

        elif stripped.startswith("I "):
            parts = stripped.split()
            if len(parts) >= 2:
                filenum = int(parts[1])
                name_to_index[str(filenum)] = filenum

    return name_to_index, n_frames


def _build_selection_commands(
    seq_name: str,
    n_frames: int,
    accepted_frames: list[str],
    name_to_index: dict[str, int],
) -> list[str]:
    """
    Generate unselect-all + select-accepted Siril commands.

    Siril 1.4 select/unselect use 1-based indexing at runtime.
    The .seq file's I-line filenum is 0-based.
    """
    cmds: list[str] = [f"unselect {seq_name} 1 {n_frames}"]

    for filename in accepted_frames:
        key = Path(filename).stem if "." in filename else filename
        idx = name_to_index.get(key)
        if idx is None:
            idx = name_to_index.get(filename)
        if idx is not None:
            cmds.append(f"select {seq_name} {idx + 1} {idx + 1}")

    return cmds


# ── Siril method maps ──────────────────────────────────────────────────────────

_REJECTION_ALGO_MAP = {
    "sigma_clipping": "s",
    "winsorized":     "w",
    "percentile":     "p",
    "median":         "m",
    "linear_fit":     "l",
    "generalized":    "g",
    "mad":            "a",
    "none":           "n",
}

_STACK_TYPE_MAP = {
    "mean":   "rej",
    "median": "med",
    "sum":    "sum",
    "min":    "min",
    "max":    "max",
}

_REJECTION_STACK_TYPES = {"mean"}

_NORM_MAP = {
    "addscale": "-norm=addscale",
    "mulscale": "-norm=mulscale",
    "add":      "-norm=add",
    "mul":      "-norm=mul",
    "none":     "-nonorm",
}

_WEIGHTING_MAP = {
    "wfwhm":   "-weight=wfwhm",
    "noise":   "-weight=noise",
    "nbstars": "-weight=nbstars",
    "nbstack": "-weight=nbstack",
    "none":    "",
}


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool
def siril_stack(
    rejection_method: str,
    weighting: str,
    stack_method: str = "mean",
    rejection_sigma: list[float] | None = None,
    normalization: str = "addscale",
    fast_norm: bool = False,
    overlap_norm: bool = False,
    output_32bit: bool = True,
    rgb_equal: bool = False,
    output_name: str = "master_light",
    maximize: bool = False,
    upscale: bool = False,
    feather: int | None = None,
    rejection_maps: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    state: Annotated[AstroState, InjectedState] = None,
) -> Command:
    """
    Stack registered frames into a master light FITS.

    Working directory, registered sequence, and accepted frame list are read
    from state. Uses Siril select/unselect to include only the accepted frames
    from select_frames (T06), then stacks with the specified method.

    Args:
        rejection_method: Pixel rejection algorithm (for stack_method='mean').
            'winsorized', 'sigma_clipping', 'linear_fit', 'none'.
            Choose based on accepted frame count from analyze_frames.
        stack_method: 'mean' (best SNR), 'median', 'sum', 'min', 'max'.
        rejection_sigma: [low_sigma, high_sigma]. Default [3.0, 3.0].
        normalization: 'addscale', 'mulscale', 'add', 'mul', 'none'.
        weighting: Frame weighting method. Choose based on what data is available:
            'wfwhm' — weights by weighted FWHM from registration. Requires
              per-frame wFWHM data in the sequence metadata. Available when
              registration was computed on the same sequence being stacked.
              May fail if registration data was lost during sequence operations.
            'noise' — weights by inverse noise level computed from image data.
              Always available regardless of registration method.
            'nbstars' — weights by number of detected stars per frame.
            'nbstack' — weights by number of stacked pixels per frame.
            'none' — equal weighting for all frames.
        output_32bit: Write 32-bit float output. Required for linear processing.
        output_name: Output filename stem (no extension).
    """
    if rejection_sigma is None:
        rejection_sigma = [3.0, 3.0]

    working_dir = state["dataset"]["working_dir"]
    registered_sequence = state["paths"]["registered_sequence"]
    accepted_frames = state["paths"].get("selected_frames") or []

    if not registered_sequence:
        raise ValueError("registered_sequence not found in state. Run siril_register (T04) first.")

    wdir = Path(working_dir)
    seq_path = wdir / f"{registered_sequence}.seq"

    name_to_index, n_frames = _parse_seq_file(seq_path)
    if n_frames == 0:
        n_frames = max(name_to_index.values()) + 1 if name_to_index else len(accepted_frames)

    selection_cmds = _build_selection_commands(
        registered_sequence, n_frames, accepted_frames, name_to_index
    )

    # ── Build stack command ───────────────────────────────────────────────
    siril_type = _STACK_TYPE_MAP.get(stack_method, "rej")
    stack_parts: list[str] = [f"stack {registered_sequence}", siril_type]

    if stack_method in _REJECTION_STACK_TYPES:
        rej_algo = _REJECTION_ALGO_MAP.get(rejection_method, "s")
        sigma_lo, sigma_hi = rejection_sigma[0], rejection_sigma[1]
        stack_parts += [rej_algo, str(sigma_lo), str(sigma_hi)]

        if rejection_maps == "single":
            stack_parts.append("-rejmap")
        elif rejection_maps == "split":
            stack_parts.append("-rejmaps")

    # Normalization
    norm_flag = _NORM_MAP.get(normalization, "-norm=addscale")
    stack_parts.append(norm_flag)

    if fast_norm:
        stack_parts.append("-fastnorm")
    if overlap_norm:
        stack_parts.append("-overlap_norm")

    # Weighting (only for mean/rej)
    if stack_method in _REJECTION_STACK_TYPES:
        weight_str = _WEIGHTING_MAP.get(weighting, "")
        if weight_str:
            stack_parts.append(weight_str)

        if feather is not None:
            stack_parts.append(f"-feather={feather}")

    if rgb_equal:
        stack_parts.append("-rgb_equal")

    # Output normalization: required for 32-bit float to produce [0,1] range.
    # Without it, pixel values stay in raw ADU scale and downstream tools
    # (which expect [0,1]) see near-zero values → data appears destroyed.
    # Siril docs: use -output_norm for light stacking, NOT for master frames
    # (master frames are built by build_masters, not this tool).
    if output_32bit:
        stack_parts.append("-output_norm")

    stack_parts.append(f"-out={output_name}")

    if maximize:
        stack_parts.append("-maximize")
    if upscale:
        stack_parts.append("-upscale")
    if output_32bit:
        stack_parts.append("-32b")

    stack_cmd = " ".join(p for p in stack_parts if p)
    commands = selection_cmds + [stack_cmd]

    result = run_siril_script(commands, working_dir=working_dir, timeout=1200)

    master_path = wdir / f"{output_name}.fit"
    if not master_path.exists():
        master_path = wdir / f"{output_name}.fits"
    if not master_path.exists():
        raise FileNotFoundError(
            f"{output_name}.fit not found in {wdir} after stacking.\n"
            f"Siril stdout:\n{result.stdout[-1000:]}"
        )

    summary = {
        "output_path": str(master_path),
        "registered_sequence": registered_sequence,
        "accepted_frames_count": len(accepted_frames),
        "total_frames_in_seq": n_frames,
        "settings": {
            "stack_method": stack_method,
            "rejection_method": rejection_method,
            "rejection_sigma": rejection_sigma,
            "normalization": normalization,
            "weighting": weighting,
            "output_32bit": output_32bit,
            "output_name": output_name,
        },
    }

    return Command(update={
        "paths": {**state["paths"], "current_image": str(master_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
