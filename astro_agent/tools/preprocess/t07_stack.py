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

import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from astro_agent.tools._siril import run_siril_script


# ── Pydantic input schema ──────────────────────────────────────────────────────

class SirilStackInput(BaseModel):
    working_dir: str = Field(
        description="Absolute path to the Siril working directory."
    )
    registered_sequence: str = Field(
        description=(
            "Name of the registered sequence (without .seq). "
            "Typically 'r_pp_lights_seq' from T04."
        )
    )
    accepted_frames: list[str] = Field(
        description=(
            "List of accepted frame keys from select_frames (T06). "
            "Must match frame keys in the .seq file."
        )
    )

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

@tool(args_schema=SirilStackInput)
def siril_stack(
    working_dir: str,
    registered_sequence: str,
    accepted_frames: list[str],
    rejection_method: str,
    stack_method: str = "mean",
    rejection_sigma: list[float] | None = None,
    normalization: str = "addscale",
    fast_norm: bool = False,
    overlap_norm: bool = False,
    output_norm: bool = False,
    weighting: str = "wfwhm",
    output_32bit: bool = True,
    rgb_equal: bool = False,
    output_name: str = "master_light",
    maximize: bool = False,
    upscale: bool = False,
    feather: int | None = None,
    rejection_maps: str | None = None,
    total_frames_hint: int | None = None,
) -> dict:
    """
    Stack registered frames into a master light FITS.

    Uses Siril select/unselect to include only accepted_frames, then runs
    the stack command with full control over rejection, normalization,
    weighting, and output options.

    Recommended settings by dataset size:
      - < 10 frames:  rejection='winsorized', sigma=[2.5, 2.5]
      - 10-30 frames: rejection='sigma_clipping', sigma=[3.0, 3.0]
      - > 30 frames:  rejection='sigma_clipping', sigma=[2.5, 2.5] (can be tighter)
      - > 100 frames: consider fast_norm=True for speed
    """
    if rejection_sigma is None:
        rejection_sigma = [3.0, 3.0]

    wdir = Path(working_dir)
    seq_path = wdir / f"{registered_sequence}.seq"

    name_to_index, n_frames = _parse_seq_file(seq_path)
    if n_frames == 0:
        n_frames = max(name_to_index.values()) + 1 if name_to_index else len(accepted_frames)

    total_frames = total_frames_hint if total_frames_hint is not None else n_frames

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
    if output_norm:
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

    n_accepted = len(accepted_frames)
    acceptance_rate = n_accepted / max(total_frames, 1)

    stack_metrics = {
        "stacked_count":       n_accepted,
        "total_frames":        total_frames,
        "acceptance_rate":     round(acceptance_rate, 3),
        "total_integration_s": None,
        "estimated_snr_gain":  round(n_accepted ** 0.5, 3),
        "background_noise":    result.parsed.get("background_noise"),
    }

    return {
        "master_light_path": str(master_path),
        "stack_metrics":     stack_metrics,
    }
