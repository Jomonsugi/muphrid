"""
T07 — siril_stack

Combine accepted registered frames into a single high-SNR master light.

Frame exclusion mechanism:
    Siril's stack command operates on the *selected* frames in a .seq file.
    This tool:
      1. Parses the .seq file to build a filenum → 1-based-position map.
      2. Emits `unselect <seq> 1 <nb_images>` to deselect all frames.
      3. Emits `select <seq> <pos> <pos>` for each accepted frame whose
         filenum could be mapped to an included I-line position.
      4. Runs `stack` — only selected frames are included.

    IMPORTANT: Siril 1.4 `select` expects the 1-based POSITION of the frame
    within the .seq I-line ordering, NOT the filenum from the I-line. These
    coincide when filenums are dense 0-based (the common case), but can
    diverge when registration drops frames or when filenums are 1-based.
    Using position directly is always correct. Verified against Siril 1.4.2.

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
import logging
import re
from dataclasses import dataclass, field
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


logger = logging.getLogger(__name__)


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

@dataclass
class _SeqIndex:
    """Parsed .seq metadata needed to build valid select commands.

    Siril's `select seq from to` uses 1-based *position in the sequence*, NOT
    the filenum from the I-line. These only coincide when filenums are dense
    and 0-based (the usual case for pp_*/r_pp_* sequences produced by Siril
    1.4). M31's run exposed a case where that assumption broke: the agent
    generated a `select seq N N` whose N was rejected by Siril because the
    frame had been dropped or renumbered after registration.

    Fields:
        n_images:    nb_images from S-line (total frames in sequence).
        n_selected:  nb_selected from S-line (frames with selection flag on).
        filenum_to_pos:  str(filenum) → 1-based position in I-line order.
                         Only includes I-lines whose `included` flag == 1.
                         An index here is always a valid `select` target.
    """
    n_images: int = 0
    n_selected: int = 0
    filenum_to_pos: dict[str, int] = field(default_factory=dict)


def _parse_seq_file(seq_path: Path) -> _SeqIndex:
    """
    Parse a Siril .seq file and return indexing data safe for `select`.

    The returned ``filenum_to_pos`` maps each INCLUDED frame's filenum (as
    string, matching the key format used by analyze_frames / select_frames)
    to its 1-based position in I-line order. That position is what Siril's
    ``select`` expects — regardless of whether filenums are 0-based, 1-based,
    dense, or sparse.
    """
    idx = _SeqIndex()

    try:
        lines = seq_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return idx

    position = 0  # 1-based after increment
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("S "):
            s_match = re.match(
                r"S\s+'?([^']+?)'?\s+(\d+)\s+(\d+)\s+(\d+)",
                stripped,
            )
            if s_match:
                idx.n_images = int(s_match.group(3))
                idx.n_selected = int(s_match.group(4))

        elif stripped.startswith("I "):
            parts = stripped.split()
            if len(parts) >= 2:
                filenum = int(parts[1])
                # included flag is column 3 when present; default to 1 so we
                # gracefully handle older Siril variants that omit it.
                included = int(parts[2]) if len(parts) >= 3 else 1
                position += 1
                if included:
                    idx.filenum_to_pos[str(filenum)] = position

    return idx


def _parse_stacked_count(stdout: str) -> int | None:
    """
    Extract frame count from Siril's stack stdout (diagnostic, not authoritative).

    Siril 1.4 emits variations like:
        "Stacking 47 images"
        "Computing stack with 47 images"
        "log: Pixel-by-pixel rejection applied on 47 images"
        "log: Average stacking of 47 images"

    Returns None if no recognizable count is found — the master FITS existing
    is the real success signal, this is for mismatch detection.
    """
    if not stdout:
        return None

    patterns = [
        r"[Ss]tacking\s+(\d+)\s+image",
        r"[Ss]tack(?:ing)?\s+(?:of\s+)?(\d+)\s+image",
        r"rejection\s+applied\s+on\s+(\d+)\s+image",
        r"with\s+(\d+)\s+image",
    ]
    for pat in patterns:
        m = re.search(pat, stdout)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def _build_selection_commands(
    seq_name: str,
    seq_index: _SeqIndex,
    accepted_frames: list[str],
) -> tuple[list[str], list[str]]:
    """
    Generate unselect-all + select-accepted Siril commands.

    Uses 1-based position in the .seq I-line order (what Siril's ``select``
    command actually validates against), not the filenum. This handles the
    edge cases where registration renumbered / dropped frames:

      * Frame keys in ``accepted_frames`` come from analyze_frames, which
        parses the *calibrated* sequence. After registration, some frames
        may be dropped or renumbered in the registered sequence. Any frame
        whose filenum is not present in ``seq_index.filenum_to_pos`` is
        silently absent from the stack — the agent sees that mismatch via
        ``count_mismatch_note`` but the stack itself still completes.

      * Position bounds are clamped to ``[1, seq_index.n_images]``. Any
        accepted frame that would produce an out-of-range position is
        skipped with a diagnostic rather than emitting a command Siril
        would reject (the M31 stuck-loop trigger).

    Returns:
        (commands, skipped_keys):
            commands     — the Siril ssf lines to emit.
            skipped_keys — accepted_frames entries we could not map, so the
                           caller can surface them in the tool summary.
    """
    upper = seq_index.n_images if seq_index.n_images > 0 else 10_000_000
    cmds: list[str] = [f"unselect {seq_name} 1 {upper}"]
    skipped: list[str] = []

    for filename in accepted_frames:
        key = Path(filename).stem if "." in filename else filename
        position = seq_index.filenum_to_pos.get(key)
        if position is None:
            position = seq_index.filenum_to_pos.get(filename)

        if position is None:
            skipped.append(filename)
            logger.warning(
                "t07_stack: accepted frame %r not found in %s — registration "
                "may have dropped it.",
                filename, seq_name,
            )
            continue

        if position < 1 or position > upper:
            skipped.append(filename)
            logger.warning(
                "t07_stack: position %d for frame %r out of range [1, %d] "
                "in %s — skipping.",
                position, filename, upper, seq_name,
            )
            continue

        cmds.append(f"select {seq_name} {position} {position}")

    return cmds, skipped


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
    rejection_method: str,
    weighting: str,
    stack_method: str = "mean",
    rejection_sigma: list[float] | None = None,
    normalization: str = "addscale",
    fast_norm: bool = False,
    overlap_norm: bool = False,
    output_norm: bool = False,
    output_32bit: bool = True,
    rgb_equal: bool = False,
    output_name: str = "master_light",
    maximize: bool = False,
    upscale: bool = False,
    feather: int | None = None,
    rejection_maps: str | None = None,
    total_frames_hint: int | None = None,  # noqa: ARG001 — diagnostic hint, not used in command
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

    seq_index = _parse_seq_file(seq_path)
    # If the S-line is missing (parser couldn't read or it's corrupted) we
    # fall back to a permissive upper bound derived from the I-lines so the
    # unselect still covers every known frame.
    if seq_index.n_images == 0 and seq_index.filenum_to_pos:
        seq_index.n_images = max(seq_index.filenum_to_pos.values())
    if seq_index.n_images == 0:
        seq_index.n_images = len(accepted_frames)

    selection_cmds, skipped_frames = _build_selection_commands(
        registered_sequence, seq_index, accepted_frames
    )

    # Safety: if no frames could be mapped we'd end up stacking all of them
    # (the unselect-then-no-select combination leaves Siril with nothing
    # selected, and stack on an empty selection behaves erratically). Fail
    # loudly with an actionable message instead of producing junk output.
    if len(selection_cmds) == 1:  # only the unselect survived
        raise RuntimeError(
            f"siril_stack: none of the {len(accepted_frames)} accepted frames "
            f"could be mapped to positions in {registered_sequence}.seq "
            f"(sequence has {seq_index.n_images} frames, {seq_index.n_selected} "
            f"with selection flag, {len(seq_index.filenum_to_pos)} included in "
            f"I-lines). The accepted frame keys from analyze_frames do not "
            f"match any filenum in the registered sequence — this indicates "
            f"registration dropped every frame, or analyze_frames was run "
            f"against a different sequence. Re-run siril_register and "
            f"analyze_frames, then retry select_frames + siril_stack. "
            f"Skipped sample: {skipped_frames[:5]}"
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
    # 32-bit output implicitly forces normalization on; the explicit
    # output_norm field lets callers force it on for 16-bit too.
    if output_32bit or output_norm:
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

    # Artifact-based verification: the master FITS existing is authoritative
    # for "stack produced output". Supplement with stdout parse so the agent
    # can see whether Siril actually used all the frames it was told to, vs
    # silently skipping some. Mismatch vs accepted_frames_count is a signal
    # worth surfacing but not failing on (Siril may reject at stack time for
    # technical reasons unrelated to the agent's selection logic).
    stacked_count_from_stdout = _parse_stacked_count(result.stdout)
    count_mismatch_note = None
    if stacked_count_from_stdout is not None and stacked_count_from_stdout != len(accepted_frames):
        count_mismatch_note = (
            f"Siril stacked {stacked_count_from_stdout} frames but {len(accepted_frames)} "
            f"were selected. Some frames may have been rejected during stacking (e.g. "
            f"too few stars for normalization, dimensional mismatch). Check Siril output."
        )

    summary = {
        "output_path": str(master_path),
        "registered_sequence": registered_sequence,
        "accepted_frames_count": len(accepted_frames),
        "stacked_count_from_stdout": stacked_count_from_stdout,
        "total_frames_in_seq": seq_index.n_images,
        "frames_in_seq_selected_flag": seq_index.n_selected,
        "skipped_accepted_frames": skipped_frames,
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
    if count_mismatch_note:
        summary["count_mismatch_note"] = count_mismatch_note
    if skipped_frames:
        summary["skipped_note"] = (
            f"{len(skipped_frames)} accepted frame(s) were not present in the "
            f"registered sequence — likely dropped during registration. This "
            f"is informational, not an error."
        )

    return Command(update={
        "paths": {"current_image": str(master_path)},
        "messages": [ToolMessage(content=json.dumps(summary, indent=2, default=str), tool_call_id=tool_call_id)],
    })
