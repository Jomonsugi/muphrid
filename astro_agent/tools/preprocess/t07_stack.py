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

Always stack in 32-bit float to preserve dynamic range for linear processing.
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
            "List of accepted frame filenames from select_frames (T06). "
            "Pass the accepted_frames list unchanged — filenames must match "
            "those in the .seq file."
        )
    )
    stack_method: str = Field(
        default="mean",
        description="'mean' (default, best SNR with rejection) or 'median'.",
    )
    rejection_method: str = Field(
        default="sigma_clipping",
        description=(
            "'sigma_clipping' (default), 'winsorized' (better for < 15 frames), "
            "'linear_fit', or 'none'."
        ),
    )
    rejection_sigma: list[float] = Field(
        default=[3.0, 3.0],
        description="[low_sigma, high_sigma] for rejection.",
    )
    normalization: str = Field(
        default="addscale",
        description=(
            "'addscale' (additive + scale normalization, default), "
            "'multiplicative', or 'none'."
        ),
    )
    weighting: str = Field(
        default="wfwhm",
        description=(
            "Frame weighting strategy. "
            "'wfwhm' (weight by FWHM — favors sharper frames), "
            "'noise' (weight by noise level), 'nbstars', or 'none'."
        ),
    )
    output_32bit: bool = Field(
        default=True,
        description=(
            "Stack in 32-bit float. Always True for astrophotography — "
            "preserves dynamic range for subsequent linear processing."
        ),
    )
    total_frames_hint: int | None = Field(
        default=None,
        description=(
            "Total registered frames from T05 summary.frame_count. "
            "Used to compute acceptance rate for HITL. If omitted, "
            "the tool reads n_frames from the .seq file directly."
        ),
    )


# ── .seq parser ────────────────────────────────────────────────────────────────

def _parse_seq_file(seq_path: Path) -> tuple[dict[str, int], int]:
    """
    Parse a Siril .seq file and return ({frame_key: frame_index}, n_frames).

    Siril .seq format (v4+, from src/io/seqfile.c):
        S 'name' beg number selnum fixed refimage version ...
        I filenum incl [width,height]

    For FITSEQ: filenum is 0-based frame index, frame_key = str(filenum).
    For regular FITS: filenum maps to basename<filenum>.fit, frame_key = str(filenum).

    Returns both the mapping and total frame count from the S line.
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

    IMPORTANT: Siril 1.4 select/unselect use 1-based indexing at runtime,
    verified by ground-truth testing.  The .seq file's I-line filenum is
    0-based, but the select/unselect CLI commands are 1-based.
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
# Verified against Siril 1.4 docs:
#   stack seqname { rej | mean } [rejection_type] [sigma_low sigma_high] [options]
#   stack seqname { med | median } [options]    (no rejection args)
#   stack seqname { sum | min | max } [options] (no rejection args)
#
# Rejection algorithm codes (the argument AFTER the stack type for rej/mean):
#   n[one], s[igma], m[edian], w[insorized], l[inear], g[eneralized], ma[d]
#
# Note: "rej" is NOT a rejection algorithm — it is a stack type (synonym for "mean").

_REJECTION_ALGO_MAP = {
    "sigma_clipping": "s",    # Sigma clipping
    "winsorized":     "w",    # Winsorized sigma clipping (recommended for < 15 frames)
    "linear_fit":     "l",    # Linear-fit clipping
    "none":           "n",    # No rejection
}

# Maps agent stack_method name → Siril stack type token
_STACK_TYPE_MAP = {
    "mean":   "rej",    # rej and mean are synonyms in Siril; rej makes intent clear
    "median": "med",
    "sum":    "sum",
    "min":    "min",
    "max":    "max",
}

# Only these stack types support rejection parameters
_REJECTION_STACK_TYPES = {"mean"}

_WEIGHTING_MAP = {
    "wfwhm":   "-weight=wfwhm",
    "noise":   "-weight=noise",
    "nbstars": "-weight=nbstars",
    "none":    "",
}


# ── LangChain tool ─────────────────────────────────────────────────────────────

@tool(args_schema=SirilStackInput)
def siril_stack(
    working_dir: str,
    registered_sequence: str,
    accepted_frames: list[str],
    stack_method: str = "mean",
    rejection_method: str = "sigma_clipping",
    rejection_sigma: list[float] | None = None,
    normalization: str = "addscale",
    weighting: str = "wfwhm",
    output_32bit: bool = True,
    total_frames_hint: int | None = None,
) -> dict:
    """
    Stack accepted registered frames into a master light FITS.
    Uses Siril select/unselect to include only accepted_frames.
    Always stack 32-bit float. Returns master_light_path, stack metrics,
    and HITL flags if the accepted frame set is clearly too small to be useful.

    HITL fires for two obvious failure modes:
      - Fewer than 2 frames accepted (nothing meaningful to stack)
      - Less than 15% of all registered frames accepted with ≥ 5 total
        (overwhelming rejection — the entire session data may be unusable)
    """
    if rejection_sigma is None:
        rejection_sigma = [3.0, 3.0]

    wdir = Path(working_dir)
    seq_path = wdir / f"{registered_sequence}.seq"

    name_to_index, n_frames = _parse_seq_file(seq_path)
    if n_frames == 0:
        n_frames = max(name_to_index.values()) + 1 if name_to_index else len(accepted_frames)

    # Use caller-supplied total if available (more reliable than .seq header for edge cases)
    total_frames = total_frames_hint if total_frames_hint is not None else n_frames

    # Build selection preamble
    selection_cmds = _build_selection_commands(
        registered_sequence, n_frames, accepted_frames, name_to_index
    )

    siril_type = _STACK_TYPE_MAP.get(stack_method, "rej")
    weight_str = _WEIGHTING_MAP.get(weighting, "")
    sigma_lo, sigma_hi = rejection_sigma[0], rejection_sigma[1]

    # Build the stack command — rejection args only apply to "mean"/"rej" stacks
    stack_parts: list[str] = [f"stack {registered_sequence}", siril_type]

    if stack_method in _REJECTION_STACK_TYPES:
        # Verified Siril 1.4 syntax: stack seq rej {algo} {sigma_lo} {sigma_hi}
        # Rejection algo codes: s=sigma, w=winsorized, l=linear, n=none
        rej_algo = _REJECTION_ALGO_MAP.get(rejection_method, "s")
        stack_parts += [rej_algo, str(sigma_lo), str(sigma_hi)]

    stack_parts += [f"-norm={normalization}", weight_str, "-out=master_light"]
    if output_32bit:
        # Verified flag name: -32b (not -32bit) per Siril 1.4 docs
        stack_parts.append("-32b")

    stack_cmd = " ".join(p for p in stack_parts if p)

    commands = selection_cmds + [stack_cmd]

    result = run_siril_script(commands, working_dir=working_dir, timeout=1200)

    master_path = wdir / "master_light.fit"
    if not master_path.exists():
        master_path = wdir / "master_light.fits"
    if not master_path.exists():
        raise FileNotFoundError(
            f"master_light.fit not found in {wdir} after stacking.\n"
            f"Siril stdout:\n{result.stdout[-1000:]}"
        )

    n_accepted = len(accepted_frames)
    acceptance_rate = n_accepted / max(total_frames, 1)

    stack_metrics = {
        "stacked_count":      n_accepted,
        "total_frames":       total_frames,
        "acceptance_rate":    round(acceptance_rate, 3),
        "total_integration_s": None,     # populated by caller if exposure_time_s known
        "estimated_snr_gain": round(n_accepted ** 0.5, 3),
        "background_noise":   result.parsed.get("background_noise"),
    }

    # ── HITL: clearly insufficient light frames ────────────────────────────
    hitl_reasons: list[str] = []

    if n_accepted < 2:
        hitl_reasons.append(
            f"Only {n_accepted} light frame(s) accepted for stacking. "
            f"A minimum of 2 is needed for any meaningful integration. "
            f"Review frame selection criteria (T06) or capture more frames."
        )
    elif total_frames >= 5 and acceptance_rate < 0.15:
        hitl_reasons.append(
            f"Only {n_accepted}/{total_frames} frames accepted ({acceptance_rate:.0%}). "
            f"More than 85% of light frames were rejected — the session data may have "
            f"a systematic issue (severe seeing, tracking failure, clouds, or overly "
            f"strict selection thresholds in T06)."
        )

    hitl_required = len(hitl_reasons) > 0
    hitl_context = ""
    if hitl_required:
        hitl_context = (
            "Light frame stacking quality check failed.\n"
            + "\n".join(f"  - {r}" for r in hitl_reasons)
            + f"\nStack metrics: {stack_metrics}"
        )

    return {
        "master_light_path": str(master_path),
        "stack_metrics":     stack_metrics,
        "hitl_required":     hitl_required,
        "hitl_context":      hitl_context,
    }
