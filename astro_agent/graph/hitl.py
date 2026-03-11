"""
HITL helpers — config loading, affirmative detection, image extraction.

All HITL policy lives in hitl_config.toml. These helpers are used by the
hitl_check node in graph/nodes.py. See hitl_design.md for the full design.

HITL is policy enforcement, not agent decision. The hitl_check node sits
between action and agent. After every tool execution it checks: is this tool
in the config and enabled? If yes, fire interrupt(). The agent never decides
whether to pause — it just calls tools and gets intercepted.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from langchain_core.messages import ToolMessage


# ── Config loading ────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "hitl_config.toml"


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


_CFG = _load_config()


def tool_cfg(tool_id: str) -> dict:
    """Return the HITL config for a specific tool. Defaults to disabled."""
    return _CFG.get("hitl", {}).get(tool_id, {"enabled": False})


def is_enabled(tool_id: str) -> bool:
    """Check both the autonomous master flag and the per-tool enabled flag."""
    if _CFG.get("autonomous", False):
        return False
    return tool_cfg(tool_id).get("enabled", False)


def vlm_enabled() -> bool:
    """Check if VLM image injection is enabled for HITL feedback loops."""
    return _CFG.get("vlm_enabled", False)


# ── Tool name → HITL config key mapping ──────────────────────────────────────
#
# After action_node executes a tool, hitl_check looks up the tool name here.
# If the tool has a mapping AND is_enabled() returns True, interrupt() fires.
# Tools not in this mapping (e.g. analyze_image, auto_crop) never trigger HITL.

TOOL_TO_HITL: dict[str, str] = {
    # Preprocessing
    "build_masters":          "T02_masters",
    "siril_stack":            "T06_T07_stack",
    "select_frames":          "T06_T07_stack",
    # Linear
    "remove_gradient":        "T09_gradient",
    "color_calibrate":        "T10_color",
    "noise_reduction":        "T12_denoise",
    "deconvolution":          "T13_decon",
    # Stretch
    "stretch_image":          "T14_stretch",
    # Non-linear
    "star_removal":           "T15_star_removal",
    "curves_adjust":          "T16_curves",
    "local_contrast_enhance": "T17_local_contrast",
    "saturation_adjust":      "T18_saturation",
    "star_restoration":       "T19_star_restoration",
    "multiscale_process":     "T27_multiscale",
}


def resolve_hitl_checkpoint(messages: list) -> tuple[str | None, str | None]:
    """
    Check the most recent ToolMessage(s) for HITL-triggering tools.

    Returns (hitl_config_key, tool_name) or (None, None) if no HITL applies.

    The action node may have executed multiple tools in one step (parallel
    tool calls). We check the most recent batch of ToolMessages (walking
    backward until we hit a non-ToolMessage) and return the LAST one that
    has a HITL mapping — that's the most consequential tool to review.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        if msg.name in TOOL_TO_HITL:
            hitl_key = TOOL_TO_HITL[msg.name]
            return hitl_key, msg.name
    return None, None


# ── Affirmative detection ─────────────────────────────────────────────────────
# HITL responses are structured, not free-text parsed. The UI (CLI, Streamlit,
# etc.) presents an explicit approve/revise choice and sends this sentinel
# when the user approves. Any other string is treated as revision feedback.

APPROVE_SENTINEL = "__APPROVE__"


def is_affirmative(response: str) -> bool:
    return response.strip() == APPROVE_SENTINEL


# ── Image extraction from message history ─────────────────────────────────────

# Keys that tools use to report output image paths in their result dicts
_IMAGE_PATH_KEYS = (
    "output_path",
    "result_path",
    "stretched_image_path",
    "starless_image_path",
    "star_mask_path",
    "exported_path",
    "preview_path",
    "mask_path",
)


def images_from_tool(messages: list, tool_name: str) -> list[str]:
    """
    Scan message history for all image paths produced by a given tool.
    Returns paths in chronological order — first attempt through most recent.
    The UI decides how many to show (all, last N, before/after first and last).
    """
    paths = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            try:
                result = ast.literal_eval(msg.content)
                if isinstance(result, dict):
                    for key in _IMAGE_PATH_KEYS:
                        if path := result.get(key):
                            paths.append(path)
                            break
            except Exception:
                pass
    return paths
