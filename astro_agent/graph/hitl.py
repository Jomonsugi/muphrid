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

import json
import tomllib

from astro_agent.graph.content import text_content
from pathlib import Path

from langchain_core.messages import ToolMessage


# ── Config loading ────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "hitl_config.toml"


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


_CFG = _load_config()

# Runtime overrides — CLI/Gradio can override toml settings.
_RUNTIME_AUTONOMOUS: bool = False
_RUNTIME_VLM_HITL: bool | None = None
_RUNTIME_VLM_AUTONOMOUS: bool | None = None


def set_autonomous(value: bool) -> None:
    """Called by the CLI/app to override the toml autonomous flag."""
    global _RUNTIME_AUTONOMOUS
    _RUNTIME_AUTONOMOUS = value


def set_vlm_hitl(value: bool) -> None:
    """Called by the app to override the toml vlm_hitl flag."""
    global _RUNTIME_VLM_HITL
    _RUNTIME_VLM_HITL = value


def set_vlm_autonomous(value: bool) -> None:
    """Called by the app to override the toml vlm_autonomous flag."""
    global _RUNTIME_VLM_AUTONOMOUS
    _RUNTIME_VLM_AUTONOMOUS = value


def tool_cfg(tool_id: str) -> dict:
    """Return the HITL config for a specific tool. Defaults to disabled."""
    return _CFG.get("hitl", {}).get(tool_id, {"enabled": False})


def is_enabled(tool_id: str) -> bool:
    """Check both the autonomous master flag and the per-tool enabled flag."""
    if is_autonomous():
        return False
    return tool_cfg(tool_id).get("enabled", False)


def is_autonomous() -> bool:
    """Check if the pipeline is running in autonomous mode (no human available)."""
    return _RUNTIME_AUTONOMOUS or _CFG.get("autonomous", False)


def vlm_hitl() -> bool:
    """Check if VLM is enabled during HITL conversations."""
    if _RUNTIME_VLM_HITL is not None:
        return _RUNTIME_VLM_HITL
    return _CFG.get("vlm_hitl", False)


def vlm_autonomous() -> bool:
    """Check if VLM is enabled outside HITL (agent self-inspection)."""
    if _RUNTIME_VLM_AUTONOMOUS is not None:
        return _RUNTIME_VLM_AUTONOMOUS
    return _CFG.get("vlm_autonomous", False)


# ── Tool name → HITL config key mapping ──────────────────────────────────────
#
# After action_node executes a tool, hitl_check looks up the tool name here.
# If the tool has a mapping AND is_enabled() returns True, interrupt() fires.
# Tools not in this mapping (e.g. analyze_image, auto_crop) never trigger HITL.

TOOL_TO_HITL: dict[str, str] = {
    # Preprocessing
    "build_masters":          "T02_masters",
    "select_frames":          "T06_select",
    "siril_stack":            "T07_stack",
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
# HITL responses are structured, not free-text parsed. The UI (CLI, Gradio,
# etc.) presents an explicit approve/revise choice and sends this sentinel
# when the user approves. Any other string is treated as revision feedback.

APPROVE_SENTINEL = "__APPROVE__"


def is_affirmative(response: str) -> bool:
    """Check if the response starts with the approval sentinel."""
    return response.strip().startswith(APPROVE_SENTINEL)


def extract_approval_note(response: str) -> str:
    """Extract the human's note from an approve-with-note response, or empty string."""
    text = response.strip()
    if text.startswith(APPROVE_SENTINEL):
        note = text[len(APPROVE_SENTINEL):].strip()
        return note
    return ""


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
                content = text_content(msg.content)
                result = json.loads(content)
                if isinstance(result, dict):
                    for key in _IMAGE_PATH_KEYS:
                        if path := result.get(key):
                            paths.append(path)
                            break
            except Exception:
                pass
    return paths
