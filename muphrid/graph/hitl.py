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
import os
import tomllib

from muphrid.graph.content import text_content
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
_RUNTIME_VLM_AUTONOMOUS: bool | None = None
_RUNTIME_VLM_RETENTION_MAX: int | None = None


def set_autonomous(value: bool) -> None:
    """Called by the CLI/app to override the toml autonomous flag."""
    global _RUNTIME_AUTONOMOUS
    _RUNTIME_AUTONOMOUS = value


def set_vlm_autonomous(value: bool) -> None:
    """Called by the app to override the toml vlm_autonomous flag."""
    global _RUNTIME_VLM_AUTONOMOUS
    _RUNTIME_VLM_AUTONOMOUS = value


def set_vlm_retention_max(value: int) -> None:
    """Called by the app to override the toml vlm_retention_max_images value."""
    global _RUNTIME_VLM_RETENTION_MAX
    _RUNTIME_VLM_RETENTION_MAX = int(value)


def tool_cfg(tool_id: str) -> dict:
    """Return the HITL config for a specific tool. Defaults to disabled."""
    return _CFG.get("hitl", {}).get(tool_id, {"enabled": False})


def is_enabled(tool_id: str) -> bool:
    """Check autonomous flag, runtime overrides, then toml config."""
    if is_autonomous():
        return False
    # App runtime overrides take priority over toml config
    if tool_id in _RUNTIME_HITL_OVERRIDES:
        return _RUNTIME_HITL_OVERRIDES[tool_id]
    return tool_cfg(tool_id).get("enabled", False)


def is_autonomous() -> bool:
    """Check if the pipeline is running in autonomous mode (no human available)."""
    return _RUNTIME_AUTONOMOUS or _CFG.get("autonomous", False)


def vlm_hitl() -> bool:
    """
    VLM access during HITL conversations.

    Always True — collaboration is the entire point of HITL, and collaboration
    over images requires the model to see what the human is referencing. There
    is no user toggle for this; the toml ignores any value, the Gradio settings
    panel does not expose it. The retention cap and the autonomous-mode toggle
    remain user-controllable.
    """
    return True


def vlm_autonomous() -> bool:
    """Check if VLM is enabled outside HITL (agent self-inspection)."""
    if _RUNTIME_VLM_AUTONOMOUS is not None:
        return _RUNTIME_VLM_AUTONOMOUS
    return _CFG.get("vlm_autonomous", True)


def vlm_window_cap() -> int:
    """
    Hard total cap on images retained in the agent's view per agent_node call.

    Applies whenever the VLM view is being built. HITL gate variants count
    against this cap like any other source. The single exception is "gate
    overflow": when the active HITL gate alone contains more images than the
    cap, the gate is shown in full (so the user can reference any variant
    they see in Gradio) and no remaining budget is granted to older non-gate
    images. Effective ceiling:

        max(cap, |images in current HITL gate|)

    Resolution order: runtime override (Gradio) → env var → toml → default 8.
    """
    if _RUNTIME_VLM_RETENTION_MAX is not None:
        return _RUNTIME_VLM_RETENTION_MAX
    env = os.environ.get("VLM_RETENTION_MAX_IMAGES")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return int(_CFG.get("vlm_retention_max_images", 8))


# ── Phase eligibility for auto-current-image projection ──────────────────────
# Only the auto-current-image source consults this. Explicit visual
# affordances — present_images, variant_pool, phase_carry — bypass it: the
# agent retains full visual access on demand in any phase if it decides the
# image will inform a decision the metrics cannot.

_VLM_AUTO_PHASES = frozenset({
    "stacking",   # current_image becomes meaningful after siril_stack
    "linear",
    "stretch",
    "nonlinear",
    "review",
    "export",
})


def vlm_phase_eligible(phase) -> bool:
    """
    True when the auto-current-image source should project a preview.

    INGEST/CALIBRATION/REGISTRATION/ANALYSIS return False — current_image
    either does not exist yet (pre-stack) or pixel inspection adds nothing
    over the analytical metrics. STACKING onward returns True.

    Affordances the agent invokes deliberately (present_images, variant
    capture during a HITL-mapped tool, phase-carry from a prior promotion)
    are not gated by this — the agent's explicit choice always wins.
    """
    val = getattr(phase, "value", phase)
    if not isinstance(val, str):
        return False
    return val in _VLM_AUTO_PHASES


# ── Tool name → HITL config key mapping ──────────────────────────────────────
#
# After action_node executes a tool, hitl_check looks up the tool name here.
# If the tool has a mapping AND is_enabled() returns True, interrupt() fires.
# ALL phase-specific tools are mapped — the human can choose to get involved
# at any step. Utility tools (analyze_image, etc.) are not mapped because
# they are diagnostic, not transformative.

TOOL_TO_HITL: dict[str, str] = {
    # Calibration
    "build_masters":          "T02_masters",
    "convert_sequence":       "T02b_convert",
    "calibrate":              "T03_calibrate",
    # Registration
    "siril_register":         "T04_register",
    # Analysis
    "analyze_frames":         "T05_analyze",
    # Stacking
    "select_frames":          "T06_select",
    "siril_stack":            "T07_stack",
    "auto_crop":              "T08_crop",
    # Linear
    "remove_gradient":        "T09_gradient",
    "color_calibrate":        "T10_color",
    "remove_green_noise":     "T11_green",
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
    "create_mask":            "T25_mask",
    "reduce_stars":           "T26_reduce_stars",
    "multiscale_process":     "T27_multiscale",
}


# ── Runtime HITL overrides (app sets these, CLI uses toml config) ────────────

_RUNTIME_HITL_OVERRIDES: dict[str, bool] = {}


def set_hitl_tool_enabled(hitl_key: str, enabled: bool) -> None:
    """Called by the app to override per-tool HITL state at runtime."""
    _RUNTIME_HITL_OVERRIDES[hitl_key] = enabled


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
