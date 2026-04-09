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
_RUNTIME_VLM_HITL: bool | None = None
_RUNTIME_VLM_AUTONOMOUS: bool | None = None
_RUNTIME_VLM_RETENTION_MAX: int | None = None
_RUNTIME_MEMORY_ENABLED: bool = False


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


def set_vlm_retention_max(value: int) -> None:
    """Called by the app to override the toml vlm_retention_max_images value."""
    global _RUNTIME_VLM_RETENTION_MAX
    _RUNTIME_VLM_RETENTION_MAX = int(value)


def set_memory_enabled(value: bool) -> None:
    """Called by the CLI/app to enable long-term memory."""
    global _RUNTIME_MEMORY_ENABLED
    _RUNTIME_MEMORY_ENABLED = value


def is_memory_enabled() -> bool:
    """Check if long-term memory is active for this session."""
    return _RUNTIME_MEMORY_ENABLED


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
    """Check if VLM is enabled during HITL conversations."""
    if _RUNTIME_VLM_HITL is not None:
        return _RUNTIME_VLM_HITL
    return _CFG.get("vlm_hitl", False)


def vlm_autonomous() -> bool:
    """Check if VLM is enabled outside HITL (agent self-inspection)."""
    if _RUNTIME_VLM_AUTONOMOUS is not None:
        return _RUNTIME_VLM_AUTONOMOUS
    return _CFG.get("vlm_autonomous", False)


def vlm_window_cap() -> int:
    """
    Hard total cap on images retained in the agent's view per agent_node call.

    Applies when vlm_autonomous is on; vlm_hitl alone uses gate-only
    visibility (no persistence outside the active gate).

    HITL gate images count against this cap like any other multimodal
    HumanMessage. The single exception is "gate overflow": when the active
    HITL gate alone contains more images than the cap, the gate is shown in
    full (so the user can reference any variant they see in Gradio) and no
    remaining budget is granted to older non-gate images. Effective ceiling:

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


# ── Affirmative detection ─────────────────────────────────────────────────────
# HITL responses are structured, not free-text parsed. The UI (CLI, Gradio,
# etc.) presents an explicit approve/revise choice and sends a sentinel string
# on approval. Any other string is treated as revision feedback.
#
# Two flavors of approval sentinel:
#
#   APPROVE_SENTINEL — bare approval ("approve current state, advance phase").
#     Used by clients that don't render the variant pool (e.g. CLI). Optionally
#     followed by a free-text note.
#
#   APPROVE_VARIANT_SENTINEL — explicit variant approval, payload is JSON with
#     {"id": "<variant_id>", "rationale": "<optional text>"}. Sent by the
#     Gradio UI when the human clicks a specific variant's Approve button.
#     The variant_id is unambiguous; the rationale is whatever was in the
#     textbox at click time, recorded as the human's stated reason.

APPROVE_SENTINEL = "__APPROVE__"
APPROVE_VARIANT_SENTINEL = "__APPROVE_VARIANT__"


def is_affirmative(response: str) -> bool:
    """Check if the response is any kind of approval (bare or variant-specific)."""
    text = response.strip()
    return text.startswith(APPROVE_SENTINEL) or text.startswith(APPROVE_VARIANT_SENTINEL)


def is_variant_approval(response: str) -> bool:
    """Check if the response is a variant-specific approval."""
    return response.strip().startswith(APPROVE_VARIANT_SENTINEL)


def extract_approval_note(response: str) -> str:
    """
    Extract the human's note from a bare approve-with-note response.
    Returns empty string for variant approvals (use parse_variant_approval
    for those).
    """
    text = response.strip()
    if text.startswith(APPROVE_VARIANT_SENTINEL):
        return ""
    if text.startswith(APPROVE_SENTINEL):
        return text[len(APPROVE_SENTINEL):].strip()
    return ""


def parse_variant_approval(response: str) -> tuple[str | None, str]:
    """
    Parse a variant approval sentinel into (variant_id, rationale).

    Format: __APPROVE_VARIANT__{"id": "T09_v3", "rationale": "..."}

    Returns (None, "") if the payload is malformed — the caller should
    treat this as revision feedback rather than approval.
    """
    text = response.strip()
    if not text.startswith(APPROVE_VARIANT_SENTINEL):
        return None, ""
    payload = text[len(APPROVE_VARIANT_SENTINEL):].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None, ""
    if not isinstance(data, dict):
        return None, ""
    variant_id = data.get("id")
    if not isinstance(variant_id, str) or not variant_id:
        return None, ""
    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""
    return variant_id, rationale


def build_variant_approval(variant_id: str, rationale: str = "") -> str:
    """
    Build the approval sentinel string for a variant. Used by UI clients.
    Always pairs with parse_variant_approval on the receiving side.
    """
    payload = json.dumps({"id": variant_id, "rationale": rationale})
    return f"{APPROVE_VARIANT_SENTINEL}{payload}"


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
