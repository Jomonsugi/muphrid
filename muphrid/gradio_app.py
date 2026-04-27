# Muphrid - LLM agent for autonomous astrophotography post-processing
# Copyright (C) 2026 Micah Shanks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Gradio 6 async HITL interface for Muphrid.

Async-first design:
  - All event handlers are async generators
  - Graph built with AsyncSqliteSaver, initialized on app.load()
  - Settings applied reactively via .change() events
  - Error recovery in every handler — app never crashes
  - .queue() enabled for generator support and cancellation

Layout: two-panel (50/50) with chat on the left, gallery + activity log
on the right. Settings organized in tabs.

Usage:
    uv run python -m muphrid.gradio_app
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from muphrid.config import check_dependencies, load_settings, make_llm
from muphrid.graph.graph import build_graph
from muphrid.graph.hitl import (
    build_variant_approval,
    set_autonomous,
    set_vlm_autonomous,
    set_vlm_retention_max,
    set_memory_enabled,
    is_memory_enabled,
    vlm_window_cap,
)
from muphrid.graph.content import text_content
from muphrid.graph.memory import make_memory_store
from muphrid.graph.state import (
    ProcessingPhase,
    SessionContext,
    build_initial_message,
    make_empty_state,
)
from muphrid.tools.preprocess.t01_ingest import ingest_dataset

logger = logging.getLogger(__name__)


# ── Config loaders (sync, pure functions) ────────────────────────────────────


def _load_hitl_defaults() -> dict:
    """Read hitl_config.toml for initial checkbox values."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    config_path = Path(__file__).resolve().parent.parent / "hitl_config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def _load_equipment_defaults() -> dict:
    """Read equipment.toml for initial widget values."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    config_path = Path(__file__).resolve().parent.parent / "equipment.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


def _load_processing_defaults() -> dict:
    """Read defaults from processing.toml, with env var overrides for backwards compat."""
    from muphrid.config import _pcfg
    per_phase = _pcfg("limits", "per_phase") or {}
    return {
        "llm_model": os.environ.get("LLM_MODEL", "") or _pcfg("model", "default", "moonshotai/Kimi-K2.5"),
        "recursion_limit": int(os.environ.get("RECURSION_LIMIT", "") or _pcfg("limits", "recursion_limit", 200)),
        "max_tools_per_phase": int(os.environ.get("MAX_TOOLS_PER_PHASE", "") or _pcfg("limits", "max_tools_per_phase", 30)),
        "max_consecutive_same_tool": int(os.environ.get("MAX_CONSECUTIVE_SAME_TOOL", "") or _pcfg("limits", "max_consecutive_same_tool", 3)),
        "max_autonomous_nudges": int(os.environ.get("MAX_AUTONOMOUS_NUDGES", "") or _pcfg("limits", "max_autonomous_nudges", 2)),
        "max_silent_hitl_tools": int(os.environ.get("MAX_SILENT_HITL_TOOLS", "") or _pcfg("limits", "max_silent_hitl_tools", 3)),
        "max_tools_ingest": int(os.environ.get("MAX_TOOLS_INGEST", "") or per_phase.get("ingest", 5)),
        "max_tools_calibration": int(os.environ.get("MAX_TOOLS_CALIBRATION", "") or per_phase.get("calibration", 10)),
        "max_tools_registration": int(os.environ.get("MAX_TOOLS_REGISTRATION", "") or per_phase.get("registration", 5)),
        "max_tools_analysis": int(os.environ.get("MAX_TOOLS_ANALYSIS", "") or per_phase.get("analysis", 5)),
        "max_tools_stacking": int(os.environ.get("MAX_TOOLS_STACKING", "") or per_phase.get("stacking", 10)),
        "max_tools_linear": int(os.environ.get("MAX_TOOLS_LINEAR", "") or per_phase.get("linear", 20)),
        "max_tools_stretch": int(os.environ.get("MAX_TOOLS_STRETCH", "") or per_phase.get("stretch", 25)),
        "max_tools_nonlinear": int(os.environ.get("MAX_TOOLS_NONLINEAR", "") or per_phase.get("nonlinear", 25)),
        "max_tools_export": int(os.environ.get("MAX_TOOLS_EXPORT", "") or per_phase.get("export", 5)),
        "cleanup_previous_runs": _pcfg("behavior", "cleanup_previous_runs", True),
        "prune_phase_analysis": _pcfg("behavior", "prune_phase_analysis", True),
    }


def _make_thread_id(target: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = target.lower().replace(" ", "-")[:30]
    return f"run-{slug}-{ts}"


# ── Async resource lifecycle ─────────────────────────────────────────────────

_GRAPH = None


_DEPENDENCY_ERROR: str | None = None


async def _init_async_resources():
    """Build the LangGraph with AsyncSqliteSaver. Called once on app.load()."""
    global _GRAPH, _DEPENDENCY_ERROR
    if _GRAPH is not None:
        return

    # Check external dependencies (Siril, GraXpert, StarNet, ExifTool) early
    # so users see actionable errors on launch, not mid-pipeline.
    try:
        check_dependencies(load_settings())
    except Exception as e:
        _DEPENDENCY_ERROR = str(e)
        logger.error(f"Dependency check failed: {e}")

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("muphrid.graph.state", "ProcessingPhase")]
    )
    conn = await aiosqlite.connect("checkpoints.db")
    checkpointer = AsyncSqliteSaver(conn=conn, serde=serde)
    await checkpointer.setup()
    store = make_memory_store()
    _GRAPH = build_graph(checkpointer=checkpointer, store=store)
    logger.info("Async resources initialized: graph + AsyncSqliteSaver")


# ── FITS preview conversion (sync utility) ───────────────────────────────────


def _convert_fits_to_preview(
    image_paths: list[str],
    working_dir: str,
    is_linear: bool = True,
) -> list[str]:
    """Convert FITS paths to displayable JPG previews. Non-FITS pass through."""
    from muphrid.tools.utility.t22_generate_preview import generate_preview

    preview_paths = []
    for img in image_paths:
        p = Path(img)
        if p.suffix.lower() in (".fit", ".fits", ".fts"):
            preview_dir = Path(working_dir) / "previews"
            expected = preview_dir / f"preview_{p.stem}.jpg"
            if expected.exists():
                preview_paths.append(str(expected))
            else:
                try:
                    result = generate_preview(
                        working_dir=working_dir,
                        fits_path=str(p),
                        format="jpg",
                        quality=95,
                        auto_stretch_linear=is_linear,
                    )
                    preview_paths.append(result["preview_path"])
                except Exception as e:
                    logger.warning(f"Preview generation failed for {p.name}: {e}")
        else:
            preview_paths.append(img)
    return preview_paths


# ── Stream chunk parsing (sync, defensive) ───────────────────────────────────


def _parse_stream_chunks(
    chunk: dict,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    working_dir: str,
    is_linear: bool,
    active_hitl: bool = False,
) -> tuple[dict | None, bool]:
    """
    Parse a single stream chunk (stream_mode="updates") and update the UI lists.

    Returns (interrupt_payload_or_None, updated_active_hitl).
    Routes agent text to chat during HITL, activity log during autonomous.
    Defensive: handles None values, non-dict updates, unexpected chunk formats.
    """
    if "__interrupt__" in chunk:
        return chunk["__interrupt__"][0].value, active_hitl

    for node_name, update in chunk.items():
        if node_name == "__interrupt__":
            continue
        if not isinstance(update, dict):
            continue

        # Track active_hitl from state updates
        if "active_hitl" in update:
            active_hitl = update["active_hitl"]

        messages = update.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_str = json.dumps(tc.get("args", {}), indent=2, default=str)
                        activity_log.append({
                            "role": "assistant",
                            "content": f"**{tc['name']}**",
                            "metadata": {"title": f"Tool Call: {tc['name']}", "log": args_str},
                        })
                agent_text = text_content(msg.content)
                if agent_text.strip():
                    if active_hitl:
                        # During HITL: skip — the post-stream HITL panel renders
                        # this same text with the tool title and review footer.
                        pass
                    else:
                        # Autonomous: agent text goes to ACTIVITY LOG
                        activity_log.append({
                            "role": "assistant",
                            "content": agent_text[:200] + "..." if len(agent_text) > 200 else agent_text,
                            "metadata": {"title": "Agent reasoning"},
                        })

            elif isinstance(msg, ToolMessage):
                result_text = text_content(msg.content)

                if msg.name == "present_images":
                    try:
                        result = json.loads(result_text)
                        if result.get("status") == "presented":
                            title = result.get("title", "")
                            description = result.get("description", "")
                            images = result.get("images", [])
                            paths = [img["path"] for img in images]
                            labels = [img.get("label", f"Image {i+1}") for i, img in enumerate(images)]
                            preview_paths = _convert_fits_to_preview(paths, working_dir, is_linear)
                            if gallery_images is None:
                                gallery_images = []
                            gallery_images.clear()
                            for path, label in zip(preview_paths, labels):
                                if Path(path).exists():
                                    gallery_images.append((path, label))
                            if description:
                                chat_messages.append({
                                    "role": "assistant",
                                    "content": f"**{title}**\n\n{description}" if title else description,
                                })
                    except (json.JSONDecodeError, KeyError):
                        pass

                # Announce export paths in chat as soon as export_final completes,
                # so the human knows where the finished pipeline wrote its outputs.
                # Without this they'd have to dig through the activity log or the
                # filesystem to find the result.
                elif msg.name == "export_final":
                    try:
                        result = json.loads(result_text)
                        files = result.get("exported_files", [])
                        if files:
                            lines = ["**Pipeline complete — output files written:**", ""]
                            for f in files:
                                size_mb = f.get("file_size_mb", "?")
                                fmt = f.get("format", "?")
                                profile = f.get("icc_profile", "?")
                                path = f.get("path", "?")
                                lines.append(f"- `{path}`")
                                lines.append(f"  *{fmt} · {profile} · {size_mb} MB*")
                            chat_messages.append({
                                "role": "assistant",
                                "content": "\n".join(lines),
                            })
                    except (json.JSONDecodeError, KeyError):
                        pass

                display_text = result_text[:500] + "..." if len(result_text) > 500 else result_text
                activity_log.append({
                    "role": "assistant",
                    "content": f"**{msg.name}** result",
                    "metadata": {"title": f"Result: {msg.name}", "log": display_text},
                })

            elif isinstance(msg, HumanMessage):
                # Don't echo HITL prompts, nudges, or auto-approval messages
                kwargs = msg.additional_kwargs or {}
                if kwargs.get("is_hitl_prompt") or kwargs.get("is_nudge"):
                    pass
                elif text_content(msg.content).startswith("Approved"):
                    pass
                # else: could show human messages during HITL conversation if needed

        new_phase = update.get("phase")
        if new_phase is not None:
            phase_name = new_phase.value if isinstance(new_phase, ProcessingPhase) else str(new_phase)
            chat_messages.append({
                "role": "assistant",
                "content": f"--- Phase: **{phase_name.upper()}** ---",
            })

    return None, active_hitl


# ── Core streaming handler (async generator) ────────────────────────────────


async def _stream_graph(
    graph,
    config: dict,
    stream_input,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    variant_pool: list[dict],
    working_dir: str,
    is_linear: bool,
):
    """
    Core async streaming loop. Yields (chat, activity, gallery, variant_pool).
    Routes agent text to chat (HITL) or activity log (autonomous).
    Handles interrupt payloads for HITL display.

    variant_pool is replaced fresh from each HITL payload (not mutated in
    place) so the UI's variant panel re-renders via @gr.render when it
    changes. Empty list outside of HITL.
    """
    interrupt_payload = None
    active_hitl = False

    # Snapshot gallery content before streaming so we can detect whether
    # present_images updated it during this round.
    gallery_before_stream = list(gallery_images)

    async for chunk in graph.astream(stream_input, config=config, stream_mode="updates"):
        result, active_hitl = _parse_stream_chunks(
            chunk, chat_messages, activity_log, gallery_images,
            working_dir, is_linear, active_hitl,
        )
        if result is not None:
            interrupt_payload = result

        yield chat_messages, activity_log, gallery_images, variant_pool

    # After stream ends, handle any interrupt payload for UI display
    if interrupt_payload is not None:
        title = interrupt_payload.get("title", "Review")
        images = interrupt_payload.get("images", [])

        # Refresh the variant pool from the payload — replaces any prior
        # pool in the UI. Backend already cleared it on phase advance or
        # variant approval, so this naturally reflects the current gate.
        variant_pool = list(interrupt_payload.get("variant_pool", []) or [])

        # Keep the gallery in sync with the variant panel. If the agent
        # called present_images during this stream, its curated comparison
        # set (which may include reference images like the original) takes
        # precedence. Otherwise, populate the gallery directly from the
        # variant pool so the viewer matches the Approve buttons below.
        # This covers the silent-reexecution / backstop case where the
        # agent never narrated or presented images.
        gallery_refreshed = (gallery_images != gallery_before_stream)

        if variant_pool and working_dir and not gallery_refreshed:
            variant_paths = [v["file_path"] for v in variant_pool if v.get("file_path")]
            if variant_paths:
                preview_paths = _convert_fits_to_preview(variant_paths, working_dir, is_linear)
                gallery_images.clear()
                for v, p in zip(variant_pool, preview_paths):
                    if Path(p).exists():
                        gallery_images.append((p, f"{v['id']} — {v.get('label', '')}"))
        elif images and working_dir and not gallery_images:
            preview_paths = _convert_fits_to_preview(images, working_dir, is_linear)
            if gallery_images is None:
                gallery_images = []
            for i, p in enumerate(preview_paths):
                if Path(p).exists():
                    gallery_images.append((p, f"Variant {i + 1}"))

        # Show HITL review prompt in chat. Agent text is rendered earlier
        # in the streaming loop during active HITL — we don't repeat it here.
        if variant_pool:
            footer = (
                f"**{title}** — {len(variant_pool)} variant"
                f"{'s' if len(variant_pool) != 1 else ''} ready for review.\n\n"
                f"---\n"
                f"*Pick a variant from the panel to commit and advance, "
                f"or send feedback to iterate. Type a rationale in the box "
                f"to record it with your approval.*"
            )
        else:
            footer = (
                f"**{title}** — awaiting your review.\n\n"
                f"---\n"
                f"*No variants in the pool yet — send feedback to guide the agent.*"
            )

        chat_messages.append({"role": "assistant", "content": footer})

        yield chat_messages, activity_log, gallery_images, variant_pool


# ── Event handlers (async generators, direct binding) ────────────────────────


async def start_session(
    dataset_path: str,
    target_name: str,
    bortle: int | None,
    sqm: float | None,
    remove_stars: str,
    notes: str,
    pixel_size: float | None,
    sensor_type: str | None,
    focal_length: float | None,
    state: dict,
):
    """Start a new processing session. Async generator — yields UI updates."""
    chat_messages: list[dict] = []
    activity_log: list[dict] = []
    gallery_images: list[tuple] = []
    variant_pool: list[dict] = []

    if not dataset_path or not target_name:
        chat_messages.append({"role": "assistant", "content": "Please provide both a dataset path and target name."})
        yield chat_messages, activity_log, gallery_images, variant_pool, state
        return

    # Block session start if dependencies are missing (checked at app startup)
    if _DEPENDENCY_ERROR:
        chat_messages.append({
            "role": "assistant",
            "content": f"**Cannot start — missing dependencies.**\n\n{_DEPENDENCY_ERROR}",
        })
        yield chat_messages, activity_log, gallery_images, variant_pool, state
        return

    # Parse remove_stars dropdown value
    if remove_stars == "yes":
        rs = True
    elif remove_stars == "no":
        rs = False
    else:
        rs = None

    thread_id = _make_thread_id(target_name)
    chat_messages.append({"role": "assistant", "content": f"Starting session **{thread_id}**\n\nIngesting dataset from `{dataset_path}`..."})
    yield chat_messages, activity_log, gallery_images, variant_pool, state

    try:
        # Apply equipment overrides from UI to os.environ
        if pixel_size and pixel_size > 0:
            os.environ["PIXEL_SIZE_UM"] = str(pixel_size)
        if sensor_type:
            os.environ["SENSOR_TYPE_OVERRIDE"] = sensor_type
        if focal_length and focal_length > 0:
            os.environ["FOCAL_LENGTH_MM"] = str(focal_length)

        # Ingest dataset
        ingest_result = ingest_dataset.invoke({
            "root_directory": dataset_path,
            "thread_id": thread_id,
        })

        # Build session context
        session = SessionContext(
            target_name=target_name,
            bortle=bortle if bortle and bortle > 0 else None,
            sqm_reading=sqm if sqm and sqm > 0 else None,
            remove_stars=rs,
            notes=notes if notes else None,
        )

        # Build initial state
        dataset = ingest_result["dataset"]
        initial_state = make_empty_state(dataset=dataset, session=session)
        initial_state["messages"] = [
            HumanMessage(content=build_initial_message(
                dataset=dataset,
                session=session,
                ingest_summary=ingest_result.get("summary", {}),
            ))
        ]

        working_dir = dataset.get("working_dir", "")

        # Build config
        config = {"configurable": {"thread_id": thread_id}}
        recursion_limit = int(os.environ.get("RECURSION_LIMIT", "200"))
        if recursion_limit > 0:
            config["recursion_limit"] = recursion_limit

        # Update session state
        state = {
            "thread_id": thread_id,
            "config": config,
            "working_dir": working_dir,
            "is_linear": True,
        }

        # Display ingest results
        warnings = ingest_result.get("warnings", [])
        if warnings:
            warn_text = "\n".join(f"- {w}" for w in warnings)
            chat_messages.append({"role": "assistant", "content": f"Ingest warnings:\n{warn_text}"})

        files = dataset.get("files", {})
        chat_messages.append({
            "role": "assistant",
            "content": (
                f"Dataset ingested: {len(files.get('lights', []))} lights, "
                f"{len(files.get('darks', []))} darks, "
                f"{len(files.get('flats', []))} flats, "
                f"{len(files.get('biases', []))} biases\n\n"
                f"Processing **{target_name}**..."
            ),
        })

        # Save settings snapshot and register session for resume diff detection
        _save_settings_snapshot(working_dir)
        _register_session(thread_id, working_dir)

        # Log model and provider at pipeline start
        model = os.environ.get("LLM_MODEL", "unknown")
        provider = os.environ.get("LLM_PROVIDER", "unknown")
        logger.info(f"Pipeline starting: model={model}, provider={provider}, thread={thread_id}")

        yield chat_messages, activity_log, gallery_images, variant_pool, state

        # Initialize memory if enabled — fail loud, don't start with degraded search
        if is_memory_enabled():
            err = _init_memory()
            if err:
                chat_messages.append({
                    "role": "assistant",
                    "content": (
                        f"**Memory initialization failed — session cannot start.**\n\n{err}"
                    ),
                })
                yield chat_messages, activity_log, gallery_images, variant_pool, state
                return

        # Stream the graph
        async for chat_msgs, act_log, gal_imgs, vpool in _stream_graph(
            _GRAPH, config, initial_state,
            chat_messages, activity_log, gallery_images, variant_pool,
            working_dir, state.get("is_linear", True),
        ):
            yield chat_msgs, act_log, gal_imgs, vpool, state

    except Exception as e:
        logger.exception(f"Session error: {e}")
        chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        yield chat_messages, activity_log, gallery_images, variant_pool, state


async def send_message(
    user_text: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    variant_pool: list[dict],
    state: dict,
):
    """Handle user message during HITL — resume graph with feedback."""
    # Gradio 6 Gallery.preprocess(None) returns None for empty gallery
    if gallery_images is None:
        gallery_images = []
    if variant_pool is None:
        variant_pool = []

    if not user_text.strip():
        yield chat_messages, activity_log, gallery_images, variant_pool, state, ""
        return

    config = state.get("config")
    working_dir = state.get("working_dir", "")
    is_linear = state.get("is_linear", True)

    if not config:
        chat_messages.append({"role": "assistant", "content": "No active session. Start a new session first."})
        yield chat_messages, activity_log, gallery_images, variant_pool, state, ""
        return

    chat_messages.append({"role": "user", "content": user_text})
    yield chat_messages, activity_log, gallery_images, variant_pool, state, ""

    try:
        async for chat_msgs, act_log, gal_imgs, vpool in _stream_graph(
            _GRAPH, config, Command(resume=user_text),
            chat_messages, activity_log, gallery_images, variant_pool,
            working_dir, is_linear,
        ):
            yield chat_msgs, act_log, gal_imgs, vpool, state, ""
    except Exception as e:
        logger.exception(f"Send message error: {e}")
        chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        yield chat_messages, activity_log, gallery_images, variant_pool, state, ""


async def approve_variant_action(
    variant_id: str,
    variant_label: str,
    rationale: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    variant_pool: list[dict],
    state: dict,
):
    """
    Handle a variant Approve button click — resume the graph with a structured
    APPROVE_VARIANT sentinel that names the chosen variant and carries the
    textbox content as the human's rationale.

    The backend's hitl_check parses the sentinel, calls promote_variant to
    move the chosen variant's file to current_image, and clears the pool.
    """
    if gallery_images is None:
        gallery_images = []
    if variant_pool is None:
        variant_pool = []

    config = state.get("config")
    working_dir = state.get("working_dir", "")
    is_linear = state.get("is_linear", True)

    if not config:
        chat_messages.append({"role": "assistant", "content": "No active session."})
        yield chat_messages, activity_log, gallery_images, variant_pool, state, ""
        return

    # Echo the human's choice into chat for the audit trail
    if rationale.strip():
        chat_messages.append({
            "role": "user",
            "content": f"Approving **{variant_id}** ({variant_label}) — {rationale.strip()}",
        })
    else:
        chat_messages.append({
            "role": "user",
            "content": f"Approving **{variant_id}** ({variant_label}).",
        })
    yield chat_messages, activity_log, gallery_images, variant_pool, state, ""

    # Clear the variant panel immediately so the user sees feedback that
    # their approval was registered before the backend finishes streaming.
    variant_pool = []
    yield chat_messages, activity_log, gallery_images, variant_pool, state, ""

    sentinel = build_variant_approval(variant_id, rationale.strip())

    try:
        async for chat_msgs, act_log, gal_imgs, vpool in _stream_graph(
            _GRAPH, config, Command(resume=sentinel),
            chat_messages, activity_log, gallery_images, variant_pool,
            working_dir, is_linear,
        ):
            yield chat_msgs, act_log, gal_imgs, vpool, state, ""
    except Exception as e:
        logger.exception(f"Variant approve error: {e}")
        chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        yield chat_messages, activity_log, gallery_images, variant_pool, state, ""


async def resume_session(
    resume_id: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    variant_pool: list[dict],
    state: dict,
):
    """Resume a session from an existing checkpoint."""
    if gallery_images is None:
        gallery_images = []
    if variant_pool is None:
        variant_pool = []
    if not resume_id.strip():
        chat_messages.append({"role": "assistant", "content": "Please enter a thread ID to resume."})
        yield chat_messages, activity_log, gallery_images, variant_pool, state
        return

    config = {"configurable": {"thread_id": resume_id}}
    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "200"))
    if recursion_limit > 0:
        config["recursion_limit"] = recursion_limit

    state = {
        "thread_id": resume_id,
        "config": config,
        "working_dir": "",
        "is_linear": True,
    }

    chat_messages.append({"role": "assistant", "content": f"Resuming session **{resume_id}**..."})
    yield chat_messages, activity_log, gallery_images, variant_pool, state

    try:
        async for chat_msgs, act_log, gal_imgs, vpool in _stream_graph(
            _GRAPH, config, Command(resume="Continue from checkpoint."),
            chat_messages, activity_log, gallery_images, variant_pool,
            state.get("working_dir", ""), state.get("is_linear", True),
        ):
            yield chat_msgs, act_log, gal_imgs, vpool, state
    except Exception as e:
        logger.exception(f"Resume error: {e}")
        chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        yield chat_messages, activity_log, gallery_images, variant_pool, state


async def _check_resume_diffs(
    resume_id: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    variant_pool: list[dict],
    state: dict,
):
    """
    Check for settings diffs before resuming. If diffs found, show warning
    and make Confirm Resume button visible. If no diffs, proceed with resume.

    Yields: (chat, activity, gallery, variant_pool, state, confirm_btn_update)
    """
    if gallery_images is None:
        gallery_images = []
    if variant_pool is None:
        variant_pool = []
    confirm_hidden = gr.update(visible=False)
    confirm_visible = gr.update(visible=True)

    if not resume_id.strip():
        chat_messages.append({"role": "assistant", "content": "Please enter a thread ID to resume."})
        yield chat_messages, activity_log, gallery_images, variant_pool, state, confirm_hidden
        return

    # Look up the original session's settings
    working_dir = _lookup_session_dir(resume_id)
    if working_dir:
        original = _load_settings_snapshot(working_dir)
    else:
        original = None

    if original:
        current = _build_settings_snapshot()
        diffs = _diff_settings(original, current)

        if diffs:
            diff_text = "\n".join(diffs)
            chat_messages.append({
                "role": "assistant",
                "content": (
                    f"**Settings changed** since session `{resume_id}` started:\n\n"
                    f"{diff_text}\n\n"
                    f"Click **Confirm Resume** to proceed with the new settings, "
                    f"or update the tabs and click **Resume** to re-check."
                ),
            })
            yield chat_messages, activity_log, gallery_images, variant_pool, state, confirm_visible
            return

    # No diffs (or no snapshot found) — proceed directly
    async for result in resume_session(resume_id, chat_messages, activity_log, gallery_images, variant_pool, state):
        yield *result, confirm_hidden


# ── Settings helpers ─────────────────────────────────────────────────────────


def _apply_ui_settings(
    recursion_limit, max_tools_phase, max_consecutive, max_nudges,
    max_silent_hitl,
    phase_linear, phase_stretch, phase_nonlinear,
    cleanup_runs, prune_analysis,
    llm_model,
):
    """Apply all UI settings to os.environ. Called before start_session via .then() chaining."""
    os.environ["LLM_MODEL"] = llm_model
    from muphrid.config import _get_model_defaults
    os.environ["LLM_PROVIDER"] = _get_model_defaults(llm_model)["provider"]
    os.environ["RECURSION_LIMIT"] = str(int(recursion_limit))
    os.environ["MAX_TOOLS_PER_PHASE"] = str(int(max_tools_phase))
    os.environ["MAX_CONSECUTIVE_SAME_TOOL"] = str(int(max_consecutive))
    os.environ["MAX_AUTONOMOUS_NUDGES"] = str(int(max_nudges))
    os.environ["MAX_SILENT_HITL_TOOLS"] = str(int(max_silent_hitl))
    # Per-phase overrides — only the phases exposed in the UI
    os.environ["MAX_TOOLS_LINEAR"] = str(int(phase_linear))
    os.environ["MAX_TOOLS_STRETCH"] = str(int(phase_stretch))
    os.environ["MAX_TOOLS_NONLINEAR"] = str(int(phase_nonlinear))
    # Preprocessing phases use global default (from processing.toml)
    os.environ["CLEANUP_PREVIOUS_RUNS"] = "true" if cleanup_runs else "false"
    os.environ["PRUNE_PHASE_ANALYSIS"] = "true" if prune_analysis else "false"


def _init_memory() -> str | None:
    """Initialize long-term memory store. Returns error message or None on success."""
    from muphrid.memory.embeddings import EmbeddingInitError, init_memory_system

    try:
        settings = load_settings()
        init_memory_system(settings, rebuild_embeddings=settings.memory_rebuild_embeddings)
        return None
    except EmbeddingInitError as e:
        set_memory_enabled(False)
        logger.error(f"Memory initialization failed: {e}")
        return str(e)


def _on_memory_toggle(enabled: bool):
    """Handle memory checkbox toggle."""
    set_memory_enabled(enabled)
    if enabled:
        err = _init_memory()
        if err:
            import gradio as gr
            gr.Warning(f"Memory init failed: {err}")


def _format_model_info(model_name: str) -> str:
    """Display model defaults dynamically from _MODEL_DEFAULTS config."""
    from muphrid.config import _get_model_defaults, ConfigError
    try:
        defaults = _get_model_defaults(model_name)
    except ConfigError:
        return f"**{model_name}**: Not configured. Add it to `_MODEL_DEFAULTS` in `config.py`."
    provider = defaults["provider"]
    temp = defaults["temperature"]
    temp_str = str(temp) if temp is not None else "fixed by model"
    thinking = "enabled" if defaults["thinking"] else "disabled"
    budget = defaults.get("thinking_budget", 0)
    lines = [f"**{model_name}** — provider: {provider}, temperature: {temp_str}, thinking: {thinking}"]
    if budget > 0:
        lines.append(f"Thinking budget: {budget} tokens")
    return "  \n".join(lines)


# ── Session settings snapshot & diff ──────────────────────────────────────────

_SESSIONS_INDEX = Path.home() / ".muphrid" / "sessions.json"


def _build_settings_snapshot() -> dict:
    """Capture current runtime settings into a dict for later diff comparison."""
    from muphrid.graph.hitl import (
        is_autonomous, is_memory_enabled, vlm_autonomous,
        vlm_window_cap, TOOL_TO_HITL, is_enabled,
    )
    return {
        "model": os.environ.get("LLM_MODEL", ""),
        "autonomous": is_autonomous(),
        "memory": is_memory_enabled(),
        # vlm_hitl is always True (collaboration requires it). Kept in the
        # snapshot dict for backward-compat / display.
        "vlm_hitl": True,
        "vlm_autonomous": vlm_autonomous(),
        "vlm_retention_max_images": vlm_window_cap(),
        "recursion_limit": int(os.environ.get("RECURSION_LIMIT", "200")),
        "max_tools_per_phase": int(os.environ.get("MAX_TOOLS_PER_PHASE", "30")),
        "max_consecutive_same_tool": int(os.environ.get("MAX_CONSECUTIVE_SAME_TOOL", "3")),
        "max_autonomous_nudges": int(os.environ.get("MAX_AUTONOMOUS_NUDGES", "2")),
        "max_silent_hitl_tools": int(os.environ.get("MAX_SILENT_HITL_TOOLS", "3")),
        "max_tools_ingest": int(os.environ.get("MAX_TOOLS_INGEST", "5")),
        "max_tools_calibration": int(os.environ.get("MAX_TOOLS_CALIBRATION", "10")),
        "max_tools_registration": int(os.environ.get("MAX_TOOLS_REGISTRATION", "5")),
        "max_tools_analysis": int(os.environ.get("MAX_TOOLS_ANALYSIS", "5")),
        "max_tools_stacking": int(os.environ.get("MAX_TOOLS_STACKING", "10")),
        "max_tools_linear": int(os.environ.get("MAX_TOOLS_LINEAR", "20")),
        "max_tools_stretch": int(os.environ.get("MAX_TOOLS_STRETCH", "25")),
        "max_tools_nonlinear": int(os.environ.get("MAX_TOOLS_NONLINEAR", "25")),
        "max_tools_export": int(os.environ.get("MAX_TOOLS_EXPORT", "5")),
        "cleanup_previous_runs": os.environ.get("CLEANUP_PREVIOUS_RUNS", "true").lower() == "true",
        "prune_phase_analysis": os.environ.get("PRUNE_PHASE_ANALYSIS", "true").lower() == "true",
        "hitl": {
            hitl_key: is_enabled(hitl_key)
            for hitl_key in TOOL_TO_HITL.values()
        },
    }


def _diff_settings(old: dict, new: dict) -> list[str]:
    """Compare two settings snapshots. Returns human-readable diff strings."""
    diffs = []
    # Top-level keys
    for key in old:
        if key == "hitl":
            continue
        if old.get(key) != new.get(key):
            old_val = old.get(key)
            new_val = new.get(key)
            label = key.replace("_", " ").title()
            diffs.append(f"- **{label}**: `{old_val}` → `{new_val}`")
    # HITL per-tool
    old_hitl = old.get("hitl", {})
    new_hitl = new.get("hitl", {})
    for tool_key in sorted(set(list(old_hitl.keys()) + list(new_hitl.keys()))):
        old_v = old_hitl.get(tool_key, False)
        new_v = new_hitl.get(tool_key, False)
        if old_v != new_v:
            status_old = "enabled" if old_v else "disabled"
            status_new = "enabled" if new_v else "disabled"
            diffs.append(f"- **HITL {tool_key}**: {status_old} → {status_new}")
    return diffs


def _register_session(thread_id: str, working_dir: str):
    """Record thread_id → working_dir mapping for resume lookup."""
    _SESSIONS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    index = {}
    if _SESSIONS_INDEX.exists():
        try:
            index = json.loads(_SESSIONS_INDEX.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    index[thread_id] = working_dir
    _SESSIONS_INDEX.write_text(json.dumps(index, indent=2))


def _lookup_session_dir(thread_id: str) -> str | None:
    """Look up working_dir for a thread_id from the sessions index."""
    if not _SESSIONS_INDEX.exists():
        return None
    try:
        index = json.loads(_SESSIONS_INDEX.read_text())
        return index.get(thread_id)
    except (json.JSONDecodeError, OSError):
        return None


def _save_settings_snapshot(working_dir: str):
    """Write current settings to settings.json in the run directory."""
    snapshot = _build_settings_snapshot()
    settings_path = Path(working_dir) / "settings.json"
    settings_path.write_text(json.dumps(snapshot, indent=2))
    logger.info(f"Settings snapshot saved to {settings_path}")


def _load_settings_snapshot(working_dir: str) -> dict | None:
    """Read settings.json from a run directory."""
    settings_path = Path(working_dir) / "settings.json"
    if not settings_path.exists():
        return None
    try:
        return json.loads(settings_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ── Layout ───────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    hitl_defaults = _load_hitl_defaults()
    equip_defaults = _load_equipment_defaults()
    env_defaults = _load_processing_defaults()
    hitl_tools = hitl_defaults.get("hitl", {})

    with gr.Blocks(title="Muphrid") as app:
        session_state = gr.State({
            "thread_id": None,
            "config": None,
            "working_dir": "",
            "is_linear": True,
        })
        # Per-session variant pool. Re-set fresh from each HITL payload by the
        # streaming wrappers. The variant panel below uses @gr.render against
        # this state to redraw approve buttons whenever the pool changes.
        variant_pool_state = gr.State([])

        gr.Markdown("# Muphrid")

        # ── Processing tab ───────────────────────────────────────────
        with gr.Tab("Processing"):
            with gr.Row():
                with gr.Accordion("Start New Session", open=True):
                    with gr.Row():
                        dataset_path = gr.Textbox(
                            label="Dataset path",
                            placeholder="/path/to/dataset",
                            scale=3,
                        )
                        target_name = gr.Textbox(
                            label="Target",
                            placeholder="M42, NGC 7000, etc.",
                            scale=2,
                        )
                    with gr.Row():
                        bortle_input = gr.Number(
                            label="Bortle (1-9)",
                            value=None,
                            minimum=None,
                            maximum=9,
                            precision=0,
                        )
                        sqm_input = gr.Number(
                            label="SQM reading",
                            value=None,
                        )
                        remove_stars_input = gr.Dropdown(
                            label="Star removal",
                            choices=[("Auto (ask via HITL)", "none"), ("Yes", "yes"), ("No", "no")],
                            value="none",
                        )
                    start_btn = gr.Button("Start Processing", variant="primary")

                with gr.Accordion("Resume Session", open=False):
                    with gr.Row():
                        resume_id = gr.Textbox(
                            label="Thread ID",
                            placeholder="run-m42-...",
                            scale=3,
                        )
                        resume_btn = gr.Button("Resume", scale=1)
                        confirm_resume_btn = gr.Button(
                            "Confirm Resume", variant="stop", visible=False, scale=1,
                        )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(
                        label="Muphrid",
                        height=600,
                        buttons=["copy", "copy_all"],
                    )
                    msg_input = gr.Textbox(
                        placeholder="Reply, or type a rationale before approving a variant...",
                        lines=1,
                        max_lines=20,
                        show_label=False,
                        container=False,
                        autoscroll=True,
                    )

                with gr.Column(scale=1):
                    gallery = gr.Gallery(
                        label="Image Review",
                        columns=2,
                        height=400,
                        allow_preview=True,
                        buttons=["download"],
                        object_fit="contain",
                        interactive=False,
                    )
                    gr.Markdown("**Activity Log**")
                    activity = gr.Chatbot(
                        show_label=False,
                        height=300,
                        buttons=[],
                    )

            # ── Variant approval panel ───────────────────────────────────
            # Re-rendered whenever variant_pool_state changes. Each variant
            # gets one Approve button; clicking it sends a structured
            # APPROVE_VARIANT sentinel via the streaming backend. The textbox
            # content (if any) travels along as the human's recorded rationale.
            #
            # Defined after both columns so all referenced components
            # (chatbot, msg_input, activity, gallery, session_state) are in
            # scope when the render function is invoked.
            with gr.Group():
                @gr.render(inputs=variant_pool_state)
                def render_variant_panel(variants):
                    if not variants:
                        gr.Markdown(
                            "*No variants in the pool yet — the agent will "
                            "produce them as it iterates through this phase.*"
                        )
                        return
                    gr.Markdown(f"### Variants ({len(variants)})")
                    for v in variants:
                        vid = v.get("id", "?")
                        label = v.get("label", "")
                        metrics = v.get("metrics", {}) or {}
                        metric_bits = []
                        for key in ("gradient_magnitude", "snr_estimate", "signal_coverage_pct"):
                            val = metrics.get(key)
                            if val is None:
                                continue
                            short = key.split("_")[0]
                            if isinstance(val, float):
                                metric_bits.append(f"{short}={val:.3f}")
                            else:
                                metric_bits.append(f"{short}={val}")
                        metric_str = " · ".join(metric_bits)
                        with gr.Row():
                            desc = f"**{vid}** — {label}"
                            if metric_str:
                                desc += f"  \n*{metric_str}*"
                            gr.Markdown(desc)
                            btn = gr.Button(
                                f"Approve {vid}",
                                variant="primary",
                                size="sm",
                                scale=0,
                            )
                            # Capture vid + label in a closure so each button
                            # knows which variant it commits without needing
                            # gr.State() instances inside the render loop —
                            # creating fresh gr.State on every render triggers
                            # Svelte's effect_update_depth_exceeded infinite
                            # loop guard and freezes the entire UI.
                            #
                            # The closure is an async generator (matching
                            # approve_variant_action's signature) so Gradio's
                            # inspect-based dispatch routes it correctly.
                            async def _approve(
                                rationale, chat, act, gal, pool, sess,
                                _vid=vid, _label=label,
                            ):
                                async for result in approve_variant_action(
                                    _vid, _label, rationale,
                                    chat, act, gal, pool, sess,
                                ):
                                    yield result
                            btn.click(
                                fn=_approve,
                                inputs=[
                                    msg_input,
                                    chatbot,
                                    activity,
                                    gallery,
                                    variant_pool_state,
                                    session_state,
                                ],
                                outputs=[
                                    chatbot,
                                    activity,
                                    gallery,
                                    variant_pool_state,
                                    session_state,
                                    msg_input,
                                ],
                            )

        # ── Session Notes tab ────────────────────────────────────────
        with gr.Tab("Session Notes"):
            gr.Markdown(
                "Free-text context injected into the agent's initial prompt. "
                "Use for anything that affects processing but isn't captured elsewhere. "
                "Leave empty if not needed."
            )
            notes_input = gr.Textbox(
                label="Notes",
                placeholder="Shot with Optolong L-eNhance duoband filter\nVery poor seeing — FWHM likely > 4px",
                lines=10,
                max_lines=20,
            )

        # ── Equipment tab ────────────────────────────────────────────
        with gr.Tab("Equipment"):
            gr.Markdown(
                "Override values that can't be read from file metadata. "
                "Leave empty to auto-detect from FITS headers or EXIF."
            )
            pixel_size = gr.Number(
                label="Pixel size (um)",
                value=equip_defaults.get("camera", {}).get("pixel_size_um"),
                info="Not in EXIF for any DSLR/mirrorless",
            )
            sensor_type = gr.Dropdown(
                label="Sensor type",
                choices=["", "bayer", "xtrans", "mono"],
                value=equip_defaults.get("camera", {}).get("sensor_type", ""),
                info="Required for X-Trans (Fuji) to enable 3-pass demosaic.",
            )
            focal_length = gr.Number(
                label="Focal length (mm)",
                value=equip_defaults.get("optics", {}).get("focal_length_mm"),
                info="Approximate value for plate solving. Measured value from plate solve takes precedence.",
            )

        # ── HITL Config tab ──────────────────────────────────────────
        with gr.Tab("HITL Config"):
            gr.Markdown("Control which pipeline steps pause for human review.")
            autonomous_mode = gr.Checkbox(
                label="Autonomous mode (skip all HITL)",
                value=hitl_defaults.get("autonomous", False),
            )

            gr.Markdown("### Silent-Tool Backstop")
            gr.Markdown(
                "During an active HITL conversation, the agent normally re-runs "
                "tools and narrates the comparison before the next interrupt fires "
                "— that's how multi-variant comparisons work. But an agent that "
                "never pauses to narrate can string together many tool calls while "
                "the human is locked out. This backstop forces an interrupt after N "
                "HITL-mapped tools have run since the agent last spoke. "
                "Set to 0 to disable."
            )
            max_silent_hitl = gr.Number(
                label="Max silent HITL tools before forced interrupt",
                value=env_defaults["max_silent_hitl_tools"],
                precision=0,
                info="3 fits a typical 3-variant comparison; lower = more interrupts.",
            )
            gr.Markdown("### VLM (Visual Language Model)")
            gr.Markdown(
                "Preview images are injected as base64 into the agent's context so "
                "it can visually reason about results. VLM is always on during HITL "
                "conversations — collaboration on images requires the model to see "
                "what you're referencing. The autonomous toggle controls whether "
                "the agent has visual access outside HITL gates as well."
            )
            vlm_present = gr.Checkbox(
                label="VLM autonomous (agent sees the working image outside of HITL too)",
                value=hitl_defaults.get("vlm_autonomous", True),
            )
            vlm_retention = gr.Number(
                label="VLM retention cap (images)",
                value=hitl_defaults.get("vlm_retention_max_images", 8),
                precision=0,
                minimum=1,
                maximum=64,
                info=(
                    "Sliding-window cap on the number of images retained in the agent's "
                    "view. Active HITL gate variants are always shown in addition to "
                    "this cap."
                ),
            )
            gr.Markdown("### Long-Term Memory")
            gr.Markdown(
                "When enabled, the agent can search past processing sessions for relevant "
                "experience. Memories are extracted from HITL conversations after approval. "
                "Keep OFF during debugging — only enable when agent quality is stable."
            )
            memory_enabled = gr.Checkbox(
                label="Long-term memory (agent learns from past HITL sessions)",
                value=os.environ.get("MEMORY_ENABLED", "false").lower() == "true",
            )
            gr.Markdown("### Per-tool HITL checkpoints")
            gr.Markdown("Choose where to get involved. Every tool can be toggled independently.")

            # Build all HITL checkboxes from the config, grouped by phase
            _hitl_checkboxes = {}

            def _hitl_cb(key, label, default=False):
                cb = gr.Checkbox(label=label, value=hitl_tools.get(key, {}).get("enabled", default))
                _hitl_checkboxes[key] = cb
                return cb

            gr.Markdown("**Calibration**")
            _hitl_cb("T02_masters", "T02 Master Frame Diagnostics")
            _hitl_cb("T02b_convert", "T02b Sequence Conversion")
            _hitl_cb("T03_calibrate", "T03 Calibration")
            gr.Markdown("**Registration**")
            _hitl_cb("T04_register", "T04 Registration")
            gr.Markdown("**Analysis**")
            _hitl_cb("T05_analyze", "T05 Frame Analysis")
            gr.Markdown("**Stacking**")
            _hitl_cb("T06_select", "T06 Frame Selection")
            _hitl_cb("T07_stack", "T07 Stack Results")
            _hitl_cb("T08_crop", "T08 Auto Crop")
            gr.Markdown("**Linear**")
            _hitl_cb("T09_gradient", "T09 Gradient Removal", default=True)
            _hitl_cb("T10_color", "T10 Color Calibration")
            _hitl_cb("T11_green", "T11 Green Noise Removal")
            _hitl_cb("T12_denoise", "T12 Noise Reduction")
            _hitl_cb("T13_decon", "T13 Deconvolution")
            gr.Markdown("**Stretch**")
            _hitl_cb("T14_stretch", "T14 Stretch", default=True)
            gr.Markdown("**Non-linear**")
            _hitl_cb("T15_star_removal", "T15 Star Removal")
            _hitl_cb("T16_curves", "T16 Curves", default=True)
            _hitl_cb("T17_local_contrast", "T17 Local Contrast")
            _hitl_cb("T18_saturation", "T18 Saturation")
            _hitl_cb("T19_star_restoration", "T19 Star Restoration", default=True)
            _hitl_cb("T25_mask", "T25 Mask Creation")
            _hitl_cb("T26_reduce_stars", "T26 Star Reduction")
            _hitl_cb("T27_multiscale", "T27 Multiscale Sharpening")

        # ── Model & Limits tab ───────────────────────────────────────
        with gr.Tab("Model & Limits"):
            gr.Markdown("### Model")
            llm_model = gr.Dropdown(
                label="LLM Model",
                choices=["moonshotai/Kimi-K2.5", "deepseek-ai/DeepSeek-V3.1", "claude-sonnet-4-6"],
                value=env_defaults["llm_model"],
                allow_custom_value=True,
            )
            model_info = gr.Markdown(_format_model_info(env_defaults["llm_model"]))

            gr.Markdown("### Safety Limits")
            with gr.Row():
                recursion_limit = gr.Number(
                    label="Recursion limit",
                    value=env_defaults["recursion_limit"],
                    precision=0,
                    info=(
                        "Max LangGraph node transitions per stream() call. "
                        "~4 ticks per tool call plus ~6–10 extra per HITL gate "
                        "and ~4 per feedback round. 200 covers a typical run "
                        "with 4–6 gates and some iteration; bump to 300 for "
                        "heavy iteration."
                    ),
                )
                max_tools_phase = gr.Number(
                    label="Max tools per phase (default)",
                    value=env_defaults["max_tools_per_phase"],
                    precision=0,
                    info="Global default. Per-phase overrides below. 0 = unlimited.",
                )
            with gr.Row():
                max_consecutive = gr.Number(
                    label="Max identical tool calls per segment",
                    value=env_defaults["max_consecutive_same_tool"],
                    precision=0,
                    info=(
                        "Hard fail if agent calls a tool with identical args N times "
                        "within the current segment (since last human input or phase advance). "
                        "Counts interleaved repeats, not just strictly consecutive. 0 = disabled."
                    ),
                )
                max_nudges = gr.Number(
                    label="Max autonomous nudges",
                    value=env_defaults["max_autonomous_nudges"],
                    precision=0,
                    info="Max consecutive text-only responses before failing.",
                )

            gr.Markdown("### Per-Phase Overrides")
            gr.Markdown(
                "Override the global default for phases that need more headroom. "
                "Preprocessing phases use the global default (tool gating limits their scope). "
                "Fine-grained control in `processing.toml`."
            )
            with gr.Row():
                phase_linear = gr.Number(label="Linear", value=env_defaults["max_tools_linear"], precision=0)
                phase_stretch = gr.Number(label="Stretch", value=env_defaults["max_tools_stretch"], precision=0)
                phase_nonlinear = gr.Number(label="Non-linear", value=env_defaults["max_tools_nonlinear"], precision=0)

            gr.Markdown("### Behavior")
            cleanup_runs = gr.Checkbox(
                label="Clean up previous runs on fresh ingest",
                value=env_defaults["cleanup_previous_runs"],
                info="Delete previous run folders (runs/*) when starting a new session.",
            )
            prune_analysis = gr.Checkbox(
                label="Prune phase analysis from context",
                value=env_defaults["prune_phase_analysis"],
                info="Remove analyze_image outputs from completed phases to save context window.",
            )

        # ── Event wiring ─────────────────────────────────────────────

        # Settings applied reactively via .change() — immediate effect
        autonomous_mode.change(fn=lambda v: set_autonomous(v), inputs=[autonomous_mode])
        vlm_present.change(fn=lambda v: set_vlm_autonomous(v), inputs=[vlm_present])
        vlm_retention.change(fn=lambda v: set_vlm_retention_max(int(v)), inputs=[vlm_retention])
        memory_enabled.change(fn=_on_memory_toggle, inputs=[memory_enabled])
        llm_model.change(fn=_format_model_info, inputs=[llm_model], outputs=[model_info])

        # Wire all HITL checkboxes — each calls set_hitl_tool_enabled on change
        from muphrid.graph.hitl import set_hitl_tool_enabled
        for hitl_key, cb in _hitl_checkboxes.items():
            cb.change(
                fn=lambda v, k=hitl_key: set_hitl_tool_enabled(k, v),
                inputs=[cb],
            )

        # Start session: apply settings first, then stream
        start_btn.click(
            fn=_apply_ui_settings,
            inputs=[
                recursion_limit, max_tools_phase, max_consecutive, max_nudges,
                max_silent_hitl,
                phase_linear, phase_stretch, phase_nonlinear,
                cleanup_runs, prune_analysis,
                llm_model,
            ],
        ).then(
            fn=start_session,
            inputs=[
                dataset_path, target_name, bortle_input, sqm_input,
                remove_stars_input, notes_input,
                pixel_size, sensor_type, focal_length,
                session_state,
            ],
            outputs=[chatbot, activity, gallery, variant_pool_state, session_state],
        )

        # Direct handler binding — no wrappers
        msg_input.submit(
            fn=send_message,
            inputs=[msg_input, chatbot, activity, gallery, variant_pool_state, session_state],
            outputs=[chatbot, activity, gallery, variant_pool_state, session_state, msg_input],
        )

        # Bare Approve button is removed — approval is variant-specific via the
        # variant panel above. The textbox carries the rationale.

        # Resume: apply settings, check diffs, warn if changed
        _apply_inputs = [
            recursion_limit, max_tools_phase, max_consecutive, max_nudges,
            max_silent_hitl,
            phase_linear, phase_stretch, phase_nonlinear,
            cleanup_runs, prune_analysis,
            llm_model,
        ]

        resume_btn.click(
            fn=_apply_ui_settings,
            inputs=_apply_inputs,
        ).then(
            fn=_check_resume_diffs,
            inputs=[resume_id, chatbot, activity, gallery, variant_pool_state, session_state],
            outputs=[chatbot, activity, gallery, variant_pool_state, session_state, confirm_resume_btn],
        )

        # Confirm Resume: apply settings, skip diff check, proceed directly
        confirm_resume_btn.click(
            fn=_apply_ui_settings,
            inputs=_apply_inputs,
        ).then(
            fn=resume_session,
            inputs=[resume_id, chatbot, activity, gallery, variant_pool_state, session_state],
            outputs=[chatbot, activity, gallery, variant_pool_state, session_state],
        )

        # Initialize async resources on app load
        app.load(fn=_init_async_resources)

    return app


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = build_app()
    app.queue()
    app.launch(
        theme=gr.themes.Soft(primary_hue="blue"),
        allowed_paths=["/"],  # datasets can be anywhere on disk
        css="""
            .gradio-container { max-width: 100% !important; padding: 0 0.5rem !important; margin: 0 !important; }
            .main { max-width: 100% !important; }
            .contain { max-width: 100% !important; }
        """,
    )


if __name__ == "__main__":
    main()
