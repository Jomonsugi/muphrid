"""
Gradio HITL interface for AstroAgent.

Two-panel layout: chat (conversation) on the left, activity log + image gallery
on the right. Settings are organized in tabs (Equipment, HITL Config, Model).

Usage:
    uv run python -m astro_agent.gradio_app
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from astro_agent.config import check_dependencies, load_settings, make_llm
from astro_agent.graph.graph import build_graph
from astro_agent.graph.hitl import APPROVE_SENTINEL, set_autonomous, set_vlm_hitl, set_vlm_autonomous, set_memory_enabled
from astro_agent.graph.content import text_content
from astro_agent.graph.memory import make_memory_store
from astro_agent.graph.state import (
    ProcessingPhase,
    SessionContext,
    build_initial_message,
    make_empty_state,
)
from astro_agent.tools.preprocess.t01_ingest import ingest_dataset

logger = logging.getLogger(__name__)


# ── Defaults from config files ──────────────────────────────────────────────

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


def _load_env_defaults() -> dict:
    """Read .env-sourced defaults from os.environ."""
    return {
        # Model
        "llm_model": os.environ.get("LLM_MODEL", "moonshotai/Kimi-K2.5"),
        "llm_provider": os.environ.get("LLM_PROVIDER", "together"),
        "llm_temperature": float(os.environ.get("LLM_TEMPERATURE", "0")),
        # Safety limits
        "recursion_limit": int(os.environ.get("RECURSION_LIMIT", "100")),
        "max_tools_per_phase": int(os.environ.get("MAX_TOOLS_PER_PHASE", "30")),
        "max_consecutive_same_tool": int(os.environ.get("MAX_CONSECUTIVE_SAME_TOOL", "3")),
        "max_autonomous_nudges": int(os.environ.get("MAX_AUTONOMOUS_NUDGES", "2")),
        # Per-phase limits
        "max_tools_ingest": int(os.environ.get("MAX_TOOLS_INGEST", "10")),
        "max_tools_calibration": int(os.environ.get("MAX_TOOLS_CALIBRATION", "10")),
        "max_tools_registration": int(os.environ.get("MAX_TOOLS_REGISTRATION", "10")),
        "max_tools_analysis": int(os.environ.get("MAX_TOOLS_ANALYSIS", "10")),
        "max_tools_stacking": int(os.environ.get("MAX_TOOLS_STACKING", "10")),
        "max_tools_linear": int(os.environ.get("MAX_TOOLS_LINEAR", "20")),
        "max_tools_stretch": int(os.environ.get("MAX_TOOLS_STRETCH", "25")),
        "max_tools_nonlinear": int(os.environ.get("MAX_TOOLS_NONLINEAR", "25")),
        "max_tools_export": int(os.environ.get("MAX_TOOLS_EXPORT", "5")),
        # Behavior
        "cleanup_previous_runs": os.environ.get("CLEANUP_PREVIOUS_RUNS", "true").lower() == "true",
        "prune_phase_analysis": os.environ.get("PRUNE_PHASE_ANALYSIS", "true").lower() == "true",
    }


# ── Graph helpers ────────────────────────────────────────────────────────────

def _make_thread_id(target: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = target.lower().replace(" ", "-")[:30]
    return f"run-{slug}-{ts}"


def _build_graph():
    """Build the LangGraph with SqliteSaver checkpointer."""
    store = make_memory_store()
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("astro_agent.graph.state", "ProcessingPhase")]
    )
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn, serde=serde)
    return build_graph(checkpointer=checkpointer, store=store)


# Module-level graph instance (built once on import)
_GRAPH = _build_graph()


# ── FITS → preview conversion ───────────────────────────────────────────────

def _convert_fits_to_preview(
    image_paths: list[str],
    working_dir: str,
    is_linear: bool = True,
) -> list[str]:
    """Convert FITS paths to displayable JPG previews. Non-FITS pass through."""
    from astro_agent.tools.utility.t22_generate_preview import generate_preview

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


# ── Stream chunk parsing ─────────────────────────────────────────────────────

def _parse_stream_chunks(
    chunk: dict,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    working_dir: str,
    is_linear: bool,
) -> dict | None:
    """
    Parse a single stream chunk (stream_mode="updates") and update the UI lists.

    Returns the interrupt payload if this chunk contains one, else None.
    """
    interrupt_payload = None

    if "__interrupt__" in chunk:
        interrupt_payload = chunk["__interrupt__"][0].value
        return interrupt_payload

    # Each chunk is {node_name: state_update_dict}
    for node_name, update in chunk.items():
        if node_name == "__interrupt__":
            continue

        messages = update.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Tool calls → activity log
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_str = json.dumps(tc.get("args", {}), indent=2, default=str)
                        activity_log.append({
                            "role": "assistant",
                            "content": f"**{tc['name']}**",
                            "metadata": {"title": f"Tool Call: {tc['name']}", "log": args_str},
                        })
                # Agent text → chat
                agent_text = text_content(msg.content)
                if agent_text.strip():
                    chat_messages.append({
                        "role": "assistant",
                        "content": agent_text,
                    })

            elif isinstance(msg, ToolMessage):
                result_text = text_content(msg.content)

                # present_images → populate gallery
                if msg.name == "present_images":
                    try:
                        result = json.loads(result_text)
                        if result.get("status") == "presented":
                            title = result.get("title", "")
                            description = result.get("description", "")
                            images = result.get("images", [])
                            # Convert FITS → JPG and populate gallery
                            paths = [img["path"] for img in images]
                            labels = [img.get("label", f"Image {i+1}") for i, img in enumerate(images)]
                            preview_paths = _convert_fits_to_preview(paths, working_dir, is_linear)
                            gallery_images.clear()
                            for path, label in zip(preview_paths, labels):
                                if Path(path).exists():
                                    gallery_images.append((path, label))
                            # Show description in chat if provided
                            if description:
                                chat_messages.append({
                                    "role": "assistant",
                                    "content": f"**{title}**\n\n{description}" if title else description,
                                })
                    except (json.JSONDecodeError, KeyError):
                        pass

                # All tool results → activity log (collapsible)
                display_text = result_text[:500] + "..." if len(result_text) > 500 else result_text
                activity_log.append({
                    "role": "assistant",
                    "content": f"**{msg.name}** result",
                    "metadata": {"title": f"Result: {msg.name}", "log": display_text},
                })

            elif isinstance(msg, HumanMessage):
                # HITL approval or feedback injected by the graph
                human_text = text_content(msg.content)
                if human_text and not human_text.startswith("Approved."):
                    # Don't echo auto-generated approval messages
                    pass

        # Phase changes
        new_phase = update.get("phase")
        if new_phase is not None:
            phase_name = new_phase.value if isinstance(new_phase, ProcessingPhase) else str(new_phase)
            chat_messages.append({
                "role": "assistant",
                "content": f"--- Phase: **{phase_name.upper()}** ---",
            })

    return interrupt_payload


# ── Core streaming handler ───────────────────────────────────────────────────

async def _stream_graph(
    graph,
    config: dict,
    stream_input,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    working_dir: str,
    is_linear: bool,
):
    """
    Core async streaming loop. Yields (chat, activity, gallery) tuples.
    Returns the interrupt payload if one fires, else None.
    """
    interrupt_payload = None

    async for chunk in graph.astream(stream_input, config=config, stream_mode="updates"):
        result = _parse_stream_chunks(
            chunk, chat_messages, activity_log, gallery_images,
            working_dir, is_linear,
        )
        if result is not None:
            interrupt_payload = result
            # Don't break — let the stream end naturally

        yield chat_messages, activity_log, gallery_images

    # Handle interrupt payload — update gallery with images
    if interrupt_payload is not None:
        interrupt_type = interrupt_payload.get("type", "unknown")
        title = interrupt_payload.get("title", "Review")
        agent_text = interrupt_payload.get("agent_text", "")
        images = interrupt_payload.get("images", [])

        if interrupt_type == "image_review" and images and working_dir:
            preview_paths = _convert_fits_to_preview(images, working_dir, is_linear)
            gallery_images.clear()
            for i, p in enumerate(preview_paths):
                if Path(p).exists():
                    gallery_images.append((p, f"Variant {i + 1}"))

        # Show HITL context in chat
        if interrupt_type == "agent_chat":
            if agent_text:
                chat_messages.append({
                    "role": "assistant",
                    "content": agent_text,
                })
        else:
            chat_messages.append({
                "role": "assistant",
                "content": f"**HITL: {title}**\n\n{agent_text}" if agent_text else f"**HITL: {title}** — Awaiting your review.",
            })

        yield chat_messages, activity_log, gallery_images


# ── Event handlers ───────────────────────────────────────────────────────────

async def start_session(
    dataset_path: str,
    target_name: str,
    bortle: int | None,
    sqm: float | None,
    remove_stars: bool | None,
    notes: str,
    # Equipment
    pixel_size: float | None,
    sensor_type: str | None,
    focal_length: float | None,
    # HITL
    autonomous_mode: bool,
    vlm_hitl: bool,
    vlm_present: bool,
    # Memory
    memory_mode: bool,
    # Model
    llm_model: str,
    llm_provider: str,
    llm_temp: float,
    # Safety limits
    recursion_limit: int,
    max_tools_phase: int,
    max_consecutive: int,
    max_nudges: int,
    # Per-phase limits
    phase_ingest: int,
    phase_calibration: int,
    phase_registration: int,
    phase_analysis: int,
    phase_stacking: int,
    phase_linear: int,
    phase_stretch: int,
    phase_nonlinear: int,
    phase_export: int,
    # Behavior
    cleanup_runs: bool,
    prune_analysis: bool,
    # Session state
    state: dict,
):
    """Start a new processing session."""
    if not dataset_path or not target_name:
        yield (
            [{"role": "assistant", "content": "Please provide both a dataset path and target name."}],
            [],
            [],
            state,
        )
        return

    # Apply settings to environment
    os.environ["LLM_MODEL"] = llm_model
    os.environ["LLM_PROVIDER"] = llm_provider
    os.environ["LLM_TEMPERATURE"] = str(llm_temp)
    os.environ["RECURSION_LIMIT"] = str(int(recursion_limit))
    os.environ["MAX_TOOLS_PER_PHASE"] = str(int(max_tools_phase))
    os.environ["MAX_CONSECUTIVE_SAME_TOOL"] = str(int(max_consecutive))
    os.environ["MAX_AUTONOMOUS_NUDGES"] = str(int(max_nudges))
    # Per-phase limits
    os.environ["MAX_TOOLS_INGEST"] = str(int(phase_ingest))
    os.environ["MAX_TOOLS_CALIBRATION"] = str(int(phase_calibration))
    os.environ["MAX_TOOLS_REGISTRATION"] = str(int(phase_registration))
    os.environ["MAX_TOOLS_ANALYSIS"] = str(int(phase_analysis))
    os.environ["MAX_TOOLS_STACKING"] = str(int(phase_stacking))
    os.environ["MAX_TOOLS_LINEAR"] = str(int(phase_linear))
    os.environ["MAX_TOOLS_STRETCH"] = str(int(phase_stretch))
    os.environ["MAX_TOOLS_NONLINEAR"] = str(int(phase_nonlinear))
    os.environ["MAX_TOOLS_EXPORT"] = str(int(phase_export))
    # Behavior
    os.environ["CLEANUP_PREVIOUS_RUNS"] = "true" if cleanup_runs else "false"
    os.environ["PRUNE_PHASE_ANALYSIS"] = "true" if prune_analysis else "false"

    # Set runtime mode flags
    set_autonomous(autonomous_mode)
    set_vlm_hitl(vlm_hitl)
    set_vlm_autonomous(vlm_present)
    set_memory_enabled(memory_mode)

    # Initialize long-term memory if enabled
    if memory_mode:
        try:
            from astro_agent.memory.embeddings import OllamaEmbedder
            from astro_agent.memory.store import MemoryStore
            from astro_agent.tools.utility.t33_memory_search import set_memory_store
            from astro_agent.graph.registry import register_memory_tool

            settings = load_settings()
            embedder = OllamaEmbedder(model=settings.memory_embedding_model)
            store = MemoryStore(db_path=settings.memory_db_path, embedder=embedder)
            # Share the db connection with the embedder for cache access
            embedder._db_conn = store._get_conn()
            set_memory_store(store)
            register_memory_tool()
            logger.info("Long-term memory enabled")
        except Exception as e:
            logger.warning(f"Long-term memory init failed (non-fatal): {e}")
            set_memory_enabled(False)

    # Build thread config
    thread_id = _make_thread_id(target_name)
    config = {"configurable": {"thread_id": thread_id}}
    if recursion_limit > 0:
        config["recursion_limit"] = recursion_limit

    # Ingest dataset
    chat_messages = [{"role": "assistant", "content": f"Starting session **{thread_id}**\n\nIngesting dataset from `{dataset_path}`..."}]
    activity_log: list[dict] = []
    gallery_images: list[tuple] = []

    yield chat_messages, activity_log, gallery_images, state

    try:
        ingest_result = ingest_dataset.invoke({
            "root_directory": dataset_path,
            "thread_id": thread_id,
        })
    except Exception as e:
        chat_messages.append({"role": "assistant", "content": f"Ingest failed: {e}"})
        yield chat_messages, activity_log, gallery_images, state
        return

    # Build session context
    session = SessionContext(
        target_name=target_name,
        bortle=bortle if bortle and bortle > 0 else None,
        sqm_reading=sqm if sqm and sqm > 0 else None,
        remove_stars=remove_stars,
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

    # Update session state
    state = {
        "thread_id": thread_id,
        "config": config,
        "working_dir": working_dir,
        "is_linear": True,
    }

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

    yield chat_messages, activity_log, gallery_images, state

    # Stream the graph
    async for chat_msgs, act_log, gal_imgs in _stream_graph(
        _GRAPH, config, initial_state,
        chat_messages, activity_log, gallery_images,
        working_dir, state.get("is_linear", True),
    ):
        yield chat_msgs, act_log, gal_imgs, state

    # Check if pipeline completed (no interrupt)
    # If we get here without an interrupt, the pipeline finished
    if not any("HITL" in m.get("content", "") and "Awaiting" in m.get("content", "") for m in chat_messages[-3:]):
        chat_messages.append({"role": "assistant", "content": "Pipeline complete."})
        yield chat_messages, activity_log, gallery_images, state


async def send_message(
    user_text: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    state: dict,
):
    """Handle user message during HITL — resume graph with feedback."""
    if not user_text.strip():
        yield chat_messages, activity_log, gallery_images, state, ""
        return

    config = state.get("config")
    working_dir = state.get("working_dir", "")
    is_linear = state.get("is_linear", True)

    if not config:
        chat_messages.append({"role": "assistant", "content": "No active session. Start a new session first."})
        yield chat_messages, activity_log, gallery_images, state, ""
        return

    # Add user message to chat
    chat_messages.append({"role": "user", "content": user_text})
    yield chat_messages, activity_log, gallery_images, state, ""

    # Resume graph with user feedback
    resume_value = user_text

    async for chat_msgs, act_log, gal_imgs in _stream_graph(
        _GRAPH, config, Command(resume=resume_value),
        chat_messages, activity_log, gallery_images,
        working_dir, is_linear,
    ):
        yield chat_msgs, act_log, gal_imgs, state, ""


async def approve_action(
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    state: dict,
):
    """Handle approve button — resume graph with APPROVE_SENTINEL."""
    config = state.get("config")
    working_dir = state.get("working_dir", "")
    is_linear = state.get("is_linear", True)

    if not config:
        chat_messages.append({"role": "assistant", "content": "No active session."})
        yield chat_messages, activity_log, gallery_images, state
        return

    chat_messages.append({"role": "user", "content": "Approved."})
    yield chat_messages, activity_log, gallery_images, state

    async for chat_msgs, act_log, gal_imgs in _stream_graph(
        _GRAPH, config, Command(resume=APPROVE_SENTINEL),
        chat_messages, activity_log, gallery_images,
        working_dir, is_linear,
    ):
        yield chat_msgs, act_log, gal_imgs, state


async def resume_session(
    resume_id: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    state: dict,
):
    """Resume a session from an existing checkpoint."""
    if not resume_id.strip():
        chat_messages.append({"role": "assistant", "content": "Please enter a thread ID to resume."})
        yield chat_messages, activity_log, gallery_images, state
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
    yield chat_messages, activity_log, gallery_images, state

    async for chat_msgs, act_log, gal_imgs in _stream_graph(
        _GRAPH, config, Command(resume=""),
        chat_messages, activity_log, gallery_images,
        state.get("working_dir", ""), state.get("is_linear", True),
    ):
        yield chat_msgs, act_log, gal_imgs, state


# ── Layout ───────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    hitl_defaults = _load_hitl_defaults()
    equip_defaults = _load_equipment_defaults()
    env_defaults = _load_env_defaults()

    hitl_tools = hitl_defaults.get("hitl", {})

    with gr.Blocks(title="AstroAgent") as app:
        # Session state
        session_state = gr.State({
            "thread_id": None,
            "config": None,
            "working_dir": "",
            "is_linear": True,
        })

        gr.Markdown("# AstroAgent")

        with gr.Tab("Processing"):
            with gr.Row():
                # ── Left column: Chat ──────────────────────────────
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="AstroAgent",
                        height=600,
                        buttons=["copy", "copy_all"],
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type feedback, ask questions, or discuss the result...",
                            scale=4,
                            lines=3,
                            max_lines=6,
                            show_label=False,
                            container=False,
                        )
                        send_btn = gr.Button("Send", scale=1)
                        approve_btn = gr.Button("Approve", variant="primary", scale=1)

                # ── Right column: Gallery + Activity log ───────────
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Image Review",
                        columns=2,
                        height=350,
                        allow_preview=True,
                        buttons=["download"],
                    )
                    activity = gr.Chatbot(
                        label="Activity Log",
                        height=250,
                        buttons=[],
                    )

            # ── Session controls ───────────────────────────────────
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
                            minimum=1,
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

        with gr.Tab("Session Notes"):
            gr.Markdown(
                "Free-text context injected into the agent's initial prompt. "
                "Use for anything that affects processing but isn't captured elsewhere. "
                "Leave empty if not needed."
            )
            notes_input = gr.Textbox(
                label="Notes",
                placeholder="Shot with Optolong L-eNhance duoband filter\nVery poor seeing — FWHM likely > 4px\n10-min subs, ASI2600, gain 100\nPrioritise faint outer nebulosity over star quality",
                lines=10,
                max_lines=20,
            )

        with gr.Tab("Equipment"):
            gr.Markdown(
                "Override values that can't be read from file metadata. "
                "Leave empty to auto-detect from FITS headers or EXIF.\n\n"
                "**DSLR/mirrorless** (RAF, CR2, NEF, ARW): pixel size is never in EXIF. "
                "X-Trans sensors (Fuji) need sensor_type for 3-pass demosaic.\n\n"
                "**Dedicated astro cameras** (ZWO, QHY FITS): all values are in FITS headers. "
                "Nothing needed here."
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
                info="Required for X-Trans (Fuji) to enable 3-pass demosaic. Leave empty for auto-detect.",
            )
            focal_length = gr.Number(
                label="Focal length (mm)",
                value=equip_defaults.get("optics", {}).get("focal_length_mm"),
                info="EXIF has nominal value; plate-solve-measured is more accurate",
            )

        with gr.Tab("HITL Config"):
            gr.Markdown("Control which pipeline steps pause for human review.")
            autonomous_mode = gr.Checkbox(
                label="Autonomous mode (skip all HITL)",
                value=hitl_defaults.get("autonomous", False),
            )
            gr.Markdown("### VLM (Visual Language Model)")
            gr.Markdown(
                "When enabled, preview images are injected as base64 into the agent's "
                "context so it can visually reason about results. Images are stripped "
                "from context after use to manage the context window."
            )
            vlm_hitl = gr.Checkbox(
                label="VLM during HITL (agent sees all images during human review — HITL-triggered and tool-triggered)",
                value=False,
            )
            vlm_present = gr.Checkbox(
                label="VLM autonomous (agent can visually inspect images outside of HITL)",
                value=False,
            )
            gr.Markdown("### Long-Term Memory")
            gr.Markdown(
                "When enabled, the agent can search past processing sessions for relevant "
                "experience (what worked, what failed, user preferences). Memories are "
                "extracted from HITL conversations after approval. Keep OFF during debugging "
                "and test runs — only enable when agent quality is stable."
            )
            memory_enabled = gr.Checkbox(
                label="Long-term memory (agent learns from past HITL sessions)",
                value=os.environ.get("MEMORY_ENABLED", "false").lower() == "true",
            )
            gr.Markdown("### Per-tool HITL checkpoints")
            gr.Markdown("**Preprocessing**")
            hitl_t02 = gr.Checkbox(label="T02 Master Frame Diagnostics", value=hitl_tools.get("T02_masters", {}).get("enabled", False))
            hitl_t06 = gr.Checkbox(label="T06 Frame Selection", value=hitl_tools.get("T06_select", {}).get("enabled", False))
            hitl_t07 = gr.Checkbox(label="T07 Stack Results", value=hitl_tools.get("T07_stack", {}).get("enabled", False))
            gr.Markdown("**Linear**")
            hitl_t09 = gr.Checkbox(label="T09 Gradient Removal", value=hitl_tools.get("T09_gradient", {}).get("enabled", True))
            hitl_t10 = gr.Checkbox(label="T10 Color Calibration", value=hitl_tools.get("T10_color", {}).get("enabled", False))
            hitl_t12 = gr.Checkbox(label="T12 Noise Reduction", value=hitl_tools.get("T12_denoise", {}).get("enabled", False))
            hitl_t13 = gr.Checkbox(label="T13 Deconvolution", value=hitl_tools.get("T13_decon", {}).get("enabled", False))
            gr.Markdown("**Stretch**")
            hitl_t14 = gr.Checkbox(label="T14 Stretch", value=hitl_tools.get("T14_stretch", {}).get("enabled", True))
            gr.Markdown("**Non-linear**")
            hitl_t15 = gr.Checkbox(label="T15 Star Removal", value=hitl_tools.get("T15_star_removal", {}).get("enabled", False))
            hitl_t16 = gr.Checkbox(label="T16 Curves", value=hitl_tools.get("T16_curves", {}).get("enabled", True))
            hitl_t17 = gr.Checkbox(label="T17 Local Contrast", value=hitl_tools.get("T17_local_contrast", {}).get("enabled", False))
            hitl_t18 = gr.Checkbox(label="T18 Saturation", value=hitl_tools.get("T18_saturation", {}).get("enabled", False))
            hitl_t19 = gr.Checkbox(label="T19 Star Restoration", value=hitl_tools.get("T19_star_restoration", {}).get("enabled", True))
            hitl_t27 = gr.Checkbox(label="T27 Multiscale Sharpening", value=hitl_tools.get("T27_multiscale", {}).get("enabled", False))

        with gr.Tab("Model & Limits"):
            gr.Markdown("### Model")
            llm_model = gr.Dropdown(
                label="LLM Model",
                choices=["moonshotai/Kimi-K2.5", "deepseek-ai/DeepSeek-V3", "claude-sonnet-4-6"],
                value=env_defaults["llm_model"],
                allow_custom_value=True,
            )
            llm_provider = gr.Dropdown(
                label="Provider",
                choices=["together", "anthropic", "openai"],
                value=env_defaults["llm_provider"],
            )
            llm_temp = gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1,
                step=0.1,
                value=env_defaults["llm_temperature"],
            )

            gr.Markdown("### Safety Limits")
            with gr.Row():
                recursion_limit = gr.Number(
                    label="Recursion limit",
                    value=env_defaults["recursion_limit"],
                    precision=0,
                    info="Max LangGraph node transitions per stream() call. ~4 per tool call.",
                )
                max_tools_phase = gr.Number(
                    label="Max tools per phase (default)",
                    value=env_defaults["max_tools_per_phase"],
                    precision=0,
                    info="Global default. Per-phase overrides below. 0 = unlimited.",
                )
            with gr.Row():
                max_consecutive = gr.Number(
                    label="Max consecutive same tool",
                    value=env_defaults["max_consecutive_same_tool"],
                    precision=0,
                    info="Hard fail if agent calls the same tool N times in a row. 0 = disabled.",
                )
                max_nudges = gr.Number(
                    label="Max autonomous nudges",
                    value=env_defaults["max_autonomous_nudges"],
                    precision=0,
                    info="Max consecutive text-only responses before failing.",
                )

            gr.Markdown("### Per-Phase Tool Limits")
            gr.Markdown("Override the global default for specific phases. 0 = use global default.")
            with gr.Row():
                phase_ingest = gr.Number(label="Ingest", value=env_defaults["max_tools_ingest"], precision=0)
                phase_calibration = gr.Number(label="Calibration", value=env_defaults["max_tools_calibration"], precision=0)
                phase_registration = gr.Number(label="Registration", value=env_defaults["max_tools_registration"], precision=0)
            with gr.Row():
                phase_analysis = gr.Number(label="Analysis", value=env_defaults["max_tools_analysis"], precision=0)
                phase_stacking = gr.Number(label="Stacking", value=env_defaults["max_tools_stacking"], precision=0)
                phase_linear = gr.Number(label="Linear", value=env_defaults["max_tools_linear"], precision=0)
            with gr.Row():
                phase_stretch = gr.Number(label="Stretch", value=env_defaults["max_tools_stretch"], precision=0)
                phase_nonlinear = gr.Number(label="Non-linear", value=env_defaults["max_tools_nonlinear"], precision=0)
                phase_export = gr.Number(label="Export", value=env_defaults["max_tools_export"], precision=0)

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

        # ── Wire events ──────────────────────────────────────────────────

        # Helper to convert remove_stars dropdown to bool | None
        def _parse_remove_stars(val: str) -> bool | None:
            if val == "yes":
                return True
            elif val == "no":
                return False
            return None

        # Start session
        start_btn.click(
            fn=lambda *args: start_session(
                dataset_path=args[0],
                target_name=args[1],
                bortle=args[2],
                sqm=args[3],
                remove_stars=_parse_remove_stars(args[4]),
                notes=args[5],
                pixel_size=args[6],
                sensor_type=args[7] if args[7] else None,
                focal_length=args[8],
                autonomous_mode=args[9],
                vlm_hitl=args[10],
                vlm_present=args[11],
                memory_mode=args[12],
                llm_model=args[13],
                llm_provider=args[14],
                llm_temp=args[15],
                recursion_limit=int(args[16]),
                max_tools_phase=int(args[17]),
                max_consecutive=int(args[18]),
                max_nudges=int(args[19]),
                phase_ingest=int(args[20]),
                phase_calibration=int(args[21]),
                phase_registration=int(args[22]),
                phase_analysis=int(args[23]),
                phase_stacking=int(args[24]),
                phase_linear=int(args[25]),
                phase_stretch=int(args[26]),
                phase_nonlinear=int(args[27]),
                phase_export=int(args[28]),
                cleanup_runs=args[29],
                prune_analysis=args[30],
                state=args[31],
            ),
            inputs=[
                dataset_path, target_name, bortle_input, sqm_input,
                remove_stars_input, notes_input,
                pixel_size, sensor_type, focal_length,
                autonomous_mode, vlm_hitl, vlm_present, memory_enabled,
                llm_model, llm_provider, llm_temp,
                recursion_limit, max_tools_phase,
                max_consecutive, max_nudges,
                phase_ingest, phase_calibration, phase_registration,
                phase_analysis, phase_stacking, phase_linear,
                phase_stretch, phase_nonlinear, phase_export,
                cleanup_runs, prune_analysis,
                session_state,
            ],
            outputs=[chatbot, activity, gallery, session_state],
        )

        # Send message
        send_btn.click(
            fn=send_message,
            inputs=[msg_input, chatbot, activity, gallery, session_state],
            outputs=[chatbot, activity, gallery, session_state, msg_input],
        )
        msg_input.submit(
            fn=send_message,
            inputs=[msg_input, chatbot, activity, gallery, session_state],
            outputs=[chatbot, activity, gallery, session_state, msg_input],
        )

        # Approve
        approve_btn.click(
            fn=approve_action,
            inputs=[chatbot, activity, gallery, session_state],
            outputs=[chatbot, activity, gallery, session_state],
        )

        # Resume
        resume_btn.click(
            fn=lambda rid, chat, act, gal, st: resume_session(rid, chat, act, gal, st),
            inputs=[resume_id, chatbot, activity, gallery, session_state],
            outputs=[chatbot, activity, gallery, session_state],
        )

    return app


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = build_app()
    app.launch(theme=gr.themes.Soft(primary_hue="blue"))


if __name__ == "__main__":
    main()
