"""
AstroAgent Streamlit App

Usage:
    streamlit run astro_agent/app.py

Architecture:
    The LangGraph pipeline runs in a background thread. A queue.Queue bridges
    the thread and the Streamlit UI. The main thread polls the queue on each
    rerun and calls st.rerun() while the graph is active. HITL interrupts pause
    the thread; the user's response resumes it via a second input queue.

    Sessions are persisted to ~/.astroagent/sessions.json so thread IDs survive
    Streamlit restarts. SQLite checkpointing means any crash or sleep can be
    resumed by entering the thread ID in the Resume section.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from astro_agent.config import check_dependencies, load_settings
from astro_agent.graph.graph import build_graph
from astro_agent.graph.hitl import APPROVE_SENTINEL
from astro_agent.graph.memory import make_memory_store
from astro_agent.graph.state import SessionContext, build_initial_message, make_empty_state
from astro_agent.tools.preprocess.t01_ingest import ingest_dataset

# ── Paths ──────────────────────────────────────────────────────────────────────

_ASTROAGENT_DIR = Path.home() / ".astroagent"
_SESSIONS_FILE = _ASTROAGENT_DIR / "sessions.json"
_DB_PATH = str(_ASTROAGENT_DIR / "checkpoints.db")


# ── Session persistence ────────────────────────────────────────────────────────


def _load_sessions() -> list[dict]:
    if _SESSIONS_FILE.exists():
        try:
            return json.loads(_SESSIONS_FILE.read_text())
        except Exception:
            return []
    return []


def _save_session(thread_id: str, target: str) -> None:
    _ASTROAGENT_DIR.mkdir(exist_ok=True)
    sessions = _load_sessions()
    sessions = [s for s in sessions if s["thread_id"] != thread_id]
    sessions.insert(0, {
        "thread_id": thread_id,
        "target": target,
        "started": datetime.now().isoformat()[:19],
    })
    _SESSIONS_FILE.write_text(json.dumps(sessions[:20], indent=2))


# ── Logging ───────────────────────────────────────────────────────────────────


def _configure_logging(thread_id: str) -> None:
    """
    Configure root logger once per Streamlit process.

    Writes to both the terminal (where `streamlit run` is invoked) and a
    per-session file at ~/.astroagent/logs/<thread_id>.log.

    The `if root.handlers` guard prevents duplicate handlers — Streamlit
    re-executes the script on every UI interaction.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    log_dir = _ASTROAGENT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{thread_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )


# ── Graph initialisation (once per Streamlit session) ─────────────────────────


def _init_graph() -> None:
    if "graph" not in st.session_state:
        settings = load_settings()
        check_dependencies(settings)
        _ASTROAGENT_DIR.mkdir(exist_ok=True)
        store = make_memory_store()
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("astro_agent.graph.state", "ProcessingPhase")])
        checkpointer = SqliteSaver(sqlite3.connect(_DB_PATH, check_same_thread=False), serde=serde)
        st.session_state["graph"] = build_graph(checkpointer=checkpointer, store=store)


# ── Thread ID helper ───────────────────────────────────────────────────────────


def _make_thread_id(target: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = target.lower().replace(" ", "-")[:30]
    return f"run-{slug}-{ts}"


# ── Background graph thread ────────────────────────────────────────────────────


def _graph_thread(
    graph: Any,
    config: dict,
    stream_input: Any,
    event_queue: queue.Queue,
    input_queue: queue.Queue,
) -> None:
    """
    Stream the graph in a background thread, pushing typed events to event_queue.
    Blocks on input_queue when a HITL interrupt fires.
    """
    current_input = stream_input

    while True:
        interrupted = False
        try:
            for chunk in graph.stream(current_input, config=config, stream_mode="updates"):

                if "phase_advance" in chunk:
                    update = chunk["phase_advance"]
                    phase = update.get("phase")
                    if phase is not None:
                        val = phase.value if hasattr(phase, "value") else str(phase)
                        event_queue.put({"type": "phase_change", "phase": val})

                elif "agent" in chunk:
                    msgs = chunk["agent"].get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                event_queue.put({
                                    "type": "tool_call",
                                    "name": tc["name"],
                                    "args": tc.get("args", {}),
                                })
                        elif getattr(msg, "content", None):
                            event_queue.put({
                                "type": "agent_text",
                                "content": msg.content,
                            })

                elif "action" in chunk:
                    msgs = chunk["action"].get("messages", [])
                    for msg in msgs:
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        is_error = (
                            getattr(msg, "status", None) == "error"
                            or content.strip().startswith("Error")
                        )
                        event_queue.put({
                            "type": "tool_error" if is_error else "tool_result",
                            "name": getattr(msg, "name", "unknown"),
                            "content": content,
                        })

                elif "__interrupt__" in chunk:
                    payload = chunk["__interrupt__"][0].value
                    event_queue.put({"type": "interrupt", "payload": payload})
                    interrupted = True
                    break

        except Exception as exc:
            event_queue.put({"type": "error", "content": str(exc)})
            return

        if not interrupted:
            event_queue.put({"type": "done"})
            return

        # Wait for human input
        response = input_queue.get()
        current_input = Command(resume=response)


# ── Session launchers ──────────────────────────────────────────────────────────


def _launch_thread(stream_input: Any, config: dict, thread_id: str) -> None:
    """Start the background graph thread and initialise session state."""
    _configure_logging(thread_id)
    eq: queue.Queue = queue.Queue()
    iq: queue.Queue = queue.Queue()

    st.session_state.update({
        "thread_id": thread_id,
        "config": config,
        "event_queue": eq,
        "input_queue": iq,
        "events": st.session_state.get("events", []),  # preserve on HITL resume
        "pending_interrupt": None,
        "running": True,
    })

    t = threading.Thread(
        target=_graph_thread,
        args=(st.session_state["graph"], config, stream_input, eq, iq),
        daemon=True,
    )
    st.session_state["graph_thread"] = t
    t.start()


def _start_new_session(
    directory: str,
    target: str,
    bortle: int,
    sqm: float | None,
    remove_stars: bool | None,
    notes: str | None,
) -> None:
    _init_graph()

    thread_id = _make_thread_id(target)
    result = ingest_dataset.invoke({"root_directory": directory, "thread_id": thread_id})
    dataset = result["dataset"]

    st.session_state["ingest_summary"] = result.get("summary", {})
    st.session_state["ingest_warnings"] = result.get("warnings", [])

    session: SessionContext = {
        "target_name": target,
        "bortle": bortle,
        "sqm_reading": sqm,
        "remove_stars": remove_stars,
        "notes": notes or None,
    }

    initial_state = make_empty_state(dataset=dataset, session=session)
    initial_state["messages"] = [
        HumanMessage(content=build_initial_message(
            dataset=dataset,
            session=session,
            ingest_summary=result.get("summary", {}),
        ))
    ]
    config = {"configurable": {"thread_id": thread_id}}
    _save_session(thread_id, target)

    st.session_state["events"] = []  # fresh event log for new session
    _launch_thread(initial_state, config, thread_id)


def _resume_session(thread_id: str) -> None:
    _init_graph()
    config = {"configurable": {"thread_id": thread_id}}
    graph = st.session_state["graph"]

    # Check for a pending interrupt in the checkpointer
    state = graph.get_state(config)
    pending: list = []
    for task in (state.tasks or []):
        if hasattr(task, "interrupts") and task.interrupts:
            pending.extend(task.interrupts)

    _configure_logging(thread_id)
    st.session_state["events"] = []  # fresh view for resume
    st.session_state["thread_id"] = thread_id
    st.session_state["config"] = config
    st.session_state["running"] = False

    if pending:
        # Surface the pending interrupt immediately; thread starts on user response
        payload = pending[0].value
        st.session_state["pending_interrupt"] = payload
        # We still need queues ready for when the user responds
        st.session_state["event_queue"] = queue.Queue()
        st.session_state["input_queue"] = queue.Queue()
        st.session_state["events"].append({"type": "interrupt", "payload": payload})
    else:
        # No pending interrupt — resume streaming from last checkpoint
        st.session_state["pending_interrupt"] = None
        _launch_thread(None, config, thread_id)


# ── Key metric extraction ──────────────────────────────────────────────────────

_METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"background_flatness['\"]?\s*:\s*([\d.]+)", "flatness"),
    (r"flatness_score['\"]?\s*:\s*([\d.]+)", "flatness"),
    (r"snr_estimate['\"]?\s*:\s*([\d.]+)", "SNR"),
    (r"gradient_magnitude['\"]?\s*:\s*([\d.]+)", "gradient"),
    (r"noise_after['\"]?\s*:\s*([\d.]+)", "noise_after"),
    (r"noise_before['\"]?\s*:\s*([\d.]+)", "noise_before"),
    (r"fwhm_median['\"]?\s*:\s*([\d.]+)", "FWHM"),
    (r"registered_count['\"]?\s*:\s*(\d+)", "registered"),
    (r"selected_count['\"]?\s*:\s*(\d+)", "selected"),
    (r"lights_count['\"]?\s*:\s*(\d+)", "lights"),
    (r"master_path['\"]?\s*:\s*['\"]([^'\"]+)['\"]", "master"),
    (r"output_path['\"]?\s*:\s*['\"]([^'\"]+)['\"]", "output"),
]


def _key_metric(content: str) -> str:
    for pattern, label in _METRIC_PATTERNS:
        m = re.search(pattern, content)
        if m:
            val = m.group(1)
            # Truncate long paths
            if "/" in val:
                val = Path(val).name
            return f"{label}: {val}"
    return ""


# ── Event rendering ────────────────────────────────────────────────────────────

_PHASE_LABELS = {
    "ingest": "INGEST",
    "calibration": "CALIBRATION",
    "registration": "REGISTRATION",
    "analysis": "ANALYSIS",
    "stacking": "STACKING",
    "linear": "LINEAR PROCESSING",
    "stretch": "STRETCH",
    "nonlinear": "NON-LINEAR PROCESSING",
    "export": "EXPORT",
    "review": "REVIEW",
    "complete": "COMPLETE",
}


def _render_events(events: list[dict]) -> None:
    for event in events:
        etype = event["type"]

        if etype == "phase_change":
            label = _PHASE_LABELS.get(event["phase"].lower(), event["phase"].upper())
            st.markdown(f"---\n#### ── {label} ──")

        elif etype == "agent_text":
            with st.chat_message("assistant"):
                st.write(event["content"])

        elif etype == "tool_call":
            args = event.get("args", {})
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            with st.expander(f"▶ {event['name']}({args_str})", expanded=False):
                st.json(args)

        elif etype == "tool_result":
            content = event.get("content", "")
            metric = _key_metric(content)
            label = f"✓ {event['name']}" + (f"  —  {metric}" if metric else "")
            with st.expander(label, expanded=False):
                st.code(content, language="json")

        elif etype == "tool_error":
            with st.expander(f"✗ {event['name']}", expanded=True):
                st.error(event.get("content", "Unknown error"))

        elif etype == "human_feedback":
            with st.chat_message("human"):
                st.write(event["content"])

        elif etype == "error":
            st.error(f"Graph error: {event['content']}")

        elif etype == "done":
            st.success("Pipeline complete.")


# ── HITL panel ─────────────────────────────────────────────────────────────────


def _render_hitl_panel(payload: dict) -> None:
    st.divider()

    ptype = payload.get("type", "data_review")
    images = payload.get("images", [])

    if ptype == "agent_chat":
        # Agent sent a text message — display it and let the human respond
        st.subheader(f"Agent message ({payload.get('phase', '')} phase)")
        agent_text = payload.get("agent_text", "")
        if agent_text:
            with st.chat_message("assistant"):
                st.write(agent_text)
    else:
        st.subheader(f"Review required: {payload.get('title', 'Human Review')}")

        if ptype == "image_review" and images:
            cols = st.columns(min(len(images), 2))
            for i, img_path in enumerate(images):
                p = Path(img_path)
                with cols[i % len(cols)]:
                    if p.exists():
                        st.image(str(p), caption=p.name, use_container_width=True)
                    else:
                        st.warning(f"Image not found: {img_path}")

        elif ptype == "data_review":
            # Show last tool result from context for data review
            context = payload.get("context", [])
            for msg in reversed(context):
                if hasattr(msg, "name") and hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, list):
                        content = content[0].get("text", str(content)) if content else ""
                    with st.expander(f"Result: {msg.name}", expanded=True):
                        st.code(content, language="json")
                    break

    st.divider()
    feedback = st.text_area(
        "Leave empty and click Approve to continue, or type feedback/questions:",
        key="hitl_feedback",
        height=80,
    )

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("Approve ✓", type="primary"):
            _submit_hitl_response(APPROVE_SENTINEL)
    with col2:
        if st.button("Send") and feedback.strip():
            _submit_hitl_response(feedback.strip(), record_feedback=True)


def _submit_hitl_response(response: str, record_feedback: bool = False) -> None:
    if record_feedback:
        st.session_state["events"].append({
            "type": "human_feedback",
            "content": response,
        })

    if "input_queue" not in st.session_state:
        # Resume case: thread hasn't started yet — launch it now with resume Command
        thread_id = st.session_state["thread_id"]
        config = st.session_state["config"]
        eq = st.session_state["event_queue"]
        iq = st.session_state["input_queue"] = queue.Queue()
        # Re-init with the correct input queue in place
        st.session_state["event_queue"] = queue.Queue()
        _launch_thread(Command(resume=response), config, thread_id)
    else:
        st.session_state["input_queue"].put(response)
        st.session_state["pending_interrupt"] = None
        st.session_state["running"] = True

    st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="AstroAgent",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("AstroAgent")
    st.caption("Autonomous astrophotography post-processing")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("New Session")

        directory = st.text_input(
            "Dataset directory",
            placeholder="/path/to/dataset",
            help="Root directory containing lights/, darks/, flats/, bias/ subfolders.",
        )
        target = st.text_input("Target name", placeholder="M42, NGC 7000, etc.")
        bortle = st.number_input("Bortle scale", min_value=1, max_value=9, value=5, step=1)
        sqm_val = st.number_input(
            "SQM reading (optional)", value=0.0, step=0.1,
            help="Leave 0 to omit. Enter SQM-L mag/arcsec² if you have it.",
        )
        sqm = sqm_val if sqm_val > 0 else None
        rs_opt = st.selectbox(
            "Star removal",
            ["ask via HITL", "yes", "no"],
            help="Whether to run star_removal + star_restoration.",
        )
        remove_stars = None if rs_opt == "ask via HITL" else (rs_opt == "yes")
        notes = st.text_area("Session notes (optional)", height=60, placeholder="e.g. L-eNhance filter, gain 100, poor seeing")

        start_disabled = not (directory.strip() and target.strip())
        if st.button("Start →", type="primary", disabled=start_disabled):
            with st.spinner("Ingesting dataset..."):
                try:
                    _start_new_session(directory.strip(), target.strip(), bortle, sqm, remove_stars, notes.strip() or None)
                except Exception as e:
                    st.error(f"Failed to start: {e}")
                    st.stop()
            st.rerun()

        # Ingest summary feedback
        if "ingest_summary" in st.session_state:
            s = st.session_state["ingest_summary"]
            st.caption(
                f"Lights: **{s.get('lights_count', 0)}**  "
                f"Darks: **{s.get('darks_count', 0)}**  "
                f"Flats: **{s.get('flats_count', 0)}**  "
                f"Biases: **{s.get('biases_count', 0)}**"
            )
            for w in st.session_state.get("ingest_warnings", []):
                st.warning(w, icon="⚠️")

        st.divider()
        st.header("Resume Session")

        sessions = _load_sessions()
        if sessions:
            options = [f"{s['target']} ({s['started'][:10]})" for s in sessions[:8]]
            idx = st.selectbox("Recent sessions", range(len(options)), format_func=lambda i: options[i])
            resume_thread = sessions[idx]["thread_id"]
            st.caption(f"`{resume_thread}`")
        else:
            resume_thread = st.text_input("Thread ID", placeholder="run-m42-20260311-120000")

        if st.button("Resume →", disabled=not resume_thread):
            with st.spinner("Loading checkpoint..."):
                try:
                    _resume_session(resume_thread)
                except Exception as e:
                    st.error(f"Failed to resume: {e}")
                    st.stop()
            st.rerun()

        # Active thread info
        if "thread_id" in st.session_state:
            st.divider()
            st.caption(f"Active thread:")
            st.code(st.session_state["thread_id"], language=None)

    # ── Main area ──────────────────────────────────────────────────────────────

    if "events" not in st.session_state:
        st.info(
            "Enter your dataset directory and target name in the sidebar, then click **Start →**.\n\n"
            "Or resume a previous session with the **Resume** section."
        )
        return

    # Drain event queue
    if st.session_state.get("running") and "event_queue" in st.session_state:
        eq: queue.Queue = st.session_state["event_queue"]
        new_events: list[dict] = []
        try:
            while True:
                new_events.append(eq.get_nowait())
        except queue.Empty:
            pass

        if new_events:
            st.session_state["events"].extend(new_events)
            for ev in new_events:
                if ev["type"] == "interrupt":
                    st.session_state["pending_interrupt"] = ev["payload"]
                    st.session_state["running"] = False
                    break
                if ev["type"] in ("done", "error"):
                    st.session_state["running"] = False
                    break

    # Render accumulated event log
    _render_events(st.session_state["events"])

    # Running indicator
    if st.session_state.get("running"):
        with st.spinner("Agent working…"):
            time.sleep(0.5)
        st.rerun()

    # HITL panel (shown when interrupted)
    if st.session_state.get("pending_interrupt"):
        _render_hitl_panel(st.session_state["pending_interrupt"])


if __name__ == "__main__":
    main()
