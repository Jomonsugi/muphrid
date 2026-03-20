"""
AstroAgent Streamlit App

Usage:
    streamlit run astro_agent/app.py

Architecture:
    The LangGraph pipeline runs in a background daemon thread. An event queue
    bridges graph events to the Streamlit UI. The main thread polls the queue
    every 0.5s via st.rerun() and renders events as they arrive.

    When an HITL interrupt fires, the thread blocks on an input queue waiting
    for the user's response. The user types feedback or clicks Approve, the
    response goes on the input queue, and the thread resumes. One code path
    for all responses — no bifurcation.

    Sessions are persisted to ~/.astroagent/sessions.json. SQLite checkpointing
    means any crash can be resumed.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from astro_agent.config import check_dependencies, load_settings
from astro_agent.graph.content import text_content
from astro_agent.graph.graph import build_graph
from astro_agent.graph.hitl import APPROVE_SENTINEL, images_from_tool
from astro_agent.graph.memory import make_memory_store
from astro_agent.graph.state import SessionContext, build_initial_message, make_empty_state
from astro_agent.tools.preprocess.t01_ingest import ingest_dataset

# ── Paths ──────────────────────────────────────────────────────────────────────

_ASTROAGENT_DIR = Path.home() / ".astroagent"
_SESSIONS_FILE = _ASTROAGENT_DIR / "sessions.json"
_DB_PATH = str(_ASTROAGENT_DIR / "checkpoints.db")


# ── Session persistence ────────────────────────────────────────────────────────


def _load_sessions() -> list[dict]:
    """Load sessions, hiding any whose run data no longer exists on disk."""
    if not _SESSIONS_FILE.exists():
        return []
    try:
        sessions = json.loads(_SESSIONS_FILE.read_text())
    except Exception:
        return []
    return [s for s in sessions if Path(s.get("working_dir", "")).is_dir()]


def _save_session(thread_id: str, target: str, working_dir: str) -> None:
    _ASTROAGENT_DIR.mkdir(exist_ok=True)
    try:
        all_sessions = json.loads(_SESSIONS_FILE.read_text()) if _SESSIONS_FILE.exists() else []
    except Exception:
        all_sessions = []
    all_sessions = [s for s in all_sessions if s["thread_id"] != thread_id]
    all_sessions.insert(0, {
        "thread_id": thread_id,
        "target": target,
        "started": datetime.now().isoformat()[:19],
        "working_dir": working_dir,
    })
    _SESSIONS_FILE.write_text(json.dumps(all_sessions[:20], indent=2))


def _prune_stale_checkpoints() -> None:
    """Remove checkpoint rows for sessions whose run data no longer exists."""
    if not _SESSIONS_FILE.exists() or not Path(_DB_PATH).exists():
        return
    try:
        all_sessions = json.loads(_SESSIONS_FILE.read_text())
    except Exception:
        return
    stale_ids = [
        s["thread_id"] for s in all_sessions
        if not Path(s.get("working_dir", "")).is_dir()
    ]
    if not stale_ids:
        return
    try:
        conn = sqlite3.connect(_DB_PATH)
        for tid in stale_ids:
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (tid,))
        conn.commit()
        conn.close()
        logging.getLogger(__name__).info(
            f"Pruned checkpoints for {len(stale_ids)} stale session(s)"
        )
    except Exception:
        pass


# ── Logging ───────────────────────────────────────────────────────────────────


def _configure_logging(thread_id: str) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
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


# ── Graph initialisation ────────────────────────────────────────────────────


def _init_graph() -> None:
    if "graph" not in st.session_state:
        settings = load_settings()
        check_dependencies(settings)
        _ASTROAGENT_DIR.mkdir(exist_ok=True)
        store = make_memory_store()
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[("astro_agent.graph.state", "ProcessingPhase")]
        )
        checkpointer = SqliteSaver(
            sqlite3.connect(_DB_PATH, check_same_thread=False), serde=serde
        )
        st.session_state["graph"] = build_graph(checkpointer=checkpointer, store=store)


def _make_thread_id(target: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = target.lower().replace(" ", "-")[:30]
    return f"run-{slug}-{ts}"


# ── Chunk → event conversion ─────────────────────────────────────────────────


def _chunk_to_events(chunk: dict) -> list[dict]:
    """Convert a graph stream chunk to a list of UI events."""
    events: list[dict] = []

    if "phase_advance" in chunk:
        update = chunk["phase_advance"]
        phase = update.get("phase")
        if phase is not None:
            val = phase.value if hasattr(phase, "value") else str(phase)
            events.append({"type": "phase_change", "phase": val})

    elif "agent" in chunk:
        msgs = chunk["agent"].get("messages", [])
        for msg in msgs:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    events.append({
                        "type": "tool_call",
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                    })
            elif getattr(msg, "content", None):
                events.append({
                    "type": "agent_text",
                    "content": text_content(msg.content),
                })

    elif "action" in chunk:
        msgs = chunk["action"].get("messages", [])
        for msg in msgs:
            content = text_content(msg.content)
            is_error = (
                getattr(msg, "status", None) == "error"
                or content.strip().startswith("Error")
            )
            events.append({
                "type": "tool_error" if is_error else "tool_result",
                "name": getattr(msg, "name", "unknown"),
                "content": content,
            })

    return events


# ── Background graph thread ──────────────────────────────────────────────────


def _graph_thread(
    graph: Any,
    config: dict,
    stream_input: Any,
    event_q: queue.Queue,
    input_q: queue.Queue,
) -> None:
    """
    Stream the graph in a background thread. Push events to event_q.
    Block on input_q when an HITL interrupt fires.
    """
    current_input = stream_input

    while True:
        interrupted = False
        stream = graph.stream(current_input, config=config, stream_mode="updates")
        try:
            for chunk in stream:
                # Push UI events
                for ev in _chunk_to_events(chunk):
                    event_q.put(ev)

                # Interrupt — push payload and wait for human
                if "__interrupt__" in chunk:
                    payload = chunk["__interrupt__"][0].value
                    # Re-derive images from current state
                    payload = _refresh_interrupt_payload(payload, graph, config)
                    event_q.put({"type": "interrupt", "payload": payload})
                    interrupted = True
                    break

        except Exception as exc:
            event_q.put({"type": "error", "content": str(exc)})
            return
        finally:
            stream.close()

        if not interrupted:
            event_q.put({"type": "done"})
            return

        # Wait for human response
        response = input_q.get()
        current_input = Command(resume=response)


def _refresh_interrupt_payload(payload: dict, graph, config: dict) -> dict:
    """Re-derive display data (images, previews) from current graph state."""
    ptype = payload.get("type", "")
    tool_name = payload.get("tool_name", "")

    if ptype == "image_review" and tool_name:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        working_dir = state.values.get("dataset", {}).get("working_dir", "")
        raw_paths = images_from_tool(messages, tool_name)

        if raw_paths and working_dir:
            from astro_agent.tools.utility.t22_generate_preview import generate_preview
            preview_paths = []
            for img in raw_paths:
                p = Path(img)
                if p.suffix.lower() in (".fit", ".fits", ".fts"):
                    preview_dir = Path(working_dir) / "previews"
                    expected = preview_dir / f"preview_{p.stem}.jpg"
                    if expected.exists():
                        preview_paths.append(str(expected))
                    else:
                        try:
                            result = generate_preview(
                                working_dir=working_dir, fits_path=str(p),
                                format="jpg", quality=95,
                            )
                            preview_paths.append(result["preview_path"])
                        except Exception:
                            pass
                else:
                    preview_paths.append(img)
            payload["images"] = preview_paths

    return payload


# ── Thread management ────────────────────────────────────────────────────────


def _launch_thread(stream_input: Any) -> None:
    """Start the background graph thread with fresh queues."""
    eq: queue.Queue = queue.Queue()
    iq: queue.Queue = queue.Queue()

    st.session_state["event_queue"] = eq
    st.session_state["input_queue"] = iq
    st.session_state["pending_interrupt"] = None
    st.session_state["running"] = True

    t = threading.Thread(
        target=_graph_thread,
        args=(
            st.session_state["graph"],
            st.session_state["config"],
            stream_input,
            eq,
            iq,
        ),
        daemon=True,
    )
    t.start()


# ── Session launchers ────────────────────────────────────────────────────────


def _start_new_session(
    directory: str,
    target: str,
    bortle: int,
    sqm: float | None,
    remove_stars: bool | None,
    notes: str | None,
) -> None:
    _init_graph()
    _prune_stale_checkpoints()

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
    _save_session(thread_id, target, working_dir=dataset["working_dir"])
    _configure_logging(thread_id)

    st.session_state.update({
        "thread_id": thread_id,
        "config": config,
        "events": [],
    })
    _launch_thread(initial_state)


def _resume_session(thread_id: str) -> None:
    _init_graph()
    config = {"configurable": {"thread_id": thread_id}}
    graph = st.session_state["graph"]

    state = graph.get_state(config)
    pending: list = []
    for task in (state.tasks or []):
        if hasattr(task, "interrupts") and task.interrupts:
            pending.extend(task.interrupts)

    _configure_logging(thread_id)

    st.session_state.update({
        "thread_id": thread_id,
        "config": config,
        "events": [],
    })

    if pending:
        # Re-derive display payload from current state, then show immediately
        payload = pending[0].value
        payload = _refresh_interrupt_payload(payload, graph, config)
        st.session_state["pending_interrupt"] = payload
        st.session_state["running"] = False
        st.session_state["events"].append({"type": "interrupt", "payload": payload})
        # Create queues — thread will start when user responds
        st.session_state["event_queue"] = queue.Queue()
        st.session_state["input_queue"] = queue.Queue()
    else:
        # No pending interrupt — start streaming from checkpoint
        _launch_thread(None)


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
            st.markdown(f"---\n#### {label}")

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
        st.subheader(f"Agent message ({payload.get('phase', '')} phase)")
        agent_text = payload.get("agent_text", "")
        if agent_text:
            with st.chat_message("assistant"):
                st.write(agent_text)
    else:
        st.subheader(f"Review required: {payload.get('title', 'Human Review')}")

        # Show agent's response during multi-turn HITL
        agent_reply = payload.get("agent_text", "")
        if agent_reply:
            with st.chat_message("assistant"):
                st.write(agent_reply)

        if ptype == "image_review" and images:
            cols = st.columns(min(len(images), 2))
            for i, img_path in enumerate(images):
                p = Path(img_path)
                with cols[i % len(cols)]:
                    if p.exists():
                        st.image(str(p), caption=p.name, width="stretch")
                    else:
                        st.warning(f"Image not found: {img_path}")

        elif ptype == "data_review":
            context = payload.get("context", [])
            for msg in reversed(context):
                if hasattr(msg, "name") and hasattr(msg, "content"):
                    content = text_content(msg.content)
                    with st.expander(f"Result: {msg.name}", expanded=True):
                        st.code(content, language="json")
                    break

    st.divider()
    feedback_key = f"hitl_feedback_{st.session_state.get('_hitl_counter', 0)}"
    feedback = st.text_area(
        "Leave empty and click Approve to continue, or type feedback/questions:",
        key=feedback_key,
        height=80,
    )

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("Approve ✓", type="primary"):
            if feedback.strip():
                # Send the human's note, then approve
                _submit_response(feedback.strip(), approve=True, record_feedback=True)
            else:
                _submit_response(APPROVE_SENTINEL)
    with col2:
        if st.button("Send") and feedback.strip():
            _submit_response(feedback.strip(), record_feedback=True)


def _submit_response(
    response: str,
    record_feedback: bool = False,
    approve: bool = False,
) -> None:
    """Put response on the input queue. One code path for all cases."""
    if record_feedback:
        st.session_state.get("events", []).append({
            "type": "human_feedback",
            "content": response,
        })

    # Approve with a note: prefix the sentinel so hitl_check recognizes
    # it as approval AND includes the human's text as a message.
    if approve:
        response = f"{APPROVE_SENTINEL}\n{response}"

    # If thread is waiting on input_queue, feed it
    if "input_queue" in st.session_state:
        st.session_state["input_queue"].put(response)
    else:
        # Resume case: thread hasn't started. Launch with resume Command.
        _launch_thread(Command(resume=response))

    st.session_state["pending_interrupt"] = None
    st.session_state["running"] = True
    st.session_state["_hitl_counter"] = st.session_state.get("_hitl_counter", 0) + 1
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
        notes = st.text_area(
            "Session notes (optional)", height=60,
            placeholder="e.g. L-eNhance filter, gain 100, poor seeing",
        )

        start_disabled = not (directory.strip() and target.strip())
        if st.button("Start →", type="primary", disabled=start_disabled):
            with st.spinner("Ingesting dataset..."):
                try:
                    _start_new_session(
                        directory.strip(), target.strip(), bortle, sqm,
                        remove_stars, notes.strip() or None,
                    )
                except Exception as e:
                    st.error(f"Failed to start: {e}")
                    st.stop()
            st.rerun()

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
            options = [f"{s['target']} — {s['started'][:19]}" for s in sessions[:8]]
            idx = st.selectbox(
                "Recent sessions", range(len(options)),
                format_func=lambda i: options[i],
            )
            resume_thread = sessions[idx]["thread_id"]
            st.caption(f"`{resume_thread}`")
        else:
            resume_thread = st.text_input(
                "Thread ID", placeholder="run-m42-20260311-120000",
            )

        if st.button("Resume →", disabled=not resume_thread):
            with st.spinner("Loading checkpoint..."):
                try:
                    _resume_session(resume_thread)
                except Exception as e:
                    st.error(f"Failed to resume: {e}")
                    st.stop()
            st.rerun()

        if "thread_id" in st.session_state:
            st.divider()
            st.caption("Active thread:")
            st.code(st.session_state["thread_id"], language=None)

    # ── Main area ──────────────────────────────────────────────────────────────

    if "events" not in st.session_state:
        st.info(
            "Enter your dataset directory and target name in the sidebar, "
            "then click **Start →**.\n\n"
            "Or resume a previous session with the **Resume** section."
        )
        return

    # Drain event queue — collect everything the background thread produced
    if "event_queue" in st.session_state:
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
                if ev["type"] in ("done", "error"):
                    st.session_state["running"] = False

    # Render the full event log — this is the conversation history
    _render_events(st.session_state.get("events", []))

    # HITL panel (shown when interrupted — user can see and respond)
    if st.session_state.get("pending_interrupt"):
        _render_hitl_panel(st.session_state["pending_interrupt"])

    # Poll while running
    elif st.session_state.get("running"):
        with st.spinner("Agent working..."):
            time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
