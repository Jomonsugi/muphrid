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
from muphrid.graph import review as review_ctl
from muphrid.graph.hitl import (
    set_autonomous,
    set_vlm_autonomous,
    set_vlm_retention_max,
    vlm_window_cap,
)
from muphrid.graph.content import text_content
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

    # Verify the checkpoint DB is intact before LangGraph starts using it.
    # Without this, a corrupt SQLite file (caused by a previously killed
    # process, two Gradio instances writing concurrently, or a disk hiccup)
    # surfaces as a cryptic "database disk image is malformed" stack trace
    # mid-stream, after the user has already started a session and lost
    # in-flight work. Catching it here lets us either fail loud at launch
    # or, when LANGGRAPH_AUTO_RECOVER_DB is set, attempt the recovery
    # routine (see scripts/recover_checkpoint_db.py — quarantines the
    # broken file and rebuilds a clean one from readable rows).
    _check_checkpoint_db_integrity("checkpoints.db")

    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("muphrid.graph.state", "ProcessingPhase")]
    )
    conn = await aiosqlite.connect("checkpoints.db")
    checkpointer = AsyncSqliteSaver(conn=conn, serde=serde)
    await checkpointer.setup()
    _GRAPH = build_graph(checkpointer=checkpointer)
    logger.info("Async resources initialized: graph + AsyncSqliteSaver")


def _check_checkpoint_db_integrity(db_path: str) -> None:
    """
    Run SQLite's PRAGMA integrity_check before LangGraph opens the DB.

    On corruption: log a clear, actionable error and set _DEPENDENCY_ERROR
    so start_session refuses to begin a run that would crash anyway.
    Recovery is a separate, deliberate action (see scripts/recover_checkpoint_db.py).
    Auto-recovery is intentionally NOT enabled by default — silently
    rebuilding the DB out from under a running process can mask real disk
    issues and lose context the user might want to inspect first.
    """
    import sqlite3
    from pathlib import Path

    global _DEPENDENCY_ERROR
    p = Path(db_path)
    if not p.exists():
        # Fresh install — AsyncSqliteSaver.setup() will create it. Nothing to check.
        return
    try:
        conn = sqlite3.connect(str(p))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        msg = (
            f"checkpoint DB at {p} is unreadable ({e}). "
            f"Quarantine the file and start fresh, or run "
            f"`python scripts/recover_checkpoint_db.py` to attempt recovery."
        )
        logger.error(msg)
        _DEPENDENCY_ERROR = (_DEPENDENCY_ERROR + "\n\n" if _DEPENDENCY_ERROR else "") + msg
        return

    ok = result == [("ok",)]
    if not ok:
        n_issues = len(result)
        sample = "; ".join(str(r[0])[:120] for r in result[:3])
        msg = (
            f"checkpoint DB at {p} reports {n_issues} integrity issue(s) — "
            f"first: {sample}. The database is corrupted and LangGraph will "
            f"crash mid-stream when it tries to write. Quarantine the file "
            f"(rename to checkpoints.db.corrupt-<ts>) and let LangGraph "
            f"create a fresh DB on next launch, or run "
            f"`python scripts/recover_checkpoint_db.py` to extract readable "
            f"checkpoints into a new DB before starting any session. "
            f"Common causes: a previous Python process killed mid-write, "
            f"two Gradio instances writing concurrently to the same file, "
            f"or a disk-level fault."
        )
        logger.error(msg)
        _DEPENDENCY_ERROR = (_DEPENDENCY_ERROR + "\n\n" if _DEPENDENCY_ERROR else "") + msg


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


def _working_dir_from_variants(variants: list[dict], fallback: str = "") -> str:
    """Return a usable working directory from explicit state or variant paths."""
    if fallback:
        return fallback
    for v in variants:
        if not isinstance(v, dict):
            continue
        fp = v.get("file_path")
        if fp:
            return str(Path(fp).parent)
    return ""


def _variant_gallery_items(
    variants: list[dict],
    working_dir: str,
    is_linear: bool,
    *,
    prefix: str = "",
) -> list[tuple]:
    """Build Gradio gallery entries for variant dicts."""
    if not variants:
        return []
    wd = _working_dir_from_variants(variants, working_dir)
    if not wd:
        return []

    paths = [v.get("file_path") for v in variants if isinstance(v, dict) and v.get("file_path")]
    if not paths:
        return []

    try:
        preview_paths = _convert_fits_to_preview(paths, wd, is_linear)
    except Exception as e:
        logger.warning(f"Variant preview generation failed: {e}")
        return []

    items: list[tuple] = []
    path_idx = 0
    for v in variants:
        if not isinstance(v, dict) or not v.get("file_path"):
            continue
        if path_idx >= len(preview_paths):
            break
        preview = preview_paths[path_idx]
        path_idx += 1
        if not Path(preview).exists():
            continue
        vid = v.get("id", "?")
        label = v.get("label", "")
        caption = f"{vid} — {label}" if label else str(vid)
        if prefix:
            caption = f"{prefix}{caption}"
        items.append((preview, caption))
    return items


def _proposal_variants(proposal: list[dict]) -> list[dict]:
    """Extract Variant dicts from HITL proposal entries."""
    out: list[dict] = []
    for entry in proposal or []:
        if not isinstance(entry, dict):
            continue
        variant = entry.get("variant")
        if isinstance(variant, dict):
            out.append(variant)
    return out


def _proposal_from_review_session(
    review_session: dict | None,
    variant_pool: list[dict],
) -> list[dict]:
    """Resolve ReviewSession proposal candidates into UI proposal entries."""
    if not review_ctl.review_is_open(review_session):
        return []
    artifact = (review_session or {}).get("proposal", {}) or {}
    candidates = artifact.get("candidates", []) or []
    pool_by_id = {
        v.get("id"): v for v in variant_pool or []
        if isinstance(v, dict) and v.get("id")
    }
    resolved: list[dict] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        vid = candidate.get("variant_id")
        variant = pool_by_id.get(vid)
        if variant is None:
            continue
        resolved.append({
            "variant": variant,
            "rationale": candidate.get("rationale", ""),
            "presented_at": candidate.get("presented_at", ""),
            "recommendation": artifact.get("recommendation"),
            "tradeoffs": artifact.get("tradeoffs", []),
            "metric_highlights": artifact.get("metric_highlights", {}),
            "proposal_rationale": artifact.get("rationale", ""),
        })
    return resolved


def _append_chat_once(chat_messages: list[dict], role: str, content: str) -> None:
    """Append a chat message unless it is an immediate duplicate."""
    text = (content or "").strip()
    if not text:
        return
    if chat_messages:
        last = chat_messages[-1]
        if last.get("role") == role and last.get("content") == text:
            return
    chat_messages.append({"role": role, "content": text})


def _proposal_rationale_summary(proposal: list[dict]) -> str:
    """Fallback chat summary when a proposal has rationale but no agent text."""
    if not proposal:
        return ""
    lines = ["**Agent-presented candidate rationale:**"]
    for entry in proposal:
        if not isinstance(entry, dict):
            continue
        variant = entry.get("variant") or {}
        if not isinstance(variant, dict):
            continue
        vid = variant.get("id", "?")
        label = variant.get("label", "")
        rationale = (entry.get("rationale", "") or "").strip()
        recommendation = entry.get("recommendation")
        heading = f"- **{vid}**"
        if label:
            heading += f" — {label}"
        if recommendation == vid:
            heading += " (recommended)"
        if rationale:
            heading += f": {rationale}"
        lines.append(heading)
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Stream chunk parsing (sync, defensive) ───────────────────────────────────


def _parse_stream_chunks(
    chunk: dict,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    working_dir: str,
    is_linear: bool,
    in_review: bool = False,
) -> tuple[dict | None, bool]:
    """
    Parse a single stream chunk (stream_mode="updates") and update the UI lists.

    Returns (interrupt_payload_or_None, updated_in_review).
    Routes agent text to chat during HITL, activity log during autonomous.

    The variant_pool and proposal lists are mutated in place from any state
    updates carried in this chunk. That gives the UI a live view of the
    pool growing while the agent runs, even in autonomous phases where no
    HITL interrupt fires. The bottom pool filmstrip mirrors this workbench
    history; the main gallery stays reserved for presented review candidates
    and explicit present_images context.

    Defensive: handles None values, non-dict updates, unexpected chunk formats.
    """
    if "__interrupt__" in chunk:
        return chunk["__interrupt__"][0].value, in_review

    for node_name, update in chunk.items():
        if node_name == "__interrupt__":
            continue
        if not isinstance(update, dict):
            continue

        if "review_session" in update:
            review_session = update.get("review_session")
            if review_ctl.review_is_open(review_session):
                in_review = True
                resolved = _proposal_from_review_session(review_session, variant_pool)
                if resolved:
                    proposal.clear()
                    proposal.extend(resolved)
            elif review_session and review_session.get("status") == "closed":
                in_review = False

        # Live pool/proposal updates — autonomous-mode visibility.
        # Without these, variant_pool_state and proposal_state are only
        # refreshed at HITL gates (from interrupt_payload), so a user
        # running a phase autonomously sees nothing until they stop. With
        # them, the panels update each chunk, the same way chat and
        # activity log do.
        if "variant_pool" in update:
            new_pool = update.get("variant_pool")
            if isinstance(new_pool, list):
                # Replace semantics — variant_pool reducer is plain replace
                # (the upstream variant_snapshot computes the full new list),
                # so mirror that here and rebuild the pool filmstrip.
                variant_pool.clear()
                variant_pool.extend(new_pool)

                # Auto-refresh the bottom workbench filmstrip from the pool.
                # The main review gallery is reserved for agent-presented
                # candidates/context; raw pool history stays observational.
                pool_gallery_images.clear()
                if new_pool:
                    pool_gallery_images.extend(
                        _variant_gallery_items(new_pool, working_dir, is_linear)
                    )

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
                    if in_review:
                        # During HITL, assistant text is part of the
                        # collaboration. Show it even when the same message
                        # also contains tool calls; otherwise the UI feels
                        # like a silent command queue.
                        _append_chat_once(chat_messages, "assistant", agent_text)
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

    return None, in_review


# ── Core streaming handler (async generator) ────────────────────────────────


async def _stream_graph(
    graph,
    config: dict,
    stream_input,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    working_dir: str,
    is_linear: bool,
):
    """
    Core async streaming loop. Yields
    (chat, activity, review_gallery, pool_gallery, pool, proposal).

    Two distinct UI surfaces populated from the HITL payload:
      * variant_pool — passive history (every variant produced this segment)
      * proposal     — agent's curation (only what was passed to
                       present_for_review). Approve buttons attach here.

    The proposal is replaced fresh from each HITL payload. The pool can also
    update outside HITL as a live workbench/observability surface.
    """
    interrupt_payload = None
    in_review = False

    # Snapshot gallery content before streaming so we can detect whether
    # present_images updated it during this round.
    gallery_before_stream = list(gallery_images)

    async for chunk in graph.astream(stream_input, config=config, stream_mode="updates"):
        result, in_review = _parse_stream_chunks(
            chunk, chat_messages, activity_log, gallery_images, pool_gallery_images,
            variant_pool, proposal,
            working_dir, is_linear, in_review,
        )
        if result is not None:
            interrupt_payload = result

        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal

    # After stream ends, handle any interrupt payload for UI display
    if interrupt_payload is not None:
        title = interrupt_payload.get("title", "Review")
        images = interrupt_payload.get("images", [])
        agent_text = (interrupt_payload.get("agent_text", "") or "").strip()

        # Refresh both panels from the payload. Backend already cleared
        # them on phase advance / variant approval, so each new gate
        # starts with the right state.
        variant_pool = list(interrupt_payload.get("variant_pool", []) or [])
        proposal = list(interrupt_payload.get("proposal", []) or [])
        approval_allowed = bool(interrupt_payload.get("approval_allowed", True))
        review_state = interrupt_payload.get("review_state", "ready")

        pool_gallery_images.clear()
        pool_gallery_images.extend(
            _variant_gallery_items(variant_pool, working_dir, is_linear)
        )

        # Keep the gallery in sync with the variant panel. If the agent
        # called present_images during this stream, its curated comparison
        # set (which may include reference images like the original) takes
        # precedence. Otherwise, populate the gallery directly from the
        # variant pool so the viewer matches the Approve buttons below.
        # This covers the silent-reexecution / backstop case where the
        # agent never narrated or presented images.
        gallery_refreshed = (gallery_images != gallery_before_stream)

        # Resumed sessions originating from a CLI run never went through
        # _register_session, so working_dir falls back to "" in the UI
        # state. The variant entries themselves carry full FITS paths
        # whose parent directory IS the working_dir, so derive it from
        # there as a fallback for preview generation.
        if not working_dir and variant_pool:
            for v in variant_pool:
                fp = v.get("file_path")
                if fp:
                    working_dir = str(Path(fp).parent)
                    break
        if not working_dir and images:
            for img in images:
                if img:
                    working_dir = str(Path(img).parent)
                    break

        # Main review gallery population + canonical labeling.
        #
        # Two invariants we hold here:
        #   (1) Every agent-presented variant in the approve panel has a
        #       gallery preview. Raw pool entries are shown only in the bottom
        #       workbench filmstrip.
        #   (2) Every gallery entry whose source matches a variant has the
        #       variant id visible in its caption. This is the fix for
        #       the user's "neither matches the image" confusion: the
        #       Approve button says "T14_v1" so the gallery caption needs
        #       to start with "T14_v1" too. The agent's descriptive label
        #       (e.g. "GHS D3 SP0.15") is preserved when it provided one
        #       via present_images — we just prefix it with the id.
        review_variants = _proposal_variants(proposal)
        if review_variants and working_dir:
            # Compute predicted preview paths for every variant once and
            # build a lookup: preview_path → (variant_id, variant_label).
            # Reusing this map below avoids the cost of re-converting and
            # gives both the merge logic AND the caption normalizer a
            # single source of truth for "this preview belongs to that
            # variant". _convert_fits_to_preview is idempotent — it
            # generates JPGs once and returns cached paths thereafter,
            # so calling it here is cheap even when previews already exist.
            variant_paths = [v["file_path"] for v in review_variants if v.get("file_path")]
            preview_paths = (
                _convert_fits_to_preview(variant_paths, working_dir, is_linear)
                if variant_paths else []
            )
            variant_info_by_preview: dict[str, tuple[str, str]] = {}
            for v, p in zip(review_variants, preview_paths):
                if Path(p).exists():
                    variant_info_by_preview[str(p)] = (
                        v.get("id", "?"),
                        v.get("label", ""),
                    )

            # Existing-gallery lookup (ignore non-tuple entries defensively).
            existing_preview_paths = {
                str(entry[0]) for entry in gallery_images
                if isinstance(entry, (tuple, list)) and entry
            }

            if not gallery_refreshed:
                # Case A: present_images was NOT called this stream — replace
                # gallery wholesale from variant_pool with canonical captions.
                gallery_images.clear()
                for v, p in zip(review_variants, preview_paths):
                    if Path(p).exists():
                        gallery_images.append((p, f"{v['id']} — {v.get('label', '')}"))
            else:
                # Case B/C: present_images WAS called. Two passes:
                #   1) Append any variant whose preview isn't already shown.
                #   2) Normalize captions: any existing gallery entry whose
                #      path matches a known variant gets the id prefixed,
                #      preserving the agent's descriptive label.
                for v, p in zip(review_variants, preview_paths):
                    if Path(p).exists() and str(p) not in existing_preview_paths:
                        gallery_images.append(
                            (p, f"{v['id']} — {v.get('label', '')}")
                        )
                # Caption normalization pass — runs over the FULL gallery
                # (both pre-existing and just-appended entries) so every
                # variant-backed preview is consistently labeled.
                for i, entry in enumerate(gallery_images):
                    if not (isinstance(entry, (tuple, list)) and len(entry) >= 2):
                        continue
                    path, current_caption = entry[0], entry[1]
                    info = variant_info_by_preview.get(str(path))
                    if info is None:
                        # Not a variant preview — agent's reference image
                        # (e.g. the linear-stage original); leave caption alone.
                        continue
                    vid, vlabel = info
                    # If the caption already begins with the variant id,
                    # we're done. Otherwise, preserve the agent's chosen
                    # descriptive label by prefixing it with the id.
                    caption_str = str(current_caption or "")
                    if caption_str.startswith(vid):
                        continue
                    if caption_str and caption_str != vlabel:
                        # Agent supplied a meaningful descriptive label —
                        # keep it, just identify which variant it is.
                        new_caption = f"{vid}: {caption_str}"
                    else:
                        # No agent label or it's redundant with the variant
                        # label — fall back to the canonical id+label form.
                        new_caption = f"{vid} — {vlabel}"
                    gallery_images[i] = (path, new_caption)
        elif images and working_dir and not gallery_images:
            # No variants but the interrupt payload had images (older flow
            # or non-variant HITL gate). Show those.
            preview_paths = _convert_fits_to_preview(images, working_dir, is_linear)
            if gallery_images is None:
                gallery_images = []
            for i, p in enumerate(preview_paths):
                if Path(p).exists():
                    gallery_images.append((p, f"Variant {i + 1}"))

        # Show HITL review prompt in chat. Agent text is first-class in
        # collaboration mode; if the model only provided rationale through
        # present_for_review, surface that rationale conversationally too.
        if agent_text:
            _append_chat_once(chat_messages, "assistant", agent_text)
        elif proposal:
            _append_chat_once(
                chat_messages,
                "assistant",
                _proposal_rationale_summary(proposal),
            )

        if review_state == "needs_curation":
            footer = (
                f"**{title}** — agent needs to select candidates.\n\n"
                f"---\n"
                f"*The pool filmstrip below is visible for inspection, but "
                f"nothing is approvable yet. Ask the agent what to compare, "
                f"or tell it to present the candidate(s) it recommends.*"
            )
        elif proposal and approval_allowed:
            footer = (
                f"**{title}** — {len(proposal)} candidate"
                f"{'s' if len(proposal) != 1 else ''} presented for review.\n\n"
                f"---\n"
                f"*Pick a presented candidate to commit and advance, "
                f"or send feedback to iterate. Use the Proposal panel's "
                f"approval note if you want to record why you approved it.*"
            )
        else:
            footer = (
                f"**{title}** — awaiting your review.\n\n"
                f"---\n"
                f"*No variants in the pool yet — send feedback to guide the agent.*"
            )

        chat_messages.append({"role": "assistant", "content": footer})

        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal


async def _recover_active_hitl_ui(
    config: dict,
    state: dict,
    chat_messages: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    *,
    message: str,
) -> bool:
    """
    Re-render a paused HITL gate after a streaming failure or timeout.

    If the graph already reached interrupt(), the checkpoint has enough state
    for the UI to recover without consuming the human response slot.
    """
    if not config:
        return False

    try:
        snapshot = await _GRAPH.aget_state(config)
        saved_values = snapshot.values if snapshot else {}
    except Exception as e:
        logger.warning(f"HITL recovery state read failed: {e}")
        return False

    if not review_ctl.review_is_open(saved_values.get("review_session")):
        return False

    saved_pool = list(saved_values.get("variant_pool", []) or [])
    saved_proposal = _proposal_from_review_session(
        saved_values.get("review_session"), saved_pool
    )

    variant_pool.clear()
    variant_pool.extend(saved_pool)
    proposal.clear()
    proposal.extend(saved_proposal)

    wd = state.get("working_dir") or (
        saved_values.get("dataset", {}).get("working_dir", "")
        if isinstance(saved_values.get("dataset"), dict) else ""
    )
    if not wd and saved_pool:
        first_path = saved_pool[0].get("file_path") if isinstance(saved_pool[0], dict) else None
        if first_path:
            wd = str(Path(first_path).parent)
    is_linear = bool((saved_values.get("metrics", {}) or {}).get("is_linear_estimate", True))

    if wd:
        try:
            pool_gallery_images.clear()
            pool_gallery_images.extend(_variant_gallery_items(saved_pool, wd, is_linear))
            gallery_images.clear()
            gallery_images.extend(
                _variant_gallery_items(_proposal_variants(saved_proposal), wd, is_linear)
            )
            state["working_dir"] = wd
            state["is_linear"] = is_linear
        except Exception as e:
            logger.warning(f"HITL recovery preview generation failed: {e}")

    chat_messages.append({"role": "assistant", "content": message})
    return True


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
    pool_gallery_images: list[tuple] = []
    variant_pool: list[dict] = []
    proposal: list[dict] = []

    if not dataset_path or not target_name:
        chat_messages.append({"role": "assistant", "content": "Please provide both a dataset path and target name."})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state
        return

    # Block session start if dependencies are missing (checked at app startup)
    if _DEPENDENCY_ERROR:
        chat_messages.append({
            "role": "assistant",
            "content": f"**Cannot start — missing dependencies.**\n\n{_DEPENDENCY_ERROR}",
        })
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state
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
    yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state

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

        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state

        # Stream the graph
        async for chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop in _stream_graph(
            _GRAPH, config, initial_state,
            chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal,
            working_dir, state.get("is_linear", True),
        ):
            yield chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop, state

    except Exception as e:
        logger.exception(f"Session error: {e}")
        recovered = await _recover_active_hitl_ui(
            state.get("config") or {}, state, chat_messages, gallery_images,
            pool_gallery_images, variant_pool, proposal,
            message=(
                f"Streaming stopped before the current turn completed: `{e}`\n\n"
                "If the graph reached an HITL pause, the review state has "
                "been restored below."
            ),
        )
        if not recovered:
            chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state


async def send_message(
    user_text: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    state: dict,
):
    """
    Handle a user feedback message during a HITL gate — resume the graph
    with the typed text as the interrupt response.

    Failure modes that this function explicitly handles instead of silently
    dropping (see Issue #71 for the regression that motivated each):

      * Empty / whitespace-only submit: surface a toast so the user knows
        the click registered. Without this, users typing into a placeholder-
        empty textbox saw the box clear with no chat-history echo and no
        warning, indistinguishable from a successful send.
      * No active session yet: append a chat message naming the missing
        prerequisite. Same rationale — silent drop is worse than a noisy
        guard rail.
      * Streaming exception: catch and append the error to chat so the
        human sees what went wrong and the textbox isn't permanently
        disabled.

    Note on Gradio 6 Chatbot/Textbox interaction: chat_messages is the
    same list object across the streaming yields — _stream_graph mutates
    it in place via _parse_stream_chunks. The user message we append on
    line ~? therefore survives all subsequent yields; if you ever see the
    user message disappear, look for accidental list reassignment.
    """
    # Gradio 6 Gallery.preprocess(None) returns None for empty gallery.
    if gallery_images is None:
        gallery_images = []
    if pool_gallery_images is None:
        pool_gallery_images = []
    if variant_pool is None:
        variant_pool = []
    if proposal is None:
        proposal = []
    if chat_messages is None:
        chat_messages = []

    if not user_text or not user_text.strip():
        # Surface as a Gradio Warning toast — visible feedback that the
        # submit fired, rather than silently clearing the textbox. Use
        # gr.skip() for msg_input so we don't disturb whatever the user
        # has actually typed (spaces, partial draft, etc).
        gr.Warning("Type something before sending.", duration=3)
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, gr.skip()
        return

    config = state.get("config")
    working_dir = state.get("working_dir", "")
    is_linear = state.get("is_linear", True)

    if not config:
        chat_messages.append({
            "role": "assistant",
            "content": "No active session. Start a new session or resume one first.",
        })
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, gr.skip()
        return

    # Echo the user's message FIRST and clear the textbox in the SAME yield.
    # This is the only point where we want msg_input to be overwritten —
    # after this we use gr.skip() so background streaming yields don't
    # clobber whatever the user starts typing next. (Bug we hit: lots of
    # streaming chunks each yield an empty string for msg_input, which
    # erases keystrokes mid-typing as the agent processes the previous
    # turn in the background.)
    chat_messages.append({"role": "user", "content": user_text})
    yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, ""

    try:
        async for chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop in _stream_graph(
            _GRAPH, config, Command(resume=review_ctl.feedback_resume_event(user_text)),
            chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal,
            working_dir, is_linear,
        ):
            # gr.skip() means "leave msg_input alone" — the textbox is
            # already empty from the echo yield above; the user may have
            # started typing their next message while we stream.
            yield chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop, state, gr.skip()
    except Exception as e:
        logger.exception(f"Send message error: {e}")
        recovered = await _recover_active_hitl_ui(
            config, state, chat_messages, gallery_images, pool_gallery_images,
            variant_pool, proposal,
            message=(
                f"Model streaming stopped before the HITL turn completed: `{e}`\n\n"
                "The active review state was restored from the checkpoint. "
                "You can approve a presented candidate or send feedback again."
            ),
        )
        if not recovered:
            chat_messages.append({"role": "assistant", "content": f"Error during streaming: {e}"})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, gr.skip()


async def approve_variant_action(
    variant_id: str,
    variant_label: str,
    rationale: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    state: dict,
):
    """
    Handle a variant Approve button click — resume the graph with a typed
    ReviewHumanEvent that names the chosen variant and carries the proposal
    panel's optional approval note as the human's rationale.

    The backend's hitl_check applies the event, calls promote_variant to
    move the chosen variant's file to current_image, and clears the pool.
    """
    if gallery_images is None:
        gallery_images = []
    if pool_gallery_images is None:
        pool_gallery_images = []
    if variant_pool is None:
        variant_pool = []
    if proposal is None:
        proposal = []

    config = state.get("config")
    working_dir = state.get("working_dir", "")
    is_linear = state.get("is_linear", True)

    if not config:
        chat_messages.append({"role": "assistant", "content": "No active session."})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, gr.skip()
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
    # Clear the approval note HERE (the rationale is now consumed/echoed).
    # Subsequent yields use gr.skip() to avoid clobbering any new note the
    # user starts typing while the graph resumes.
    yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, ""

    try:
        async for chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop in _stream_graph(
            _GRAPH,
            config,
            Command(resume=review_ctl.approval_resume_event(
                variant_id, rationale.strip()
            )),
            chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal,
            working_dir, is_linear,
        ):
            yield chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop, state, gr.skip()
    except Exception as e:
        logger.exception(f"Variant approve error: {e}")
        recovered = await _recover_active_hitl_ui(
            config, state, chat_messages, gallery_images, pool_gallery_images,
            variant_pool, proposal,
            message=(
                f"Approval hit an error before the gate closed: `{e}`\n\n"
                "The active review state was restored from the checkpoint. "
                "You can approve again or send feedback."
            ),
        )
        if not recovered:
            chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, gr.skip()


def _proposal_choice_options(proposal: list[dict]) -> list[tuple[str, str]]:
    """Build stable dropdown choices for presented candidates."""
    choices: list[tuple[str, str]] = []
    for entry in proposal or []:
        if not isinstance(entry, dict):
            continue
        variant = entry.get("variant")
        if not isinstance(variant, dict):
            continue
        vid = variant.get("id")
        if not vid:
            continue
        label = variant.get("label", "")
        display = f"{vid} — {label}" if label else str(vid)
        choices.append((display, str(vid)))
    return choices


def update_approval_controls(proposal: list[dict], streaming: bool):
    """
    Keep approval controls stable outside @gr.render.

    Gradio dynamic render functions can redraw their component tree when state
    changes. Event handlers attached to components inside that tree may become
    stale on the client. The approval action is a state transition, so it uses
    fixed top-level controls that are only updated from proposal_state.
    """
    choices = _proposal_choice_options(proposal)
    enabled = bool(choices) and not bool(streaming)
    value = choices[0][1] if choices else None
    return (
        gr.update(choices=choices, value=value, interactive=enabled),
        gr.update(value="", interactive=enabled),
        gr.update(interactive=enabled),
    )


async def approve_selected_variant_action(
    variant_id: str | None,
    rationale: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    state: dict,
):
    """Approve the candidate selected in the stable Proposal controls."""
    if not variant_id:
        gr.Warning("Select a presented candidate to approve.", duration=3)
        yield (
            chat_messages, activity_log, gallery_images, pool_gallery_images,
            variant_pool, proposal, state, gr.skip(),
        )
        return

    variant_label = ""
    for entry in proposal or []:
        variant = entry.get("variant") if isinstance(entry, dict) else None
        if isinstance(variant, dict) and variant.get("id") == variant_id:
            variant_label = str(variant.get("label", ""))
            break

    async for result in approve_variant_action(
        variant_id, variant_label, rationale,
        chat_messages, activity_log, gallery_images, pool_gallery_images,
        variant_pool, proposal, state,
    ):
        yield result


async def resume_session(
    resume_id: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    state: dict,
):
    """Resume a session from an existing checkpoint."""
    if gallery_images is None:
        gallery_images = []
    if pool_gallery_images is None:
        pool_gallery_images = []
    if variant_pool is None:
        variant_pool = []
    if proposal is None:
        proposal = []
    if not resume_id.strip():
        # Don't clear UI state for the validation-error case — preserve
        # whatever was on screen so the user keeps their context.
        chat_messages.append({"role": "assistant", "content": "Please enter a thread ID to resume."})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state
        return

    # Clear UI state from any prior session in this Gradio process before
    # streaming the resumed thread. The chat / activity log / gallery /
    # variant pool are session-scoped client state, not graph state — they
    # don't get reset by LangGraph when a different thread is resumed, so
    # without this you'd see stale messages from whatever session ran
    # earlier in the same Gradio process. Mutate in place so the Gradio
    # State bindings see the updated lists.
    chat_messages.clear()
    activity_log.clear()
    gallery_images.clear()
    pool_gallery_images.clear()
    variant_pool.clear()
    proposal.clear()

    config = {"configurable": {"thread_id": resume_id}}
    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "200"))
    if recursion_limit > 0:
        config["recursion_limit"] = recursion_limit

    # Recover the run's working directory from the sessions index so
    # downstream consumers (preview generation, FITS lookups, audit
    # report writes) have the right path. Falls back to empty if the
    # session was never registered (e.g. CLI run that didn't go through
    # _register_session); the graph state's dataset.working_dir is the
    # source of truth and will be present after the first super-step.
    resumed_working_dir = _lookup_session_dir(resume_id) or ""

    state = {
        "thread_id": resume_id,
        "config": config,
        "working_dir": resumed_working_dir,
        "is_linear": True,
    }

    chat_messages.append({"role": "assistant", "content": f"Resuming session **{resume_id}**..."})
    yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state

    # Active-HITL-aware resume. Inspect the saved state BEFORE streaming.
    # If the run was paused at a review_session gate, do NOT
    # send a Command(resume=...) value into the graph — any value would
    # be treated as the human's response to the pending interrupt and
    # silently consume the gate. Instead, re-render the gate UI from saved
    # state so the human can take real action; send_message carries
    # feedback, while approve_variant_click carries structured approval.
    try:
        snapshot = await _GRAPH.aget_state(config)
        saved_values = snapshot.values if snapshot else {}
        is_paused_at_hitl = review_ctl.review_is_open(saved_values.get("review_session"))
        is_legacy_hitl = bool(saved_values.get("active_hitl")) and not is_paused_at_hitl
    except Exception as e:
        logger.warning(f"resume_session: aget_state failed (non-fatal): {e}")
        saved_values = {}
        is_paused_at_hitl = False
        is_legacy_hitl = False

    if is_legacy_hitl:
        chat_messages.append({
            "role": "assistant",
            "content": (
                "**This saved checkpoint predates Review Mode state.** It has "
                "`active_hitl=True` but no open `review_session`, so the app "
                "cannot safely reconstruct the approval contract or consume a "
                "resume value as human approval. Start from a clean cloned "
                "checkpoint before the gate, or rerun this phase with the "
                "current Review Mode implementation."
            ),
        })
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state
        return

    if is_paused_at_hitl:
        # Re-render the gate UI from saved state instead of streaming.
        saved_pool = list(saved_values.get("variant_pool", []) or [])
        saved_proposal = _proposal_from_review_session(
            saved_values.get("review_session"), saved_pool
        )
        proposal.clear()
        proposal.extend(saved_proposal)
        if saved_pool:
            variant_pool.clear()
            variant_pool.extend(saved_pool)
            # Materialize previews for the bottom workbench filmstrip and,
            # separately, the main review gallery if the agent had already
            # presented candidates.
            try:
                wd = state.get("working_dir") or (
                    saved_values.get("dataset", {}).get("working_dir", "")
                    if isinstance(saved_values.get("dataset"), dict) else ""
                )
                if not wd and saved_pool:
                    fp = saved_pool[0].get("file_path")
                    if fp:
                        wd = str(Path(fp).parent)
                if wd:
                    is_linear_resumed = bool(
                        (saved_values.get("metrics", {}) or {}).get("is_linear_estimate", True)
                    )
                    pool_gallery_images.clear()
                    pool_gallery_images.extend(
                        _variant_gallery_items(saved_pool, wd, is_linear_resumed)
                    )
                    review_variants = _proposal_variants(saved_proposal)
                    gallery_images.clear()
                    gallery_images.extend(
                        _variant_gallery_items(review_variants, wd, is_linear_resumed)
                    )
                    state["working_dir"] = wd
            except Exception as e:
                logger.warning(
                    f"resume_session: variant preview generation failed (non-fatal): {e}"
                )

        chat_messages.append({
            "role": "assistant",
            "content": (
                "**Session paused at an HITL gate.** Review the presented "
                "candidate(s), or use the pool filmstrip for context/debugging. "
                "Approve a presented candidate or send feedback to iterate. "
                "Pasting the resume thread does not auto-advance — your "
                "response is what closes the gate."
            ),
        })
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state
        return

    # Not paused at a gate — safe to stream a continuation. Empty resume
    # value is inert: it doesn't get treated as approval or feedback by
    # any hitl_check path because no interrupt is currently waiting.
    try:
        async for chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop in _stream_graph(
            _GRAPH, config, Command(resume=""),
            chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal,
            state.get("working_dir", ""), state.get("is_linear", True),
        ):
            yield chat_msgs, act_log, gal_imgs, pool_gal_imgs, vpool, vprop, state
    except Exception as e:
        logger.exception(f"Resume error: {e}")
        chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state


async def _check_resume_diffs(
    resume_id: str,
    chat_messages: list[dict],
    activity_log: list[dict],
    gallery_images: list[tuple],
    pool_gallery_images: list[tuple],
    variant_pool: list[dict],
    proposal: list[dict],
    state: dict,
):
    """
    Check for settings diffs before resuming. If diffs found, show warning
    and make Confirm Resume button visible. If no diffs, proceed with resume.

    Yields: (chat, activity, review_gallery, pool_gallery, variant_pool, proposal, state, confirm_btn_update)
    """
    if gallery_images is None:
        gallery_images = []
    if pool_gallery_images is None:
        pool_gallery_images = []
    if variant_pool is None:
        variant_pool = []
    if proposal is None:
        proposal = []
    confirm_hidden = gr.update(visible=False)
    confirm_visible = gr.update(visible=True)

    if not resume_id.strip():
        chat_messages.append({"role": "assistant", "content": "Please enter a thread ID to resume."})
        yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, confirm_hidden
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
            yield chat_messages, activity_log, gallery_images, pool_gallery_images, variant_pool, proposal, state, confirm_visible
            return

    # No diffs (or no snapshot found) — proceed directly
    async for result in resume_session(
        resume_id, chat_messages, activity_log, gallery_images,
        pool_gallery_images, variant_pool, proposal, state,
    ):
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


def _build_settings_snapshot() -> dict:
    """Capture current runtime settings into a dict for later diff comparison."""
    from muphrid.graph.hitl import (
        is_autonomous, vlm_autonomous,
        vlm_window_cap, TOOL_TO_HITL, is_enabled,
    )
    return {
        "model": os.environ.get("LLM_MODEL", ""),
        "autonomous": is_autonomous(),
        # vlm_hitl is always True (collaboration requires it). Kept in the
        # snapshot dict for display.
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


# Session-index helpers moved to muphrid.sessions so CLI and Gradio share them.
from muphrid.sessions import (
    register_session as _register_session,
    lookup_session_dir as _lookup_session_dir,
)


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

    # CSS: pulsing animation for the "Agent is working" banner. The pulse is
    # the visual signal that the agent is alive and iterating, not the UI
    # being frozen. Background-color modulation reads better than opacity at
    # a glance because the surrounding chrome stays at full contrast.
    blocks_css = """
    @keyframes muphrid-working-pulse {
        0%, 100% { background-color: rgba(96, 156, 247, 0.08); }
        50%      { background-color: rgba(96, 156, 247, 0.28); }
    }
    .muphrid-working-banner {
        animation: muphrid-working-pulse 1.6s ease-in-out infinite;
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        border: 1px solid rgba(96, 156, 247, 0.35);
        margin-bottom: 0.4rem;
    }
    """

    # NOTE on css placement: in Gradio 6.0+, `css` was moved from
    # gr.Blocks() to launch(). build_app() returns the Blocks object;
    # the caller (main()) is responsible for passing css= to launch().
    # We attach the CSS as an attribute on the returned Blocks so
    # main() can pick it up without having to import blocks_css.
    with gr.Blocks(title="Muphrid") as app:
        app._muphrid_css = blocks_css  # consumed by main() at launch time
        session_state = gr.State({
            "thread_id": None,
            "config": None,
            "working_dir": "",
            "is_linear": True,
        })
        # Per-session variant pool — passive history of every variant the
        # agent has produced this segment. Drives the read-only "Pool"
        # panel; the agent does not have to call any tool to populate it
        # (variant_snapshot in the graph builds it automatically).
        variant_pool_state = gr.State([])
        # Per-session proposal — the agent's curated subset, populated only
        # when the agent calls the `present_for_review` tool. Drives the
        # Approve buttons. Each entry: {"variant": Variant, "rationale":
        # str, "presented_at": str}. Empty until the agent surfaces
        # candidates; the panel shows an explicit empty-state message in
        # that case rather than auto-populating from the pool.
        proposal_state = gr.State([])
        # Boolean flag tracking whether a graph stream is currently in flight.
        # Drives the working-banner visibility and msg_input.interactive lock.
        # Each async-generator handler sets True on entry, False on exit
        # (including the except path). See render_streaming_state below for
        # the reactive bindings.
        is_streaming_state = gr.State(False)

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
                        value=[],  # explicit empty list — Gradio 6 Chatbot uses messages format
                        height=600,
                        buttons=["copy", "copy_all"],
                    )
                    # Pulsing "agent is working" banner. Lives BETWEEN the
                    # message list and the input textbox so it's visually
                    # adjacent to the control it locks. This matches the
                    # Cursor / Claude.app pattern: the "thinking…" indicator
                    # appears just above the input, where the user's eyes
                    # already are when they're about to type. A banner at the
                    # top of the column reads as a header and gets ignored
                    # while the user looks at the bottom for the typing
                    # affordance — exactly the wrong place to put the
                    # "stop, controls are locked" signal.
                    working_banner = gr.Markdown(
                        "🔄 **Agent is working** — controls are locked while it iterates. "
                        "Tool activity is shown on the right.",
                        visible=False,
                        elem_classes=["muphrid-working-banner"],
                    )
                    # ── Chat input ──────────────────────────────────────
                    # Design notes (every detail here is a fix for an actual bug
                    # we observed in HITL sessions, don't simplify without a
                    # replacement plan):
                    #
                    #  * lines=2 + submit_btn=True: with lines=1 the textbox
                    #    auto-submits on Enter, which silently consumed
                    #    rationale-in-progress when the user expected Enter to
                    #    insert a newline. Pairing lines>=2 with an explicit
                    #    submit button matches gr.ChatInterface and the
                    #    Cursor / Claude-coding-agent style: Enter for newline,
                    #    button (or Shift+Enter) to send.
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder=(
                                "Ask a question or request another iteration. "
                                "Approval happens in the Proposal panel."
                                " Shift+Enter or click Send."
                            ),
                            lines=2,
                            max_lines=20,
                            show_label=False,
                            container=False,
                            autoscroll=True,
                            submit_btn=True,
                            scale=10,
                        )

                with gr.Column(scale=1):
                    gallery = gr.Gallery(
                        label="HITL Review / Presented Candidates",
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

            with gr.Accordion("Agent Workbench / Pool", open=False):
                pool_gallery = gr.Gallery(
                    label="Pool Filmstrip",
                    columns=8,
                    height=170,
                    allow_preview=True,
                    buttons=["download"],
                    object_fit="contain",
                    interactive=False,
                )
                gr.Markdown(
                    "*Live workbench history for transparency/debugging. "
                    "Images here are not approvable until the agent presents them for review.*"
                )

                # ── Pool details (read-only history) ──────────────────────────
                # Every variant the agent has produced this segment, listed with
                # its label and key metrics. NO Approve buttons here — this is
                # status, not a control surface. The user can refer to variant
                # ids in chat when asking questions or requesting iterations, but
                # approval is only available from the Proposal panel.
                #
                # The panel populates passively as the agent runs HITL-mapped
                # tools; nothing the agent does affects this panel directly.
                @gr.render(inputs=variant_pool_state)
                def render_pool_panel(variants):
                    if not variants:
                        gr.Markdown(
                            "*No variants yet this segment. "
                            "The pool fills automatically as the agent iterates.*"
                        )
                        return
                    gr.Markdown(f"*{len(variants)} variant{'s' if len(variants) != 1 else ''} in the current workbench.*")
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
                            metric_bits.append(
                                f"{short}={val:.3f}" if isinstance(val, float) else f"{short}={val}"
                            )
                        metric_str = " · ".join(metric_bits)
                        desc = f"**{vid}** — {label}"
                        if metric_str:
                            desc += f"  \n*{metric_str}*"
                        gr.Markdown(desc)

            # ── Proposal panel (the agent's curation, with Approve buttons) ──
            # Driven by `proposal_state`, which the streaming layer
            # populates from the ReviewSession proposal artifact.
            #
            # Empty until the agent calls present_for_review. The
            # empty-state message is the explicit signal that the agent
            # has not yet curated a candidate — the human's path forward
            # is to chat with the agent ("which one do you recommend?")
            # rather than reach for a UI shortcut.
            #
            # Each entry is a dict {"variant": Variant, "rationale": str,
            # "presented_at": str}. Per-variant attribution: each variant
            # shows the rationale it was presented with, so the agent's
            # iterative reasoning is visible alongside each candidate.
            #
            # Approval controls live outside this dynamic render block.
            # The render block is display-only; the fixed controls below
            # avoid stale Gradio fn_index handlers when proposal_state redraws.
            with gr.Group():
                @gr.render(inputs=[proposal_state, is_streaming_state])
                def render_proposal_panel(proposal, streaming):
                    if streaming:
                        gr.Markdown(
                            "### Proposal\n*Agent is iterating — proposals will "
                            "be available for approval when it pauses.*"
                        )
                        return
                    if not proposal:
                        gr.Markdown(
                            "### Proposal\n*Agent hasn't surfaced anything for "
                            "review at this gate yet. The workbench filmstrip "
                            "shows what it's been working on; chat with the "
                            "agent to ask it to present candidates.*"
                        )
                        return
                    gr.Markdown(
                        f"### Proposal ({len(proposal)} for review)\n"
                        f"*Send questions or revision requests in chat. To "
                        f"approve, use the stable approval controls below.*"
                    )
                    artifact_entry = proposal[0] if proposal else {}
                    proposal_rationale = (
                        artifact_entry.get("proposal_rationale", "") or ""
                    ).strip()
                    recommendation = artifact_entry.get("recommendation")
                    tradeoffs = artifact_entry.get("tradeoffs", []) or []
                    metric_highlights = artifact_entry.get("metric_highlights", {}) or {}
                    summary_lines = []
                    if recommendation:
                        summary_lines.append(f"**Recommended:** `{recommendation}`")
                    if proposal_rationale:
                        summary_lines.append(f"> {proposal_rationale}")
                    if tradeoffs:
                        summary_lines.append("**Tradeoffs**")
                        summary_lines.extend(f"- {t}" for t in tradeoffs)
                    if metric_highlights:
                        bits = [f"{k}={v}" for k, v in metric_highlights.items()]
                        if bits:
                            summary_lines.append(
                                f"**Proposal metrics:** {' · '.join(bits)}"
                            )
                    if summary_lines:
                        gr.Markdown("\n".join(summary_lines))
                    for entry in proposal:
                        v = entry.get("variant", {}) or {}
                        rationale = entry.get("rationale", "") or ""
                        recommendation = entry.get("recommendation")
                        vid = v.get("id", "?")
                        label = v.get("label", "")
                        metrics = v.get("metrics", {}) or {}
                        metric_bits = []
                        for key in ("gradient_magnitude", "snr_estimate", "signal_coverage_pct"):
                            val = metrics.get(key)
                            if val is None:
                                continue
                            short = key.split("_")[0]
                            metric_bits.append(
                                f"{short}={val:.3f}" if isinstance(val, float) else f"{short}={val}"
                            )
                        metric_str = " · ".join(metric_bits)
                        with gr.Row():
                            desc = f"**{vid}** — {label}"
                            if metric_str:
                                desc += f"  \n*{metric_str}*"
                            if rationale:
                                desc += f"  \n> {rationale}"
                            if recommendation == vid:
                                desc += "  \n`Recommended`"
                            gr.Markdown(desc)

            with gr.Group():
                gr.Markdown("### Approval")
                approval_candidate = gr.Dropdown(
                    label="Presented candidate",
                    choices=[],
                    value=None,
                    interactive=False,
                )
                approval_note = gr.Textbox(
                    label="Approval note (optional)",
                    placeholder=(
                        "Why this candidate is approved, e.g. best gradient "
                        "balance without clipping. This is recorded with the "
                        "approval action, not sent as chat feedback."
                    ),
                    lines=2,
                    max_lines=8,
                    interactive=False,
                )
                approve_candidate_btn = gr.Button(
                    "Approve selected candidate",
                    variant="primary",
                    interactive=False,
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
        llm_model.change(fn=_format_model_info, inputs=[llm_model], outputs=[model_info])

        # Wire all HITL checkboxes — each calls set_hitl_tool_enabled on change
        from muphrid.graph.hitl import set_hitl_tool_enabled
        for hitl_key, cb in _hitl_checkboxes.items():
            cb.change(
                fn=lambda v, k=hitl_key: set_hitl_tool_enabled(k, v),
                inputs=[cb],
            )

        # Reactive: when is_streaming flips, toggle the working banner and
        # lock/unlock the textbox in lockstep. Doing this via a .change()
        # listener (rather than threading the streaming flag through every
        # handler signature) keeps the streaming handlers simple — they just
        # flip the flag True on entry and False on exit via .then() chains.
        is_streaming_state.change(
            fn=lambda streaming: (
                gr.update(visible=bool(streaming)),
                gr.update(interactive=not bool(streaming)),
            ),
            inputs=[is_streaming_state],
            outputs=[working_banner, msg_input],
        )
        proposal_state.change(
            fn=update_approval_controls,
            inputs=[proposal_state, is_streaming_state],
            outputs=[approval_candidate, approval_note, approve_candidate_btn],
        )
        is_streaming_state.change(
            fn=update_approval_controls,
            inputs=[proposal_state, is_streaming_state],
            outputs=[approval_candidate, approval_note, approve_candidate_btn],
        )

        # Helpers to flip the streaming flag at the start and end of each
        # streaming binding via .then() chains. The async generators
        # themselves don't need to know about the flag — they just run.
        # If a generator errors, Gradio still fires the trailing .then()
        # with the False payload, so the UI never gets stuck "locked".
        _stream_start = lambda: True       # noqa: E731
        _stream_end = lambda: False        # noqa: E731

        # Start session: apply settings → flip streaming on → stream → flip off
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
            fn=_stream_start, inputs=None, outputs=[is_streaming_state],
        ).then(
            fn=start_session,
            inputs=[
                dataset_path, target_name, bortle_input, sqm_input,
                remove_stars_input, notes_input,
                pixel_size, sensor_type, focal_length,
                session_state,
            ],
            outputs=[chatbot, activity, gallery, pool_gallery, variant_pool_state, proposal_state, session_state],
        ).then(
            fn=_stream_end, inputs=None, outputs=[is_streaming_state],
        )

        # User feedback during HITL: lock controls → run handler → unlock.
        msg_input.submit(
            fn=_stream_start, inputs=None, outputs=[is_streaming_state],
        ).then(
            fn=send_message,
            inputs=[msg_input, chatbot, activity, gallery, pool_gallery, variant_pool_state, proposal_state, session_state],
            outputs=[chatbot, activity, gallery, pool_gallery, variant_pool_state, proposal_state, session_state, msg_input],
        ).then(
            fn=_stream_end, inputs=None, outputs=[is_streaming_state],
        )

        # Bare Approve button is removed — approval is variant-specific via the
        # stable Proposal approval controls above. The proposal-local approval
        # note carries the rationale; chat remains feedback/questions only.
        approve_candidate_btn.click(
            fn=_stream_start, inputs=None, outputs=[is_streaming_state],
        ).then(
            fn=approve_selected_variant_action,
            inputs=[
                approval_candidate,
                approval_note,
                chatbot,
                activity,
                gallery,
                pool_gallery,
                variant_pool_state,
                proposal_state,
                session_state,
            ],
            outputs=[
                chatbot,
                activity,
                gallery,
                pool_gallery,
                variant_pool_state,
                proposal_state,
                session_state,
                approval_note,
            ],
        ).then(
            fn=_stream_end, inputs=None, outputs=[is_streaming_state],
        )

        # Resume: apply settings, check diffs, warn if changed
        _apply_inputs = [
            recursion_limit, max_tools_phase, max_consecutive, max_nudges,
            max_silent_hitl,
            phase_linear, phase_stretch, phase_nonlinear,
            cleanup_runs, prune_analysis,
            llm_model,
        ]

        # Resume (with diff-check pre-flight): apply settings → check diffs.
        # The diff-check itself is fast and doesn't stream, so we don't lock
        # controls for it. The actual stream happens on confirm.
        resume_btn.click(
            fn=_apply_ui_settings,
            inputs=_apply_inputs,
        ).then(
            fn=_check_resume_diffs,
            inputs=[resume_id, chatbot, activity, gallery, pool_gallery, variant_pool_state, proposal_state, session_state],
            outputs=[chatbot, activity, gallery, pool_gallery, variant_pool_state, proposal_state, session_state, confirm_resume_btn],
        )

        # Confirm Resume: lock controls → stream → unlock.
        confirm_resume_btn.click(
            fn=_apply_ui_settings,
            inputs=_apply_inputs,
        ).then(
            fn=_stream_start, inputs=None, outputs=[is_streaming_state],
        ).then(
            fn=resume_session,
            inputs=[resume_id, chatbot, activity, gallery, pool_gallery, variant_pool_state, proposal_state, session_state],
            outputs=[chatbot, activity, gallery, pool_gallery, variant_pool_state, proposal_state, session_state],
        ).then(
            fn=_stream_end, inputs=None, outputs=[is_streaming_state],
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
    # Merge build_app's component-level CSS (e.g. the working-banner pulse
    # animation) with main()'s layout CSS. In Gradio 6.0+, css MUST be
    # passed to launch() — the gr.Blocks(css=...) constructor parameter
    # was removed.
    layout_css = """
        .gradio-container { max-width: 100% !important; padding: 0 0.5rem !important; margin: 0 !important; }
        .main { max-width: 100% !important; }
        .contain { max-width: 100% !important; }
    """
    component_css = getattr(app, "_muphrid_css", "") or ""
    app.launch(
        theme=gr.themes.Soft(primary_hue="blue"),
        allowed_paths=["/"],  # datasets can be anywhere on disk
        css=layout_css + "\n" + component_css,
    )


if __name__ == "__main__":
    main()
