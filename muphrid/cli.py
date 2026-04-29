"""
CLI entry point — connects the LangGraph pipeline to the command line.

Usage:
    python -m muphrid.cli process /path/to/dataset --target "M42 Orion Nebula" --bortle 4
    python -m muphrid.cli process /path/to/dataset --resume "run-m42-20260311"
"""

from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
from datetime import datetime

import typer
from langchain_core.messages import HumanMessage
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from muphrid.config import check_dependencies, load_settings
from muphrid.graph.graph import build_graph
from muphrid.graph import review as review_ctl
from muphrid.graph.state import (
    AstroState,
    ProcessingPhase,
    SessionContext,
    build_initial_message,
    make_empty_state,
)
from muphrid.tools.preprocess.t01_ingest import ingest_dataset

app = typer.Typer(help="Muphrid: autonomous astrophotography post-processing.")

logger = logging.getLogger(__name__)


# Ordered phase list for "stop after phase N" comparisons.
_PHASE_ORDER: list[ProcessingPhase] = [
    ProcessingPhase.INGEST,
    ProcessingPhase.CALIBRATION,
    ProcessingPhase.REGISTRATION,
    ProcessingPhase.ANALYSIS,
    ProcessingPhase.STACKING,
    ProcessingPhase.LINEAR,
    ProcessingPhase.STRETCH,
    ProcessingPhase.NONLINEAR,
    ProcessingPhase.REVIEW,
    ProcessingPhase.EXPORT,
    ProcessingPhase.COMPLETE,
]

# Friendly aliases for phase groups. "preprocess" ends when stacking completes.
_PHASE_ALIASES: dict[str, ProcessingPhase] = {
    "preprocess":    ProcessingPhase.STACKING,
    "preprocessing": ProcessingPhase.STACKING,
}


def _resolve_stop_phase(name: str | None) -> ProcessingPhase | None:
    """Map a user-supplied phase name (or alias) to a ProcessingPhase."""
    if not name:
        return None
    key = name.lower().strip()
    if key in _PHASE_ALIASES:
        return _PHASE_ALIASES[key]
    try:
        return ProcessingPhase(key)
    except ValueError:
        valid = ", ".join(p.value for p in _PHASE_ORDER)
        alias_list = ", ".join(sorted(_PHASE_ALIASES))
        raise typer.BadParameter(
            f"Unknown phase '{name}'. Valid phases: {valid}. Aliases: {alias_list}."
        )


def _phase_index(p) -> int:
    """Index of a ProcessingPhase (or its string value) in _PHASE_ORDER."""
    if isinstance(p, ProcessingPhase):
        try:
            return _PHASE_ORDER.index(p)
        except ValueError:
            return -1
    if isinstance(p, str):
        try:
            return _PHASE_ORDER.index(ProcessingPhase(p))
        except ValueError:
            return -1
    return -1


@app.callback()
def _cli_callback() -> None:
    """
    Muphrid CLI.

    An empty callback forces Typer to treat this as a multi-command app so
    the docstring's stated usage — ``python -m muphrid.cli process ...`` —
    actually works. Without it, a single-command Typer app collapses to
    ``python -m muphrid.cli ...`` and rejects the subcommand name as an
    extra argument.
    """


# Smart-quote → straight-quote map. Terminals (macOS "Smart Quotes" setting,
# and some copy-paste flows) silently substitute U+2018/U+2019/U+201C/U+201D
# into user input. These characters break cfitsio FITS path parsing, Siril's
# working-dir argument handling, and thread-id URL slugs. Normalize them to
# ASCII before anything sees the string.
_SMART_QUOTE_MAP = str.maketrans({
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201f": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2013": "-",  # EN DASH
    "\u2014": "-",  # EM DASH
    "\u2212": "-",  # MINUS SIGN
    "\u00a0": " ",  # NO-BREAK SPACE
})

# Characters the slug is allowed to contain. Anything else is dropped.
# Stays filesystem-safe on macOS/Linux/Windows and URL-safe for thread ids.
_SLUG_ALLOWED = re.compile(r"[^a-z0-9_\-]+")


def _sanitize_target(raw: str) -> tuple[str, list[str]]:
    """
    Canonicalize a user-supplied target name.

    Returns (clean_target, notes). `clean_target` is the string that should be
    used downstream (display, SIMBAD lookup, session context). `notes` is a list
    of human-readable warnings to surface via typer.echo when sanitization
    changed the input.

    Steps:
      1. Translate smart quotes / typographic dashes / NBSP to ASCII equivalents.
      2. NFKC-normalize so visually-identical compatibility characters collapse.
      3. Strip surrounding quotes the shell left behind (rare, but cheap to check).
      4. Collapse internal whitespace.

    No character class filtering is done on the display value — we want to keep
    things like apostrophes in "Bode's Galaxy" or "47 Tucanae". Filesystem-safe
    slug generation is a separate step (see `_make_thread_id`).
    """
    notes: list[str] = []
    original = raw

    translated = raw.translate(_SMART_QUOTE_MAP)
    if translated != raw:
        notes.append(
            "target: translated non-ASCII typographic characters "
            "(smart quotes / em-dashes / NBSP) to ASCII equivalents"
        )

    normalized = unicodedata.normalize("NFKC", translated)
    if normalized != translated:
        notes.append("target: applied NFKC Unicode normalization")

    # Strip one layer of leading/trailing matched quotes (terminal sometimes
    # preserves them on paste; Typer does not strip).
    stripped = normalized.strip()
    for pair in ('""', "''"):
        if len(stripped) >= 2 and stripped[0] == pair[0] and stripped[-1] == pair[-1]:
            stripped = stripped[1:-1].strip()
            notes.append("target: removed surrounding quote characters")
            break

    # Collapse internal whitespace runs to single spaces.
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    if collapsed != stripped:
        notes.append("target: collapsed internal whitespace")

    if not collapsed:
        raise typer.BadParameter(
            f"--target resolved to empty string after sanitization "
            f"(original: {original!r}). Provide a non-empty target name."
        )

    return collapsed, notes


def _make_thread_id(target: str) -> str:
    """
    Build a filesystem- and URL-safe thread id from a target display name.

    The slug is derived by:
      - lowercasing,
      - replacing whitespace with '-',
      - ASCII-folding (NFKD decompose, drop non-ASCII combining chars),
      - dropping anything outside ``[a-z0-9_-]``.

    The slug is then capped at 30 chars and suffixed with a timestamp so
    each run has a unique, filesystem-safe, ASCII-only thread id. Any thread
    id used with SqliteSaver and LangGraph's checkpoint system eventually
    becomes part of a filesystem path (run output dir) — non-ASCII slugs here
    have historically caused cfitsio/Siril path failures downstream, which is
    why we sanitize aggressively even though the display target name is kept.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # NFKD decompose, then drop anything that isn't ASCII. This converts
    # accented letters ("Mü" → "Mu") and strips combining marks.
    ascii_fold = (
        unicodedata.normalize("NFKD", target)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    lowered = ascii_fold.lower().replace(" ", "-")
    cleaned = _SLUG_ALLOWED.sub("", lowered)
    # Collapse repeated dashes and strip leading/trailing dashes so the
    # final thread id reads cleanly in logs and shell output.
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")

    if not cleaned:
        cleaned = "target"  # last-ditch fallback so the thread id is never empty

    slug = cleaned[:30].rstrip("-") or "target"
    thread_id = f"run-{slug}-{ts}"

    # Belt-and-suspenders: verify the result is pure ASCII before returning.
    # If this ever fails it means the slug-generation logic above has a bug,
    # not that the input was malicious — raise loudly rather than silently
    # corrupt downstream paths.
    if not thread_id.isascii():
        raise RuntimeError(
            f"_make_thread_id produced non-ASCII thread_id: {thread_id!r}. "
            f"This is a slug-generation bug, not a user-input issue."
        )
    return thread_id


@app.command()
def process(
    directory: str = typer.Argument(..., help="Path to dataset directory."),
    target: str = typer.Option(..., help="Target name (e.g. 'M42 Orion Nebula')."),
    bortle: int = typer.Option(None, help="Bortle scale of imaging site (1-9). If unknown, omit."),
    sqm: float = typer.Option(None, help="SQM-L reading in mag/arcsec²."),
    remove_stars: bool = typer.Option(None, help="Run star removal/restoration. None=ask via HITL."),
    notes: str = typer.Option(None, help="Free-text session notes."),
    resume: str = typer.Option("", help="Thread ID to resume from checkpoint."),
    autonomous: bool = typer.Option(False, "--autonomous", help="Skip all HITL interrupts."),
    stop_after_phase: str = typer.Option(
        None,
        "--stop-after-phase",
        help=(
            "Halt once the pipeline advances past the named phase. Accepts a "
            "ProcessingPhase value (e.g. 'stacking', 'linear') or the alias "
            "'preprocess' (= stop after stacking). Checkpoint is preserved on "
            "halt; resume the run with --resume <thread_id>."
        ),
    ),
    db: str = typer.Option("checkpoints.db", help="Path to SQLite checkpoint database."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Process a dataset from raw frames to final export."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Sanitize --target once, up front. The cleaned value replaces the raw
    # CLI argument so every downstream consumer (thread_id slug, session
    # context, SIMBAD lookup, phase reports) sees a canonical ASCII-friendly
    # string. Terminals that auto-substitute straight quotes into smart
    # quotes have historically poisoned run directories and Siril path args;
    # this catches that and any future variant (em-dashes, NBSP, etc.)
    # before it can cascade.
    clean_target, target_notes = _sanitize_target(target)
    for note in target_notes:
        typer.echo(f"Warning: {note}", err=True)
    if clean_target != target:
        typer.echo(f"Target normalized: {target!r} → {clean_target!r}", err=True)
    target = clean_target

    # Set autonomous mode before anything else
    if autonomous:
        from muphrid.graph.hitl import set_autonomous
        set_autonomous(True)

    # Resolve stop-after-phase early so we can typer.BadParameter before work starts.
    stop_phase = _resolve_stop_phase(stop_after_phase)
    if stop_phase is not None:
        typer.echo(f"Stop-after-phase: {stop_phase.value}")

    # Validate dependencies
    settings = load_settings()
    check_dependencies(settings)

    # Build graph
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("muphrid.graph.state", "ProcessingPhase")])
    checkpointer = SqliteSaver(sqlite3.connect(db, check_same_thread=False), serde=serde)
    graph = build_graph(checkpointer=checkpointer)

    # Thread config
    thread_id = resume if resume else _make_thread_id(target)
    # Fall back to processing.toml [limits].recursion_limit (200) when the
    # env var is unset. Previously this defaulted to 0 (= LangGraph's
    # built-in 25), which silently throttled the CLI below the Gradio app.
    from muphrid.config import _pcfg
    env_recursion = os.environ.get("RECURSION_LIMIT", "")
    recursion_limit = int(env_recursion) if env_recursion else int(_pcfg("limits", "recursion_limit", 200))
    config = {"configurable": {"thread_id": thread_id}}
    if recursion_limit > 0:
        config["recursion_limit"] = recursion_limit
    typer.echo(f"Thread ID: {thread_id}")

    if resume:
        # Resume from existing checkpoint
        typer.echo(f"Resuming from checkpoint: {resume}")
        _run_graph(
            graph, config,
            resume=True, autonomous=autonomous, stop_phase=stop_phase,
        )
    else:
        # New run — build initial state
        session = {
            "target_name": target,
            "bortle": bortle,
            "sqm_reading": sqm,
            "remove_stars": remove_stars,
            "notes": notes,
        }

        # T01 ingest runs first to discover the dataset
        typer.echo(f"Ingesting dataset from: {directory}")
        typer.echo(f"Run output dir: {directory}/runs/{thread_id}/")
        ingest_result = ingest_dataset.invoke({
            "root_directory": directory,
            "thread_id": thread_id,
        })

        for warning in ingest_result.get("warnings", []):
            typer.echo(f"Warning: {warning}", err=True)

        # Register the session in the shared sessions index so a later
        # Gradio resume of this thread_id can recover the working_dir
        # without falling back to deriving it from variant file paths.
        from muphrid.sessions import register_session
        working_dir_for_index = ingest_result["dataset"].get("working_dir", "")
        if working_dir_for_index:
            register_session(thread_id, working_dir_for_index)

        initial_state = make_empty_state(
            dataset=ingest_result["dataset"],
            session=session,
        )

        # Add initial instruction with full dataset context
        initial_state["messages"] = [
            HumanMessage(content=build_initial_message(
                dataset=ingest_result["dataset"],
                session=session,
                ingest_summary=ingest_result.get("summary", {}),
            ))
        ]

        _run_graph(
            graph, config,
            initial_state=initial_state, autonomous=autonomous, stop_phase=stop_phase,
        )


def _run_graph(
    graph, config,
    initial_state=None, resume=False, autonomous=False,
    stop_phase: ProcessingPhase | None = None,
):
    """
    Run the graph, handling HITL interrupts interactively.

    When interrupt() fires, we present the payload to the user via CLI
    and collect their response. The response is passed back via
    Command(resume=response).

    If ``stop_phase`` is set, the loop halts as soon as the state advances
    past that phase. We detect this by sniffing ``phase`` updates emitted
    by ``advance_phase``; checkpoint state up through that tool is fully
    persisted, so the run can be resumed with ``--resume <thread_id>``.
    """
    # First invocation
    if resume:
        # Resume with empty command to pick up where we left off
        stream_input = Command(resume="")
    else:
        stream_input = initial_state

    stop_idx = _phase_index(stop_phase) if stop_phase is not None else None

    while True:
        interrupt_payload = None
        stop_triggered = False

        for chunk in graph.stream(stream_input, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupt_payload = chunk["__interrupt__"][0].value
                # No break — LangGraph ends the stream naturally after the interrupt
                # chunk. Breaking early calls generator.close() which throws
                # GeneratorExit into LangGraph's cleanup code, propagating as a
                # crash and poisoning LangSmith traces for all tool calls.
                continue

            # Sniff node updates for a phase transition past the stop point.
            # Each chunk in stream_mode="updates" is {node_name: updates_dict}.
            if stop_idx is not None and not stop_triggered:
                for node_updates in chunk.values():
                    if not isinstance(node_updates, dict):
                        continue
                    new_phase = node_updates.get("phase")
                    if new_phase is None:
                        continue
                    if _phase_index(new_phase) > stop_idx:
                        stop_triggered = True
                        new_name = (
                            new_phase.value
                            if isinstance(new_phase, ProcessingPhase)
                            else str(new_phase)
                        )
                        typer.echo(
                            f"\n[stop-after-phase] Advanced to '{new_name}' "
                            f"(past '{stop_phase.value}'). Halting stream."
                        )
                        break

            if stop_triggered:
                # Close the stream. LangGraph raises GeneratorExit into its
                # running super-step — the checkpoint for all completed nodes
                # (including advance_phase) is already durable, so
                # `--resume <thread_id>` picks up cleanly at the new phase.
                # The LangSmith trace for this super-step may be incomplete.
                break

        if stop_triggered:
            thread_id = config.get("configurable", {}).get("thread_id", "?")
            typer.echo(
                f"\nStopped after phase '{stop_phase.value}'. "
                f"Resume with:  python -m muphrid.cli process ... --resume {thread_id}"
            )
            break

        if interrupt_payload is None:
            # Graph completed without interrupt
            typer.echo("\nPipeline complete.")
            break

        interrupt_type = interrupt_payload.get("type", "unknown")

        if interrupt_type == "flag_dataset_issue":
            # Agent-initiated escape hatch. Fires regardless of autonomous
            # mode — this is the documented exception. CLI behavior is
            # mode-aware: unattended runs exit cleanly with a non-zero
            # status so the user returns to a clear signal; attended runs
            # prompt and resume.
            phase = interrupt_payload.get("phase", "?")
            reason = interrupt_payload.get("reason", "(no reason given)")
            current_image = interrupt_payload.get("current_image")

            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"DATASET ISSUE FLAGGED — {phase} phase")
            typer.echo(f"{'=' * 60}")
            typer.echo(f"\n{reason}\n")
            if current_image:
                typer.echo(f"Working image at flag time: {current_image}")
            metrics_snap = interrupt_payload.get("metrics_snapshot") or {}
            if metrics_snap:
                typer.echo("\nMetrics at flag time:")
                for k, v in metrics_snap.items():
                    typer.echo(f"  {k}: {v}")
            typer.echo(f"\n{'=' * 60}")

            if autonomous:
                # Unattended run: exit cleanly. Dataset state is on disk;
                # processing_log.md has the flag; the run can be resumed
                # from the same thread_id once the human has decided how
                # to proceed (open Gradio for collaboration, fix the data
                # manually, etc.).
                thread_id = config.get("configurable", {}).get("thread_id", "?")
                typer.echo(
                    f"\n[autonomous] Halting on flagged issue. "
                    f"Thread '{thread_id}' is preserved on disk. "
                    f"Open Gradio or re-run with --resume to continue."
                )
                raise typer.Exit(code=2)

            # Attended CLI: prompt for response, resume normally.
            response = typer.prompt(
                "Your response (or instructions to the agent)"
            ).strip()
            stream_input = Command(resume=response)
            continue

        if interrupt_type == "agent_chat":
            # Agent wants to talk — show its message and get human response
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"Agent ({interrupt_payload.get('phase', '?')} phase):")
            typer.echo(interrupt_payload.get("agent_text", "(no text)"))
            typer.echo(f"{'=' * 60}")

            if autonomous:
                # Should not happen — agent_chat node handles autonomous mode
                # internally. But just in case:
                typer.echo("[autonomous] Auto-responding with nudge.")
                response = (
                    "Either call a tool to continue processing, or call "
                    "advance_phase to move to the next phase. Do not respond "
                    "with text without calling a tool."
                )
            else:
                response = typer.prompt("Your response").strip()
        else:
            # HITL gate (tool review)
            typer.echo(f"\n{'=' * 60}")
            typer.echo(f"HITL: {interrupt_payload.get('title', 'Review')}")
            typer.echo(f"Type: {interrupt_type}")
            typer.echo(f"Tool: {interrupt_payload.get('tool_name', 'unknown')}")
            if interrupt_payload.get("review_state") == "needs_curation":
                typer.echo(
                    "Review state: agent still needs to select and present "
                    "candidate(s) before approval is available."
                )

            images = interrupt_payload.get("images", [])
            if images:
                typer.echo(f"Images: {images}")

            proposal = interrupt_payload.get("proposal", []) or []
            proposal_ids: list[str] = []
            if proposal:
                typer.echo("\nPresented candidates:")
                for entry in proposal:
                    variant = entry.get("variant", {}) if isinstance(entry, dict) else {}
                    vid = variant.get("id")
                    if not vid:
                        continue
                    proposal_ids.append(vid)
                    label = variant.get("label", "")
                    rationale = entry.get("rationale", "") if isinstance(entry, dict) else ""
                    typer.echo(f"  {vid}: {label}")
                    if rationale:
                        typer.echo(f"    {rationale}")

            typer.echo(f"{'=' * 60}")

            if autonomous:
                typer.echo("[autonomous] Auto-approving.")
                response = {"type": "approve_current", "text": "Auto-approve"}
            else:
                approval_allowed = interrupt_payload.get("approval_allowed", True)
                if approval_allowed and proposal_ids:
                    prompt = "approve <variant_id> or message"
                else:
                    prompt = "approve (a) or message" if approval_allowed else "message"
                response = typer.prompt(prompt).strip()
                if approval_allowed:
                    lowered = response.lower()
                    if proposal_ids:
                        parts = response.split()
                        if lowered in ("a", "approve") and len(proposal_ids) == 1:
                            response = review_ctl.approval_resume_event(proposal_ids[0])
                        elif len(parts) == 2 and parts[0].lower() in ("a", "approve"):
                            requested = parts[1]
                            if requested in proposal_ids:
                                response = review_ctl.approval_resume_event(requested)
                            else:
                                typer.echo(
                                    f"Variant '{requested}' is not in the presented set; "
                                    "treating input as feedback."
                                )
                    elif lowered in ("a", "approve"):
                        response = {"type": "approve_current", "text": "Approve current"}
                if isinstance(response, str):
                    response = review_ctl.feedback_resume_event(response)

        # Resume with human response
        stream_input = Command(resume=response)


if __name__ == "__main__":
    app()
