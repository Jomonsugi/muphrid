"""
CLI entry point — connects the LangGraph pipeline to the command line.

Usage:
    python -m astro_agent.cli process /path/to/dataset --target "M42 Orion Nebula" --bortle 4
    python -m astro_agent.cli process /path/to/dataset --resume "run-m42-20260311"
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import typer
from langchain_core.messages import HumanMessage
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from astro_agent.config import check_dependencies, load_settings
from astro_agent.graph.graph import build_graph
from astro_agent.graph.hitl import APPROVE_SENTINEL
from astro_agent.graph.memory import make_memory_store
from astro_agent.graph.state import (
    AstroState,
    SessionContext,
    build_initial_message,
    make_empty_state,
)
from astro_agent.tools.preprocess.t01_ingest import ingest_dataset

app = typer.Typer(help="AstroAgent: autonomous astrophotography post-processing.")

logger = logging.getLogger(__name__)


def _make_thread_id(target: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = target.lower().replace(" ", "-")[:30]
    return f"run-{slug}-{ts}"


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
    memory: bool = typer.Option(False, "--memory", help="Enable long-term memory (learns from HITL sessions)."),
    rebuild_embeddings: bool = typer.Option(False, "--rebuild-embeddings", help="Rebuild vector index after model/provider change."),
    db: str = typer.Option("checkpoints.db", help="Path to SQLite checkpoint database."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Process a dataset from raw frames to final export."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Set autonomous mode before anything else
    if autonomous:
        from astro_agent.graph.hitl import set_autonomous
        set_autonomous(True)

    # Initialize long-term memory if enabled
    if memory:
        from astro_agent.memory.embeddings import EmbeddingInitError, init_memory_system

        try:
            settings_pre = load_settings()
            rebuild = rebuild_embeddings or settings_pre.memory_rebuild_embeddings
            init_memory_system(settings_pre, rebuild_embeddings=rebuild)
            logger.info("Long-term memory enabled")
        except EmbeddingInitError as e:
            typer.echo(f"Error: Memory initialization failed.\n\n{e}", err=True)
            raise typer.Exit(code=1)

    # Validate dependencies
    settings = load_settings()
    check_dependencies(settings)

    # Build graph
    store = make_memory_store()
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("astro_agent.graph.state", "ProcessingPhase")])
    checkpointer = SqliteSaver(sqlite3.connect(db, check_same_thread=False), serde=serde)
    graph = build_graph(checkpointer=checkpointer, store=store)

    # Thread config
    thread_id = resume if resume else _make_thread_id(target)
    recursion_limit = int(os.environ.get("RECURSION_LIMIT", "0"))
    config = {"configurable": {"thread_id": thread_id}}
    if recursion_limit > 0:
        config["recursion_limit"] = recursion_limit
    typer.echo(f"Thread ID: {thread_id}")

    if resume:
        # Resume from existing checkpoint
        typer.echo(f"Resuming from checkpoint: {resume}")
        _run_graph(graph, config, resume=True, autonomous=autonomous)
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

        _run_graph(graph, config, initial_state=initial_state, autonomous=autonomous)


def _run_graph(graph, config, initial_state=None, resume=False, autonomous=False):
    """
    Run the graph, handling HITL interrupts interactively.

    When interrupt() fires, we present the payload to the user via CLI
    and collect their response. The response is passed back via
    Command(resume=response).
    """
    # First invocation
    if resume:
        # Resume with empty command to pick up where we left off
        stream_input = Command(resume="")
    else:
        stream_input = initial_state

    while True:
        interrupt_payload = None

        for chunk in graph.stream(stream_input, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupt_payload = chunk["__interrupt__"][0].value
                # No break — LangGraph ends the stream naturally after the interrupt
                # chunk. Breaking early calls generator.close() which throws
                # GeneratorExit into LangGraph's cleanup code, propagating as a
                # crash and poisoning LangSmith traces for all tool calls.

        if interrupt_payload is None:
            # Graph completed without interrupt
            typer.echo("\nPipeline complete.")
            break

        interrupt_type = interrupt_payload.get("type", "unknown")

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

            images = interrupt_payload.get("images", [])
            if images:
                typer.echo(f"Images: {images}")

            typer.echo(f"{'=' * 60}")

            if autonomous:
                typer.echo("[autonomous] Auto-approving.")
                response = APPROVE_SENTINEL
            else:
                response = typer.prompt("approve (a) or message").strip()
                if response.lower() in ("a", "approve"):
                    response = APPROVE_SENTINEL

        # Resume with human response
        stream_input = Command(resume=response)


if __name__ == "__main__":
    app()
