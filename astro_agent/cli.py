"""
CLI entry point — connects the LangGraph pipeline to the command line.

Usage:
    python -m astro_agent.cli process /path/to/dataset --target "M42 Orion Nebula" --bortle 4
    python -m astro_agent.cli process /path/to/dataset --resume "run-m42-20260311"
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

import typer
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from astro_agent.config import check_dependencies, load_settings
from astro_agent.graph.graph import build_graph
from astro_agent.graph.hitl import APPROVE_SENTINEL
from astro_agent.graph.memory import make_memory_store
from astro_agent.graph.state import (
    AstroState,
    SessionContext,
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
    bortle: int = typer.Option(5, help="Bortle scale of imaging site (1-9)."),
    sqm: float = typer.Option(None, help="SQM-L reading in mag/arcsec²."),
    remove_stars: bool = typer.Option(None, help="Run star removal/restoration. None=ask via HITL."),
    notes: str = typer.Option(None, help="Free-text session notes."),
    resume: str = typer.Option("", help="Thread ID to resume from checkpoint."),
    autonomous: bool = typer.Option(False, "--autonomous", help="Skip all HITL interrupts."),
    db: str = typer.Option("checkpoints.db", help="Path to SQLite checkpoint database."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Process a dataset from raw frames to final export."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Validate dependencies
    settings = load_settings()
    check_dependencies(settings)

    # Build graph
    store = make_memory_store()
    checkpointer = SqliteSaver.from_conn_string(db)
    graph = build_graph(checkpointer=checkpointer, store=store)

    # Thread config
    thread_id = resume if resume else _make_thread_id(target)
    config = {"configurable": {"thread_id": thread_id}}
    typer.echo(f"Thread ID: {thread_id}")

    if resume:
        # Resume from existing checkpoint
        typer.echo(f"Resuming from checkpoint: {resume}")
        _run_graph(graph, config, resume=True)
    else:
        # New run — build initial state
        session = SessionContext(
            target_name=target,
            bortle=bortle,
            sqm_reading=sqm,
            remove_stars=remove_stars,
            notes=notes,
        )

        # T01 ingest runs first to discover the dataset
        typer.echo(f"Ingesting dataset from: {directory}")
        ingest_result = ingest_dataset.invoke({"root_directory": directory})

        for warning in ingest_result.get("warnings", []):
            typer.echo(f"Warning: {warning}", err=True)

        initial_state = make_empty_state(
            dataset=ingest_result["dataset"],
            session=session,
        )

        # Add initial instruction
        initial_state["messages"] = [
            HumanMessage(content=f"Process the dataset for {target}. Begin preprocessing.")
        ]

        _run_graph(graph, config, initial_state=initial_state)


def _run_graph(graph, config, initial_state=None, resume=False):
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
                break

        if interrupt_payload is None:
            # Graph completed without interrupt
            typer.echo("\nPipeline complete.")
            break

        # Present HITL to user
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"HITL: {interrupt_payload.get('title', 'Review')}")
        typer.echo(f"Type: {interrupt_payload.get('type', 'unknown')}")
        typer.echo(f"Tool: {interrupt_payload.get('tool_name', 'unknown')}")

        images = interrupt_payload.get("images", [])
        if images:
            typer.echo(f"Images: {images}")

        typer.echo(f"{'=' * 60}")

        # HITL gate: approve to pass, anything else is conversation.
        # The agent handles feedback/questions/revision requests agentically.
        # The gate re-fires every time until explicit approval.
        response = typer.prompt("approve (a) or message").strip()
        if response.lower() in ("a", "approve"):
            response = APPROVE_SENTINEL

        # Resume with human response
        stream_input = Command(resume=response)


if __name__ == "__main__":
    app()
