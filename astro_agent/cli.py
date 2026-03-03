"""
Entry point — Phase 10.

Stub only. Wired in Phase 10 after the graph and HITL presenter are complete.
"""

import typer

app = typer.Typer(help="AstroAgent: autonomous astrophotography post-processing.")


@app.command()
def process(
    directory: str = typer.Argument(..., help="Path to dataset directory."),
    profile: str = typer.Option("balanced", help="conservative | balanced | aggressive"),
    resume: str = typer.Option("", help="Thread ID to resume from checkpoint."),
    no_hitl: bool = typer.Option(False, "--no-hitl", help="Auto-select HITL options."),
) -> None:
    """Process a dataset from raw frames to final export."""
    typer.echo("Phase 10 not yet implemented. Build Phases 2–9 first.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
