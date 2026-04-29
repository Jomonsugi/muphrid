"""
Clone a LangGraph checkpoint thread for repeatable dev testing.

Typical workflow:

    # Create a clean seed once.
    uv run python -m muphrid.cli process /data/m42 --target "M42" \
      --autonomous --stop-after-phase preprocess

    # Fork a fresh Gradio-resumable test thread from that seed.
    uv run python scripts/clone_thread.py run-m42-... hitl-test-001

    # Later, fork from a historical phase boundary inside a longer run.
    uv run python scripts/clone_thread.py run-m42-full-... nonlinear-test-001 \
      --at-phase nonlinear

The clone reuses the source run's working directory and artifact paths. It
copies checkpoint state/message history, not files on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from muphrid.sessions import lookup_session_dir, register_session


DEFAULT_DB = "checkpoints.db"


@dataclass(frozen=True)
class CheckpointRow:
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    type: str | None
    checkpoint: bytes
    metadata: bytes | None


@dataclass(frozen=True)
class CheckpointSummary:
    row: CheckpointRow
    step: int | None
    source: str
    phase: str
    active_hitl: bool | None
    current_image: str | None
    messages_count: int
    next_nodes: tuple[str, ...]
    working_dir: str | None


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[("muphrid.graph.state", "ProcessingPhase")]
    )


def _phase_value(value: Any) -> str:
    return getattr(value, "value", str(value)) if value is not None else "(none)"


def _loads_metadata(blob: bytes | None) -> dict:
    if not blob:
        return {}
    try:
        return json.loads(blob.decode("utf-8") if isinstance(blob, bytes) else blob)
    except Exception:
        return {}


def _loads_checkpoint(row: CheckpointRow) -> dict:
    if row.type is None:
        raise ValueError(f"checkpoint {row.checkpoint_id} has no serializer type")
    return _serializer().loads_typed((row.type, row.checkpoint))


def _row_to_summary(row: CheckpointRow) -> CheckpointSummary:
    metadata = _loads_metadata(row.metadata)
    checkpoint = _loads_checkpoint(row)
    values = checkpoint.get("channel_values", {}) or {}
    dataset = values.get("dataset", {}) or {}
    paths = values.get("paths", {}) or {}
    messages = values.get("messages", []) or []
    next_nodes = tuple(
        key.removeprefix("branch:to:")
        for key in values
        if isinstance(key, str) and key.startswith("branch:to:")
    )

    return CheckpointSummary(
        row=row,
        step=metadata.get("step") if isinstance(metadata.get("step"), int) else None,
        source=str(metadata.get("source", "")),
        phase=_phase_value(values.get("phase")),
        active_hitl=values.get("active_hitl"),
        current_image=paths.get("current_image") if isinstance(paths, dict) else None,
        messages_count=len(messages),
        next_nodes=next_nodes,
        working_dir=dataset.get("working_dir") if isinstance(dataset, dict) else None,
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"checkpoint DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_rows(conn: sqlite3.Connection, thread_id: str) -> list[CheckpointRow]:
    rows = conn.execute(
        """
        SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
               type, checkpoint, metadata
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY checkpoint_ns, checkpoint_id
        """,
        (thread_id,),
    ).fetchall()
    return [
        CheckpointRow(
            thread_id=r["thread_id"],
            checkpoint_ns=r["checkpoint_ns"],
            checkpoint_id=r["checkpoint_id"],
            parent_checkpoint_id=r["parent_checkpoint_id"],
            type=r["type"],
            checkpoint=r["checkpoint"],
            metadata=r["metadata"],
        )
        for r in rows
    ]


def _sort_summaries(summaries: list[CheckpointSummary]) -> list[CheckpointSummary]:
    return sorted(
        summaries,
        key=lambda s: (
            s.row.checkpoint_ns,
            s.step if s.step is not None else 10**12,
            s.row.checkpoint_id,
        ),
    )


def _latest_summary(summaries: list[CheckpointSummary]) -> CheckpointSummary:
    if not summaries:
        raise ValueError("source thread has no checkpoints")
    return _sort_summaries(summaries)[-1]


def _select_at_phase(
    summaries: list[CheckpointSummary],
    phase: str,
) -> CheckpointSummary:
    target = phase.lower()
    ordered = [s for s in _sort_summaries(summaries) if s.row.checkpoint_ns == ""]
    if not ordered:
        ordered = _sort_summaries(summaries)

    phase_matches = [s for s in ordered if s.phase.lower() == target]
    if not phase_matches:
        available = ", ".join(sorted({s.phase for s in ordered}))
        raise ValueError(
            f"no checkpoint in phase {phase!r}; available phases: {available}"
        )

    boundary_matches: list[CheckpointSummary] = []
    previous_phase: str | None = None
    for summary in ordered:
        if summary.phase.lower() == target and previous_phase != summary.phase.lower():
            boundary_matches.append(summary)
        previous_phase = summary.phase.lower()

    return boundary_matches[-1] if boundary_matches else phase_matches[-1]


def _select_checkpoint(
    summaries: list[CheckpointSummary],
    checkpoint_id: str,
) -> CheckpointSummary:
    matches = [s for s in summaries if s.row.checkpoint_id == checkpoint_id]
    if not matches:
        raise ValueError(f"checkpoint_id not found in source thread: {checkpoint_id}")
    if len(matches) > 1:
        namespaces = ", ".join(repr(s.row.checkpoint_ns) for s in matches)
        raise ValueError(
            f"checkpoint_id {checkpoint_id!r} exists in multiple namespaces "
            f"({namespaces}); this script currently expects a unique id"
        )
    return matches[0]


def _ancestor_keys(
    rows: list[CheckpointRow],
    selected: CheckpointRow,
) -> set[tuple[str, str]]:
    by_key = {(r.checkpoint_ns, r.checkpoint_id): r for r in rows}
    included: set[tuple[str, str]] = set()
    current = selected
    while True:
        key = (current.checkpoint_ns, current.checkpoint_id)
        included.add(key)
        parent = current.parent_checkpoint_id
        if not parent:
            break
        parent_key = (current.checkpoint_ns, parent)
        parent_row = by_key.get(parent_key)
        if parent_row is None:
            raise ValueError(
                f"checkpoint chain is broken: parent {parent!r} for "
                f"{current.checkpoint_id!r} not found"
            )
        current = parent_row
    return included


def _destination_exists(conn: sqlite3.Connection, thread_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1",
        (thread_id,),
    ).fetchone()
    return row is not None


def _delete_destination(conn: sqlite3.Connection, thread_id: str) -> None:
    conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))


def _copy_rows(
    conn: sqlite3.Connection,
    source_thread_id: str,
    dest_thread_id: str,
    included_keys: set[tuple[str, str]] | None,
) -> tuple[int, int]:
    if included_keys is None:
        checkpoint_rows = conn.execute(
            """
            SELECT checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                   type, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ?
            """,
            (source_thread_id,),
        ).fetchall()
    else:
        checkpoint_rows = [
            r for r in conn.execute(
                """
                SELECT checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                       type, checkpoint, metadata
                FROM checkpoints
                WHERE thread_id = ?
                """,
                (source_thread_id,),
            ).fetchall()
            if (r["checkpoint_ns"], r["checkpoint_id"]) in included_keys
        ]

    conn.executemany(
        """
        INSERT INTO checkpoints (
            thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
            type, checkpoint, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                dest_thread_id,
                r["checkpoint_ns"],
                r["checkpoint_id"],
                r["parent_checkpoint_id"],
                r["type"],
                r["checkpoint"],
                r["metadata"],
            )
            for r in checkpoint_rows
        ],
    )

    if included_keys is None:
        write_rows = conn.execute(
            """
            SELECT checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value
            FROM writes
            WHERE thread_id = ?
            """,
            (source_thread_id,),
        ).fetchall()
    else:
        write_rows = [
            r for r in conn.execute(
                """
                SELECT checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value
                FROM writes
                WHERE thread_id = ?
                """,
                (source_thread_id,),
            ).fetchall()
            if (r["checkpoint_ns"], r["checkpoint_id"]) in included_keys
        ]

    conn.executemany(
        """
        INSERT INTO writes (
            thread_id, checkpoint_ns, checkpoint_id, task_id,
            idx, channel, type, value
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                dest_thread_id,
                r["checkpoint_ns"],
                r["checkpoint_id"],
                r["task_id"],
                r["idx"],
                r["channel"],
                r["type"],
                r["value"],
            )
            for r in write_rows
        ],
    )
    return len(checkpoint_rows), len(write_rows)


def _print_list(summaries: list[CheckpointSummary]) -> None:
    print("checkpoint_id                           step   phase        hitl  msgs  next")
    print("-" * 84)
    previous_phase: str | None = None
    for summary in _sort_summaries(summaries):
        marker = ">"
        phase = summary.phase
        if previous_phase == phase:
            marker = " "
        previous_phase = phase
        next_label = ",".join(summary.next_nodes) if summary.next_nodes else "-"
        step_label = str(summary.step) if summary.step is not None else "-"
        print(
            f"{marker} {summary.row.checkpoint_id:<36} "
            f"{step_label:>5}  {phase:<11} "
            f"{str(summary.active_hitl):<5} "
            f"{summary.messages_count:>5}  {next_label}"
        )


def _working_dir_for_clone(
    source_thread_id: str,
    selected_summary: CheckpointSummary,
) -> str | None:
    return (
        lookup_session_dir(source_thread_id)
        or selected_summary.working_dir
        or (
            str(Path(selected_summary.current_image).parent)
            if selected_summary.current_image
            else None
        )
    )


def _default_dest(source_thread_id: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{source_thread_id}-clone-{ts}"


async def _aget_graph_summary(db_path: Path, thread_id: str) -> dict[str, Any]:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from muphrid.graph.graph import build_graph

    conn = await aiosqlite.connect(str(db_path))
    try:
        checkpointer = AsyncSqliteSaver(conn=conn, serde=_serializer())
        await checkpointer.setup()
        graph = build_graph(checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snapshot:
            raise ValueError(f"LangGraph returned no state for cloned thread {thread_id!r}")
        values = snapshot.values or {}
        paths = values.get("paths", {}) or {}
        messages = values.get("messages", []) or []
        return {
            "phase": _phase_value(values.get("phase")),
            "active_hitl": values.get("active_hitl"),
            "current_image": paths.get("current_image") if isinstance(paths, dict) else None,
            "messages_count": len(messages),
            "next_nodes": tuple(snapshot.next or ()),
            "tasks": tuple(t.name for t in (snapshot.tasks or ())),
        }
    finally:
        await conn.close()


def _get_graph_summary(db_path: Path, thread_id: str) -> dict[str, Any]:
    return asyncio.run(_aget_graph_summary(db_path, thread_id))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone a LangGraph checkpoint thread to a fresh thread id.",
    )
    parser.add_argument("source_thread_id", help="Thread id to clone from.")
    parser.add_argument(
        "dest_thread_id",
        nargs="?",
        help="New thread id. Defaults to <source>-clone-<UTC timestamp>.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"Checkpoint DB path. Default: {DEFAULT_DB}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination thread if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and summarize the source checkpoint without writing a clone.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List checkpoints for the source thread and exit.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint-id",
        help="Clone the ancestor chain ending at this checkpoint id.",
    )
    group.add_argument(
        "--at-phase",
        help=(
            "Clone from the latest phase-entry checkpoint for this phase "
            "(e.g. linear, stretch, nonlinear)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    try:
        graph_summary: dict[str, Any] | None = None

        with _connect(db_path) as conn:
            rows = _load_rows(conn, args.source_thread_id)
            if not rows:
                print(
                    f"No checkpoints found for source thread "
                    f"{args.source_thread_id!r} in {db_path}",
                    file=sys.stderr,
                )
                return 1

            summaries = [_row_to_summary(row) for row in rows]
            if args.list:
                print(f"Thread: {args.source_thread_id}")
                print(f"DB:     {db_path}")
                print()
                _print_list(summaries)
                return 0

            if args.checkpoint_id:
                selected = _select_checkpoint(summaries, args.checkpoint_id)
                included_keys: set[tuple[str, str]] | None = _ancestor_keys(
                    rows, selected.row
                )
                selection_label = f"checkpoint {selected.row.checkpoint_id}"
            elif args.at_phase:
                selected = _select_at_phase(summaries, args.at_phase)
                included_keys = _ancestor_keys(rows, selected.row)
                selection_label = (
                    f"phase {args.at_phase!r} at checkpoint "
                    f"{selected.row.checkpoint_id}"
                )
            else:
                selected = _latest_summary(summaries)
                included_keys = None
                selection_label = f"latest checkpoint {selected.row.checkpoint_id}"

            dest_thread_id = args.dest_thread_id or _default_dest(args.source_thread_id)
            working_dir = _working_dir_for_clone(args.source_thread_id, selected)

            if args.dry_run:
                print("Dry run — no checkpoint rows copied.")
                print(f"Source:     {args.source_thread_id}")
                print(f"Dest:       {dest_thread_id}")
                print(f"Selection:  {selection_label}")
                print(f"Phase:      {selected.phase}")
                print(f"Active HITL:{selected.active_hitl}")
                print(f"Messages:   {selected.messages_count}")
                print(f"Current:    {selected.current_image or '(none)'}")
                print(f"Work dir:   {working_dir or '(not found)'}")
                return 0

            if dest_thread_id == args.source_thread_id:
                print("Destination thread id must differ from source.", file=sys.stderr)
                return 1
            if not dest_thread_id.isascii():
                print("Destination thread id must be ASCII.", file=sys.stderr)
                return 1

            if _destination_exists(conn, dest_thread_id):
                if not args.force:
                    print(
                        f"Destination thread {dest_thread_id!r} already exists. "
                        "Use --force to overwrite it.",
                        file=sys.stderr,
                    )
                    return 1
                _delete_destination(conn, dest_thread_id)

            checkpoint_count, write_count = _copy_rows(
                conn,
                args.source_thread_id,
                dest_thread_id,
                included_keys,
            )

            if working_dir:
                register_session(dest_thread_id, working_dir)

            conn.commit()

        graph_summary = _get_graph_summary(db_path, dest_thread_id)

    except Exception as e:
        print(f"clone_thread failed: {e}", file=sys.stderr)
        return 1

    print(f"Cloned {checkpoint_count} checkpoint row(s), {write_count} write row(s)")
    print(f"Source:     {args.source_thread_id}")
    print(f"Dest:       {dest_thread_id}")
    print(f"Selection:  {selection_label}")
    print(f"Phase:      {graph_summary.get('phase') if graph_summary else selected.phase}")
    print(
        "Active HITL:"
        f"{graph_summary.get('active_hitl') if graph_summary else selected.active_hitl}"
    )
    print(
        f"Messages:   "
        f"{graph_summary.get('messages_count') if graph_summary else selected.messages_count}"
    )
    print(
        f"Current:    "
        f"{(graph_summary.get('current_image') if graph_summary else selected.current_image) or '(none)'}"
    )
    if graph_summary:
        next_nodes = graph_summary.get("next_nodes") or ()
        tasks = graph_summary.get("tasks") or ()
        next_or_tasks = next_nodes or tasks
        print(f"Next/tasks: {', '.join(next_or_tasks) if next_or_tasks else '-'}")
    if working_dir:
        print(f"Work dir:   {working_dir}")
    else:
        print("Work dir:   (not found; Gradio may derive previews from state paths)")
    print()
    print(f"Resume in Gradio with: {dest_thread_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
