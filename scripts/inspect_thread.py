"""
Quick diagnostic: load a saved thread's state and print key fields.

Usage:
    uv run python scripts/inspect_thread.py <thread_id> [<db_path>]

Defaults:
    thread_id = run-m20-trifid-nebula-20260428-154311
    db_path   = checkpoints.db
"""

import asyncio
import sys

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from muphrid.graph.graph import build_graph


THREAD_ID = sys.argv[1] if len(sys.argv) > 1 else "run-m20-trifid-nebula-20260428-154311"
DB_PATH = sys.argv[2] if len(sys.argv) > 2 else "checkpoints.db"


async def main() -> None:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("muphrid.graph.state", "ProcessingPhase")]
    )
    conn = await aiosqlite.connect(DB_PATH)
    checkpointer = AsyncSqliteSaver(conn=conn, serde=serde)
    await checkpointer.setup()

    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": THREAD_ID}}
    snap = await graph.aget_state(config)

    if not snap:
        print(f"No saved state for thread '{THREAD_ID}' in {DB_PATH}")
        return

    v = snap.values or {}
    print(f"Thread: {THREAD_ID}")
    print(f"DB:     {DB_PATH}")
    print("---")
    print(f"phase:               {v.get('phase')}")
    print(f"active_hitl:         {v.get('active_hitl')}")
    print(f"current_image:       {(v.get('paths', {}) or {}).get('current_image')}")
    print(f"variant_pool count:  {len(v.get('variant_pool', []) or [])}")
    print(f"variant_pool ids:    {[x.get('id') for x in (v.get('variant_pool') or [])]}")
    print(f"messages count:      {len(v.get('messages', []) or [])}")
    print(f"snapshot.next:       {snap.next}")
    print(f"snapshot.tasks:      {[t.name for t in (snap.tasks or [])]}")

    await conn.close()


asyncio.run(main())
