# Notes for Claude when working on muphrid

## Debugging tool failures — read the message stream first

When the agent loops on a tool error, **never guess what the error is**. The
LangGraph SQLite checkpoint always has the actual `ToolMessage.content` —
that is the literal string the agent saw and reacted to. Read it first;
patch second.

The default checkpoint DB lives at `checkpoints.db` in the working dir
(default project root) or wherever the CLI was launched. There is also a
fallback at `<dataset>/runs/checkpoints.db` for some Gradio sessions. Use
`scripts/inspect_thread.py` for the quick state summary. For the full
message tail (which is what shows the actual tool error), drop into the
DB directly:

```python
import sqlite3
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

conn = sqlite3.connect("checkpoints.db")
cur = conn.cursor()
cur.execute(
    "SELECT type, checkpoint FROM checkpoints "
    "WHERE thread_id=? ORDER BY checkpoint_id DESC LIMIT 1",
    (thread_id,),
)
type_, blob = cur.fetchone()
cp = JsonPlusSerializer().loads_typed((type_, blob))
for m in cp["channel_values"]["messages"][-15:]:
    print(type(m).__name__, getattr(m, "name", ""), str(getattr(m, "content", ""))[:400])
    if tc := getattr(m, "tool_calls", None):
        print("  tool_calls:", [(t["name"], {k: v for k, v in t.get("args", {}).items() if k not in ("state", "tool_call_id")}) for t in tc])
```

The audit-report markdown files (`runs/<id>/reports/NN_<phase>.md`) only
capture phases the agent successfully advanced through. If the run failed
inside a phase, that phase's report won't exist yet — so the markdown
**will not** show the failing tool's output. Only the checkpoint DB has it.

## Stuck-loop errors are symptoms, not causes

`StuckLoopError: Agent called 'X' 3 times and every result was an error`
means the same tool returned the same error class three times. The
detector itself is fine. The interesting question is always *what was the
error message* — that's in the `ToolMessage` immediately preceding the
StuckLoopError, in the checkpoint. Don't infer; read it.

## When tempted to add defensive fallbacks, ask whether the real bug is upstream

Speculative "if X is empty, fall back to Y" patches are the easy reach
when a stuck loop happens. They almost always paper over a real bug
elsewhere — schema/function drift, a tool returning a non-actionable
error string, the agent prompt missing a step, etc. The empty-state
fallback I added once for `siril_stack` was wrong: the agent had actually
populated state correctly; the real bug was a schema field
(`output_norm`) that the function signature didn't accept. The fallback
would have silently passed bad data through if it had ever fired.

Default to: read the error → trace the path → fix at the source.

## Schema/function drift is the recurring gotcha

The codebase uses `@tool(args_schema=Foo)` everywhere (per #59). When you
edit either the schema or the function signature, sweep the other side.
A quick check:

```bash
python3 - <<'EOF'
import ast
from pathlib import Path
for f in sorted(Path("muphrid/tools").rglob("*.py")):
    src = f.read_text()
    tree = ast.parse(src)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for dec in fn.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "tool":
                schema = next((kw.value.id for kw in dec.keywords
                               if kw.arg == "args_schema" and isinstance(kw.value, ast.Name)), None)
                if not schema or schema not in classes:
                    continue
                fields = {s.target.id for s in classes[schema].body
                          if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
                params = {a.arg for a in fn.args.args} - {"tool_call_id", "state"}
                if fields - params or params - fields:
                    print(f"{f}: {fn.name} drift: schema-only={fields-params} func-only={params-fields}")
EOF
```

## Tool state updates: emit deltas, never spread the whole field

`AstroState.paths` and `AstroState.metadata` use `_merge_dicts` (deep-merge)
as their reducer. The reducer composes parallel updates correctly — but
*only if each update names just the keys it changes*. Spreading the
existing dict and adding one key is a footgun:

```python
# BAD — silently breaks under parallel execution
return Command(update={
    "paths": {**state["paths"], "current_image": output_path}
})

# GOOD — delta only; the reducer composes with siblings
return Command(update={
    "paths": {"current_image": output_path}
})
```

Why it matters: modern LLMs run tool calls in parallel, and the agent will
do exactly that (e.g. `build_masters(bias)` + `build_masters(dark)` +
`build_masters(flat)` in one turn). Each parallel call reads `state` at
its own start, computes its update, and emits. The reducer applies them in
some order. If A spreads `state["paths"]` and changes only `current_image`
while B (parallel) spreads `state["paths"]` and changes only `latest_mask`,
A's emitted `latest_mask` value is *whatever it was before B ran* — when
B's update merges first, then A's, A's stale `latest_mask` overwrites
B's update. Last writer wins; sibling changes vanish.

The catastrophic flavor of this bug shows up post-`rewind_phase`: rewind
restores `paths.masters = {"bias": None, "dark": None, "flat": None}`. A
3-way parallel `build_masters` then emits, each carrying explicit `None`
values for the sibling masters. The reducer sees `{"bias": None}` from one
call and `{"bias": "/path/master_bias.fit"}` from another and overwrites
according to merge order — most updates lost, `advance_phase` blocks
forever on "missing master_bias", agent loops, eventually emits an empty
turn that 400s the API.

**Sweep:** before merging changes that touch `Command(update={"paths": ...})`
or `Command(update={"metadata": ...})`, run this:

```bash
grep -rn '{\*\*state\["paths"\]\|{\*\*state\["metadata"\]' muphrid/tools \
  | grep -v '^[^:]*:\s*#'
```

Any hit not in a comment is a candidate parallel-update clobber bug.

**Replace-aware fields** (`metrics`) use `_dict_merge_or_replace` (defined
in `muphrid/graph/state.py`). The reducer composes parallel deltas as
deep-merge by default, but treats `Replace(value)` as a full overwrite.
Use `Replace` from clear/restore code paths so the field can still be
reset:

```python
from muphrid.graph.state import Replace

# Tool writer (composes with siblings under parallel execution):
return Command(update={"metrics": {"is_linear_estimate": False}})

# Clearer / restorer (rewind_phase, etc.):
return Command(update={"metrics": Replace(restored_metrics)})
```

Three tools write `metrics`: `analyze_image` (per-image stats),
`analyze_frames` (frame_stats / frame_summary), and `stretch_image`
(is_linear_estimate). All three now emit deltas. Modern parallel-tool-call
models can batch them safely.

**Plain replace-semantics fields** (`variant_pool`, `visual_context`,
`regression_warnings`) still have no reducer. Their writers compute the
full new list from scratch (e.g. `variant_snapshot` rebuilds the pool
from messages), so additive composition would be incorrect for them. They
have a single writer per super-step in practice, so parallel-write data
loss doesn't occur. If you ever add a second writer that should compose
additively, add a `_list_extend_or_replace` reducer mirroring the dict
version above.

## Synthetic state for direct tool .func() calls is NOT a Command.update

When a tool calls another tool's underlying function directly — e.g.
`create_mask.func(state=synth, ...)` inside `t35_hdr_composite` —
the synthetic `synth` dict does NOT go through any reducer. The inner
tool reads `synth["paths"]` directly and expects a complete paths dict,
not a delta. So the `{**state["paths"], "X": value}` spread is
*required* in that idiom, and the delta-only rule does not apply.

The two patterns look almost identical, so be deliberate when sweeping.
Command.update payloads compose through reducers; synthetic state for
direct .func() invocation does not.

## Checkpoint DB corruption — diagnose and recover, don't auto-heal

`sqlite3.DatabaseError: database disk image is malformed` on a `astream` /
`aput_writes` call means the SQLite checkpoint file is physically damaged.
Common causes, in order of frequency:

  1. A Python process holding a write was killed (SIGKILL, OOM kill,
     parent terminal closed, `kill -9` on Gradio while the agent was
     mid-stream).
  2. Two processes writing the same file concurrently (Gradio started
     while CLI was running, or two Gradio instances pointed at
     `./checkpoints.db`). SQLite's default journaling does not protect
     across multiple write-holders if one was started carelessly.
  3. Disk-level fault — rare on local SSDs, more common on networked
     filesystems.

`muphrid/gradio_app.py:_check_checkpoint_db_integrity` runs
`PRAGMA integrity_check` at startup and refuses to start a session if the
DB is corrupt. The trace you'd otherwise see is buried 20+ frames deep in
`langgraph/pregel/_loop.py`, which is unhelpful for triage; the startup
check turns it into a clear actionable message.

Recovery: `uv run python scripts/recover_checkpoint_db.py [<db_path>]`.
The script row-walks the corrupt file, copies readable rows into a fresh
DB, quarantines the broken file as `<db>.corrupt-<UTC-ts>`, and promotes
the recovered DB into place. SQLite's recovery is per-row, so corruption
in B-tree pages typically only loses the rows on those specific pages —
in practice every checkpoint we've seen has been fully recoverable.

This is intentionally not automatic on every launch. Silent self-healing
can paper over a real disk fault that the operator might want to inspect
first; opt-in keeps the operator in the loop.

## Empty Pydantic args_schema is a langchain-core trap

`langchain-core` 1.3.x `BaseTool._to_args_and_kwargs` has an early-return
path:

```python
if (self.args_schema is not None
    and is_basemodel_subclass(self.args_schema)
    and not get_fields(self.args_schema)):
    return (), {}
```

If your tool's args_schema is fields-empty, that path discards the
state/tool_call_id LangGraph just injected, and the function gets called
with no args at all. Symptom: `<tool>() missing 2 required positional
arguments: 'tool_call_id' and 'state'`. Workaround: add a single optional
field to the schema (see `AnalyzeFramesInput.note`).
