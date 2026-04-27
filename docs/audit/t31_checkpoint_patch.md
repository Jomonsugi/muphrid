# Patch — Checkpoint hardening (t31)

**File:** `/Users/micahshanks/Dev/muphrid/muphrid/tools/nonlinear/t31_checkpoint.py`

## Why

In the M20 run, the agent called `save_checkpoint("starless_base")` expecting
to bookmark the starless image. But `state.paths.current_image` at that moment
was still pointing at the pre-starless vibrant_nebula FITS — `star_removal`
had written the starless file as a separate sibling but `current_image` hadn't
been updated (or the agent's mental model had drifted). The checkpoint locked
in the wrong reference. When LCE was then applied and produced a starfull
output, the agent restored to "starless_base" three times trying to recover,
each restore returning the same wrong pre-starless state. The agent gave up,
called `commit_variant`, and moved on with a flawed image.

The tool's return message only shows `Path(current_image).name` — the bare
filename — so the agent can't easily verify what was actually bookmarked.
And there is no way to bookmark a specific path independent of what
`current_image` happens to point at.

## Shape of the change

Two small edits in one file. No schema changes for existing callers; the
new `explicit_path` argument is optional and backward compatible.

### Change 1 — Make `save_checkpoint` echo the full resolved path

Current response payload:

```python
summary = {
    "saved": name,
    "image": Path(current_image).name,
    "overwritten": name in existing,
    "all_checkpoints": {k: Path(v).name for k, v in updated.items()},
}
```

Replace with:

```python
bookmarked_path = Path(current_image).resolve()
summary = {
    "saved": name,
    "bookmarked_path": str(bookmarked_path),
    "bookmarked_file": bookmarked_path.name,
    "bookmarked_size_bytes": bookmarked_path.stat().st_size,
    "bookmarked_mtime": bookmarked_path.stat().st_mtime,
    "overwritten": name in existing,
    "verify": (
        "Confirm that 'bookmarked_path' is the file you intended to "
        "checkpoint. If your last tool call produced a different output "
        "file (e.g. a starless sibling) and current_image did not follow, "
        "call save_checkpoint(name=..., explicit_path='<path>') instead."
    ),
    "all_checkpoints": {k: str(Path(v).resolve()) for k, v in updated.items()},
}
```

The extra size/mtime fields let the agent sanity-check — if the mtime is
older than the last processing step, the bookmark is almost certainly stale.

### Change 2 — Optional `explicit_path` argument

Extend the input schema:

```python
class SaveCheckpointInput(BaseModel):
    name: str = Field(
        description=(
            "A short, descriptive name for this checkpoint. Use names that "
            "describe the processing state: 'starless_base', 'after_curves', "
            "'pre_saturation', 'curves_v2_good'."
        ),
    )
    explicit_path: str | None = Field(
        default=None,
        description=(
            "Optional absolute path to bookmark instead of current_image. "
            "Use this when a recent tool produced a specific output file "
            "that current_image did not follow (e.g. star_removal produces "
            "both a starless and a stars-only FITS; current_image may "
            "point at one but you want to bookmark the other). When "
            "explicit_path is provided, current_image is left unchanged "
            "and only the checkpoint reference is stored. When omitted, "
            "current_image is used."
        ),
    )
```

Then in the tool body, resolve the target path up-front:

```python
target_path = explicit_path if explicit_path else state["paths"].get("current_image")

if not target_path:
    return Command(update={
        "messages": [ToolMessage(
            content=(
                "Cannot save checkpoint: no current_image is set and no "
                "explicit_path was provided."
            ),
            tool_call_id=tool_call_id,
        )],
    })

resolved = Path(target_path).resolve()
if not resolved.exists():
    return Command(update={
        "messages": [ToolMessage(
            content=(
                f"Cannot save checkpoint '{name}': file does not exist at "
                f"{resolved} (source: "
                f"{'explicit_path' if explicit_path else 'current_image'})."
            ),
            tool_call_id=tool_call_id,
        )],
    })

existing = state.get("metadata", {}).get("checkpoints") or {}
updated = {**existing, name: str(resolved)}
```

### Change 3 — `restore_checkpoint` reports a diff summary

Currently the restore payload is:

```python
summary = {
    "restored": name,
    "current_image": Path(restore_path).name,
    "previous_image": ...,
    "all_checkpoints": {...},
}
```

Replace with:

```python
restored = Path(restore_path).resolve()
prev = (
    Path(state["paths"]["current_image"]).resolve()
    if state["paths"].get("current_image")
    else None
)

summary = {
    "restored_checkpoint": name,
    "current_image_now": str(restored),
    "current_image_before_restore": str(prev) if prev else None,
    "same_file": prev is not None and prev == restored,
    "diff_note": (
        "current_image has been reset to the checkpointed path. Any "
        "output files produced between the checkpoint and now still "
        "exist on disk; they are just no longer the active image. "
        "If you want to return to the pre-restore state, call "
        "save_checkpoint with explicit_path=<the pre-restore path> "
        "BEFORE calling restore."
    ) if not (prev is not None and prev == restored) else "No-op: already at that checkpoint.",
    "all_checkpoints": {k: str(Path(v).resolve()) for k, v in checkpoints.items()},
}
```

The `same_file: true` signal catches the M20 loop directly — when three
consecutive restores all return `same_file: true`, the agent knows the
restore isn't doing what it wants and can branch instead of retrying.

## Prompt-side lever

In `prompts.py` SYSTEM_BASE or the NONLINEAR prompt (covered in
`prompts_nonlinear_rewrite.md` §2), add:

```
save_checkpoint bookmarks whatever path state.paths.current_image points at
right now. It does NOT copy files. The returned message includes the resolved
path it bookmarked — verify that path matches what you intended before
relying on it. If the path looks stale (older mtime than your last tool
call, or a filename from before the last star_removal / stretch / similar),
re-run save_checkpoint with explicit_path=<the correct absolute path>.

If restore_checkpoint returns same_file=true, the restore was a no-op —
current_image already matched the checkpoint. In that case the checkpoint
was stale at save time; create a fresh checkpoint from the correct path
before retrying.
```

## Regression / test plan

This change is tractable to unit-test:

1. Create two temporary FITS files, set `current_image` to one of them.
2. Call `save_checkpoint("a")` with no explicit_path → assert bookmark
   resolves to file #1.
3. Call `save_checkpoint("b", explicit_path=<file #2>)` → assert bookmark
   resolves to file #2 AND `current_image` is unchanged.
4. Call `restore_checkpoint("a")` → assert `current_image` == file #1,
   `same_file` reflects whether file #1 was already current.
5. Reproduce the M20 scenario: `current_image` = pre-starless, write a
   starless sibling without updating `current_image`, call
   `save_checkpoint("starless_base")` with no explicit_path. Assert the
   resolved path in the response is the pre-starless file (the bug
   reproduces). Then call with `explicit_path=<starless sibling>`. Assert
   the checkpoint now points at the starless.

## Effort

~40 lines changed in `t31_checkpoint.py`. ~10 lines added to prompts.py.
Fully backward compatible — no existing caller needs to change.

## Root-cause pair

The upstream fix — having `star_removal` always update `current_image` to
the starless path, or returning a structured `outputs: {starless: ..., stars: ...}`
dict that the agent explicitly chooses from — is a larger change and should
be considered separately. The checkpoint hardening here is the defense-in-depth
layer: even if upstream tools leave `current_image` inconsistent, the agent
now has the signal (full path in the response) and the escape hatch
(explicit_path) to recover without looping.
