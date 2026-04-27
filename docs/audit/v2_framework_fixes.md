# Framework Fixes (v2)

_April 19, 2026 · supersedes sections of the original audit that framed the checkpoint bug as a tool-level issue; this doc relocates the fix to the framework layer and adds four more framework-level issues surfaced by the Sonnet 4.6 run `run-m20-20260419-005415`._

## Why this memo replaces `t31_checkpoint_patch.md`

The original patch added an `explicit_path` argument to `save_checkpoint`. That moves in the wrong direction. The operator's mental model is "checkpoint this processing state and restore it later" — same as PixInsight or Siril's History. The agent should never have to reason about which FITS file represents the state. Adding `explicit_path` forces it to. The fix belongs upstream, in the tools that produce sibling files without updating `current_image`.

The original `t31_checkpoint_patch.md` should be retired. Replace it with this memo.

## The shared root cause: tools that write siblings but don't update `current_image`

`star_removal` (t15_star_removal.py, lines 210–213) returns:

```python
return Command(update={
    "paths": {**state["paths"], "starless_image": str(starless_fits), "star_mask": mask_fits_path},
    "messages": [ToolMessage(...)],
})
```

It sets `paths.starless_image` and `paths.star_mask`. It does _not_ touch `paths.current_image`. So after `star_removal` runs, `current_image` still points at the starred input. Any tool that reads `paths.current_image` next — `save_checkpoint`, `local_contrast_enhance`, `multiscale_process`, `pixel_math`, almost everything — operates on the starred image.

This is not a subtle issue. Both the Kimi run (M20, Apr 17) and the Sonnet run (M20, Apr 19) hit it identically:

- Agent calls `star_removal` → starless file exists on disk, `current_image` unchanged.
- Agent calls `save_checkpoint("starless_base")` → bookmarks `current_image`, which is the _starred_ file.
- Agent calls `local_contrast_enhance` → runs on the _starred_ file, produces a starred output.
- Agent sees stars in the output and is confused. Calls `restore_checkpoint("starless_base")` to recover.
- Restore returns to `current_image` = the same starred file. The "restore" is a no-op by design, but the agent cannot see that.
- Agent retries restore two or three more times. Loop.

Sonnet's own narration on the third restore is precise: _"the 'starless_base' checkpoint is just pointing back to the same blown v6 image."_ The agent correctly diagnosed the bug without being able to fix it.

### The fix: `star_removal` must update `current_image` to the starless output

One-line semantic change:

```python
return Command(update={
    "paths": {
        **state["paths"],
        "current_image": str(starless_fits),          # <-- add
        "previous_image": state["paths"].get("current_image"),  # <-- optional, for audit trail
        "starless_image": str(starless_fits),
        "star_mask": mask_fits_path,
    },
    ...
})
```

Rationale: the whole point of `star_removal` in a starless workflow is that the starless file _is_ the working image going forward. Until `star_restoration` runs, every subsequent step should operate on the starless. Continuing to point `current_image` at the starred input means every tool that reads `current_image` is reading the wrong file — which is what happened.

`star_restoration` (t19) already reads `paths.starless_image` explicitly, so it doesn't need to rely on `current_image`. Its own output assignment should also update `current_image` back to the restored file — worth double-checking.

### Audit every tool that writes a sibling file

A clean way to prevent this class of bug:

- **Rule:** any tool that writes a new FITS must update `paths.current_image` to that file unless the tool is documented as a side-output tool (e.g., a tool that writes only a mask or only a background model and leaves the main working image alone).
- **Action:** grep the codebase for tools whose `Command` update touches `paths` but does not set `current_image`. Each of them must be reviewed. Candidates to check: `star_removal` (fixed above), `create_mask` (writes a mask — should NOT update `current_image` because the mask is not the working image), `remove_gradient` when `save_background_model=True` writes a background sibling, any tool that produces `*_before.fits` / `*_after.fits` snapshots.

Proposed convention and documentation (to include in a CLAUDE.md or tools/README.md):

> Every tool that produces the new working image must set `paths.current_image` to that file. Tools that produce auxiliary outputs (masks, backgrounds, debug frames) must set an explicit named path in `paths` (`star_mask`, `background_model`, etc.) and leave `current_image` alone. The agent never references `current_image` — it is an internal framework concept. The agent references processing states by checkpoint name.

### `save_checkpoint` / `restore_checkpoint` — no API change needed

With the upstream fix, `save_checkpoint` bookmarks the right thing by construction, because `current_image` is always what the agent just produced. No `explicit_path` argument. No agent-facing paths. The docstring can then be written in processing-state language only:

```python
"""
Bookmark the current processing state under a name you can restore later.

Use this before any adjustment you might want to undo. restore_checkpoint
returns you to that state. Think of checkpoints as branch labels in your
processing tree — cheap to create, cheap to restore.
"""
```

### One defense-in-depth guard

Even with the upstream fix, `restore_checkpoint` should return a clear signal when the restore is a no-op (stored path == current path). This catches any _future_ tool that regresses the convention. The minimal signal:

```python
summary = {
    "restored_checkpoint": name,
    "noop": restored_path == prev_path,   # agent can see this and branch
    "diff_note": "Restore was a no-op — checkpoint and current state are the same file." if noop else "State restored.",
}
```

If three consecutive restores return `noop: true`, the framework should consider surfacing an explicit advisory via a ToolMessage rather than leave the agent to diagnose it. A small stuck-loop detector for restore specifically.

## Issue #2 — `commit_variant` pool-cleared race

**Symptom.** Log line from Sonnet run:

```
2026-04-19 01:03:23,679 [muphrid.graph.nodes] INFO: promote_variant: T09_v1 → current_image ...
2026-04-19 01:03:26,902 [muphrid.tools.utility.t31_commit_variant] WARNING: commit_variant: id='T09_v1' not in pool (valid=[])
```

And again for T14_v6.

**Cause.** In `nodes.py`, `promote_variant` (HITL approval path) runs `build_variant_promotion_update` which clears `variant_pool = []`. The HumanMessage approval text then cues the agent. The agent reasonably reads "Approved T09_v1" as an instruction to also call `commit_variant(T09_v1)` — which looks up the now-empty pool and errors.

**Fix.** `commit_variant` must handle "already promoted" gracefully rather than failing. One approach:

```python
def commit_variant(variant_id, rationale, state, tool_call_id):
    pool = list(state.get("variant_pool", []) or [])
    current_image = state.get("paths", {}).get("current_image", "")

    # Case 1: normal path — variant is in the pool
    result = build_variant_promotion_update(state, variant_id)
    if result is not None:
        variant, update = result
        # ... existing logic
        return Command(update=update)

    # Case 2: pool is empty but a recent message shows an HITL approval for this id.
    # Treat as an acknowledgement, not an error.
    recent_approval = _scan_for_recent_hitl_approval(state, variant_id)
    if recent_approval:
        return Command(update={
            "messages": [ToolMessage(
                content=json.dumps({
                    "status": "already_committed",
                    "variant_id": variant_id,
                    "current_image": current_image,
                    "note": (
                        "This variant was already promoted via HITL approval. "
                        "current_image reflects the approved variant. "
                        "No further commit action was needed."
                    ),
                }),
                tool_call_id=tool_call_id,
            )],
        })

    # Case 3: unknown variant — original error path
    return _error_payload(...)
```

The "scan for recent HITL approval" helper looks at the last few HumanMessage entries for "Approved {variant_id}" (emitted by `promote_variant` at nodes.py:1205–1209). Alternatively, the framework can set a short-lived metadata flag `metadata.last_hitl_approved_variant` that `commit_variant` checks directly.

**Better fix (upstream):** the HITL-approval HumanMessage should include an instruction like "No `commit_variant` call needed — the variant is already your current image." This eliminates the spurious commit_variant call at the source. Minimal change to the template in `promote_variant`.

**Effort.** Either approach is ~15 lines. The "better fix" is preferred because it stops the agent from making a pointless tool call in the first place.

## Issue #3 — Phase gates: too rigid, no backward path

Three distinct pain points surfaced in the Sonnet run.

### 3a. Hard forward-only phase order

`t30_advance_phase.py` defines:

```python
_NEXT_PHASE = {
    INGEST → CALIBRATION → REGISTRATION → ANALYSIS → STACKING →
    LINEAR → STRETCH → NONLINEAR → EXPORT → COMPLETE,
}
```

and `advance_phase` only moves to `_NEXT_PHASE[current]`. There is no way to go backward.

When Sonnet realized the stretch was bad and wanted to redo the linear, it called `advance_phase`. That took it from NONLINEAR to EXPORT. The agent's narration after the fact:

> "The system only allows forward movement in phases. When I called `advance_phase`, it moved to Export instead of going back to Linear."

This is a real workflow need. Backtracking — redoing a linear step after seeing the stretched result is bad — is standard practice in human astrophotography. The operator says "let me rewind to before the stretch and try again."

**Proposal.** Add a symmetric `rewind_phase` tool (or a `target_phase` argument on `advance_phase`). The rewind:

- Must explicitly declare intent: `rewind_phase(target="linear", reason="stretch revealed residual gradient; need to redo gradient removal")`.
- Restores `current_image` to the last checkpoint or saved snapshot from that phase, or to the canonical phase-entry file (e.g., `master_light_crop.fit` for LINEAR entry).
- Logs the rewind distinctly from advances so the processing log and HITL UI show it clearly.
- Clears any `variant_pool` from the phases it is unwinding past.

The rewind target list can be restricted (you can't rewind past a phase whose inputs are gone — e.g., can't go back to STACKING if the calibrated sequence was cleaned up).

**Simpler interim version.** If a full rewind is too invasive, expose a `return_to_phase(target="linear", snapshot="master_light_crop")` variant that reads from named snapshots and re-enters the earlier phase. The snapshot registry is already implicit — each phase has a canonical output — so this is mostly wiring.

### 3b. Phase gate blocks cross-phase utility tools that pros compose

Sonnet correctly attempted an HDR masked-stretch workflow while in STRETCH phase:

```
stretch_image (core-protected) → stretch_image (faint boost) → create_mask → pixel_math blend
```

The phase gate rejected `create_mask` in STRETCH. Agent had to advance to NONLINEAR first, do the blend there, and then proceed. But the blend is conceptually a stretch-phase operation — its result is what STRETCH should produce.

This is a category error in the phase model. `create_mask` is a utility. It doesn't operate in a phase-specific way. Denying it in STRETCH forces the agent to fake a phase advance before it is actually done stretching.

**Proposal.** Move `create_mask` into the always-available utility group (like `pixel_math`, `analyze_image`, `plate_solve`). The tool is pure: it takes an image and options and returns a mask file. It does not care what phase you are in.

Similarly audit the other phase-gated tools for ones that are functionally utilities and could be re-homed to utility: `saturation_adjust`, `curves_adjust`, and `local_contrast_enhance` are non-linear-only for good physics reasons, but `pixel_math` is already global, so `create_mask` as its companion should be too.

### 3c. `analyze_frames` in REGISTRATION

Minor but telling. Sonnet called `analyze_frames` while in REGISTRATION to check the register results. Got phase-gated. That is functionally backwards — the tool reads the registration cache, which is what you'd want right after registering. The gate exists to keep analysis on the ANALYSIS phase, but `analyze_frames` is also the natural diagnostic for "did my register work?"

**Proposal.** Expose `analyze_frames` as a utility (available in REGISTRATION, ANALYSIS, and subsequent phases) rather than restricting it to ANALYSIS. The tool is read-only against the registration cache; there is no state risk in letting the agent call it early.

### Broader principle

Phase gates should not prevent the agent from using tools that a human would reach for while thinking in that phase. Their purpose is to prevent physics mistakes (applying linear tools to stretched data). If a tool is not physics-sensitive, it should be a utility or at worst advisory — a warning, not a block.

## Issue #4 — "No narration" pressure + silent-tool backstop = lose/lose

The SYSTEM_BASE prompt says: _"If you respond with text instead of a tool call, you will be redirected to act."_ The framework also has a silent-tool backstop that fires when three HITL-mapped tools run without narration (nodes.py; triggered in the Sonnet log after six back-to-back `stretch_image` calls).

The combination punishes the agent on both sides:

- If it narrates, it gets "redirected to act."
- If it doesn't narrate, after three quiet tool calls the backstop forces a HITL interrupt anyway.

Net effect in Sonnet's case: six stretch variants in 90 seconds, each worse than the last, no reflection between them, and the human ends up intervening when the backstop fires. Kimi happened to narrate between tool calls so never tripped the backstop, but took the same damage-from-rushing on the M20 saturation passes.

**Fix.** Remove the "redirected to act" instruction. Soften to:

> Think briefly between tool calls when a decision is nontrivial. A sentence or two of reasoning before a consequential step produces better outcomes. Don't narrate routine actions and don't summarize completed work for the human — but don't suppress thinking.

Keep the silent-tool backstop — it is a useful safety net — but re-frame the reason it fires. Current-style: forces an HITL interrupt. Better: the framework nudges the agent with a ToolMessage like "You've run 3 tools in a row without analysis. Pause and evaluate the current state with `analyze_image` or reason in text before the next call." Only escalate to HITL if the agent doesn't respond.

## Issue #5 — HITL response handling: the agent doesn't converse

Both runs show this: when the user responds during HITL with words rather than approve/reject, the agent reads the words as a cue and immediately re-tools without acknowledging the message or engaging with the content. From Sonnet's run:

```
HITL response: 'this image has completely removed almost all the color ... m20 is about gone.'
Agent response: tool_calls (['stretch_image'])   # no acknowledgement
```

The user gave specific feedback. The agent should respond conversationally: confirm what it heard, describe its plan, _then_ tool-call. That is how a collaborator behaves. The current behavior reads as "your feedback is just a prompt for my next action."

**Fix.** In the HITL resume branch, the prompt should say something like:

> If the user's HITL response contains conversation or feedback rather than an approval token, reply in text first: acknowledge what they said, state your interpretation, and describe your plan. Then tool-call.

And relax the "no narration" rule after HITL input specifically — that is exactly when narration is appropriate.

## Issue #6 — `advance_phase` gate checks are valuable but incomplete

The existing phase gate checks (t30, lines 115–179) are good at catching skipped work — for example, refusing to advance from CALIBRATION if masters are missing. They don't catch regressed work: advancing from LINEAR after gradient_magnitude went 0.012 → 0.462 is allowed.

**Proposal.** Add regression gates:

- LINEAR → STRETCH should refuse when a tool returned a `regression_warning` since the last phase entry and the warning wasn't explicitly addressed (revert + re-run, or user acknowledged).
- NONLINEAR → EXPORT should refuse when the last nonlinear tool introduced visible clipping or the target-type visibility checklist (v2 prompt) flags missing features.

The gates return the same kind of structured error as the existing "missing masters" case, so the agent gets a concrete instruction rather than a vague "don't advance."

## Summary of required changes

| Area | Change | Effort | Blast radius |
|---|---|---|---|
| `t15_star_removal.py` | Set `paths.current_image = starless_fits` in the return Command | 3 lines | contained |
| `t19_star_restoration.py` | Verify it sets `current_image` back to the restored file | audit + ~3 lines if missing | contained |
| `t31_checkpoint.py` | Drop `explicit_path` proposal; add `noop` flag to restore response | 10 lines | contained |
| Restore-loop detector | Stuck-loop detector for 3 consecutive no-op restores | ~20 lines in nodes.py | contained |
| `t31_commit_variant.py` + `promote_variant` HumanMessage | Graceful "already_committed" path; adjust HITL approval message to suppress spurious commit calls | ~20 lines | contained |
| Phase gate — `create_mask`, `analyze_frames` as utilities | Move registry entries | ~10 lines in `registry.py` | small |
| `rewind_phase` tool (or `advance_phase` backward variant) | New tool + integration | ~80 lines + tests | moderate |
| Prompt — remove "no narration" pressure | String replacement | 1 paragraph | contained |
| Silent-tool backstop | Nudge via ToolMessage before escalating to HITL | ~30 lines in nodes.py | contained |
| HITL conversation cue | Prompt addition | 1 paragraph | contained |
| Regression gates on `advance_phase` | New checks keyed off `regression_warning` / visibility checklist | ~40 lines in t30 | contained |

Total: none of these changes is architectural. The framework is largely sound; it just needs the post-conditions tightened so the agent never has to reason about files, and the transitions loosened so the agent is not forced to fake phase advances to reach for legitimate tools.

## What changed vs. the original audit

- `t31_checkpoint_patch.md` (original): superseded. Agent-facing `explicit_path` argument is the wrong direction. Fix is at the producing tool (`star_removal` sets `current_image`), not the consuming tool.
- Original §4.4 "Checkpoint semantics are easy to misuse" reads as if the agent needs better discipline. It does not. The framework writes the wrong state and the agent correctly works from it.
- Original §8 "Model-Size Reading" claimed a stronger model would produce marginally better output. Sonnet 4.6 produced a _worse_ output because it attempted more pro moves (HDR composite, aggressive stretch variants, masked blend) and the framework's rigidity broke each one. The model-size finding must be retracted and replaced with: framework resilience matters more than model strength for this task. See the v2 README.
