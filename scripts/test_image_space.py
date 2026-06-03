#!/usr/bin/env python3
"""
Focused smoke tests for the metadata.image_space authoritative render-state
contract.

Run from project root:
    uv run python scripts/test_image_space.py

Exit 0 = all checks pass.

What this exercises (not a full pipeline run):

  1. The structural drift check at module import — registry refuses to
     load if any current_image writer fails to also write image_space.
     We synthesize a violator and prove the static analyzer catches it.

  2. Writer round-trips — every writer that emits paths.current_image
     also emits a valid metadata.image_space ('linear' | 'display').
     We don't run the underlying tools (they'd need Siril); we rely on
     the static drift check + spot-check the literal payloads.

  3. State authority on read sites — export refuses when state's
     image_space is missing or invalid. analyze_image does NOT clobber
     image_space (its metadata delta only touches last_analysis_snapshot).

  4. Checkpoint round-trip — save_checkpoint and auto_checkpoint both build
     entries via the single writer make_checkpoint_entry, which records
     image_space alongside path and is the only image_space guard.

  5. commit_export round-trip — refuses without tentative_export, moves
     files on success, clears the tentative marker.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)


_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "ok" if ok else "FAIL"
    msg = f" {status} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


# ── 1. Structural drift check catches violators ──────────────────────────────


def test_drift_check_catches_violator() -> None:
    """
    Synthesize a tool function whose Command.update writes paths.current_image
    but NOT metadata.image_space, then run the same AST predicate the
    registry uses, and confirm it flags the violation.
    """
    print("\n[1] structural drift check catches missing image_space writers")

    bad_src = textwrap.dedent(
        """
        def fake_tool():
            return Command(update={
                "paths": {"current_image": "x.fit"},
                "messages": ["hi"],
            })
        """
    )
    good_src = textwrap.dedent(
        """
        def fake_tool():
            return Command(update={
                "paths": {"current_image": "x.fit"},
                "metadata": {"image_space": "linear"},
                "messages": ["hi"],
            })
        """
    )

    def _has_key(d: ast.Dict, key: str) -> bool:
        for k in d.keys:
            if isinstance(k, ast.Constant) and k.value == key:
                return True
        return False

    def _get_value(d: ast.Dict, key: str) -> ast.expr | None:
        for k, v in zip(d.keys, d.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
        return None

    def violates(src: str) -> bool:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Name) and f.id == "Command"):
                continue
            update_kw = next((kw for kw in node.keywords if kw.arg == "update"), None)
            if not update_kw or not isinstance(update_kw.value, ast.Dict):
                continue
            ud: ast.Dict = update_kw.value
            paths_val = _get_value(ud, "paths")
            if not isinstance(paths_val, ast.Dict):
                continue
            if not _has_key(paths_val, "current_image"):
                continue
            md_val = _get_value(ud, "metadata")
            ok = isinstance(md_val, ast.Dict) and _has_key(md_val, "image_space")
            if not ok:
                return True
        return False

    check("violator detected", violates(bad_src) is True)
    check("conforming payload accepted", violates(good_src) is False)


# ── 2. Registered tools all pass the structural drift check ──────────────────


def test_registry_imports_clean() -> None:
    """
    Importing muphrid.graph.registry runs both _assert_no_schema_drift and
    _assert_image_space_writers at module load. If either trips, the import
    raises ImportError. Reaching all_tools() proves both checks are clean.
    """
    print("\n[2] registry imports cleanly with all drift checks")

    try:
        from muphrid.graph.registry import all_tools
        tools = all_tools()
        check(
            "registry import + drift checks pass",
            isinstance(tools, list) and len(tools) > 0,
            f"{len(tools)} tools registered",
        )
    except ImportError as e:
        # Print the FULL error, no truncation, so the violator list is visible.
        print()
        print("== FULL ImportError ==")
        print(str(e))
        print("== end ==")
        check("registry import + drift checks pass", False, "see ImportError above")


# ── 3. Read sites refuse on missing/invalid image_space ──────────────────────


def test_export_refuses_missing_image_space() -> None:
    """export.export_final must refuse when state.metadata.image_space is missing."""
    print("\n[3] export_final refuses on missing image_space")

    from muphrid.tools.utility.export import export_final

    state = {
        "dataset": {"working_dir": tempfile.mkdtemp(prefix="img_space_test_")},
        "paths": {"current_image": "/nonexistent.fit"},
        "metadata": {}, # no image_space
        "metrics": {"is_linear_estimate": True},
    }

    try:
        export_final.func(state=state, tool_call_id="test")
        check("missing image_space rejected", False, "export_final did not raise")
    except RuntimeError as e:
        msg = str(e).lower()
        check(
            "missing image_space rejected",
            "image_space" in msg and ("missing" in msg or "invalid" in msg),
            "RuntimeError raised with image_space context",
        )
    except Exception as e:
        check("missing image_space rejected", False, f"unexpected: {type(e).__name__}: {e}")

    # Invalid value should also be rejected.
    state["metadata"] = {"image_space": "rainbow"}
    try:
        export_final.func(state=state, tool_call_id="test")
        check("invalid image_space rejected", False, "export_final did not raise")
    except RuntimeError as e:
        check("invalid image_space rejected", "image_space" in str(e).lower())
    except Exception as e:
        check("invalid image_space rejected", False, f"unexpected: {type(e).__name__}: {e}")


def test_export_picks_correct_source_profile() -> None:
    """
    With image_space='display', the default source_profile should resolve
    to 'sRGB' (data already gamma-encoded). With 'linear' it should be
    'sRGBlinear'. We verify by inspecting the function (not running Siril).
    """
    print("\n[4] export_final picks ICC source profile from image_space")

    from muphrid.tools.utility import export
    src = inspect_source(export.export_final.func)
    check(
        "display path → sRGB",
        '"sRGBlinear" if incoming_image_space == "linear" else "sRGB"' in src,
        "source-profile branch present",
    )


def test_export_review_is_system_owned() -> None:
    """
    Final export review must be system-owned: export_final derives tentative
    staging from export_final HITL policy, commit_export is not model-bound, and
    backend commit updates metadata without mutating paths.current_image.
    """
    print("\n[4b] export review is system-owned")

    from muphrid.graph.registry import tools_for_phase
    from muphrid.graph.state import ProcessingPhase
    from muphrid.tools.utility import export

    src = inspect_source(export.export_final.func)
    check(
        "export_final derives tentative from export_final",
        'is_enabled("export_final")' in src and "effective_tentative" in src,
    )

    export_tool_names = {t.name for t in tools_for_phase(ProcessingPhase.EXPORT)}
    check(
        "commit_export not model-bound in export phase",
        "commit_export" not in export_tool_names,
        f"tools={sorted(export_tool_names)}",
    )

    wd = tempfile.mkdtemp(prefix="backend_commit_")
    final_dir = Path(wd) / "export"
    staging = final_dir / ".tentative_master"
    staging.mkdir(parents=True)
    jpg = staging / "master_web.jpg"
    jpg.write_bytes(b"jpg")
    state = {
        "dataset": {"working_dir": wd},
        "paths": {"current_image": str(Path(wd) / "working.fit")},
        "metadata": {
            "image_space": "display",
            "tentative_export": {
                "stem": "master",
                "write_dir": str(staging),
                "final_dir": str(final_dir),
                "exported_files": [
                    {"path": str(jpg), "format": "jpg",
                     "icc_profile": "sRGB", "file_size_mb": 0.0},
                ],
                "preview_jpg": str(jpg),
            },
        },
    }
    try:
        update, _summary = export.commit_export_update(state, note="approved")
        check("backend export commit does not mutate paths", "paths" not in update)
        check("backend export commit sets export_done", update["metadata"]["export_done"] is True)
    except Exception as e:
        check("backend export commit", False, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def test_preview_cache_is_render_mode_aware() -> None:
    """Preview filenames must differ for linear autostretch vs display faithful."""
    print("\n[4c] preview cache is render-mode aware")

    from muphrid.tools.utility import generate_preview
    import muphrid.gradio_app as gradio_app

    preview_src = inspect_source(generate_preview.generate_preview)
    gradio_src = inspect_source(gradio_app._convert_fits_to_preview)
    check(
        "generate_preview includes render-mode suffix",
        "linear_autostretch" in preview_src and "display_faithful" in preview_src,
    )
    check(
        "gradio preview cache includes render-mode suffix",
        "linear_autostretch" in gradio_src and "display_faithful" in gradio_src,
    )


def inspect_source(func) -> str:
    import inspect as _inspect
    return _inspect.getsource(func)


# ── 4. analyze_image does not clobber image_space ────────────────────────────


def test_analyze_image_does_not_clobber_image_space() -> None:
    """
    analyze_image's only metadata write is last_analysis_snapshot. Because
    metadata uses the deep-merge reducer, image_space survives.
    """
    print("\n[5] analyze_image preserves image_space")

    from muphrid.tools.utility import analyze
    src = inspect_source(analyze.analyze_image.func)
    # Only one metadata literal in the body, and it must NOT contain image_space.
    check(
        "no image_space write in analyze_image",
        '"image_space"' not in src,
        "image_space key not present in analyze_image source",
    )
    check(
        "metadata delta is last_analysis_snapshot only",
        '"metadata": {"last_analysis_snapshot"' in src,
        "delta-only emit confirmed",
    )


# ── 5. Checkpoint round-trip ─────────────────────────────────────────────────


def test_checkpoint_save_restore_round_trip() -> None:
    """
    All checkpoint entries are {"path", "image_space"}, built by the single
    writer make_checkpoint_entry. save_checkpoint and the graph's
    auto_checkpoint both go through it, so restore trusts the shape and
    re-asserts image_space without defensive checks.
    """
    print("\n[6] checkpoint save/restore round-trip preserves image_space")

    from muphrid.tools.nonlinear.checkpoint import (
        make_checkpoint_entry,
        restore_checkpoint,
        save_checkpoint,
    )

    wd = tempfile.mkdtemp(prefix="cp_test_")
    fit = Path(wd) / "img.fit"
    fit.write_bytes(b"fake")

    state = {
        "dataset": {"working_dir": wd},
        "paths": {"current_image": str(fit)},
        "metadata": {"image_space": "display"},
        "metrics": {},
    }

    try:
        cmd = save_checkpoint.func(name="ck1", state=state, tool_call_id="test")
        entry = cmd.update["metadata"]["checkpoints"]["ck1"]
        check(
            "save records {path, image_space}",
            isinstance(entry, dict) and entry.get("image_space") == "display",
            f"entry={entry}",
        )
    except Exception as e:
        check("save records {path, image_space}", False, f"{type(e).__name__}: {e}")

    # Now restore from a state that already has the new-format checkpoint.
    state["metadata"]["checkpoints"] = {
        "ck1": {"path": str(fit), "image_space": "display"}
    }
    try:
        cmd = restore_checkpoint.func(name="ck1", state=state, tool_call_id="test")
        check(
            "restore re-asserts image_space",
            cmd.update["metadata"]["image_space"] == "display",
        )
    except Exception as e:
        check("restore re-asserts image_space", False, f"{type(e).__name__}: {e}")

    # The single writer is the only place the entry shape is constructed and
    # the only place image_space is guarded. There is no bare-string format to
    # support, so readers do not defensively re-check the shape.
    entry = make_checkpoint_entry(str(fit), "linear")
    check(
        "make_checkpoint_entry builds {path, image_space}",
        entry == {"path": str(fit), "image_space": "linear"},
        f"entry={entry}",
    )
    try:
        make_checkpoint_entry(str(fit), "garbage")
        check("make_checkpoint_entry refuses invalid image_space", False, "did not raise")
    except ValueError:
        check("make_checkpoint_entry refuses invalid image_space", True)
    except Exception as e:
        check(
            "make_checkpoint_entry refuses invalid image_space",
            False,
            f"unexpected: {type(e).__name__}: {e}",
        )

    # The graph's auto_checkpoint must emit the SAME structured entry shape —
    # this is the writer that historically wrote bare strings.
    from langchain_core.messages import AIMessage

    from muphrid.graph.nodes import auto_checkpoint
    from muphrid.graph.state import ProcessingPhase

    auto_state = {
        "phase": ProcessingPhase.LINEAR,
        "paths": {"current_image": str(fit)},
        "metadata": {"image_space": "linear"},
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "remove_gradient",
                    "args": {},
                    "id": "call_1",
                    "type": "tool_call",
                }],
            )
        ],
    }
    auto_update = auto_checkpoint(auto_state)
    auto_entries = (auto_update.get("metadata", {}).get("checkpoints") or {})
    check(
        "auto_checkpoint emits {path, image_space} entries",
        bool(auto_entries)
        and all(
            isinstance(v, dict) and v.get("image_space") == "linear" and "path" in v
            for v in auto_entries.values()
        ),
        f"entries={auto_entries}",
    )

    # If image_space is missing, auto_checkpoint records NO checkpoint entry
    # rather than write an unrestorable one. (It may still emit the
    # pre_action_image capture for the effect detector — that's orthogonal.)
    auto_state_no_space = {**auto_state, "metadata": {}}
    skipped = auto_checkpoint(auto_state_no_space)
    check(
        "auto_checkpoint records no checkpoint when image_space missing",
        "checkpoints" not in skipped.get("metadata", {}),
        f"got={skipped}",
    )


# ── 6. commit_export round-trip ──────────────────────────────────────────────


def test_commit_export_round_trip() -> None:
    """
    commit_export refuses without tentative_export; on success, moves files
    from staging into the final dir and clears the tentative marker.
    """
    print("\n[7] commit_export tentative → final round-trip")

    from muphrid.tools.utility.export import commit_export

    wd = tempfile.mkdtemp(prefix="commit_export_test_")
    final_dir = Path(wd) / "export"
    staging = final_dir / ".tentative_master"
    staging.mkdir(parents=True)

    tif = staging / "master.tif"
    jpg = staging / "master_web.jpg"
    tif.write_bytes(b"tif")
    jpg.write_bytes(b"jpg")

    state_no_tentative = {
        "dataset": {"working_dir": wd},
        "paths": {},
        "metadata": {"image_space": "display"},
        "metrics": {},
    }
    try:
        commit_export.func(state=state_no_tentative, tool_call_id="test")
        check("refuses without tentative_export", False, "did not raise")
    except RuntimeError as e:
        check("refuses without tentative_export", "tentative_export" in str(e))
    except Exception as e:
        check("refuses without tentative_export", False, f"unexpected: {type(e).__name__}: {e}")

    state_with_tentative = {
        "dataset": {"working_dir": wd},
        "paths": {},
        "metadata": {
            "image_space": "display",
            "tentative_export": {
                "stem": "master",
                "write_dir": str(staging),
                "final_dir": str(final_dir),
                "exported_files": [
                    {"path": str(tif), "format": "tiff16",
                     "icc_profile": "Rec2020", "file_size_mb": 0.0},
                    {"path": str(jpg), "format": "jpg",
                     "icc_profile": "sRGB", "file_size_mb": 0.0},
                ],
                "preview_jpg": str(jpg),
            },
        },
        "metrics": {},
    }
    try:
        cmd = commit_export.func(state=state_with_tentative, tool_call_id="test")
        moved = cmd.update["metadata"]["exported_files"]
        check("commit reports moved files", len(moved) == 2)
        check(
            "tif moved into final_dir",
            (final_dir / "master.tif").exists() and not tif.exists(),
        )
        check(
            "jpg moved into final_dir",
            (final_dir / "master_web.jpg").exists() and not jpg.exists(),
        )
        check(
            "tentative_export cleared",
            cmd.update["metadata"]["tentative_export"] is None,
        )
        check(
            "export_done flipped",
            cmd.update["metadata"]["export_done"] is True,
        )
    except Exception as e:
        check("commit_export round-trip", False, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ── 7. Stretch is THE transition ─────────────────────────────────────────────


def test_stretch_writers_set_display() -> None:
    """
    stretch.stretch_image and select_stretch_variant must emit
    metadata.image_space="display" (the linear→display transition).
    Inspect the source instead of running Siril.
    """
    print("\n[8] stretch_image emits image_space='display'")

    from muphrid.tools.nonlinear import stretch
    src = inspect_source(stretch.stretch_image.func)
    check(
        'stretch_image: "image_space": "display"',
        '"image_space": "display"' in src,
    )

    src2 = inspect_source(stretch.select_stretch_variant.func)
    check(
        'select_stretch_variant: "image_space": "display"',
        '"image_space": "display"' in src2,
    )


# ── 8. Linear writers set linear ─────────────────────────────────────────────


def test_linear_writers_set_linear() -> None:
    """
    Spot-check a few linear-stage writers: stack, crop, gradient, color
    calibrate, deconvolution.
    """
    print("\n[9] linear-stage writers emit image_space='linear'")

    from muphrid.tools.preprocess import stack, crop
    from muphrid.tools.linear import gradient, color_calibrate, deconvolution

    for tool, label in [
        (stack.siril_stack, "siril_stack"),
        (crop.auto_crop, "auto_crop"),
        (gradient.remove_gradient, "remove_gradient"),
        (color_calibrate.color_calibrate, "color_calibrate"),
        (deconvolution.deconvolution, "deconvolution"),
    ]:
        src = inspect_source(tool.func)
        check(
            f"{label}: image_space='linear'",
            '"image_space": "linear"' in src,
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    print("Image-space contract tests")
    test_drift_check_catches_violator()
    test_registry_imports_clean()
    test_export_refuses_missing_image_space()
    test_export_picks_correct_source_profile()
    test_export_review_is_system_owned()
    test_preview_cache_is_render_mode_aware()
    test_analyze_image_does_not_clobber_image_space()
    test_checkpoint_save_restore_round_trip()
    test_commit_export_round_trip()
    test_stretch_writers_set_display()
    test_linear_writers_set_linear()

    print()
    if _failures:
        print(f"FAIL — {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f" - {f}")
        return 1
    print("All image_space contract tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
