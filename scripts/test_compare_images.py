#!/usr/bin/env python3
"""
Focused tests for compare_images — read-only, metric-major comparison of
images by handle (variant id / checkpoint name / "current").

Run from project root:
    uv run python scripts/test_compare_images.py

Exit 0 = all checks pass.

What this exercises (the capability the agent lacked when comparing GraXpert
vs Siril gradient variants):
  1. Compare N images by handle in one call → metric-major table.
  2. Handles span variant ids, checkpoint names, and "current" — never paths.
  3. Read-only: the call mutates no state.
  4. Render-space tagging + cross-space warning.
  5. Unknown refs are refused with the available handles listed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

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


def _write_fits(path: Path, r_lvl: float, g_lvl: float, b_lvl: float) -> None:
    """3-layer (3,H,W) linear-ish frame with a per-channel background level."""
    h = w = 256
    rng = np.random.default_rng(0)
    chans = [
        (lvl + rng.normal(0, 0.002, (h, w))).clip(0, 1).astype("float32")
        for lvl in (r_lvl, g_lvl, b_lvl)
    ]
    fits.writeto(path, np.stack(chans), overwrite=True)


def _payload(cmd) -> dict:
    return json.loads(cmd.update["messages"][0].content)


def main() -> int:
    from muphrid.tools.utility.compare_images import compare_images

    wd = tempfile.mkdtemp(prefix="cmp_test_")
    img_green = Path(wd) / "a.fits"   # green-biased
    img_neutral = Path(wd) / "b.fits"
    img_red = Path(wd) / "c.fits"     # red-biased
    _write_fits(img_green, 0.10, 0.16, 0.12)
    _write_fits(img_neutral, 0.12, 0.12, 0.12)
    _write_fits(img_red, 0.16, 0.10, 0.10)

    state = {
        "dataset": {"working_dir": wd},
        "paths": {"current_image": str(img_neutral)},
        "metadata": {
            "image_space": "linear",
            "checkpoints": {
                "pre_gradient_linear": {"path": str(img_neutral), "image_space": "linear"},
            },
        },
        "variant_pool": [
            {"id": "remove_gradient_v1", "file_path": str(img_green),
             "image_space": "linear", "label": "graxpert div", "params": {"correction": "division"}},
            {"id": "remove_gradient_v2", "file_path": str(img_red),
             "image_space": "linear", "label": "subsky poly", "params": {"degree": 2}},
        ],
        "phase": "linear",
    }

    print("\n[1] metric-major comparison across variant ids + current")
    cmd = compare_images.func(
        refs=["remove_gradient_v1", "remove_gradient_v2", "current"],
        metrics=["color"], detect_stars=False, state=state, tool_call_id="t",
    )
    p = _payload(cmd)
    tbl = p.get("metrics", {})
    check("returns a metric-major table", isinstance(tbl, dict) and "green_excess" in tbl,
          f"keys={list(tbl)[:6]}")
    ge = tbl.get("green_excess", {})
    check("all three refs present in each metric row",
          set(ge) == {"remove_gradient_v1", "remove_gradient_v2", "current"}, f"row={ge}")
    check("green-biased variant has highest green_excess; red-biased lowest",
          ge.get("remove_gradient_v1", -9) > ge.get("current", 0) > ge.get("remove_gradient_v2", 9),
          f"row={ge}")

    print("\n[2] read-only — the call mutates no state")
    check("Command.update contains only messages",
          set(cmd.update.keys()) == {"messages"}, f"keys={set(cmd.update.keys())}")

    print("\n[3] legend tags handles + render space, never leaks paths")
    legend = p.get("compared", [])
    check("legend has one entry per ref", len(legend) == 3, f"n={len(legend)}")
    check("legend exposes no path field", all("path" not in e for e in legend))
    check("legend tags kind + image_space",
          all(e.get("kind") and e.get("image_space") == "linear" for e in legend))

    print("\n[4] checkpoint handle + 'all' metrics")
    cmd2 = compare_images.func(
        refs=["pre_gradient_linear", "remove_gradient_v1"],
        metrics=["all"], detect_stars=False, state=state, tool_call_id="t",
    )
    p2 = _payload(cmd2)
    check("checkpoint name resolves as a ref",
          {e["ref"] for e in p2["compared"]} == {"pre_gradient_linear", "remove_gradient_v1"})
    check("'all' returns many metric rows", len(p2["metrics"]) > 10, f"n={len(p2['metrics'])}")

    print("\n[5] cross render-space warning")
    state_mixed = json.loads(json.dumps(state))
    state_mixed["variant_pool"][0]["image_space"] = "display"
    cmd3 = compare_images.func(
        refs=["remove_gradient_v1", "remove_gradient_v2"],
        metrics=["color"], detect_stars=False, state=state_mixed, tool_call_id="t",
    )
    check("mixing linear+display emits image_space_warning",
          "image_space_warning" in _payload(cmd3))

    print("\n[6] unknown ref refused with available handles")
    cmd4 = compare_images.func(
        refs=["does_not_exist"], detect_stars=False, state=state, tool_call_id="t",
    )
    p4 = _payload(cmd4)
    check("unknown ref returns error + available lists",
          "error" in p4 and "remove_gradient_v1" in p4.get("available_variant_ids", [])
          and "pre_gradient_linear" in p4.get("available_checkpoints", []),
          f"payload keys={list(p4)}")

    print("\n[7] single ref works (read-only analysis by handle)")
    cmd5 = compare_images.func(
        refs=["remove_gradient_v2"], metrics=["snr", "color"],
        detect_stars=False, state=state, tool_call_id="t",
    )
    p5 = _payload(cmd5)
    check("single-ref comparison returns that ref's metrics",
          all(set(row) == {"remove_gradient_v2"} for row in p5["metrics"].values()),
          f"metrics={list(p5['metrics'])}")

    print()
    if _failures:
        print(f"FAIL — {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f" - {f}")
        return 1
    print("All compare_images tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
