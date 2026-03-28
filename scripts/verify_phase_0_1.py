#!/usr/bin/env python3
"""
Verify Phase 0 and Phase 1 are complete.

Run from project root:
    uv run python scripts/verify_phase_0_1.py

Exit 0 = all checks pass.  Exit 1 = one or more checks failed.
All checks run regardless of earlier failures so you see the full picture.
"""

import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv

load_dotenv(project_root / ".env")
load_dotenv(project_root / "muphrid" / ".env")  # fallback if stored in package

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "\u2713" if ok else "\u2717"
    msg = f"  {status} {name}"
    if detail:
        msg += f" \u2014 {detail}"
    print(msg)
    if not ok:
        _failures.append(name)


# ---------------------------------------------------------------------------
# Phase 0 — Config
# ---------------------------------------------------------------------------
print("Phase 0+1 Verification\n" + "=" * 40)
print("\nPhase 0 \u2014 Config")

try:
    from muphrid.config import load_settings, check_dependencies, PROFILE_DEFAULTS, make_llm
    check("imports (config module)", True)
except Exception as e:
    check("imports (config module)", False, str(e))
    print("\nCannot continue — config module failed to import.")
    sys.exit(1)

settings = None
try:
    settings = load_settings()
    check("load_settings()", True)
except Exception as e:
    check("load_settings()", False, str(e))

if settings is not None:
    provider = settings.llm_provider
    check(
        f"API key present for provider '{provider}'",
        bool(
            settings.together_api_key
            if provider == "together"
            else settings.anthropic_api_key
            if provider == "anthropic"
            else True  # openai key not stored in Settings
        ),
        "set in .env or shell environment",
    )

    expected_profiles = {"conservative", "balanced", "aggressive"}
    missing = expected_profiles - set(PROFILE_DEFAULTS.keys())
    check(
        "PROFILE_DEFAULTS has all expected profiles",
        not missing,
        f"missing: {missing}" if missing else f"keys: {sorted(PROFILE_DEFAULTS.keys())}",
    )

    try:
        check_dependencies(settings)
        check("check_dependencies()", True)
    except Exception as e:
        check("check_dependencies()", False, str(e))

    try:
        llm = make_llm(settings)
        check("make_llm()", llm is not None, type(llm).__name__)
    except Exception as e:
        check("make_llm()", False, str(e))

# ---------------------------------------------------------------------------
# Phase 0 — Siril engine
# ---------------------------------------------------------------------------
print("\nPhase 0 \u2014 Siril engine")

try:
    from muphrid.tools._siril import (
        run_siril_script,
        build_script,
        SirilResult,
        SirilError,
    )
    check("imports (_siril module)", True)
except Exception as e:
    check("imports (_siril module)", False, str(e))
    _failures.append("_siril import")
else:
    script = build_script([], requires="1.4.0")
    check(
        "build_script()",
        "requires 1.4.0" in script and "close" in script,
        "requires + close present",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = run_siril_script([], tmpdir, timeout=15)
            check(
                "run_siril_script()",
                isinstance(result, SirilResult),
                f"exit_code={result.exit_code}",
            )
            check(
                "SirilResult has stdout/stderr",
                hasattr(result, "stdout") and hasattr(result, "stderr"),
            )
        except SirilError as e:
            check("run_siril_script()", False, f"SirilError: {e}")
        except FileNotFoundError as e:
            check("run_siril_script()", False, f"binary not found: {e}")
        except Exception as e:
            check("run_siril_script()", False, str(e))

# ---------------------------------------------------------------------------
# Phase 1 — State schemas
# ---------------------------------------------------------------------------
print("\nPhase 1 \u2014 State schemas")

try:
    from muphrid.graph.state import (
        AstroState,
        PathState,
        MasterPaths,
        Metadata,
        Metrics,
        Dataset,
        FileInventory,
        AcquisitionMeta,
        FrameMetrics,
        ReportEntry,
        HITLPayload,
        ProcessingPhase,
        is_linear_phase,
        make_empty_state,
    )
    check("imports (state module)", True)
except Exception as e:
    check("imports (state module)", False, str(e))
    _failures.append("state import")
else:
    check("AstroState defined", "phase" in AstroState.__annotations__)
    check("PathState has latest_preview", "latest_preview" in PathState.__annotations__)
    check("MasterPaths defined", "bias" in MasterPaths.__annotations__ or len(MasterPaths.__annotations__) > 0)
    check("ReportEntry defined", "tool" in ReportEntry.__annotations__)

    # ProcessingPhase enum
    check("ProcessingPhase.LINEAR", ProcessingPhase.LINEAR.value == "linear")
    check("ProcessingPhase.NONLINEAR", ProcessingPhase.NONLINEAR.value == "nonlinear")
    check("is_linear_phase(LINEAR) == True", is_linear_phase(ProcessingPhase.LINEAR))
    check("is_linear_phase(NONLINEAR) == False", not is_linear_phase(ProcessingPhase.NONLINEAR))

    # HITLPayload fields — these are the stable contract for the presenter pattern
    required_hitl_fields = {"trigger", "checkpoint", "question", "options",
                            "allow_free_text", "preview_paths", "preview_labels", "context"}
    actual_hitl_fields = set(HITLPayload.__annotations__.keys())
    missing_hitl = required_hitl_fields - actual_hitl_fields
    check(
        "HITLPayload has all required fields",
        not missing_hitl,
        f"missing: {missing_hitl}" if missing_hitl else f"{sorted(actual_hitl_fields)}",
    )

    # make_empty_state round-trip
    ds: Dataset = {
        "id": "test",
        "working_dir": "/tmp",
        "files": {"lights": [], "darks": [], "flats": [], "biases": []},
        "acquisition_meta": {},
    }
    session = {
        "target_name": "test",
        "bortle": 4,
        "sqm_reading": None,
        "remove_stars": None,
        "notes": None,
    }
    try:
        state = make_empty_state(ds, session)
        check("make_empty_state() returns AstroState", state["phase"] == ProcessingPhase.INGEST)
        check("make_empty_state() messages list", isinstance(state.get("messages", []), list))
        check("make_empty_state() session present", state.get("session") == session)
    except Exception as e:
        check("make_empty_state()", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 40)
if _failures:
    print(f"FAILED ({len(_failures)} check(s)):")
    for f in _failures:
        print(f"  \u2717 {f}")
    sys.exit(1)
else:
    print("Phase 0+1 verification complete — all checks passed.")
    sys.exit(0)
