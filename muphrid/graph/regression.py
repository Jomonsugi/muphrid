"""
Regression detection for analyze_image.

Pure helper module. No I/O, no state mutation — just compares two metrics
snapshots and returns RegressionWarning entries for any monotonic quality
metric that worsened beyond configured thresholds.

Philosophy:
  - Informational, not prescriptive. The framework emits warnings but never
    blocks. The agent decides whether to revert, re-parameterize, or proceed.
  - Conservative thresholds. A warning should correspond to a meaningful
    change, not histogram jitter. Both an absolute AND a relative threshold
    must be crossed.
  - Phase-agnostic metrics only. Metrics whose "worse" direction inverts
    depending on phase (e.g. star_count after intentional star_removal,
    histogram_skewness after stretch) are deliberately excluded — noisy
    warnings waste attention. The agent's own judgment covers phase-specific
    cases.

Exposed symbols:
  - METRIC_RULES: the monitored-metric table
  - detect_regressions(current, baseline, phase) -> list[RegressionWarning]
  - filter_resolved(warnings, current, tolerance_ratio) -> list[RegressionWarning]
  - merge_warnings(existing, new, current) -> list[RegressionWarning]
  - format_warnings(warnings) -> str
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from muphrid.graph.state import RegressionWarning


# ── Monitored metrics ─────────────────────────────────────────────────────────

# For each monitored metric:
#   direction:  "higher" or "lower" — which way is "worse"
#   min_abs:    absolute-delta threshold in the metric's native units
#   min_rel:    relative-delta threshold (fraction; applies when baseline ≠ 0)
#   label:      human-readable name for the summary string
#   unit:       display unit ("%", "", "dB", etc.) for the summary string
#
# Both thresholds must be crossed for a warning to fire — this keeps
# near-zero baselines from producing spurious relative-delta warnings and
# keeps large-baseline jitter from producing spurious absolute-delta
# warnings. Tune conservatively; rare false positives are better than a
# warning firehose the agent learns to ignore.

METRIC_RULES: dict[str, dict[str, Any]] = {
    "clipped_shadows_pct": {
        "direction": "higher",
        "min_abs":   0.5,    # +0.5 percentage points
        "min_rel":   0.25,   # +25% relative
        "label":     "Shadow clipping",
        "unit":      "%",
    },
    "clipped_highlights_pct": {
        "direction": "higher",
        "min_abs":   0.3,
        "min_rel":   0.25,
        "label":     "Highlight clipping",
        "unit":      "%",
    },
    "gradient_magnitude": {
        "direction": "higher",
        "min_abs":   0.01,
        "min_rel":   0.25,
        "label":     "Gradient magnitude",
        "unit":      "",
    },
    "background_flatness": {
        "direction": "lower",
        "min_abs":   0.05,   # drop of at least 0.05 on the 0–1 scale
        "min_rel":   0.10,
        "label":     "Background flatness",
        "unit":      "",
    },
    "current_noise": {
        "direction": "higher",
        "min_abs":   0.0,    # any relative increase that crosses min_rel counts
        "min_rel":   0.25,
        "label":     "Noise",
        "unit":      "",
    },
    "wavelet_noise": {
        "direction": "higher",
        "min_abs":   0.0,
        "min_rel":   0.25,
        "label":     "Wavelet noise",
        "unit":      "",
    },
    "channel_imbalance": {
        "direction": "higher",
        "min_abs":   0.01,
        "min_rel":   0.25,
        "label":     "Channel imbalance",
        "unit":      "",
    },
    "snr_estimate": {
        "direction": "lower",
        "min_abs":   0.0,
        "min_rel":   0.15,   # 15% SNR drop is significant
        "label":     "SNR estimate",
        "unit":      "",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_regressions(
    current: dict | None,
    baseline: dict | None,
    phase: str,
    *,
    now: datetime | None = None,
) -> list[RegressionWarning]:
    """
    Compare two metrics snapshots and return warnings for any monitored
    metric that worsened past its configured thresholds.

    Args:
      current:  metrics dict from the just-completed analyze_image
      baseline: metrics dict from the prior analyze_image (from
                metadata.last_analysis_snapshot). If None or missing keys,
                no comparison runs for those keys.
      phase:    ProcessingPhase value at detection time (for the warning's
                phase_origin field).
      now:      clock-injection hook for tests; defaults to real UTC now.

    Returns a possibly-empty list of RegressionWarning dicts. Order
    follows METRIC_RULES key order for deterministic output.
    """
    if not current or not baseline:
        return []

    ts = (now or datetime.now(timezone.utc)).isoformat()
    warnings: list[RegressionWarning] = []

    for metric, rule in METRIC_RULES.items():
        cur = current.get(metric)
        base = baseline.get(metric)
        if cur is None or base is None:
            continue
        try:
            cur_v = float(cur)
            base_v = float(base)
        except (TypeError, ValueError):
            continue

        delta = cur_v - base_v
        rel = (delta / base_v) if base_v != 0 else None

        direction = rule["direction"]
        worse = (direction == "higher" and delta > 0) or (
            direction == "lower" and delta < 0
        )
        if not worse:
            continue

        abs_ok = abs(delta) >= rule["min_abs"]
        # Relative check: skip when baseline is zero (can't compute ratio);
        # the absolute threshold alone then decides.
        rel_ok = rel is not None and abs(rel) >= rule["min_rel"]
        if base_v == 0:
            triggered = abs_ok
        else:
            triggered = abs_ok and rel_ok
        if not triggered:
            continue

        summary = _format_summary(rule, base_v, cur_v, delta, rel)
        warnings.append(RegressionWarning(
            metric=metric,
            baseline=base_v,
            current=cur_v,
            delta=delta,
            relative_delta=rel,
            direction="worse",
            summary=summary,
            phase_origin=phase,
            detected_at=ts,
        ))

    return warnings


def filter_resolved(
    warnings: list[RegressionWarning],
    current: dict | None,
    *,
    tolerance_ratio: float = 0.20,
) -> list[RegressionWarning]:
    """
    Remove warnings whose metric has returned within tolerance of the
    warning's own baseline value (i.e., the "known-good" target).

    A warning is "resolved" when the current measurement is no worse than
    baseline * (1 + tolerance_ratio) for higher-is-worse metrics, or
    baseline * (1 - tolerance_ratio) for lower-is-worse metrics. The
    tolerance dead-band prevents churn from normal measurement jitter.

    Warnings whose metric is missing from `current` are retained as-is —
    we have no evidence of recovery so we leave the alert visible.
    """
    if not warnings or not current:
        return list(warnings)

    kept: list[RegressionWarning] = []
    for w in warnings:
        metric = w["metric"]
        rule = METRIC_RULES.get(metric)
        base = w.get("baseline")
        cur = current.get(metric)
        if rule is None or base is None or cur is None:
            kept.append(w)
            continue
        try:
            base_v = float(base)
            cur_v = float(cur)
        except (TypeError, ValueError):
            kept.append(w)
            continue

        direction = rule["direction"]
        if direction == "higher":
            threshold = base_v * (1 + tolerance_ratio) if base_v != 0 else rule["min_abs"]
            resolved = cur_v <= threshold
        else:  # "lower"
            threshold = base_v * (1 - tolerance_ratio) if base_v != 0 else -rule["min_abs"]
            resolved = cur_v >= threshold
        if not resolved:
            kept.append(w)

    return kept


def merge_warnings(
    existing: list[RegressionWarning],
    new: list[RegressionWarning],
    current: dict | None,
    *,
    now: datetime | None = None,
) -> list[RegressionWarning]:
    """
    Combine existing (pending) warnings with newly-detected ones, keyed by
    metric. Rules:

      - If a metric has both an existing warning and a new detection, the
        existing warning wins — its baseline value is preserved (it's the
        known-good reference from before the regression chain started) and
        its current/delta/relative_delta/summary/detected_at are refreshed
        to reflect the latest measurement.
      - If a metric has only an existing warning, its current/delta fields
        are still refreshed against `current` so the warning's summary
        string stays accurate across tool calls that didn't produce a new
        detection.
      - If a metric has only a new warning, it's added as-is.

    The resulting list is ordered by METRIC_RULES insertion order for
    deterministic output.
    """
    ts = (now or datetime.now(timezone.utc)).isoformat()
    by_metric: dict[str, RegressionWarning] = {
        w["metric"]: dict(w) for w in existing
    }

    # Refresh current/delta/summary on every kept warning against the latest
    # measurement, even when no new detection fired for that metric.
    if current:
        for metric, w in by_metric.items():
            cur = current.get(metric)
            base = w.get("baseline")
            if cur is None or base is None:
                continue
            try:
                cur_v = float(cur)
                base_v = float(base)
            except (TypeError, ValueError):
                continue
            delta = cur_v - base_v
            rel = (delta / base_v) if base_v != 0 else None
            w["current"] = cur_v
            w["delta"] = delta
            w["relative_delta"] = rel
            w["detected_at"] = ts
            rule = METRIC_RULES.get(metric)
            if rule is not None:
                w["summary"] = _format_summary(rule, base_v, cur_v, delta, rel)

    # Add any freshly-detected metric not already tracked. Existing warnings
    # for the same metric shadow the new detection (baseline preservation).
    for nw in new:
        if nw["metric"] not in by_metric:
            by_metric[nw["metric"]] = dict(nw)

    return [by_metric[m] for m in METRIC_RULES if m in by_metric]


def format_warnings(warnings: list[RegressionWarning]) -> str:
    """
    Render warnings as a short prose block suitable for inclusion in a
    ToolMessage content string or a Markdown log entry.

    Empty list yields an empty string so callers can concatenate without
    branching.
    """
    if not warnings:
        return ""
    lines = ["Outstanding regression warnings:"]
    for w in warnings:
        lines.append(f"  • {w['summary']}")
    return "\n".join(lines)


# ── Internal ──────────────────────────────────────────────────────────────────

def _format_summary(
    rule: dict[str, Any],
    base_v: float,
    cur_v: float,
    delta: float,
    rel: float | None,
) -> str:
    """Produce a single line like 'Highlight clipping rose 0.1% → 3.2% (+3.1%)'."""
    unit = rule["unit"]
    label = rule["label"]
    verb = "rose" if delta > 0 else "fell"
    base_str = _fmt(base_v, unit)
    cur_str = _fmt(cur_v, unit)
    delta_str = _fmt(delta, unit, signed=True)
    rel_str = f", {rel * 100:+.0f}%" if rel is not None else ""
    return f"{label} {verb} {base_str} → {cur_str} ({delta_str}{rel_str})"


def _fmt(v: float, unit: str, *, signed: bool = False) -> str:
    """Format a number with sensible precision for the given unit."""
    if unit == "%":
        fmt = "{:+.2f}%" if signed else "{:.2f}%"
        return fmt.format(v)
    # Unitless — pick precision by magnitude.
    if abs(v) >= 10:
        fmt = "{:+.1f}" if signed else "{:.1f}"
    elif abs(v) >= 1:
        fmt = "{:+.2f}" if signed else "{:.2f}"
    else:
        fmt = "{:+.3f}" if signed else "{:.3f}"
    return fmt.format(v)
