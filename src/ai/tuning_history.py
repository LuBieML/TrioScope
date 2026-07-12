"""Tuning iteration history — Qt-free logic.

Tuning is a loop: capture → ANALYZE → tweak Pn → repeat. This module keeps
one record per ANALYZE (key metrics + the axis's Pn snapshot at that
moment) and answers the question a tuning engineer asks every iteration:
*did my change make it better?* — by comparing each run against the most
recent earlier run on the same axis.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime

from .tuning_rules import tuning_score

# Attr → drive parameter code, for Pn-change summaries ("Pn102 500→600").
_PN_LABELS: dict[str, str] = {
    "drive_type": "Drive",
    "pn100_tuning_mode": "Pn100.0",
    "pn100_vibration": "Pn100.2",
    "pn100_damping": "Pn100.3",
    "pn101": "Pn101", "pn102": "Pn102", "pn103": "Pn103", "pn104": "Pn104",
    "pn106": "Pn106", "pn112": "Pn112", "pn113": "Pn113", "pn114": "Pn114",
    "pn115": "Pn115", "pn135": "Pn135",
}

MAX_RUNS = 50

# Relative change below which two values count as "same" (plus an absolute
# floor so noise-level values don't flip verdicts).
_REL_EPSILON = 0.05


@dataclass(frozen=True)
class KpiDef:
    """One comparable tuning metric.

    direction: "lower"  — smaller is better
               "higher" — larger is better (tuning score)
               "unity"  — closer to 1.0 is better
               "absent" — None/0 is good, any value is bad (e.g. oscillation)
    """
    key: str
    label: str
    unit: str
    direction: str
    abs_epsilon: float = 0.0
    fmt: str = "{:.4g}"


KPI_DEFS: tuple[KpiDef, ...] = (
    KpiDef("score", "Score", "/10", "higher", abs_epsilon=0.4, fmt="{:.1f}"),
    KpiDef("settle_ms", "Settle", "ms", "lower", abs_epsilon=2.0, fmt="{:.0f}"),
    KpiDef("fe_settle_peak", "FE pk", "u", "lower"),
    KpiDef("ringing", "Ring", "", "lower", fmt="{:.0f}"),
    KpiDef("fe_cruise_rms", "Cruise", "u", "lower"),
    KpiDef("fe_accel_peak", "Accel", "u", "lower"),
    KpiDef("osc_hz", "Osc", "Hz", "absent", fmt="{:.0f}"),
    KpiDef("vel_reach", "VelTrk", "", "unity", abs_epsilon=0.01, fmt="{:.3f}"),
    KpiDef("vel_overshoot_pct", "VelOS", "%", "lower", abs_epsilon=0.5,
           fmt="{:.1f}"),
    KpiDef("settle_spread_ms", "Spread", "ms", "lower", abs_epsilon=2.0,
           fmt="{:.0f}"),
)


@dataclass(eq=False)  # identity semantics — two runs can hold equal values
class TuningRun:
    """One ANALYZE result, condensed for comparison."""
    timestamp: str
    axis: int
    kpis: dict            # KPI key → float | None
    pn_snapshot: dict | None   # DriveProfile.to_dict() at analyze time
    context: dict              # n_moves, duration_s, band, …
    full_metrics: dict = field(repr=False, default_factory=dict)


# ---------------------------------------------------------------- extraction
def extract_kpis(metrics: dict) -> dict:
    """Pull the comparable scalars out of a SignalMetrics result dict."""
    settle = metrics.get("settle") or {}
    fe = metrics.get("fe") or {}
    vel = metrics.get("velocity") or {}
    osc_fe = (metrics.get("oscillation") or {}).get("fe") or {}

    settle_ms = settle.get("time_to_band_ms")
    if not settle.get("settled_within_window", True):
        settle_ms = None  # not settled — treated as worse than any number

    spread = None
    per_move = settle.get("per_move") or []
    times = [m["time_to_band_ms"] for m in per_move
             if m.get("time_to_band_ms") is not None]
    if len(times) >= 2:
        spread = max(times) - min(times)

    overshoot = vel.get("velocity_overshoot_per_move") or {}

    return {
        "score": tuning_score(metrics),
        "settle_ms": settle_ms,
        "fe_settle_peak": settle.get("fe_peak_during_settle"),
        "ringing": settle.get("zero_crossings"),
        "fe_cruise_rms": (fe.get("cruise") or {}).get("rms"),
        "fe_accel_peak": (fe.get("accel") or {}).get("peak_abs"),
        "osc_hz": (osc_fe.get("dominant_hz")
                   if osc_fe.get("has_significant_oscillation") else None),
        "vel_reach": vel.get("cruise_velocity_reach_ratio"),
        "vel_overshoot_pct": overshoot.get("max_pct"),
        "settle_spread_ms": spread,
    }


def make_run(metrics: dict, axis: int, pn_snapshot: dict | None,
             timestamp: str | None = None) -> TuningRun:
    cap = metrics.get("capture") or {}
    phases = metrics.get("phases") or {}
    return TuningRun(
        timestamp=timestamp or datetime.now().strftime("%H:%M:%S"),
        axis=axis,
        kpis=extract_kpis(metrics),
        pn_snapshot=dict(pn_snapshot) if pn_snapshot else None,
        context={
            "n_samples": cap.get("n_samples"),
            "duration_s": cap.get("duration_s"),
            "n_moves": phases.get("n_moves"),
            "band": cap.get("settle_band"),
            "band_source": cap.get("settle_band_source"),
        },
        full_metrics=metrics,
    )


# ---------------------------------------------------------------- comparison
def _same(a: float, b: float, kpi: KpiDef) -> bool:
    tol = max(kpi.abs_epsilon, _REL_EPSILON * max(abs(a), abs(b)))
    return abs(a - b) <= tol


def compare_kpi(prev: float | None, cur: float | None, kpi: KpiDef) -> str | None:
    """Verdict for one KPI vs the previous run.

    Returns "better" | "worse" | "same" | None (not comparable).
    """
    if kpi.direction == "absent":
        prev_bad = prev is not None
        cur_bad = cur is not None
        if prev_bad == cur_bad:
            return "same"
        return "worse" if cur_bad else "better"

    if prev is None and cur is None:
        return "same"
    if cur is None:
        return "worse"    # e.g. no longer settles within the window
    if prev is None:
        return "better"

    if kpi.direction == "unity":
        prev_err, cur_err = abs(prev - 1.0), abs(cur - 1.0)
    elif kpi.direction == "higher":
        prev_err, cur_err = -prev, -cur
    else:  # "lower"
        prev_err, cur_err = prev, cur
    if _same(prev_err, cur_err, kpi):
        return "same"
    return "better" if cur_err < prev_err else "worse"


def compare_runs(prev: TuningRun | None, cur: TuningRun) -> dict:
    """{kpi_key: verdict} for every KPI; all None when there is no baseline."""
    if prev is None:
        return {kpi.key: None for kpi in KPI_DEFS}
    return {
        kpi.key: compare_kpi(prev.kpis.get(kpi.key), cur.kpis.get(kpi.key), kpi)
        for kpi in KPI_DEFS
    }


def pn_changes(prev_snapshot: dict | None, snapshot: dict | None) -> list[str]:
    """Human-readable parameter changes between two Pn snapshots."""
    if not prev_snapshot or not snapshot:
        return []
    changes = []
    for attr, label in _PN_LABELS.items():
        old = prev_snapshot.get(attr)
        new = snapshot.get(attr)
        if old != new:
            changes.append(f"{label} {old if old is not None else '—'}"
                           f"→{new if new is not None else '—'}")
    return changes


def format_kpi(value: float | None, kpi: KpiDef) -> str:
    if value is None:
        return "∅" if kpi.key == "settle_ms" else "—"
    return kpi.fmt.format(value)


# ---------------------------------------------------------------- container
class TuningHistory:
    """Ordered run store (oldest → newest), capped at MAX_RUNS."""

    def __init__(self, max_runs: int = MAX_RUNS):
        self._runs: list[TuningRun] = []
        self._max_runs = max_runs

    def __len__(self) -> int:
        return len(self._runs)

    @property
    def runs(self) -> list[TuningRun]:
        return list(self._runs)

    def add(self, run: TuningRun) -> None:
        self._runs.append(run)
        if len(self._runs) > self._max_runs:
            self._runs = self._runs[-self._max_runs:]

    def clear(self) -> None:
        self._runs.clear()

    def previous_for(self, run: TuningRun) -> TuningRun | None:
        """Most recent earlier run on the same axis (the comparison baseline)."""
        idx = next(
            (i for i, r in enumerate(self._runs) if r is run),
            len(self._runs),
        )
        for earlier in reversed(self._runs[:idx]):
            if earlier.axis == run.axis:
                return earlier
        return None

    # ------------------------------------------------------------- export
    def to_csv(self) -> str:
        """All runs as CSV (chronological), including Pn values and changes."""
        pn_attrs = list(_PN_LABELS)
        header = (["timestamp", "axis"]
                  + [kpi.key for kpi in KPI_DEFS]
                  + ["n_moves", "duration_s", "band", "band_source"]
                  + pn_attrs + ["pn_changes"])
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        for run in self._runs:
            prev = self.previous_for(run)
            changes = pn_changes(prev.pn_snapshot if prev else None,
                                 run.pn_snapshot)
            snapshot = run.pn_snapshot or {}
            writer.writerow(
                [run.timestamp, run.axis]
                + [run.kpis.get(kpi.key) for kpi in KPI_DEFS]
                + [run.context.get("n_moves"), run.context.get("duration_s"),
                   run.context.get("band"), run.context.get("band_source")]
                + [snapshot.get(attr) for attr in pn_attrs]
                + ["; ".join(changes)]
            )
        return buf.getvalue()
