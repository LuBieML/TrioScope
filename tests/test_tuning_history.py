"""Tests for the Qt-free tuning iteration history logic."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ai.signal_metrics import SignalMetrics
from src.ai.tuning_history import (
    KPI_DEFS, TuningHistory, compare_kpi, compare_runs, extract_kpis,
    format_kpi, make_run, pn_changes,
)

FS = 1000.0


def _kpi(key):
    return next(k for k in KPI_DEFS if k.key == key)


def _capture_metrics(fe_scale=1.0, seed=42):
    """compute_all result for a clean single move with scalable FE noise."""
    n_pre, n_acc, n_cr, n_post = 300, 200, 600, 800
    dvel = np.concatenate([
        np.zeros(n_pre),
        np.linspace(0.0, 100.0, n_acc, endpoint=False),
        np.full(n_cr, 100.0),
        np.linspace(100.0, 0.0, n_acc, endpoint=False),
        np.zeros(n_post),
    ])
    t = np.arange(len(dvel)) / FS
    dpos = np.cumsum(dvel) / FS
    rng = np.random.default_rng(seed)
    fe = rng.normal(0, 0.002, len(t)) * fe_scale
    return SignalMetrics.compute_all(
        t, {"DPOS(0)": dpos, "DRIVE_FE(0)": fe}, axis=0, settle_band=0.05)


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------
class TestExtractKpis:
    def test_pulls_scalars_from_real_metrics(self):
        kpis = extract_kpis(_capture_metrics())
        assert kpis["settle_ms"] is not None
        assert kpis["fe_cruise_rms"] > 0
        assert kpis["osc_hz"] is None          # no oscillation seeded
        assert kpis["ringing"] is not None

    def test_not_settled_becomes_none(self):
        metrics = {"settle": {"time_to_band_ms": None,
                              "settled_within_window": False,
                              "zero_crossings": 9}}
        kpis = extract_kpis(metrics)
        assert kpis["settle_ms"] is None
        assert kpis["ringing"] == 9

    def test_spread_needs_two_settled_moves(self):
        metrics = {"settle": {
            "settled_within_window": True, "time_to_band_ms": 30.0,
            "per_move": [{"time_to_band_ms": 20.0},
                         {"time_to_band_ms": 30.0}],
        }}
        assert extract_kpis(metrics)["settle_spread_ms"] == pytest.approx(10.0)
        metrics["settle"]["per_move"] = [{"time_to_band_ms": 20.0}]
        assert extract_kpis(metrics)["settle_spread_ms"] is None

    def test_oscillation_only_when_significant(self):
        metrics = {"oscillation": {"fe": {
            "dominant_hz": 120.0, "has_significant_oscillation": True}}}
        assert extract_kpis(metrics)["osc_hz"] == 120.0
        metrics["oscillation"]["fe"]["has_significant_oscillation"] = False
        assert extract_kpis(metrics)["osc_hz"] is None


# ---------------------------------------------------------------------------
# Comparison verdicts
# ---------------------------------------------------------------------------
class TestCompareKpi:
    def test_lower_is_better(self):
        kpi = _kpi("settle_ms")
        assert compare_kpi(180.0, 95.0, kpi) == "better"
        assert compare_kpi(95.0, 180.0, kpi) == "worse"
        assert compare_kpi(100.0, 101.0, kpi) == "same"   # within epsilon

    def test_unity_direction_compares_distance_to_one(self):
        kpi = _kpi("vel_reach")
        assert compare_kpi(0.90, 0.99, kpi) == "better"
        assert compare_kpi(0.99, 1.20, kpi) == "worse"
        assert compare_kpi(1.02, 0.98, kpi) == "same"     # equal distance

    def test_absent_direction_flags_appearance(self):
        kpi = _kpi("osc_hz")
        assert compare_kpi(None, 120.0, kpi) == "worse"    # oscillation appeared
        assert compare_kpi(120.0, None, kpi) == "better"   # oscillation gone
        assert compare_kpi(None, None, kpi) == "same"
        assert compare_kpi(120.0, 80.0, kpi) == "same"     # still oscillating

    def test_none_means_not_settled_and_is_worse(self):
        kpi = _kpi("settle_ms")
        assert compare_kpi(100.0, None, kpi) == "worse"
        assert compare_kpi(None, 100.0, kpi) == "better"
        assert compare_kpi(None, None, kpi) == "same"

    def test_no_baseline_gives_no_verdicts(self):
        run = make_run(_capture_metrics(), axis=0, pn_snapshot=None)
        verdicts = compare_runs(None, run)
        assert all(v is None for v in verdicts.values())

    def test_improved_capture_reads_better(self):
        noisy = make_run(_capture_metrics(fe_scale=10.0), 0, None)
        clean = make_run(_capture_metrics(fe_scale=1.0, seed=7), 0, None)
        verdicts = compare_runs(noisy, clean)
        assert verdicts["fe_cruise_rms"] == "better"


# ---------------------------------------------------------------------------
# Pn snapshot diff
# ---------------------------------------------------------------------------
class TestPnChanges:
    def test_changed_parameter_formatted(self):
        prev = {"drive_type": "DX4", "pn102": 500, "pn112": 30}
        cur = {"drive_type": "DX4", "pn102": 600, "pn112": 30}
        assert pn_changes(prev, cur) == ["Pn102 500→600"]

    def test_multiple_and_none_values(self):
        prev = {"pn102": 500, "pn114": None}
        cur = {"pn102": 550, "pn114": 20}
        changes = pn_changes(prev, cur)
        assert "Pn102 500→550" in changes
        assert "Pn114 —→20" in changes

    def test_missing_snapshots_give_no_changes(self):
        assert pn_changes(None, {"pn102": 500}) == []
        assert pn_changes({"pn102": 500}, None) == []
        assert pn_changes({"pn102": 500}, {"pn102": 500}) == []


# ---------------------------------------------------------------------------
# History container + CSV
# ---------------------------------------------------------------------------
class TestTuningHistory:
    def _run(self, axis=0, **kpi_overrides):
        metrics = _capture_metrics()
        run = make_run(metrics, axis=axis, pn_snapshot={"pn102": 500})
        run.kpis.update(kpi_overrides)
        return run

    def test_add_and_cap(self):
        history = TuningHistory(max_runs=3)
        for i in range(5):
            history.add(self._run(settle_ms=float(i)))
        assert len(history) == 3
        assert [r.kpis["settle_ms"] for r in history.runs] == [2.0, 3.0, 4.0]

    def test_previous_for_matches_same_axis_only(self):
        history = TuningHistory()
        run_a0 = self._run(axis=0)
        run_a1 = self._run(axis=1)
        run_b0 = self._run(axis=0)
        for run in (run_a0, run_a1, run_b0):
            history.add(run)
        assert history.previous_for(run_b0) is run_a0
        assert history.previous_for(run_a1) is None
        assert history.previous_for(run_a0) is None

    def test_clear(self):
        history = TuningHistory()
        history.add(self._run())
        history.clear()
        assert len(history) == 0

    def test_csv_contains_runs_and_changes(self):
        history = TuningHistory()
        first = make_run(_capture_metrics(), 0, {"pn102": 500},
                         timestamp="10:00:00")
        second = make_run(_capture_metrics(seed=7), 0, {"pn102": 600},
                          timestamp="10:05:00")
        history.add(first)
        history.add(second)
        csv_text = history.to_csv()
        lines = csv_text.strip().splitlines()
        assert len(lines) == 3  # header + 2 runs
        assert "timestamp" in lines[0] and "settle_ms" in lines[0]
        assert "10:00:00" in lines[1]
        assert "Pn102 500→600" in lines[2]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
class TestFormatting:
    def test_format_kpi_handles_none(self):
        assert format_kpi(None, _kpi("settle_ms")) == "∅"
        assert format_kpi(None, _kpi("osc_hz")) == "—"
        assert format_kpi(95.4, _kpi("settle_ms")) == "95"
