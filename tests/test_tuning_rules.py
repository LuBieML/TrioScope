"""Tests for the offline rule-based tuning recommendations."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ai.signal_metrics import SignalMetrics
from src.ai.tuning_rules import (
    MAX_PARAMETER_CHANGES, evaluate, tuning_score,
)

FS = 1000.0


# ---------------------------------------------------------------------------
# Handcrafted metrics fragments (unit-level rule triggers)
# ---------------------------------------------------------------------------
def _ok_metrics(**overrides) -> dict:
    """A minimal healthy OK-result skeleton; overrides merge shallowly."""
    metrics = {
        "data_sufficiency": "OK",
        "capture": {}, "channels_detected": {}, "phases": {},
        "fe": {
            "cruise": {"mean": 0.0, "std": 0.001, "rms": 0.001,
                       "peak_abs": 0.004},
            "accel": {"mean": 0.0, "std": 0.001, "rms": 0.001,
                      "peak_abs": 0.005},
            "decel": {"mean": 0.0, "std": 0.001, "rms": 0.001,
                      "peak_abs": 0.005},
        },
        "velocity": {}, "current": {}, "oscillation": {}, "asymmetry": {},
        "settle": {"band": 0.01, "settled_within_window": True,
                   "time_to_band_ms": 5.0, "zero_crossings": 1,
                   "ringing": False, "steady_state_offset_nonzero": False,
                   "fe_steady_state": 0.0, "fe_peak_during_settle": 0.005},
        "health": {}, "warnings": [],
    }
    metrics.update(overrides)
    return metrics


_DX4_PROFILE = {
    "drive_type": "DX4", "pn100_tuning_mode": 5,
    "pn101": 40, "pn102": 500, "pn103": 125, "pn104": 40,
    "pn106": 100, "pn112": 30, "pn113": 0, "pn114": 20, "pn115": 0,
    "pn135": 4,
}


def _rule_ids(report):
    return [r.rule_id for r in report.recommendations]


def _obs_ids(report):
    return [o.rule_id for o in report.observations]


# ---------------------------------------------------------------------------
# Blocking rules (current loop first)
# ---------------------------------------------------------------------------
class TestBlockingRules:
    def test_saturation_blocks_gain_advice(self):
        metrics = _ok_metrics(
            velocity={"cruise_velocity_reach_ratio": 0.85,
                      "velocity_overshoot_per_move": {"max_pct": 0.0}},
            current={"accel": {"saturation_pct": 12.0}},
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert _rule_ids(report) == ["torque_saturation"]
        assert report.root_cause == "profile"
        # Explicitly no Pn102 advice while torque-limited
        assert all(r.parameter != "Pn102" for r in report.recommendations)

    def test_resonance_recommends_notch_not_gains(self):
        metrics = _ok_metrics(
            oscillation={
                "fe": {"has_significant_oscillation": True,
                       "dominant_hz": 340.0},
                "current_vs_velocity_phase": {"phase_deg": 88.0,
                                              "dominant_freq_hz": 340.0},
            },
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert _rule_ids(report) == ["mechanical_resonance"]
        assert "notch" in report.recommendations[0].action.lower()
        assert "340" in report.recommendations[0].action
        assert report.root_cause == "mechanical"

    def test_resonance_below_drive_limit_does_not_recommend_notch(self):
        metrics = _ok_metrics(
            oscillation={
                "fe": {"has_significant_oscillation": True,
                       "dominant_hz": 25.0},
                "current_vs_velocity_phase": {"phase_deg": 88.0,
                                              "dominant_freq_hz": 25.0},
            },
        )
        report = evaluate(metrics, _DX4_PROFILE)
        rec = report.recommendations[0]
        assert rec.rule_id == "mechanical_resonance"
        assert "notch filter cannot be set below 50 Hz" in rec.action
        assert "notch filter at ~25" not in rec.action

    def test_instability_recommends_gain_reduction(self):
        metrics = _ok_metrics(
            oscillation={
                "fe": {"has_significant_oscillation": True,
                       "dominant_hz": 45.0},
                "current_vs_velocity_phase": {"phase_deg": 5.0,
                                              "dominant_freq_hz": 45.0},
            },
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert _rule_ids(report) == ["loop_instability"]
        rec = report.recommendations[0]
        assert rec.parameter == "Pn104"
        assert rec.proposed == "Pn104 40 → 32"   # −20%

    def test_oscillation_without_phase_asks_for_current_capture(self):
        metrics = _ok_metrics(
            oscillation={"fe": {"has_significant_oscillation": True,
                                "dominant_hz": 120.0}},
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert _rule_ids(report) == ["oscillation_unresolved"]
        assert "DRIVE_TORQUE" in report.recommendations[0].action

    def test_ambiguous_phase_with_current_captured_gives_gain_probe(self):
        # Field case: current IS captured, phase 122° falls between the
        # instability and resonance windows — must NOT ask to capture what
        # is already on screen
        metrics = _ok_metrics(
            channels_detected={"current": "DRIVE_CURRENT(0)",
                               "measured_vel": "MSPEED(0)"},
            oscillation={
                "fe": {"has_significant_oscillation": True,
                       "dominant_hz": 25.0},
                "current_vs_velocity_phase": {
                    "phase_deg": 122.0, "dominant_freq_hz": 25.0,
                    "coherence": 0.96},
            },
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert _rule_ids(report) == ["oscillation_ambiguous_phase"]
        rec = report.recommendations[0]
        assert "Capture DRIVE_TORQUE" not in rec.action
        assert "122°" in rec.action
        assert "notch filter cannot target frequencies below 50 Hz" in rec.action
        assert "notch at 25.0 Hz" not in rec.action
        assert rec.proposed == "Pn104 40 → 34"   # −15%
        assert "coherence 0.96" in rec.diagnosis

    def test_ambiguous_phase_gated_to_rigidity_in_auto_mode(self):
        metrics = _ok_metrics(
            channels_detected={"current": "DRIVE_CURRENT(0)",
                               "measured_vel": "MSPEED(0)"},
            oscillation={
                "fe": {"has_significant_oscillation": True,
                       "dominant_hz": 25.0},
                "current_vs_velocity_phase": {
                    "phase_deg": 122.0, "dominant_freq_hz": 25.0},
            },
        )
        profile = dict(_DX4_PROFILE, pn100_tuning_mode=3, pn101=71)
        report = evaluate(metrics, profile)
        rec = report.recommendations[0]
        assert rec.parameter == "Pn101"
        assert rec.proposed == "Pn101 71 → 60"   # softer → −15%

    def test_no_coherent_phase_with_signals_asks_for_longer_capture(self):
        metrics = _ok_metrics(
            channels_detected={"current": "DRIVE_CURRENT(0)",
                               "measured_vel": "MSPEED(0)"},
            oscillation={
                "fe": {"has_significant_oscillation": True,
                       "dominant_hz": 120.0},
                "current_vs_velocity_phase": {
                    "note": "no coherent oscillation detected",
                    "dominant_freq_hz": None},
            },
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert _rule_ids(report) == ["oscillation_no_coherent_phase"]
        assert "Re-capture" in report.recommendations[0].action


# ---------------------------------------------------------------------------
# Velocity and FE rules
# ---------------------------------------------------------------------------
class TestVelocityRules:
    def test_under_responsive_proposes_pn102_increase(self):
        metrics = _ok_metrics(
            velocity={"cruise_velocity_reach_ratio": 0.88,
                      "velocity_overshoot_per_move": {"max_pct": 0.0}},
            current={"accel": {"saturation_pct": 0.5}},
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert "velocity_under_responsive" in _rule_ids(report)
        rec = next(r for r in report.recommendations
                   if r.rule_id == "velocity_under_responsive")
        assert rec.proposed == "Pn102 500 → 575"   # +15%

    def test_no_current_channel_adds_torque_caveat(self):
        metrics = _ok_metrics(
            velocity={"cruise_velocity_reach_ratio": 0.88,
                      "velocity_overshoot_per_move": {"max_pct": 0.0}},
        )
        report = evaluate(metrics, _DX4_PROFILE)
        rec = next(r for r in report.recommendations
                   if r.rule_id == "velocity_under_responsive")
        assert "torque-limited" in rec.diagnosis

    def test_velocity_overshoot_proposes_reduction(self):
        metrics = _ok_metrics(
            velocity={"cruise_velocity_reach_ratio": 1.0,
                      "velocity_overshoot_per_move": {"max_pct": 22.0}},
        )
        report = evaluate(metrics, _DX4_PROFILE)
        rec = next(r for r in report.recommendations
                   if r.rule_id == "velocity_overshoot")
        assert rec.proposed == "Pn102 500 → 425"   # −15%


class TestFeRules:
    def test_vff_rule_fires_and_clamps_at_100(self):
        metrics = _ok_metrics(fe={
            **_ok_metrics()["fe"],
            "cruise_fe_vs_velocity": {"slope": 0.002, "intercept": 0.0,
                                      "proportional_to_velocity": True},
        })
        profile = dict(_DX4_PROFILE, pn112=95)
        report = evaluate(metrics, profile)
        rec = next(r for r in report.recommendations
                   if r.rule_id == "vff_insufficient")
        assert rec.parameter == "Pn112"
        assert rec.proposed == "Pn112 95 → 100"   # +10 clamped to max
        assert "cruise_fe_vs_velocity" in rec.diagnosis

    def test_aff_rule_fires_on_ramp_spikes_and_defers_to_vff(self):
        fe = {
            "cruise": {"mean": 0.0, "std": 0.001, "rms": 0.001,
                       "peak_abs": 0.01},
            "accel": {"mean": 0.0, "std": 0.02, "rms": 0.02,
                      "peak_abs": 0.08},   # 8× cruise
            "decel": {"mean": 0.0, "std": 0.02, "rms": 0.02,
                      "peak_abs": 0.06},
            "cruise_fe_vs_velocity": {"slope": 0.002, "intercept": 0.0,
                                      "proportional_to_velocity": True},
        }
        report = evaluate(_ok_metrics(fe=fe), _DX4_PROFILE)
        ids = _rule_ids(report)
        assert ids.index("vff_insufficient") < ids.index("aff_insufficient")
        aff = next(r for r in report.recommendations
                   if r.rule_id == "aff_insufficient")
        assert "confirm VFF first" in aff.action
        assert aff.proposed == "Pn114 20 → 30"

    def test_ringing_and_offset_rules(self):
        metrics = _ok_metrics(settle={
            "band": 0.01, "settled_within_window": True,
            "time_to_band_ms": 150.0, "zero_crossings": 7, "ringing": True,
            "steady_state_offset_nonzero": True, "fe_steady_state": 0.03,
            "fe_peak_during_settle": 0.2, "damping_ratio": 0.12,
        })
        report = evaluate(metrics, _DX4_PROFILE)
        ids = _rule_ids(report)
        assert "underdamped_settle" in ids
        assert "integral_insufficient" in ids
        integral = next(r for r in report.recommendations
                        if r.rule_id == "integral_insufficient")
        assert integral.proposed == "Pn103 125 → 106"   # −15%

    def test_at_most_three_parameter_changes(self):
        # Fire five actionable rules at once
        metrics = _ok_metrics(
            velocity={"cruise_velocity_reach_ratio": 0.88,
                      "velocity_overshoot_per_move": {"max_pct": 22.0}},
            fe={
                "cruise": {"mean": 0.0, "std": 0.001, "rms": 0.001,
                           "peak_abs": 0.01},
                "accel": {"mean": 0.0, "std": 0.02, "rms": 0.02,
                          "peak_abs": 0.08},
                "decel": {"mean": 0.0, "std": 0.02, "rms": 0.02,
                          "peak_abs": 0.06},
                "cruise_fe_vs_velocity": {
                    "slope": 0.002, "intercept": 0.0,
                    "proportional_to_velocity": True},
            },
            settle={"band": 0.01, "settled_within_window": True,
                    "time_to_band_ms": 150.0, "zero_crossings": 7,
                    "ringing": True, "steady_state_offset_nonzero": True,
                    "fe_steady_state": 0.03, "fe_peak_during_settle": 0.2},
        )
        report = evaluate(metrics, _DX4_PROFILE)
        assert len(report.recommendations) == MAX_PARAMETER_CHANGES


# ---------------------------------------------------------------------------
# Mechanical observations
# ---------------------------------------------------------------------------
class TestMechanicalRules:
    def test_reversal_spikes_are_mechanical_not_gains(self):
        metrics = _ok_metrics(fe={
            **_ok_metrics()["fe"],
            "reversal": {"mean": 0.0, "std": 0.05, "rms": 0.05,
                         "peak_abs": 0.1},   # 25× cruise peak 0.004
        })
        report = evaluate(metrics, _DX4_PROFILE)
        assert "reversal_transients" in _obs_ids(report)
        obs = report.observations[0]
        assert "S-curve" in obs.action
        assert "NOT" in obs.action
        # Mechanical root cause deducts only one point from the score
        assert tuning_score(metrics) == pytest.approx(9.0)

    def test_asymmetry_is_observation(self):
        metrics = _ok_metrics(asymmetry={
            "asymmetry_ratio": 0.7, "significant": True,
            "cruise_fe_pos_dir_mean": 0.3, "cruise_fe_neg_dir_mean": -0.05,
        })
        report = evaluate(metrics, _DX4_PROFILE)
        assert "direction_asymmetry" in _obs_ids(report)


# ---------------------------------------------------------------------------
# Tuning-mode gating
# ---------------------------------------------------------------------------
class TestModeGating:
    def test_tuningless_mode_redirects_to_rigidity(self):
        metrics = _ok_metrics(
            velocity={"cruise_velocity_reach_ratio": 0.88,
                      "velocity_overshoot_per_move": {"max_pct": 0.0}},
            current={"accel": {"saturation_pct": 0.5}},
        )
        profile = dict(_DX4_PROFILE, pn100_tuning_mode=1)
        report = evaluate(metrics, profile)
        rec = report.recommendations[0]
        assert rec.parameter == "Pn101"
        assert "Tuning-less" in rec.action
        assert "Pn101" in rec.action
        assert rec.proposed == "Pn101 40 → 46"   # +15%

    def test_manual_mode_keeps_direct_gain_advice(self):
        metrics = _ok_metrics(
            velocity={"cruise_velocity_reach_ratio": 0.88,
                      "velocity_overshoot_per_move": {"max_pct": 0.0}},
            current={"accel": {"saturation_pct": 0.5}},
        )
        report = evaluate(metrics, _DX4_PROFILE)   # mode 5
        assert report.recommendations[0].parameter == "Pn102"

    def test_feedforward_rules_not_gated(self):
        metrics = _ok_metrics(fe={
            **_ok_metrics()["fe"],
            "cruise_fe_vs_velocity": {"slope": 0.002, "intercept": 0.0,
                                      "proportional_to_velocity": True},
        })
        profile = dict(_DX4_PROFILE, pn100_tuning_mode=1)
        report = evaluate(metrics, profile)
        assert _rule_ids(report) == ["vff_insufficient"]


# ---------------------------------------------------------------------------
# Score rubric
# ---------------------------------------------------------------------------
class TestTuningScore:
    def test_perfect_capture_scores_ten(self):
        assert tuning_score(_ok_metrics()) == pytest.approx(10.0)

    def test_rubric_arithmetic(self):
        metrics = _ok_metrics(
            oscillation={"fe": {"has_significant_oscillation": True,
                                "dominant_hz": 120.0}},     # -2
            current={"accel": {"saturation_pct": 8.0}},      # -2
            settle={**_ok_metrics()["settle"],
                    "ringing": True, "zero_crossings": 6},   # -1
        )
        assert tuning_score(metrics) == pytest.approx(5.0)

    def test_insufficient_data_has_no_score(self):
        assert tuning_score({"data_sufficiency": "INSUFFICIENT"}) is None
        report = evaluate({"data_sufficiency": "INSUFFICIENT",
                           "warnings": ["no motion"]})
        assert report.score is None
        assert report.root_cause == "insufficient-data"
        assert "no motion" in report.observations[0].diagnosis

    def test_well_tuned_omits_recommendations(self):
        # A mild single deduction keeps the score at 9 → well tuned
        metrics = _ok_metrics(asymmetry={"asymmetry_ratio": 0.5,
                                         "significant": True})
        report = evaluate(metrics, _DX4_PROFILE)
        assert report.score == pytest.approx(9.0)
        assert report.well_tuned is True
        assert report.recommendations == []
        assert "direction_asymmetry" in _obs_ids(report)   # still observed


# ---------------------------------------------------------------------------
# End-to-end on a synthetic capture
# ---------------------------------------------------------------------------
class TestEndToEnd:
    @staticmethod
    def _capture(fe_fn):
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
        fe = fe_fn(t, dvel)
        return SignalMetrics.compute_all(
            t, {"DPOS(0)": dpos, "DRIVE_FE(0)": fe}, axis=0)

    def test_quiet_axis_reports_well_tuned(self):
        rng = np.random.default_rng(42)
        metrics = self._capture(lambda t, v: rng.normal(0, 0.002, len(t)))
        report = evaluate(metrics, _DX4_PROFILE)
        assert report.well_tuned is True
        assert report.score is not None and report.score >= 8.0
        assert report.recommendations == []

    def test_viscous_fe_yields_vff_recommendation(self):
        rng = np.random.default_rng(42)
        # FE ∝ velocity needs speed spread: two moves at 50 and 100 u/s
        n_acc, n_cr = 200, 400
        dvel = np.concatenate([
            np.zeros(300),
            np.linspace(0, 50.0, n_acc, endpoint=False),
            np.full(n_cr, 50.0),
            np.linspace(50.0, 0, n_acc, endpoint=False),
            np.zeros(500),
            np.linspace(0, 100.0, n_acc, endpoint=False),
            np.full(n_cr, 100.0),
            np.linspace(100.0, 0, n_acc, endpoint=False),
            np.zeros(800),
        ])
        t = np.arange(len(dvel)) / FS
        dpos = np.cumsum(dvel) / FS
        fe = 0.002 * dvel + rng.normal(0, 1e-4, len(t))
        metrics = SignalMetrics.compute_all(
            t, {"DPOS(0)": dpos, "DRIVE_FE(0)": fe}, axis=0)
        report = evaluate(metrics, _DX4_PROFILE)
        assert "vff_insufficient" in _rule_ids(report)
        rec = next(r for r in report.recommendations
                   if r.rule_id == "vff_insufficient")
        assert rec.proposed == "Pn112 30 → 40"
