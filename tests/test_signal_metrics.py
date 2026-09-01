"""Synthetic-profile tests for the SignalMetrics analysis engine.

Every test builds a trapezoidal profiled move with a known injected defect
(VFF lag, resonance, ringdown of known damping, stiction spike, splice…)
and asserts the engine recovers the seeded truth — and, just as important,
that a well-tuned axis is NOT flagged.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ai.signal_metrics import SignalMetrics
from src.ai.signal_channels import resolve_channels, detect_axes
from src.ai.signal_phases import segment_phases

FS = 1000.0
DT = 1.0 / FS


# ---------------------------------------------------------------------------
# Synthetic profile generation
# ---------------------------------------------------------------------------
def trapezoid_dvel(v=100.0, t_acc=0.2, t_cruise=0.6, t_pre=0.3, t_post=0.8,
                   direction=1.0):
    """Demand velocity of one trapezoidal move (units/s at FS)."""
    n_pre = int(t_pre * FS)
    n_acc = int(t_acc * FS)
    n_cr = int(t_cruise * FS)
    n_post = int(t_post * FS)
    return np.concatenate([
        np.zeros(n_pre),
        np.linspace(0.0, v, n_acc, endpoint=False),
        np.full(n_cr, v),
        np.linspace(v, 0.0, n_acc, endpoint=False),
        np.zeros(n_post),
    ]) * direction


def profile(dvel):
    """(t, dpos) for a demand-velocity array."""
    t = np.arange(len(dvel)) / FS
    dpos = np.cumsum(dvel) / FS
    return t, dpos


def noise(n, sigma, seed=42):
    return np.random.default_rng(seed).normal(0.0, sigma, n)


def base_params(dpos, fe, **extra):
    params = {"DPOS(0)": dpos, "DRIVE_FE(0)": fe}
    params.update(extra)
    return params


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------
class TestChannelResolution:
    def test_fe_never_binds_to_latch_or_limit(self):
        params = {"FE_LATCH(0)": [], "FE_LIMIT(0)": [], "FE_RANGE(0)": []}
        assert resolve_channels(params, axis=0)["fe"] is None

    def test_exact_fe_wins_over_latch(self):
        params = {"FE_LATCH(0)": [], "FE(0)": []}
        assert resolve_channels(params, axis=0)["fe"] == "FE(0)"

    def test_drive_fe_preferred_over_controller_fe(self):
        params = {"FE(0)": [], "DRIVE_FE(0)": []}
        assert resolve_channels(params, axis=0)["fe"] == "DRIVE_FE(0)"

    def test_torque_limits_never_matched_as_current(self):
        params = {"AXIS_MAX_TORQUE(0)": [], "DRIVE_POS_TORQUE(0)": []}
        assert resolve_channels(params, axis=0)["current"] is None
        params["DRIVE_TORQUE(0)"] = []
        assert resolve_channels(params, axis=0)["current"] == "DRIVE_TORQUE(0)"

    def test_demand_speed_raw_vs_normalised(self):
        params = {"DEMAND_SPEED_NORMALISED(0)": [], "DEMAND_SPEED(0)": []}
        ch = resolve_channels(params, axis=0)
        assert ch["demand_vel_native"] == "DEMAND_SPEED_NORMALISED(0)"
        assert ch["demand_vel_raw"] == "DEMAND_SPEED(0)"

    def test_axis_filter_and_detection(self):
        params = {"DPOS(0)": [], "DPOS(3)": [], "MPOS(3)": []}
        assert detect_axes(params) == [0, 3]
        assert resolve_channels(params, axis=3)["dpos"] == "DPOS(3)"
        assert resolve_channels(params, axis=0)["mpos"] is None

    def test_drive_mode_labels_have_no_axis(self):
        params = {"Speed Feedback (rpm)": [], "Torque Command (%Tn)": []}
        ch = resolve_channels(params, axis=5)  # axis-less keys match any axis
        assert ch["measured_vel"] == "Speed Feedback (rpm)"
        assert ch["current"] == "Torque Command (%Tn)"


# ---------------------------------------------------------------------------
# Well-tuned axis must not be flagged
# ---------------------------------------------------------------------------
class TestQuietAxis:
    def _metrics(self):
        dvel = trapezoid_dvel()
        t, dpos = profile(dvel)
        fe = noise(len(t), 0.002)
        mvel = dvel + noise(len(t), 0.05, seed=7)
        return SignalMetrics.compute_all(
            t, base_params(dpos, fe, **{"MSPEED(0)": mvel}), axis=0)

    def test_settles_immediately_within_auto_band(self):
        m = self._metrics()
        assert m["data_sufficiency"] == "OK"
        settle = m["settle"]
        assert settle["settled_within_window"]
        assert settle["time_to_band_ms"] is not None
        assert settle["time_to_band_ms"] <= 30.0

    def test_noise_is_not_ringing(self):
        settle = self._metrics()["settle"]
        assert settle["ringing"] is False
        assert settle["zero_crossings"] <= 3

    def test_health_verdicts_green(self):
        m = self._metrics()
        assert m["health"]["position"] is True
        assert m["health"]["velocity"] is True
        assert m["health"]["position_issues"] == []

    def test_no_oscillation_reported(self):
        m = self._metrics()
        assert m["oscillation"]["fe"]["has_significant_oscillation"] is False


# ---------------------------------------------------------------------------
# Injected defects must be recovered
# ---------------------------------------------------------------------------
class TestVffSlope:
    def test_fe_proportional_to_velocity_detected(self):
        # Two moves at different speeds so the cruise fit has velocity spread
        k = 0.002
        dvel = np.concatenate([
            trapezoid_dvel(v=50.0, t_post=0.5),
            trapezoid_dvel(v=100.0, t_pre=0.0),
        ])
        t, dpos = profile(dvel)
        fe = k * dvel + noise(len(t), 1e-4)
        m = SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)
        fit = m["fe"]["cruise_fe_vs_velocity"]
        assert fit["proportional_to_velocity"] is True
        assert fit["slope"] == pytest.approx(k, rel=0.25)

    def test_one_cruise_speed_does_not_invent_vff_slope(self):
        dvel = trapezoid_dvel(v=50.0)
        t, dpos = profile(dvel)
        fe = 0.002 * dvel + noise(len(t), 1e-4)
        m = SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)
        fit = m["fe"]["cruise_fe_vs_velocity"]
        assert fit["proportional_to_velocity"] is False
        assert "slope" not in fit
        assert "two materially different speeds" in fit["note"]

    def test_symmetric_viscous_fe_is_not_asymmetry(self):
        # +kv one way, -kv the other: symmetric physics, must NOT flag
        k = 0.002
        dvel = np.concatenate([
            trapezoid_dvel(v=100.0, t_post=0.5),
            trapezoid_dvel(v=100.0, t_pre=0.0, direction=-1.0),
        ])
        t, dpos = profile(dvel)
        fe = k * dvel + noise(len(t), 1e-4)
        m = SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)
        assert m["asymmetry"]["significant"] is False

    def test_direction_dependent_magnitude_is_asymmetry(self):
        # Gravity-like: large FE one way, small the other
        dvel = np.concatenate([
            trapezoid_dvel(v=100.0, t_post=0.5),
            trapezoid_dvel(v=100.0, t_pre=0.0, direction=-1.0),
        ])
        t, dpos = profile(dvel)
        fe = np.where(dvel > 0, 0.3, 0.05) * (np.abs(dvel) > 1e-9)
        fe = fe + noise(len(t), 1e-4)
        m = SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)
        assert m["asymmetry"]["significant"] is True


class TestResonance:
    def test_120hz_peak_recovered(self):
        dvel = trapezoid_dvel(t_cruise=0.6)
        t, dpos = profile(dvel)
        fe = 0.05 * np.sin(2 * np.pi * 120.0 * t) + noise(len(t), 0.001)
        m = SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)
        osc = m["oscillation"]["fe"]
        assert osc["has_significant_oscillation"] is True
        assert osc["dominant_hz"] == pytest.approx(120.0, abs=3.0)

    def test_oscillation_becomes_position_issue(self):
        dvel = trapezoid_dvel(t_cruise=0.6)
        t, dpos = profile(dvel)
        fe = 0.05 * np.sin(2 * np.pi * 120.0 * t) + noise(len(t), 0.001)
        m = SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)
        assert m["health"]["position"] is False
        assert m["oscillation"]["fe"]["dominant_hz"] == pytest.approx(
            120.0, abs=3.0)


class TestDirectionIndependentPhases:
    def test_negative_move_accel_and_decel_are_not_swapped(self):
        dvel = trapezoid_dvel(direction=-1.0)
        t = np.arange(len(dvel)) / FS
        phases = segment_phases(t, dvel, DT, [(0, len(dvel))])

        accel_slice = slice(300, 500)
        decel_slice = slice(1100, 1300)
        assert phases["accel"][accel_slice].sum() > 150
        assert phases["accel"][decel_slice].sum() == 0
        assert phases["decel"][decel_slice].sum() > 150
        assert phases["decel"][accel_slice].sum() == 0

    @pytest.mark.parametrize("direction", [1.0, -1.0])
    def test_overshoot_uses_peak_after_demand_ramp(self, direction):
        dvel = trapezoid_dvel(direction=direction)
        t, dpos = profile(dvel)
        mvel = dvel.copy()
        mvel[520] = direction * 120.0  # peak is in cruise, not accel
        metrics = SignalMetrics.compute_all(
            t, base_params(dpos, noise(len(t), 0.001),
                           **{"MSPEED(0)": mvel}),
            axis=0,
        )
        overshoot = metrics["velocity"]["velocity_overshoot_per_move"]
        assert overshoot["max_pct"] == pytest.approx(20.0, abs=0.1)


class TestRingdown:
    ZETA = 0.15
    FN = 40.0

    def _metrics(self, band=0.01):
        dvel = trapezoid_dvel(t_post=0.5)
        t, dpos = profile(dvel)
        fe = noise(len(t), 5e-4)
        # Inject a decaying oscillation from the end of the demand profile
        move_end = int((0.3 + 0.2 + 0.6 + 0.2) * FS)
        tt = t[move_end:] - t[move_end]
        wn = 2 * np.pi * self.FN
        wd = wn * np.sqrt(1 - self.ZETA ** 2)
        fe[move_end:] += 0.5 * np.exp(-self.ZETA * wn * tt) * np.sin(wd * tt)
        return SignalMetrics.compute_all(
            t, base_params(dpos, fe), axis=0, settle_band=band)

    def test_ringing_flagged(self):
        settle = self._metrics()["settle"]
        assert settle["ringing"] is True
        assert settle["zero_crossings"] > 3

    def test_settle_time_matches_envelope_decay(self):
        settle = self._metrics()
        s = settle["settle"]
        # envelope 0.5·e^(−ζωn·t) crosses 0.01 at t = ln(50)/ζωn ≈ 104 ms
        assert s["settled_within_window"]
        assert 60.0 <= s["time_to_band_ms"] <= 150.0

    def test_damping_ratio_recovered(self):
        settle = self._metrics()["settle"]
        assert settle["damping_ratio"] == pytest.approx(self.ZETA, abs=0.08)

    def test_natural_frequency_recovered(self):
        settle = self._metrics()["settle"]
        assert settle["natural_freq_hz"] == pytest.approx(self.FN, abs=6.0)

    def test_user_band_reported(self):
        settle = self._metrics(band=0.01)["settle"]
        assert settle["band"] == pytest.approx(0.01)
        assert settle["band_source"] == "user"


# ---------------------------------------------------------------------------
# Multi-move and segment robustness
# ---------------------------------------------------------------------------
class TestMultiMove:
    def _close_moves(self):
        """Two moves 80 ms apart; big FE spike during move 2's accel."""
        m1 = trapezoid_dvel(v=100.0, t_acc=0.2, t_cruise=0.4, t_pre=0.3,
                            t_post=0.08)
        m2 = trapezoid_dvel(v=100.0, t_acc=0.2, t_cruise=0.4, t_pre=0.0,
                            t_post=0.5)
        dvel = np.concatenate([m1, m2])
        t, dpos = profile(dvel)
        fe = noise(len(t), 0.001)
        accel2_start = len(m1)
        fe[accel2_start + 10:accel2_start + 180] += 2.0
        return SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)

    def test_settle_window_clipped_at_next_move(self):
        m = self._close_moves()
        # The 2.0 spike lives in move 2's accel — it must NOT appear in the
        # settle stats of move 1 (whose 200 ms window is cut at 80 ms).
        assert m["settle"]["fe_peak_during_settle"] < 0.1
        assert m["fe"]["accel"]["peak_abs"] > 1.5

    def test_moves_counted_and_warned(self):
        m = self._close_moves()
        assert m["phases"]["n_moves"] == 2
        assert m["settle"]["n_windows"] == 2
        assert any("multi-move" in w for w in m["warnings"])


class TestSegmentBreaks:
    def test_splice_jump_is_not_motion(self):
        # Two idle segments captured at different positions
        dpos = np.concatenate([np.zeros(400), np.full(400, 100.0)])
        t = np.arange(len(dpos)) / FS
        fe = noise(len(t), 0.001)
        m = SignalMetrics.compute_all(
            t, base_params(dpos, fe), axis=0, segment_breaks=[400])
        assert m["data_sufficiency"] == "INSUFFICIENT"
        assert any("no motion" in w for w in m["warnings"])

    def test_splice_does_not_inflate_velocity_peak(self):
        dvel = trapezoid_dvel(v=100.0, t_post=0.3)
        t1, dpos1 = profile(dvel)
        dpos2 = dpos1 + 500.0  # second capture starts elsewhere
        dpos = np.concatenate([dpos1, dpos2])
        t = np.arange(len(dpos)) / FS
        fe = noise(len(t), 0.001)
        m = SignalMetrics.compute_all(
            t, base_params(dpos, fe), axis=0, segment_breaks=[len(dpos1)])
        assert m["capture"]["n_segments"] == 2
        assert m["phases"]["n_moves"] == 2
        assert m["phases"]["peak_demand_velocity"] == pytest.approx(100.0, rel=0.05)


# ---------------------------------------------------------------------------
# Reversal detection and robustness against field captures
# ---------------------------------------------------------------------------
class TestReversalDetection:
    def test_move_edges_are_not_reversals(self):
        # Out-and-back with a long dwell: two separate moves, no reversal.
        # (Treating 0→v / v→0 edges as reversals blankets ±80 ms of every
        # move boundary and eats short moves entirely.)
        dvel = np.concatenate([
            trapezoid_dvel(v=100.0, t_post=0.5),
            trapezoid_dvel(v=100.0, t_pre=0.0, direction=-1.0),
        ])
        t, dpos = profile(dvel)
        m = SignalMetrics.compute_all(
            t, base_params(dpos, noise(len(t), 0.002)), axis=0)
        assert m["phases"]["n_reversals"] == 0
        assert m["phases"]["reversal_pct"] == 0.0
        assert m["phases"]["n_moves"] == 2

    def test_rect_velocity_pulse_counts_both_moves(self):
        # Constant-velocity out-and-back (rectangular velocity) — the
        # capture shape from the field screenshot that read as "1 move".
        dvel = np.concatenate([
            np.zeros(500),
            np.full(300, 64.0),    # out
            np.zeros(1100),        # hold
            np.full(300, -64.0),   # back
            np.zeros(800),
        ])
        t, dpos = profile(dvel)
        m = SignalMetrics.compute_all(
            t, base_params(dpos, noise(len(t), 0.002)), axis=0)
        assert m["phases"]["n_moves"] == 2
        assert m["phases"]["n_reversals"] == 0
        # Constant-velocity ramps classify as cruise in both directions,
        # so the VFF fit and asymmetry check have what they need
        assert m["phases"]["cruise_pct"] > 5.0
        assert "asymmetry_ratio" in m["asymmetry"]

    def test_true_reversal_through_zero_detected(self):
        # Triangle profile: continuous motion through a direction change
        dvel = np.concatenate([
            np.zeros(300),
            np.linspace(0.0, 100.0, 200, endpoint=False),
            np.linspace(100.0, -100.0, 400, endpoint=False),
            np.linspace(-100.0, 0.0, 200, endpoint=False),
            np.zeros(500),
        ])
        t, dpos = profile(dvel)
        m = SignalMetrics.compute_all(
            t, base_params(dpos, noise(len(t), 0.002)), axis=0)
        assert m["phases"]["n_reversals"] == 1
        assert m["phases"]["reversal_pct"] > 0.0

    def test_dither_capture_is_not_analysable(self):
        # Rapid demand dither: every crossing is a reversal, no clean move
        t = np.arange(3000) / FS
        dpos = 0.2 * np.sin(2 * np.pi * 30.0 * t)
        m = SignalMetrics.compute_all(
            t, base_params(dpos, noise(len(t), 0.002)), axis=0)
        assert m["data_sufficiency"] == "INSUFFICIENT"
        assert any("no analysable move" in w for w in m["warnings"])


class TestAutoBandRobustness:
    def test_correlated_noise_does_not_collapse_band(self):
        # Real FE noise is often slow wander with tiny sample-to-sample
        # steps (quantization). A diff-only sigma collapses the band to
        # ~1e-5 and everything reads "not settled" + false integral advice.
        dvel = trapezoid_dvel()
        t, dpos = profile(dvel)
        white = np.random.default_rng(3).normal(0, 1.0, len(t))
        slow = np.convolve(white, np.ones(400) / 400, mode="same")
        slow *= 0.0025 / max(float(np.std(slow)), 1e-12)
        fe = slow + 0.0016  # wander + small offset, well inside real noise
        m = SignalMetrics.compute_all(t, base_params(dpos, fe), axis=0)
        settle = m["settle"]
        assert settle["band"] > 0.004  # floored by idle value spread
        assert settle["settled_within_window"] is True
        assert settle["ringing"] is False
        assert settle["steady_state_offset_nonzero"] is False
        assert settle["natural_freq_hz"] is None
        # The collapsed band previously produced these exact false alarms
        issues = " ".join(m["health"]["position_issues"])
        assert "not settled" not in issues
        assert "ringing" not in issues.lower()
        assert "integral" not in issues


# ---------------------------------------------------------------------------
# Axis binding
# ---------------------------------------------------------------------------
class TestAxisBinding:
    def test_analysis_never_mixes_axes(self):
        dvel = trapezoid_dvel()
        t, dpos = profile(dvel)
        fe = noise(len(t), 0.002)
        params = {
            "DPOS(0)": dpos, "DRIVE_FE(0)": fe,
            "MPOS(1)": np.zeros(len(t)),
        }
        m = SignalMetrics.compute_all(t, params, axis=0)
        assert m["channels_detected"]["dpos"] == "DPOS(0)"
        assert m["channels_detected"]["fe"] == "DRIVE_FE(0)"
        assert m["channels_detected"]["mpos"] is None

    def test_missing_axis_reported(self):
        dvel = trapezoid_dvel()
        t, dpos = profile(dvel)
        params = {"DPOS(0)": dpos, "DRIVE_FE(0)": noise(len(t), 0.002)}
        m = SignalMetrics.compute_all(t, params, axis=2)
        assert m["data_sufficiency"] == "INSUFFICIENT"
        assert any("captured axes: [0]" in w for w in m["warnings"])


# ---------------------------------------------------------------------------
# Demand-velocity scaling
# ---------------------------------------------------------------------------
class TestDemandVelocityScaling:
    def test_normalised_channel_used_without_rescaling(self):
        dvel = trapezoid_dvel()
        t = np.arange(len(dvel)) / FS
        params = {
            "DEMAND_SPEED_NORMALISED(0)": dvel,             # already units/s
            "MSPEED(0)": dvel + noise(len(t), 0.05),
            "DRIVE_FE(0)": noise(len(t), 0.002),
        }
        m = SignalMetrics.compute_all(t, params, axis=0, servo_period_sec=0.001)
        assert (m["channels_detected"]["demand_vel"]
                == "DEMAND_SPEED_NORMALISED(0)")
        # Double-scaling by 1/servo_period would make this ratio ~0.001
        assert m["velocity"]["cruise_velocity_reach_ratio"] == pytest.approx(
            1.0, abs=0.05)

    def test_raw_channel_scaled_by_servo_period(self):
        dvel = trapezoid_dvel()
        t = np.arange(len(dvel)) / FS
        params = {
            "DEMAND_SPEED(0)": dvel * 0.001,                # units/servocycle
            "MSPEED(0)": dvel + noise(len(t), 0.05),
            "DRIVE_FE(0)": noise(len(t), 0.002),
        }
        m = SignalMetrics.compute_all(t, params, axis=0, servo_period_sec=0.001)
        assert m["velocity"]["cruise_velocity_reach_ratio"] == pytest.approx(
            1.0, abs=0.05)

    def test_raw_channel_without_period_skips_tracking(self):
        dvel = trapezoid_dvel()
        t = np.arange(len(dvel)) / FS
        params = {
            "DEMAND_SPEED(0)": dvel * 0.001,
            "MSPEED(0)": dvel + noise(len(t), 0.05),
            "DRIVE_FE(0)": noise(len(t), 0.002),
        }
        m = SignalMetrics.compute_all(t, params, axis=0)  # no servo period
        assert m["velocity"] == {}
        assert any("servo period unknown" in w for w in m["warnings"])
        # Segmentation is threshold-relative, so it still works
        assert m["phases"]["n_moves"] == 1


# ---------------------------------------------------------------------------
# API and report
# ---------------------------------------------------------------------------
class TestApi:
    def test_positional_call_still_works(self):
        dvel = trapezoid_dvel()
        t, dpos = profile(dvel)
        m = SignalMetrics.compute_all(t, base_params(dpos, noise(len(t), 0.002)))
        assert m["data_sufficiency"] == "OK"

    def test_too_short_capture(self):
        m = SignalMetrics.compute_all(np.arange(10) / FS, {"DPOS(0)": np.zeros(10)})
        assert m["data_sufficiency"] == "INSUFFICIENT"

    def test_format_for_llm_contains_key_sections(self):
        dvel = trapezoid_dvel()
        t, dpos = profile(dvel)
        mvel = dvel + noise(len(t), 0.05, seed=3)
        m = SignalMetrics.compute_all(
            t, base_params(dpos, noise(len(t), 0.002), **{"MSPEED(0)": mvel}),
            axis=0)
        text = SignalMetrics.format_for_llm(m)
        assert "DATA SUFFICIENCY: OK" in text
        assert "## Phase segmentation" in text
        assert "## Settling" in text
        assert "band" in text
        assert "## Velocity tracking" in text
