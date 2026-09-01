"""Tests for the Welch-averaged spectral analysis with coherence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ai.signal_metrics import SignalMetrics
from src.ai.signal_spectral import cross_phase, fft_peaks

FS = 1000.0


def _move_dvel(t_cruise=0.6, t_post=0.5, v=100.0):
    return np.concatenate([
        np.zeros(300),
        np.linspace(0.0, v, 200, endpoint=False),
        np.full(int(t_cruise * FS), v),
        np.linspace(v, 0.0, 200, endpoint=False),
        np.zeros(int(t_post * FS)),
    ])


def _capture(n_moves=1, fe_fn=None, t_cruise=0.6):
    dvel = np.concatenate([_move_dvel(t_cruise=t_cruise)] * n_moves)
    t = np.arange(len(dvel)) / FS
    dpos = np.cumsum(dvel) / FS
    rng = np.random.default_rng(42)
    fe = fe_fn(t) if fe_fn else rng.normal(0, 0.001, len(t))
    return t, dpos, fe


def _cruise_setup(n_moves=1, fe_fn=None, t_cruise=0.6):
    """(fe, cruise_mask, bounds) via the real segmentation engine."""
    t, dpos, fe = _capture(n_moves, fe_fn, t_cruise)
    metrics = SignalMetrics.compute_all(
        t, {"DPOS(0)": dpos, "DRIVE_FE(0)": fe}, axis=0)
    return metrics


class TestInterpolatedPeaks:
    def test_off_bin_frequency_recovered_within_1hz(self):
        # 123.7 Hz sits between bins for any realistic window length
        target = 123.7
        metrics = _cruise_setup(fe_fn=lambda t: (
            0.05 * np.sin(2 * np.pi * target * t)
            + np.random.default_rng(1).normal(0, 0.001, len(t))))
        osc = metrics["oscillation"]["fe"]
        assert osc["has_significant_oscillation"] is True
        assert osc["dominant_hz"] == pytest.approx(target, abs=1.0)

    def test_resolution_and_note_reported(self):
        metrics = _cruise_setup(fe_fn=lambda t: (
            0.05 * np.sin(2 * np.pi * 120.0 * t)))
        osc = metrics["oscillation"]["fe"]
        assert osc["resolution_hz"] > 0
        assert "interpolated" in osc["note"]


class TestWelchAveraging:
    def test_repeated_moves_average_together(self):
        one = _cruise_setup(n_moves=1)["oscillation"]["fe"]
        three = _cruise_setup(n_moves=3)["oscillation"]["fe"]
        assert one["n_averages"] >= 1
        assert three["n_averages"] > one["n_averages"]
        assert three["n_cruise_runs"] == 3

    def test_quiet_axis_still_clean_with_averaging(self):
        metrics = _cruise_setup(n_moves=3)
        assert metrics["oscillation"]["fe"]["has_significant_oscillation"] is False


class TestCoherence:
    @staticmethod
    def _phase_inputs(shared_hz=200.0, phase_rad=np.pi / 2, coherent=True,
                      n_moves=3):
        """current/velocity arrays sharing (or not) a resonance line."""
        dvel = np.concatenate([_move_dvel()] * n_moves)
        t = np.arange(len(dvel)) / FS
        rng = np.random.default_rng(7)
        osc_v = 0.5 * np.sin(2 * np.pi * shared_hz * t)
        if coherent:
            osc_c = 2.0 * np.sin(2 * np.pi * shared_hz * t + phase_rad)
        else:
            osc_c = rng.normal(0, 2.0, len(t))  # independent noise
        velocity = dvel + osc_v + rng.normal(0, 0.02, len(t))
        current = osc_c + rng.normal(0, 0.02, len(t))
        dpos = np.cumsum(dvel) / FS
        return t, dpos, velocity, current

    def _cross(self, **kwargs):
        t, dpos, velocity, current = self._phase_inputs(**kwargs)
        metrics = SignalMetrics.compute_all(
            t, {"DPOS(0)": dpos, "MSPEED(0)": velocity,
                "DRIVE_TORQUE(0)": current,
                "DRIVE_FE(0)": np.zeros(len(t))},
            axis=0)
        return metrics["oscillation"]["current_vs_velocity_phase"]

    def test_coherent_resonance_passes_gate_with_90_degrees(self):
        cvp = self._cross(coherent=True, phase_rad=np.pi / 2)
        assert cvp["dominant_freq_hz"] == pytest.approx(200.0, abs=3.0)
        assert cvp["coherence"] is not None and cvp["coherence"] >= 0.7
        assert 60 < cvp["phase_deg"] < 120
        assert "RESONANCE" in cvp["interpretation"]
        assert cvp["n_averages"] >= 2
        assert cvp["classification_reliable"] is True

    def test_low_frequency_resonance_is_outside_notch_range(self):
        cvp = self._cross(shared_hz=25.0, coherent=True,
                          phase_rad=np.pi / 2)
        assert cvp["dominant_freq_hz"] == pytest.approx(25.0, abs=3.0)
        assert "RESONANCE" in cvp["interpretation"]
        assert "below 50 Hz notch-filter range" in cvp["interpretation"]

    def test_in_phase_oscillation_reads_instability(self):
        cvp = self._cross(coherent=True, phase_rad=0.0)
        assert -30 < cvp["phase_deg"] < 30
        assert "INSTABILITY" in cvp["interpretation"]

    def test_incoherent_noise_rejected(self):
        cvp = self._cross(coherent=False)
        assert cvp["dominant_freq_hz"] is None
        assert "no coherent oscillation" in cvp["note"]

    def test_single_window_falls_back_to_proxy(self):
        t, dpos, velocity, current = self._phase_inputs(n_moves=1)
        metrics = SignalMetrics.compute_all(
            t, {"DPOS(0)": dpos, "MSPEED(0)": velocity,
                "DRIVE_TORQUE(0)": current,
                "DRIVE_FE(0)": np.zeros(len(t))},
            axis=0)
        cvp = metrics["oscillation"]["current_vs_velocity_phase"]
        if cvp.get("n_averages", 0) >= 2:
            pytest.skip("segmentation produced multiple windows")
        assert "proxy" in cvp["method"]
        assert cvp["coherence"] is None
        assert cvp["classification_reliable"] is False

    def test_fe_mode_does_not_borrow_phase_from_different_frequency(self):
        t, dpos, velocity, current = self._phase_inputs(
            shared_hz=200.0, coherent=True, n_moves=3)
        fe = 0.05 * np.sin(2 * np.pi * 30.0 * t)
        metrics = SignalMetrics.compute_all(
            t, {"DPOS(0)": dpos, "MSPEED(0)": velocity,
                "DRIVE_TORQUE(0)": current, "DRIVE_FE(0)": fe},
            axis=0)
        cvp = metrics["oscillation"]["current_vs_velocity_phase"]
        assert cvp["target_freq_hz"] == pytest.approx(30.0, abs=2.0)
        assert cvp["dominant_freq_hz"] is None
        assert cvp["classification_reliable"] is False
        assert "at the FE oscillation frequency" in cvp["note"]


class TestGates:
    def test_short_cruise_still_insufficient(self):
        metrics = _cruise_setup(t_cruise=0.1)
        osc = metrics["oscillation"]["fe"]
        assert osc["has_significant_oscillation"] is False
        assert osc.get("insufficient_duration") is True

    def test_direct_call_with_no_cruise(self):
        mask = np.zeros(1000, dtype=bool)
        result = fft_peaks(np.zeros(1000), mask, FS, [(0, 1000)])
        assert result["has_significant_oscillation"] is False
        assert cross_phase(np.zeros(1000), np.zeros(1000), mask, FS,
                           [(0, 1000)]) is None
