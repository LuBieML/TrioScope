"""
Signal metrics for ScopeEngine AI analysis.

The LLM cannot read raw numeric arrays — it pattern-matches named scalars
against the rules in its system prompt. This module turns a full-rate scope
capture into a compact, structured report of named metrics the LLM can
reason over directly.

Every number the LLM is allowed to trust comes from here.

The implementation is split across focused modules:
    signal_constants.py  tuning thresholds
    signal_phases.py     channel detection, PhaseStats, phase segmentation
    signal_analyzers.py  FE / velocity / current / asymmetry / settling
    signal_spectral.py   FFT peaks and current-vs-velocity cross phase
    signal_report.py     LLM-facing text formatting
"""

from __future__ import annotations

import numpy as np

from . import signal_analyzers, signal_report, signal_spectral
from .signal_phases import PhaseStats, _find_channel, segment_phases

__all__ = ["SignalMetrics", "PhaseStats", "_find_channel"]


class SignalMetrics:
    """Compute and format a structured metrics report from a scope capture."""

    # Static aliases keep the historical SignalMetrics._xyz entry points.
    _segment_phases = staticmethod(segment_phases)
    _analyze_fe = staticmethod(signal_analyzers.analyze_fe)
    _analyze_velocity = staticmethod(signal_analyzers.analyze_velocity)
    _analyze_current = staticmethod(signal_analyzers.analyze_current)
    _analyze_asymmetry = staticmethod(signal_analyzers.analyze_asymmetry)
    _analyze_settling = staticmethod(signal_analyzers.analyze_settling)
    _fft_peaks = staticmethod(signal_spectral.fft_peaks)
    _cross_phase = staticmethod(signal_spectral.cross_phase)
    format_for_llm = staticmethod(signal_report.format_for_llm)

    @classmethod
    def compute_all(cls, time_arr: np.ndarray, params: dict) -> dict:
        result: dict = {
            "capture": {},
            "channels_detected": {},
            "phases": {},
            "fe": {},
            "velocity": {},
            "current": {},
            "oscillation": {},
            "asymmetry": {},
            "settle": {},
            "data_sufficiency": "OK",
            "warnings": [],
        }

        n = len(time_arr)
        if n < 32:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append(f"capture too short ({n} samples, need >=32)")
            return result

        dt = float(np.median(np.diff(time_arr)))
        fs = 1.0 / dt if dt > 0 else 0.0
        duration = float(time_arr[-1] - time_arr[0])
        result["capture"] = {
            "duration_s": round(duration, 4),
            "n_samples": n,
            "sample_rate_hz": round(fs, 1),
            "dt_ms": round(dt * 1000, 3),
            "nyquist_hz": round(fs / 2, 1),
        }

        # --- Channel detection ---
        ch_dpos = _find_channel(params, "dpos", "demandposition", "targetposition")
        ch_mpos = _find_channel(params, "mpos", "measuredposition", "actualposition")
        ch_fe = _find_channel(params, "fe", "followingerror")
        ch_dvel = _find_channel(params, "demandspeed", "demandvel", "dspeed")
        ch_mvel = _find_channel(params, "mspeed", "measuredvel", "actualvel", "vactual")
        ch_cur = _find_channel(params, "current", "torque", "dacout")

        result["channels_detected"] = {
            "dpos": ch_dpos, "mpos": ch_mpos, "fe": ch_fe,
            "demand_vel": ch_dvel, "measured_vel": ch_mvel, "current": ch_cur,
        }

        dpos = params.get(ch_dpos) if ch_dpos else None
        mpos = params.get(ch_mpos) if ch_mpos else None
        fe = params.get(ch_fe) if ch_fe else None
        mvel = params.get(ch_mvel) if ch_mvel else None
        cur = params.get(ch_cur) if ch_cur else None

        # Derive FE from DPOS/MPOS if not captured directly
        if fe is None and dpos is not None and mpos is not None:
            fe = dpos - mpos
            result["warnings"].append("FE derived from DPOS-MPOS (not captured directly)")

        # Derive demand velocity
        if dpos is not None:
            dvel = np.gradient(dpos, time_arr)
        elif ch_dvel:
            dvel = params[ch_dvel]
        else:
            dvel = None
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append(
                "no DPOS or demand-velocity channel — cannot segment motion phases"
            )
            return result

        # Data sufficiency: is anything actually moving?
        v_peak = float(np.max(np.abs(dvel)))
        if v_peak < 1e-9:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append("no motion detected in demand velocity (idle capture)")
            return result

        # --- Phase segmentation ---
        phases = cls._segment_phases(time_arr, dvel, dt)
        result["phases"] = {
            "n_moves": int(phases["n_moves"]),
            "n_reversals": int(phases["n_reversals"]),
            "idle_pct": round(100 * phases["idle"].sum() / n, 1),
            "accel_pct": round(100 * phases["accel"].sum() / n, 1),
            "cruise_pct": round(100 * phases["cruise"].sum() / n, 1),
            "decel_pct": round(100 * phases["decel"].sum() / n, 1),
            "settle_pct": round(100 * phases["settle"].sum() / n, 1),
            "reversal_pct": round(100 * phases["reversal"].sum() / n, 1),
            "peak_demand_velocity": round(v_peak, 4),
        }

        # --- FE per-phase + cruise slope vs velocity ---
        if fe is not None:
            result["fe"] = cls._analyze_fe(fe, dvel, phases)
        else:
            result["warnings"].append("no FE channel (and no DPOS+MPOS to derive from)")

        # --- Velocity tracking error ---
        if mvel is not None:
            result["velocity"] = cls._analyze_velocity(mvel, dvel, phases)

        # --- Current / torque ---
        if cur is not None:
            result["current"] = cls._analyze_current(cur, phases, dt)

        # --- Oscillation (FFT, cruise-only) ---
        cruise_mask = phases["cruise"]
        if fe is not None:
            result["oscillation"]["fe"] = cls._fft_peaks(fe, cruise_mask, fs)
        if mvel is not None:
            vel_err = mvel - dvel
            result["oscillation"]["velocity_error"] = cls._fft_peaks(
                vel_err, cruise_mask, fs)
        if cur is not None:
            result["oscillation"]["current"] = cls._fft_peaks(
                cur - np.mean(cur), cruise_mask, fs)

        # --- Phase relationship between current and velocity ---
        if cur is not None and mvel is not None:
            result["oscillation"]["current_vs_velocity_phase"] = cls._cross_phase(
                cur - np.mean(cur), mvel - np.mean(mvel), cruise_mask, fs
            )

        # --- Asymmetry (+ vs - direction) ---
        if fe is not None:
            result["asymmetry"] = cls._analyze_asymmetry(fe, dvel, phases)

        # --- Settling ---
        if fe is not None:
            result["settle"] = cls._analyze_settling(fe, phases, dt)

        # --- New warnings ---
        # Insufficient cruise duration for oscillation
        for sig_name in ("fe", "velocity_error", "current"):
            fft_res = result["oscillation"].get(sig_name, {})
            if ("insufficient cruise duration" in fft_res.get("note", "")
                    and not any("oscillation analysis skipped" in w
                                for w in result["warnings"])):
                dur = fft_res["note"].split("(")[1].split(" s")[0]
                result["warnings"].append(
                    f"oscillation analysis skipped: insufficient cruise duration "
                    f"({dur}s)")
                break

        # Multi-move capture
        n_moves = phases["n_moves"]
        if n_moves > 1:
            result["warnings"].append(
                f"multi-move capture detected ({n_moves} moves) — per-move stats "
                f"aggregated; check reversal transient stats for stiction/backlash "
                f"signatures")

        # Large FE spikes at reversals
        fe_data = result.get("fe", {})
        if "reversal" in fe_data and "cruise" in fe_data:
            rev_peak = fe_data["reversal"]["peak_abs"]
            cruise_peak = fe_data["cruise"]["peak_abs"]
            if cruise_peak > 0 and rev_peak > 5 * cruise_peak:
                result["warnings"].append(
                    f"large FE spikes at reversals ({rev_peak} vs cruise "
                    f"{cruise_peak}) — likely stiction, backlash, or triangle-wave "
                    f"acceleration discontinuity")

        return result
