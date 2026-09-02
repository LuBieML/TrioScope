"""
Signal metrics for scope-capture servo analysis.

Single analysis engine shared by the Servo Loop Analyser panel and the AI
chat panel: it turns a full-rate scope capture into a structured report of
named metrics (phase-segmented FE, velocity tracking, current saturation,
oscillation, per-move settling). Every number the panel cards display and
the LLM is allowed to trust comes from here.

Robustness properties:
  - axis-aware, exact-first channel matching (multi-axis captures never mix
    axes; FE never binds to FE_LATCH etc.)
  - segment-break aware (continuous-mode splices never fake motion)
  - per-move settle windows clipped at the next move and segment ends
  - settling / ringing judged against an absolute tolerance band, not a
    percentage of the noise peak
"""

from __future__ import annotations

import numpy as np

from .signal_channels import detect_axes, resolve_channels
from .signal_phases import (
    segment_bounds, per_segment_gradient, median_dt, segment_phases,
)
from .signal_analyzers import (
    analyze_asymmetry, analyze_current, analyze_fe, analyze_settling,
    analyze_velocity, build_health, estimate_settle_band,
)
from .signal_spectral import cross_phase, fft_peaks
from .signal_report import format_for_llm
from .signal_constants import (
    FE_OSCILLATION_SIGNAL_FRAC, VELOCITY_OSCILLATION_DEMAND_FRAC,
)

__all__ = ["SignalMetrics"]


def _apply_engineering_oscillation_floor(
        spectral: dict, threshold: float, basis: str) -> None:
    """Separate a repeatable FFT line from a control-significant oscillation."""
    spectral_peak = bool(spectral.get("has_significant_oscillation"))
    spectral["has_spectral_peak"] = spectral_peak
    spectral["significance_threshold"] = round(max(0.0, threshold), 6)
    spectral["significance_basis"] = basis
    peaks = spectral.get("peaks") or []
    amplitude = float(peaks[0].get("amplitude") or 0.0) if peaks else 0.0
    significant = spectral_peak and amplitude >= threshold
    spectral["has_significant_oscillation"] = significant
    if spectral_peak and not significant:
        spectral["note"] = (
            f"{spectral.get('note', '')}; repeatable spectral line retained "
            f"for diagnostics, but amplitude {amplitude:g} is below the "
            f"control-significance floor {threshold:g} ({basis})"
        ).lstrip("; ")


class SignalMetrics:
    """Compute and format a structured metrics report from a scope capture."""

    @classmethod
    def compute_all(cls, time_arr: np.ndarray, params: dict, *,
                    axis: int | None = None,
                    servo_period_sec: float | None = None,
                    segment_breaks=None,
                    settle_band: float | None = None) -> dict:
        """Analyze one capture.

        axis: restrict channel matching to this axis (None = any).
        servo_period_sec: scales raw DEMAND_SPEED (units/servocycle) to
            units/s so it is comparable with MSPEED.
        segment_breaks: buffer indices where continuous-mode segments were
            spliced; analysis never crosses them.
        settle_band: settling tolerance in user units (None/0 = derive
            from the capture's noise floor).
        """
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
            "health": {},
            "data_sufficiency": "OK",
            "warnings": [],
        }

        n = len(time_arr)
        if n < 32:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append(f"capture too short ({n} samples, need >=32)")
            return result

        t = np.asarray(time_arr, dtype=np.float64)
        bounds = segment_bounds(n, segment_breaks)
        dt = median_dt(t, bounds)
        fs = 1.0 / dt if dt > 0 else 0.0
        duration = float(t[-1] - t[0])
        result["capture"] = {
            "duration_s": round(duration, 4),
            "n_samples": n,
            "sample_rate_hz": round(fs, 1),
            "dt_ms": round(dt * 1000, 3),
            "nyquist_hz": round(fs / 2, 1),
            "n_segments": len(bounds),
            "axis_analyzed": axis if axis is not None else "any",
        }
        if len(bounds) > 1:
            result["warnings"].append(
                f"capture contains {len(bounds)} spliced segments — each is "
                f"analyzed independently (no cross-splice gradients or FFTs)")

        # --- Channel detection ---
        ch = resolve_channels(params, axis)
        ch_demand_vel = ch["demand_vel_native"] or ch["demand_vel_raw"]
        result["channels_detected"] = {
            "dpos": ch["dpos"], "mpos": ch["mpos"], "fe": ch["fe"],
            "demand_vel": ch_demand_vel,
            "measured_vel": ch["measured_vel"], "current": ch["current"],
        }
        if axis is not None and not any(ch.values()):
            captured_axes = detect_axes(params)
            if captured_axes and axis not in captured_axes:
                result["data_sufficiency"] = "INSUFFICIENT"
                result["warnings"].append(
                    f"no channels captured for axis {axis} "
                    f"(captured axes: {captured_axes})")
                return result

        def _arr(key: str | None) -> np.ndarray | None:
            if key is None:
                return None
            return np.asarray(params[key], dtype=np.float64)

        dpos = _arr(ch["dpos"])
        mpos = _arr(ch["mpos"])
        fe = _arr(ch["fe"])
        mvel = _arr(ch["measured_vel"])
        cur = _arr(ch["current"])

        # Derive FE from DPOS/MPOS if not captured directly
        if fe is None and dpos is not None and mpos is not None:
            fe = dpos - mpos
            result["warnings"].append("FE derived from DPOS-MPOS (not captured directly)")

        # --- Demand velocity (drives segmentation; scaled for tracking) ---
        velocity_units_known = True
        if dpos is not None:
            dvel = per_segment_gradient(dpos, t, bounds)
            demand_vel_source = "derived from DPOS (units/s)"
        elif ch["demand_vel_native"] is not None:
            dvel = _arr(ch["demand_vel_native"])
            demand_vel_source = f"{ch['demand_vel_native']} (native velocity units)"
        elif ch["demand_vel_raw"] is not None:
            raw = _arr(ch["demand_vel_raw"])
            assert raw is not None
            if servo_period_sec and servo_period_sec > 0:
                dvel = raw / float(servo_period_sec)
                demand_vel_source = (
                    f"{ch['demand_vel_raw']} scaled by 1/servo_period (units/s)")
            else:
                dvel = raw
                velocity_units_known = False
                demand_vel_source = (
                    f"{ch['demand_vel_raw']} in units/servocycle "
                    f"(servo period unknown)")
                result["warnings"].append(
                    "servo period unknown — DEMAND_SPEED left in "
                    "units/servocycle; velocity-tracking metrics skipped")
        else:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append(
                "no DPOS or demand-velocity channel — cannot segment motion phases")
            return result
        result["channels_detected"]["demand_vel_source"] = demand_vel_source
        assert dvel is not None  # every surviving branch assigned an array

        # Data sufficiency: is anything actually moving?
        v_peak = float(np.max(np.abs(dvel)))
        if v_peak < 1e-9:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append("no motion detected in demand velocity (idle capture)")
            return result

        # --- Phase segmentation ---
        phases = segment_phases(t, dvel, dt, bounds)
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
        if phases["n_moves"] == 0:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append(
                "no analysable move — demand velocity never rose above the "
                "motion threshold outside reversal windows; capture DPOS "
                "over at least one complete move (with post-move dwell)")
            return result

        # --- FE per-phase + cruise slope vs velocity ---
        if fe is not None:
            result["fe"] = analyze_fe(fe, dvel, phases, velocity_units_known)
        else:
            result["warnings"].append("no FE channel (and no DPOS+MPOS to derive from)")

        # --- Settling against the tolerance band ---
        if fe is not None:
            band, band_source = estimate_settle_band(fe, phases, settle_band)
            result["capture"]["settle_band"] = round(band, 6)
            result["capture"]["settle_band_source"] = band_source
            result["settle"] = analyze_settling(fe, t, phases, band, band_source)

        # --- Velocity tracking error (units must be consistent) ---
        if mvel is not None and velocity_units_known:
            result["velocity"] = analyze_velocity(mvel, dvel, phases)

        # --- Current / torque ---
        if cur is not None:
            result["current"] = analyze_current(cur, phases, dt)

        # --- Oscillation (FFT, longest contiguous cruise run) ---
        cruise_mask = phases["cruise"]
        if fe is not None:
            result["oscillation"]["fe"] = fft_peaks(fe, cruise_mask, fs, bounds)
            fe_typical = float(np.percentile(np.abs(fe), 95))
            fe_band = float(result["capture"].get("settle_band") or 0.0)
            user_band = (
                fe_band
                if result["capture"].get("settle_band_source") == "user"
                else 0.0
            )
            _apply_engineering_oscillation_floor(
                result["oscillation"]["fe"],
                max(user_band, FE_OSCILLATION_SIGNAL_FRAC * fe_typical),
                "max(user settle band, 10% of typical move FE)",
            )
        if mvel is not None and velocity_units_known:
            result["oscillation"]["velocity_error"] = fft_peaks(
                mvel - dvel, cruise_mask, fs, bounds)
            _apply_engineering_oscillation_floor(
                result["oscillation"]["velocity_error"],
                VELOCITY_OSCILLATION_DEMAND_FRAC * v_peak,
                "0.5% of peak demand velocity",
            )
        if cur is not None:
            result["oscillation"]["current"] = fft_peaks(
                cur - np.mean(cur), cruise_mask, fs, bounds)

        # --- Phase relationship between current and velocity ---
        if cur is not None and mvel is not None:
            fe_osc = result["oscillation"].get("fe", {})
            phase_target = (fe_osc.get("dominant_hz")
                            if fe_osc.get("has_significant_oscillation")
                            else None)
            result["oscillation"]["current_vs_velocity_phase"] = cross_phase(
                cur - np.mean(cur), mvel - np.mean(mvel), cruise_mask, fs, bounds,
                target_freq_hz=phase_target)

        # --- Asymmetry (+ vs - direction) ---
        if fe is not None:
            result["asymmetry"] = analyze_asymmetry(fe, dvel, phases)

        # --- Aggregate warnings ---
        for sig_name in ("fe", "velocity_error", "current"):
            fft_res = result["oscillation"].get(sig_name, {})
            if fft_res.get("insufficient_duration"):
                result["warnings"].append(
                    f"oscillation analysis skipped: insufficient contiguous "
                    f"cruise duration ({fft_res.get('cruise_duration_s', 0)}s)")
                break

        n_moves = phases["n_moves"]
        if n_moves > 1:
            result["warnings"].append(
                f"multi-move capture detected ({n_moves} moves) — settling is "
                f"per-move; check reversal transient stats for stiction/backlash "
                f"signatures")

        fe_data = result.get("fe", {})
        if "reversal" in fe_data and "cruise" in fe_data:
            rev_peak = fe_data["reversal"]["peak_abs"]
            cruise_peak = fe_data["cruise"]["peak_abs"]
            if cruise_peak > 0 and rev_peak > 5 * cruise_peak:
                result["warnings"].append(
                    f"large FE spikes at reversals ({rev_peak} vs cruise "
                    f"{cruise_peak}) — likely stiction, backlash, or triangle-wave "
                    f"acceleration discontinuity")

        # --- Loop health verdicts (panel cards + report) ---
        result["health"] = build_health(result)

        return result

    # ---------------------------------------------------------------- format
    @staticmethod
    def format_for_llm(metrics: dict) -> str:
        """Format the metrics dict as a compact named-scalar block for the LLM."""
        return format_for_llm(metrics)
