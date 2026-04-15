"""
Classical servo-loop analyser.

Two-level analysis:
  1. Velocity loop — compares MSPEED against demand velocity derived from the
     command profile.  Detects overshoot, ringing, and tracking errors during
     accel/cruise phases.
  2. Position loop — analyses DRIVE_FE (drive following error) during and after
     the profiled move.  DPOS provides the motion-profile structure (boundaries,
     step size, phase segmentation).  Measures overshoot, oscillation, settling
     time, steady-state error, damping, and following error during motion.

     MPOS is NOT required — DRIVE_FE is the single source of error information,
     giving the most accurate view of what the drive itself sees.

All analysis is designed for real profiled moves (trapezoidal / S-curve ramps),
NOT ideal step inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class VelocityLoopMetrics:
    """Velocity-loop observations from MSPEED vs demand velocity."""
    accel_overshoot_pct: float = 0.0
    cruise_tracking_ratio: float = 1.0
    cruise_velocity_std: float = 0.0
    accel_settle_time_ms: float = 0.0
    accel_oscillation_count: int = 0
    is_healthy: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass
class StepResponseMetrics:
    """Position-loop observations from following-error analysis."""
    overshoot_pct: float = 0.0
    settling_time_ms: float = 0.0
    rise_time_ms: float = 0.0
    steady_state_error: float = 0.0
    oscillation_count: int = 0
    damping_ratio: float = 0.0
    natural_freq_est_hz: float = 0.0
    drive_fe_peak: float = 0.0            # peak |FE| during motion (user units)
    drive_fe_cruise_mean: float = 0.0     # mean |FE| during cruise (user units)
    drive_fe_peak_pct: float = 0.0        # peak |FE| during motion / step_size
    drive_fe_cruise_mean_pct: float = 0.0  # mean |FE| during cruise / step_size


# ---------------------------------------------------------------------------
# Classical analyser
# ---------------------------------------------------------------------------
class ClassicalTuner:
    """Analyse profiled-move captures of the servo loops."""

    # --------------------------------------------------------------- velocity
    @staticmethod
    def analyze_velocity_loop(
        time_arr: np.ndarray,
        measured_velocity: np.ndarray,
        demand_velocity: np.ndarray,
    ) -> VelocityLoopMetrics:
        """Observe velocity-loop behaviour from MSPEED vs captured DEMAND_SPEED.

        Both inputs must be in the same units (user units per second). The
        caller is responsible for scaling DEMAND_SPEED (captured as
        units/servocycle) by 1/servo_period_sec before passing it in.
        """
        dvel = demand_velocity
        dacc = np.gradient(dvel, time_arr)
        verr = measured_velocity - dvel

        v_peak = float(np.max(np.abs(dvel)))
        a_peak = float(np.max(np.abs(dacc)))
        if v_peak < 1e-9:
            return VelocityLoopMetrics()

        # Segment
        moving = np.abs(dvel) > 0.02 * v_peak
        accel = moving & (np.abs(dacc) > 0.10 * a_peak)
        cruise = moving & ~accel

        metrics = VelocityLoopMetrics()

        # Accel overshoot — evaluated in the transition window where
        # acceleration drops to near zero (velocity peaks), normalized
        # against the commanded velocity at that instant, and filtered
        # to require a local maximum (rejects single-sample noise spikes).
        if a_peak > 1e-9:
            transition_mask = moving & (np.abs(dacc) <= 0.05 * a_peak)
            if transition_mask.sum() > 2:
                v_target = float(np.mean(np.abs(dvel[transition_mask])))
                if v_target > 1e-9:
                    signed_err = verr[transition_mask] * np.sign(dvel[transition_mask])
                    peak_idx = int(np.argmax(signed_err))
                    if 0 < peak_idx < len(signed_err) - 1:
                        local_max = float(signed_err[peak_idx])
                        if (
                            local_max > float(signed_err[peak_idx - 1])
                            and local_max > float(signed_err[peak_idx + 1])
                        ):
                            metrics.accel_overshoot_pct = max(
                                0.0, local_max / v_target * 100.0
                            )

        # Cruise tracking ratio
        if cruise.sum() > 10:
            denom = float(np.mean(np.abs(dvel[cruise])))
            if denom > 1e-9:
                metrics.cruise_tracking_ratio = float(
                    np.mean(np.abs(measured_velocity[cruise]))
                ) / denom

        # Cruise velocity std
        if cruise.sum() > 10:
            metrics.cruise_velocity_std = float(np.std(verr[cruise]))

        # Accel-to-cruise transition settle time
        transition_indices = np.where(accel[:-1] & cruise[1:])[0]
        last_transition_idx: int | None = None
        if len(transition_indices) > 0:
            last_transition_idx = int(transition_indices[-1]) + 1
            cruise_speed = float(np.mean(np.abs(dvel[cruise])))
            band = 0.05 * cruise_speed if cruise_speed > 1e-9 else 1e-9
            t_trans = time_arr[last_transition_idx]
            for i in range(last_transition_idx, len(time_arr)):
                if abs(verr[i]) <= band:
                    settled = True
                    for j in range(i, min(i + 10, len(time_arr))):
                        if abs(verr[j]) > band:
                            settled = False
                            break
                    if settled:
                        metrics.accel_settle_time_ms = float(
                            time_arr[i] - t_trans
                        ) * 1000.0
                        break

        # Accel oscillation count — 200ms window after last transition
        if last_transition_idx is not None:
            dt = float(np.median(np.diff(time_arr)))
            window_samples = max(1, int(0.200 / dt))
            win_end = min(len(verr), last_transition_idx + window_samples)
            win_verr = verr[last_transition_idx:win_end]
            if len(win_verr) > 2:
                signs = np.sign(win_verr)
                sign_changes = np.diff(signs)
                zero_crossings = int(np.sum(sign_changes != 0))
                metrics.accel_oscillation_count = zero_crossings // 2

        # Observations
        issues: list[str] = []
        if metrics.accel_overshoot_pct > 15:
            issues.append(
                f"Velocity overshoot {metrics.accel_overshoot_pct:.1f}% "
                f"during accel (>15%)"
            )
        if metrics.cruise_tracking_ratio < 0.90:
            issues.append(
                f"Velocity not reaching demand "
                f"(ratio {metrics.cruise_tracking_ratio:.3f})"
            )
        if metrics.cruise_tracking_ratio > 1.10:
            issues.append(
                f"Velocity exceeding demand during cruise "
                f"(ratio {metrics.cruise_tracking_ratio:.3f})"
            )
        if metrics.accel_oscillation_count > 3:
            issues.append(
                f"Velocity ringing at accel/cruise transition "
                f"({metrics.accel_oscillation_count} oscillations)"
            )
        if metrics.accel_settle_time_ms > 100:
            issues.append(
                f"Slow velocity settling ({metrics.accel_settle_time_ms:.0f}ms)"
            )

        if issues:
            metrics.is_healthy = False
            metrics.issues = issues

        return metrics

    # ---------------------------------------------------------- step response
    @staticmethod
    def analyze_step_response(
        time_arr: np.ndarray,
        command: np.ndarray,
        drive_fe: np.ndarray,
        velocity: np.ndarray | None = None,
        demand_velocity: np.ndarray | None = None,
    ) -> tuple[StepResponseMetrics, VelocityLoopMetrics | None]:
        """Analyse a profiled-move capture for position-loop behaviour.

        Uses DRIVE_FE (drive following error) as the sole error source and
        DPOS (command) for profile structure: boundaries, step size, and
        phase segmentation.  MPOS is not required.

        Returns (position_metrics, velocity_metrics_or_None).
        """
        if len(time_arr) < 20:
            return (StepResponseMetrics(), None)

        dt = float(np.median(np.diff(time_arr)))
        fe = np.asarray(drive_fe, dtype=np.float64)
        dvel = np.gradient(command, time_arr)
        v_peak = float(np.max(np.abs(dvel)))

        if v_peak < 1e-9:
            return (StepResponseMetrics(), None)

        # Find move-complete — last transition from moving to stopped
        moving = np.abs(dvel) > 0.02 * v_peak
        transitions = np.where(moving[:-1] & ~moving[1:])[0]
        if len(transitions) == 0:
            return (StepResponseMetrics(), None)

        # Settle window — anchor move-end to when DPOS itself stops changing,
        # not to demand-velocity threshold.  Settling time must be counted
        # from the moment the commanded profile has truly reached its final
        # value, otherwise we count samples where DPOS is still ramping.
        move_end_idx = transitions[-1] + 1
        cmd_final_est = float(command[-1])
        cmd_span = float(np.max(command) - np.min(command))
        dpos_stable_thresh = max(1e-4 * cmd_span, 1e-9)
        dpos_changing = np.abs(command - cmd_final_est) > dpos_stable_thresh
        dpos_change_idx = np.where(dpos_changing)[0]
        if len(dpos_change_idx) > 0:
            dpos_end_idx = int(dpos_change_idx[-1]) + 1
            # Use whichever is later — DPOS-based or velocity-based.
            move_end_idx = max(move_end_idx, dpos_end_idx)
            move_end_idx = min(move_end_idx, len(time_arr) - 1)
        t_end = time_arr[move_end_idx]
        settle_mask = time_arr >= t_end
        if settle_mask.sum() < 10:
            return (StepResponseMetrics(), None)

        # Step size for normalisation
        move_start_indices = np.where(~moving[:-1] & moving[1:])[0]
        if len(move_start_indices) > 0:
            move_start_idx = int(move_start_indices[-1])
        else:
            move_start_idx = 0
        cmd_start = float(command[move_start_idx])
        cmd_final = float(command[move_end_idx])
        step_size = abs(cmd_final - cmd_start)
        if step_size < 1e-9:
            step_size = 1.0

        # Overshoot — peak FE after move-end
        t_settle = time_arr[settle_mask] - t_end
        fe_settle = fe[settle_mask]
        peak_fe = float(np.max(np.abs(fe_settle)))
        overshoot_pct = (peak_fe / step_size) * 100.0

        # Oscillation count — zero-crossings of FE in settle window
        signs = np.sign(fe_settle)
        sign_changes = np.diff(signs)
        zero_crossings = int(np.sum(sign_changes != 0))
        oscillation_count = zero_crossings // 2

        # Steady-state error — mean FE in last 10% of settle window
        ss_tail_len = max(1, len(fe_settle) // 10)
        fe_steady = float(np.mean(fe_settle[-ss_tail_len:]))
        steady_state_error = abs(fe_steady) / step_size

        # Settling time — when the peak-to-peak ringing envelope decays to a steady state.
        settling_time_ms = 0.0
        if len(fe_settle) >= 2:
            # You suggested checking if it reaches a steady value for 10 ms.
            win_samples = max(10, int(round(0.010 / dt))) if dt > 0 else 10
            
            if len(fe_settle) <= win_samples:
                # Not enough data for a full window, fallback to end of capture
                settling_time_ms = float(t_settle[-1]) * 1000.0
            else:
                from numpy.lib.stride_tricks import sliding_window_view
                
                # Calculate rolling peak-to-peak variation (high-pass filter to ignore slow drift)
                # By looking at the difference between the max and min over a 10ms sliding window,
                # we precisely isolate the high-frequency ringing and completely ignore slow baseline drift.
                windows = sliding_window_view(fe_settle, win_samples)
                rolling_envelope = np.max(windows, axis=1) - np.min(windows, axis=1)
                
                # We declare it "steady" when the ringing stops. 
                # Ringing has stopped when the peak-to-peak variation drops to 5% of peak overshoot.
                peak_fe = float(np.max(np.abs(fe_settle)))
                envelope_band = max(0.05 * peak_fe, 1e-6)
                
                unstable = rolling_envelope > envelope_band
                unstable_idx = np.where(unstable)[0]
                
                if len(unstable_idx) == 0:
                    # Already settled at move-end
                    settling_time_ms = 0.0
                else:
                    # last_unstable is the start index of the last sliding window that was ringing.
                    # The signal is mathematically considered "settled for 10ms" at the exact sample
                    # where this final unstable 10ms window ends.
                    last_unstable = int(unstable_idx[-1])
                    settle_idx = last_unstable + win_samples
                    
                    if settle_idx < len(t_settle):
                        settling_time_ms = float(t_settle[settle_idx]) * 1000.0
                    else:
                        settling_time_ms = float(t_settle[-1]) * 1000.0

        # Damping ratio from overshoot
        damping_ratio = 0.0
        if overshoot_pct > 0.1:
            os_frac = overshoot_pct / 100.0
            ln_os = math.log(os_frac)
            damping_ratio = -ln_os / math.sqrt(math.pi**2 + ln_os**2)

        # Rise time — move duration (not classical 10%-90% for profiled moves)
        rise_time_ms = float(
            time_arr[move_end_idx] - time_arr[move_start_idx]
        ) * 1000.0

        # Natural frequency from oscillation period during settle
        natural_freq_hz = 0.0
        if oscillation_count >= 2:
            zc_indices = np.where(sign_changes != 0)[0]
            if len(zc_indices) >= 4:
                zc_times = t_settle[zc_indices]
                half_periods = np.diff(zc_times)
                period = float(np.median(half_periods)) * 2.0
                if period > 0:
                    natural_freq_hz = 1.0 / period

        # Following-error peak — absolute max |FE| across the whole capture
        # (covers accel/decel transitions and ringdown, not just cruise).
        drive_fe_peak = float(np.max(np.abs(fe)))
        drive_fe_peak_pct = drive_fe_peak / step_size * 100.0

        # Mean |FE| during cruise phase (constant-velocity segment).
        drive_fe_cruise_mean = 0.0
        drive_fe_cruise_mean_pct = 0.0
        dacc = np.gradient(dvel, time_arr)
        a_peak = float(np.max(np.abs(dacc)))
        if a_peak > 1e-9:
            cruise_mask = moving & (np.abs(dacc) <= 0.10 * a_peak)
            if cruise_mask.sum() > 5:
                drive_fe_cruise_mean = float(np.mean(np.abs(fe[cruise_mask])))
                drive_fe_cruise_mean_pct = drive_fe_cruise_mean / step_size * 100.0

        # Velocity loop analysis — requires captured DEMAND_SPEED (units/s).
        vel_metrics: VelocityLoopMetrics | None = None
        if velocity is not None and demand_velocity is not None:
            vel_metrics = ClassicalTuner.analyze_velocity_loop(
                time_arr, velocity, demand_velocity
            )

        metrics = StepResponseMetrics(
            overshoot_pct=overshoot_pct,
            settling_time_ms=settling_time_ms,
            rise_time_ms=rise_time_ms,
            steady_state_error=steady_state_error,
            oscillation_count=oscillation_count,
            damping_ratio=damping_ratio,
            natural_freq_est_hz=natural_freq_hz,
            drive_fe_peak=drive_fe_peak,
            drive_fe_cruise_mean=drive_fe_cruise_mean,
            drive_fe_peak_pct=drive_fe_peak_pct,
            drive_fe_cruise_mean_pct=drive_fe_cruise_mean_pct,
        )

        return (metrics, vel_metrics)

    # --------------------------------------------------------- bandwidth calc
    @staticmethod
    def bandwidth_calculate(
        pn102: float, pn103: float, pn104: float
    ) -> dict[str, float]:
        """Estimate closed-loop bandwidths from gain parameters.

        pn102: speed loop gain (rad/s)
        pn103: speed loop integral time (×0.1 ms)
        pn104: position loop gain (1/s)
        """
        ti_s = pn103 * 0.1e-3
        speed_bw_hz = pn102 / (2 * math.pi)
        pos_bw_hz = pn104 / (2 * math.pi)
        integral_freq_hz = 1.0 / (2 * math.pi * ti_s) if ti_s > 0 else 0.0
        return {
            "speed_loop_bw_hz": round(speed_bw_hz, 1),
            "position_loop_bw_hz": round(pos_bw_hz, 1),
            "integral_freq_hz": round(integral_freq_hz, 1),
            "speed_to_position_ratio": (
                round(speed_bw_hz / pos_bw_hz, 1) if pos_bw_hz > 0 else float("inf")
            ),
        }

    # ------------------------------------------------------- oscillation det
    @staticmethod
    def detect_oscillation(
        time_arr: np.ndarray, signal: np.ndarray, min_freq: float = 5.0
    ) -> dict:
        """Detect dominant oscillation frequency via FFT.

        Returns dict with freq_hz, amplitude, and is_oscillating flag.
        """
        n = len(signal)
        if n < 64:
            return {"freq_hz": 0.0, "amplitude": 0.0, "is_oscillating": False}

        dt = float(np.median(np.diff(time_arr)))
        fs = 1.0 / dt if dt > 0 else 0.0
        if fs <= 0:
            return {"freq_hz": 0.0, "amplitude": 0.0, "is_oscillating": False}

        x = signal - np.mean(signal)
        window = np.hanning(n)
        X = np.fft.rfft(x * window)
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        mag = np.abs(X) * 2.0 / n / np.mean(window)

        valid = freqs >= min_freq
        if not np.any(valid):
            return {"freq_hz": 0.0, "amplitude": 0.0, "is_oscillating": False}

        mag_v = mag[valid]
        freqs_v = freqs[valid]
        noise_floor = float(np.median(mag_v))
        idx = int(np.argmax(mag_v))
        peak_freq = float(freqs_v[idx])
        peak_amp = float(mag_v[idx])
        is_osc = peak_amp > 5.0 * noise_floor

        return {
            "freq_hz": round(peak_freq, 1),
            "amplitude": round(peak_amp, 6),
            "noise_floor": round(noise_floor, 6),
            "is_oscillating": bool(is_osc),
        }
