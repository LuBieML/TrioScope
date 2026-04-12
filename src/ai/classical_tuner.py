"""
Classical servo-loop analyser and correction engine.

Two-level analysis:
  1. Velocity loop — compares MSPEED against demand velocity derived from the
     command profile.  Detects overshoot, ringing, and tracking errors during
     accel/cruise phases.
  2. Position loop — analyses following-error (FE = command − response) after
     the profiled move completes.  Measures overshoot, oscillation, settling
     time, steady-state error, and damping.

Corrections follow inside-out cascade priority: the velocity loop (Pn102/Pn103)
must be healthy before the position loop (Pn104) is touched.

Tuning-mode awareness (Pn100.0):
  1 = Tuningless  → only Pn101 (servo rigidity 1–31)
  3 = One-Param   → only Pn101
  5 = Manual       → Pn102, Pn103, Pn104

All analysis is designed for real profiled moves (trapezoidal / S-curve ramps),
NOT ideal step inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .drive_profile import DriveProfile

# ---------------------------------------------------------------------------
# Parameter limits (from PARAM_DEFS in drive_profile.py)
# ---------------------------------------------------------------------------
_PN_LIMITS: dict[str, tuple[int, int]] = {
    "pn101": (1, 31),
    "pn102": (1, 10000),
    "pn103": (1, 5000),
    "pn104": (0, 1000),
    "pn112": (0, 100),
    "pn113": (0, 640),
    "pn114": (0, 100),
    "pn115": (0, 640),
    "pn135": (0, 30000),
}


def _clamp_pn(name: str, value: float) -> int:
    """Clamp a parameter value to its valid range and round to int."""
    lo, hi = _PN_LIMITS.get(name, (0, 65535))
    return max(lo, min(hi, round(value)))


# ---------------------------------------------------------------------------
# Bandwidth presets
# ---------------------------------------------------------------------------
BANDWIDTH_PRESETS: dict[str, dict[str, float]] = {
    "conservative": {"pn102": 200, "pn103": 200, "pn104": 20},
    "moderate":     {"pn102": 500, "pn103": 125, "pn104": 40},
    "aggressive":   {"pn102": 1000, "pn103": 80, "pn104": 80},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class VelocityLoopMetrics:
    """Velocity loop quality assessed from MSPEED vs demand velocity."""
    accel_overshoot_pct: float = 0.0
    cruise_tracking_ratio: float = 1.0
    cruise_velocity_std: float = 0.0
    accel_settle_time_ms: float = 0.0
    accel_oscillation_count: int = 0
    is_healthy: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass
class StepResponseMetrics:
    """Position-loop quality from following-error analysis after move end."""
    overshoot_pct: float = 0.0
    settling_time_ms: float = 0.0
    rise_time_ms: float = 0.0
    steady_state_error: float = 0.0
    oscillation_count: int = 0
    damping_ratio: float = 0.0
    natural_freq_est_hz: float = 0.0
    velocity_overshoot_pct: float = 0.0   # from VelocityLoopMetrics when available

    def grade(self) -> tuple[float, str, list[str]]:
        """Score the response 0–10 with symptom-only detail messages.

        Returns (score, verdict, detail_list).
        """
        score = 10.0
        details: list[str] = []

        # --- Overshoot ---
        if self.overshoot_pct <= 5:
            details.append(f"Overshoot {self.overshoot_pct:.1f}% — excellent")
        elif self.overshoot_pct <= 15:
            score -= 2.0
            details.append(
                f"Overshoot {self.overshoot_pct:.1f}% — loop response too aggressive"
            )
        elif self.overshoot_pct <= 25:
            score -= 4.0
            details.append(
                f"Overshoot {self.overshoot_pct:.1f}% — high, loop response too aggressive"
            )
        else:
            score -= 6.0
            details.append(
                f"Overshoot {self.overshoot_pct:.1f}% — excessive, loop response too aggressive"
            )

        # --- Oscillations ---
        if self.oscillation_count <= 1:
            details.append(
                f"Oscillations: {self.oscillation_count} — excellent (well damped)"
            )
        elif self.oscillation_count <= 3:
            score -= 1.5
            details.append(
                f"Oscillations: {self.oscillation_count} — moderate ringing"
            )
        else:
            score -= 3.0
            details.append(
                f"Oscillations: {self.oscillation_count} — significant ringing, "
                f"gains need reduction"
            )

        # --- Steady-state error ---
        ss_pct = self.steady_state_error * 100.0
        has_overshoot = self.overshoot_pct > 5
        if ss_pct <= 0.5:
            details.append(f"Steady-state error {ss_pct:.2f}% — excellent")
        elif ss_pct <= 2.0:
            score -= 1.0
            details.append(f"Steady-state error {ss_pct:.2f}% — good")
        else:
            score -= 2.0
            if has_overshoot:
                details.append(
                    f"Steady-state error {ss_pct:.2f}% — resolve overshoot first, "
                    f"then assess"
                )
            else:
                details.append(
                    f"Steady-state error {ss_pct:.2f}% — consider faster integral action"
                )

        # --- Settling time ---
        if self.settling_time_ms <= 50:
            details.append(f"Settling time {self.settling_time_ms:.0f}ms — excellent")
        elif self.settling_time_ms <= 200:
            score -= 0.5
            details.append(f"Settling time {self.settling_time_ms:.0f}ms — good")
        elif self.settling_time_ms <= 500:
            score -= 1.5
            details.append(f"Settling time {self.settling_time_ms:.0f}ms — slow")
        else:
            score -= 3.0
            if has_overshoot:
                details.append(
                    f"Settling time {self.settling_time_ms:.0f}ms — very slow, "
                    f"caused by overshoot"
                )
            else:
                details.append(
                    f"Settling time {self.settling_time_ms:.0f}ms — very slow, "
                    f"consider faster integral action"
                )

        score = max(0.0, min(10.0, score))

        if score >= 8:
            verdict = "Well tuned"
        elif score >= 5:
            verdict = "Acceptable — minor improvements possible"
        else:
            verdict = "Needs attention"

        return score, verdict, details


@dataclass
class TuningResult:
    """Proposed parameter changes with diagnostics."""
    method: str = ""
    pn101: Optional[int] = None
    pn102: Optional[int] = None
    pn103: Optional[int] = None
    pn104: Optional[int] = None
    pn106: Optional[int] = None
    pn112: Optional[int] = None
    pn113: Optional[int] = None
    pn114: Optional[int] = None
    pn115: Optional[int] = None
    pn135: Optional[int] = None
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    confidence: str = "low"

    def to_profile_delta(self) -> dict[str, int]:
        """Return only the Pn fields that have proposed values."""
        delta = {}
        for attr in (
            "pn101", "pn102", "pn103", "pn104", "pn106",
            "pn112", "pn113", "pn114", "pn115", "pn135",
        ):
            val = getattr(self, attr)
            if val is not None:
                delta[attr] = val
        return delta


# ---------------------------------------------------------------------------
# Classical tuner
# ---------------------------------------------------------------------------
class ClassicalTuner:
    """Analyse profiled-move captures and suggest gain corrections."""

    # --------------------------------------------------------------- velocity
    @staticmethod
    def analyze_velocity_loop(
        time_arr: np.ndarray,
        measured_velocity: np.ndarray,
        demand_velocity: np.ndarray,
    ) -> VelocityLoopMetrics:
        """Assess velocity-loop quality from MSPEED vs captured DEMAND_SPEED.

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
                    # Check all remaining in a small window stay within band
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

        # Health assessment
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
        response: np.ndarray,
        command: np.ndarray,
        velocity: np.ndarray | None = None,
        demand_velocity: np.ndarray | None = None,
    ) -> tuple[StepResponseMetrics, VelocityLoopMetrics | None]:
        """Analyse a profiled-move capture for position-loop quality.

        Returns (position_metrics, velocity_metrics_or_None).
        """
        if len(time_arr) < 20:
            return (StepResponseMetrics(), None)

        dt = float(np.median(np.diff(time_arr)))
        fe = command - response
        dvel = np.gradient(command, time_arr)
        v_peak = float(np.max(np.abs(dvel)))

        if v_peak < 1e-9:
            return (StepResponseMetrics(), None)

        # Find move-complete — last transition from moving to stopped
        moving = np.abs(dvel) > 0.02 * v_peak
        transitions = np.where(moving[:-1] & ~moving[1:])[0]
        if len(transitions) == 0:
            return (StepResponseMetrics(), None)

        # Settle window
        move_end_idx = transitions[-1] + 1
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

        # Settling time — time from move-end until FE stays within +/-2% of step_size
        band = 0.02 * step_size
        if band < 1e-9:
            band = peak_fe * 0.05
        within_band = np.abs(fe_settle) <= band
        settling_time_ms = 0.0
        for i in range(len(within_band) - 1, -1, -1):
            if not within_band[i]:
                if i < len(t_settle) - 1:
                    settling_time_ms = float(t_settle[i + 1]) * 1000.0
                break

        # Steady-state error — mean FE in last 10% of settle window
        tail_len = max(1, len(fe_settle) // 10)
        steady_state_error = abs(float(np.mean(fe_settle[-tail_len:]))) / step_size

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
            velocity_overshoot_pct=(
                vel_metrics.accel_overshoot_pct if vel_metrics is not None else 0.0
            ),
        )

        return (metrics, vel_metrics)

    # ----------------------------------------------------------- corrections
    @staticmethod
    def suggest_corrections(
        pos_metrics: StepResponseMetrics,
        vel_metrics: VelocityLoopMetrics | None,
        current_profile: DriveProfile,
    ) -> TuningResult:
        """Propose gain changes based on analysis results and tuning mode."""
        tuning_mode = getattr(current_profile, "pn100_tuning_mode", None)

        if tuning_mode in (1, 3):
            return ClassicalTuner._suggest_rigidity(pos_metrics, current_profile)
        elif tuning_mode == 5:
            return ClassicalTuner._suggest_manual(
                pos_metrics, vel_metrics, current_profile
            )
        else:
            result = ClassicalTuner._suggest_manual(
                pos_metrics, vel_metrics, current_profile
            )
            result.warnings.insert(
                0,
                "Tuning mode unknown — assuming Manual (Pn100.0=5). "
                "Read drive profile to confirm.",
            )
            return result

    # -------------------------------------------------- rigidity (mode 1 / 3)
    @staticmethod
    def _suggest_rigidity(
        pos_metrics: StepResponseMetrics,
        current_profile: DriveProfile,
    ) -> TuningResult:
        """Suggest Pn101 (servo rigidity) changes for Tuningless / One-Param modes."""
        pn101 = current_profile.pn101
        if pn101 is None:
            return TuningResult(
                method="step_response",
                warnings=["Pn101 unknown — read drive profile first."],
                confidence="low",
            )

        score, verdict, grade_details = pos_metrics.grade()

        if score >= 8:
            return TuningResult(
                method="step_response",
                diagnostics={
                    "tuning_score": f"{score}/10",
                    "verdict": verdict,
                    "assessment": grade_details,
                },
                warnings=[
                    f"Tuning Score: {score}/10 — {verdict}. No changes needed."
                ],
                confidence="high",
            )

        new_101 = float(pn101)
        reasons: list[str] = []
        has_overshoot = pos_metrics.overshoot_pct > 5

        if pos_metrics.oscillation_count > 3 or pos_metrics.overshoot_pct > 25:
            new_101 *= 0.80
            reasons.append(
                f"Overshoot {pos_metrics.overshoot_pct:.1f}% / "
                f"{pos_metrics.oscillation_count} oscillations "
                f"→ Pn101 Servo Rigidity -20% (softer)"
            )
        elif pos_metrics.overshoot_pct > 10:
            new_101 *= 0.90
            reasons.append(
                f"Overshoot {pos_metrics.overshoot_pct:.1f}% "
                f"→ Pn101 Servo Rigidity -10%"
            )
        elif pos_metrics.overshoot_pct > 5:
            new_101 *= 0.95
            reasons.append(
                f"Overshoot {pos_metrics.overshoot_pct:.1f}% "
                f"→ Pn101 Servo Rigidity -5%"
            )
        elif pos_metrics.settling_time_ms > 200 and not has_overshoot:
            new_101 *= 1.15
            reasons.append(
                f"Slow settling ({pos_metrics.settling_time_ms:.0f}ms) "
                f"→ Pn101 Servo Rigidity +15% (stiffer)"
            )
        elif pos_metrics.steady_state_error > 0.02 and not has_overshoot:
            new_101 *= 1.10
            reasons.append(
                f"Steady-state error {pos_metrics.steady_state_error:.4f} "
                f"→ Pn101 Servo Rigidity +10%"
            )

        new_101_clamped = max(1, min(31, round(new_101)))

        result = TuningResult(
            method="step_response",
            diagnostics={
                "tuning_score": f"{score}/10",
                "verdict": verdict,
                "metrics": {
                    "overshoot_pct": pos_metrics.overshoot_pct,
                    "settling_time_ms": pos_metrics.settling_time_ms,
                    "oscillation_count": pos_metrics.oscillation_count,
                    "steady_state_error": pos_metrics.steady_state_error,
                },
                "assessment": grade_details,
                "corrections": reasons,
            },
            warnings=[f"Tuning Score: {score}/10 — {verdict}."],
            confidence="medium",
        )

        if int(new_101_clamped) != pn101:
            result.pn101 = int(new_101_clamped)

        return result

    # --------------------------------------------------- manual (mode 5)
    @staticmethod
    def _suggest_manual(
        pos_metrics: StepResponseMetrics,
        vel_metrics: VelocityLoopMetrics | None,
        current_profile: DriveProfile,
    ) -> TuningResult:
        """Suggest Pn102/Pn103/Pn104 changes for Manual tuning mode."""
        pn102 = current_profile.pn102
        pn103 = current_profile.pn103
        pn104 = current_profile.pn104

        if any(v is None for v in (pn102, pn103, pn104)):
            missing = [
                f"Pn{code}"
                for code, val in [("102", pn102), ("103", pn103), ("104", pn104)]
                if val is None
            ]
            return TuningResult(
                method="step_response",
                warnings=[
                    f"{', '.join(missing)} unknown — read drive profile first."
                ],
                confidence="low",
            )

        score, verdict, grade_details = pos_metrics.grade()

        if score >= 8:
            return TuningResult(
                method="step_response",
                diagnostics={
                    "tuning_score": f"{score}/10",
                    "verdict": verdict,
                    "assessment": grade_details,
                },
                warnings=[
                    f"Tuning Score: {score}/10 — {verdict}. No changes needed."
                ],
                confidence="high",
            )

        new_102 = float(pn102)
        new_103 = float(pn103)
        new_104 = float(pn104)
        reasons: list[str] = []
        loop_assessment = "full_cascade"

        # ---- Step 1: Velocity loop (inside-out — MUST come first) ----
        if vel_metrics is not None:
            if not vel_metrics.is_healthy:
                loop_assessment = "velocity_only"

                if vel_metrics.accel_overshoot_pct > 30:
                    new_102 *= 0.85
                    reasons.append(
                        f"Velocity overshoot {vel_metrics.accel_overshoot_pct:.1f}% "
                        f"(>30%) → Pn102 -15%"
                    )
                elif vel_metrics.accel_overshoot_pct > 15:
                    new_102 *= 0.90
                    reasons.append(
                        f"Velocity overshoot {vel_metrics.accel_overshoot_pct:.1f}% "
                        f"(>15%) → Pn102 -10%"
                    )

                if vel_metrics.accel_oscillation_count > 3:
                    new_102 *= 0.90
                    new_103 *= 1.15
                    reasons.append(
                        f"Velocity ringing "
                        f"({vel_metrics.accel_oscillation_count} oscillations) "
                        f"→ Pn102 -10%, Pn103 +15%"
                    )

                if vel_metrics.cruise_tracking_ratio < 0.95:
                    new_102 *= 1.15
                    reasons.append(
                        f"Velocity not reaching demand "
                        f"(ratio {vel_metrics.cruise_tracking_ratio:.3f}) "
                        f"→ Pn102 +15%"
                    )

                # Return immediately — do NOT assess position loop
                return TuningResult(
                    method="step_response",
                    pn102=_clamp_pn("pn102", new_102),
                    pn103=_clamp_pn("pn103", new_103),
                    diagnostics={
                        "tuning_score": f"{score}/10",
                        "verdict": verdict,
                        "loop_assessment": loop_assessment,
                        "velocity_issues": vel_metrics.issues,
                        "assessment": grade_details,
                        "corrections": reasons,
                    },
                    warnings=[
                        f"Tuning Score: {score}/10 — {verdict}.",
                        "Velocity loop has issues — fix Pn102/Pn103 first, "
                        "then re-capture to assess position loop.",
                    ],
                    confidence="medium",
                )
            # else: velocity loop healthy, proceed to position loop
        else:
            loop_assessment = "position_only"

        # ---- Step 2: Position loop (velocity healthy or unavailable) ----
        has_overshoot = pos_metrics.overshoot_pct > 5

        # High-frequency ringing → mechanical resonance
        if (
            pos_metrics.oscillation_count > 3
            and pos_metrics.natural_freq_est_hz > 100
        ):
            reasons.append(
                f"Ringing at {pos_metrics.natural_freq_est_hz:.0f} Hz — "
                f"likely mechanical resonance. Use notch filter (Pn4xx) or "
                f"vibration suppression, not gain changes."
            )
            return TuningResult(
                method="step_response",
                diagnostics={
                    "tuning_score": f"{score}/10",
                    "verdict": verdict,
                    "loop_assessment": loop_assessment,
                    "assessment": grade_details,
                    "corrections": reasons,
                },
                warnings=[
                    f"Tuning Score: {score}/10 — {verdict}.",
                    "Mechanical resonance detected — gains not adjusted.",
                ],
                confidence="medium",
            )

        # Ringing at lower frequency → position loop issue
        if pos_metrics.oscillation_count > 3:
            new_104 *= 0.85
            new_103 *= 1.20
            reasons.append(
                f"Ringing ({pos_metrics.oscillation_count} oscillations) "
                f"→ Pn104 -15%, Pn103 +20%"
            )
        elif pos_metrics.overshoot_pct > 25:
            new_104 *= 0.85
            reasons.append(
                f"Overshoot {pos_metrics.overshoot_pct:.1f}% (>25%) → Pn104 -15%"
            )
        elif pos_metrics.overshoot_pct > 10:
            new_104 *= 0.90
            reasons.append(
                f"Overshoot {pos_metrics.overshoot_pct:.1f}% (>10%) → Pn104 -10%"
            )
        elif pos_metrics.overshoot_pct > 5:
            new_104 *= 0.95
            reasons.append(
                f"Overshoot {pos_metrics.overshoot_pct:.1f}% (>5%) → Pn104 -5%"
            )
        elif pos_metrics.settling_time_ms > 200 and not has_overshoot:
            new_104 *= 1.15
            reasons.append(
                f"Sluggish settling ({pos_metrics.settling_time_ms:.0f}ms) "
                f"→ Pn104 +15%"
            )

        # Steady-state error — only speed up integral if NOT fighting overshoot
        if pos_metrics.steady_state_error > 0.02 and not has_overshoot:
            new_103 *= 0.85
            reasons.append(
                f"Steady-state error {pos_metrics.steady_state_error:.4f} "
                f"→ Pn103 -15% (faster integral)"
            )

        # Slow settling caused by overshoot → need to damp, not tighten
        if pos_metrics.settling_time_ms > 500 and has_overshoot:
            new_104 *= 0.90
            reasons.append(
                f"Slow settling ({pos_metrics.settling_time_ms:.0f}ms with overshoot) "
                f"→ Pn104 -10%"
            )

        # Build result — NEVER set pn102 in position-loop block
        result_warnings = [f"Tuning Score: {score}/10 — {verdict}."]
        if vel_metrics is None:
            result_warnings.append(
                "No MSPEED data — velocity loop not assessed. "
                "Capture MSPEED for full cascade analysis."
            )

        result = TuningResult(
            method="step_response",
            diagnostics={
                "tuning_score": f"{score}/10",
                "verdict": verdict,
                "loop_assessment": loop_assessment,
                "metrics": {
                    "overshoot_pct": pos_metrics.overshoot_pct,
                    "settling_time_ms": pos_metrics.settling_time_ms,
                    "oscillation_count": pos_metrics.oscillation_count,
                    "steady_state_error": pos_metrics.steady_state_error,
                    "natural_freq_hz": pos_metrics.natural_freq_est_hz,
                },
                "assessment": grade_details,
                "corrections": reasons,
            },
            warnings=result_warnings,
            confidence="medium",
        )

        clamped_103 = _clamp_pn("pn103", new_103)
        clamped_104 = _clamp_pn("pn104", new_104)
        if clamped_103 != pn103:
            result.pn103 = clamped_103
        if clamped_104 != pn104:
            result.pn104 = clamped_104

        return result

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
        ti_s = pn103 * 0.1e-3  # convert to seconds
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

    # ------------------------------------------------------ Ziegler-Nichols
    @staticmethod
    def ziegler_nichols_pi(ku: float, tu: float) -> dict[str, float]:
        """Ziegler-Nichols PI tuning from ultimate gain and period.

        ku: ultimate gain (where sustained oscillation begins)
        tu: ultimate period (seconds)
        """
        kp = 0.45 * ku
        ti = tu / 1.2
        return {"kp": round(kp, 4), "ti_s": round(ti, 6)}

    @staticmethod
    def ziegler_nichols_pid(ku: float, tu: float) -> dict[str, float]:
        """Ziegler-Nichols PID tuning from ultimate gain and period."""
        kp = 0.6 * ku
        ti = tu / 2.0
        td = tu / 8.0
        return {"kp": round(kp, 4), "ti_s": round(ti, 6), "td_s": round(td, 6)}

    # --------------------------------------------------------- feedforward
    @staticmethod
    def estimate_feedforward(
        time_arr: np.ndarray,
        command: np.ndarray,
        response: np.ndarray,
        demand_velocity: np.ndarray,
        current_pn112: int = 0,
        current_pn114: int = 0,
    ) -> dict:
        """Estimate speed and torque feedforward percentages from a capture.

        Analyses cruise-phase FE proportionality to velocity (→ Pn112) and
        accel-phase FE proportionality to acceleration (→ Pn114).

        demand_velocity must be DEMAND_SPEED already scaled to units/second.
        """
        dvel = demand_velocity
        dacc = np.gradient(dvel, time_arr)
        fe = command - response

        v_peak = float(np.max(np.abs(dvel)))
        a_peak = float(np.max(np.abs(dacc)))
        if v_peak < 1e-9:
            return {"note": "no motion detected"}

        moving = np.abs(dvel) > 0.02 * v_peak
        accel = moving & (np.abs(dacc) > 0.10 * a_peak) if a_peak > 1e-9 else np.zeros_like(moving)
        cruise = moving & ~accel

        result: dict = {}

        # Speed feedforward (Pn112) — from cruise FE vs velocity slope
        if cruise.sum() > 20:
            v = dvel[cruise]
            f = fe[cruise]
            if np.std(v) > 1e-9:
                slope, _ = np.polyfit(v, f, 1)
                # slope ≈ missing VFF fraction * (1/Kp_pos)
                # Suggest increasing Pn112 proportionally
                residual_vff_pct = abs(slope) * 100
                suggested = min(100, current_pn112 + round(residual_vff_pct))
                result["speed_feedforward"] = {
                    "current_pn112": current_pn112,
                    "fe_velocity_slope": round(float(slope), 6),
                    "suggested_pn112": suggested,
                    "note": "slope>0 → FE proportional to velocity → increase Pn112",
                }

        # Torque feedforward (Pn114) — from accel FE vs acceleration slope
        if accel.sum() > 20:
            a = dacc[accel]
            f = fe[accel]
            if np.std(a) > 1e-9:
                slope, _ = np.polyfit(a, f, 1)
                residual_tff_pct = abs(slope) * 100
                suggested = min(100, current_pn114 + round(residual_tff_pct))
                result["torque_feedforward"] = {
                    "current_pn114": current_pn114,
                    "fe_accel_slope": round(float(slope), 6),
                    "suggested_pn114": suggested,
                    "note": "slope>0 → FE proportional to accel → increase Pn114",
                }

        if not result:
            result["note"] = "insufficient cruise/accel data for feedforward estimation"

        return result
