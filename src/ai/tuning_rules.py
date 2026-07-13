"""Offline rule-based tuning recommendations — Qt-free.

The same diagnostic table the AI panel's system prompt encodes, as plain
Python over the SignalMetrics result dict, so the Servo Loop Analyser can
say *"increase Pn112 by ≤10 points — cruise FE slope ∝ velocity"* without
an API key.

Structure follows the mandated inside-out decision order:

  1. current loop   — saturation / resonance / instability are BLOCKING:
                      when one fires, no other gain advice is issued
  2. velocity loop  — reach ratio, overshoot
  3. position / FE  — feedforward (VFF then AFF), ringing, integral action
  m. mechanical     — reversal spikes, asymmetry (observations, not knobs)

Step limits per iteration: gains ±15-20 %, feedforward ±10 points, at most
three parameter changes. Pn100 auto-tuning modes gate direct gain advice
to Pn101 rigidity. Operation mode is assumed CSP (controller closes the
position loop) unless stated — recommendations name both the controller
parameter and the drive equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .drive_profile import PARAM_DEFS

# Spinbox bounds from the drive profile definitions: attr → (min, max)
_PN_BOUNDS: dict[str, tuple[int, int]] = {
    entry[0]: (entry[4], entry[5])
    for entry in PARAM_DEFS
    if entry[4] is not None and entry[5] is not None
}

WELL_TUNED_SCORE = 8.0
MAX_PARAMETER_CHANGES = 3

# Accel/decel FE peak vs cruise peak ratio that reads as "missing AFF"
_AFF_PEAK_RATIO = 3.0
# Reversal FE peak vs cruise peak ratio that reads as mechanical
_REVERSAL_PEAK_RATIO = 5.0
_SATURATION_PCT_LIMIT = 5.0


@dataclass
class Recommendation:
    rule_id: str
    severity: str            # "action" | "observe" | "info"
    root_cause: str          # current|velocity|position|mechanical|profile
    action: str              # headline: what to do
    diagnosis: str           # why: symptom with cited metric names
    expected: str = ""       # what should improve on the next capture
    parameter: str | None = None    # primary drive parameter, e.g. "Pn112"
    proposed: str | None = None     # "30 → 40" when computable


@dataclass
class TuningReport:
    score: float | None
    well_tuned: bool
    root_cause: str
    recommendations: list[Recommendation] = field(default_factory=list)
    observations: list[Recommendation] = field(default_factory=list)
    profile_attached: bool = False  # Pn snapshot was available for proposals


# ---------------------------------------------------------------- helpers
def _pn(profile: dict | None, attr: str) -> int | None:
    if not profile:
        return None
    value = profile.get(attr)
    return int(value) if value is not None else None


def _propose(profile: dict | None, attr: str, label: str, *,
             factor: float | None = None, delta: int | None = None) -> str | None:
    """"500 → 575" for a relative change to a known Pn value, else None."""
    current = _pn(profile, attr)
    if current is None:
        return None
    if factor is not None:
        target = round(current * factor)
    else:
        target = current + (delta or 0)
    lo, hi = _PN_BOUNDS.get(attr, (0, 10 ** 9))
    target = max(lo, min(hi, target))
    if target == current:
        return None
    return f"{label} {current} → {target}"


def _tuning_mode(profile: dict | None) -> int | None:
    return _pn(profile, "pn100_tuning_mode")


def _phase_band(metrics: dict) -> tuple[float | None, float | None, str]:
    """(phase_deg, dominant_freq_hz, confidence_note) from the
    current-vs-velocity cross-spectrum."""
    cvp = (metrics.get("oscillation") or {}).get("current_vs_velocity_phase")
    if not cvp:
        return None, None, ""
    coherence = cvp.get("coherence")
    note = f", coherence {coherence}" if coherence is not None else ""
    return cvp.get("phase_deg"), cvp.get("dominant_freq_hz"), note


def _max_saturation(metrics: dict) -> float:
    current = metrics.get("current") or {}
    return max(
        (current.get(phase, {}).get("saturation_pct", 0.0)
         for phase in ("accel", "decel")),
        default=0.0,
    )


def _fe_stat(metrics: dict, phase: str, key: str) -> float | None:
    stats = (metrics.get("fe") or {}).get(phase)
    return stats.get(key) if stats else None


def _reversal_is_mechanical(metrics: dict) -> bool:
    rev_peak = _fe_stat(metrics, "reversal", "peak_abs")
    cruise_peak = _fe_stat(metrics, "cruise", "peak_abs")
    osc = (metrics.get("oscillation") or {}).get("fe") or {}
    return (rev_peak is not None and cruise_peak is not None
            and cruise_peak > 0
            and rev_peak > _REVERSAL_PEAK_RATIO * cruise_peak
            and not osc.get("has_significant_oscillation"))


# ---------------------------------------------------------------- score
def tuning_score(metrics: dict) -> float | None:
    """0-10 tuning score (the AI prompt's rubric: start at 10, subtract)."""
    if metrics.get("data_sufficiency") != "OK":
        return None

    score = 10.0
    fe_cruise_mean = _fe_stat(metrics, "cruise", "mean")
    fe_accel_peak = _fe_stat(metrics, "accel", "peak_abs")
    fe_decel_peak = _fe_stat(metrics, "decel", "peak_abs")
    fe_cruise_peak = _fe_stat(metrics, "cruise", "peak_abs")

    if (fe_cruise_mean is not None and fe_accel_peak is not None
            and fe_accel_peak > 0
            and abs(fe_cruise_mean) > 0.10 * fe_accel_peak):
        score -= 1.0

    if fe_cruise_peak is not None and fe_cruise_peak > 0:
        worst_ramp = max(fe_accel_peak or 0.0, fe_decel_peak or 0.0)
        if worst_ramp > _AFF_PEAK_RATIO * fe_cruise_peak:
            score -= 1.0

    osc_fe = (metrics.get("oscillation") or {}).get("fe") or {}
    if osc_fe.get("has_significant_oscillation"):
        score -= 2.0

    if _max_saturation(metrics) > _SATURATION_PCT_LIMIT:
        score -= 2.0

    if (metrics.get("asymmetry") or {}).get("significant"):
        score -= 1.0

    settle = metrics.get("settle") or {}
    if settle.get("ringing") or settle.get("zero_crossings", 0) > 3:
        score -= 1.0
    if settle.get("steady_state_offset_nonzero"):
        score -= 1.0

    if _reversal_is_mechanical(metrics):
        score -= 1.0  # mechanical root cause — deduct only one point

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------- rules
def _blocking_rules(metrics: dict, profile: dict | None) -> list[Recommendation]:
    """Current-loop findings that suppress all other gain advice."""
    found: list[Recommendation] = []

    saturation = _max_saturation(metrics)
    if saturation > _SATURATION_PCT_LIMIT:
        found.append(Recommendation(
            rule_id="torque_saturation",
            severity="action", root_cause="profile",
            action=("Reduce ACCEL/DECEL in the motion profile (or upsize "
                    "the motor) — do NOT tune gains while torque-limited"),
            diagnosis=(f"current saturation_pct {saturation:.1f}% during "
                       f"accel/decel (>{_SATURATION_PCT_LIMIT:g}%) — the "
                       f"drive is torque-limited"),
            expected="saturation_pct < 5, velocity reaches demand",
        ))

    osc_fe = (metrics.get("oscillation") or {}).get("fe") or {}
    if osc_fe.get("has_significant_oscillation"):
        phase_deg, phase_freq, confidence = _phase_band(metrics)
        freq = osc_fe.get("dominant_hz")
        if phase_deg is not None and 60 < phase_deg < 120:
            found.append(Recommendation(
                rule_id="mechanical_resonance",
                severity="action", root_cause="mechanical",
                action=(f"Apply the drive's notch filter at ~{phase_freq or freq} Hz "
                        f"(drive commissioning tool); do NOT increase "
                        f"position gains"),
                diagnosis=(f"oscillation.fe dominant_hz={freq} with "
                           f"current_vs_velocity_phase ≈ +90° "
                           f"({phase_deg:.0f}°{confidence}) — current leads "
                           f"velocity: mechanical resonance"),
                expected="oscillation peak gone at the notch frequency",
            ))
        elif phase_deg is not None and -30 < phase_deg < 30:
            proposed = _propose(profile, "pn104", "Pn104", factor=0.8)
            found.append(Recommendation(
                rule_id="loop_instability",
                severity="action", root_cause="position",
                action=("Reduce P_GAIN (CSP) or Pn104 (drive-closed loop) "
                        "by ~20%; if a low-frequency oscillation persists, "
                        "reduce integral action instead"),
                diagnosis=(f"oscillation.fe dominant_hz={freq} with "
                           f"current_vs_velocity_phase ≈ 0° "
                           f"({phase_deg:.0f}°{confidence}) — in phase: "
                           f"loop instability"),
                expected="has_significant_oscillation = false",
                parameter="Pn104", proposed=proposed,
            ))
        else:
            channels = metrics.get("channels_detected") or {}
            have_signals = bool(channels.get("current")) and bool(
                channels.get("measured_vel"))
            if not have_signals:
                found.append(Recommendation(
                    rule_id="oscillation_unresolved",
                    severity="action", root_cause="position",
                    action=("Capture DRIVE_TORQUE (or current) together "
                            "with MSPEED to discriminate resonance (+90°) "
                            "from loop instability (0°) before changing "
                            "gains"),
                    diagnosis=(f"oscillation.fe dominant_hz={freq} is "
                               f"significant but no current/velocity "
                               f"channels were captured for the phase test"),
                    expected="a phase verdict on the next capture",
                ))
            elif phase_deg is None:
                found.append(Recommendation(
                    rule_id="oscillation_no_coherent_phase",
                    severity="action", root_cause="position",
                    action=("Re-capture with 2-3 repeated moves and a "
                            "longer cruise (>0.3 s each) — current and "
                            "velocity are captured but no coherent shared "
                            "line passed the gate, and the resonance-vs-"
                            "instability verdict needs more spectral "
                            "averages"),
                    diagnosis=(f"oscillation.fe dominant_hz={freq} is "
                               f"significant but the current-vs-velocity "
                               f"cross-spectrum found no coherent bin"),
                    expected="a phase verdict on the next capture",
                ))
            else:
                found.append(Recommendation(
                    rule_id="oscillation_ambiguous_phase",
                    severity="action", root_cause="position",
                    action=(f"Phase {phase_deg:.0f}° sits between the "
                            f"instability (~0°) and resonance (~+90°) "
                            f"signatures — reduce P_GAIN (CSP) or Pn104 "
                            f"(drive-closed loop) by ~15% and re-capture: "
                            f"if the {freq} Hz peak shifts with the gain it "
                            f"is loop-related; if it stays fixed, apply a "
                            f"notch at {freq} Hz instead. Do not increase "
                            f"any gain while the oscillation persists"),
                    diagnosis=(f"oscillation.fe dominant_hz={freq} with "
                               f"current_vs_velocity_phase "
                               f"{phase_deg:.0f}°{confidence} — outside both "
                               f"classification windows (measurement and "
                               f"filter lag can shift a true resonance "
                               f"beyond +90°)"),
                    expected=("the {0} Hz peak's response to the gain "
                              "change identifies the root cause".format(freq)),
                    parameter="Pn104",
                    proposed=_propose(profile, "pn104", "Pn104", factor=0.85),
                ))

    return found


def _velocity_rules(metrics: dict, profile: dict | None) -> list[Recommendation]:
    found: list[Recommendation] = []
    vel = metrics.get("velocity") or {}

    ratio = vel.get("cruise_velocity_reach_ratio")
    if ratio is not None and ratio < 0.95:
        # saturation case is handled by the blocking rule
        torque_note = ("with current not saturated"
                       if metrics.get("current")
                       else "— no current channel captured, so confirm the "
                            "drive is not torque-limited first")
        found.append(Recommendation(
            rule_id="velocity_under_responsive",
            severity="action", root_cause="velocity",
            action=("Increase Pn102 speed loop gain by 15-20% "
                    "(velocity loop under-responsive)"),
            diagnosis=(f"cruise_velocity_reach_ratio {ratio:.3f} < 0.95 "
                       f"{torque_note}"),
            expected="reach ratio → 1.0",
            parameter="Pn102",
            proposed=_propose(profile, "pn102", "Pn102", factor=1.15),
        ))

    overshoot = (vel.get("velocity_overshoot_per_move") or {})
    if overshoot.get("max_pct", 0.0) > 15.0:
        found.append(Recommendation(
            rule_id="velocity_overshoot",
            severity="action", root_cause="velocity",
            action=("Reduce Pn102 speed loop gain, or reduce Pn112 speed "
                    "feedforward, by 15% (velocity loop too aggressive)"),
            diagnosis=(f"velocity_overshoot_per_move.max_pct "
                       f"{overshoot['max_pct']:.1f}% > 15% during accel"),
            expected="accel overshoot < 15%",
            parameter="Pn102",
            proposed=_propose(profile, "pn102", "Pn102", factor=0.85),
        ))

    return found


def _fe_rules(metrics: dict, profile: dict | None) -> list[Recommendation]:
    found: list[Recommendation] = []
    fe = metrics.get("fe") or {}
    settle = metrics.get("settle") or {}

    fit = fe.get("cruise_fe_vs_velocity") or {}
    vff_needed = bool(fit.get("proportional_to_velocity"))
    if vff_needed:
        found.append(Recommendation(
            rule_id="vff_insufficient",
            severity="action", root_cause="position",
            action=("Increase VFF_GAIN toward 1.0 (CSP) or Pn112 speed "
                    "feedforward (drive-closed loop) by ≤10 points"),
            diagnosis=(f"cruise_fe_vs_velocity slope {fit.get('slope')} with "
                       f"proportional_to_velocity=true — FE scales with speed"),
            expected="slope → 0, cruise fe.mean → 0 (if FE flips sign, "
                     "feedforward is too high)",
            parameter="Pn112",
            proposed=_propose(profile, "pn112", "Pn112", delta=10),
        ))

    cruise_peak = _fe_stat(metrics, "cruise", "peak_abs")
    worst_ramp = max(_fe_stat(metrics, "accel", "peak_abs") or 0.0,
                     _fe_stat(metrics, "decel", "peak_abs") or 0.0)
    if (cruise_peak is not None and cruise_peak > 0
            and worst_ramp > _AFF_PEAK_RATIO * cruise_peak):
        order_note = (" — confirm VFF first (rule above), then re-evaluate"
                      if vff_needed else "")
        found.append(Recommendation(
            rule_id="aff_insufficient",
            severity="action", root_cause="position",
            action=("Increase AFF_GAIN (CSP) or Pn114 torque feedforward "
                    f"(drive-closed loop) by ≤10 points, target 60-80%"
                    f"{order_note}"),
            diagnosis=(f"fe accel/decel peak_abs {worst_ramp:g} is "
                       f">{_AFF_PEAK_RATIO:g}× cruise peak {cruise_peak:g} "
                       f"— FE spikes only while accelerating"),
            expected="accel/decel FE peaks approach the cruise level",
            parameter="Pn114",
            proposed=_propose(profile, "pn114", "Pn114", delta=10),
        ))

    if settle.get("ringing"):
        found.append(Recommendation(
            rule_id="underdamped_settle",
            severity="action", root_cause="position",
            action=("Increase D_GAIN or reduce P_GAIN (CSP); for "
                    "drive-closed loops reduce Pn104 by 15%"),
            diagnosis=(f"settle.ringing=true "
                       f"({settle.get('zero_crossings')} band crossings, "
                       f"damping_ratio={settle.get('damping_ratio')})"),
            expected="zero_crossings ≤ 3",
            parameter="Pn104",
            proposed=_propose(profile, "pn104", "Pn104", factor=0.85),
        ))

    if settle.get("steady_state_offset_nonzero"):
        found.append(Recommendation(
            rule_id="integral_insufficient",
            severity="action", root_cause="position",
            action=("Increase I_GAIN (CSP) or decrease Pn103 speed-loop Ti "
                    "(drive-closed loop) by ~15% — smallest steps: integral "
                    "is the most stability-threatening gain"),
            diagnosis=(f"settle.steady_state_offset_nonzero=true "
                       f"(fe_steady_state {settle.get('fe_steady_state')} "
                       f"outside ±{settle.get('band')})"),
            expected="steady-state FE inside the tolerance band",
            parameter="Pn103",
            proposed=_propose(profile, "pn103", "Pn103", factor=0.85),
        ))

    return found


def _mechanical_rules(metrics: dict) -> list[Recommendation]:
    found: list[Recommendation] = []

    if _reversal_is_mechanical(metrics):
        rev_peak = _fe_stat(metrics, "reversal", "peak_abs")
        cruise_peak = _fe_stat(metrics, "cruise", "peak_abs")
        found.append(Recommendation(
            rule_id="reversal_transients",
            severity="observe", root_cause="mechanical",
            action=("Investigate stiction/backlash at zero speed; use an "
                    "S-curve profile instead of triangular. Do NOT soften "
                    "gains — that makes reversal spikes larger"),
            diagnosis=(f"fe.reversal.peak_abs {rev_peak:g} > "
                       f"{_REVERSAL_PEAK_RATIO:g}× cruise peak "
                       f"{cruise_peak:g} with quiet cruise and no "
                       f"oscillation — reversal transient, not a tuning "
                       f"problem"),
            expected="smaller spikes after mechanical fix or S-curve",
        ))

    asym = metrics.get("asymmetry") or {}
    if asym.get("significant"):
        found.append(Recommendation(
            rule_id="direction_asymmetry",
            severity="observe", root_cause="mechanical",
            action=("Investigate direction-dependent mechanics: friction, "
                    "backlash, or gravity load (consider compensation) — "
                    "not a gain problem"),
            diagnosis=(f"asymmetry_ratio {asym.get('asymmetry_ratio')} "
                       f"(pos {asym.get('cruise_fe_pos_dir_mean')} vs neg "
                       f"{asym.get('cruise_fe_neg_dir_mean')})"),
            expected="|FE| magnitudes match in both directions",
        ))

    return found


# ------------------------------------------------------------- mode gating
_GATED_PARAMS = {"Pn102", "Pn103", "Pn104"}


def _apply_mode_gating(recs: list[Recommendation],
                       profile: dict | None) -> list[Recommendation]:
    """Auto-tuning modes manage Pn102/103/104 internally → advise Pn101."""
    mode = _tuning_mode(profile)
    if mode not in (1, 3):
        return recs

    mode_name = "Tuning-less" if mode == 1 else "One-parameter auto-tuning"
    gated: list[Recommendation] = []
    rigidity_added = False
    for rec in recs:
        if rec.parameter not in _GATED_PARAMS:
            gated.append(rec)
            continue
        if rigidity_added:
            continue
        softer = rec.rule_id in ("loop_instability", "velocity_overshoot",
                                 "underdamped_settle",
                                 "oscillation_ambiguous_phase")
        direction = "Reduce" if softer else "Increase"
        factor = 0.85 if softer else 1.15
        gated.append(Recommendation(
            rule_id=f"{rec.rule_id}_rigidity",
            severity="action", root_cause=rec.root_cause,
            action=(f"Drive is in {mode_name} mode (Pn100.0={mode}) — "
                    f"{direction} Pn101 servo rigidity by 15%, or switch "
                    f"Pn100.0 to Manual (5) to tune Pn102/Pn103/Pn104 "
                    f"directly (drive restart required)"),
            diagnosis=rec.diagnosis,
            expected=rec.expected,
            parameter="Pn101",
            proposed=_propose(profile, "pn101", "Pn101", factor=factor),
        ))
        rigidity_added = True
    return gated


# ---------------------------------------------------------------- evaluate
def evaluate(metrics: dict, profile: dict | None = None) -> TuningReport:
    """Full offline diagnosis of one SignalMetrics result.

    profile: DriveProfile.to_dict() for the analyzed axis (or None) — used
    for concrete Pn value proposals and auto-tuning-mode gating.
    """
    profile_attached = bool(profile)
    if metrics.get("data_sufficiency") != "OK":
        reason = (metrics.get("warnings") or ["insufficient data"])[-1]
        return TuningReport(
            score=None, well_tuned=False, root_cause="insufficient-data",
            observations=[Recommendation(
                rule_id="insufficient_data", severity="info",
                root_cause="insufficient-data",
                action="Capture DPOS + DRIVE_FE (and ideally MSPEED, "
                       "DEMAND_SPEED, DRIVE_TORQUE) over at least one move",
                diagnosis=str(reason),
            )],
            profile_attached=profile_attached,
        )

    score = tuning_score(metrics)
    observations = _mechanical_rules(metrics)

    blocking = _blocking_rules(metrics, profile)
    if blocking:
        recommendations = _apply_mode_gating(blocking, profile)
        return TuningReport(
            score=score, well_tuned=False,
            root_cause=recommendations[0].root_cause,
            recommendations=recommendations[:MAX_PARAMETER_CHANGES],
            observations=observations,
            profile_attached=profile_attached,
        )

    recommendations = _velocity_rules(metrics, profile) + _fe_rules(metrics, profile)
    recommendations = _apply_mode_gating(recommendations, profile)

    # "Well tuned" means no rule found anything actionable AND the score is
    # high — a fired rule always cites a concrete defect, so it is never
    # suppressed by a good overall score.
    well_tuned = (score is not None and score >= WELL_TUNED_SCORE
                  and not recommendations)

    if recommendations:
        root_cause = recommendations[0].root_cause
    elif observations:
        root_cause = "mechanical"
    else:
        root_cause = "well-tuned"

    return TuningReport(
        score=score,
        well_tuned=well_tuned,
        root_cause=root_cause,
        recommendations=recommendations[:MAX_PARAMETER_CHANGES],
        observations=observations,
        profile_attached=profile_attached,
    )
