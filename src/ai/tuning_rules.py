"""Offline rule-based tuning recommendations — Qt-free.

The same diagnostic table the AI panel's system prompt encodes, as plain
Python over the SignalMetrics result dict, so the Servo Loop Analyser can
say *"increase Pn112 by ≤10 points — cruise FE slope ∝ velocity"* without
an API key.

Structure follows the mandated inside-out decision order:

  1. current loop   — saturation / resonance / instability are BLOCKING:
                      when one fires, no other gain advice is issued
  2. velocity loop  — reach ratio, overshoot
  3. position / FE  — feedforward (VFF then AFF), ringing, residual offset
  m. mechanical     — reversal spikes, asymmetry (observations, not knobs)

Step limits per iteration: gains ±15-20 %, feedforward ±10 points, at most
three parameter changes. Pn100 auto-tuning modes gate direct gain advice
to Pn101 rigidity. Operation mode is assumed CSP (controller closes the
position loop) unless stated — recommendations name both the controller
parameter and the drive equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .drive_profile import MIN_NOTCH_FILTER_HZ, PARAM_DEFS

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
# When two FE spectral lines are almost equally strong, treating the first
# list entry as *the* plant mode is not a safe basis for a notch/gain change.
_COMPARABLE_MODE_RATIO = 0.90


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


def _phase_band(
        metrics: dict,
) -> tuple[float | None, float | None, str, bool]:
    """Phase result plus whether it can safely drive a tuning rule.

    A phase claim is actionable only when it has real multi-window
    coherence and refers to the same frequency as the dominant FE mode.
    Single-window magnitude proxies remain useful diagnostics but must not
    propose gain changes.
    """
    cvp = (metrics.get("oscillation") or {}).get("current_vs_velocity_phase")
    if not cvp:
        return None, None, "", False
    coherence = cvp.get("coherence")
    note = f", coherence {coherence}" if coherence is not None else ""
    phase = cvp.get("phase_deg")
    phase_freq = cvp.get("dominant_freq_hz")
    fe_freq = ((metrics.get("oscillation") or {}).get("fe") or {}).get(
        "dominant_hz")
    resolution = float(cvp.get("resolution_hz") or 0.0)
    tolerance = max(2.0, 1.5 * resolution,
                    0.05 * float(fe_freq or 0.0))
    same_frequency = (phase_freq is not None and fe_freq is not None
                      and abs(float(phase_freq) - float(fe_freq)) <= tolerance)
    explicitly_reliable = cvp.get("classification_reliable")
    coherent = (bool(explicitly_reliable) if explicitly_reliable is not None
                else coherence is not None)
    reliable = bool(phase is not None and coherent and same_frequency)
    if phase_freq is not None and fe_freq is not None and not same_frequency:
        note += f", frequency mismatch {phase_freq:g} vs FE {fe_freq:g} Hz"
    elif not coherent and phase is not None:
        note += ", single-window phase only"
    return phase, phase_freq, note, reliable


def _notch_frequency(phase_freq: float | None,
                     oscillation_freq: float | None) -> float | None:
    """Return a detected frequency only when the drive can notch it."""
    frequency = phase_freq if phase_freq is not None else oscillation_freq
    if frequency is None or frequency < MIN_NOTCH_FILTER_HZ:
        return None
    return frequency


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


def _significant_modes(oscillation: dict) -> list[dict]:
    """Strong FE peaks that are comparable with the reported dominant line."""
    peaks = [
        peak for peak in (oscillation.get("peaks") or [])
        if peak.get("freq_hz") is not None and peak.get("amplitude") is not None
    ]
    if len(peaks) < 2:
        return peaks[:1]
    strongest = max(float(peak["amplitude"]) for peak in peaks)
    if strongest <= 0:
        return peaks[:1]
    return [
        peak for peak in peaks
        if float(peak["amplitude"]) >= _COMPARABLE_MODE_RATIO * strongest
    ]


def _mode_evidence(oscillation: dict) -> str:
    """Compact frequency/amplitude evidence for recommendation text."""
    modes = _significant_modes(oscillation)
    if modes:
        return ", ".join(
            f"{float(peak['freq_hz']):g} Hz @ {float(peak['amplitude']):.4g} u"
            for peak in modes
        )
    frequency = oscillation.get("dominant_hz")
    return f"{frequency:g} Hz" if frequency is not None else "unknown frequency"


def _repeat_move_protocol(*, include_speed_sweep: bool = False) -> str:
    """A reproducible capture, replacing vague requests for 'more data'."""
    text = (
        "Capture DPOS, DRIVE_FE, MSPEED, and DRIVE_CURRENT for 3 identical "
        "moves with at least 0.8 s of steady cruise in every move"
    )
    if include_speed_sweep:
        text += (
            "; repeat the capture at 50%, 100%, and 150% speed while "
            "keeping accel, load, and gains unchanged"
        )
    return text


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
        phase_deg, phase_freq, confidence, phase_reliable = _phase_band(metrics)
        freq = osc_fe.get("dominant_hz")
        evidence = _mode_evidence(osc_fe)
        comparable_modes = _significant_modes(osc_fe)
        channels = metrics.get("channels_detected") or {}
        have_signals = bool(channels.get("current")) and bool(
            channels.get("measured_vel"))

        if len(comparable_modes) > 1:
            found.append(Recommendation(
                rule_id="oscillation_multiple_modes",
                severity="action", root_cause="position",
                action=(
                    "Keep gains and notch settings unchanged: more than one "
                    "FE line is comparably strong, so there is no unique mode to "
                    "tune. " + _repeat_move_protocol(include_speed_sweep=True)
                    + ". A frequency that scales with speed indicates periodic "
                    "mechanical forcing (belt/screw pitch, cogging, or runout); "
                    "a fixed frequency is a structural or control-loop mode."
                ),
                diagnosis=(
                    f"oscillation.fe.peaks contains comparable modes: {evidence} "
                    f"(within {100 * (1 - _COMPARABLE_MODE_RATIO):.0f}% of the "
                    "strongest amplitude); a single dominant-mode phase verdict "
                    "would be ambiguous"
                ),
                expected=(
                    "one repeatable dominant line, then a frequency-matched "
                    "coherent phase result from at least 3 moves"
                ),
            ))
        elif not have_signals:
            found.append(Recommendation(
                rule_id="oscillation_unresolved",
                severity="action", root_cause="position",
                action=(
                    "Keep gains unchanged. "
                    + _repeat_move_protocol(include_speed_sweep=True)
                    + ". Use coherent current-versus-velocity phase at the FE "
                    "frequency: about +90° indicates mechanical resonance; "
                    "about 0° indicates loop instability."
                ),
                diagnosis=(f"oscillation.fe reports {evidence}, but no "
                           "DRIVE_CURRENT/MSPEED pair was captured for the "
                           "phase test"),
                expected=("classification_reliable=true at the same FE "
                          "frequency, plus fixed-versus-speed-scaled behaviour"),
            ))
        elif phase_deg is None or not phase_reliable:
            phase_info = ((metrics.get("oscillation") or {}).get(
                "current_vs_velocity_phase") or {})
            method = phase_info.get("method") or "unavailable"
            averages = phase_info.get("n_averages")
            found.append(Recommendation(
                rule_id="oscillation_no_coherent_phase",
                severity="action", root_cause="position",
                action=(
                    "Keep gains unchanged. "
                    + _repeat_move_protocol(include_speed_sweep=True)
                    + ". Accept a diagnosis only when "
                    "current_vs_velocity_phase.classification_reliable=true "
                    "at the same FE line. About +90° means resonance; about "
                    "0° means loop instability."
                ),
                diagnosis=(f"oscillation.fe reports {evidence}, but its "
                           f"frequency-matched current/velocity phase is not "
                           f"reliable ({method}, n_averages={averages}"
                           f"{confidence})"),
                expected=("at least 3 usable spectral averages, a coherent "
                          "same-frequency phase verdict, and a fixed or "
                          "speed-scaled frequency trend"),
            ))
        elif 60 < phase_deg < 120:
            notch_freq = _notch_frequency(phase_freq, freq)
            if notch_freq is not None:
                action = (
                    "Save the current settings. In the drive commissioning "
                    f"tool, start with a shallow notch centred at ~{notch_freq:g} "
                    "Hz; change no gains, then repeat the identical 3-move "
                    "capture. Deepen the notch only if that same line remains."
                )
                expected = (
                    f"FE amplitude at ~{notch_freq:g} Hz falls by at least "
                    "50%, without worse current RMS, velocity overshoot, or "
                    "settling"
                )
            else:
                mode_freq = phase_freq or freq
                action = (
                    f"Do not set a notch: ~{mode_freq:g} Hz is below the "
                    f"drive's {MIN_NOTCH_FILTER_HZ:g} Hz limit. With power "
                    "isolated, inspect motor/load mounting, coupling or belt "
                    "compliance, backlash, and load looseness; then compare an "
                    "unloaded 3-move capture with the loaded capture. Do not "
                    "increase position gain."
                )
                expected = (
                    "the fixed-frequency line is materially lower after the "
                    "mechanical correction, with no invalid notch setting"
                )
            found.append(Recommendation(
                rule_id="mechanical_resonance",
                severity="action", root_cause="mechanical",
                action=action,
                diagnosis=(f"oscillation.fe reports {evidence} and "
                           f"current_vs_velocity_phase={phase_deg:.0f}° at "
                           f"{phase_freq:g} Hz{confidence}; current leads "
                           "velocity in the mechanical-resonance band"),
                expected=expected,
            ))
        elif -30 < phase_deg < 30:
            proposed = _propose(profile, "pn104", "Pn104", factor=0.9)
            found.append(Recommendation(
                rule_id="loop_instability",
                severity="action", root_cause="position",
                action=(
                    "Save the current settings and change exactly one active "
                    "outer-loop gain: in CSP reduce controller P_GAIN by 10%; "
                    "only when the drive closes the position loop reduce Pn104 "
                    "by 10%. Leave Pn103 unchanged because it is velocity-loop "
                    "integral time. Repeat the identical 3-move capture; restore "
                    "the setting if the FE line does not fall."
                ),
                diagnosis=(f"oscillation.fe reports {evidence} and "
                           f"current_vs_velocity_phase={phase_deg:.0f}° at "
                           f"{phase_freq:g} Hz{confidence}; current and velocity "
                           "are in the loop-instability band"),
                expected=("FE amplitude at that frequency falls by at least "
                          "30%, with no loss of cruise tracking; otherwise "
                          "restore the gain and investigate mechanics"),
                parameter="Pn104", proposed=proposed,
            ))
        else:
            found.append(Recommendation(
                rule_id="oscillation_ambiguous_phase",
                severity="action", root_cause="position",
                action=(
                    f"Keep gains unchanged: phase {phase_deg:.0f}° is outside "
                    "the validated ~0° instability and ~+90° resonance bands. "
                    + _repeat_move_protocol(include_speed_sweep=True)
                    + ". A speed-scaled frequency points to periodic mechanical "
                    "forcing; a fixed line must be re-tested for coherent phase."
                ),
                diagnosis=(f"oscillation.fe reports {evidence}, with matched "
                           f"current_vs_velocity_phase={phase_deg:.0f}° at "
                           f"{phase_freq:g} Hz{confidence}; root cause remains "
                           "unclassified"),
                expected=("a repeatable fixed or speed-scaled frequency trend "
                          "and a phase inside a validated band before any change"),
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
    slope = fit.get("slope")
    vff_needed = bool(fit.get("proportional_to_velocity")
                      and slope is not None and slope > 0)
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
        crossings = int(settle.get("zero_crossings") or 0)
        damping = settle.get("damping_ratio")
        severe = crossings >= 8 or (damping is not None and damping < 0.15)
        reduction_pct = 15 if severe else 10
        factor = 1.0 - reduction_pct / 100.0
        band = float(settle.get("band") or 0.0)
        settle_peak = float(settle.get("fe_peak_during_settle") or 0.0)
        peak_ratio = settle_peak / band if band > 0 else None
        natural_freq = settle.get("natural_freq_hz")
        frequency_note = (f", natural_freq_hz={natural_freq}"
                          if natural_freq is not None else "")
        ratio_note = (f", peak/band={peak_ratio:.1f}×"
                      if peak_ratio is not None else "")
        found.append(Recommendation(
            rule_id="underdamped_settle",
            severity="action", root_cause="position",
            action=(
                "Save the current settings and verify which device closes the "
                f"position loop. Change one parameter only: reduce controller "
                f"P_GAIN by {reduction_pct}% in CSP, or reduce Pn104 by "
                f"{reduction_pct}% only for a drive-internal position loop. "
                "Do not change Pn103; it is velocity-loop integral time and "
                "there is no drive position-loop integral action. Repeat 3 "
                "identical moves and restore the original gain if ringing does "
                "not improve."
            ),
            diagnosis=(f"settle.ringing=true: zero_crossings={crossings}, "
                       f"damping_ratio={damping}, "
                       f"fe_peak_during_settle={settle_peak:g}, band={band:g}"
                       f"{ratio_note}{frequency_note}"),
            expected=("zero_crossings ≤ 3, shorter time_to_band_ms, and at "
                      "least 30% lower fe_peak_during_settle without worse "
                      "cruise FE; otherwise restore the gain and inspect "
                      "coupling/backlash/compliance"),
            parameter="Pn104",
            proposed=_propose(profile, "pn104", "Pn104", factor=factor),
        ))

    if settle.get("steady_state_offset_nonzero"):
        found.append(Recommendation(
            rule_id="position_offset_investigation",
            severity="action", root_cause="position",
            action=("Do not change Pn103 or position gain from this metric "
                    "alone. Set an application tolerance, extend the dwell, "
                    "then check static load/friction, brake release, and "
                    "command-versus-feedback position bias"),
            diagnosis=(f"settle.steady_state_offset_nonzero=true "
                       f"(fe_steady_state {settle.get('fe_steady_state')} "
                       f"outside ±{settle.get('band')}); the drive position "
                       "loop has no integral term, and Pn103 belongs to the "
                       "velocity loop"),
            expected="late-window FE inside a user-defined application tolerance",
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
                                 "underdamped_settle")
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
