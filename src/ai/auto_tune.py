"""Conservative optimizer for a DX drive-closed position/velocity loop.

The optimizer is deliberately Qt- and hardware-free.  A caller owns motion,
capture and CoE I/O, while this module decides which *single* parameter to
probe and whether a repeated trial is safe to keep.  That separation keeps the
decision logic deterministic and makes rollback possible even when the UI or
hardware orchestration changes.

Version 1 tunes the active feedback and feedforward parameters in cascade
order while the drive is in manual tuning mode (Pn100.0 = 5):

    Pn102 -> Pn103 -> Pn104 -> Pn112 -> Pn114

Pn106 is a required, fixed inertia input.  Filters are intentionally excluded
until the captured response gives a reliable noise/resonance classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Iterable, Mapping, Sequence

from .drive_profile import PARAM_DEFS
from .tuning_rules import tuning_score
MANUAL_TUNING_MODE = 5
MIN_IMPROVEMENT = 0.03
MAX_GUARD_REGRESSION = 0.10
MAX_SCORE_REGRESSION = 0.5
class AutoTuneError(RuntimeError):
    """Raised when a session cannot safely be started or advanced."""


class TuneStage(str, Enum):
    VELOCITY_GAIN = "velocity_gain"
    VELOCITY_INTEGRAL = "velocity_integral"
    POSITION_GAIN = "position_gain"
    SPEED_FEEDFORWARD = "speed_feedforward"
    TORQUE_FEEDFORWARD = "torque_feedforward"
    COMPLETE = "complete"


@dataclass(frozen=True)
class StageSpec:
    stage: TuneStage
    parameter: str
    label: str
    initial_direction: int
    relative_steps: tuple[float, ...] = ()
    additive_steps: tuple[int, ...] = ()
    max_trials: int = 8


STAGES: tuple[StageSpec, ...] = (
    StageSpec(TuneStage.VELOCITY_GAIN, "pn102", "Pn102", 1,
              relative_steps=(0.10, 0.05, 0.02)),
    # A smaller Pn103 means stronger/faster velocity integral action.
    StageSpec(TuneStage.VELOCITY_INTEGRAL, "pn103", "Pn103", -1,
              relative_steps=(0.10, 0.05, 0.02)),
    StageSpec(TuneStage.POSITION_GAIN, "pn104", "Pn104", 1,
              relative_steps=(0.10, 0.05, 0.02)),
    StageSpec(TuneStage.SPEED_FEEDFORWARD, "pn112", "Pn112", 1,
              additive_steps=(10, 5, 2)),
    StageSpec(TuneStage.TORQUE_FEEDFORWARD, "pn114", "Pn114", 1,
              additive_steps=(10, 5, 2)),
)


_BOUNDS: dict[str, tuple[int, int]] = {
    entry[0]: (int(entry[4]), int(entry[5]))
    for entry in PARAM_DEFS
    if entry[4] is not None and entry[5] is not None
}


@dataclass(frozen=True)
class TrialSummary:
    """Median response metrics from repeated, identical captures."""

    repeats: int
    safe: bool
    failures: tuple[str, ...]
    score: float | None
    velocity_error_rms: float | None
    velocity_reach_error: float | None
    velocity_overshoot_pct: float | None
    settle_ms: float | None
    settle_peak: float | None
    zero_crossings: float | None
    cruise_fe_rms: float | None
    cruise_fe_mean_abs: float | None
    ramp_fe_peak: float | None
    ramp_velocity_error_rms: float | None
    saturation_pct: float


@dataclass(frozen=True)
class TuneCandidate:
    stage: TuneStage
    parameter: str
    label: str
    current: int
    proposed: int
    step: float
    direction: int
    trial_number: int


@dataclass(frozen=True)
class TrialDecision:
    candidate: TuneCandidate
    accepted: bool
    unsafe: bool
    improvement: float | None
    reason: str
    rollback_value: int | None
    summary: TrialSummary


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _nested(metrics: Mapping, *path: str) -> float | None:
    value = metrics
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _number(value)


def _median(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return median(valid) if valid else None


def _max_saturation(metrics: Mapping) -> float:
    current = metrics.get("current") or {}
    return max(
        (_number((current.get(phase) or {}).get("saturation_pct")) or 0.0
         for phase in ("accel", "decel")),
        default=0.0,
    )


def _capture_failures(metrics: Mapping) -> list[str]:
    failures: list[str] = []
    if metrics.get("data_sufficiency") != "OK":
        failures.append("insufficient capture data")

    saturation = _max_saturation(metrics)
    if saturation > 5.0:
        failures.append(f"torque/current saturation {saturation:g}%")

    oscillation = metrics.get("oscillation") or {}
    for role in ("fe", "velocity_error"):
        if (oscillation.get(role) or {}).get("has_significant_oscillation"):
            failures.append(f"significant {role.replace('_', ' ')} oscillation")

    settle = metrics.get("settle") or {}
    if settle and not settle.get("settled_within_window", True):
        failures.append("position failed to settle")
    return failures


def summarize_trials(metrics_runs: Sequence[Mapping]) -> TrialSummary:
    """Aggregate repeated captures and apply non-negotiable safety gates."""
    if not metrics_runs:
        raise AutoTuneError("At least one capture is required.")

    failures: list[str] = []
    for metrics in metrics_runs:
        for failure in _capture_failures(metrics):
            if failure not in failures:
                failures.append(failure)

    def med(*path: str) -> float | None:
        return _median(_nested(metrics, *path) for metrics in metrics_runs)

    velocity_rms = _median(
        _median((
            _nested(metrics, "velocity", "accel_err", "rms"),
            _nested(metrics, "velocity", "cruise_err", "rms"),
            _nested(metrics, "velocity", "decel_err", "rms"),
        ))
        for metrics in metrics_runs
    )
    ramp_velocity_rms = _median(
        _median((
            _nested(metrics, "velocity", "accel_err", "rms"),
            _nested(metrics, "velocity", "decel_err", "rms"),
        ))
        for metrics in metrics_runs
    )
    ramp_peak = _median(
        max(
            _nested(metrics, "fe", "accel", "peak_abs") or 0.0,
            _nested(metrics, "fe", "decel", "peak_abs") or 0.0,
        )
        for metrics in metrics_runs
    )
    score = _median(_number(tuning_score(dict(metrics)))
                    for metrics in metrics_runs)
    reach = med("velocity", "cruise_velocity_reach_ratio")

    return TrialSummary(
        repeats=len(metrics_runs),
        safe=not failures,
        failures=tuple(failures),
        score=score,
        velocity_error_rms=velocity_rms,
        velocity_reach_error=(abs(reach - 1.0) if reach is not None else None),
        velocity_overshoot_pct=med(
            "velocity", "velocity_overshoot_per_move", "max_pct"),
        settle_ms=med("settle", "time_to_band_ms"),
        settle_peak=med("settle", "fe_peak_during_settle"),
        zero_crossings=med("settle", "zero_crossings"),
        cruise_fe_rms=med("fe", "cruise", "rms"),
        cruise_fe_mean_abs=(
            abs(med("fe", "cruise", "mean"))
            if med("fe", "cruise", "mean") is not None else None
        ),
        ramp_fe_peak=ramp_peak,
        ramp_velocity_error_rms=ramp_velocity_rms,
        saturation_pct=max(_max_saturation(metrics) for metrics in metrics_runs),
    )


_OBJECTIVES: dict[TuneStage, tuple[tuple[str, float], ...]] = {
    TuneStage.VELOCITY_GAIN: (
        ("velocity_error_rms", 0.55),
        ("velocity_reach_error", 0.25),
        ("velocity_overshoot_pct", 0.20),
    ),
    TuneStage.VELOCITY_INTEGRAL: (
        ("velocity_error_rms", 0.55),
        ("velocity_reach_error", 0.25),
        ("velocity_overshoot_pct", 0.20),
    ),
    TuneStage.POSITION_GAIN: (
        ("settle_ms", 0.35),
        ("settle_peak", 0.35),
        ("zero_crossings", 0.15),
        ("cruise_fe_rms", 0.15),
    ),
    TuneStage.SPEED_FEEDFORWARD: (
        ("cruise_fe_rms", 0.70),
        ("cruise_fe_mean_abs", 0.30),
    ),
    TuneStage.TORQUE_FEEDFORWARD: (
        ("ramp_fe_peak", 0.70),
        ("ramp_velocity_error_rms", 0.30),
    ),
}


def _ratio(reference: float, candidate: float) -> float:
    floor = max(abs(reference) * 0.01, 1e-9)
    return (candidate + floor) / (reference + floor)


def compare_trial(
    stage: TuneStage,
    reference: TrialSummary,
    candidate: TrialSummary,
) -> tuple[bool, bool, float | None, str]:
    """Return ``accepted, unsafe, improvement, reason`` for one probe."""
    if not candidate.safe:
        return False, True, None, "; ".join(candidate.failures)

    if (reference.score is not None and candidate.score is not None
            and candidate.score < reference.score - MAX_SCORE_REGRESSION):
        return False, False, None, (
            f"overall score regressed {reference.score:.1f} -> "
            f"{candidate.score:.1f}"
        )

    weighted_ratio = 0.0
    used_weight = 0.0
    details: list[str] = []
    for field, weight in _OBJECTIVES[stage]:
        old = getattr(reference, field)
        new = getattr(candidate, field)
        if old is None or new is None:
            continue
        ratio = _ratio(old, new)
        weighted_ratio += weight * ratio
        used_weight += weight
        details.append(f"{field} {old:.4g}->{new:.4g}")

    if used_weight <= 0:
        return False, False, None, "required stage metrics are unavailable"

    cost_ratio = weighted_ratio / used_weight
    improvement = 1.0 - cost_ratio

    guard_fields = (
        "velocity_overshoot_pct", "settle_ms", "settle_peak",
        "cruise_fe_rms", "ramp_fe_peak",
    )
    regressions: list[str] = []
    for field in guard_fields:
        old = getattr(reference, field)
        new = getattr(candidate, field)
        if old is None or new is None:
            continue
        if _ratio(old, new) > 1.0 + MAX_GUARD_REGRESSION:
            regressions.append(f"{field} worsened {old:.4g}->{new:.4g}")
    if regressions:
        return False, False, improvement, "; ".join(regressions)

    accepted = improvement >= MIN_IMPROVEMENT
    reason = (
        f"stage objective improved {100 * improvement:.1f}%"
        if accepted else
        f"stage improvement {100 * improvement:.1f}% is below "
        f"the {100 * MIN_IMPROVEMENT:.0f}% acceptance threshold"
    )
    if details:
        reason += " (" + ", ".join(details) + ")"
    return accepted, False, improvement, reason


def validate_manual_drive_profile(profile: Mapping) -> None:
    """Reject profiles that cannot support drive-position manual tuning."""
    if profile.get("drive_type") not in ("DX3", "DX4"):
        raise AutoTuneError("Select a DX3 or DX4 drive profile.")
    if profile.get("pn100_tuning_mode") != MANUAL_TUNING_MODE:
        raise AutoTuneError("Pn100.0 must be Manual tuning mode (5).")
    pn106 = profile.get("pn106")
    if pn106 is None or int(pn106) <= 0:
        raise AutoTuneError("Pn106 load inertia must be measured and greater than zero.")
    missing = [spec.label for spec in STAGES if profile.get(spec.parameter) is None]
    if missing:
        raise AutoTuneError("Missing drive parameters: " + ", ".join(missing))


class ManualDrivePositionOptimizer:
    """Bounded one-parameter-at-a-time optimizer for supervised sessions."""

    def __init__(self, profile: Mapping):
        validate_manual_drive_profile(profile)
        self.original_profile = dict(profile)
        self.accepted_profile = dict(profile)
        self.reference: TrialSummary | None = None
        self._stage_index = 0
        self._step_index = 0
        self._direction = STAGES[0].initial_direction
        self._tried_directions: set[int] = set()
        self._stage_had_accept = False
        self._stage_trials = 0
        self._pending: TuneCandidate | None = None
        self.decisions: list[TrialDecision] = []

    @property
    def stage(self) -> TuneStage:
        if self._stage_index >= len(STAGES):
            return TuneStage.COMPLETE
        return STAGES[self._stage_index].stage

    @property
    def pending_candidate(self) -> TuneCandidate | None:
        return self._pending

    @property
    def complete(self) -> bool:
        return self.stage is TuneStage.COMPLETE

    def set_baseline(self, metrics_runs: Sequence[Mapping]) -> TrialSummary:
        if self.reference is not None:
            raise AutoTuneError("A baseline has already been recorded.")
        summary = summarize_trials(metrics_runs)
        if not summary.safe:
            raise AutoTuneError(
                "Baseline is unsafe: " + "; ".join(summary.failures)
            )
        self.reference = summary
        return summary

    def _spec(self) -> StageSpec:
        if self.complete:
            raise AutoTuneError("Auto-tuning is complete.")
        return STAGES[self._stage_index]

    def _steps(self, spec: StageSpec) -> Sequence[float | int]:
        return spec.relative_steps or spec.additive_steps

    def _candidate_value(self, spec: StageSpec, current: int) -> tuple[int, float]:
        steps = self._steps(spec)
        step = steps[self._step_index]
        if spec.relative_steps:
            proposed = round(current * (1.0 + self._direction * float(step)))
        else:
            proposed = current + self._direction * int(step)
        lo, hi = _BOUNDS[spec.parameter]
        return max(lo, min(hi, proposed)), float(step)

    def next_candidate(self) -> TuneCandidate | None:
        if self.reference is None:
            raise AutoTuneError("Record a safe baseline before requesting a candidate.")
        if self._pending is not None:
            return self._pending

        while not self.complete:
            spec = self._spec()
            if self._stage_trials >= spec.max_trials:
                self._advance_stage()
                continue
            proposed, step = self._candidate_value(
                spec, int(self.accepted_profile[spec.parameter]))
            current = int(self.accepted_profile[spec.parameter])
            if proposed == current:
                if not self._advance_search_after_rejection():
                    self._advance_stage()
                continue
            self._stage_trials += 1
            self._pending = TuneCandidate(
                stage=spec.stage,
                parameter=spec.parameter,
                label=spec.label,
                current=current,
                proposed=proposed,
                step=step,
                direction=self._direction,
                trial_number=self._stage_trials,
            )
            return self._pending
        return None

    def assess(self, metrics_runs: Sequence[Mapping]) -> TrialDecision:
        if self.reference is None or self._pending is None:
            raise AutoTuneError("There is no pending candidate to assess.")
        candidate = self._pending
        summary = summarize_trials(metrics_runs)
        accepted, unsafe, improvement, reason = compare_trial(
            candidate.stage, self.reference, summary)
        decision = TrialDecision(
            candidate=candidate,
            accepted=accepted,
            unsafe=unsafe,
            improvement=improvement,
            reason=reason,
            rollback_value=None if accepted else candidate.current,
            summary=summary,
        )
        self.decisions.append(decision)
        self._pending = None
        self._tried_directions.add(candidate.direction)

        if accepted:
            self.accepted_profile[candidate.parameter] = candidate.proposed
            self.reference = summary
            self._stage_had_accept = True
        elif not self._advance_search_after_rejection():
            self._advance_stage()
        return decision

    def _advance_search_after_rejection(self) -> bool:
        spec = self._spec()
        steps = self._steps(spec)
        if self._stage_had_accept:
            self._step_index += 1
            self._tried_directions.clear()
        elif -self._direction not in self._tried_directions:
            self._direction *= -1
        else:
            self._step_index += 1
            self._direction = spec.initial_direction
            self._tried_directions.clear()
        return self._step_index < len(steps)

    def _advance_stage(self) -> None:
        self._stage_index += 1
        self._step_index = 0
        self._tried_directions.clear()
        self._stage_had_accept = False
        self._stage_trials = 0
        self._pending = None
        if not self.complete:
            self._direction = self._spec().initial_direction

    def changed_parameters(self) -> dict[str, tuple[int, int]]:
        """Accepted changes as ``parameter -> (original, accepted)``."""
        changed: dict[str, tuple[int, int]] = {}
        for spec in STAGES:
            old = int(self.original_profile[spec.parameter])
            new = int(self.accepted_profile[spec.parameter])
            if old != new:
                changed[spec.parameter] = (old, new)
        return changed
