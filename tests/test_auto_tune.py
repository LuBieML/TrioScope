"""Manual drive-position auto-tuner decision and CoE safety tests."""

import pytest

from src.ai import coe_io
from src.ai.auto_tune import (
    AutoTuneError,
    ManualDrivePositionOptimizer,
    TuneStage,
    compare_trial,
    summarize_trials,
    validate_manual_drive_profile,
)


def _profile(**changes):
    profile = {
        "drive_type": "DX4",
        "pn100_tuning_mode": 5,
        "pn106": 250,
        "pn102": 500,
        "pn103": 125,
        "pn104": 40,
        "pn112": 20,
        "pn114": 10,
    }
    profile.update(changes)
    return profile


def _metrics(
    *,
    velocity_rms=10.0,
    reach=0.90,
    overshoot=5.0,
    settle_ms=100.0,
    settle_peak=10.0,
    crossings=5,
    cruise_rms=3.0,
    cruise_mean=2.0,
    ramp_peak=8.0,
    saturation=0.0,
    oscillation=False,
    sufficient=True,
):
    return {
        "data_sufficiency": "OK" if sufficient else "INSUFFICIENT",
        "velocity": {
            "accel_err": {"rms": velocity_rms},
            "cruise_err": {"rms": velocity_rms},
            "decel_err": {"rms": velocity_rms},
            "cruise_velocity_reach_ratio": reach,
            "velocity_overshoot_per_move": {"max_pct": overshoot},
        },
        "settle": {
            "settled_within_window": True,
            "time_to_band_ms": settle_ms,
            "fe_peak_during_settle": settle_peak,
            "zero_crossings": crossings,
            "ringing": crossings > 3,
            "steady_state_offset_nonzero": False,
        },
        "fe": {
            "cruise": {"rms": cruise_rms, "mean": cruise_mean,
                       "peak_abs": cruise_rms * 2},
            "accel": {"peak_abs": ramp_peak},
            "decel": {"peak_abs": ramp_peak * 0.9},
        },
        "current": {
            "accel": {"saturation_pct": saturation},
            "decel": {"saturation_pct": saturation},
        },
        "oscillation": {
            "fe": {"has_significant_oscillation": oscillation},
            "velocity_error": {"has_significant_oscillation": False},
        },
        "asymmetry": {"significant": False},
    }


def test_manual_drive_profile_requires_mode_inertia_and_all_stage_values():
    validate_manual_drive_profile(_profile())

    with pytest.raises(AutoTuneError, match="Manual"):
        validate_manual_drive_profile(_profile(pn100_tuning_mode=3))
    with pytest.raises(AutoTuneError, match="Pn106"):
        validate_manual_drive_profile(_profile(pn106=0))
    with pytest.raises(AutoTuneError, match="Pn104"):
        validate_manual_drive_profile(_profile(pn104=None))


def test_trial_summary_uses_medians_and_safety_gates():
    summary = summarize_trials([
        _metrics(velocity_rms=12.0),
        _metrics(velocity_rms=8.0),
        _metrics(velocity_rms=10.0),
    ])
    assert summary.repeats == 3
    assert summary.velocity_error_rms == pytest.approx(10.0)
    assert summary.safe

    unsafe = summarize_trials([_metrics(saturation=8.0)])
    assert not unsafe.safe
    assert "saturation" in unsafe.failures[0]


def test_optimizer_starts_inside_out_with_ten_percent_pn102_probe():
    optimizer = ManualDrivePositionOptimizer(_profile())
    optimizer.set_baseline([_metrics()] * 3)

    candidate = optimizer.next_candidate()

    assert candidate is not None
    assert candidate.stage is TuneStage.VELOCITY_GAIN
    assert candidate.parameter == "pn102"
    assert candidate.current == 500
    assert candidate.proposed == 550


def test_improved_candidate_is_accepted_and_becomes_new_reference():
    optimizer = ManualDrivePositionOptimizer(_profile())
    optimizer.set_baseline([_metrics()] * 3)
    optimizer.next_candidate()

    improved = _metrics(velocity_rms=7.5, reach=0.96, overshoot=4.0)
    decision = optimizer.assess([improved] * 3)

    assert decision.accepted
    assert decision.rollback_value is None
    assert optimizer.accepted_profile["pn102"] == 550
    next_candidate = optimizer.next_candidate()
    assert next_candidate is not None
    assert next_candidate.current == 550
    assert next_candidate.proposed == 605


def test_unhelpful_candidate_is_rejected_then_opposite_direction_is_tried():
    optimizer = ManualDrivePositionOptimizer(_profile())
    optimizer.set_baseline([_metrics()] * 3)
    optimizer.next_candidate()

    decision = optimizer.assess([_metrics()] * 3)

    assert not decision.accepted
    assert decision.rollback_value == 500
    opposite = optimizer.next_candidate()
    assert opposite is not None
    assert opposite.current == 500
    assert opposite.proposed == 450


def test_unsafe_candidate_demands_rollback_even_if_other_metrics_improve():
    optimizer = ManualDrivePositionOptimizer(_profile())
    reference = optimizer.set_baseline([_metrics()] * 3)
    candidate = optimizer.next_candidate()
    assert candidate is not None
    unsafe = summarize_trials([
        _metrics(velocity_rms=5.0, reach=1.0, saturation=9.0)
    ] * 3)

    accepted, is_unsafe, improvement, reason = compare_trial(
        TuneStage.VELOCITY_GAIN, reference, unsafe)

    assert not accepted
    assert is_unsafe
    assert improvement is None
    assert "saturation" in reason


def test_verified_single_pn_write_reads_old_value_and_confirms_new(monkeypatch):
    values = iter([500, 550])
    writes = []
    monkeypatch.setattr(
        coe_io, "read_single_pn",
        lambda *_args, **_kwargs: next(values),
    )
    monkeypatch.setattr(
        coe_io, "write_single_pn",
        lambda _connection, axis, attr, value: writes.append(
            (axis, attr, value)),
    )

    result = coe_io.write_single_pn_verified(
        object(), axis=2, pn_attr="pn102", value=550,
    )

    assert result.previous == 500
    assert result.readback == 550
    assert writes == [(2, "pn102", 550)]


def test_verified_single_pn_write_rolls_back_readback_mismatch(monkeypatch):
    # old value, mismatched readback, successful rollback readback
    values = iter([500, 540, 500])
    writes = []
    monkeypatch.setattr(
        coe_io, "read_single_pn",
        lambda *_args, **_kwargs: next(values),
    )
    monkeypatch.setattr(
        coe_io, "write_single_pn",
        lambda _connection, axis, attr, value: writes.append(
            (axis, attr, value)),
    )

    with pytest.raises(coe_io.PnWriteVerificationError) as raised:
        coe_io.write_single_pn_verified(
            object(), axis=2, pn_attr="pn102", value=550,
        )

    assert raised.value.rollback_succeeded
    assert writes == [(2, "pn102", 550), (2, "pn102", 500)]


def test_verified_single_pn_write_rejects_bounds_before_hardware(monkeypatch):
    monkeypatch.setattr(
        coe_io, "read_single_pn",
        lambda *_args, **_kwargs: pytest.fail("hardware should not be read"),
    )
    with pytest.raises(ValueError, match="outside"):
        coe_io.write_single_pn_verified(
            object(), axis=0, pn_attr="pn112", value=101,
        )
