import math

import pytest

from src.ai.inertia_estimator import (
    ACCEL_MINUS_STEADY,
    ACCEL_VS_DECEL,
    CURRENT_AMPS,
    RAW_CURRENT,
    RAW_TORQUE,
    acceleration_to_motor_rev_s2,
    estimate_inertia,
    motor_inertia_units_to_kgm2,
)


def test_raw_torque_estimates_load_inertia_and_pn106():
    result = estimate_inertia(
        acceleration_average=250.0,
        steady_average=50.0,
        motor_acceleration_rpm_s=600.0,
        rated_torque_nm=2.0,
        motor_inertia_kgm2=0.001,
        signal_mode=RAW_TORQUE,
    )

    assert result.signal_delta == 200.0
    assert result.torque_per_motor_nm == pytest.approx(0.4)
    assert result.angular_acceleration_rad_s2 == pytest.approx(20.0 * math.pi)
    assert result.total_inertia_kgm2 == pytest.approx(0.4 / (20.0 * math.pi))
    assert result.load_inertia_kgm2 == pytest.approx(
        result.total_inertia_kgm2 - 0.001
    )
    assert result.pn106_percent == pytest.approx(
        result.load_inertia_kgm2 / 0.001 * 100.0
    )


def test_dx_motor_table_inertia_units_convert_to_si():
    assert motor_inertia_units_to_kgm2(230) == pytest.approx(2.30e-6)


def test_acceleration_scales_to_revolutions_with_encoder_resolution():
    assert acceleration_to_motor_rev_s2(
        acceleration_units_s2=500.0,
        axis_units_counts=400.0,
        encoder_counts_per_rev=4000.0,
    ) == pytest.approx(50.0)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf])
def test_acceleration_conversion_rejects_invalid_encoder_resolution(value):
    with pytest.raises(ValueError):
        acceleration_to_motor_rev_s2(500.0, 400.0, value)


def test_raw_current_reports_acceleration_current_and_uses_motor_ratio():
    result = estimate_inertia(
        acceleration_average=450.0,
        steady_average=300.0,
        motor_acceleration_rpm_s=1000.0,
        rated_torque_nm=3.0,
        rated_current_a=6.0,
        motor_inertia_kgm2=0.0002,
        signal_mode=RAW_CURRENT,
    )

    assert result.acceleration_current_a == pytest.approx(0.9)
    assert result.torque_per_motor_nm == pytest.approx(0.45)


def test_recommended_method_uses_accel_and_decel_and_reports_symmetry():
    result = estimate_inertia(
        acceleration_average=220.0,
        steady_average=60.0,
        deceleration_average=-80.0,
        motor_acceleration_rpm_s=500.0,
        rated_torque_nm=2.5,
        motor_inertia_kgm2=0.0003,
        signal_mode=RAW_TORQUE,
        method=ACCEL_VS_DECEL,
    )

    assert result.signal_delta == pytest.approx(150.0)
    assert result.symmetry_error_percent == pytest.approx(20.0 / 150.0 * 100.0)


def test_identical_gantry_motors_sum_torque_and_rotor_inertia():
    one = estimate_inertia(
        acceleration_average=200.0,
        steady_average=0.0,
        motor_acceleration_rpm_s=600.0,
        rated_torque_nm=2.0,
        motor_inertia_kgm2=0.001,
        motor_count=1,
        signal_mode=RAW_TORQUE,
    )
    two = estimate_inertia(
        acceleration_average=200.0,
        steady_average=0.0,
        motor_acceleration_rpm_s=600.0,
        rated_torque_nm=2.0,
        motor_inertia_kgm2=0.001,
        motor_count=2,
        signal_mode=RAW_TORQUE,
    )

    assert two.combined_torque_nm == pytest.approx(2 * one.combined_torque_nm)
    assert two.total_inertia_kgm2 == pytest.approx(2 * one.total_inertia_kgm2)
    assert two.load_inertia_kgm2 == pytest.approx(2 * one.load_inertia_kgm2)
    assert two.pn106_percent == pytest.approx(one.pn106_percent)


def test_current_amps_requires_rated_current():
    with pytest.raises(ValueError, match="Rated current"):
        estimate_inertia(
            acceleration_average=10.0,
            steady_average=2.0,
            motor_acceleration_rpm_s=100.0,
            rated_torque_nm=1.0,
            motor_inertia_kgm2=0.001,
            signal_mode=CURRENT_AMPS,
            method=ACCEL_MINUS_STEADY,
        )


def test_raw_current_can_omit_rated_current_when_only_torque_is_needed():
    result = estimate_inertia(
        acceleration_average=10.0,
        steady_average=2.0,
        motor_acceleration_rpm_s=100.0,
        rated_torque_nm=1.0,
        motor_inertia_kgm2=0.001,
        signal_mode=RAW_CURRENT,
        method=ACCEL_MINUS_STEADY,
    )

    assert result.acceleration_current_a is None
    assert result.torque_per_motor_nm == pytest.approx(0.008)
