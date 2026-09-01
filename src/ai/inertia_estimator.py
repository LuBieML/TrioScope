"""Qt-free motor-side inertia estimation from captured torque/current."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


RAW_TORQUE = "raw_torque"
RAW_CURRENT = "raw_current"
CURRENT_AMPS = "current_amps"
TORQUE_NM = "torque_nm"

SIGNAL_MODES = {RAW_TORQUE, RAW_CURRENT, CURRENT_AMPS, TORQUE_NM}

# DX motor tables express rotor inertia as an integer/fixed-point count where
# one displayed unit equals 1e-8 kg·m² (for example 230 -> 2.30e-6 kg·m²).
MOTOR_INERTIA_UNIT_KGM2 = 1e-8
DX_ENCODER_COUNTS_PER_REV = 2**21


def motor_inertia_units_to_kgm2(value: float) -> float:
    """Convert the DX manual's 1e-8 kg·m² inertia value to SI units."""
    return float(value) * MOTOR_INERTIA_UNIT_KGM2


def acceleration_to_motor_rev_s2(
    acceleration_units_s2: float,
    axis_units_counts: float,
    encoder_counts_per_rev: float = DX_ENCODER_COUNTS_PER_REV,
) -> float:
    """Scale controller acceleration to motor revolutions/s².

    ``axis_units_counts`` is the controller UNITS value in counts/user-unit.
    It is independent of the motor encoder resolution in counts/revolution.
    """
    values = {
        "Acceleration": acceleration_units_s2,
        "Axis UNITS": axis_units_counts,
        "Encoder counts/rev": encoder_counts_per_rev,
    }
    for label, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")
        if float(value) <= 0:
            raise ValueError(f"{label} must be greater than zero.")
    return (
        float(acceleration_units_s2)
        * float(axis_units_counts)
        / float(encoder_counts_per_rev)
    )


@dataclass(frozen=True)
class InertiaEstimate:
    """Calculated inertia, reflected to the coupled motor shafts."""

    signal_delta: float
    acceleration_current_a: Optional[float]
    torque_per_motor_nm: float
    combined_torque_nm: float
    angular_acceleration_rad_s2: float
    total_inertia_kgm2: float
    load_inertia_kgm2: float
    load_inertia_per_motor_kgm2: float
    load_motor_ratio: float
    pn106_value: float
    symmetry_error_percent: Optional[float]


def estimate_inertia(
    *,
    acceleration_average: float,
    steady_average: float,
    deceleration_average: float,
    motor_acceleration_rpm_s: float,
    rated_torque_nm: float,
    motor_inertia_kgm2: float,
    motor_count: int = 1,
    signal_mode: str = RAW_TORQUE,
    rated_current_a: Optional[float] = None,
) -> InertiaEstimate:
    """Estimate load inertia from acceleration/deceleration phase averages.

    Raw DX values use 0.1% of rated torque/current. For multiple identical
    gantry motors the supplied measurement is treated as the representative
    value for each equally-loaded motor. The steady-speed average is used only
    to assess acceleration/deceleration symmetry.
    """

    if signal_mode not in SIGNAL_MODES:
        raise ValueError(f"Unsupported signal mode: {signal_mode}")
    if isinstance(motor_count, bool) or not isinstance(motor_count, int):
        raise ValueError("Motor count must be a whole number.")
    if motor_count < 1:
        raise ValueError("Motor count must be at least one.")

    required_values = {
        "Acceleration average": acceleration_average,
        "Steady average": steady_average,
        "Deceleration average": deceleration_average,
        "Motor-speed slope": motor_acceleration_rpm_s,
        "Rated torque": rated_torque_nm,
        "Motor inertia": motor_inertia_kgm2,
    }
    for label, value in required_values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite.")

    if motor_acceleration_rpm_s <= 0:
        raise ValueError("Motor-speed slope must be greater than zero.")
    if rated_torque_nm <= 0:
        raise ValueError("Rated torque must be greater than zero.")
    if motor_inertia_kgm2 <= 0:
        raise ValueError("Motor inertia must be greater than zero.")

    if signal_mode == CURRENT_AMPS and (
        rated_current_a is None or rated_current_a <= 0
    ):
        raise ValueError("Rated current must be greater than zero for a current signal.")
    if rated_current_a is not None and not math.isfinite(float(rated_current_a)):
        raise ValueError("Rated current must be finite.")
    if rated_current_a is not None and rated_current_a < 0:
        raise ValueError("Rated current cannot be negative.")

    signal_delta = (acceleration_average - deceleration_average) / 2.0
    accel_component = abs(acceleration_average - steady_average)
    decel_component = abs(steady_average - deceleration_average)
    component_mean = (accel_component + decel_component) / 2.0
    symmetry_error = None
    if component_mean > 0:
        symmetry_error = (
            abs(accel_component - decel_component) / component_mean * 100.0
        )

    magnitude = abs(signal_delta)
    acceleration_current_a: Optional[float] = None
    if signal_mode == RAW_TORQUE:
        torque_per_motor = magnitude / 1000.0 * rated_torque_nm
    elif signal_mode == RAW_CURRENT:
        current_fraction = magnitude / 1000.0
        if rated_current_a is not None and rated_current_a > 0:
            acceleration_current_a = current_fraction * rated_current_a
        torque_per_motor = current_fraction * rated_torque_nm
    elif signal_mode == CURRENT_AMPS:
        assert rated_current_a is not None
        acceleration_current_a = magnitude
        torque_per_motor = magnitude * rated_torque_nm / rated_current_a
    else:  # TORQUE_NM
        torque_per_motor = magnitude

    angular_acceleration = motor_acceleration_rpm_s * 2.0 * math.pi / 60.0
    combined_torque = torque_per_motor * motor_count
    total_inertia = combined_torque / angular_acceleration
    combined_motor_inertia = motor_inertia_kgm2 * motor_count
    load_inertia = total_inertia - combined_motor_inertia
    load_per_motor = load_inertia / motor_count
    load_motor_ratio = load_per_motor / motor_inertia_kgm2
    pn106 = load_motor_ratio * 100.0

    return InertiaEstimate(
        signal_delta=signal_delta,
        acceleration_current_a=acceleration_current_a,
        torque_per_motor_nm=torque_per_motor,
        combined_torque_nm=combined_torque,
        angular_acceleration_rad_s2=angular_acceleration,
        total_inertia_kgm2=total_inertia,
        load_inertia_kgm2=load_inertia,
        load_inertia_per_motor_kgm2=load_per_motor,
        load_motor_ratio=load_motor_ratio,
        pn106_value=pn106,
        symmetry_error_percent=symmetry_error,
    )
