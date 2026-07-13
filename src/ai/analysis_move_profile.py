"""Recommended motion profile for signal-analysis captures.

The signal analyzer needs a long enough constant-speed section to resolve
low-frequency oscillation and separate acceleration, cruise, deceleration,
and settling behavior.  This module turns a requested speed and symmetric
acceleration into the minimum useful trapezoidal-move distance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .signal_constants import (
    MIN_CYCLES_FOR_PEAK,
    MIN_CRUISE_DURATION_S,
    MIN_OSCILLATION_HZ,
)


WELCH_LONGEST_RUN_FRACTION = 0.9
SPECTRAL_CRUISE_FLOOR_S = max(
    MIN_CRUISE_DURATION_S,
    MIN_CYCLES_FOR_PEAK / MIN_OSCILLATION_HZ / WELCH_LONGEST_RUN_FRACTION,
)
# 0.8 s leaves useful margin above the current ~0.667 s spectral floor.
RECOMMENDED_CRUISE_DURATION_S = 0.8
RECOMMENDED_IDLE_DURATION_S = 0.3
RECOMMENDED_DWELL_DURATION_S = 0.3


@dataclass(frozen=True)
class AnalysisMoveProfile:
    """Calculated symmetric trapezoidal move in configured axis units."""

    speed: float
    acceleration: float
    cruise_duration_s: float
    acceleration_duration_s: float
    acceleration_distance: float
    recommended_distance: float
    total_move_duration_s: float
    minimum_capture_duration_s: float


def calculate_analysis_move(
    speed: float,
    acceleration: float,
    *,
    cruise_duration_s: float = RECOMMENDED_CRUISE_DURATION_S,
) -> AnalysisMoveProfile:
    """Return an analyzer-ready trapezoidal move for symmetric accel/decel.

    ``speed`` is in configured axis-units/s and ``acceleration`` is in
    axis-units/s².  The result uses ``distance = V*Tcruise + V²/A``.
    """

    values = {
        "Speed": float(speed),
        "Acceleration": float(acceleration),
        "Cruise duration": float(cruise_duration_s),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite value greater than zero.")

    speed = values["Speed"]
    acceleration = values["Acceleration"]
    cruise_duration_s = values["Cruise duration"]
    acceleration_duration_s = speed / acceleration
    acceleration_distance = 0.5 * speed * acceleration_duration_s
    recommended_distance = (
        speed * cruise_duration_s + 2.0 * acceleration_distance
    )
    total_move_duration_s = (
        2.0 * acceleration_duration_s + cruise_duration_s
    )

    return AnalysisMoveProfile(
        speed=speed,
        acceleration=acceleration,
        cruise_duration_s=cruise_duration_s,
        acceleration_duration_s=acceleration_duration_s,
        acceleration_distance=acceleration_distance,
        recommended_distance=recommended_distance,
        total_move_duration_s=total_move_duration_s,
        minimum_capture_duration_s=(
            RECOMMENDED_IDLE_DURATION_S
            + total_move_duration_s
            + RECOMMENDED_DWELL_DURATION_S
        ),
    )
