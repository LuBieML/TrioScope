import math

import pytest

from src.ai.analysis_move_profile import (
    RECOMMENDED_CRUISE_DURATION_S,
    SPECTRAL_CRUISE_FLOOR_S,
    calculate_analysis_move,
)


def test_recommended_cruise_exceeds_spectral_floor():
    assert RECOMMENDED_CRUISE_DURATION_S > SPECTRAL_CRUISE_FLOOR_S


def test_calculates_symmetric_trapezoidal_test_move():
    profile = calculate_analysis_move(speed=100.0, acceleration=500.0)

    assert profile.acceleration_duration_s == pytest.approx(0.2)
    assert profile.acceleration_distance == pytest.approx(10.0)
    assert profile.recommended_distance == pytest.approx(100.0)
    assert profile.total_move_duration_s == pytest.approx(1.2)
    assert profile.minimum_capture_duration_s == pytest.approx(1.8)


@pytest.mark.parametrize(
    ("speed", "acceleration"),
    [(0, 1), (-1, 1), (1, 0), (1, -1), (math.inf, 1), (1, math.nan)],
)
def test_rejects_non_positive_or_non_finite_inputs(speed, acceleration):
    with pytest.raises(ValueError, match="finite value greater than zero"):
        calculate_analysis_move(speed, acceleration)
