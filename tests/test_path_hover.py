import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ui.path_hover import (  # noqa: E402
    nearest_projected_point_index,
    nearest_xy_point_index,
)


class TestNearestXYPoint(unittest.TestCase):
    def test_returns_actual_nearest_sample_in_pixel_space(self):
        index = nearest_xy_point_index(
            np.array([0.0, 10.0, 20.0]),
            np.array([0.0, 5.0, 0.0]),
            mouse_x=10.4,
            mouse_y=4.8,
            x_units_per_pixel=0.1,
            y_units_per_pixel=0.1,
        )

        self.assertEqual(index, 1)

    def test_returns_none_when_mouse_is_not_touching_path(self):
        index = nearest_xy_point_index(
            [0.0, 1.0],
            [0.0, 1.0],
            mouse_x=50.0,
            mouse_y=50.0,
            x_units_per_pixel=1.0,
            y_units_per_pixel=1.0,
        )

        self.assertIsNone(index)

    def test_uses_independent_axis_scales(self):
        index = nearest_xy_point_index(
            [0.0, 1.0],
            [0.0, 100.0],
            mouse_x=0.9,
            mouse_y=1.0,
            x_units_per_pixel=0.1,
            y_units_per_pixel=100.0,
        )

        self.assertEqual(index, 1)


class TestNearestProjectedPoint(unittest.TestCase):
    def test_ignores_invalid_projection_and_selects_nearest_point(self):
        points = np.array([
            [np.nan, np.nan],
            [20.0, 20.0],
            [25.0, 24.0],
        ])

        self.assertEqual(nearest_projected_point_index(points, 24.0, 24.0), 2)

    def test_requires_mouse_to_be_close_to_projected_trace(self):
        points = np.array([[20.0, 20.0], [30.0, 30.0]])

        self.assertIsNone(nearest_projected_point_index(points, 100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
