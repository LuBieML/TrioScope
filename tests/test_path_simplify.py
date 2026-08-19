import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ui.path_hover import (  # noqa: E402
    PATH_HOVER_DISTANCE_PX,
    block_screen_distances_sq,
)
from src.ui.path_simplify import (  # noqa: E402
    MAX_KEPT_FRACTION,
    MIN_SPLIT_BUDGET,
    aabb_corners,
    block_bounds,
    neighbour_chord_distances,
    simplify_indices,
)


def _polyline_distance(points, kept_indices):
    """Largest distance from any original point to the simplified polyline."""
    kept = points[kept_indices]
    starts = kept[:-1]
    ends = kept[1:]
    worst = 0.0
    for point in points:
        offsets = point - starts
        chords = ends - starts
        chord_sq = np.einsum("ij,ij->i", chords, chords)
        t = np.divide(
            np.einsum("ij,ij->i", offsets, chords),
            np.where(chord_sq > 0, chord_sq, 1.0),
        )
        np.clip(t, 0.0, 1.0, out=t)
        perp = offsets - t[:, None] * chords
        worst = max(
            worst,
            float(np.sqrt(np.einsum("ij,ij->i", perp, perp)).min()),
        )
    return worst


class TestSimplifyIndices(unittest.TestCase):
    def test_straight_run_collapses_to_its_endpoints(self):
        points = np.column_stack([
            np.linspace(0, 100, 5000),
            np.zeros(5000),
            np.zeros(5000),
        ])
        kept = simplify_indices(points, 0.01)
        np.testing.assert_array_equal(kept, [0, 4999])

    def test_corners_are_never_dropped(self):
        points = np.array([
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (20.0, 10.0, 0.0),
            (20.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
        ])
        kept = simplify_indices(points, 0.5)
        np.testing.assert_array_equal(kept, np.arange(len(points)))

    def test_every_dropped_point_stays_within_epsilon(self):
        rng = np.random.default_rng(1234)
        t = np.linspace(0, 8 * np.pi, 20000)
        points = np.column_stack([
            np.cos(t) * 30 + rng.normal(0, 0.02, t.size),
            np.sin(t) * 30 + rng.normal(0, 0.02, t.size),
            t * 0.5,
        ])
        epsilon = 0.25
        kept = simplify_indices(points, epsilon)
        self.assertLess(len(kept), len(points) / 10)
        self.assertLessEqual(_polyline_distance(points, kept), epsilon + 1e-09)

    def test_endpoints_are_always_kept_so_chunks_join(self):
        points = np.column_stack([
            np.linspace(0, 1, 100),
            np.zeros(100),
            np.zeros(100),
        ])
        kept = simplify_indices(points, 0.01)
        self.assertEqual(kept[0], 0)
        self.assertEqual(kept[-1], len(points) - 1)

    def test_path_doubling_back_is_not_flattened(self):
        points = np.array([
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ])
        kept = simplify_indices(points, 0.1)
        self.assertIn(2, kept.tolist())

    def test_degenerate_inputs_are_returned_untouched(self):
        points = np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        np.testing.assert_array_equal(simplify_indices(points, 0.1), [0, 1])
        empty = np.empty((0, 3))
        np.testing.assert_array_equal(simplify_indices(empty, 0.1), [])
        many = np.column_stack([np.arange(10), np.zeros(10), np.zeros(10)])
        np.testing.assert_array_equal(simplify_indices(many, 0), np.arange(10))


class TestSimplifyCostBounds(unittest.TestCase):
    """Douglas-Peucker either stays cheap or gives up and keeps every sample."""

    def _paths(self):
        rng = np.random.default_rng(2024)
        t = np.linspace(0, 20, 30000)
        yield "noise", np.cumsum(rng.normal(0, 1.0, (30000, 3)), axis=0)
        yield "smooth", np.column_stack([np.cos(t) * 20, np.sin(t) * 20, t])
        yield "straight", np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
        yield "jittered_line", np.column_stack([
            t,
            rng.normal(0, 0.05, t.size),
            rng.normal(0, 0.05, t.size),
        ])

    def test_kept_count_is_bounded_or_the_run_is_kept_whole(self):
        for name, points in self._paths():
            with self.subTest(path=name):
                n = len(points)
                budget = max(MIN_SPLIT_BUDGET, n // MAX_KEPT_FRACTION)
                kept = simplify_indices(points, 0.02)
                self.assertTrue(
                    len(kept) <= budget or len(kept) == n,
                    f"{name}: kept {len(kept)} (budget {budget}, n {n})",
                )

    def test_giving_up_keeps_every_point_so_epsilon_still_holds(self):
        rng = np.random.default_rng(7)
        points = np.cumsum(rng.normal(0, 1.0, (800, 3)), axis=0)
        kept = simplify_indices(points, 1e-9)
        np.testing.assert_array_equal(kept, np.arange(len(points)))

    def test_screening_does_not_spoil_a_compressible_path(self):
        segments = []
        for index in range(8):
            step = np.linspace(0, 10, 200)
            zeros = np.zeros_like(step)
            if index % 2 == 0:
                segments.append(np.column_stack([step + index * 10, zeros, zeros]))
            else:
                segments.append(np.column_stack([zeros + index * 10, step, zeros]))
        points = np.concatenate(segments)
        kept = simplify_indices(points, 0.05)
        self.assertLess(len(kept), len(points) / 10)
        self.assertLessEqual(_polyline_distance(points, kept), 0.05 + 1e-09)


class TestNeighbourChordDistances(unittest.TestCase):
    def test_collinear_points_measure_zero(self):
        points = np.column_stack([
            np.linspace(0, 10, 20),
            np.zeros(20),
            np.zeros(20),
        ])
        np.testing.assert_allclose(neighbour_chord_distances(points), 0.0, atol=1e-12)

    def test_a_spike_measures_its_own_height(self):
        points = np.array([
            (0.0, 0.0, 0.0),
            (1.0, 3.0, 0.0),
            (2.0, 0.0, 0.0),
        ])
        np.testing.assert_allclose(neighbour_chord_distances(points), [3.0])

    def test_short_runs_yield_nothing(self):
        self.assertEqual(len(neighbour_chord_distances(np.zeros((2, 3)))), 0)
        self.assertEqual(len(neighbour_chord_distances(np.zeros((1, 3)))), 0)
        self.assertEqual(len(neighbour_chord_distances(np.zeros((0, 3)))), 0)


class TestBlockBounds(unittest.TestCase):
    def test_bounds_cover_every_point_in_each_block(self):
        rng = np.random.default_rng(7)
        points = rng.normal(0, 5, (2500, 3))
        bounds = block_bounds(points, 1000)
        self.assertEqual(bounds.shape, (3, 2, 3))
        for index, (low, high) in enumerate(bounds):
            block = points[index * 1000:(index + 1) * 1000]
            self.assertTrue(np.all(block >= low - 1e-05))
            self.assertTrue(np.all(block <= high + 1e-05))

    def test_empty_input_yields_no_blocks(self):
        self.assertEqual(block_bounds(np.empty((0, 3)), 8).shape, (0, 2, 3))

    def test_corners_span_the_bounding_box(self):
        bounds = np.array([[(-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)]])
        corners = aabb_corners(bounds)
        self.assertEqual(corners.shape, (1, 8, 3))
        unique = {tuple(corner) for corner in corners[0]}
        self.assertEqual(len(unique), 8)
        np.testing.assert_array_equal(corners[0].min(axis=0), [-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(corners[0].max(axis=0), [1.0, 2.0, 3.0])


class TestBlockScreenDistances(unittest.TestCase):
    PATH_HOVER_DISTANCE_PX = PATH_HOVER_DISTANCE_PX
    RADIUS_SQ = PATH_HOVER_DISTANCE_PX ** 2

    def _square(self, x0, y0, size):
        offsets = np.array([
            [0.0, 0.0], [size, 0.0], [size, size], [0.0, size],
            [0.0, 0.0], [size, 0.0], [size, size], [0.0, size],
        ], dtype=float)
        return offsets + np.array([x0, y0], dtype=float)

    def test_far_blocks_are_culled_and_near_blocks_kept(self):
        corners = np.stack([
            self._square(0, 0, 10),
            self._square(500, 500, 10),
        ])
        distances = block_screen_distances_sq(corners, 5.0, 5.0)
        np.testing.assert_array_equal(distances <= self.RADIUS_SQ, [True, False])

    def test_distance_is_measured_to_the_box_not_its_centre(self):
        corners = np.stack([self._square(10, 0, 10)])
        self.assertAlmostEqual(
            float(block_screen_distances_sq(corners, 0.0, 5.0)[0]),
            10.0 ** 2,
        )

    def test_a_block_containing_the_cursor_is_at_distance_zero(self):
        corners = np.stack([self._square(0, 0, 10)])
        self.assertEqual(block_screen_distances_sq(corners, 5.0, 5.0)[0], 0.0)

    def test_blocks_behind_the_camera_are_never_culled(self):
        corners = np.stack([self._square(0, 0, 10)])
        corners[0, 0] = np.nan
        self.assertEqual(block_screen_distances_sq(corners, 1000.0, 1000.0)[0], 0.0)

    def test_no_blocks_gives_empty_result(self):
        self.assertEqual(
            block_screen_distances_sq(np.empty((0, 8, 2)), 0.0, 0.0).shape,
            (0,),
        )


if __name__ == "__main__":
    unittest.main()
