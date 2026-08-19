import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ui.path_3d_view import (  # noqa: E402
    CHUNK_SAMPLES,
    CURSOR_MARKER_SIZE_PX,
    DEFAULT_CAMERA_DISTANCE,
    FROZEN_SEGMENT_VERTICES,
    HOVER_BLOCK_SIZE,
    MAX_HOVER_CANDIDATES,
    Path3DView,
)


class _ViewStub:
    set_view_scale = Path3DView.set_view_scale

    def __init__(self):
        self.view_scale = 1.0
        self.camera_distance = DEFAULT_CAMERA_DISTANCE

    def setCameraPosition(self, *, distance):
        self.camera_distance = distance


class TestPath3DViewScale(unittest.TestCase):
    def test_scale_changes_camera_distance_for_the_complete_scene(self):
        view = _ViewStub()

        view.set_view_scale(2.0)

        self.assertEqual(view.view_scale, 2.0)
        self.assertEqual(view.camera_distance, DEFAULT_CAMERA_DISTANCE / 2.0)

    def test_scale_is_clamped_to_supported_range(self):
        view = _ViewStub()

        view.set_view_scale(0.01)
        self.assertEqual(view.view_scale, 0.25)
        self.assertEqual(view.camera_distance, DEFAULT_CAMERA_DISTANCE / 0.25)

        view.set_view_scale(10.0)
        self.assertEqual(view.view_scale, 4.0)
        self.assertEqual(view.camera_distance, DEFAULT_CAMERA_DISTANCE / 4.0)


class _CursorViewStub:
    _update_cursor = Path3DView._update_cursor

    def __init__(self):
        self.cursor_item = None
        self.added_items = []

    def addItem(self, item):
        self.added_items.append(item)


class TestPath3DViewCursor(unittest.TestCase):
    @patch("src.ui.path_3d_view.gl.GLScatterPlotItem")
    def test_cursor_uses_constant_pixel_size(self, scatter_item_cls):
        view = _CursorViewStub()

        view._update_cursor(1.0, 2.0, 3.0)

        kwargs = scatter_item_cls.call_args.kwargs
        np.testing.assert_array_equal(
            kwargs["pos"], np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        )
        self.assertEqual(kwargs["size"], CURSOR_MARKER_SIZE_PX)
        self.assertTrue(kwargs["pxMode"])
        self.assertEqual(view.added_items, [scatter_item_cls.return_value])

    @patch("src.ui.path_3d_view.gl.GLScatterPlotItem")
    def test_existing_cursor_moves_without_recreating_marker(self, scatter_item_cls):
        view = _CursorViewStub()
        view._update_cursor(1.0, 2.0, 3.0)

        view._update_cursor(4.0, 5.0, 6.0)

        scatter_item_cls.assert_called_once()
        np.testing.assert_array_equal(
            scatter_item_cls.return_value.setData.call_args.kwargs["pos"],
            np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        )


class _PathStub:
    """Exercises the sample-ingest pipeline without an OpenGL context."""

    _reset_path_buffers = Path3DView._reset_path_buffers
    _drop_path_items = Path3DView._drop_path_items
    _ingest = Path3DView._ingest
    _update_extents = Path3DView._update_extents
    _current_epsilon = Path3DView._current_epsilon
    _append_tail = Path3DView._append_tail
    _freeze_tail = Path3DView._freeze_tail
    _append_raw = Path3DView._append_raw
    _append_frozen = Path3DView._append_frozen
    _coarsen_frozen = Path3DView._coarsen_frozen
    _camera_pixel_epsilon = Path3DView._camera_pixel_epsilon
    _extent_epsilon = Path3DView._extent_epsilon
    _rebuild_display_from_raw = Path3DView._rebuild_display_from_raw
    _maybe_refine_for_view = Path3DView._maybe_refine_for_view
    _frozen_block_bounds = Path3DView._frozen_block_bounds
    _raw_block_bounds = Path3DView._raw_block_bounds
    _bounds_for_buffer = Path3DView._bounds_for_buffer
    _blocks_near = Path3DView._blocks_near
    _candidate_blocks = Path3DView._candidate_blocks
    _refresh_frozen_items = Path3DView._refresh_frozen_items
    _drop_frozen_items = Path3DView._drop_frozen_items

    def _upload_span(self, index):
        self.uploads.append((index, tuple(self._frozen_spans[index])))
        if self.frozen_items[index] is None:
            self.frozen_items[index] = object()

    def _project(self, points):
        return np.asarray(points, dtype=float)[:, :2]

    def __init__(self):
        self.frozen_items = []
        self.tail_item = None
        self.cursor_item = None
        self.hover_item = None
        self.uploads = []
        self._line_width = 1
        self._has_w = False
        self._reset_path_buffers()

    def removeItem(self, item):
        return None

    def frozen(self):
        return np.array(self._frozen_pts[:self._frozen_len])

    def tail(self):
        return np.array(self._tail_pts[:self._tail_len])

    def drawn_vertices(self):
        return self._frozen_len + self._tail_len


def _helix(n, turns=40.0, radius=30.0):
    """A smooth toolpath-like curve with real corners and long smooth runs."""
    t = np.linspace(0.0, turns * 2 * np.pi, n)
    return (
        np.cos(t) * radius,
        np.sin(t) * radius,
        np.linspace(0.0, 50.0, n),
    )


def _feed(view, x, y, z, step, w=None):
    """Push samples in frame-sized increments, as the renderer does."""
    snapshots = []
    for end in range(step, len(x) + step, step):
        end = min(end, len(x))
        view._ingest(
            x[:end], y[:end], z[:end],
            None if w is None else w[:end],
        )
        snapshots.append(view.frozen())
    return snapshots


class TestPath3DViewIngest(unittest.TestCase):
    def test_frozen_geometry_is_never_rewritten_as_the_path_grows(self):
        x, y, z = _helix(150000)
        view = _PathStub()
        snapshots = _feed(view, x, y, z, step=4096)
        self.assertGreater(len(snapshots[-1]), 0)
        for earlier, later in zip(snapshots, snapshots[1:]):
            self.assertGreaterEqual(len(later), len(earlier))
            np.testing.assert_array_equal(later[:len(earlier)], earlier)

    def test_drawn_vertices_stay_far_below_the_raw_sample_count(self):
        x, y, z = _helix(1000000)
        view = _PathStub()
        _feed(view, x, y, z, step=50000)
        self.assertLess(view.drawn_vertices(), len(x) / 50)

    def test_each_chunk_is_simplified_exactly_once(self):
        x, y, z = _helix(CHUNK_SAMPLES * 6)
        view = _PathStub()
        calls = []
        real = Path3DView._freeze_tail

        def counting_freeze(self):
            calls.append(self._tail_len)
            return real(self)

        with patch.object(_PathStub, "_freeze_tail", counting_freeze):
            _feed(view, x, y, z, step=1000)

        total = CHUNK_SAMPLES * 6
        expected = 1 + (total - CHUNK_SAMPLES) // (CHUNK_SAMPLES - 1)
        self.assertEqual(len(calls), expected)
        self.assertTrue(all(count == CHUNK_SAMPLES for count in calls))

    def test_geometry_does_not_depend_on_how_frames_split_the_samples(self):
        x, y, z = _helix(120000)
        by_frame = _PathStub()
        _feed(by_frame, x, y, z, step=997)
        in_bulk = _PathStub()
        in_bulk._ingest(x, y, z, None)
        np.testing.assert_array_equal(by_frame.frozen(), in_bulk.frozen())
        np.testing.assert_array_equal(by_frame.tail(), in_bulk.tail())

    def test_chunks_join_without_a_gap(self):
        x, y, z = _helix(CHUNK_SAMPLES * 2)
        view = _PathStub()
        view._ingest(x, y, z, None)
        np.testing.assert_array_equal(view.frozen()[-1], view.tail()[0])

    def test_shorter_input_rebuilds_from_scratch(self):
        x, y, z = _helix(CHUNK_SAMPLES * 3)
        view = _PathStub()
        view._ingest(x, y, z, None)
        shorter = len(x) // 2
        view._ingest(x[:shorter], y[:shorter], z[:shorter], None)
        rebuilt = _PathStub()
        rebuilt._ingest(x[:shorter], y[:shorter], z[:shorter], None)
        np.testing.assert_array_equal(view.frozen(), rebuilt.frozen())
        np.testing.assert_array_equal(view.tail(), rebuilt.tail())

    def test_switching_to_w_mode_rebuilds_with_w_tracked(self):
        x, y, z = _helix(CHUNK_SAMPLES * 2)
        view = _PathStub()
        view._ingest(x, y, z, None)
        self.assertFalse(view._has_w)
        view._ingest(x, y, z, z)
        self.assertTrue(view._has_w)
        self.assertEqual(view._consumed, len(x))

    def test_w_values_stay_aligned_with_their_samples(self):
        x, y, z = _helix(CHUNK_SAMPLES * 2)
        w = np.asarray(z) * 3.0
        view = _PathStub()
        view._ingest(x, y, z, w)
        frozen_z = view._frozen_pts[:view._frozen_len, 2]
        frozen_w = view._frozen_w[:view._frozen_len]
        np.testing.assert_allclose(frozen_w, frozen_z * 3.0, rtol=0.0001, atol=0.001)

    def test_running_extents_match_a_full_scan(self):
        x, y, z = _helix(20000)
        view = _PathStub()
        _feed(view, x, y, z, step=1500)
        pts = np.column_stack([x, y, z])
        np.testing.assert_allclose(view._extent_min, pts.min(axis=0))
        np.testing.assert_allclose(view._extent_max, pts.max(axis=0))

    def test_unsimplifiable_path_is_capped_instead_of_growing_unbounded(self):
        rng = np.random.default_rng(99)
        steps = rng.normal(0, 5.0, (200000, 3))
        walk = np.cumsum(steps, axis=0)
        view = _PathStub()
        with patch("src.ui.path_3d_view.MAX_FROZEN_VERTICES", 20000):
            _feed(view, walk[:, 0], walk[:, 1], walk[:, 2], step=50000)
        self.assertLessEqual(view._frozen_len, 20000)
        self.assertGreater(view._epsilon_floor, 0.0)

    def test_block_bounds_cover_the_whole_frozen_path(self):
        x, y, z = _helix(CHUNK_SAMPLES * 5)
        view = _PathStub()
        _feed(view, x, y, z, step=CHUNK_SAMPLES)
        bounds = view._frozen_block_bounds()
        expected_blocks = (view._frozen_len + HOVER_BLOCK_SIZE - 1) // HOVER_BLOCK_SIZE
        self.assertEqual(len(bounds), expected_blocks)
        for index, (low, high) in enumerate(bounds):
            block = view._frozen_pts[index * HOVER_BLOCK_SIZE:(index + 1) * HOVER_BLOCK_SIZE]
            block = block[:view._frozen_len - index * HOVER_BLOCK_SIZE]
            self.assertTrue(np.all(block >= low))
            self.assertTrue(np.all(block <= high))

    def test_frozen_path_is_drawn_as_bounded_continuous_segments(self):
        view = _PathStub()
        view._frozen_pts = np.zeros(
            (FROZEN_SEGMENT_VERTICES * 3 + 500, 3), dtype=np.float32)
        view._frozen_len = len(view._frozen_pts)
        view._refresh_frozen_items()
        spans = view._frozen_spans
        self.assertEqual(len(spans), 4)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], view._frozen_len)
        for start, end in spans:
            self.assertLessEqual(end - start, FROZEN_SEGMENT_VERTICES)
        for (_, end), (next_start, _) in zip(spans, spans[1:]):
            self.assertEqual(next_start, end - 1)

    def test_closed_segments_are_never_re_uploaded(self):
        view = _PathStub()
        total = FROZEN_SEGMENT_VERTICES * 3
        view._frozen_pts = np.zeros((total, 3), dtype=np.float32)
        for end in range(CHUNK_SAMPLES, total + 1, CHUNK_SAMPLES):
            view._frozen_len = end
            view._refresh_frozen_items()
        self.assertGreater(len(view._frozen_spans), 2)
        order = [index for index, _ in view.uploads]
        self.assertEqual(order, sorted(order))
        for index, (start, end) in view.uploads:
            self.assertLessEqual(end - start, FROZEN_SEGMENT_VERTICES)

    def test_coarsening_rebuilds_the_drawn_segments(self):
        view = _PathStub()
        view._frozen_pts = np.zeros((FROZEN_SEGMENT_VERTICES + 10, 3), dtype=np.float32)
        view._frozen_len = len(view._frozen_pts)
        view._refresh_frozen_items()
        self.assertGreater(len(view.frozen_items), 0)
        view._coarsen_frozen()
        self.assertEqual(view._frozen_spans, [])
        self.assertEqual(view.frozen_items, [])

    def test_distant_blocks_are_skipped_when_picking(self):
        view = _PathStub()
        blocks = np.concatenate([
            np.zeros((HOVER_BLOCK_SIZE, 3)),
            np.full((HOVER_BLOCK_SIZE, 3), 5000.0),
            np.full((HOVER_BLOCK_SIZE, 3), 9000.0),
        ]).astype(np.float32)
        view._frozen_pts = blocks
        view._frozen_len = len(blocks)
        np.testing.assert_array_equal(view._candidate_blocks(0.0, 0.0), [0])
        np.testing.assert_array_equal(view._candidate_blocks(5000.0, 5000.0), [1])

    def test_candidate_blocks_are_capped_when_everything_overlaps(self):
        view = _PathStub()
        view._frozen_pts = np.zeros((HOVER_BLOCK_SIZE * 40, 3), dtype=np.float32)
        view._frozen_len = len(view._frozen_pts)
        blocks = view._candidate_blocks(0.0, 0.0)
        self.assertLessEqual(len(blocks), MAX_HOVER_CANDIDATES)
        self.assertGreater(len(blocks), 0)
        candidates = view._candidate_blocks(0.0, 0.0)
        self.assertLessEqual(len(candidates), max(1, MAX_HOVER_CANDIDATES // HOVER_BLOCK_SIZE))

    def test_block_bounds_cache_extends_rather_than_recomputing(self):
        x, y, z = _helix(CHUNK_SAMPLES * 4)
        view = _PathStub()
        _feed(view, x, y, z, step=CHUNK_SAMPLES)
        first = view._frozen_block_bounds()
        cached_full_blocks = len(view._block_bounds_cache)
        again = view._frozen_block_bounds()
        np.testing.assert_array_equal(first[:cached_full_blocks], again[:cached_full_blocks])
        self.assertEqual(
            len(view._block_bounds_cache),
            view._frozen_len // HOVER_BLOCK_SIZE,
        )

    def test_zoom_in_restores_a_feature_smaller_than_the_coarse_tolerance(self):
        n = CHUNK_SAMPLES
        x = np.linspace(0.0, 100.0, n)
        y = np.zeros(n)
        z = np.zeros(n)
        spike = n // 2
        y[spike] = 0.01
        view = _PathStub()
        view._ingest(x, y, z, None)
        coarse = view.frozen()
        self.assertFalse(np.any(np.abs(coarse[:, 1]) > 0.005))

        view._epsilon_override = 0.001
        view._rebuild_display_from_raw()
        fine = view.frozen()
        self.assertTrue(np.any(np.abs(fine[:, 1]) > 0.005))
        self.assertGreater(len(fine), len(coarse))
        self.assertEqual(view._raw_len, n)

    def test_raw_samples_are_kept_when_the_display_is_simplified(self):
        x, y, z = _helix(CHUNK_SAMPLES * 3)
        view = _PathStub()
        view._ingest(x, y, z, None)
        self.assertGreater(view._raw_len, view._frozen_len)
        self.assertGreater(len(view._raw_chunks), 0)


if __name__ == "__main__":
    unittest.main()
