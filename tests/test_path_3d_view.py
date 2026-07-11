import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ui.path_3d_view import (
    CURSOR_MARKER_SIZE_PX,
    DEFAULT_CAMERA_DISTANCE,
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


if __name__ == "__main__":
    unittest.main()
