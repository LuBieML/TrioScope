"""OpenGL path view for XYZ and XYZW plot modes.

Built for multi-hour captures. The path is split into a short live *tail* of raw
samples and a *frozen* buffer of geometry-simplified vertices. A chunk is simplified when it leaves the tail, but the raw samples are kept.
Zooming in rebuilds the drawn path from those samples at pixel-scale
tolerance, so close-up views recover full capture resolution. Per-frame work
while capturing stays proportional to the samples that just arrived.
"""

from html import escape
import logging
import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from ui.path_hover import (
    PATH_HOVER_DISTANCE_PX,
    block_screen_distances_sq,
    nearest_projected_point_index,
)
from ui.path_simplify import (
    aabb_corners,
    block_bounds,
    decimate_to_budget,
    simplify_indices,
)

logger = logging.getLogger(__name__)

GOLDEN_PATH_COLOR = (1.0, 0.843, 0.0, 1.0)  # #FFD700
DEFAULT_CAMERA_DISTANCE = 50.0
CURSOR_MARKER_SIZE_PX = 8.0
HOVER_MARKER_SIZE_PX = 11.0
# Raw samples held at full resolution until the chunk is retired.
CHUNK_SAMPLES = 4096
# Frozen-path hover culling block size (same as a chunk so bounds stay cheap).
HOVER_BLOCK_SIZE = 4096
# Independent GL line items so only the newest segment is re-uploaded.
FROZEN_SEGMENT_VERTICES = 65536
MAX_HOVER_CANDIDATES = 65536
# World-space tolerance as a fraction of the running path extent (zoomed out).
EPSILON_FRACTION = 0.00025
# Drawn vertices stay within this many pixels of the raw path when zoomed in.
PIXEL_EPSILON = 0.5
# Rebuild when camera pixel size halves (zoom in) or grows 4x (zoom out).
REFINE_IN_RATIO = 0.5
REFINE_OUT_RATIO = 4.0
# Hard cap on frozen vertices; only noisy unsimplifiable paths reach this.
MAX_FROZEN_VERTICES = 2000000
W_RANGE_RECOLOR_FRACTION = 0.02

_W_COLORMAP = None


def _w_colormap():
    """Cached turbo colormap -- the tail is recoloured on every frame."""
    global _W_COLORMAP
    if _W_COLORMAP is None:
        _W_COLORMAP = pg.colormap.get("turbo")
    return _W_COLORMAP


class Path3DView(gl.GLViewWidget):
    """OpenGL path view for XYZ and XYZW plot modes."""

    viewScaleChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor("#0A0A0A")
        self.view_scale = 1.0
        self.setCameraPosition(distance=DEFAULT_CAMERA_DISTANCE)
        self.grid_item = None
        self.frozen_items = []
        self.tail_item = None
        self.cursor_item = None
        self.hover_item = None
        self.colorbar_items = []
        self._cb_min_label = None
        self._cb_max_label = None
        self._cb_title_label = None
        self._coordinate_names = ("X", "Y", "Z")
        self._last_hover_pos = None
        self._line_width = 1
        self._reset_path_buffers()

        self.setMouseTracking(True)
        self.coordinate_label = QLabel(self)
        self.coordinate_label.setTextFormat(Qt.RichText)
        self.coordinate_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.coordinate_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.coordinate_label.setStyleSheet(
            "QLabel { color: #d4d4d4; background-color: rgba(26, 26, 46, 220);"
            " border: 1px solid #55556a; border-radius: 4px; padding: 5px;"
            " font-family: Consolas; font-size: 9pt; }"
        )
        self.coordinate_label.hide()

    def setup_view(self, mode, enabled_traces):
        for item in self.items[:]:
            self.removeItem(item)
        self.frozen_items = []
        self.tail_item = None
        self.cursor_item = None
        self.hover_item = None
        self.colorbar_items = []
        self._reset_path_buffers()
        self._last_hover_pos = None
        self.coordinate_label.hide()

        if len(enabled_traces) >= 3:
            names = [trace.get_display_name() for trace in enabled_traces[:4]]
            self._coordinate_names = tuple(names)

        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(100, 100)
        self.grid_item.setSpacing(10, 10)
        self.grid_item.setColor((255, 255, 255, 40))
        self.addItem(self.grid_item)

        axis_len = 50
        for direction, color in [
            ([axis_len, 0, 0], (1, 0.2, 0.2, 0.8)),
            ([0, axis_len, 0], (0.2, 1, 0.2, 0.8)),
            ([0, 0, axis_len], (0.4, 0.4, 1, 0.8)),
        ]:
            pts = np.array([[0, 0, 0], direction], dtype=np.float32)
            self.addItem(gl.GLLinePlotItem(
                pos=pts, color=color, width=2, antialias=True))

        if len(enabled_traces) >= 3:
            labels = [
                (enabled_traces[0].get_display_name(), [axis_len + 3, 0, 0]),
                (enabled_traces[1].get_display_name(), [0, axis_len + 3, 0]),
                (enabled_traces[2].get_display_name(), [0, 0, axis_len + 3]),
            ]
            font = QFont("Segoe UI", 8)
            for text, pos in labels:
                self.addItem(gl.GLTextItem(
                    pos=pos, text=text, color=(212, 212, 212, 200), font=font))

        if mode == "xyzw" and len(enabled_traces) >= 4:
            self.build_colorbar(axis_len, enabled_traces[3].get_display_name())

    def build_colorbar(self, axis_len, w_label):
        self.colorbar_items = []
        cmap = pg.colormap.get("turbo")
        num_segments = 30
        bar_x = axis_len + 10
        bar_height = axis_len * 0.8
        seg_h = bar_height / num_segments

        for i in range(num_segments):
            t = i / (num_segments - 1)
            color = cmap.map([t], mode="float")[0]
            z_base = i * seg_h
            pts = np.array([
                [bar_x, 0, z_base],
                [bar_x, 0, z_base + seg_h],
            ], dtype=np.float32)
            seg = gl.GLLinePlotItem(
                pos=pts, color=tuple(color), width=12, antialias=True)
            self.addItem(seg)
            self.colorbar_items.append(seg)

        font = QFont("Segoe UI", 7)
        self._cb_min_label = gl.GLTextItem(
            pos=[bar_x + 2, 0, -2], text="min",
            color=(212, 212, 212, 200), font=font)
        self.addItem(self._cb_min_label)
        self.colorbar_items.append(self._cb_min_label)

        self._cb_max_label = gl.GLTextItem(
            pos=[bar_x + 2, 0, bar_height + 1], text="max",
            color=(212, 212, 212, 200), font=font)
        self.addItem(self._cb_max_label)
        self.colorbar_items.append(self._cb_max_label)

        title_font = QFont("Segoe UI", 8)
        self._cb_title_label = gl.GLTextItem(
            pos=[bar_x - 2, 0, bar_height + 5], text=w_label,
            color=(255, 165, 0, 220), font=title_font)
        self.addItem(self._cb_title_label)
        self.colorbar_items.append(self._cb_title_label)

    def update_colorbar_range(self, w_min, w_max):
        if self._cb_min_label is not None:
            self._cb_min_label.setData(text=f"{w_min:.2f}")
        if self._cb_max_label is not None:
            self._cb_max_label.setData(text=f"{w_max:.2f}")

    def render_xyz(self, x_vals, y_vals, z_vals, line_width=1):
        self._line_width = line_width
        ingested = self._ingest(x_vals, y_vals, z_vals, None)
        refined = self._maybe_refine_for_view()
        if ingested or refined:
            self._refresh_frozen_items()
            self._refresh_tail_item()
            self._fit_grid()
        self._update_cursor(float(x_vals[-1]), float(y_vals[-1]), float(z_vals[-1]))

    def render_xyzw(self, x_vals, y_vals, z_vals, w_vals, line_width=1):
        self._line_width = line_width
        ingested = self._ingest(x_vals, y_vals, z_vals, w_vals)
        refined = self._maybe_refine_for_view()
        if ingested or refined:
            self._refresh_frozen_items()
            self._refresh_tail_item()
            self._fit_grid()
            self.update_colorbar_range(self._w_min, self._w_max)
        self._update_cursor(float(x_vals[-1]), float(y_vals[-1]), float(z_vals[-1]))

    def clear_path(self):
        self._drop_path_items()
        self._reset_path_buffers()
        self.coordinate_label.hide()

    def _reset_path_buffers(self):
        self._consumed = 0
        self._has_w = False
        self._frozen_pts = np.empty((0, 3), dtype=np.float32)
        self._frozen_w = np.empty(0, dtype=np.float32)
        self._frozen_len = 0
        self._frozen_spans = []
        self._raw_pts = np.empty((0, 3), dtype=np.float32)
        self._raw_w = np.empty(0, dtype=np.float32)
        self._raw_len = 0
        self._raw_chunks = []
        self._raw_bounds_cache = None
        self._display_camera_epsilon = None
        self._tail_pts = np.empty((CHUNK_SAMPLES, 3), dtype=np.float32)
        self._tail_w = np.empty(CHUNK_SAMPLES, dtype=np.float32)
        self._tail_len = 0
        self._epsilon_floor = 0.0
        self._extent_min = None
        self._extent_max = None
        self._w_min = np.inf
        self._w_max = -np.inf
        self._colored_w_min = np.inf
        self._colored_w_max = -np.inf
        self._block_bounds_cache = None

    def _drop_path_items(self):
        self._drop_frozen_items()
        for item in (self.tail_item, self.cursor_item, self.hover_item):
            if item is None:
                continue
            self.removeItem(item)
        self.tail_item = None
        self.cursor_item = None
        self.hover_item = None

    def _drop_frozen_items(self):
        for item in self.frozen_items:
            if item is None:
                continue
            self.removeItem(item)
        self.frozen_items = []
        self._frozen_spans = []

    def _ingest(self, x_vals, y_vals, z_vals, w_vals):
        """Absorb samples appended since the last frame. Returns True if any."""
        has_w = w_vals is not None
        n = len(x_vals)

        if n < self._consumed or has_w != self._has_w:
            self._drop_path_items()
            self._reset_path_buffers()
            self._has_w = has_w

        if n <= self._consumed:
            return False

        start = self._consumed
        new_pts = np.column_stack([
            x_vals[start:n], y_vals[start:n], z_vals[start:n],
        ]).astype(np.float32, copy=False)
        new_w = (
            np.asarray(w_vals[start:n], dtype=np.float32) if has_w else None
        )
        self._consumed = n
        self._append_tail(new_pts, new_w)
        return True

    def _update_extents(self, new_pts, new_w):
        """Track running bounds so per-frame cost stays O(new samples)."""
        low = new_pts.min(axis=0)
        high = new_pts.max(axis=0)
        if self._extent_min is None:
            self._extent_min = low
            self._extent_max = high
        else:
            self._extent_min = np.minimum(self._extent_min, low)
            self._extent_max = np.maximum(self._extent_max, high)
        if new_w is not None and len(new_w):
            self._w_min = min(self._w_min, float(new_w.min()))
            self._w_max = max(self._w_max, float(new_w.max()))

    def _camera_pixel_epsilon(self):
        """World-space size of PIXEL_EPSILON screen pixels, or None if unknown."""
        opts = getattr(self, "opts", None)
        if not opts:
            return None
        try:
            height = max(int(self.height()), 1)
            distance = max(float(opts.get("distance", DEFAULT_CAMERA_DISTANCE)), 1e-6)
            fov = float(opts.get("fov", 60.0))
        except (TypeError, ValueError, AttributeError):
            return None
        pixel = 2.0 * distance * math.tan(math.radians(fov) * 0.5) / height
        return pixel * PIXEL_EPSILON

    def _extent_epsilon(self):
        if self._extent_min is None:
            return self._epsilon_floor
        span = float(np.max(self._extent_max - self._extent_min))
        return max(self._epsilon_floor, span * EPSILON_FRACTION)

    def _current_epsilon(self):
        override = getattr(self, "_epsilon_override", None)
        if override is not None:
            return max(self._epsilon_floor, float(override))
        extent_eps = self._extent_epsilon()
        camera_eps = self._camera_pixel_epsilon()
        if camera_eps is None:
            return extent_eps
        if self._extent_min is None:
            return max(self._epsilon_floor, camera_eps)
        # Zoom in uses the tighter camera tolerance so new chunks stay sharp.
        return max(self._epsilon_floor, min(extent_eps, camera_eps))

    def _append_tail(self, pts, w):
        offset = 0
        total = len(pts)
        while offset < total:
            take = min(CHUNK_SAMPLES - self._tail_len, total - offset)
            end = self._tail_len + take
            self._tail_pts[self._tail_len:end] = pts[offset:offset + take]
            if w is not None:
                self._tail_w[self._tail_len:end] = w[offset:offset + take]
            self._tail_len = end
            self._update_extents(
                pts[offset:offset + take],
                None if w is None else w[offset:offset + take],
            )
            offset += take
            if self._tail_len >= CHUNK_SAMPLES:
                self._freeze_tail()

    def _freeze_tail(self):
        """Retire a finished tail chunk: keep the raw samples, draw a
        simplified copy. Zoom-in rebuilds the drawing from the raw samples.
        """
        if self._tail_len < 2:
            return
        pts = np.array(self._tail_pts[:self._tail_len], dtype=np.float32)
        w = (
            np.array(self._tail_w[:self._tail_len], dtype=np.float32)
            if self._has_w else None
        )
        self._append_raw(pts, w)
        keep = simplify_indices(pts, self._current_epsilon())
        kept_w = w[keep] if w is not None else None
        self._append_frozen(pts[keep], kept_w)
        if self._display_camera_epsilon is None:
            self._display_camera_epsilon = self._camera_pixel_epsilon()
        # Leave the last sample as the first of the next tail so chunks join.
        self._tail_pts[0] = self._tail_pts[self._tail_len - 1]
        if self._has_w:
            self._tail_w[0] = self._tail_w[self._tail_len - 1]
        self._tail_len = 1

    def _append_raw(self, pts, w):
        """Keep every retired sample so zoom-in can restore full resolution."""
        if len(pts) == 0:
            return
        start = self._raw_len
        needed = start + len(pts)
        if needed > len(self._raw_pts):
            capacity = max(65536, len(self._raw_pts) * 2, needed)
            grown = np.empty((capacity, 3), dtype=np.float32)
            grown[:start] = self._raw_pts[:start]
            self._raw_pts = grown
            grown_w = np.empty(capacity, dtype=np.float32)
            grown_w[:start] = self._raw_w[:start]
            self._raw_w = grown_w
        self._raw_pts[start:needed] = pts
        if w is not None:
            self._raw_w[start:needed] = w
        self._raw_len = needed
        self._raw_chunks.append((start, needed))

    def _maybe_refine_for_view(self):
        """Rebuild drawn vertices from raw samples when zoom changes enough."""
        camera = self._camera_pixel_epsilon()
        if camera is None or self._raw_len < 2:
            return False
        previous = self._display_camera_epsilon
        if previous is None:
            self._display_camera_epsilon = camera
            return False
        if camera < previous * REFINE_IN_RATIO or camera > previous * REFINE_OUT_RATIO:
            self._rebuild_display_from_raw()
            return True
        return False

    def _rebuild_display_from_raw(self):
        """Re-simplify every raw chunk at the current zoom tolerance."""
        epsilon = self._current_epsilon()
        self._drop_frozen_items()
        self._frozen_len = 0
        self._frozen_pts = np.empty((0, 3), dtype=np.float32)
        self._frozen_w = np.empty(0, dtype=np.float32)
        self._block_bounds_cache = None
        for start, end in self._raw_chunks:
            pts = self._raw_pts[start:end]
            keep = simplify_indices(pts, epsilon)
            kept_w = self._raw_w[start:end][keep] if self._has_w else None
            self._append_frozen(pts[keep], kept_w)
        self._display_camera_epsilon = self._camera_pixel_epsilon()

    def _append_frozen(self, pts, w):
        if self._frozen_len > 0 and len(pts) > 0:
            # Drop the duplicated join vertex already stored at the seam.
            pts = pts[1:]
            w = w[1:] if w is not None else None
        if len(pts) == 0:
            return

        needed = self._frozen_len + len(pts)
        if needed > len(self._frozen_pts):
            capacity = max(65536, len(self._frozen_pts) * 2, needed)
            grown = np.empty((capacity, 3), dtype=np.float32)
            grown[:self._frozen_len] = self._frozen_pts[:self._frozen_len]
            self._frozen_pts = grown
            grown_w = np.empty(capacity, dtype=np.float32)
            grown_w[:self._frozen_len] = self._frozen_w[:self._frozen_len]
            self._frozen_w = grown_w

        self._frozen_pts[self._frozen_len:needed] = pts
        if w is not None:
            self._frozen_w[self._frozen_len:needed] = w
        self._frozen_len = needed

        if self._frozen_len > MAX_FROZEN_VERTICES:
            self._coarsen_frozen()

    def _coarsen_frozen(self):
        """Last-resort reduction when a path refuses to simplify honestly.

        Only reachable when sample noise exceeds the tolerance everywhere, so
        nothing is collinear. This drops detail beyond the epsilon guarantee, so
        it is logged rather than done silently.
        """
        before = self._frozen_len
        keep = decimate_to_budget(
            self._frozen_pts[:self._frozen_len],
            MAX_FROZEN_VERTICES // 2,
        )
        self._frozen_pts[:len(keep)] = self._frozen_pts[:self._frozen_len][keep]
        if self._has_w:
            self._frozen_w[:len(keep)] = self._frozen_w[:self._frozen_len][keep]
        self._frozen_len = len(keep)
        self._block_bounds_cache = None
        self._drop_frozen_items()
        self._epsilon_floor = max(
            self._current_epsilon() * 2.0,
            np.finfo(np.float32).tiny,
        )
        logger.info(
            "3D path exceeded %d vertices; reduced %d -> %d and raised "
            "tolerance to %.6g. Fine detail on the retained path is lost.",
            MAX_FROZEN_VERTICES,
            before,
            self._frozen_len,
            self._epsilon_floor,
        )

    def _refresh_frozen_items(self):
        """Draw the frozen path as a chain of bounded, independent segments.

        Only the newest segment is re-uploaded as the path grows, so the cost of
        a freeze is bounded by the segment size rather than by how long the
        session has been running. Closed segments are never touched again.
        """
        if self._frozen_len < 2:
            return

        if self._has_w and self._w_range_moved():
            self._colored_w_min = self._w_min
            self._colored_w_max = self._w_max
            for index in range(len(self._frozen_spans)):
                self._upload_span(index)

        while True:
            if not self._frozen_spans:
                self._frozen_spans.append([0, 0])
                self.frozen_items.append(None)

            start, end = self._frozen_spans[-1]
            limit = min(start + FROZEN_SEGMENT_VERTICES, self._frozen_len)
            if limit > end:
                self._frozen_spans[-1][1] = limit
                self._upload_span(len(self._frozen_spans) - 1)

            filled = self._frozen_spans[-1][1]
            if filled - start < FROZEN_SEGMENT_VERTICES or filled >= self._frozen_len:
                return

            # Overlap by one vertex so adjacent segments stay continuous.
            self._frozen_spans.append([filled - 1, filled - 1])
            self.frozen_items.append(None)

    def _upload_span(self, index):
        start, end = self._frozen_spans[index]
        if end - start < 2:
            return

        pts = np.array(self._frozen_pts[start:end])
        if self._has_w:
            colors = self._map_w_colors(self._frozen_w[start:end])
        else:
            colors = GOLDEN_PATH_COLOR

        item = self.frozen_items[index]
        if item is None:
            item = gl.GLLinePlotItem(
                pos=pts, color=colors,
                width=self._line_width, antialias=True)
            self.addItem(item)
            self.frozen_items[index] = item
            return
        item.setData(pos=pts, color=colors)

    def _refresh_tail_item(self):
        if self._tail_len < 2:
            if self.tail_item is not None:
                self.tail_item.setVisible(False)
            return

        pts = np.array(self._tail_pts[:self._tail_len])
        if self._has_w:
            colors = self._map_w_colors(self._tail_w[:self._tail_len])
        else:
            colors = GOLDEN_PATH_COLOR

        if self.tail_item is None:
            self.tail_item = gl.GLLinePlotItem(
                pos=pts, color=colors,
                width=self._line_width, antialias=True)
            self.addItem(self.tail_item)
            return
        self.tail_item.setData(pos=pts, color=colors)
        self.tail_item.setVisible(True)

    def _w_range_moved(self):
        """True when W has grown enough to be worth recolouring the whole path."""
        if not np.isfinite(self._colored_w_min):
            return True
        span = max(self._w_max - self._w_min, 1e-12)
        grew = max(
            self._colored_w_min - self._w_min,
            self._w_max - self._colored_w_max,
        )
        return grew / span > W_RANGE_RECOLOR_FRACTION

    def _map_w_colors(self, w_values):
        span = self._w_max - self._w_min
        if span > 1e-12:
            normalized = (w_values - self._w_min) / span
        else:
            normalized = np.full(len(w_values), 0.5, dtype=np.float32)
        return _w_colormap().map(normalized, mode="float")

    def set_view_scale(self, scale):
        """Scale the complete 3D scene by changing camera distance."""
        self.view_scale = max(0.25, min(4.0, float(scale)))
        self.setCameraPosition(distance=DEFAULT_CAMERA_DISTANCE / self.view_scale)
        if getattr(self, "_raw_len", 0) > 1 and self._maybe_refine_for_view():
            self._refresh_frozen_items()
            self._refresh_tail_item()

    def wheelEvent(self, event):
        """Keep the bottom scale slider synchronized with wheel zoom."""
        super().wheelEvent(event)
        distance = max(float(self.opts.get("distance", DEFAULT_CAMERA_DISTANCE)), 1e-6)
        scale = max(0.25, min(4.0, DEFAULT_CAMERA_DISTANCE / distance))
        self.set_view_scale(scale)
        self.viewScaleChanged.emit(self.view_scale)
        self._refresh_hover()

    def mouseMoveEvent(self, event):
        """Pick and display the sampled path point nearest the mouse."""
        super().mouseMoveEvent(event)
        self._last_hover_pos = event.position()
        self._refresh_hover()

    def leaveEvent(self, event):
        self._last_hover_pos = None
        self._hide_hover()
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_coordinate_label()
        if getattr(self, "_raw_len", 0) > 1 and self._maybe_refine_for_view():
            self._refresh_frozen_items()
            self._refresh_tail_item()
            self._refresh_hover()

    def _refresh_hover(self):
        if self._last_hover_pos is None:
            self._hide_hover()
            return

        mouse_x = float(self._last_hover_pos.x())
        mouse_y = float(self._last_hover_pos.y())
        points, w_values = self._hover_candidates(mouse_x, mouse_y)
        if len(points) == 0:
            self._hide_hover()
            return

        index = nearest_projected_point_index(
            self._project(points), mouse_x, mouse_y,
        )
        if index is None:
            self._hide_hover()
            return

        point = points[index]
        pos = np.asarray([point], dtype=np.float32)
        if self.hover_item is None:
            self.hover_item = gl.GLScatterPlotItem(
                pos=pos,
                color=(1.0, 1.0, 1.0, 1.0),
                size=HOVER_MARKER_SIZE_PX,
                pxMode=True,
            )
            self.addItem(self.hover_item)
        else:
            self.hover_item.setData(pos=pos)
            self.hover_item.setVisible(True)

        names = self._coordinate_names
        rows = [
            ("X", names[0] if len(names) > 0 else "X", float(point[0]), "#ff6666"),
            ("Y", names[1] if len(names) > 1 else "Y", float(point[1]), "#66ff66"),
            ("Z", names[2] if len(names) > 2 else "Z", float(point[2]), "#8080ff"),
        ]
        if w_values is not None:
            rows.append((
                "W",
                names[3] if len(names) > 3 else "W",
                float(w_values[index]),
                "#ffa500",
            ))
        html = "".join(
            f"<div><span style='color:{color}; font-weight:600'>{axis}</span> "
            f"<span style='color:#999'>({escape(name)})</span>: {value:.6g}</div>"
            for axis, name, value, color in rows
        )
        self.coordinate_label.setText(html)
        self.coordinate_label.adjustSize()
        self._position_coordinate_label()
        self.coordinate_label.show()
        self.coordinate_label.raise_()

    def _hover_candidates(self, mouse_x, mouse_y):
        """Samples that could be the nearest one, without scanning the path.

        The live tail is always in play. Hover prefers the raw samples so the
        readout stays at capture resolution; blocks are culled by projected
        bounding box first.
        """
        parts = []
        w_parts = []

        if self._tail_len > 0:
            parts.append(self._tail_pts[:self._tail_len])
            if self._has_w:
                w_parts.append(self._tail_w[:self._tail_len])

        if self._raw_len > 0:
            hover_len = self._raw_len
            hover_pts = self._raw_pts
            hover_w = self._raw_w
            blocks = self._blocks_near(mouse_x, mouse_y, self._raw_block_bounds())
        elif self._frozen_len > 0:
            hover_len = self._frozen_len
            hover_pts = self._frozen_pts
            hover_w = self._frozen_w
            blocks = self._candidate_blocks(mouse_x, mouse_y)
        else:
            blocks = ()
            hover_len = 0
            hover_pts = None
            hover_w = None

        for block in blocks:
            start = int(block) * HOVER_BLOCK_SIZE
            end = min(start + HOVER_BLOCK_SIZE, hover_len)
            parts.append(hover_pts[start:end])
            if self._has_w:
                w_parts.append(hover_w[start:end])

        if not parts:
            return np.empty((0, 3), dtype=np.float32), None
        points = np.concatenate(parts) if len(parts) > 1 else parts[0]
        if not self._has_w:
            return points, None
        w_values = np.concatenate(w_parts) if len(w_parts) > 1 else w_parts[0]
        return points, w_values

    def _candidate_blocks(self, mouse_x, mouse_y):
        """Frozen blocks worth examining for this cursor position."""
        return self._blocks_near(mouse_x, mouse_y, self._frozen_block_bounds())

    def _blocks_near(self, mouse_x, mouse_y, bounds):
        if len(bounds) == 0:
            return np.empty(0, dtype=np.intp)

        corners = aabb_corners(bounds)
        projected = self._project(corners.reshape(-1, 3)).reshape(
            len(bounds), 8, 2)
        distances = block_screen_distances_sq(projected, mouse_x, mouse_y)
        near = np.flatnonzero(distances <= PATH_HOVER_DISTANCE_PX ** 2)

        budget = max(1, MAX_HOVER_CANDIDATES // HOVER_BLOCK_SIZE)
        if len(near) > budget:
            near = near[np.argsort(distances[near], kind="stable")[:budget]]
        return near

    def _frozen_block_bounds(self):
        """Per-block bounds, computed once per block as the path grows."""
        cached, result = self._bounds_for_buffer(
            self._frozen_pts, self._frozen_len, self._block_bounds_cache)
        self._block_bounds_cache = cached
        return result

    def _raw_block_bounds(self):
        cached, result = self._bounds_for_buffer(
            self._raw_pts, self._raw_len, self._raw_bounds_cache)
        self._raw_bounds_cache = cached
        return result

    def _bounds_for_buffer(self, pts, length, cached):
        full_blocks = length // HOVER_BLOCK_SIZE
        done = 0 if cached is None else len(cached)
        if done < full_blocks:
            fresh = block_bounds(
                pts[done * HOVER_BLOCK_SIZE:full_blocks * HOVER_BLOCK_SIZE],
                HOVER_BLOCK_SIZE,
            )
            cached = fresh if cached is None else np.concatenate([cached, fresh])

        partial_start = full_blocks * HOVER_BLOCK_SIZE
        if partial_start < length:
            partial = block_bounds(
                pts[partial_start:length],
                HOVER_BLOCK_SIZE,
            )
            if cached is None:
                return cached, partial
            return cached, np.concatenate([cached, partial])
        if cached is not None:
            return cached, cached
        return cached, np.empty((0, 2, 3), dtype=np.float32)

    def _project(self, points):
        """Project XYZ samples into widget pixel coordinates."""
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        viewport = (0, 0, width, height)
        transform = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
        matrix = np.asarray(transform.copyDataTo(), dtype=float).reshape(4, 4)

        homogeneous = np.column_stack([
            np.asarray(points, dtype=float),
            np.ones(len(points)),
        ])
        clip = homogeneous @ matrix.T
        clip_w = clip[:, 3]
        valid = np.isfinite(clip).all(axis=1) & (clip_w > 1e-12)
        projected = np.full((len(clip), 2), np.nan, dtype=float)
        if valid.any():
            ndc = clip[valid, :3] / clip_w[valid, None]
            in_depth = (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
            valid_indices = np.flatnonzero(valid)[in_depth]
            ndc = ndc[in_depth]
            projected[valid_indices, 0] = (ndc[:, 0] + 1.0) * width * 0.5
            projected[valid_indices, 1] = (1.0 - ndc[:, 1]) * height * 0.5
        return projected

    def _hide_hover(self):
        self.coordinate_label.hide()
        if self.hover_item is not None:
            self.hover_item.setVisible(False)

    def _position_coordinate_label(self):
        margin = 10
        self.coordinate_label.move(
            max(margin, self.width() - self.coordinate_label.width() - margin),
            margin,
        )

    def _fit_grid(self):
        if self.grid_item is None or self._extent_min is None:
            return

        x_min = min(float(self._extent_min[0]), 0)
        x_max = max(float(self._extent_max[0]), 0)
        y_min = min(float(self._extent_min[1]), 0)
        y_max = max(float(self._extent_max[1]), 0)
        margin = max(x_max - x_min, y_max - y_min, 10) * 0.2
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        sx = x_max - x_min
        sy = y_max - y_min
        grid_span = max(sx, sy)
        spacing = max(
            1,
            round(
                grid_span / 15,
                -int(np.floor(np.log10(max(grid_span / 15, 0.1)))),
            ),
        )
        self.grid_item.setSize(sx, sy)
        self.grid_item.setSpacing(spacing, spacing)
        self.grid_item.resetTransform()
        self.grid_item.translate((x_max + x_min) / 2, (y_max + y_min) / 2, 0)

    def _update_cursor(self, x, y, z):
        pos = np.array([[x, y, z]], dtype=np.float32)
        if self.cursor_item is None:
            self.cursor_item = gl.GLScatterPlotItem(
                pos=pos,
                color=(1.0, 0.33, 0.33, 1.0),
                size=CURSOR_MARKER_SIZE_PX,
                pxMode=True,
            )
            self.addItem(self.cursor_item)
        else:
            self.cursor_item.setData(pos=pos)
