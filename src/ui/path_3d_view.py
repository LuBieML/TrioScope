import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl


GOLDEN_PATH_COLOR = (1.0, 0.843, 0.0, 1.0)  # #FFD700
DEFAULT_CAMERA_DISTANCE = 50.0
CURSOR_MARKER_SIZE_PX = 8.0


class Path3DView(gl.GLViewWidget):
    """OpenGL path view for XYZ and XYZW plot modes."""

    viewScaleChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundColor("#0A0A0A")
        self.view_scale = 1.0
        self.setCameraPosition(distance=DEFAULT_CAMERA_DISTANCE)
        self.grid_item = None
        self.line_item = None
        self.cursor_item = None
        self.line_segments = None
        self.colorbar_items = []
        self._cb_min_label = None
        self._cb_max_label = None
        self._cb_title_label = None

    def setup_view(self, mode, enabled_traces):
        for item in self.items[:]:
            self.removeItem(item)
        self.line_item = None
        self.cursor_item = None
        self.line_segments = None
        self.colorbar_items = []

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
        pts = self._downsample_points(x_vals, y_vals, z_vals)
        if self.line_item is None:
            self.line_item = gl.GLLinePlotItem(
                pos=pts, color=GOLDEN_PATH_COLOR,
                width=line_width, antialias=True)
            self.addItem(self.line_item)
        else:
            self.line_item.setData(pos=pts)

        self._fit_grid(x_vals, y_vals)
        self._update_cursor(float(x_vals[-1]), float(y_vals[-1]), float(z_vals[-1]))

    def render_xyzw(self, x_vals, y_vals, z_vals, w_vals, line_width=1):
        x_ds, y_ds, z_ds, w_ds = self._downsample_arrays(
            x_vals, y_vals, z_vals, w_vals)
        pts = np.column_stack([x_ds, y_ds, z_ds]).astype(np.float32)

        w_min, w_max = float(w_ds.min()), float(w_ds.max())
        if w_max - w_min > 1e-12:
            w_norm = (w_ds - w_min) / (w_max - w_min)
        else:
            w_norm = np.full_like(w_ds, 0.5)
        colors = pg.colormap.get("turbo").map(w_norm, mode="float")

        if self.line_item is None:
            self.line_item = gl.GLScatterPlotItem(
                pos=pts, color=colors, size=2.5, pxMode=True)
            self.addItem(self.line_item)
            if len(pts) > 1:
                self.line_segments = gl.GLLinePlotItem(
                    pos=pts, color=(colors[:-1] + colors[1:]) / 2.0,
                    width=line_width, antialias=True)
                self.addItem(self.line_segments)
        else:
            self.line_item.setData(pos=pts, color=colors, size=2.5)
            if self.line_segments is not None and len(pts) > 1:
                self.line_segments.setData(
                    pos=pts, color=(colors[:-1] + colors[1:]) / 2.0)

        self._fit_grid(x_vals, y_vals)
        self._update_cursor(float(x_vals[-1]), float(y_vals[-1]), float(z_vals[-1]))
        self.update_colorbar_range(w_min, w_max)

    def clear_path(self):
        self.line_item = None
        self.cursor_item = None
        self.line_segments = None

    def set_view_scale(self, scale):
        """Scale the complete 3D scene by changing camera distance."""
        self.view_scale = max(0.25, min(4.0, float(scale)))
        self.setCameraPosition(distance=DEFAULT_CAMERA_DISTANCE / self.view_scale)

    def wheelEvent(self, event):
        """Keep the bottom scale slider synchronized with wheel zoom."""
        super().wheelEvent(event)
        distance = max(float(self.opts.get("distance", DEFAULT_CAMERA_DISTANCE)), 1e-6)
        scale = max(0.25, min(4.0, DEFAULT_CAMERA_DISTANCE / distance))
        self.set_view_scale(scale)
        self.viewScaleChanged.emit(self.view_scale)

    @staticmethod
    def _downsample_arrays(*arrays, max_points=8000):
        n = len(arrays[0])
        if n <= max_points:
            return arrays
        step = n // max_points
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return tuple(arr[idx] for arr in arrays)

    def _downsample_points(self, x_vals, y_vals, z_vals):
        x_ds, y_ds, z_ds = self._downsample_arrays(x_vals, y_vals, z_vals)
        return np.column_stack([x_ds, y_ds, z_ds]).astype(np.float32)

    def _fit_grid(self, x_vals, y_vals):
        if self.grid_item is None or len(x_vals) == 0:
            return

        x_min = min(float(x_vals.min()), 0)
        x_max = max(float(x_vals.max()), 0)
        y_min = min(float(y_vals.min()), 0)
        y_max = max(float(y_vals.max()), 0)
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
