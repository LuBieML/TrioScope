import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QCheckBox, QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
import pyqtgraph as pg

from plot.viewbox import ScopeViewBox
from ui.theme import CURSOR_COLORS

class CompareWindow(QMainWindow):
    """Separate window for comparing two or more live traces on one plot.

    Each trace draws on its own ViewBox stacked on a shared main PlotItem so
    Y-scales stay independent (traces can have different units). X axis is
    shared. Data is pushed from the main app on every timer tick.
    """

    closed = Signal()

    MIN_TRACES = 2

    def __init__(self, traces, fft_mode, parent=None, single_trace=False):
        super().__init__(parent, Qt.Window)
        self.single_trace = single_trace
        self.setWindowTitle("Trace Scope" if single_trace else "Compare Scopes")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumSize(760, 480)
        self.resize(1180, 720)
        self.fft_mode = fft_mode
        self.traces = list(traces)  # TraceControl refs
        self.trace_keys = [t.get_display_name() for t in self.traces]

        central = QWidget()
        central.setStyleSheet("background-color: #0A0A0A;")
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top bar: title + close hint
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        names = ", ".join(t.get_display_name() for t in self.traces)
        mode = "FFT" if fft_mode else "time-domain"
        if single_trace:
            title_text = f"{mode}: {names}"
        else:
            title_names = names if len(self.traces) <= 3 else f"{len(self.traces)} scopes"
            title_text = f"Comparing {mode}: {title_names}"
        title = QLabel(title_text)
        title.setToolTip(names)
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        title.setStyleSheet("color: #d4d4d4; font-size: 10pt; font-weight: bold;")
        top.addWidget(title, 1)
        self.btn_link_y = QPushButton("\U0001f517 Unify Y")
        self.btn_link_y.setCheckable(True)
        self.btn_link_y.setToolTip(
            "Link Y axes of all compared traces to a shared range")
        self.btn_link_y.setStyleSheet(
            "QPushButton { background-color: #2e2e2e; color: #d4d4d4; "
            "padding: 4px 10px; border: 1px solid #555; } "
            "QPushButton:checked { background-color: #03DAC6; color: #000; "
            "font-weight: bold; } "
            "QPushButton:hover { background-color: #3a3a3a; }"
        )
        self.btn_link_y.toggled.connect(self._on_link_y_toggled)
        top.addWidget(self.btn_link_y)
        self.btn_link_y.setVisible(len(self.traces) > 1)
        self.btn_cursors = QPushButton("\u2295 Cursors")
        self.btn_cursors.setCheckable(True)
        self.btn_cursors.setToolTip("Show draggable C1/C2 measurement cursors")
        self.btn_cursors.setStyleSheet(
            "QPushButton { background-color: #2e2e2e; color: #d4d4d4; "
            "padding: 4px 10px; border: 1px solid #555; } "
            "QPushButton:checked { background-color: #3a3a5c; color: #fff; "
            "border: 1px solid #FFD700; font-weight: bold; } "
            "QPushButton:hover { background-color: #3a3a3a; }"
        )
        self.btn_cursors.toggled.connect(self._toggle_cursors)
        top.addWidget(self.btn_cursors)
        hint = QLabel("Esc closes window")
        hint.setStyleSheet("color: #888888; font-size: 9pt; margin-left: 10px;")
        top.addWidget(hint)
        layout.addLayout(top)

        self.cursor_readout = QLabel("")
        self.cursor_readout.setTextFormat(Qt.RichText)
        self.cursor_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cursor_readout.setWordWrap(True)
        self.cursor_readout.setStyleSheet(
            "QLabel { background-color: #151515; color: #d4d4d4; "
            "border: 1px solid #333; padding: 6px; "
            "font-family: Consolas; font-size: 9pt; }"
        )
        self.cursor_readout.hide()
        layout.addWidget(self.cursor_readout)

        # Graphics layout widget holding the single overlay plot
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground('#0A0A0A')
        layout.addWidget(self.glw, 1)

        # Main plot (owns the X axis and the first Y axis). Uses the
        # oscilloscope-style ViewBox so right-drag rubber-band zoom,
        # wheel-zoom and double-click reset work the same as the main window.
        self._main_vb = ScopeViewBox()
        self.main_plot = self.glw.addPlot(row=0, col=0, viewBox=self._main_vb)
        self._main_vb.doubleClicked.connect(self._reset_view)
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        self.main_plot.setLabel(
            'bottom',
            'Frequency (Hz)' if fft_mode else 'Time (seconds)',
            color='#d4d4d4',
        )

        # Primary axis config — color-code to first trace
        first_color = self.traces[0].get_color() if self.traces else '#d4d4d4'
        self.main_plot.getAxis('left').setPen(pg.mkPen(first_color))
        self.main_plot.getAxis('left').setTextPen(pg.mkPen(first_color))
        self.main_plot.setLabel('left', self.traces[0].get_display_name(),
                                color=first_color)

        # Primary curve goes on main_plot's ViewBox
        width = 1
        self.curves = []
        self.viewboxes = [self.main_plot.vb]
        self.axes = [self.main_plot.getAxis('left')]
        pen0 = pg.mkPen(first_color, width=width)
        curve0 = self.main_plot.plot(pen=pen0)
        curve0.setClipToView(True)
        curve0.setDownsampling(auto=True, method='subsample')
        self.curves.append(curve0)

        # Extra traces: one extra ViewBox + right-side AxisItem per trace
        # Standard pyqtgraph multi-axis pattern.
        for i, trace in enumerate(self.traces[1:], start=1):
            color = trace.get_color()
            vb = pg.ViewBox()
            axis = pg.AxisItem('right')
            axis.setPen(pg.mkPen(color))
            axis.setTextPen(pg.mkPen(color))
            axis.setLabel(trace.get_display_name(), color=color)
            # Append axis into the layout on the right side of the main plot
            self.glw.addItem(axis, row=0, col=i + 1)
            self.glw.scene().addItem(vb)
            axis.linkToView(vb)
            vb.setXLink(self.main_plot.vb)
            curve = pg.PlotDataItem(pen=pg.mkPen(color, width=width))
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method='subsample')
            vb.addItem(curve)
            self.curves.append(curve)
            self.viewboxes.append(vb)
            self.axes.append(axis)

        # Sync the extra ViewBoxes' geometry to the main plot's ViewBox
        self.main_plot.vb.sigResized.connect(self._sync_viewboxes)
        self._sync_viewboxes()

        # --- Hover crosshair + coordinate readout ---
        # Vertical line that follows the mouse; per-trace values are read out
        # in a translucent label anchored in the top-left of the plot.
        self._vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#888888', width=1, style=Qt.DashLine),
        )
        self._vline.setZValue(1000)
        self.main_plot.addItem(self._vline, ignoreBounds=True)
        self._vline.hide()

        self._hover_label = pg.TextItem(anchor=(0, 0), color='#d4d4d4',
                                        fill=pg.mkBrush(0, 0, 0, 180))
        self._hover_label.setZValue(1001)
        self.main_plot.addItem(self._hover_label, ignoreBounds=True)
        self._hover_label.hide()

        # Cache of latest x-array + per-trace y-array for interpolation
        self._last_x = None
        self._last_y = [None] * len(self.traces)
        self._cursors_enabled = False
        self._cursor_lines = {}
        self._cursor_pos = {'c1': 0.0, 'c2': 1.0}
        self._cursor_updating = False

        self.main_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _sync_viewboxes(self):
        rect = self.main_plot.vb.sceneBoundingRect()
        for vb in self.viewboxes[1:]:
            vb.setGeometry(rect)
            vb.linkedViewChanged(self.main_plot.vb, vb.XAxis)

    def _on_link_y_toggled(self, checked):
        """Link/unlink all extra ViewBoxes' Y axes to the main one.

        When linked, all traces share the widest Y range (union of individual
        ranges), so they scale together. When unlinked, each trace gets back
        its own auto-ranged Y.
        """
        main_vb = self.main_plot.vb
        if checked:
            # Compute union of current Y ranges across all ViewBoxes
            y_mins, y_maxs = [], []
            for vb in self.viewboxes:
                y_min, y_max = vb.viewRange()[1]
                y_mins.append(y_min)
                y_maxs.append(y_max)
            y_min = min(y_mins)
            y_max = max(y_maxs)
            # Disable auto-range on the driver, fix range to the union, then link
            main_vb.enableAutoRange(axis='y', enable=False)
            main_vb.setYRange(y_min, y_max, padding=0)
            for vb in self.viewboxes[1:]:
                vb.enableAutoRange(axis='y', enable=False)
                vb.setYLink(main_vb)
        else:
            # Break the link and restore per-trace auto-ranging
            for vb in self.viewboxes[1:]:
                vb.setYLink(None)
                vb.enableAutoRange(axis='y', enable=True)
            main_vb.enableAutoRange(axis='y', enable=True)

    def update_data(self, time_arr, params_by_name, fft_freqs=None,
                    fft_magnitudes=None):
        """Push latest data into each curve. For FFT mode, supply freqs+mags."""
        if self.fft_mode:
            if fft_freqs is None or not fft_magnitudes:
                return
            self._last_x = fft_freqs
            for i, (curve, trace) in enumerate(zip(self.curves, self.traces)):
                name = trace.get_display_name()
                mag = fft_magnitudes.get(name)
                if mag is None or len(mag) == 0:
                    continue
                curve.setData(fft_freqs, mag, skipFiniteCheck=True)
                self._last_y[i] = mag
        else:
            if time_arr is None or len(time_arr) == 0:
                return
            self._last_x = time_arr
            for i, (curve, trace) in enumerate(zip(self.curves, self.traces)):
                name = trace.get_display_name()
                values = params_by_name.get(name)
                if values is None or len(values) == 0:
                    continue
                curve.setData(time_arr, values, skipFiniteCheck=True)
                self._last_y[i] = values
        if self._cursors_enabled:
            self._update_cursor_readout()

    def _toggle_cursors(self, checked):
        self._cursors_enabled = checked
        if checked:
            self._init_cursor_positions()
            self._add_cursor_lines()
            self.cursor_readout.show()
            self._update_cursor_readout()
        else:
            self._remove_cursor_lines()
            self.cursor_readout.hide()

    def _init_cursor_positions(self):
        if self._last_x is None or len(self._last_x) == 0:
            self._cursor_pos['c1'] = 0.0
            self._cursor_pos['c2'] = 1.0
            return
        data_min = float(self._last_x[0])
        data_max = float(self._last_x[-1])
        x_min, x_max = self.main_plot.vb.viewRange()[0]
        x_min = max(float(x_min), data_min)
        x_max = min(float(x_max), data_max)
        if x_max <= x_min:
            x_min, x_max = data_min, data_max
        span = x_max - x_min
        self._cursor_pos['c1'] = x_min + span * 0.33
        self._cursor_pos['c2'] = x_min + span * 0.67

    def _add_cursor_lines(self):
        self._remove_cursor_lines()
        for cid in ('c1', 'c2'):
            color = CURSOR_COLORS[cid]
            line = pg.InfiniteLine(
                pos=self._cursor_pos[cid],
                angle=90,
                movable=True,
                pen=pg.mkPen(color, width=1.5, style=Qt.DashLine),
                hoverPen=pg.mkPen(color, width=2.5),
                label=cid.upper(),
                labelOpts={
                    'position': 0.95,
                    'color': color,
                    'fill': pg.mkBrush('#2B2B2BBB'),
                    'movable': True,
                },
            )
            line.setZValue(1200)
            line._cursor_id = cid
            line.sigPositionChanged.connect(self._on_cursor_line_moved)
            self.main_plot.addItem(line, ignoreBounds=True)
            self._cursor_lines[cid] = line

    def _remove_cursor_lines(self):
        for line in self._cursor_lines.values():
            try:
                self.main_plot.removeItem(line)
            except Exception:
                pass
        self._cursor_lines.clear()

    def _on_cursor_line_moved(self, line):
        if self._cursor_updating:
            return
        cid = line._cursor_id
        self._cursor_pos[cid] = float(line.value())
        self._cursor_updating = True
        try:
            other = self._cursor_lines.get(cid)
            if other is not None and other is not line:
                other.setValue(self._cursor_pos[cid])
        finally:
            self._cursor_updating = False
        self._update_cursor_readout()

    def _sample_trace_at(self, y_values, x):
        if self._last_x is None or y_values is None:
            return None
        xs = self._last_x
        if len(xs) == 0 or len(y_values) != len(xs):
            return None
        if x < xs[0] or x > xs[-1]:
            return None
        return float(np.interp(x, xs, y_values))

    def _format_x(self, value):
        if self.fft_mode:
            return f"{value:.6g} Hz"
        return f"{value:.6f} s"

    def _format_delta_x(self, value):
        if self.fft_mode:
            return f"{value:+.6g} Hz"
        return f"{value:+.6f} s"

    def _format_value(self, value):
        if value is None:
            return "---"
        return f"{value:.6g}"

    def _update_cursor_readout(self):
        if not self._cursors_enabled:
            self.cursor_readout.setText("")
            return
        if self._last_x is None or len(self._last_x) == 0:
            msg = "No trace data available." if self.single_trace else "No compare data available."
            self.cursor_readout.setText(msg)
            return

        x1 = self._cursor_pos['c1']
        x2 = self._cursor_pos['c2']
        dx = x2 - x1
        rows = []
        for trace, y_values in zip(self.traces, self._last_y):
            v1 = self._sample_trace_at(y_values, x1)
            v2 = self._sample_trace_at(y_values, x2)
            dv = None if v1 is None or v2 is None else v2 - v1
            color = trace.get_color()
            name = trace.get_display_name()
            rows.append(
                "<tr>"
                f"<td style='padding:2px 10px 2px 0; color:{color}; font-weight:bold;'>{name}</td>"
                f"<td style='padding:2px 10px;'>{self._format_value(v1)}</td>"
                f"<td style='padding:2px 10px;'>{self._format_value(v2)}</td>"
                f"<td style='padding:2px 10px;'>{self._format_value(dv)}</td>"
                "</tr>"
            )

        c1_color = CURSOR_COLORS['c1']
        c2_color = CURSOR_COLORS['c2']
        interval = (
            f"&Delta;f = {self._format_delta_x(dx)}"
            if self.fft_mode else
            f"&Delta;t = {self._format_delta_x(dx)}"
        )
        if not self.fft_mode and abs(dx) > 1e-12:
            interval += f" &nbsp; | &nbsp; f = {1.0 / abs(dx):.6g} Hz"

        html = (
            "<div>"
            f"<span style='color:{c1_color}; font-weight:bold;'>C1</span> "
            f"{self._format_x(x1)} &nbsp; | &nbsp; "
            f"<span style='color:{c2_color}; font-weight:bold;'>C2</span> "
            f"{self._format_x(x2)} &nbsp; | &nbsp; {interval}"
            "</div>"
            "<table cellspacing='0' cellpadding='0'>"
            "<tr style='color:#999;'>"
            "<td style='padding-right:10px;'>Signal</td>"
            "<td style='padding:2px 10px;'>C1</td>"
            "<td style='padding:2px 10px;'>C2</td>"
            "<td style='padding:2px 10px;'>&Delta;</td>"
            "</tr>"
            + "".join(rows) +
            "</table>"
        )
        self.cursor_readout.setText(html)

    def _reset_view(self):
        """Re-enable auto-range on all ViewBoxes after a double-click reset."""
        for vb in self.viewboxes:
            vb.enableAutoRange()
        if self.btn_link_y.isChecked():
            self.btn_link_y.setChecked(False)

    def _on_mouse_moved(self, scene_pos):
        """Update crosshair + coordinate readout from the mouse scene position."""
        vb = self.main_plot.vb
        if not self.main_plot.sceneBoundingRect().contains(scene_pos):
            self._vline.hide()
            self._hover_label.hide()
            return
        mouse_point = vb.mapSceneToView(scene_pos)
        x = mouse_point.x()
        self._vline.setPos(x)
        self._vline.show()

        if self._last_x is None or len(self._last_x) == 0:
            self._hover_label.hide()
            return

        xs = self._last_x
        if x < xs[0] or x > xs[-1]:
            self._hover_label.hide()
            return

        idx = np.searchsorted(xs, x)
        if idx > 0 and (idx == len(xs) or abs(xs[idx - 1] - x) <= abs(xs[idx] - x)):
            idx -= 1

        x_label = ("f" if self.fft_mode else "t")
        x_unit = ("Hz" if self.fft_mode else "s")
        lines = [f"{x_label} = {x:.4g} {x_unit}"]
        for trace, ys in zip(self.traces, self._last_y):
            if ys is None or len(ys) != len(xs):
                continue
            y = float(ys[idx])
            color = trace.get_color()
            name = trace.get_display_name()
            lines.append(
                f"<span style='color:{color};'>{name} = {y:.4g}</span>")
        self._hover_label.setHtml(
            "<div style='font-family:monospace;font-size:9pt;'>"
            + "<br/>".join(lines) + "</div>"
        )
        # Anchor the label just right of the cursor, near the top of the view
        x_range, y_range = vb.viewRange()
        y_top = y_range[1]
        x_span = x_range[1] - x_range[0]
        self._hover_label.setPos(x + x_span * 0.01, y_top)
        self._hover_label.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class _CompareTracePicker(QDialog):
    """Small modal: pick two or more same-type enabled traces to compare."""

    def __init__(self, candidates, fft_mode, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Traces")
        self.setStyleSheet(
            "QDialog { background-color: #1a1a1a; color: #d4d4d4; } "
            "QCheckBox { color: #d4d4d4; padding: 4px; } "
            "QPushButton { background-color: #2e2e2e; color: #d4d4d4; "
            "padding: 6px 14px; border: 1px solid #555; } "
            "QPushButton:hover { background-color: #3a3a3a; }"
        )
        self.fft_mode = fft_mode
        self.candidates = candidates  # list[TraceControl]
        self.checks = []

        layout = QVBoxLayout(self)
        kind = "FFT" if fft_mode else "time-domain"
        header = QLabel(f"Select 2 or more {kind} scopes to overlay:")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        for t in candidates:
            cb = QCheckBox(t.get_display_name())
            cb.setStyleSheet(f"color: {t.get_color()}; font-weight: bold;")
            cb.toggled.connect(self._update_ok_state)
            self.checks.append(cb)
            layout.addWidget(cb)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_ok = QPushButton("Compare")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_ok.setEnabled(False)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def _update_ok_state(self):
        selected_count = sum(cb.isChecked() for cb in self.checks)
        self.btn_ok.setEnabled(selected_count >= CompareWindow.MIN_TRACES)

    def selected_traces(self):
        return [t for t, cb in zip(self.candidates, self.checks)
                if cb.isChecked()]


class TraceWindow(CompareWindow):
    """Single-trace companion window using the compare plot/cursor tools."""

    def __init__(self, trace, fft_mode, parent=None):
        super().__init__([trace], fft_mode, parent=parent, single_trace=True)
