from html import escape
import weakref

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QMessageBox
import pyqtgraph as pg

from plot.viewbox import ScopeViewBox
from ui.compare_window import CompareWindow, _CompareTracePicker, TraceWindow
from ui.path_hover import nearest_xy_point_index
from ui.theme import CURSOR_COLORS
from ui.trace_control import TraceControl
from ui.window_controller import WindowBackedController


class PlotRenderer(WindowBackedController):
    _local_attrs = WindowBackedController._local_attrs | frozenset({
        "_fft_cache", "_fft_window_cache", "_fft_dirty", "_fft_peak_cache",
        "_fft_max_samples", "_last_data_len", "_stats_cache", "_ref_set",
        "_stats_pos_cache", "_last_render_data_len", "_stats_reposition_scheduled",
        "_pending_stats_vbs", "_pending_stats_vb_refs", "_detail_update_scheduled",
        "_pending_detail_vbs", "_pending_detail_vb_refs", "_hover_vlines",
        "_hover_labels", "_last_freqs", "_hover_pending_pos",
        "_hover_update_scheduled",
    })

    def __init__(self, window):
        super().__init__(window)
        self._fft_cache = {}
        self._fft_window_cache = (0, None)
        self._fft_dirty = True
        self._fft_peak_cache = {}
        self._fft_max_samples = 16384
        self._last_data_len = 0
        self._stats_cache = {}
        self._ref_set = {}
        self._stats_pos_cache = {}
        self._last_render_data_len = 0
        self._stats_reposition_scheduled = False
        self._pending_stats_vbs = set()
        self._pending_stats_vb_refs = {}
        self._detail_update_scheduled = False
        self._pending_detail_vbs = set()
        self._pending_detail_vb_refs = {}
        self._hover_vlines = {}
        self._hover_labels = {}
        self._last_freqs = None
        self._hover_pending_pos = None
        self._hover_update_scheduled = False

    def _create_scope_plot(self):
        """Add a PlotItem as a new row in the shared GraphicsLayoutWidget."""
        vb = ScopeViewBox()
        vb.doubleClicked.connect(self.window._on_plot_double_click)
        pi = pg.PlotItem(viewBox=vb)
        # Append as next row in the shared layout (single scene → single repaint)
        self.plot_layout_widget.addItem(pi)
        self.plot_layout_widget.nextRow()
        return pi

    def _on_plot_double_click(self):
        """Re-enable auto-scroll on double-click during capture."""
        if self.is_running:
            self.auto_scroll = True
            self._xy_auto_range = True
            self._update_auto_scroll_button()

    def _recreate_subplots(self):
        """Recreate subplots — one row per enabled trace for independent Y-scales.
        Each trace gets its own left Y-axis, color-coded. X-axes are linked.
        In XY mode, a single plot shows trace1 vs trace2."""
        # Clear all PlotItems from the shared layout (keeps the widget itself).
        # clear() removes items but doesn't reset the row/col cursor — reset manually
        # so the next addItem() starts at (0, 0) instead of after the old positions.
        self.plot_layout_widget.clear()
        self.plot_layout_widget.ci.currentRow = 0
        self.plot_layout_widget.ci.currentCol = 0
        self.plot_items = {}
        self.curves = {}
        self.ref_curves = {}
        self.stats_texts = {}
        self._ref_set = {}
        self._stats_pos_cache = {}
        self._hover_vlines.clear()
        self._hover_labels.clear()
        self._stats_cache = {}
        self._cursor_lines_c1.clear()
        self._cursor_lines_c2.clear()
        self._xy_auto_range = True

        enabled_traces = self.get_enabled_traces()

        if not enabled_traces:
            pi = self._create_scope_plot()
            self._configure_plot(pi, show_xlabel=True)
            self.plot_items['empty'] = pi
            self._add_hover_elements_to_plot(pi, 'empty')
            return

        # XY mode: single 2D plot with first two traces as X and Y
        if self.plot_mode == 'xy':
            self._update_path_info_label()
            if len(enabled_traces) < 2:
                pi = self._create_scope_plot()
                self._configure_plot(pi, show_xlabel=True)
                self.plot_items['empty'] = pi
                self._add_hover_elements_to_plot(pi, 'empty')
                return

            pi = self._create_scope_plot()
            vb = pi.getViewBox()
            vb.setBackgroundColor(self.plot_bg_color)
            pi.showGrid(x=True, y=True, alpha=self.grid_alpha)
            vb.uniform_zoom = True
            vb.setAspectLocked(True)
            x_trace = enabled_traces[0]
            y_trace = enabled_traces[1]
            pi.setLabel('bottom', x_trace.get_display_name(),
                        color=x_trace.get_color())
            pi.setLabel('left', y_trace.get_display_name(),
                        color=y_trace.get_color())
            pi.getAxis('bottom').setPen(pg.mkPen(x_trace.get_color()))
            pi.getAxis('bottom').setTextPen(pg.mkPen(x_trace.get_color()))
            pi.getAxis('left').setPen(pg.mkPen(y_trace.get_color()))
            pi.getAxis('left').setTextPen(pg.mkPen(y_trace.get_color()))
            pi.disableAutoRange()
            vb.sigRangeChangedManually.connect(self.window._on_manual_range_change)
            vb.sigRangeChangedManually.connect(self.window._on_xy_manual_zoom)
            self.plot_items['xy'] = pi
            self._add_hover_elements_to_plot(pi, 'xy')
            return

        # XYZ/XYZW mode: 3D OpenGL view
        if self.plot_mode in ('xyz', 'xyzw'):
            self._update_path_info_label()
            self._setup_3d_view()
            return

        num_subplots = len(enabled_traces)

        for row, trace in enumerate(enabled_traces):
            pi = self._create_scope_plot()
            is_last = (row == num_subplots - 1)
            if trace.is_fft():
                self._configure_fft_plot(pi, show_xlabel=is_last)
            else:
                self._configure_plot(pi, show_xlabel=is_last)

            # Color-code left Y-axis to match trace
            color = trace.get_color()
            pi.getAxis('left').setPen(pg.mkPen(color))
            pi.getAxis('left').setTextPen(pg.mkPen(color))

            self.plot_items[id(trace)] = pi
            self._add_hover_elements_to_plot(pi, id(trace))

        # Link X-axes for synchronized scrolling (partitioned by time/FFT)
        self._update_x_links()

        # Re-add cursor lines if cursors are active
        if self._cursors_enabled:
            self._add_cursors_to_plots()

    def _configure_plot(self, plot_item, show_xlabel=True):
        """Configure a PlotItem with standard settings"""
        vb = plot_item.getViewBox()
        vb.setBackgroundColor(self.plot_bg_color)
        plot_item.showGrid(x=True, y=True, alpha=self.grid_alpha)
        if show_xlabel:
            plot_item.setLabel('bottom', 'Time (seconds)', color='#d4d4d4')
        else:
            plot_item.setLabel('bottom', '')
        plot_item.setLabel('left', '')

        # Fix left-axis width so all plots align regardless of label width
        plot_item.getAxis('left').setWidth(65)

        # Y auto-range follows visible data
        plot_item.enableAutoRange(axis='y', enable=True)
        # NOTE: setAutoVisible(y=True) forces a Y-bounds rescan of visible X data
        # on every pan tick — ~100 Hz × N curves × N points. Leave it off;
        # Y still autoranges on data updates via enableAutoRange.

        # Disable auto-scroll when user manually interacts
        vb.sigRangeChangedManually.connect(self.window._on_manual_range_change)

        # Reposition stats text and update dot visibility when view range changes
        vb.sigRangeChanged.connect(self.window._reposition_stats_texts)
        vb.sigRangeChanged.connect(self.window._update_curve_detail)

    def _configure_fft_plot(self, plot_item, show_xlabel=True):
        """Configure a PlotItem for FFT spectrum display."""
        vb = plot_item.getViewBox()
        vb.setBackgroundColor(self.plot_bg_color)
        plot_item.showGrid(x=True, y=True, alpha=self.grid_alpha)
        if show_xlabel:
            plot_item.setLabel('bottom', 'Frequency (Hz)', color='#d4d4d4')
        else:
            plot_item.setLabel('bottom', '')
        plot_item.setLabel('left', 'Magnitude', color='#d4d4d4')
        plot_item.getAxis('left').setWidth(65)
        plot_item.enableAutoRange(axis='y', enable=True)
        # NOTE: setAutoVisible(y=True) forces a Y-bounds rescan of visible X data
        # on every pan tick — ~100 Hz × N curves × N points. Leave it off;
        # Y still autoranges on data updates via enableAutoRange.
        vb.sigRangeChanged.connect(self.window._reposition_stats_texts)

    def _on_manual_range_change(self, _changes):
        """When user manually pans/zooms, disable auto-scroll"""
        if self.is_running and self.auto_scroll:
            self.auto_scroll = False
            self._update_auto_scroll_button()

    def _reposition_stats_texts(self, vb):
        """Debounced: coalesce pan events via a 0ms timer, then reposition."""
        self._pending_stats_vbs.add(id(vb))
        self._pending_stats_vb_refs[id(vb)] = vb
        if not self._stats_reposition_scheduled:
            self._stats_reposition_scheduled = True
            QTimer.singleShot(16, self.window._flush_stats_reposition)

    def _flush_stats_reposition(self):
        self._stats_reposition_scheduled = False
        vbs = [self._pending_stats_vb_refs[i] for i in self._pending_stats_vbs
               if i in self._pending_stats_vb_refs]
        self._pending_stats_vbs.clear()
        self._pending_stats_vb_refs.clear()
        for vb in vbs:
            for trace_id, pi in self.plot_items.items():
                if pi.getViewBox() is vb:
                    view_range = vb.viewRange()
                    new_pos = (view_range[0][1], view_range[1][1])
                    if trace_id in self.stats_texts:
                        if self._stats_pos_cache.get(trace_id) != new_pos:
                            self.stats_texts[trace_id].setPos(*new_pos)
                            self._stats_pos_cache[trace_id] = new_pos
                    if trace_id in self._hover_labels:
                        self._hover_labels[trace_id].setPos(*new_pos)

    def _update_curve_detail(self, vb):
        """Debounced: coalesce dot-detail updates to end of pan burst."""
        if self.plot_mode in ('xy', 'xyz'):
            return
        if not self.curves:
            return
        self._pending_detail_vbs.add(id(vb))
        self._pending_detail_vb_refs[id(vb)] = vb
        if not self._detail_update_scheduled:
            self._detail_update_scheduled = True
            QTimer.singleShot(50, self.window._flush_curve_detail)

    def _flush_curve_detail(self):
        self._detail_update_scheduled = False
        vbs = [self._pending_detail_vb_refs[i] for i in self._pending_detail_vbs
               if i in self._pending_detail_vb_refs]
        self._pending_detail_vbs.clear()
        self._pending_detail_vb_refs.clear()
        if self.plot_mode in ('xy', 'xyz') or not self.curves:
            return
        for vb in vbs:
            self._do_update_curve_detail(vb)

    def _do_update_curve_detail(self, vb):
        view_range = vb.viewRange()
        visible_span = view_range[0][1] - view_range[0][0]
        for trace_id, pi in self.plot_items.items():
            if pi.getViewBox() is not vb or trace_id not in self.curves:
                continue
            # Skip FFT traces — dot detail only applies to time-domain
            trace_obj = next((t for t in self.traces if id(t) == trace_id), None)
            if trace_obj is not None and trace_obj.is_fft():
                continue
            curve = self.curves[trace_id]
            xData = curve.xData
            if xData is None or len(xData) < 2:
                continue
            sample_dt = xData[1] - xData[0]
            if sample_dt <= 0:
                continue
            visible_points = visible_span / sample_dt
            want_dots = visible_points <= 2000
            # Track state to avoid redundant symbol toggling (each call triggers repaint)
            had_dots = getattr(curve, '_has_dots', False)
            if want_dots and not had_dots:
                curve.setDownsampling(ds=1)
                color = curve.opts['pen'].color()
                curve.setSymbol('o')
                curve.setSymbolSize(4)
                curve.setSymbolBrush(color)
                curve.setSymbolPen(None)
                curve._has_dots = True
            elif not want_dots and had_dots:
                curve.setSymbol(None)
                curve.setDownsampling(auto=True, method='subsample')
                curve._has_dots = False

    def _on_xy_manual_zoom(self, _changes):
        """When user manually pans/zooms in XY mode, stop auto-fitting."""
        self._xy_auto_range = False
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()

    def _add_hover_elements_to_plot(self, plot_item, plot_key):
        """Add hover crosshair and label to a plot item."""
        vline = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#888888', width=1, style=Qt.DashLine),
        )
        vline.setZValue(1000)
        plot_item.addItem(vline, ignoreBounds=True)
        vline.hide()
        self._hover_vlines[plot_key] = vline

        label = pg.TextItem(anchor=(1, 0), color='#d4d4d4', fill=None)
        label.setZValue(1001)
        plot_item.addItem(label, ignoreBounds=True)
        label.hide()
        self._hover_labels[plot_key] = label

    def _on_main_plot_mouse_moved(self, scene_pos):
        """Debounced: coalesce mouse-move events (can exceed 100Hz on
        high-polling-rate mice) into at most one hover update per frame."""
        self._hover_pending_pos = scene_pos
        if not self._hover_update_scheduled:
            self._hover_update_scheduled = True
            QTimer.singleShot(16, self.window._flush_hover_update)

    def _flush_hover_update(self):
        self._hover_update_scheduled = False
        scene_pos = self._hover_pending_pos
        if scene_pos is None:
            return
        self._do_hover_update(scene_pos)

    def _do_hover_update(self, scene_pos):
        """Update crosshair and value readout on all main window plots."""
        if self.plot_mode in ('xyz', 'xyzw') or not self.plot_layout_widget.isVisible():
            return

        active_plot_key = None
        active_pi = None
        for key, pi in self.plot_items.items():
            if pi.sceneBoundingRect().contains(scene_pos):
                active_plot_key = key
                active_pi = pi
                break

        if active_pi is None:
            for vline in self._hover_vlines.values(): vline.hide()
            for label in self._hover_labels.values(): label.hide()
            self._hide_xy_path_hover()
            return

        vb_active = active_pi.getViewBox()
        mouse_point = vb_active.mapSceneToView(scene_pos)
        x = mouse_point.x()

        if self.plot_mode == 'xy':
            self._update_xy_path_hover(
                active_plot_key, active_pi, mouse_point.x(), mouse_point.y())
            return

        for key, pi in self.plot_items.items():
            vline = self._hover_vlines.get(key)
            label = self._hover_labels.get(key)
            if not vline or not label:
                continue

            # Update crosshair and value readout on all plots natively
            vb = pi.getViewBox()
            vline.setPos(x)
            vline.show()

            if self.accumulated_data is None or len(self.accumulated_data['time']) == 0:
                label.hide()
                continue

            html_lines = []
            if self.plot_mode != 'xy':  # time mode
                trace = next((t for t in self.traces if id(t) == key), None)
                if not trace:
                    continue

                if trace.is_fft():
                    freqs = self._last_freqs
                    cached = self._fft_cache.get(key)
                    if freqs is None or not cached or 'magnitude' not in cached:
                        continue
                    mags = cached['magnitude']
                    if x < freqs[0] or x > freqs[-1]: continue
                    idx = np.searchsorted(freqs, x)
                    html_lines.append(f"f = {freqs[idx]:.4g} Hz")
                    html_lines.append(f"<span style='color:{trace.get_color()};'>Mag: {mags[idx]:.4g}</span>")
                else:
                    time_arr = self.accumulated_data['time']
                    pname = trace.get_display_name()
                    values = self.accumulated_data['params'].get(pname)
                    if values is None or x < time_arr[0] or x > time_arr[-1]: continue
                    idx = np.searchsorted(time_arr, x)
                    html_lines.append(f"t = {time_arr[idx]:.4g} s")
                    html_lines.append(f"<span style='color:{trace.get_color()};'>Cur: {values[idx]:.4f}</span>")

            if html_lines:
                html = (
                    "<div style='font-family: Segoe UI; font-size: 8pt; text-align: right;'>"
                    "<br><br>" + "<br/>".join(html_lines) + "</div>"
                )
                # setHtml triggers a rich-text re-layout — skip when unchanged
                if html != getattr(label, '_last_html', None):
                    label.setHtml(html)
                    label._last_html = html
                x_range, y_range = vb.viewRange()
                x_right = x_range[1]
                y_top = y_range[1]
                label.setPos(x_right, y_top)
                label.show()
            else:
                label.hide()

    def _update_xy_path_hover(self, active_key, plot_item, mouse_x, mouse_y):
        """Snap the XY readout to the nearest rendered path sample."""
        for vline in self._hover_vlines.values():
            vline.hide()
        for key, label in self._hover_labels.items():
            if key != active_key:
                label.hide()

        label = self._hover_labels.get(active_key)
        path_curve = self.curves.get('xy_path')
        if label is None or path_curve is None:
            self._hide_xy_path_hover()
            return
        x_values = path_curve.xData
        y_values = path_curve.yData
        if x_values is None or y_values is None:
            self._hide_xy_path_hover()
            return

        vb = plot_item.getViewBox()
        x_range, y_range = vb.viewRange()
        scene_rect = vb.sceneBoundingRect()
        x_units_per_pixel = (x_range[1] - x_range[0]) / max(scene_rect.width(), 1.0)
        y_units_per_pixel = (y_range[1] - y_range[0]) / max(scene_rect.height(), 1.0)
        index = nearest_xy_point_index(
            x_values,
            y_values,
            mouse_x,
            mouse_y,
            x_units_per_pixel,
            y_units_per_pixel,
        )
        if index is None:
            self._hide_xy_path_hover()
            return

        x_value = float(x_values[index])
        y_value = float(y_values[index])
        enabled = self.get_enabled_traces()
        if len(enabled) < 2:
            self._hide_xy_path_hover()
            return

        x_name = escape(enabled[0].get_display_name())
        y_name = escape(enabled[1].get_display_name())
        label.setHtml(
            "<div style='font-family: Consolas; font-size: 9pt; text-align: right;"
            " background-color: rgba(26,26,46,220);'>"
            f"<span style='color:{enabled[0].get_color()}; font-weight:600'>X</span> "
            f"<span style='color:#999'>({x_name})</span>: {x_value:.6g}<br/>"
            f"<span style='color:{enabled[1].get_color()}; font-weight:600'>Y</span> "
            f"<span style='color:#999'>({y_name})</span>: {y_value:.6g}</div>"
        )
        label.setPos(x_range[1], y_range[1])
        label.show()

        marker = self.curves.get('xy_hover')
        if marker is not None:
            marker.setData([x_value], [y_value])

    def _hide_xy_path_hover(self):
        label = self._hover_labels.get('xy')
        if label is not None:
            label.hide()
        marker = self.curves.get('xy_hover')
        if marker is not None:
            marker.setData([], [])

    def _open_trace_window(self, trace):
        """Open a resizable companion window for one enabled trace."""
        if trace not in self.traces or trace.parent() is None:
            return

        trace_name = trace.get_display_name()
        if not trace.is_enabled():
            QMessageBox.information(
                self.window, "Trace Window",
                "Enable this trace before opening it in a separate window.")
            return

        if self.accumulated_data is None or not self.accumulated_data['params']:
            QMessageBox.information(
                self.window, "Trace Window",
                "Start a capture first - the trace window needs live data.")
            return

        if trace_name not in self.accumulated_data['params']:
            QMessageBox.information(
                self.window, "Trace Window",
                f"No captured data is available for {trace_name}.")
            return

        self._trace_window_counter += 1
        trace_window = TraceWindow(trace, trace.is_fft(), parent=self.window,
                                   line_width=self.line_width)
        trace_window.setWindowTitle(
            f"Trace Scope {self._trace_window_counter}: {trace_name}")
        trace_window.closed.connect(
            lambda w=trace_window: self._on_compare_closed(w))
        self._compare_windows.append(trace_window)
        trace_window.show()
        trace_window.raise_()
        trace_window.activateWindow()
        self._push_compare_data(trace_window)

    def _open_compare(self):
        """Open another resizable compare window with selected live scopes."""
        if self.accumulated_data is None or not self.accumulated_data['params']:
            QMessageBox.information(
                self.window, "Compare",
                "Start a capture first — compare needs live data.")
            return

        enabled = self.get_enabled_traces()
        if len(enabled) < 2:
            QMessageBox.information(
                self.window, "Compare",
                "Enable at least 2 traces before comparing.")
            return

        # Split by kind — FFT and time-domain can't be mixed in one overlay
        time_traces = [t for t in enabled if not t.is_fft()]
        fft_traces = [t for t in enabled if t.is_fft()]

        # If both kinds exist, let the user pick which bucket; otherwise use
        # the only one that's available.
        if len(time_traces) >= 2 and len(fft_traces) >= 2:
            kind = QMessageBox.question(
                self.window, "Compare",
                "Compare time-domain traces? "
                "(choose No for FFT traces)",
                QMessageBox.Yes | QMessageBox.No)
            candidates = time_traces if kind == QMessageBox.Yes else fft_traces
            fft_mode = kind != QMessageBox.Yes
        elif len(time_traces) >= 2:
            candidates, fft_mode = time_traces, False
        elif len(fft_traces) >= 2:
            candidates, fft_mode = fft_traces, True
        else:
            QMessageBox.information(
                self.window, "Compare",
                "Need at least 2 traces of the same type (time or FFT).")
            return

        dlg = _CompareTracePicker(candidates, fft_mode, parent=self.window)
        if dlg.exec() != QDialog.Accepted:
            return
        chosen = dlg.selected_traces()
        if len(chosen) < 2:
            return

        self._compare_window_counter += 1
        compare_window = CompareWindow(chosen, fft_mode, parent=self.window,
                                       line_width=self.line_width)
        compare_window.setWindowTitle(
            f"Compare Scopes {self._compare_window_counter}")
        compare_window.closed.connect(
            lambda w=compare_window: self._on_compare_closed(w))
        self._compare_windows.append(compare_window)
        compare_window.show()
        compare_window.raise_()
        compare_window.activateWindow()
        # Push current data immediately so the view isn't blank until next tick
        self._push_compare_data(compare_window)

    def _on_compare_closed(self, compare_window):
        if compare_window in self._compare_windows:
            self._compare_windows.remove(compare_window)

    def _push_compare_data(self, compare_window=None):
        """Send latest accumulated data into one or all compare windows."""
        if self.accumulated_data is None:
            return
        data = self.accumulated_data
        windows = (
            [compare_window]
            if compare_window is not None
            else list(self._compare_windows)
        )
        for window in windows:
            if window.fft_mode:
                # Reuse FFT cache if available; compute locally for pop-outs
                # that are open while the main plot is in a non-FFT mode.
                mags = {}
                freqs = None
                for trace in window.traces:
                    cached = self._fft_cache.get(id(trace))
                    trace_freqs = None
                    magnitude = None
                    if cached and 'magnitude' in cached:
                        magnitude = cached['magnitude']
                        if len(data['time']) > 1:
                            sample_dt = float(data['time'][1] - data['time'][0])
                            if sample_dt > 0:
                                n_fft = max(1, len(magnitude) * 2 - 2)
                                trace_freqs = np.fft.rfftfreq(n_fft, d=sample_dt)
                    if magnitude is None or trace_freqs is None:
                        trace_freqs, magnitude = self._compute_fft_payload_for_trace(
                            trace, data)
                    if trace_freqs is None or magnitude is None:
                        continue
                    if freqs is None:
                        freqs = trace_freqs
                    if len(trace_freqs) != len(freqs):
                        continue
                    mags[trace.get_display_name()] = magnitude
                window.update_data(
                    None, None, fft_freqs=freqs, fft_magnitudes=mags)
            else:
                window.update_data(data['time'], data['params'])

    def _compute_fft_payload_for_trace(self, trace, data):
        """Compute FFT data for a companion window when the main cache is absent."""
        time_arr = data['time']
        values = data['params'].get(trace.get_display_name())
        if values is None or len(time_arr) < 2 or len(values) < 2:
            return None, None

        sample_dt = float(time_arr[1] - time_arr[0])
        if sample_dt <= 0:
            return None, None

        fft_values = values
        if len(fft_values) > self._fft_max_samples:
            fft_values = fft_values[-self._fft_max_samples:]

        n_fft = len(fft_values)
        freqs = np.fft.rfftfreq(n_fft, d=sample_dt)
        window = np.hanning(n_fft)
        window_sum = np.sum(window)
        if window_sum <= 0:
            window = np.ones(n_fft)
            window_sum = n_fft
        centered = fft_values - np.mean(fft_values)
        fft_vals = np.fft.rfft(centered * window)
        magnitude = np.abs(fft_vals) * 2.0 / window_sum
        magnitude[0] /= 2.0
        return freqs, magnitude

    def _toggle_cursors(self, checked):
        """Toggle cursor measurement mode on/off."""
        self._cursors_enabled = checked
        if checked:
            self._init_cursor_positions()
            self._add_cursors_to_plots()
            has_time = any(not t.is_fft() for t in self.get_enabled_traces())
            if self.plot_mode == 'time' and has_time:
                self.cursor_readout.show()
            self._update_cursor_readout()
            self.btn_cursors.setStyleSheet(
                "background-color: #3a3a5c; border: 1px solid #FFD700;"
            )
        else:
            self._remove_cursors_from_plots()
            self.cursor_readout.hide()
            self.btn_cursors.setStyleSheet("")
        self._sync_measurement_panel(force=True)
        # Re-render FFT traces with/without cursor window
        if any(t.is_fft() for t in self.get_enabled_traces()) and self.accumulated_data is not None:
            self.curves = {}
            self.stats_texts = {}
            self._recreate_subplots()
            self._render_plots()

    def _init_cursor_positions(self):
        """Set initial cursor positions to 1/3 and 2/3 of the visible range."""
        if self.accumulated_data is not None and len(self.accumulated_data['time']) > 0:
            t = self.accumulated_data['time']
            # Use visible range if available, otherwise full data range
            # Use first time-domain plot for visible range
            first_pi = None
            for tr in self.get_enabled_traces():
                if not tr.is_fft() and id(tr) in self.plot_items:
                    first_pi = self.plot_items[id(tr)]
                    break
            if first_pi and self.plot_mode == 'time':
                vr = first_pi.getViewBox().viewRange()
                t_min, t_max = vr[0]
            else:
                t_min, t_max = float(t[0]), float(t[-1])
            span = t_max - t_min
            self._cursor_pos['c1'] = t_min + span * 0.33
            self._cursor_pos['c2'] = t_min + span * 0.67
        else:
            self._cursor_pos['c1'] = 0.0
            self._cursor_pos['c2'] = 1.0

    def _add_cursors_to_plots(self):
        """Add draggable cursor lines to time-domain subplots."""
        self._remove_cursors_from_plots()
        if self.plot_mode in ('xy', 'xyz'):
            return
        # Only add cursors to time-domain traces (not FFT subplots)
        fft_plot_keys = {id(t) for t in self.get_enabled_traces() if t.is_fft()}
        for plot_key, pi in self.plot_items.items():
            if plot_key in fft_plot_keys:
                continue
            for cid, color, store in [
                ('c1', CURSOR_COLORS['c1'], self._cursor_lines_c1),
                ('c2', CURSOR_COLORS['c2'], self._cursor_lines_c2),
            ]:
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
                line.setZValue(1000)
                # Tag line so the callback knows which cursor it is
                line._cursor_id = cid
                line.sigPositionChanged.connect(self.window._on_cursor_line_moved)
                pi.addItem(line)
                store[plot_key] = line

    def _remove_cursors_from_plots(self):
        """Remove all cursor lines from plots."""
        for store in (self._cursor_lines_c1, self._cursor_lines_c2):
            for plot_key, line in store.items():
                if plot_key in self.plot_items:
                    try:
                        self.plot_items[plot_key].removeItem(line)
                    except Exception:
                        pass
            store.clear()

    def _on_cursor_line_moved(self, line):
        """Called when any cursor line is dragged — sync all lines of same cursor."""
        if self._cursor_updating:
            return
        cid = line._cursor_id
        new_x = line.value()
        self._cursor_pos[cid] = new_x
        self._cursor_updating = True
        try:
            store = self._cursor_lines_c1 if cid == 'c1' else self._cursor_lines_c2
            for pk, other_line in store.items():
                if other_line is not line:
                    other_line.setValue(new_x)
        finally:
            self._cursor_updating = False
        self._update_cursor_readout()
        self._sync_measurement_panel()
        # Mark FFT dirty so the next timer tick re-renders (avoid per-pixel recompute)
        if any(t.is_fft() for t in self.get_enabled_traces()):
            self._fft_dirty = True
            # If capture is stopped (timer not running), render directly
            if not self._update_timer.isActive() and self.accumulated_data is not None:
                self._render_plots()

    def _get_nearest_index(self, t):
        if self.accumulated_data is None:
            return None
        time_arr = self.accumulated_data['time']
        if len(time_arr) == 0:
            return None
        if t <= time_arr[0]:
            return 0
        if t >= time_arr[-1]:
            return len(time_arr) - 1
        idx = np.searchsorted(time_arr, t)
        if idx > 0 and (idx >= len(time_arr) or
                        abs(time_arr[idx - 1] - t) <= abs(time_arr[idx] - t)):
            idx -= 1
        return idx

    def _update_cursor_readout(self):
        """Update the cursor readout panel with current cursor values."""
        if not self._cursors_enabled or self.plot_mode in ('xy', 'xyz'):
            self.cursor_readout_label.setText("")
            return

        # Only show readout for time-domain traces
        enabled_traces = [t for t in self.get_enabled_traces() if not t.is_fft()]
        if not enabled_traces:
            self.cursor_readout_label.setText("")
            return
        t1 = self._cursor_pos['c1']
        t2 = self._cursor_pos['c2']
        dt = t2 - t1

        idx1 = self._get_nearest_index(t1)
        idx2 = self._get_nearest_index(t2)

        # Build HTML table for readout
        param_cells_c1 = []
        param_cells_c2 = []
        param_cells_delta = []

        for trace in enabled_traces:
            pname = trace.get_display_name()
            color = trace.get_color()
            if self.accumulated_data is not None and pname in self.accumulated_data['params'] and idx1 is not None and idx2 is not None:
                v1 = float(self.accumulated_data['params'][pname][idx1])
                v2 = float(self.accumulated_data['params'][pname][idx2])
            else:
                v1 = v2 = None
            v1_str = f"{v1:.4f}" if v1 is not None else "---"
            v2_str = f"{v2:.4f}" if v2 is not None else "---"
            if v1 is not None and v2 is not None:
                dv = v2 - v1
                dv_str = f"{dv:+.4f}"
            else:
                dv_str = "---"
            param_cells_c1.append(
                f'<td style="padding: 0 12px;">'
                f'<span style="color:{color};">{pname}:</span> {v1_str}</td>'
            )
            param_cells_c2.append(
                f'<td style="padding: 0 12px;">'
                f'<span style="color:{color};">{pname}:</span> {v2_str}</td>'
            )
            param_cells_delta.append(
                f'<td style="padding: 0 12px;">'
                f'<span style="color:{color};">\u0394{pname}:</span> {dv_str}</td>'
            )

        # Frequency from delta-t
        if abs(dt) > 1e-9:
            freq = 1.0 / abs(dt)
            freq_str = f"{freq:.2f} Hz"
        else:
            freq_str = "--- Hz"

        c1_color = CURSOR_COLORS['c1']
        c2_color = CURSOR_COLORS['c2']

        html = (
            '<table cellspacing="0" cellpadding="1" style="font-family: Consolas; font-size: 9pt;">'
            f'<tr><td style="color:{c1_color}; font-weight:bold; padding-right:8px;">C1</td>'
            f'<td style="padding: 0 12px;">t = {t1:.6f} s</td>'
            f'{"".join(param_cells_c1)}</tr>'
            f'<tr><td style="color:{c2_color}; font-weight:bold; padding-right:8px;">C2</td>'
            f'<td style="padding: 0 12px;">t = {t2:.6f} s</td>'
            f'{"".join(param_cells_c2)}</tr>'
            f'<tr><td style="color:#FFA500; font-weight:bold; padding-right:8px;">\u0394</td>'
            f'<td style="padding: 0 12px;">\u0394t = {dt:+.6f} s</td>'
            f'{"".join(param_cells_delta)}'
            f'<td style="padding: 0 12px; color:#FFA500;">f = {freq_str}</td></tr>'
            '</table>'
        )
        self.cursor_readout_label.setText(html)

    def _setup_3d_view(self):
        """Set up the dedicated 3D OpenGL path widget."""
        self.gl_widget.setup_view(self.plot_mode, self.get_enabled_traces())


    def _build_3d_colorbar(self, axis_len, w_label):
        """Compatibility wrapper for the path widget color bar."""
        self.gl_widget.build_colorbar(axis_len, w_label)


    def _update_colorbar_range(self, w_min, w_max):
        """Compatibility wrapper for the path widget color range labels."""
        self.gl_widget.update_colorbar_range(w_min, w_max)


    def _update_x_links(self):
        """Link or unlink X-axes across subplots, partitioned by time/FFT."""
        # Unlink everything first
        for pi in self.plot_items.values():
            pi.setXLink(None)

        if not self.lock_x_axis:
            return

        # Partition into time and FFT groups
        time_plots = []
        fft_plots = []
        for trace in self.get_enabled_traces():
            tid = id(trace)
            if tid in self.plot_items:
                if trace.is_fft():
                    fft_plots.append(self.plot_items[tid])
                else:
                    time_plots.append(self.plot_items[tid])

        # Link within each group
        for group in (time_plots, fft_plots):
            if len(group) > 1:
                for pi in group[1:]:
                    pi.setXLink(group[0])

    def _on_lock_x_changed(self, checked):
        self.lock_x_axis = checked
        self._update_x_links()

    def _on_plot_mode_changed(self, index):
        modes = ['time', 'xy', 'xyz', 'xyzw']
        self.plot_mode = modes[index]
        self.path_scale_control.setVisible(self.plot_mode in ('xyz', 'xyzw'))
        self._update_path_info_label()
        self.curves = {}
        self.stats_texts = {}

        # Show/hide 2D vs 3D widgets
        if self.plot_mode in ('xyz', 'xyzw'):
            self.plot_splitter.hide()
            self.gl_widget.show()
        else:
            self.gl_widget.hide()
            self.plot_splitter.show()

        # Cursor readout visible when any trace is in time mode
        if self._cursors_enabled:
            has_time = self.plot_mode == 'time' and any(
                not t.is_fft() for t in self.get_enabled_traces())
            self.cursor_readout.setVisible(has_time)

        self._recreate_subplots()

        # Re-render static data in new mode (e.g. switching to FFT after capture)
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()

    def _on_path_view_scale_changed(self, value):
        """Scale the complete XYZ/XYZW scene from the bottom slider."""
        self.path_view_scale = max(0.25, min(4.0, value / 100.0))
        self.path_view_scale_value.setText(f"{self.path_view_scale:.2f}×")
        if self.gl_widget is not None:
            self.gl_widget.set_view_scale(self.path_view_scale)

    def _sync_path_view_scale(self, scale):
        """Reflect mouse-wheel 3D zoom in the scale slider and readout."""
        self.path_view_scale = max(0.25, min(4.0, float(scale)))
        slider_value = round(self.path_view_scale * 100)
        self.path_view_scale_slider.blockSignals(True)
        self.path_view_scale_slider.setValue(slider_value)
        self.path_view_scale_slider.blockSignals(False)
        self.path_view_scale_value.setText(f"{self.path_view_scale:.2f}×")

    def _update_path_info_label(self):
        """Update path mode info label showing axis assignments."""
        if self.plot_mode == 'time':
            fft_count = sum(1 for t in self.get_enabled_traces() if t.is_fft())
            if fft_count > 0:
                self.path_info_label.setText(
                    f"FFT: {fft_count} trace(s) in spectrum mode")
            else:
                self.path_info_label.setText("")
            return
        enabled = self.get_enabled_traces()
        if self.plot_mode == 'xy':
            if len(enabled) < 2:
                self.path_info_label.setText("Enable at least 2 traces for XY")
            else:
                self.path_info_label.setText(
                    f"X: {enabled[0].get_display_name()}  |  "
                    f"Y: {enabled[1].get_display_name()}")
        elif self.plot_mode == 'xyz':
            if len(enabled) < 3:
                self.path_info_label.setText("Enable at least 3 traces for XYZ")
            else:
                self.path_info_label.setText(
                    f"X: {enabled[0].get_display_name()}  |  "
                    f"Y: {enabled[1].get_display_name()}  |  "
                    f"Z: {enabled[2].get_display_name()}")
        elif self.plot_mode == 'xyzw':
            if len(enabled) < 4:
                self.path_info_label.setText("Enable at least 4 traces for XYZW")
            else:
                self.path_info_label.setText(
                    f"X: {enabled[0].get_display_name()}  |  "
                    f"Y: {enabled[1].get_display_name()}  |  "
                    f"Z: {enabled[2].get_display_name()}  |  "
                    f"W(color): {enabled[3].get_display_name()}")

    def add_trace(self):
        if len(self.traces) >= self.max_traces:
            QMessageBox.warning(self.window, "Maximum Traces", f"Maximum {self.max_traces} traces allowed")
            return

        trace_idx = len(self.traces)
        trace = TraceControl(trace_idx, parent=self.traces_container)
        trace.changed.connect(self.window.on_trace_changed)
        trace.btn_pin.toggled.connect(lambda checked, t=trace: self._on_pin_toggled(t, checked))
        trace.btn_popout.clicked.connect(
            lambda checked=False, t=trace: self._open_trace_window(t))
        # Set drive mode if currently in drive scope source
        if self.capture_source == 'drive':
            trace.set_drive_mode(True)
            # Auto-select different drive variables for each trace
            n_vars = trace.drive_var_combo.count()
            if trace_idx < n_vars:
                trace.drive_var_combo.setCurrentIndex(trace_idx)
        self.traces_layout.addWidget(trace)
        self.traces.append(trace)

        if len(self.traces) == 1:
            trace.chk_enable.setChecked(True)

    def on_trace_changed(self):
        # Remove destroyed traces — clear ref data for deleted ones
        alive_traces = [t for t in self.traces if t.parent() is not None]
        deleted_ids = {id(t) for t in self.traces} - {id(t) for t in alive_traces}
        for tid in deleted_ids:
            self.ref_curves.pop(tid, None)
            self._ref_set.pop(tid, None)
        self.traces = alive_traces
        self.curves = {}
        self.stats_texts = {}
        self._fft_cache = {}
        self._fft_peak_cache = {}
        self._stats_cache = {}
        self._ref_set = {}
        self._stats_pos_cache = {}
        # Close compare windows whose selected traces were deleted/disabled/retyped.
        for cw in list(self._compare_windows):
            current_keys = [t.get_display_name() for t in cw.traces]
            expected_keys = getattr(cw, 'trace_keys', current_keys)
            still_valid = (
                current_keys == expected_keys and all(
                    t in self.traces and t.is_enabled()
                    and t.is_fft() == cw.fft_mode
                    for t in cw.traces
                )
            )
            if not still_valid:
                cw.close()
        self._update_path_info_label()
        self._recreate_subplots()

        # Re-render captured data when scope is stopped (e.g. toggling FFT)
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()
            self._sync_measurement_panel(force=True)

    def _on_pin_toggled(self, trace, checked):
        """Pin or unpin the current trace data as a reference."""
        trace_id = id(trace)
        if checked:
            # Snapshot current accumulated data for this trace
            if self.accumulated_data is None:
                trace.btn_pin.setChecked(False)
                return
            param_name = trace.get_display_name()
            if param_name not in self.accumulated_data['params']:
                trace.btn_pin.setChecked(False)
                return
            trace.ref_data = {
                'time': self.accumulated_data['time'].copy(),
                'values': self.accumulated_data['params'][param_name].copy(),
            }
            # If FFT mode, also snapshot the computed FFT spectrum
            if trace.is_fft():
                cached = self._fft_cache.get(trace_id)
                if cached and 'magnitude' in cached:
                    sample_dt = float(
                        self.accumulated_data['time'][1]
                        - self.accumulated_data['time'][0]
                    ) if len(self.accumulated_data['time']) > 1 else 1.0
                    n_fft = len(cached['magnitude']) * 2 - 2  # inverse of rfftfreq
                    trace.ref_data['fft_freqs'] = np.fft.rfftfreq(
                        n_fft, d=sample_dt).copy()
                    trace.ref_data['fft_magnitude'] = cached['magnitude'].copy()
        else:
            trace.ref_data = None
            # Remove the reference curve from the plot
            if trace_id in self.ref_curves:
                ref_curve = self.ref_curves.pop(trace_id)
                if trace_id in self.plot_items:
                    self.plot_items[trace_id].removeItem(ref_curve)
            self._ref_set.pop(trace_id, None)
        # Re-render to show/hide reference
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()

    def get_enabled_traces(self):
        return [t for t in self.traces if t.is_enabled()]

    def _render_plots(self):
        """Update all plot curves with current accumulated data"""
        if self.accumulated_data is None:
            return

        plot_data = self.accumulated_data
        time_arr = plot_data['time']
        if len(time_arr) == 0:
            return

        enabled_traces = self.get_enabled_traces()

        # ── XY Mode ──
        if self.plot_mode == 'xy' and len(enabled_traces) >= 2 and 'xy' in self.plot_items:
            pi = self.plot_items['xy']
            x_name = enabled_traces[0].get_display_name()
            y_name = enabled_traces[1].get_display_name()

            if x_name not in plot_data['params'] or y_name not in plot_data['params']:
                return

            x_vals = plot_data['params'][x_name]
            y_vals = plot_data['params'][y_name]

            # Downsample for rendering if too many points
            max_xy_points = 8000
            n = len(x_vals)
            if n > max_xy_points:
                # Keep every Nth point, but always include the last point
                step = n // max_xy_points
                idx = np.arange(0, n, step)
                if idx[-1] != n - 1:
                    idx = np.append(idx, n - 1)
                x_plot = x_vals[idx]
                y_plot = y_vals[idx]
            else:
                x_plot = x_vals
                y_plot = y_vals

            if 'xy_path' not in self.curves:
                pen = pg.mkPen('#03DAC6', width=self.line_width)
                curve = pi.plot(pen=pen)
                curve._viewBox = weakref.ref(pi.getViewBox())
                self.curves['xy_path'] = curve
            if 'xy_cursor' not in self.curves:
                curve = pi.plot(
                    symbol='o', symbolSize=8,
                    symbolBrush='#FF5555', symbolPen=None, pen=None)
                curve._viewBox = weakref.ref(pi.getViewBox())
                self.curves['xy_cursor'] = curve
            if 'xy_hover' not in self.curves:
                curve = pi.plot(
                    symbol='o', symbolSize=11,
                    symbolBrush='#FFFFFF', symbolPen=None, pen=None)
                curve.setZValue(1002)
                curve._viewBox = weakref.ref(pi.getViewBox())
                self.curves['xy_hover'] = curve

            vb = pi.getViewBox()
            if self._xy_auto_range:
                # Update data then fit view to all data
                self.curves['xy_path'].setData(x_plot, y_plot)
                self.curves['xy_cursor'].setData([x_vals[-1]], [y_vals[-1]])
                margin = 0.05
                # Incremental min/max — scan only newly appended samples
                # (full-array min/max is O(n) per frame on a growing buffer)
                prev_mm = self._stats_cache.get(('xy_minmax',))
                if prev_mm is not None and prev_mm[0] <= n:
                    mm_start, x_min, x_max, y_min, y_max = prev_mm
                else:
                    mm_start = 0
                    x_min = y_min = np.inf
                    x_max = y_max = -np.inf
                if n > mm_start:
                    x_min = min(x_min, float(np.min(x_vals[mm_start:])))
                    x_max = max(x_max, float(np.max(x_vals[mm_start:])))
                    y_min = min(y_min, float(np.min(y_vals[mm_start:])))
                    y_max = max(y_max, float(np.max(y_vals[mm_start:])))
                    self._stats_cache[('xy_minmax',)] = (n, x_min, x_max, y_min, y_max)
                # Make X and Y spans equal so the view is symmetric
                x_span = x_max - x_min
                y_span = y_max - y_min
                max_span = max(x_span, y_span, 1.0)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                half = max_span / 2 * (1 + margin)
                pi.setXRange(x_center - half, x_center + half, padding=0)
                pi.setYRange(y_center - half, y_center + half, padding=0)
            else:
                # User has zoomed/panned — update data without touching the range
                # Block ViewBox auto-range signals so setData cannot move the view
                vb.blockSignals(True)
                try:
                    self.curves['xy_path'].setData(x_plot, y_plot)
                    self.curves['xy_cursor'].setData([x_vals[-1]], [y_vals[-1]])
                finally:
                    vb.blockSignals(False)
            return

        # ── XYZ Mode ──
        if self.plot_mode == 'xyz' and len(enabled_traces) >= 3:
            x_name = enabled_traces[0].get_display_name()
            y_name = enabled_traces[1].get_display_name()
            z_name = enabled_traces[2].get_display_name()

            if not all(n in plot_data['params'] for n in (x_name, y_name, z_name)):
                return

            self.gl_widget.render_xyz(
                plot_data['params'][x_name],
                plot_data['params'][y_name],
                plot_data['params'][z_name],
                line_width=self.line_width,
            )
            return

        # ── XYZW Mode (4D: color-mapped W) ──
        if self.plot_mode == 'xyzw' and len(enabled_traces) >= 4:
            x_name = enabled_traces[0].get_display_name()
            y_name = enabled_traces[1].get_display_name()
            z_name = enabled_traces[2].get_display_name()
            w_name = enabled_traces[3].get_display_name()

            if not all(n in plot_data['params'] for n in (x_name, y_name, z_name, w_name)):
                return

            self.gl_widget.render_xyzw(
                plot_data['params'][x_name],
                plot_data['params'][y_name],
                plot_data['params'][z_name],
                plot_data['params'][w_name],
                line_width=self.line_width,
            )
            return

        # ── Per-trace rendering (time or FFT per trace) ──

        # Auto-scroll: apply only to the first time-domain plot
        if self.auto_scroll and self.is_running:
            max_time = time_arr[-1]
            min_time = max(0, max_time - self.window_duration)
            slice_min_time = max(0, min_time - self.window_duration * 0.1)
            slice_idx = np.searchsorted(time_arr, slice_min_time)
            render_time_arr = time_arr[slice_idx:]
            for trace in enabled_traces:
                if not trace.is_fft() and id(trace) in self.plot_items:
                    self.plot_items[id(trace)].setXRange(min_time, max_time, padding=0)
                    break
        else:
            render_time_arr = time_arr
            slice_idx = 0

        # Precompute FFT shared data if any trace needs it
        has_fft_traces = any(t.is_fft() for t in enabled_traces)
        freqs = None
        fft_time = time_arr
        fft_params = plot_data['params']
        fft_cursor_key = None
        # Only slice params for traces that actually need FFT (avoid O(N_traces) copies)
        fft_needed_names = (
            {t.get_display_name() for t in enabled_traces if t.is_fft()}
            if has_fft_traces else set()
        )
        if has_fft_traces and len(time_arr) >= 2:
            sample_dt = float(time_arr[1] - time_arr[0])
            if sample_dt > 0:
                # Windowed FFT: if cursors are enabled, use C1\u2013C2 time window
                if self._cursors_enabled:
                    t1 = min(self._cursor_pos['c1'], self._cursor_pos['c2'])
                    t2 = max(self._cursor_pos['c1'], self._cursor_pos['c2'])
                    mask = (time_arr >= t1) & (time_arr <= t2)
                    if np.sum(mask) >= 2:
                        fft_time = time_arr[mask]
                        fft_params = {
                            k: plot_data['params'][k][mask]
                            for k in fft_needed_names
                            if k in plot_data['params']
                        }
                        duration = float(fft_time[-1] - fft_time[0])
                        freq_res = 1.0 / duration if duration > 0 else 0
                        self.path_info_label.setText(
                            f"FFT window: {t1:.3f}s \u2192 {t2:.3f}s "
                            f"({duration:.3f}s, {len(fft_time)} pts, "
                            f"\u0394f={freq_res:.2f} Hz)")
                    else:
                        self.path_info_label.setText(
                            "FFT: cursor window too narrow \u2014 using full data")
                else:
                    # Cap FFT size to last N samples for performance
                    if len(fft_time) > self._fft_max_samples:
                        fft_time = fft_time[-self._fft_max_samples:]
                        fft_params = {
                            k: plot_data['params'][k][-self._fft_max_samples:]
                            for k in fft_needed_names
                            if k in plot_data['params']
                        }
                    else:
                        fft_params = {
                            k: plot_data['params'][k]
                            for k in fft_needed_names
                            if k in plot_data['params']
                        }
                    self.path_info_label.setText(
                        f"FFT: last {len(fft_time)} pts (enable cursors to window)")
                n_fft = len(fft_time)
                freqs = np.fft.rfftfreq(n_fft, d=sample_dt)
                # Cache Hanning window — reuse if size unchanged
                if self._fft_window_cache[0] != n_fft:
                    self._fft_window_cache = (n_fft, np.hanning(n_fft))
                fft_cursor_key = (round(self._cursor_pos['c1'], 6),
                                  round(self._cursor_pos['c2'], 6)) if self._cursors_enabled else None
        self._fft_dirty = False
        if has_fft_traces:
            self._last_freqs = freqs


        for trace in enabled_traces:
            trace_id = id(trace)
            if trace_id not in self.plot_items:
                continue

            pi = self.plot_items[trace_id]
            param_name = trace.get_display_name()
            color = trace.get_color()

            if trace.is_fft():
                # ── FFT rendering for this trace ──
                if freqs is None or param_name not in fft_params:
                    continue
                values = fft_params[param_name]
                n_fft = len(fft_time)

                # Check FFT cache — throttle recompute to ~10Hz by bucketing last-sample time
                last_t_bucket = int(float(time_arr[-1]) * 10) if len(time_arr) else 0
                cache_key = (n_fft, last_t_bucket, fft_cursor_key)
                cached = self._fft_cache.get(trace_id)
                if cached and cached['key'] == cache_key:
                    magnitude = cached['magnitude']
                else:
                    # Compute single-sided amplitude spectrum
                    centered = values - np.mean(values)
                    window = self._fft_window_cache[1]
                    windowed = centered * window
                    fft_vals = np.fft.rfft(windowed)
                    window_sum = np.sum(window)
                    magnitude = np.abs(fft_vals) * 2.0 / window_sum
                    magnitude[0] /= 2.0
                    self._fft_cache[trace_id] = {
                        'key': cache_key,
                        'magnitude': magnitude,
                    }

                if trace_id not in self.curves:
                    if pi.legend is None:
                        pi.addLegend(
                            offset=(10, 5),
                            brush=pg.mkBrush('#2B2B2BBB'),
                            pen=pg.mkPen('#606060'),
                            labelTextColor='#d4d4d4',
                            labelTextSize='9pt',
                        )
                    pen = pg.mkPen(color, width=self.line_width)
                    curve = pi.plot(name=param_name, pen=pen)
                    curve._viewBox = weakref.ref(pi.getViewBox())
                    curve.setClipToView(True)
                    curve.setDownsampling(auto=True, method='subsample')
                    self.curves[trace_id] = curve

                self.curves[trace_id].setData(freqs, magnitude, skipFiniteCheck=True)

                # ── Reference (pinned) FFT overlay ──
                if (trace.has_ref_data()
                        and 'fft_freqs' in trace.ref_data
                        and 'fft_magnitude' in trace.ref_data):
                    if trace_id not in self.ref_curves:
                        ref_pen = pg.mkPen(trace.ref_color,
                                           width=self.line_width)
                        ref_curve = pi.plot(
                            name=f"{param_name} (REF)", pen=ref_pen)
                        ref_curve._viewBox = weakref.ref(pi.getViewBox())
                        ref_curve.setClipToView(True)
                        ref_curve.setDownsampling(auto=True, method='subsample')
                        self.ref_curves[trace_id] = ref_curve
                    ref_key = ('fft', id(trace.ref_data))
                    if self._ref_set.get(trace_id) != ref_key:
                        self.ref_curves[trace_id].setData(
                            trace.ref_data['fft_freqs'],
                            trace.ref_data['fft_magnitude'])
                        self._ref_set[trace_id] = ref_key

                # Peak frequency annotation (throttled — only update when values change)
                if len(magnitude) > 1:
                    peak_idx = np.argmax(magnitude[1:]) + 1
                    peak_freq = round(float(freqs[peak_idx]), 2)
                    peak_mag = round(float(magnitude[peak_idx]), 4)
                    prev_peak = self._fft_peak_cache.get(trace_id)
                    if prev_peak != (peak_freq, peak_mag):
                        self._fft_peak_cache[trace_id] = (peak_freq, peak_mag)
                        stats_html = (
                            f'<span style="font-family: Segoe UI; font-size: 8pt;">'
                            f'<span style="color: #FFA500;">Peak: {peak_freq:.2f} Hz</span><br>'
                            f'<span style="color: #99FF99;">Mag: {peak_mag:.4f}</span>'
                            f'</span>'
                        )
                        if trace_id not in self.stats_texts:
                            txt = pg.TextItem(anchor=(1, 0))
                            txt.setHtml(stats_html)
                            pi.getViewBox().addItem(txt, ignoreBounds=True)
                            self.stats_texts[trace_id] = txt
                        else:
                            self.stats_texts[trace_id].setHtml(stats_html)
                    if trace_id in self.stats_texts:
                        vb = pi.getViewBox()
                        view_range = vb.viewRange()
                        new_pos = (view_range[0][1], view_range[1][1])
                        if self._stats_pos_cache.get(trace_id) != new_pos:
                            self.stats_texts[trace_id].setPos(*new_pos)
                            self._stats_pos_cache[trace_id] = new_pos
                        if trace_id in self._hover_labels:
                            self._hover_labels[trace_id].setPos(*new_pos)
            else:
                # ── Time-domain rendering for this trace ──
                if param_name not in plot_data['params']:
                    continue

                values = plot_data['params'][param_name]
                render_values = values[slice_idx:] if self.auto_scroll and self.is_running else values

                if trace_id not in self.curves:
                    if pi.legend is None:
                        pi.addLegend(
                            offset=(10, 5),
                            brush=pg.mkBrush('#2B2B2BBB'),
                            pen=pg.mkPen('#606060'),
                            labelTextColor='#d4d4d4',
                            labelTextSize='9pt'
                        )
                    pen = pg.mkPen(color, width=self.line_width)
                    curve = pi.plot(name=param_name, pen=pen)
                    curve._viewBox = weakref.ref(pi.getViewBox())
                    curve.setClipToView(True)
                    curve.setDownsampling(auto=True, method='subsample')
                    self.curves[trace_id] = curve

                # Skip setData when the rendered window has not changed. During
                # capture auto-scroll may leave the curve holding only the last
                # visible slice; stopping must redraw the full buffer even when
                # the sample count is unchanged.
                cur_len = len(values)
                render_key = (cur_len, slice_idx, len(render_time_arr))
                prev_render_key = self._stats_cache.get(('render', trace_id))
                data_changed = render_key != prev_render_key
                if data_changed:
                    self.curves[trace_id].setData(render_time_arr, render_values, skipFiniteCheck=True)
                    self._stats_cache[('render', trace_id)] = render_key

                # ── Reference (pinned) trace overlay ──
                if trace.has_ref_data():
                    if trace_id not in self.ref_curves:
                        ref_pen = pg.mkPen(trace.ref_color,
                                           width=self.line_width)
                        ref_curve = pi.plot(
                            name=f"{param_name} (REF)", pen=ref_pen)
                        ref_curve._viewBox = weakref.ref(pi.getViewBox())
                        ref_curve.setClipToView(True)
                        ref_curve.setDownsampling(auto=True, method='subsample')
                        self.ref_curves[trace_id] = ref_curve
                    ref_key = ('time', id(trace.ref_data))
                    if self._ref_set.get(trace_id) != ref_key:
                        self.ref_curves[trace_id].setData(
                            trace.ref_data['time'], trace.ref_data['values'])
                        self._ref_set[trace_id] = ref_key

                # Update min/max stats text (only when new data arrived).
                # Incremental: scan only samples appended since the last tick —
                # a full-buffer min/max here is O(n) per frame, O(n²) over a
                # long capture. _stats_cache is cleared on trace change,
                # import, and clear_data, which invalidates this too.
                if data_changed and cur_len > 0:
                    mm_key = ('minmax', trace_id)
                    prev_mm = self._stats_cache.get(mm_key)
                    if prev_mm is not None and prev_mm[0] <= cur_len:
                        mm_start, v_min, v_max = prev_mm
                    else:
                        mm_start, v_min, v_max = 0, np.inf, -np.inf
                    if cur_len > mm_start:
                        new_vals = values[mm_start:]
                        v_min = min(v_min, float(np.min(new_vals)))
                        v_max = max(v_max, float(np.max(new_vals)))
                        self._stats_cache[mm_key] = (cur_len, v_min, v_max)
                    v_min_s = f"{v_min:.4f}"
                    v_max_s = f"{v_max:.4f}"
                    prev_stats = self._stats_cache.get(trace_id)
                    if prev_stats != (v_min_s, v_max_s):
                        self._stats_cache[trace_id] = (v_min_s, v_max_s)
                        stats_html = (
                            f'<span style="font-family: Segoe UI; font-size: 8pt;">'
                            f'<span style="color: #FF9999;">Min: {v_min_s}</span><br>'
                            f'<span style="color: #99FF99;">Max: {v_max_s}</span>'
                            f'</span>'
                        )
                        if trace_id not in self.stats_texts:
                            txt = pg.TextItem(anchor=(1, 0))
                            txt.setHtml(stats_html)
                            pi.getViewBox().addItem(txt, ignoreBounds=True)
                            self.stats_texts[trace_id] = txt
                        else:
                            self.stats_texts[trace_id].setHtml(stats_html)
                if trace_id in self.stats_texts:
                    vb = pi.getViewBox()
                    view_range = vb.viewRange()
                    new_pos = (view_range[0][1], view_range[1][1])
                    if self._stats_pos_cache.get(trace_id) != new_pos:
                        self.stats_texts[trace_id].setPos(*new_pos)
                        self._stats_pos_cache[trace_id] = new_pos
                    if trace_id in self._hover_labels:
                        self._hover_labels[trace_id].setPos(*new_pos)

        # Update cursor readout if cursors are active (time-domain traces only)
        if self._cursors_enabled:
            self._update_cursor_readout()

    def toggle_auto_scroll(self):
        self.auto_scroll = not self.auto_scroll
        self._update_auto_scroll_button()

    def _update_auto_scroll_button(self):
        if self.auto_scroll:
            self.btn_auto_scroll.setText("\u25b6 Auto-scroll ON")
            self.btn_auto_scroll.setVisible(False if self.is_running else False)
        else:
            self.btn_auto_scroll.setText("\U0001f512 Auto-scroll OFF")
            self.btn_auto_scroll.setVisible(self.is_running)

    def _fit_all_data(self):
        """Set view range to show all captured data."""
        if self.accumulated_data is None:
            return
        time_arr = self.accumulated_data['time']
        if len(time_arr) == 0:
            return
        if self.plot_mode in ('xy', 'xyz'):
            return
        # Fit only time-domain traces
        for trace in self.get_enabled_traces():
            if not trace.is_fft() and id(trace) in self.plot_items:
                self.plot_items[id(trace)].setXRange(
                    float(time_arr[0]), float(time_arr[-1]), padding=0.02)
                break

    def clear_data(self):
        self.accumulated_data = None
        self.total_samples = 0
        self._last_consumed_state = None
        self._virtual_buffers = {}
        with self._data_lock:
            self._buffer_len = 0
            self._segment_breaks = []
        self.curves = {}
        self.ref_curves = {}
        self.stats_texts = {}
        self._ref_set = {}
        self._stats_pos_cache = {}
        self._stats_cache = {}
        # Unpin all traces and clear their reference data
        for trace in self.traces:
            trace.ref_data = None
            trace.btn_pin.setChecked(False)
        if hasattr(self.gl_widget, "clear_path"):
            self.gl_widget.clear_path()
        self._recreate_subplots()
        self.sample_counter_label.setText("Samples: 0")
        self.status_label.setText("Data cleared")
        if self._cursors_enabled:
            self._update_cursor_readout()
        self._sync_measurement_panel(force=True)

    def _apply_plot_settings(self):
        """Apply settings to all plots"""
        for pi in self.plot_items.values():
            pi.getViewBox().setBackgroundColor(self.plot_bg_color)
            pi.showGrid(x=True, y=True, alpha=self.grid_alpha)
        # Update line widths
        for curve in self.curves.values():
            pen = curve.opts.get('pen')
            if pen:
                color = pen.color()
                curve.setPen(pg.mkPen(color, width=self.line_width))
        for ref_curve in self.ref_curves.values():
            pen = ref_curve.opts.get('pen')
            if pen:
                ref_curve.setPen(pg.mkPen(pen.color(), width=self.line_width))
        for compare_window in self._compare_windows:
            compare_window.set_line_width(self.line_width)
        self._update_x_links()

