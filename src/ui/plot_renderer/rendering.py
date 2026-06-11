"""Curve rendering: time/FFT/XY/3D data updates, auto-scroll, clear, styling."""

import weakref

import numpy as np
import pyqtgraph as pg


class RenderingMixin:
    """_render_plots and the view-fitting / clearing helpers around it."""

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
                # Windowed FFT: if cursors are enabled, use C1–C2 time window
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
                            f"FFT window: {t1:.3f}s → {t2:.3f}s "
                            f"({duration:.3f}s, {len(fft_time)} pts, "
                            f"Δf={freq_res:.2f} Hz)")
                    else:
                        self.path_info_label.setText(
                            "FFT: cursor window too narrow — using full data")
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
            self.btn_auto_scroll.setText("▶ Auto-scroll ON")
            self.btn_auto_scroll.setVisible(False if self.is_running else False)
        else:
            self.btn_auto_scroll.setText("🔒 Auto-scroll OFF")
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
