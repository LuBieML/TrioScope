"""Subplot creation, per-mode layout, axis linking, and plot configuration."""

import pyqtgraph as pg

from plot.viewbox import ScopeViewBox


class PlotLayoutMixin:
    """Creates and configures the subplot grid for each plot mode."""

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

    def _on_xy_manual_zoom(self, _changes):
        """When user manually pans/zooms in XY mode, stop auto-fitting."""
        self._xy_auto_range = False
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()

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
