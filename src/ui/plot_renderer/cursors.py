"""Measurement cursors: draggable C1/C2 lines and the delta readout."""

import numpy as np
from PySide6.QtCore import Qt
import pyqtgraph as pg

from ui.theme import CURSOR_COLORS


class CursorsMixin:
    """Cursor lines on time-domain subplots and the readout table."""

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
                f'<span style="color:{color};">Δ{pname}:</span> {dv_str}</td>'
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
            f'<tr><td style="color:#FFA500; font-weight:bold; padding-right:8px;">Δ</td>'
            f'<td style="padding: 0 12px;">Δt = {dt:+.6f} s</td>'
            f'{"".join(param_cells_delta)}'
            f'<td style="padding: 0 12px; color:#FFA500;">f = {freq_str}</td></tr>'
            '</table>'
        )
        self.cursor_readout_label.setText(html)
