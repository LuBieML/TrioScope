"""Debounced overlays: stats-text repositioning, curve dot detail, hover readout."""

import numpy as np
from PySide6.QtCore import Qt, QTimer
import pyqtgraph as pg


class HoverOverlaysMixin:
    """Pan/zoom-coalesced overlay updates and the hover crosshair."""

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
            return

        vb_active = active_pi.getViewBox()
        mouse_point = vb_active.mapSceneToView(scene_pos)
        x = mouse_point.x()

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
            if self.plot_mode == 'xy':
                enabled = self.get_enabled_traces()
                if len(enabled) >= 2:
                    if key == active_plot_key:
                        html_lines.append(f"<span style='color:{enabled[0].get_color()}'>{enabled[0].get_display_name()} = {mouse_point.x():.4g}</span>")
                        html_lines.append(f"<span style='color:{enabled[1].get_color()}'>{enabled[1].get_display_name()} = {mouse_point.y():.4g}</span>")
            else:  # time mode
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
