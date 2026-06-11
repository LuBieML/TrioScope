"""Trace control management: add/remove traces and pin-as-reference."""

import numpy as np
from PySide6.QtWidgets import QMessageBox

from ui.trace_control import TraceControl


class TraceManagementMixin:
    """TraceControl lifecycle and reference pinning."""

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
