"""Pop-out trace windows and multi-trace compare windows."""

import numpy as np
from PySide6.QtWidgets import QDialog, QMessageBox

from ui.compare_window import CompareWindow, _CompareTracePicker, TraceWindow


class CompareWindowsMixin:
    """Companion trace/compare window management and data mirroring."""

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
