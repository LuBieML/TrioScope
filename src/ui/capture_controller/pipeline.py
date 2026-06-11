"""Shared capture data pipeline: buffering, update timer, stop handling."""

import logging

import numpy as np

from scope.parameters import CHANNEL_PARAMETERS_SET, _VIRTUAL_PARAM_MAP

logger = logging.getLogger(__name__)


class CapturePipelineMixin:
    """Thread-safe data buffering and the ~30fps consolidation timer."""

    def _push_data(self, data):
        """Thread-safe: push new data chunk from capture thread into pre-allocated buffer"""
        time_chunk = data['time']
        n_new = len(time_chunk)
        if n_new == 0:
            return

        with self._data_lock:
            while self._buffer_len + n_new > self._buffer_capacity:
                self._buffer_capacity = max(100_000, self._buffer_capacity * 2)
                new_time = np.empty(self._buffer_capacity, dtype=np.float64)
                new_time[:self._buffer_len] = self._time_buffer[:self._buffer_len]
                self._time_buffer = new_time
                for k, v in self._param_buffers.items():
                    new_v = np.empty(self._buffer_capacity, dtype=v.dtype)
                    new_v[:self._buffer_len] = v[:self._buffer_len]
                    self._param_buffers[k] = new_v

            start = self._buffer_len
            end = start + n_new
            self._time_buffer[start:end] = time_chunk
            for param_name, values in data['params'].items():
                if param_name not in self._param_buffers:
                    self._param_buffers[param_name] = np.empty(self._buffer_capacity, dtype=values.dtype)
                    if start > 0:
                        self._param_buffers[param_name][:start] = values[0] if len(values) > 0 else 0
                self._param_buffers[param_name][start:end] = values

            self._buffer_len += n_new

    def _push_segment_break(self):
        """Record current sample count as a segment boundary (capture restart)."""
        with self._data_lock:
            self._segment_breaks.append(self._buffer_len)

    def _on_capture_progress(self, msg: str):
        self.progress_label.setText(msg)

    def _on_capture_status(self, msg: str):
        self.status_label.setText(msg)

    def _on_capture_stopped(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._update_timer.stop()
        # Force final update to consolidate and render all captured/downloaded data
        self._on_update_timer(force=True)
        # Final render: show all captured data so panning works immediately
        if self.auto_scroll:
            self.auto_scroll = False
            self._update_auto_scroll_button()
        # Draw the final buffer before fitting. Single-push captures (e.g. the
        # drive scope) deliver all their data in one _push_data right before
        # this handler runs, and stopping the timer can beat the next render
        # tick — so consolidate and render once here or the curves stay empty.
        # Forced: auto-scroll just changed, so the full buffer must be redrawn
        # even though the sample count is unchanged.
        self._on_update_timer(force=True)
        self._fit_all_data()

    def _on_update_timer(self, force=False):
        """Main-thread timer: consolidate data and update plots at ~30fps"""
        # Consolidate data chunks under lock
        with self._data_lock:
            if self._buffer_len == 0:
                return
            n = self._buffer_len
            consumed_state = (n, len(self._segment_breaks))
            if not force and consumed_state == self._last_consumed_state:
                return
            all_time = self._time_buffer[:n]
            all_params = {k: v[:n] for k, v in self._param_buffers.items()}
            seg_breaks = list(self._segment_breaks)
        self._last_consumed_state = consumed_state

        # Inject virtual derived channels.
        # For each enabled virtual-param trace, compute its value from the underlying
        # raw data that was captured under the underlying param's display-name key.
        # Currently: DEMAND_SPEED_NORMALISED = DEMAND_SPEED / servo_period_sec
        #            (converts units/servocycle → units/second to match MSPEED units).
        if self.scope_engine is not None and self.scope_engine.servo_period_sec:
            sp = self.scope_engine.servo_period_sec
            for trace in self.get_enabled_traces():
                if trace._drive_mode:
                    continue
                raw_param = trace.param_combo.currentText()
                if raw_param not in _VIRTUAL_PARAM_MAP:
                    continue
                underlying = _VIRTUAL_PARAM_MAP[raw_param]
                idx = trace.axis_spin.value()
                # Build the key under which the raw data was stored (display-name format)
                if underlying in CHANNEL_PARAMETERS_SET:
                    src_key = f"{underlying} Ch({idx})"
                else:
                    src_key = f"{underlying}({idx})"
                dst_key = trace.get_display_name()  # e.g. "DEMAND_SPEED_NORMALISED(0)"
                if src_key in all_params:
                    # Divide only the newly arrived slice into a persistent
                    # buffer — a full-array divide here is O(n) per tick.
                    src = all_params[src_key]
                    processed, buf = self._virtual_buffers.get(dst_key, (0, None))
                    if buf is None or len(buf) < n or processed > n:
                        buf = np.empty(max(n, self._buffer_capacity), dtype=np.float64)
                        processed = 0
                    if n > processed:
                        np.divide(src[processed:n], sp, out=buf[processed:n])
                        self._virtual_buffers[dst_key] = (n, buf)
                    all_params[dst_key] = buf[:n]

        self.accumulated_data = {
            'time': all_time,
            'num_samples': len(all_time),
            'params': all_params,
            'segment_breaks': seg_breaks,
        }
        self.total_samples = len(all_time)
        self.sample_counter_label.setText(f"Samples: {self.total_samples}")

        # Update plots
        self._render_plots()

        # Mirror live data into every compare window if any are open.
        if self._compare_windows:
            self._push_compare_data()

        # Update trace value labels
        for trace in self.get_enabled_traces():
            param_name = trace.get_display_name()
            if param_name in all_params and len(all_params[param_name]) > 0:
                trace.update_value(all_params[param_name][-1])

        self._sync_measurement_panel()

    def stop_capture(self):
        self.is_running = False
        if self.capture_source == 'drive':
            if self.drive_scope_engine:
                try:
                    self.drive_scope_engine.stop_capture()
                except Exception:
                    pass
        else:
            if self.scope_engine:
                try:
                    self.scope_engine.stop_capture()
                except Exception:
                    pass
        self.status_label.setText("Stopped")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_auto_scroll.setVisible(False)
