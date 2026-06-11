"""Controller SCOPE capture: start validation and single/continuous threads."""

import logging
import threading
import time

from PySide6.QtWidgets import QMessageBox

from scope.parameters import CHANNEL_PARAMETERS_SET, _VIRTUAL_PARAM_MAP

logger = logging.getLogger(__name__)


class ControllerScopeCaptureMixin:
    """start_capture() and the controller SCOPE capture threads."""

    def start_capture(self):
        if not self.trio_connected:
            QMessageBox.critical(self.window, "Error", "Not connected")
            return

        # Route to drive scope or controller scope
        if self.capture_source == 'drive':
            self._start_drive_scope_capture()
            return

        enabled_traces = self.get_enabled_traces()
        if not enabled_traces:
            QMessageBox.warning(self.window, "No Traces", "Enable at least one trace")
            return

        missing_params = [
            f"Trace {t.trace_number + 1}"
            for t in enabled_traces
            if not t._drive_mode and not t.param_combo.currentText().strip()
        ]
        if missing_params:
            QMessageBox.warning(
                self.window, "Missing Parameter",
                "Select a parameter before starting capture for:\n\n"
                + "\n".join(missing_params))
            return

        # Deduplicate parameters — Trio SCOPE supports max 8 unique params
        seen = {}
        unique_params = []
        unique_display = []
        for t in enabled_traces:
            ps = t.get_parameter_string()
            if ps not in seen:
                raw_param = t.param_combo.currentText().strip() if not t._drive_mode else None
                if raw_param in _VIRTUAL_PARAM_MAP:
                    # Virtual params map to an underlying Trio param.  Store raw data under
                    # the display-name format of the underlying param (e.g. "DEMAND_SPEED(0)")
                    # so the post-capture injection step can always find it by that key.
                    underlying = _VIRTUAL_PARAM_MAP[raw_param]
                    idx = t.axis_spin.value()
                    if underlying in CHANNEL_PARAMETERS_SET:
                        display = f"{underlying} Ch({idx})"
                    else:
                        display = f"{underlying}({idx})"
                else:
                    display = t.get_display_name()
                seen[ps] = display
                unique_params.append(ps)
                unique_display.append(display)

        if len(unique_params) > 8:
            QMessageBox.warning(self.window, "Too Many Parameters",
                                "Trio SCOPE supports max 8 unique parameters.\n"
                                f"You have {len(unique_params)} unique parameters enabled.\n"
                                "Use duplicate parameters across traces to stay within the limit.")
            return

        # Rebuild subplots and clear old curves
        self.curves = {}
        self.stats_texts = {}
        self._recreate_subplots()

        try:
            period_cycles = int(self.period_edit.text())
            duration_sec = float(self.duration_edit.text())
            num_params = len(unique_params)

            # Calculate table_start
            if self.use_end_of_table:
                sample_period_sec = period_cycles * self.scope_engine.servo_period_sec
                total_samples = int(duration_sec / sample_period_sec)
                total_entries = total_samples * num_params
                table_start = max(0, self.scope_engine.tsize - total_entries)
            else:
                try:
                    table_start = int(self.table_start_edit.text())
                except ValueError:
                    table_start = 0

            if self.radio_continuous.isChecked():
                sample_period_sec = period_cycles * self.scope_engine.servo_period_sec
                available = self.scope_engine.tsize - table_start
                max_samples = available // num_params
                config_duration = max_samples * sample_period_sec
            else:
                config_duration = duration_sec

            self.scope_engine.configure(
                unique_params, unique_display, period_cycles, config_duration, table_start)

            # Clear data
            self.accumulated_data = None
            self.total_samples = 0
            self._last_consumed_state = None
            self._virtual_buffers = {}
            with self._data_lock:
                self._buffer_len = 0
                self._segment_breaks = []

            # Update UI
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.is_running = True
            self.auto_scroll = True
            self._update_auto_scroll_button()

            # Start update timer
            self._update_timer.start()

            # Start capture thread
            if self.external_trigger_chk.isChecked() and self.radio_single.isChecked():
                self.scope_thread = threading.Thread(target=self._scope_single_external_trigger_thread, daemon=True)
            elif self.external_trigger_chk.isChecked():
                self.scope_thread = threading.Thread(target=self._scope_continuous_external_trigger_thread, daemon=True)
            elif self.radio_single.isChecked():
                self.scope_thread = threading.Thread(target=self._scope_single_shot_thread, daemon=True)
            else:
                self.scope_thread = threading.Thread(target=self._scope_continuous_thread, daemon=True)
            self.scope_thread.start()

        except Exception as e:
            QMessageBox.critical(self.window, "Start Error", str(e))
            logger.exception("Start capture failed")

    def _scope_single_shot_thread(self):
        """Single-shot capture — background thread"""
        try:
            period_cycles = int(self.period_edit.text())
            self.scope_engine.start_capture()
            capture_start_time = time.time()
            duration_sec = float(self.duration_edit.text())
            last_sample_idx = 0

            while self.is_running and (time.time() - capture_start_time) < duration_sec:
                batch_data, last_sample_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)
                if batch_data and batch_data['num_samples'] > 0:
                    self._push_data(batch_data)

                elapsed = time.time() - capture_start_time
                pct = (elapsed / duration_sec) * 100
                self.sig_capture_progress.emit(f"Progress: {pct:.1f}%")
                time.sleep(0.025)

            if not self.is_running:
                self.scope_engine.stop_capture()
                return

            # Wait for capture completion
            timeout = 10
            wait_start = time.time()
            while not self.scope_engine.is_capture_complete():
                if time.time() - wait_start > timeout:
                    break
                time.sleep(0.02)

            # Final read — only fetch samples not yet streamed
            final_batch, last_sample_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)
            self.scope_engine.stop_capture()
            if final_batch and final_batch['num_samples'] > 0:
                self._push_data(final_batch)
            self.sig_capture_status.emit(f"Captured {last_sample_idx} samples")

        except Exception as e:
            self.sig_capture_status.emit(f"Error: {e}")
            logger.exception("Single-shot error")
        finally:
            self.is_running = False
            self.sig_capture_stopped.emit()

    def _scope_continuous_thread(self):
        """Continuous capture — background thread.

        Uses TRIGGER(1) for auto-retrigger: the controller automatically
        restarts capture when the buffer fills, eliminating PC-side
        stop/restart gaps and timing compensation.
        """
        try:
            samples_per_param = (
                (self.scope_engine.table_end - self.scope_engine.table_start + 1)
                // self.scope_engine.num_params
            )
            sample_period = self.scope_engine.period_cycles * self.scope_engine.servo_period_sec

            self.scope_engine.start_capture(auto_retrigger=True)
            time.sleep(0.05)
            last_sample_idx = 0
            sample_offset = 0  # cumulative sample offset across wraps
            self.sig_capture_status.emit("Capturing (continuous)...")

            while self.is_running and self.trio_connected:
                batch_data, new_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)

                if batch_data and batch_data['num_samples'] > 0:
                    # Shift time by cumulative offset from previous scans
                    time_shift = sample_offset * sample_period
                    if time_shift > 0:
                        batch_data['time'] = batch_data['time'] + time_shift
                    self._push_data(batch_data)
                    last_sample_idx = new_idx
                else:
                    # Detect auto-retrigger wrap: SCOPE_POS resets to 0
                    try:
                        scope_pos = self.scope_engine.connection.GetSystemParameter_SCOPE_POS()
                        if scope_pos < last_sample_idx and last_sample_idx > 0:
                            sample_offset += samples_per_param
                            last_sample_idx = 0
                            continue
                    except Exception:
                        pass

                time.sleep(0.025)

            if not self.is_running:
                try:
                    self.scope_engine.stop_capture()
                except Exception:
                    pass

        except Exception as e:
            self.sig_capture_status.emit(f"Error: {e}")
            logger.exception("Continuous error")
        finally:
            self.is_running = False
            self.sig_capture_stopped.emit()
