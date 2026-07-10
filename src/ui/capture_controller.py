import logging
import os
import threading
import time

import numpy as np
from PySide6.QtCore import QStandardPaths, Signal
from PySide6.QtWidgets import QMessageBox

from scope.drive_scope_engine import NUM_CHANNELS as DRIVE_NUM_CHANNELS
from scope.parameters import CHANNEL_PARAMETERS_SET, _VIRTUAL_PARAM_MAP
from ui.window_controller import WindowBackedController

logger = logging.getLogger(__name__)


class CaptureController(WindowBackedController):
    sig_capture_progress = Signal(str)
    sig_capture_status = Signal(str)
    sig_capture_stopped = Signal()

    def _on_source_changed(self, index):
        """Toggle between Controller SCOPE and Drive Scope UI."""
        is_drive = (index == 1)
        self.capture_source = 'drive' if is_drive else 'controller'

        # Controller SCOPE widgets
        for w in (self.ctrl_period_label, self.period_edit, self.ctrl_period_unit,
                  self.ctrl_duration_label, self.duration_edit, self.ctrl_duration_unit,
                  self.ctrl_mode_label, self.ctrl_mode_widget, self.external_trigger_chk):
            w.setVisible(not is_drive)

        # Drive Scope widgets
        for w in (self.drv_sample_label, self.drv_sample_edit, self.drv_sample_unit,
                  self.drv_trigger_label, self.drv_trigger_combo,
                  self.drv_axis_label, self.drv_axis_spin, self.drv_info_label):
            w.setVisible(is_drive)

        if is_drive:
            self._update_drive_info_label()
            self._on_drive_trigger_changed()  # show/hide trigger value inputs
            # Switch trace controls to drive variable mode
            for trace in self.traces:
                trace.set_drive_mode(True)
        else:
            self.drv_info_label.setText("")
            # Hide trigger value inputs when switching away from drive mode
            self.drv_trig_val_label.setVisible(False)
            self.drv_trig_val1_edit.setVisible(False)
            self.drv_trig_val2_edit.setVisible(False)
            for trace in self.traces:
                trace.set_drive_mode(False)

    def _get_drive_sample_time_units(self) -> int:
        """Convert capture duration (seconds) to drive sample_time units (×125 μs).

        sample_time_units = duration_s / (1000 × 125 μs)
        e.g. 1.0 s → 1.0 / 0.125 = 8 units → 8 × 125 μs = 1 ms per sample
        """
        try:
            duration_s = float(self.drv_sample_edit.text())
        except ValueError:
            return 8  # default → 1 ms/sample → 1 s capture
        # duration_s = 1000_samples × sample_time_units × 125e-6
        # sample_time_units = duration_s / (1000 × 125e-6) = duration_s / 0.125
        units = max(1, round(duration_s / 0.125))
        return units

    def _update_drive_info_label(self):
        """Update the drive scope info label and resolution display."""
        units = self._get_drive_sample_time_units()
        period_us = units * 125
        # Update resolution next to the "s" unit label
        if period_us >= 1000:
            res_str = f"{period_us / 1000:.2f} ms"
        else:
            res_str = f"{period_us} μs"
        self.drv_sample_unit.setText(f"s  (res: {res_str})")

    def _on_drive_trigger_changed(self):
        """Show/hide trigger value inputs based on selected trigger mode."""
        mode = self.drv_trigger_combo.currentData()
        # Modes needing a threshold: 1=Rising, 2=Falling, 3=Greater, 4=Less
        needs_value1 = mode in (1, 2, 3, 4, 5, 6)
        # Window modes need two thresholds: 5=Inside, 6=Outside
        needs_value2 = mode in (5, 6)

        is_drive = (self.capture_source == 'drive')
        self.drv_trig_val_label.setVisible(is_drive and needs_value1)
        self.drv_trig_val1_edit.setVisible(is_drive and needs_value1)
        self.drv_trig_val2_edit.setVisible(is_drive and needs_value2)

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

    def _start_drive_scope_capture(self):
        """Start drive-based scope capture via SDO protocol."""
        enabled_traces = self.get_enabled_traces()
        if not enabled_traces:
            QMessageBox.warning(self.window, "No Traces", "Enable at least one trace")
            return

        # Collect channel addresses from enabled traces.  Deduplicate: the
        # drive samples each address once and every trace using that variable
        # shares the same display key, so duplicates would only waste channel
        # slots and overwrite each other in the parsed params dict.
        channels = []
        display_names = []
        seen_addrs = set()
        for t in enabled_traces:
            addr = t.get_drive_variable_address()
            if addr and addr != 0 and addr not in seen_addrs:
                seen_addrs.add(addr)
                channels.append(addr)
                display_names.append(t.get_display_name())

        if not channels:
            QMessageBox.warning(self.window, "No Variables",
                                "Select at least one drive variable (not '(Disabled)')")
            return

        if len(channels) > DRIVE_NUM_CHANNELS:
            QMessageBox.warning(self.window, "Too Many Channels",
                                f"Drive scope supports max {DRIVE_NUM_CHANNELS} channels.\n"
                                f"You have {len(channels)} channels enabled.")
            return

        # Rebuild subplots
        self.curves = {}
        self.stats_texts = {}
        self._recreate_subplots()

        try:
            sample_time = self._get_drive_sample_time_units()
            trigger_mode = self.drv_trigger_combo.currentData()
            axis = self.drv_axis_spin.value()

            # Parse trigger values
            try:
                trigger_value1 = int(self.drv_trig_val1_edit.text())
            except ValueError:
                trigger_value1 = 0
            try:
                trigger_value2 = int(self.drv_trig_val2_edit.text())
            except ValueError:
                trigger_value2 = 0

            # Update drive scope engine axis
            self.drive_scope_engine.axis = axis

            # Configure
            config = self.drive_scope_engine.configure(
                channels=channels,
                sample_time=sample_time,
                trigger_mode=trigger_mode,
                trigger_value1=trigger_value1,
                trigger_value2=trigger_value2,
                display_names=display_names,
            )

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

            logger.info("Drive scope: %s", config)
            self.sig_capture_status.emit(
                f"Drive scope: {config['active_channels']} ch, "
                f"{config['sample_period_ms']:.2f} ms/sample, "
                f"{config['capture_duration_sec']*1000:.1f} ms capture"
            )

            # Start capture thread (always single-shot for drive scope)
            self.scope_thread = threading.Thread(
                target=self._drive_scope_capture_thread, daemon=True)
            self.scope_thread.start()

        except Exception as e:
            QMessageBox.critical(self.window, "Drive Scope Error", str(e))
            logger.exception("Drive scope start failed")

    def _drive_scope_capture_thread(self):
        """Background thread for drive scope: start → wait → download.

        NOTE: We must NOT hold _conn_lock for long periods — the watchdog
        needs it every 0.5s.  Acquire/release per-operation instead.
        """
        try:
            engine = self.drive_scope_engine

            # Step 1: Start capture on the drive
            self.sig_capture_status.emit("Drive scope: starting capture...")
            with self._conn_lock:
                engine.start_capture()

            # Step 2: Wait for capture to complete (poll with short lock holds)
            self.sig_capture_status.emit("Drive scope: sampling...")
            capture_timeout = max(30.0, engine.capture_duration_sec * 3)
            wait_start = time.monotonic()
            completed = False
            started_sampling = bool(getattr(engine, "last_start_saw_sampling", False))
            if started_sampling:
                logger.info(
                    "Drive scope start already observed sampling; status sequence=%s",
                    getattr(engine, "last_start_status_sequence", []),
                )

            # Small initial sleep to allow the start command to execute on the controller
            time.sleep(0.05)

            while (time.monotonic() - wait_start) < capture_timeout:
                if not self.is_running:
                    with self._conn_lock:
                        engine.stop_capture()
                    return

                with self._conn_lock:
                    status = engine.get_status()

                # Status: 0/2 (idle/done from previous), 1 (sampling), 2 (done)
                if status == 1:
                    started_sampling = True

                if status == 2 and started_sampling:
                    completed = True
                    break

                elapsed = time.monotonic() - wait_start
                if (
                    status == 2
                    and not started_sampling
                    and elapsed >= engine.capture_duration_sec + 0.5
                ):
                    raise RuntimeError(
                        "Drive scope did not report a fresh sampling state; "
                        "refusing to download stale drive scope data."
                    )

                if engine.capture_duration_sec > 0:
                    pct = min(0.99, elapsed / engine.capture_duration_sec)
                    self.sig_capture_progress.emit(f"Sampling: {pct*100:.0f}%")

                time.sleep(0.05)

            if not completed:
                self.sig_capture_status.emit("Drive scope: capture timed out")
                logger.warning("Drive scope capture timed out")
                return

            self.sig_capture_progress.emit("Sampling: 100%")

            # Step 3: Download data from drive via TABLE relay
            # This takes ~20s for 8000 words — stop the watchdog so it
            # doesn't kill the connection while we hold _conn_lock.
            self._stop_watchdog()
            self.sig_capture_status.emit("Drive scope: downloading data...")

            # Determine safe, writable local filename for the downloaded bin file
            import os
            from PySide6.QtCore import QStandardPaths
            local_filename = "drive_scope.bin"
            try:
                # Test write in current working directory
                test_file = "test_write_perm.tmp"
                with open(test_file, "w") as f:
                    f.write("")
                os.remove(test_file)
            except (IOError, OSError):
                # Fallback to Documents directory
                docs_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
                if docs_dir and os.path.isdir(docs_dir):
                    local_filename = os.path.join(docs_dir, "drive_scope.bin")
                else:
                    temp_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)
                    local_filename = os.path.join(temp_dir, "drive_scope.bin")
            
            abs_local_filename = os.path.abspath(local_filename)

            def _download_cb(pct, msg):
                self.sig_capture_progress.emit(msg)

            try:
                with self._conn_lock:
                    data = engine.read_data(
                        progress_callback=_download_cb,
                        local_filename=abs_local_filename
                    )
            finally:
                # Restart the watchdog now that the long operation is done,
                # even if the FIFO transfer raises.
                self._start_watchdog()

            if not self.is_running:
                return

            # Step 4: Push data into the display pipeline
            if data and data['num_samples'] > 0:
                self._push_data(data)
                self.sig_capture_status.emit(
                    f"Drive scope: captured {data['num_samples']} samples. "
                    f"File: {abs_local_filename}"
                )
            else:
                self.sig_capture_status.emit("Drive scope: no data captured")

        except Exception as e:
            self.sig_capture_status.emit(f"Drive scope error: {e}")
            logger.exception("Drive scope capture error")
        finally:
            self.is_running = False
            self.sig_capture_stopped.emit()

    def _arm_and_wait_for_external_trigger(self, auto_retrigger: bool = False) -> bool:
        """Arm SCOPE and wait for an external TRIGGER to start the capture.

        An external program commonly issues the one-shot ``TRIGGER`` command.
        When continuous mode is selected, promote that capture to
        ``TRIGGER(1)`` after detecting it so the UI mode remains authoritative.
        """
        self.scope_engine.arm_capture()
        self.sig_capture_status.emit("SCOPE armed; waiting for external TRIGGER...")
        self.sig_capture_progress.emit("Waiting for TRIGGER")

        try:
            initial_scope_pos = self.scope_engine.connection.GetSystemParameter_SCOPE_POS()
        except Exception as exc:
            logger.debug("Could not read initial SCOPE_POS after arming: %s", exc)
            initial_scope_pos = 0

        while self.is_running and self.trio_connected:
            try:
                scope_pos = self.scope_engine.connection.GetSystemParameter_SCOPE_POS()
            except Exception as exc:
                logger.debug("Could not read SCOPE_POS while waiting for TRIGGER: %s", exc)
                scope_pos = 0

            if (
                (initial_scope_pos == 0 and scope_pos > 0)
                or (initial_scope_pos != 0 and scope_pos != initial_scope_pos)
            ):
                self.scope_engine.is_capturing = True
                if auto_retrigger:
                    self.scope_engine.trigger_capture(auto_retrigger=True)
                self.sig_capture_status.emit("External TRIGGER detected; capturing...")
                return True

            time.sleep(0.010)

        try:
            self.scope_engine.stop_capture()
        except Exception:
            pass
        return False

    def _scope_single_external_trigger_thread(self):
        """Single-shot controller SCOPE started by a Trio BASIC TRIGGER command."""
        try:
            samples_per_param = (
                (self.scope_engine.table_end - self.scope_engine.table_start + 1)
                // self.scope_engine.num_params
            )
            sample_period = self.scope_engine.period_cycles * self.scope_engine.servo_period_sec
            expected_duration = samples_per_param * sample_period

            if not self._arm_and_wait_for_external_trigger():
                return

            last_sample_idx = 0
            capture_start = time.monotonic()
            timeout = max(10.0, expected_duration + 5.0)

            while self.is_running and self.trio_connected:
                batch_data, last_sample_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)
                if batch_data and batch_data['num_samples'] > 0:
                    self._push_data(batch_data)

                pct = min(100.0, (last_sample_idx / samples_per_param) * 100.0)
                self.sig_capture_progress.emit(f"Progress: {pct:.1f}%")

                if last_sample_idx >= samples_per_param:
                    break

                if time.monotonic() - capture_start > timeout:
                    self.sig_capture_status.emit("External-trigger capture timed out")
                    logger.warning("External-trigger capture timed out")
                    return

                # Each poll costs one network round-trip per parameter; the UI
                # renders at ~30fps, so polling faster than this buys nothing.
                time.sleep(0.025)

            if not self.is_running:
                self.scope_engine.stop_capture()
                return

            self.scope_engine.stop_capture()
            self.sig_capture_status.emit(f"Captured {last_sample_idx} samples")

        except Exception as e:
            self.sig_capture_status.emit(f"Error: {e}")
            logger.exception("External-trigger single capture error")
        finally:
            self.is_running = False
            self.sig_capture_stopped.emit()

    def _scope_continuous_external_trigger_thread(self):
        """Continuous controller SCOPE started by an external TRIGGER command."""
        try:
            samples_per_param = (
                (self.scope_engine.table_end - self.scope_engine.table_start + 1)
                // self.scope_engine.num_params
            )
            sample_period = self.scope_engine.period_cycles * self.scope_engine.servo_period_sec

            if not self._arm_and_wait_for_external_trigger(auto_retrigger=True):
                return

            last_sample_idx = 0
            sample_offset = 0
            self.sig_capture_status.emit("Capturing (external continuous)...")

            while self.is_running and self.trio_connected:
                batch_data, new_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)

                if batch_data and batch_data['num_samples'] > 0:
                    time_shift = sample_offset * sample_period
                    if time_shift > 0:
                        batch_data['time'] = batch_data['time'] + time_shift
                    self._push_data(batch_data)
                    last_sample_idx = new_idx
                else:
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
            logger.exception("External-trigger continuous capture error")
        finally:
            self.is_running = False
            self.sig_capture_stopped.emit()

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

