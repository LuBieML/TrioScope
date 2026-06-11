"""Drive scope (SDO) capture: start validation and capture thread."""

import logging
import os
import threading
import time

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QMessageBox

from scope.drive_scope_engine import NUM_CHANNELS as DRIVE_NUM_CHANNELS

logger = logging.getLogger(__name__)


class DriveScopeCaptureMixin:
    """Drive-based scope capture via the SDO COMBO protocol."""

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
