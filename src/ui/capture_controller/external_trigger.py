"""Externally triggered controller SCOPE capture (Trio BASIC TRIGGER)."""

import logging
import time

logger = logging.getLogger(__name__)


class ExternalTriggerCaptureMixin:
    """Arm-and-wait capture threads driven by an external TRIGGER command."""

    def _arm_and_wait_for_external_trigger(self) -> bool:
        """Arm controller SCOPE and wait until SCOPE_POS shows TRIGGER activity."""
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
        """Continuous controller SCOPE started by a Trio BASIC TRIGGER(1) command."""
        try:
            samples_per_param = (
                (self.scope_engine.table_end - self.scope_engine.table_start + 1)
                // self.scope_engine.num_params
            )
            sample_period = self.scope_engine.period_cycles * self.scope_engine.servo_period_sec

            if not self._arm_and_wait_for_external_trigger():
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
