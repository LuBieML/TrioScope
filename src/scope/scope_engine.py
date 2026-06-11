"""
Trio Controller SCOPE Engine

Manages the lifecycle of SCOPE-based parameter capture using the Trio controller's
built-in SCOPE command for deterministic, high-speed data acquisition at servo rates.

This provides much better performance and accuracy than PC-side polling:
- Deterministic timing (controller servo clock)
- No jitter from PC load, network, or USB latency
- Multi-parameter synchronous capture
- Bulk data read-back after capture
"""

import numpy as np
import re
import logging
from typing import Optional, List, Dict, Tuple, Any

try:
    import Trio_UnifiedApi as TUA
except ImportError:
    TUA = None

from .parameter_parser import ScopeParameterParser

logger = logging.getLogger(__name__)

__all__ = ["ScopeEngine", "ScopeParameterParser"]


class ScopeEngine:
    """
    Manages Trio SCOPE capture lifecycle.

    Provides deterministic, high-speed parameter capture using the controller's
    built-in SCOPE command. Data is captured at servo rate directly on the
    controller and then bulk-read to the PC.
    """

    def __init__(self, connection):
        """
        Initialize SCOPE engine.

        Args:
            connection: Trio_UnifiedApi connection object
        """
        self.connection = connection
        self.servo_period_us = None      # Servo period in microseconds
        self.servo_period_sec = None     # Servo period in seconds
        self.is_capturing = False
        self.table_start = 0
        self.table_end = 0
        self.num_params = 0
        self.period_cycles = 1
        self.scope_params = []           # Formatted strings for ScopeOn
        self.display_names = []          # User-friendly names for plotting
        self.tsize = None                # Controller TABLE size
        self.is_armed = False
        self._armed_via_execute_fallback = False
        self._scope_on_requires_execute_fallback = False

    @staticmethod
    def _scope_command_param(param: str) -> str:
        """Return a SCOPE command-line compatible parameter token."""
        param = param.strip()
        if not param:
            raise ValueError("SCOPE parameter cannot be empty")
        out_match = re.match(r'^OUT\s*\(\s*(\d+)\s*\)$', param, re.IGNORECASE)
        if out_match:
            return f"READ_OP({out_match.group(1)})"
        return param

    def _execute_arm_scope(self) -> None:
        params_str = ", ".join(
            self._scope_command_param(param) for param in self.scope_params
        )
        scope_command = (
            f"SCOPE(ON, {self.period_cycles}, {self.table_start}, "
            f"{self.table_end}, {params_str})"
        )

        logger.debug("Arming SCOPE via Execute fallback: %s", scope_command)
        self.connection.Execute(scope_command)

    def _execute_trigger(self, auto_retrigger: bool) -> None:
        trigger_command = "TRIGGER(1)" if auto_retrigger else "TRIGGER"

        logger.debug("Starting SCOPE via Execute fallback: %s", trigger_command)
        self.connection.Execute(trigger_command)

    def read_servo_period(self) -> float:
        """
        Read SERVO_PERIOD from controller.

        Returns:
            Servo period in seconds

        Raises:
            Exception: If read fails
        """
        try:
            # GetSystemParameter_SERVO_PERIOD returns microseconds
            self.servo_period_us = self.connection.GetSystemParameter_SERVO_PERIOD()
            self.servo_period_sec = self.servo_period_us / 1_000_000.0
            logger.info(f"Servo period: {self.servo_period_us} μs ({self.servo_period_sec*1000:.3f} ms)")
            return self.servo_period_sec
        except Exception as e:
            logger.error(f"Failed to read SERVO_PERIOD: {e}")
            raise

    def read_table_size(self) -> int:
        """
        Read TSIZE (maximum TABLE size) from controller.

        Returns:
            Maximum TABLE size

        Raises:
            Exception: If read fails
        """
        try:
            self.tsize = self.connection.GetSystemParameter_TSIZE()
            logger.info(f"TABLE size: {self.tsize}")
            return self.tsize
        except Exception as e:
            logger.error(f"Failed to read TSIZE: {e}")
            raise

    def configure(self, param_strings: List[str], display_names: List[str],
                  period_cycles: int, duration_seconds: float,
                  table_start: int = 0) -> Dict[str, Any]:
        """
        Configure SCOPE capture.

        Args:
            param_strings: List of SCOPE-formatted parameter strings
                          e.g. ["MPOS AXIS(0)", "DPOS AXIS(0)", "FE AXIS(0)"]
            display_names: List of user-friendly names for display
            period_cycles: Capture every N servo cycles (1 = every cycle)
            duration_seconds: Desired capture duration
            table_start: Starting TABLE index for data storage

        Returns:
            Dict with configuration info:
                - sample_period_sec: Time between samples
                - total_samples: Number of samples to capture
                - total_table_entries: Total TABLE entries needed
                - table_end: Ending TABLE index

        Raises:
            ValueError: If TABLE range exceeds TSIZE or parameters invalid
        """
        clean_params = [
            str(param).strip() if param is not None else ""
            for param in param_strings
        ]
        if not clean_params:
            raise ValueError("No parameters specified")
        for idx, param in enumerate(clean_params, start=1):
            if not param:
                raise ValueError(f"SCOPE parameter {idx} is empty")

        if self.servo_period_sec is None:
            raise ValueError("Servo period not read. Call read_servo_period() first.")

        if self.tsize is None:
            raise ValueError("TABLE size not read. Call read_table_size() first.")

        self.scope_params = clean_params
        self.display_names = display_names
        self.num_params = len(clean_params)
        self.period_cycles = period_cycles
        self.table_start = table_start
        self.is_armed = False
        self.is_capturing = False
        self._armed_via_execute_fallback = False

        # Calculate TABLE range needed
        sample_period_sec = period_cycles * self.servo_period_sec
        total_samples = int(duration_seconds / sample_period_sec)

        if total_samples < 1:
            raise ValueError(f"Duration too short. Minimum: {sample_period_sec:.6f} seconds")

        total_table_entries = total_samples * self.num_params
        self.table_end = table_start + total_table_entries - 1

        # Check against TSIZE
        if self.table_end >= self.tsize:
            raise ValueError(
                f"TABLE range {table_start}..{self.table_end} exceeds TSIZE ({self.tsize}). "
                f"Reduce duration, increase period_cycles, or reduce number of parameters."
            )

        config_info = {
            'sample_period_sec': sample_period_sec,
            'sample_period_ms': sample_period_sec * 1000,
            'total_samples': total_samples,
            'total_table_entries': total_table_entries,
            'table_start': self.table_start,
            'table_end': self.table_end,
            'num_params': self.num_params,
        }

        logger.info(f"SCOPE configured: {total_samples} samples @ {sample_period_sec*1000:.3f} ms "
                   f"({total_table_entries} TABLE entries)")

        return config_info

    def arm_capture(self):
        """
        Arm SCOPE capture to TABLE without starting sampling.

        According to Trio documentation:
        1. SCOPE(ON, ...) loads/arms the scope.
        2. TRIGGER starts sampling later.

        Raises:
            Exception: If SCOPE fails
        """
        try:
            logger.debug(
                "Arming SCOPE: period=%s, table=%s..%s, params=%s",
                self.period_cycles,
                self.table_start,
                self.table_end,
                self.scope_params,
            )
            if self._scope_on_requires_execute_fallback:
                self._execute_arm_scope()
                self._armed_via_execute_fallback = True
            else:
                try:
                    self.connection.ScopeOn(
                        self.period_cycles,
                        self.table_start,
                        self.table_end,
                        list(self.scope_params),
                    )
                    self._armed_via_execute_fallback = False
                except RuntimeError as e:
                    if "std::basic_string_view" not in str(e):
                        raise
                    self._scope_on_requires_execute_fallback = True
                    self._execute_arm_scope()
                    self._armed_via_execute_fallback = True
            self.is_armed = True
            self.is_capturing = False
            logger.debug("SCOPE armed")

        except Exception as e:
            logger.error(f"Failed to arm SCOPE: {e}")
            raise

    def trigger_capture(self, auto_retrigger=False):
        """
        Start an already armed SCOPE capture.

        Args:
            auto_retrigger: If True, use TRIGGER(1) so the controller
                           automatically restarts capture when the buffer fills.

        Raises:
            RuntimeError: If SCOPE has not been armed.
            Exception: If TRIGGER fails.
        """
        if not self.is_armed:
            raise RuntimeError("SCOPE is not armed — call arm_capture() first")

        try:
            if self._armed_via_execute_fallback:
                self._execute_trigger(auto_retrigger)
            else:
                try:
                    self.connection.Trigger(auto_retrigger)
                except AttributeError:
                    self._execute_trigger(auto_retrigger)
            self.is_capturing = True

            if auto_retrigger:
                logger.debug("SCOPE capture started (auto-retrigger)")
            else:
                logger.debug("SCOPE capture started (single-shot)")

        except Exception as e:
            logger.error(f"Failed to start SCOPE: {e}")
            raise

    def start_capture(self, auto_retrigger=False):
        """
        Arm and immediately start SCOPE capture to TABLE.

        This preserves the original TrioScope RUN behaviour. Use
        arm_capture() when an external Trio BASIC program will issue TRIGGER.
        """
        self.arm_capture()
        self.trigger_capture(auto_retrigger)

    def stop_capture(self):
        """
        Stop SCOPE capture.

        Uses SCOPE(OFF) to disable the scope completely.

        Raises:
            Exception: If SCOPE(OFF) fails
        """
        try:
            self.connection.ScopeOff()
            self.is_armed = False
            self.is_capturing = False
            self._armed_via_execute_fallback = False
            logger.debug("SCOPE capture stopped")
        except Exception as e:
            logger.error(f"Failed to stop SCOPE: {e}")
            raise

    def get_capture_progress(self) -> Tuple[int, int, float]:
        """
        Get current capture progress.

        Returns:
            Tuple of (current_entries, total_entries, percent_complete)
        """
        try:
            scope_pos = self.connection.GetSystemParameter_SCOPE_POS()
            # SCOPE_POS is 0-based (relative to capture start, not absolute TABLE index)
            current_entries = scope_pos
            total_entries = self.table_end - self.table_start + 1
            percent = (current_entries / total_entries * 100) if total_entries > 0 else 0
            return (current_entries, total_entries, percent)
        except Exception as e:
            logger.error(f"Failed to read SCOPE_POS: {e}")
            return (0, 0, 0.0)

    def is_capture_complete(self) -> bool:
        """
        Check if capture has filled the TABLE range.

        Note: SCOPE_POS wraps around within the TABLE range in continuous mode.
        For single-shot, it stops at table_end+1.
        For continuous, we need to detect when buffer is full by checking wrap-around.

        Returns:
            True if capture is complete (single-shot reached end)
        """
        try:
            scope_pos = self.connection.GetSystemParameter_SCOPE_POS()
            # SCOPE_POS is 0-based; capture complete when it reaches total entries
            total_entries = self.table_end - self.table_start + 1
            return scope_pos >= total_entries
        except Exception as e:
            logger.error(f"Failed to check capture completion: {e}")
            return False

    def read_captured_data(self, start: Optional[int] = None,
                          count: Optional[int] = None) -> Dict[str, Any]:
        """
        Read captured data from TABLE and de-interleave.

        Args:
            start: Starting TABLE index (defaults to table_start)
            count: Number of TABLE entries to read (defaults to all)

        Returns:
            Dict containing:
                'time': np.array of time values in seconds
                'sample_period': Sample period in seconds
                'num_samples': Number of samples
                'params': Dict mapping parameter names to np.array of values

        Raises:
            Exception: If TABLE read fails
        """
        if start is None:
            start = self.table_start
        if count is None:
            count = self.table_end - self.table_start + 1

        try:
            # Bulk read TABLE data
            # GetMultiTableValues(start, count, output_array) fills output_array
            # Create pre-allocated numpy array (empty: fully overwritten below)
            raw = np.empty(count, dtype=np.float64)
            self.connection.GetMultiTableValues(start, count, raw)

            # IMPORTANT: SCOPE stores data in SEQUENTIAL BLOCKS, not interleaved!
            # Example with 2 params, 500 samples:
            #   TABLE[0..499]   = MPOS samples
            #   TABLE[500..999] = DPOS samples
            # This is different from interleaved: [MPOS0, DPOS0, MPOS1, DPOS1, ...]

            # Calculate samples per parameter
            num_samples = len(raw) // self.num_params

            # Build time array
            sample_period_sec = self.period_cycles * self.servo_period_sec
            time_array = np.arange(num_samples) * sample_period_sec

            result = {
                'time': time_array,
                'sample_period': sample_period_sec,
                'num_samples': num_samples,
                'params': {}
            }

            # Extract sequential blocks for each parameter
            for i, (param_name, display_name) in enumerate(zip(self.scope_params, self.display_names)):
                block_start = i * num_samples
                block_end = (i + 1) * num_samples
                result['params'][display_name] = raw[block_start:block_end]

            logger.info(f"Read {num_samples} samples from TABLE")

            return result

        except Exception as e:
            logger.error(f"Failed to read TABLE data: {e}")
            raise

    def read_new_data(self, last_read_pos: int, max_samples: int = 0) -> Tuple[Optional[Dict], int]:
        """
        Read only newly captured data since last_read_pos.
        For live/streaming display.

        Note: With sequential storage, we need to read from all parameter blocks.

        Args:
            last_read_pos: Last sample index that was read (not TABLE position!)
            max_samples: Maximum number of samples to read (0 = read all available)

        Returns:
            Tuple of (data_dict or None, new_last_sample_index)
        """
        try:
            current_pos = self.connection.GetSystemParameter_SCOPE_POS()

            # SCOPE_POS is 0-based (relative to capture start, not absolute TABLE index)
            current_sample_idx = current_pos

            if current_sample_idx <= last_read_pos:
                return None, last_read_pos

            # Calculate how many new samples to read
            new_samples = current_sample_idx - last_read_pos

            # Apply max_samples limit if specified
            if max_samples > 0 and new_samples > max_samples:
                new_samples = max_samples

            if new_samples <= 0:
                return None, last_read_pos

            # Calculate the actual end position after limiting samples
            actual_end = last_read_pos + new_samples

            # Build time array for new samples
            sample_period_sec = self.period_cycles * self.servo_period_sec
            time_array = np.arange(last_read_pos, actual_end) * sample_period_sec

            result = {
                'time': time_array,
                'sample_period': sample_period_sec,
                'num_samples': new_samples,
                'params': {}
            }

            # Read new data from each parameter's block.
            samples_per_param = (self.table_end - self.table_start + 1) // self.num_params

            # Each GetMultiTableValues call costs one controller round-trip,
            # so when several parameters are due, prefer a single read that
            # spans all their regions (including the gaps between blocks) and
            # slice it locally. Worth it only when the gaps waste at most as
            # much as the useful data — with small streaming batches the gaps
            # dominate and per-parameter reads win.
            span = (self.num_params - 1) * samples_per_param + new_samples
            useful = self.num_params * new_samples
            if self.num_params > 1 and span <= 2 * useful:
                raw = np.empty(span, dtype=np.float64)
                self.connection.GetMultiTableValues(
                    self.table_start + last_read_pos, span, raw)
                for i, (param_name, display_name) in enumerate(
                        zip(self.scope_params, self.display_names)):
                    offset = i * samples_per_param
                    result['params'][display_name] = raw[offset:offset + new_samples]
                return result, actual_end

            for i, (param_name, display_name) in enumerate(zip(self.scope_params, self.display_names)):
                # Calculate position in this parameter's block
                param_block_start = self.table_start + (i * samples_per_param)
                read_start = param_block_start + last_read_pos
                read_count = new_samples

                # Read this parameter's new data (empty: fully overwritten)
                param_data = np.empty(read_count, dtype=np.float64)
                self.connection.GetMultiTableValues(read_start, read_count, param_data)
                result['params'][display_name] = param_data

            return result, actual_end

        except Exception as e:
            logger.error(f"Failed to read new data: {e}")
            return None, last_read_pos
