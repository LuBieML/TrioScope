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

logger = logging.getLogger(__name__)


# Channel-type parameters that use CHANNEL(n) instead of AXIS(n)
CHANNEL_PARAMETERS = {
    "AIN", "AINBI", "AOUT",
    "DV_CONTROLWORD", "DV_IN", "DV_OUT", "DV_STATUSWORD",
    "IN", "OUT",
}

# Complete set of axis parameters that require AXIS(n) suffix
# Based on Trio Motion Perfect v4 and the prompt specification
AXIS_PARAMETERS = {
    "ACCEL", "ACCEL_FACTOR", "ADDAX_AXIS", "AFF_GAIN", "ATYPE",
    "AXIS_A_OUTPUT", "AXIS_ACCEL", "AXIS_B_OUTPUT", "AXIS_BLENDING",
    "AXIS_D_OUTPUT", "AXIS_DEBUG_A", "AXIS_DEBUG_B", "AXIS_DEBUG_C",
    "AXIS_DEBUG_D", "AXIS_DECEL", "AXIS_DIRECTION", "AXIS_DPOS",
    "AXIS_ENABLE", "AXIS_ENABLE_OVERRIDE", "AXIS_ERROR_COUNT",
    "AXIS_FASTDEC", "AXIS_FS_LIMIT", "AXIS_MAX_ACC", "AXIS_MAX_JERK",
    "AXIS_MAX_JOG_ACC", "AXIS_MAX_JOG_SPEED", "AXIS_MAX_SPEED",
    "AXIS_MAX_TORQUE", "AXIS_MODE", "AXIS_RS_LIMIT", "AXIS_SPEED",
    "AXIS_UNITS", "AXIS_Z_OUTPUT", "AXISSTATUS",
    "BACKLASH_DIST", "CHANGE_DIR_LAST", "CLOSE_WIN", "CLOSE_WINB",
    "CLUTCH_RATE", "CORNER_FACTOR", "CORNER_MODE", "CORNER_RADIUS",
    "CORNER_STATE", "CREEP",
    "D_GAIN", "D_ZONE_MAX", "D_ZONE_MIN", "DAC", "DAC_OUT", "DAC_SCALE",
    "DATUM_IN", "DECEL", "DECEL_ANGLE", "DEMAND_EDGES", "DEMAND_SPEED",
    "DISTANCE_TO_SYNC", "DPOS",
    "DRIVE_BRAKE_OUTPUT", "DRIVE_CONTROL", "DRIVE_CONTROLWORD",
    "DRIVE_CURRENT", "DRIVE_CW_MODE", "DRIVE_ENABLE", "DRIVE_FE",
    "DRIVE_FE_LIMIT", "DRIVE_INDEX", "DRIVE_INPUTS", "DRIVE_MODE",
    "DRIVE_MONITOR", "DRIVE_NEG_TORQUE", "DRIVE_PARAMETER",
    "DRIVE_POS_TORQUE", "DRIVE_PROFILE", "DRIVE_REP_DIST",
    "DRIVE_SET_VAL", "DRIVE_STATUS", "DRIVE_TORQUE", "DRIVE_TYPE",
    "DRIVE_VALUE", "DRIVE_VELOCITY",
    "ENCODER", "ENCODER_BITS", "ENCODER_CONTROL", "ENCODER_FILTER",
    "ENCODER_ID", "ENCODER_STATUS", "ENCODER_TURNS",
    "END_DIR_LAST", "END_VELOCITY", "ENDMOVE", "ENDMOVE_BUFFER",
    "ENDMOVE_SPEED",
    "FAST_JOG", "FASTDEC", "FASTJERK", "FE", "FE_LATCH", "FE_LIMIT",
    "FE_LIMIT_MODE", "FE_RANGE", "FEED_OVERRIDE", "FHOLD_IN", "FHSPEED",
    "FORCE_ACCEL", "FORCE_DECEL", "FORCE_DWELL", "FORCE_JERK",
    "FORCE_RAMP", "FORCE_SPEED", "FRAME_REP_DIST", "FRAME_SCALE",
    "FS_LIMIT", "FULL_SP_RADIUS", "FWD_IN", "FWD_JOG", "FWD_START",
    "I_GAIN", "IDLE", "IN_POS", "IN_POS_DIST", "IN_POS_SPEED",
    "INTERP_FACTOR", "INVERT_STEP",
    "JERK", "JERK_FACTOR", "JOGSPEED",
    "LINK_AXIS", "LOADED", "LOOKAHEAD_FACTOR",
    "MARK", "MARKB", "MERGE", "MICROSTEP",
    "MOVE_COUNT", "MOVE_COUNT_INC", "MOVE_ENDMOVE_SPEED",
    "MOVE_FORCE_ACCEL", "MOVE_FORCE_DECEL", "MOVE_FORCE_RAMP",
    "MOVE_FORCE_SPEED", "MOVE_PA", "MOVE_PA_CONT", "MOVE_PA_IDLE",
    "MOVE_PB", "MOVE_PB_CONT", "MOVE_PB_IDLE",
    "MOVE_STARTMOVE_SPEED", "MOVELINK_MODIFY", "MOVES_BUFFERED",
    "MPOS", "MSPEED", "MSPEED_FILTER", "MSPEEDF", "MTYPE",
    "NEG_OFFSET", "NTYPE",
    "OFFPOS", "OPEN_WIN", "OPEN_WINB", "OUTLIMIT", "OV_GAIN",
    "P_GAIN", "POS_DELAY", "POS_OFFSET", "POSI_SEQ_DELAY",
    "POSI_SEQ_MODE", "PP_STEP", "PS_ENCODER", "PWM_CYCLE", "PWM_MARK",
    "RAISE_ANGLE", "REG_INPUTS", "REG_POS", "REG_POSB",
    "REGIST_CONTROL", "REGIST_DELAY", "REGIST_SPEED", "REGIST_SPEEDB",
    "REMAIN", "REMAIN_TIME", "REP_DIST", "REP_OPTION", "REV_IN",
    "REV_JOG", "REV_START",
    "ROBOT_DPOS", "ROBOT_FE", "ROBOT_FE_LATCH", "ROBOT_FE_LIMIT",
    "ROBOT_FE_RANGE", "ROBOT_FS_LIMIT", "ROBOT_RS_LIMIT",
    "ROBOT_SP_MODE", "ROBOT_UNITS", "ROBOTSTATUS", "RS_LIMIT",
    "S_REF", "S_REF_OUT", "SERVO", "SPEED", "SPEED_FACTOR",
    "SPEED_SIGN", "SRAMP", "START_DIR_LAST", "STARTMOVE_SPEED",
    "STOP_ANGLE", "STOPPING_DISTANCE",
    "SYNC_AXIS", "SYNC_CONTROL", "SYNC_DPOS", "SYNC_DWELL",
    "SYNC_FLIGHT", "SYNC_PAUSE", "SYNC_TIME", "SYNC_TIMER",
    "SYNC_WITHDRAW",
    "T_REF", "T_REF_OUT", "TABLE_POINTER", "TANG_DIRECTION",
    "TCP_DPOS", "TCP_FS_LIMIT", "TCP_RS_LIMIT", "TCP_UNITS",
    "TRIOPCTESTVARIAB",
    "UNITS", "V_LIMIT", "VECTOR_BUFFERED", "VERIFY", "VFF_GAIN",
    "VP_ACCEL", "VP_DEMAND_ACCEL", "VP_DEMAND_DECEL", "VP_DEMAND_JERK",
    "VP_DEMAND_SPEED", "VP_ERROR", "VP_JERK", "VP_MODE", "VP_OPTIONS",
    "VP_POSITION", "VP_SPEED",
    "WORLD_ACCEL", "WORLD_DECEL", "WORLD_DPOS", "WORLD_FASTDEC",
    "WORLD_FS_LIMIT", "WORLD_JERK", "WORLD_JOGSPEED",
    "WORLD_RS_LIMIT", "WORLD_SPEED", "WORLD_UNITS",
}


class ScopeParameterParser:
    """
    Parses user-friendly parameter strings into SCOPE-compatible format.

    Handles various input formats:
    - MPOS(0) or ?mpos(0) → "MPOS AXIS(0)"
    - MPOS or ?mpos → "MPOS AXIS(0)" (default axis 0)
    - VR(5) → "VR(5)"
    - TABLE(100) → "TABLE(100)"
    - Multiple params: "MPOS(0), DPOS(0), FE(0)" → ["MPOS AXIS(0)", "DPOS AXIS(0)", "FE AXIS(0)"]
    """

    @staticmethod
    def parse_parameter_string(param_str: str) -> Tuple[str, str]:
        """
        Parse a single parameter string into SCOPE format.

        Args:
            param_str: User input like "MPOS(0)" or "VR(5)"

        Returns:
            Tuple of (scope_param_string, display_name)
            Example: ("MPOS AXIS(0)", "MPOS(0)")

        Raises:
            ValueError: If parameter format is invalid
        """
        param_str = param_str.strip()

        # Remove leading '?' if present
        if param_str.startswith('?'):
            param_str = param_str[1:]

        # Pattern 1: VR(index)
        vr_match = re.match(r'^VR\s*\(\s*(\d+)\s*\)$', param_str, re.IGNORECASE)
        if vr_match:
            index = vr_match.group(1)
            return f"VR({index})", f"VR({index})"

        # Pattern 2: TABLE(index)
        table_match = re.match(r'^TABLE\s*\(\s*(\d+)\s*\)$', param_str, re.IGNORECASE)
        if table_match:
            index = table_match.group(1)
            return f"TABLE({index})", f"TABLE({index})"

        # Pattern 3: PARAM(index) - axis or channel parameter with explicit index
        indexed_param_match = re.match(r'^(\w+)\s*\(\s*(\d+)\s*\)$', param_str, re.IGNORECASE)
        if indexed_param_match:
            param_name = indexed_param_match.group(1).upper()
            index_num = indexed_param_match.group(2)

            if param_name in CHANNEL_PARAMETERS:
                return f"{param_name}({index_num})", f"{param_name} Ch({index_num})"
            elif param_name in AXIS_PARAMETERS:
                return f"{param_name} AXIS({index_num})", f"{param_name}({index_num})"
            else:
                # Unknown parameter - might be valid on controller
                logger.warning(f"Unknown parameter: {param_name}")
                return f"{param_name} AXIS({index_num})", f"{param_name}({index_num})"

        # Pattern 4: PARAM - axis/channel parameter without explicit index (default to 0)
        param_only_match = re.match(r'^(\w+)$', param_str, re.IGNORECASE)
        if param_only_match:
            param_name = param_only_match.group(1).upper()

            if param_name in CHANNEL_PARAMETERS:
                return f"{param_name}(0)", f"{param_name} Ch(0)"
            elif param_name in AXIS_PARAMETERS:
                return f"{param_name} AXIS(0)", f"{param_name}(0)"
            else:
                # Might be a system parameter (no axis needed)
                return param_name, param_name

        raise ValueError(f"Invalid parameter format: {param_str}")

    @staticmethod
    def parse_multiple_parameters(params_str: str) -> Tuple[List[str], List[str]]:
        """
        Parse comma-separated parameter list.

        Args:
            params_str: Comma-separated parameters like "MPOS(0), DPOS(0), FE(0)"

        Returns:
            Tuple of (scope_params_list, display_names_list)

        Raises:
            ValueError: If any parameter is invalid
        """
        param_strs = [p.strip() for p in params_str.split(',') if p.strip()]

        scope_params = []
        display_names = []

        for param_str in param_strs:
            scope_param, display_name = ScopeParameterParser.parse_parameter_string(param_str)
            scope_params.append(scope_param)
            display_names.append(display_name)

        return scope_params, display_names


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
        if not param_strings:
            raise ValueError("No parameters specified")

        if self.servo_period_sec is None:
            raise ValueError("Servo period not read. Call read_servo_period() first.")

        if self.tsize is None:
            raise ValueError("TABLE size not read. Call read_table_size() first.")

        self.scope_params = param_strings
        self.display_names = display_names
        self.num_params = len(param_strings)
        self.period_cycles = period_cycles
        self.table_start = table_start

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

    def start_capture(self, auto_retrigger=False):
        """
        Start SCOPE capture to TABLE.

        According to Trio documentation:
        1. SCOPE(ON, ...) arms the scope
        2. TRIGGER starts capturing (one-shot)
        3. TRIGGER(1) starts capturing with auto-retrigger at end of each scan

        Args:
            auto_retrigger: If True, use TRIGGER(1) so the controller
                           automatically restarts capture when the buffer fills.

        Raises:
            Exception: If SCOPE or TRIGGER fails
        """
        try:
            # WORKAROUND: The Python wrapper's ScopeOn() method has a bug with string parameters
            # (pybind11 issue: "NumPy type info missing for class std::basic_string_view")
            # Solution: Use Execute() to call the SCOPE TrioBASIC command directly

            # Build SCOPE command string: SCOPE(ON, period, start, end, param1, param2, ...)
            # CRITICAL: Parameter names must NOT be quoted!
            # The Trio BASIC parser expects unquoted parameter names like: MPOS AXIS(0)
            # Using quotes causes SCOPE to arm but never capture (SCOPE_POS stays at 0)
            params_str = ", ".join(self.scope_params)
            scope_command = f"SCOPE(ON, {self.period_cycles}, {self.table_start}, {self.table_end}, {params_str})"

            logger.debug(f"Arming SCOPE: {scope_command}")
            self.connection.Execute(scope_command)

            if auto_retrigger:
                self.connection.Execute("TRIGGER(1)")
                logger.debug("SCOPE capture started (auto-retrigger)")
            else:
                self.connection.Execute("TRIGGER")
                logger.debug("SCOPE capture started (single-shot)")

        except Exception as e:
            logger.error(f"Failed to start SCOPE: {e}")
            raise

    def stop_capture(self):
        """
        Stop SCOPE capture.

        Uses SCOPE(OFF) to disable the scope completely.

        Raises:
            Exception: If SCOPE(OFF) fails
        """
        try:
            # Use Execute to send SCOPE(OFF) command
            # Note: ScopeOff() method also works, but Execute is more explicit
            self.connection.Execute("SCOPE(OFF)")
            self.is_capturing = False
            logger.debug("SCOPE capture stopped")
        except Exception as e:
            logger.error(f"Failed to stop SCOPE: {e}")
            # Try the ScopeOff() method as fallback
            try:
                self.connection.ScopeOff()
                self.is_capturing = False
                logger.debug("SCOPE stopped via ScopeOff() fallback")
            except:
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
            # Create pre-allocated numpy array
            raw = np.zeros(count, dtype=np.float64)
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

            # Read new data from each parameter's block
            samples_per_param = (self.table_end - self.table_start + 1) // self.num_params

            for i, (param_name, display_name) in enumerate(zip(self.scope_params, self.display_names)):
                # Calculate position in this parameter's block
                param_block_start = self.table_start + (i * samples_per_param)
                read_start = param_block_start + last_read_pos
                read_count = new_samples

                # Read this parameter's new data
                param_data = np.zeros(read_count, dtype=np.float64)
                self.connection.GetMultiTableValues(read_start, read_count, param_data)
                result['params'][display_name] = param_data

            return result, actual_end

        except Exception as e:
            logger.error(f"Failed to read new data: {e}")
            return None, last_read_pos
