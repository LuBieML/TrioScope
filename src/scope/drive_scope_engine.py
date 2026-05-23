"""
Drive-Based Scope Engine (COMBO protocol)

Captures internal servo drive variables using the drive's built-in scope
via CoE SDO objects. This provides access to fast internal variables
(current loops, observer estimates) sampled at the drive's internal rate
(125μs per unit), which is faster than the Trio controller servo rate.

Protocol (from IPD-PLN-T22 COMBO document):
    0x368C  Setup:    trigger mode, thresholds, channel addresses, sample time
    0x368B  Control:  start/stop capture
    0x3680  Status:   bits 14-15 indicate capture state
    0x3687  Data:     16000-byte domain (8000 words, 8 channels × 1000 samples)

Data layout is interleaved:
    Word 0-7:   Ch1[0], Ch2[0], ..., Ch8[0]
    Word 8-15:  Ch1[1], Ch2[1], ..., Ch8[1]
    ...
"""

import logging
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import Trio_UnifiedApi as TUA
except ImportError:
    TUA = None

logger = logging.getLogger(__name__)

# ── SDO object indices ──────────────────────────────────────────────────
SETUP_INDEX = 0x368C     # Capture setup (sub 1–15)
CONTROL_INDEX = 0x368B   # Start/stop (sub 0)
STATUS_INDEX = 0x3680    # Capture status (bits 14-15)
DATA_INDEX = 0x3687      # Capture data buffer (domain, 16000 bytes)

# ── Constants ───────────────────────────────────────────────────────────
NUM_CHANNELS = 8
SAMPLES_PER_CHANNEL = 1000
TOTAL_WORDS = NUM_CHANNELS * SAMPLES_PER_CHANNEL  # 8000
EXPECTED_CAPTURE_BYTES = TOTAL_WORDS * 2
FIFO_CHUNK_BYTES = 0x8100
FIFO_CONTAINER_PREFIX_BYTES = 0x100
SAMPLE_TIME_UNIT_US = 125  # each sample time unit = 125 μs

# ── Trigger modes ───────────────────────────────────────────────────────
TRIGGER_MODES = {
    0: "Free Run (no trigger)",
    1: "Rising Edge",
    2: "Falling Edge",
    3: "Greater Than",
    4: "Less Than",
    5: "Window Inside",
    6: "Window Outside",
}

# ── Data type codes (for Channel1 variable data type) ───────────────────
DATA_TYPES = {
    1: ("Int16", np.int16),
    2: ("Uint16", np.uint16),
    3: ("Int32", np.int32),
    4: ("Uint32", np.uint32),
    5: ("Int64", np.int64),
    6: ("Uint64", np.uint64),
}

# ── Common drive variable addresses ─────────────────────────────────────
DRIVE_VARIABLES = {
    0x0F10: ("SPD_FB_RPM", "Speed feedback", "rpm", 1, "Int16"),
    0x0F11: ("SPD_CMD_RPM", "Speed command", "rpm", 1, "Int16"),
    0x0F13: ("TN", "Torque command %", "%Tn", 1, "Int16"),
    0x0F16: ("CURRENT_POS_L1", "Current pos low 16b", "pulse", 5, "Int64"),
    0x0F17: ("CURRENT_POS_H1", "Current pos mid-low 16b", "pulse", 5, "Int64"),
    0x0F18: ("CURRENT_POS_L2", "Current pos mid-high 16b", "pulse", 5, "Int64"),
    0x0F19: ("CURRENT_POS_H2", "Current pos high 16b", "pulse", 5, "Int64"),
    0x0F1C: ("IU", "Phase U current", "0.1%rated", 1, "Int16"),
    0x0F1D: ("IV", "Phase V current", "0.1%rated", 1, "Int16"),
    0x0F1E: ("ID_REF", "Id reference", "0.1%rated", 1, "Int16"),
    0x0F1F: ("ID", "Id actual", "0.1%rated", 1, "Int16"),
    0x0F20: ("IQ_REF", "Iq reference", "0.1%rated", 1, "Int16"),
    0x0F21: ("IQ", "Iq actual", "0.1%rated", 1, "Int16"),
    0x0F22: ("UD", "Ud voltage", "V", 2, "Uint16"),
    0x0F23: ("UQ", "Uq voltage", "V", 2, "Uint16"),
    0x0F2A: ("EST_SPD_L", "Observer speed low 16b", "0.1rpm", 3, "Int32"),
    0x0F2B: ("EST_SPD_H", "Observer speed high 16b", "0.1rpm", 3, "Int32"),
    0x0F2C: ("EST_TORQ_PER", "Observer torque", "0.1%rated", 1, "Int16"),
    0x0F2D: ("FF_SPEED", "Speed feedforward", "rpm", 1, "Int16"),
    0x0F2E: ("FF_TORQUE", "Torque feedforward", "0.1%rated", 2, "Uint16"),
    0x0F2F: ("PGERR_SPEED", "Pos cmd speed", "rpm", 1, "Int16"),
    0x0F32: ("EK_L1", "Pos error low 16b", "enc pulse", 5, "Int64"),
    0x0F33: ("EK_H1", "Pos error mid-low 16b", "enc pulse", 5, "Int64"),
    0x0F34: ("EK_L2", "Pos error mid-high 16b", "enc pulse", 5, "Int64"),
    0x0F35: ("EK_H2", "Pos error high 16b", "enc pulse", 5, "Int64"),
    0x0F36: ("PG_L1", "Pos cmd low 16b", "pulse", 5, "Int64"),
    0x0F37: ("PG_H1", "Pos cmd mid-low 16b", "pulse", 5, "Int64"),
    0x0F38: ("PG_L2", "Pos cmd mid-high 16b", "pulse", 5, "Int64"),
    0x0F39: ("PG_H2", "Pos cmd high 16b", "pulse", 5, "Int64"),
}

# Subset of commonly used variables for the UI dropdown
COMMON_DRIVE_VARIABLES = [
    (0x0F10, "Speed Feedback (rpm)"),
    (0x0F11, "Speed Command (rpm)"),
    (0x0F13, "Torque Command (%Tn)"),
    (0x0F1E, "Id Reference (0.1%rated)"),
    (0x0F1F, "Id Actual (0.1%rated)"),
    (0x0F20, "Iq Reference (0.1%rated)"),
    (0x0F21, "Iq Actual (0.1%rated)"),
    (0x0F22, "Ud Voltage (V)"),
    (0x0F23, "Uq Voltage (V)"),
    (0x0F2A, "Observer Speed Low (0.1rpm)"),
    (0x0F2C, "Observer Torque (0.1%rated)"),
    (0x0F2D, "Speed Feedforward (rpm)"),
    (0x0F2E, "Torque Feedforward (0.1%rated)"),
    (0x0F2F, "Position Cmd Speed (rpm)"),
    (0x0000, "(Disabled)"),
]

SUPPORTED_DRIVE_TYPES = {
    41: "DX3",
    42: "DX4",
}

# SDO read sentinel and timing
_VR_SENTINEL = -9999.0
_SDO_POLL_MS = 2      # fast poll for bulk reads (ms)
_SDO_TIMEOUT = 2.0    # seconds
_FIFO_TIMEOUT = 15.0  # seconds

_U16 = None  # set lazily after TUA import


def _get_u16():
    global _U16
    if _U16 is None and TUA is not None:
        _U16 = TUA.Co_ObjectType.Unsigned16
    return _U16


def _get_u32():
    if TUA is not None:
        return TUA.Co_ObjectType.Unsigned32
    return None


def _fast_coe_read(connection, axis: int, index: int, subindex: int,
                   obj_type, vr_scratch: int = 901) -> int:
    """Optimised single SDO read with tight polling."""
    connection.SetVrValue(vr_scratch, _VR_SENTINEL)
    connection.Ethercat_CoReadAxis(axis, index, subindex, obj_type, vr_scratch)

    deadline = time.monotonic() + _SDO_TIMEOUT
    poll_s = _SDO_POLL_MS / 1000.0
    while time.monotonic() < deadline:
        val = connection.GetVrValue(vr_scratch)
        if val != _VR_SENTINEL:
            return int(val)
        time.sleep(poll_s)

    raise TimeoutError(
        f"SDO read timed out: axis {axis}, 0x{index:04X} sub {subindex}")


def _fast_coe_write(connection, axis: int, index: int, subindex: int,
                    value: int, obj_type=None):
    """Write one CoE object to the drive."""
    if obj_type is None:
        obj_type = _get_u16()
    connection.Ethercat_CoWriteAxis_Value(axis, index, subindex, obj_type, value)


class DriveScopeEngine:
    """
    Manages the drive-based scope capture lifecycle via CoE SDO.

    Usage:
        engine = DriveScopeEngine(connection, axis=0)
        engine.configure(channels=[0x0F10, 0x0F13], sample_time=8, trigger_mode=0)
        engine.start_capture()
        engine.wait_for_completion()
        data = engine.read_data()
    """

    def __init__(self, connection, axis: int = 0, vr_scratch: int = 901):
        self.connection = connection
        self.axis = axis
        self.vr_scratch = vr_scratch

        # Configuration
        self.channel_addresses: List[int] = [0] * NUM_CHANNELS
        self.active_channels: int = 0  # how many channels are in use
        self.sample_time: int = 1      # in units of 125 μs
        self.trigger_mode: int = 0
        self.trigger_value1: int = 0
        self.trigger_value2: int = 0
        self.ch1_data_type: int = 1    # Int16 by default

        # State
        self.is_configured = False
        self.is_capturing = False
        self.last_start_saw_sampling = False
        self.last_start_status_sequence: List[int] = []

    def _write_u16(self, index: int, subindex: int, value: int) -> None:
        """Write one unsigned 16-bit CoE object, with Execute fallback."""
        value = int(value) & 0xFFFF
        try:
            _fast_coe_write(
                self.connection, self.axis, index, subindex, value, _get_u16()
            )
        except Exception as exc:
            logger.debug(
                "Typed CoE write failed for axis %d 0x%04X:%d=%d; "
                "falling back to Execute: %s",
                self.axis, index, subindex, value, exc,
            )
            if value > 9:
                val_str = f"${value:x}"
            else:
                val_str = str(value)
            cmd = f"co_write_axis({self.axis}, ${index:04x}, {subindex}, 6, -1, {val_str})"
            self.connection.Execute(cmd)

    def _read_u16(self, index: int, subindex: int) -> int:
        """Read one unsigned 16-bit CoE object."""
        return _fast_coe_read(
            self.connection, self.axis, index, subindex, _get_u16(), self.vr_scratch
        ) & 0xFFFF

    @property
    def sample_period_us(self) -> float:
        return self.sample_time * SAMPLE_TIME_UNIT_US

    @property
    def sample_period_sec(self) -> float:
        return self.sample_period_us / 1_000_000.0

    @property
    def capture_duration_sec(self) -> float:
        return SAMPLES_PER_CHANNEL * self.sample_period_sec

    def configure(
        self,
        channels: List[int],
        sample_time: int = 8,
        trigger_mode: int = 0,
        trigger_value1: int = 0,
        trigger_value2: int = 0,
        ch1_data_type: int = 1,
    ) -> Dict[str, Any]:
        """
        Configure drive scope capture parameters.

        Args:
            channels: List of variable addresses (up to 8).
                      e.g. [0x0F10, 0x0F13] for speed feedback + torque cmd.
                      Unused channels are set to 0x0000.
            sample_time: Sample period in units of 125 μs (1 = 125 μs, 8 = 1 ms).
            trigger_mode: 0=free run, 1=rising, 2=falling, 3=greater, 4=less, 5/6=window.
            trigger_value1: First trigger threshold (32-bit, for modes 1-6).
            trigger_value2: Second trigger threshold (32-bit, for window modes 5-6).
            ch1_data_type: Data type code for channel 1 trigger comparison.

        Returns:
            Configuration summary dict.
        """
        if not channels:
            raise ValueError("At least one channel address is required")
        if len(channels) > NUM_CHANNELS:
            raise ValueError(f"Maximum {NUM_CHANNELS} channels supported")

        get_drive_type = getattr(self.connection, "GetAxisParameter_DRIVE_TYPE", None)
        if get_drive_type is not None:
            try:
                drive_type = int(get_drive_type(self.axis))
                if drive_type and drive_type not in SUPPORTED_DRIVE_TYPES:
                    raise RuntimeError(
                        "Drive Scope (SDO) currently supports Trio DX3/DX4 only; "
                        f"axis {self.axis} reports DRIVE_TYPE={drive_type}."
                    )
                if drive_type:
                    logger.info(
                        "Drive scope axis %d drive type: %s (%d)",
                        self.axis, SUPPORTED_DRIVE_TYPES[drive_type], drive_type,
                    )
                else:
                    logger.warning(
                        "Drive scope axis %d reports DRIVE_TYPE=0; "
                        "cannot confirm DX3/DX4 protocol compatibility",
                        self.axis,
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                logger.debug("Could not read DRIVE_TYPE for axis %d: %s", self.axis, exc)

        self.active_channels = len(channels)
        self.channel_addresses = list(channels) + [0] * (NUM_CHANNELS - len(channels))
        self.sample_time = max(1, sample_time)
        self.trigger_mode = trigger_mode
        self.trigger_value1 = trigger_value1
        self.trigger_value2 = trigger_value2
        self.ch1_data_type = ch1_data_type

        # Stop any running capture first (C# does this before configuring)
        self._write_u16(CONTROL_INDEX, 0, 0)
        time.sleep(0.02)

        # Write setup using co_write_axis via Execute — matching C# reference.
        # Syntax: co_write_axis(axis, $368c, sub, 6, -1, value)
        # type 6 = Unsigned16
        # Use ${:x} hex notation for values to match C# reference exactly.
        writes = [
            (1, self.trigger_mode),
            (2, self.trigger_value1 & 0xFFFF),         # Trigger value 1 low
            (3, (self.trigger_value1 >> 16) & 0xFFFF),  # Trigger value 1 high
            (4, self.trigger_value2 & 0xFFFF),         # Trigger value 2 low
            (5, (self.trigger_value2 >> 16) & 0xFFFF),  # Trigger value 2 high
            (6, self.ch1_data_type),
            (7, self.sample_time),
        ]
        # Sub-indices 8–15: channel variable addresses
        for i, addr in enumerate(self.channel_addresses):
            writes.append((8 + i, addr))

        for sub, val in writes:
            self._write_u16(SETUP_INDEX, sub, val)
            time.sleep(0.02)
            logger.debug("Setup 0x368C[%d] = %d", sub, val & 0xFFFF)

        try:
            readback = {
                sub: self._read_u16(SETUP_INDEX, sub)
                for sub in (1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            }
            logger.info(
                "Drive scope setup readback: trigger=%d, dtype=%d, sample=%d, "
                "channels=%s",
                readback[1],
                readback[6],
                readback[7],
                [f"0x{readback[sub]:04X}" for sub in range(8, 16)],
            )
        except Exception as exc:
            logger.warning("Could not read back drive scope setup: %s", exc)

        self.is_configured = True

        config_info = {
            'active_channels': self.active_channels,
            'sample_time_units': self.sample_time,
            'sample_period_us': self.sample_period_us,
            'sample_period_ms': self.sample_period_us / 1000.0,
            'capture_duration_sec': self.capture_duration_sec,
            'samples_per_channel': SAMPLES_PER_CHANNEL,
            'trigger_mode': TRIGGER_MODES.get(self.trigger_mode, f"Unknown ({self.trigger_mode})"),
            'channel_addresses': [f"0x{a:04X}" for a in self.channel_addresses[:self.active_channels]],
        }

        logger.info(
            "Drive scope configured: %d ch, sample_time=%d (%.1f μs), "
            "trigger=%s, duration=%.3f s",
            self.active_channels, self.sample_time, self.sample_period_us,
            config_info['trigger_mode'], self.capture_duration_sec,
        )
        return config_info

    def start_capture(self):
        """Re-arm and start drive scope capture with a fresh 0 -> 1 edge."""
        if not self.is_configured:
            raise RuntimeError("Drive scope not configured — call configure() first")

        self.last_start_saw_sampling = False
        self.last_start_status_sequence = []

        self._write_u16(CONTROL_INDEX, 0, 0)
        time.sleep(0.05)
        try:
            stop_status = self.get_status()
            self.last_start_status_sequence.append(stop_status)
            logger.info("Drive scope status after stop/re-arm: %d", stop_status)
        except Exception as exc:
            logger.warning("Could not verify drive scope stop/re-arm status: %s", exc)

        self._write_u16(CONTROL_INDEX, 0, 1)
        self.is_capturing = True

        verify_deadline = time.monotonic() + min(1.0, max(0.25, self.capture_duration_sec * 0.25))
        while time.monotonic() < verify_deadline:
            status = self.get_status()
            self.last_start_status_sequence.append(status)
            if status == 1:
                self.last_start_saw_sampling = True
                break
            time.sleep(0.02)

        logger.info(
            "Drive scope capture started; start status sequence=%s, saw_sampling=%s",
            self.last_start_status_sequence,
            self.last_start_saw_sampling,
        )

    def stop_capture(self):
        """Stop drive scope capture by writing 0 to 0x368B."""
        try:
            self._write_u16(CONTROL_INDEX, 0, 0)
        except Exception as e:
            logger.warning("Failed to stop drive scope: %s", e)
        self.is_capturing = False
        logger.info("Drive scope capture stopped")

    def get_status(self) -> int:
        """
        Read capture status from 0x3680 bits 14-15.

        Uses co_read_axis via Execute (matching C# reference):
            co_read_axis(axis, $3680, 0, 6, vr)
        type 6 = Unsigned16

        Returns:
            0 = not in sampling status
            1 = sampling in progress
            2 = sampling done
        """
        try:
            raw = self._read_u16(STATUS_INDEX, 0)
            status = (raw >> 14) & 0x3
            logger.debug("Drive scope status raw=0x%04X status=%d", raw & 0xFFFF, status)
            return status
        except Exception as exc:
            logger.debug("Typed status read failed; falling back to Execute: %s", exc)

        vr = self.vr_scratch
        self.connection.SetVrValue(vr, _VR_SENTINEL)
        cmd = f"co_read_axis({self.axis}, $3680, 0, 6, {vr})"
        self.connection.Execute(cmd)
        deadline = time.monotonic() + _SDO_TIMEOUT
        while time.monotonic() < deadline:
            val = self.connection.GetVrValue(vr)
            if val != _VR_SENTINEL:
                raw = int(val)
                status = (raw >> 14) & 0x3
                logger.debug("Drive scope status raw=0x%04X status=%d", raw & 0xFFFF, status)
                return status
            time.sleep(0.01)

        logger.warning("Status read timed out")
        return 0

    def is_capture_complete(self) -> bool:
        """Check if capture is complete (status == 2)."""
        return self.get_status() == 2

    def is_capture_in_progress(self) -> bool:
        """Check if capture is in progress (status == 1)."""
        return self.get_status() == 1

    def wait_for_completion(self, timeout: float = 30.0,
                            progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Poll status until capture is complete or timeout.

        Args:
            timeout: Maximum wait time in seconds.
            progress_callback: Called with estimated progress 0.0–1.0.

        Returns:
            True if capture completed, False if timed out.
        """
        start = time.monotonic()
        capture_duration = self.capture_duration_sec

        while (time.monotonic() - start) < timeout:
            status = self.get_status()
            if status == 2:
                if progress_callback:
                    progress_callback(1.0)
                logger.info("Drive scope capture complete")
                return True

            if progress_callback and capture_duration > 0:
                elapsed = time.monotonic() - start
                progress_callback(min(0.99, elapsed / capture_duration))

            time.sleep(0.05)

        logger.warning("Drive scope capture timed out after %.1f s", timeout)
        return False

    def _candidate_fifo_devices(self) -> List[Tuple[int, str]]:
        """Return plausible EtherCAT device numbers for the FIFO BASIC command.

        co_write_axis/co_read_axis accept Trio axis numbers, but
        ETHERCAT($161, ...) addresses the EtherCAT slave/device.  Different
        controller/API layers expose physical positions and configured station
        addresses differently, so keep all plausible IDs and try them in order.
        """
        candidates: List[Tuple[int, str]] = []

        def add(value: int, source: str) -> None:
            if value <= 0:
                return
            if any(existing == value for existing, _ in candidates):
                return
            candidates.append((value, source))

        fallback = self.axis + 1
        add(fallback, "axis+1 reference")

        try:
            get_slot_number = getattr(self.connection, "GetAxisParameter_SLOT_NUMBER", None)
            if get_slot_number is not None:
                slot_number = int(get_slot_number(self.axis))
                if slot_number > 0:
                    logger.debug(
                        "Drive scope axis %d fallback SLOT_NUMBER/device address %d",
                        self.axis, slot_number,
                    )
                    add(slot_number, "axis SLOT_NUMBER")
        except Exception as exc:
            logger.debug("Could not read SLOT_NUMBER for axis %d: %s", self.axis, exc)

        try:
            check_slaves = getattr(self.connection, "Ethercat_CheckNumberOfSlaves", None)
            get_slave_axis = getattr(self.connection, "Ethercat_GetSlaveAxis", None)
            if check_slaves is not None and get_slave_axis is not None:
                num_slaves = int(check_slaves(0))
                for pos in range(max(0, num_slaves)):
                    if int(get_slave_axis(0, pos)) != self.axis:
                        continue
                    logger.debug(
                        "Drive scope axis %d found at EtherCAT slave position %d; "
                        "adding BASIC device candidate %d",
                        self.axis, pos, pos + 1,
                    )
                    add(pos + 1, f"slave position {pos}")
        except Exception as exc:
            logger.debug("Could not resolve EtherCAT device for axis %d: %s", self.axis, exc)

        logger.debug("Drive scope FIFO device candidates for axis %d: %s", self.axis, candidates)
        return candidates

    def _execute_ethercat_function(self, args: str, timeout: float = 2.0) -> int:
        """Execute a Trio BASIC ETHERCAT(...) function and return its result.

        Execute() alone cannot report the function return value.  Assigning the
        function to a scratch VR gives us a numeric completion/error/progress
        code while still using the same BASIC command surface as the reference
        C# implementation.
        """
        vr = self.vr_scratch
        self.connection.SetVrValue(vr, _VR_SENTINEL)
        self.connection.Execute(f"VR({vr})=ETHERCAT({args})")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            val = self.connection.GetVrValue(vr)
            if val != _VR_SENTINEL:
                return int(val)
            time.sleep(_SDO_POLL_MS / 1000.0)

        raise TimeoutError(f"ETHERCAT({args}) did not return a value")

    def _wait_for_fifo_transfer(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Wait until the controller-side EC_COE_FIFO transfer is complete."""
        deadline = time.monotonic() + _FIFO_TIMEOUT
        saw_progress = False

        while time.monotonic() < deadline:
            try:
                progress = self._execute_ethercat_function("$142", timeout=0.5)
            except Exception as exc:
                logger.debug("Could not read EC_COE_FIFO progress: %s", exc)
                if not saw_progress:
                    time.sleep(2.0)
                    return
                break

            saw_progress = True
            if progress >= 100:
                if progress_callback:
                    progress_callback(0.3, "FIFO transfer complete")
                return

            if progress_callback:
                pct = max(0, min(99, progress))
                progress_callback(0.1 + pct * 0.002, f"FIFO transfer: {pct}%")
            time.sleep(0.1)

        raise TimeoutError("Timed out waiting for EC_COE_FIFO transfer to complete")

    def _start_fifo_transfer(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[int, str]:
        """Start EC_COE_FIFO transfer using the first working device candidate."""
        errors: List[str] = []

        for device, source in self._candidate_fifo_devices():
            self._delete_remote_fifo_file()
            ethercat_args = f"$161, 0, {device}, $3687, 0, {EXPECTED_CAPTURE_BYTES}"
            cmd = f"ethercat({ethercat_args})"
            logger.debug("FIFO transfer candidate from %s: %s", source, cmd)

            try:
                # Match the C# reference: start $161 as a BASIC command.  The
                # command's boolean return is not the transferred data status.
                self.connection.Execute(cmd)
            except Exception as exc:
                errors.append(f"{device} ({source}): {exc}")
                logger.debug("FIFO transfer candidate failed: %s", errors[-1])
                continue

            logger.info(
                "EC_COE_FIFO transfer command issued using device %d (%s)",
                device, source,
            )

            if progress_callback:
                progress_callback(0.1, "Waiting for FIFO transfer...")

            try:
                self._wait_for_fifo_transfer(progress_callback)
            except Exception as exc:
                errors.append(f"{device} ({source}): wait failed: {exc}")
                logger.debug("FIFO transfer candidate failed: %s", errors[-1])
                continue

            fifo_state = self._remote_file_state("EC_COE_FIFO")
            fifo_crc = self._remote_file_crc("EC_COE_FIFO")
            logger.info(
                "Controller EC_COE_FIFO after transfer candidate %d (%s): "
                "FileExists=%s, CRC=%s",
                device, source,
                fifo_state if fifo_state is not None else "n/a",
                f"0x{fifo_crc:04X}" if fifo_crc is not None else "n/a",
            )
            if fifo_state is None or fifo_state != 0:
                return device, source

            errors.append(f"{device} ({source}): EC_COE_FIFO was not created")

        raise RuntimeError(
            "Could not start/verify EC_COE_FIFO transfer. Tried: "
            + "; ".join(errors)
        )

    def _delete_remote_fifo_file(self) -> None:
        """Best-effort cleanup of the controller-side FIFO transfer file."""
        before_state = self._remote_file_state("EC_COE_FIFO")
        if before_state is not None:
            logger.info("Controller EC_COE_FIFO before cleanup: FileExists=%d", before_state)

        delete = getattr(self.connection, "Delete", None)
        if delete is not None:
            try:
                delete("EC_COE_FIFO")
                logger.info("Deleted previous controller EC_COE_FIFO file")
                return
            except Exception as exc:
                logger.debug("Could not delete previous controller EC_COE_FIFO file: %s", exc)

        try:
            self.connection.Execute('FILE "DEL" "EC_COE_FIFO"')
            time.sleep(0.05)
            after_state = self._remote_file_state("EC_COE_FIFO")
            if after_state is not None:
                logger.info("Controller EC_COE_FIFO after cleanup: FileExists=%d", after_state)
            else:
                logger.info('Issued controller cleanup with FILE "DEL" "EC_COE_FIFO"')
        except Exception as exc:
            logger.debug("Could not delete controller EC_COE_FIFO with FILE DEL: %s", exc)

    def _remote_file_state(self, name: str) -> Optional[int]:
        """Return Trio FileExists flag for a controller file, if available."""
        file_exists = getattr(self.connection, "FileExists", None)
        if file_exists is None:
            return None
        try:
            return int(file_exists(name))
        except Exception as exc:
            logger.debug("Could not check controller file %s: %s", name, exc)
            return None

    def _remote_file_crc(self, name: str) -> Optional[int]:
        """Return controller file CRC, if available."""
        get_crc = getattr(self.connection, "GetRemoteFileCRC", None)
        if get_crc is None:
            return None
        try:
            return int(get_crc(name))
        except Exception as exc:
            logger.debug("Could not read controller file CRC for %s: %s", name, exc)
            return None

    def _select_capture_bytes(self, raw_bytes: bytes) -> bytes:
        """Return the most recent 16000-byte capture payload from a FIFO file.

        The Trio controller stores EC_COE_FIFO downloads as 0x8100-byte chunks:
        a small container prefix followed by the object payload/padding.  Compose
        drive_scope.bin from the newest chunk's payload area, not from byte zero
        of the controller container.
        """
        n_bytes = len(raw_bytes)
        if n_bytes <= EXPECTED_CAPTURE_BYTES:
            return raw_bytes

        chunk_start = 0
        chunk = raw_bytes
        if n_bytes >= FIFO_CHUNK_BYTES:
            chunk_count = n_bytes // FIFO_CHUNK_BYTES
            chunk_start = (chunk_count - 1) * FIFO_CHUNK_BYTES
            chunk = raw_bytes[chunk_start:chunk_start + FIFO_CHUNK_BYTES]
            logger.info(
                "FIFO file contains %d chunk(s), using newest chunk at byte %d",
                chunk_count, chunk_start,
            )

        payload_offset = 0
        if len(chunk) >= FIFO_CONTAINER_PREFIX_BYTES + EXPECTED_CAPTURE_BYTES:
            payload_offset = FIFO_CONTAINER_PREFIX_BYTES

        payload = chunk[payload_offset:payload_offset + EXPECTED_CAPTURE_BYTES]
        if len(payload) < EXPECTED_CAPTURE_BYTES:
            logger.warning(
                "FIFO payload has %d bytes; expected %d, padding composed file",
                len(payload), EXPECTED_CAPTURE_BYTES,
            )
            payload = payload + bytes(EXPECTED_CAPTURE_BYTES - len(payload))

        logger.info(
            "Selected FIFO payload window: raw byte %d, chunk offset %d, length %d",
            chunk_start + payload_offset, payload_offset, len(payload),
        )
        return payload

    def _nonzero_byte_ranges(self, data: bytes, merge_gap: int = 16) -> List[Tuple[int, int]]:
        """Return merged nonzero byte ranges as [start, end) pairs."""
        ranges: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for idx, byte in enumerate(data):
            if byte and start is None:
                start = idx
            elif not byte and start is not None:
                ranges.append((start, idx))
                start = None
        if start is not None:
            ranges.append((start, len(data)))

        merged: List[Tuple[int, int]] = []
        for start, end in ranges:
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def read_data(
        self,
        table_start: int = 0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        local_filename: str = "drive_scope.bin",
    ) -> Dict[str, Any]:
        """
        Read captured data from drive data buffer (0x3687) using EC_COE_FIFO
        file transfer — matching the C# reference implementation.

        Steps:
          1. ethercat($161, 0, slave, $3687, 0, 16000) — initiate FIFO transfer
          2. DownloadFile("drive_scope.bin", "EC_COE_FIFO") — download to PC
          3. Parse the binary file (16-bit interleaved words)

        Args:
            table_start: (unused, kept for API compat)
            progress_callback: Called with (progress_0_to_1, status_message).
            local_filename: Local path for the downloaded binary file.

        Returns:
            Dict with 'time', 'sample_period', 'num_samples', 'params'.
        """
        if progress_callback:
            progress_callback(0.0, "Initiating FIFO transfer from drive...")

        logger.info("Reading drive scope data via EC_COE_FIFO transfer...")
        read_start = time.monotonic()

        # Step 1: Initiate CoE FIFO file transfer on the controller
        # $161 = EC_COE_FIFO transfer function
        # 16000 bytes = 8000 words × 2 bytes/word
        self._start_fifo_transfer(progress_callback)

        if progress_callback:
            progress_callback(0.3, "Downloading file from controller...")

        fifo_state = self._remote_file_state("EC_COE_FIFO")
        fifo_crc = self._remote_file_crc("EC_COE_FIFO")
        if fifo_state is not None or fifo_crc is not None:
            logger.info(
                "Controller EC_COE_FIFO before download: FileExists=%s, CRC=%s",
                fifo_state if fifo_state is not None else "n/a",
                f"0x{fifo_crc:04X}" if fifo_crc is not None else "n/a",
            )

        # Step 2: Download the controller-side FIFO to a raw diagnostic file,
        # then compose drive_scope.bin as exactly the 0x3687 payload bytes.
        file_path = pathlib.Path(local_filename)
        raw_file_path = file_path.with_name(f"{file_path.stem}_fifo_raw{file_path.suffix}")
        try:
            # Remove only the raw diagnostic file before transfer.  Keep the
            # previous composed capture until a new FIFO download succeeds.
            if raw_file_path.exists():
                raw_file_path.unlink()
        except OSError as e:
            raise RuntimeError(
                f"Failed to replace previous drive scope files for {local_filename}: {e}"
            ) from e

        # Python API requires a progress callback: (ProgressInfo) -> None
        def _download_progress(info):
            logger.debug("DownloadFile progress: pos=%s", info.current_pos)

        try:
            self.connection.DownloadFile(str(raw_file_path), "EC_COE_FIFO", _download_progress)
        except Exception as e:
            logger.error("DownloadFile failed: %s", e)
            raise RuntimeError(f"Failed to download drive scope data: {e}") from e

        if progress_callback:
            progress_callback(0.8, "Composing binary data...")

        # Step 3: Compose a clean local binary payload.
        if not raw_file_path.exists():
            raise FileNotFoundError(f"Downloaded FIFO file not found: {raw_file_path}")

        raw_bytes = raw_file_path.read_bytes()
        capture_bytes = self._select_capture_bytes(raw_bytes)
        file_path.write_bytes(capture_bytes)
        elapsed = time.monotonic() - read_start

        raw_ranges = self._nonzero_byte_ranges(raw_bytes)
        capture_ranges = self._nonzero_byte_ranges(capture_bytes)
        logger.info(
            "FIFO raw download complete: %d bytes in %.1f s, nonzero ranges=%s",
            len(raw_bytes), elapsed, raw_ranges[:8],
        )
        logger.info(
            "Composed %s: %d bytes, nonzero ranges=%s",
            file_path.name, len(capture_bytes), capture_ranges[:8],
        )

        if progress_callback:
            progress_callback(1.0, "Data download complete")

        return self._parse_raw_bytes(capture_bytes)

    def _parse_raw_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Parse binary data downloaded via EC_COE_FIFO.

        The data layout is interleaved across all 8 channels (the drive's built-in
        scope always captures and outputs 8 channels). Each sample row contains
        eight 16-bit words.
        """
        n_bytes = len(raw_bytes)
        n_words = n_bytes // 2
        n_ch = self.active_channels

        logger.info(
            "Parsing %d bytes (%d words), %d active channels, "
            "stride=%d words/sample",
            n_bytes, n_words, n_ch, NUM_CHANNELS,
        )

        # Convert bytes to uint16 array (little-endian)
        raw_words = np.frombuffer(raw_bytes[:n_words * 2], dtype=np.dtype('<u2'))

        # Expected useful data: NUM_CHANNELS * SAMPLES_PER_CHANNEL (8000 words)
        expected_words = TOTAL_WORDS
        if len(raw_words) < expected_words:
            logger.warning(
                "Got %d words, expected %d (%d ch × %d samples) — padding",
                len(raw_words), expected_words, NUM_CHANNELS, SAMPLES_PER_CHANNEL,
            )
            padded = np.zeros(expected_words, dtype=np.uint16)
            padded[:len(raw_words)] = raw_words
            raw_words = padded
        else:
            # Take only the first 8000 words (the actual captured domain data)
            if len(raw_words) > expected_words:
                extra_words = raw_words[expected_words:]
                if np.any(extra_words):
                    logger.warning(
                        "Got %d words from drive scope FIFO; using first %d words "
                        "and ignoring %d nonzero trailing words",
                        len(raw_words), expected_words, int(np.count_nonzero(extra_words)),
                    )
                else:
                    logger.debug(
                        "Got %d words from drive scope FIFO; trailing data is zero padding",
                        len(raw_words),
                    )
            raw_words = raw_words[:expected_words]

        # Reshape to (1000, 8) — each row is one sample across 8 channels
        data_2d = raw_words.reshape(SAMPLES_PER_CHANNEL, NUM_CHANNELS)

        # Build time array
        time_array = np.arange(SAMPLES_PER_CHANNEL) * self.sample_period_sec

        result = {
            'time': time_array,
            'sample_period': self.sample_period_sec,
            'num_samples': SAMPLES_PER_CHANNEL,
            'params': {},
            'raw_words': raw_words,
        }

        # Extract each active channel with signed interpretation
        for ch_idx in range(n_ch):
            addr = self.channel_addresses[ch_idx]
            if addr == 0:
                continue

            raw_ch = data_2d[:, ch_idx].copy()

            # Determine display name and data type
            if addr in DRIVE_VARIABLES:
                name, desc, unit, dtype_code, dtype_str = DRIVE_VARIABLES[addr]
                display_name = f"{name} (0x{addr:04X})"
            else:
                display_name = f"Ch{ch_idx+1} (0x{addr:04X})"
                dtype_str = "Int16"

            # Convert to signed if needed (C# does: (short)(hi<<8 | lo))
            if dtype_str in ("Int16", "Int32", "Int64"):
                values = raw_ch.astype(np.int16).astype(np.float64)
            else:
                values = raw_ch.astype(np.float64)

            result['params'][display_name] = values
            logger.debug(
                "Ch%d %s: min=%.1f max=%.1f mean=%.1f",
                ch_idx, display_name,
                values.min(), values.max(), values.mean(),
            )

        return result
