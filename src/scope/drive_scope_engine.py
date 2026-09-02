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
    0x3687  Data:     16000-byte domain (8000 words, up to 8 ch × 1000 samples)

Data layout is interleaved across the ACTIVE channels only.  The COMBO
document shows the full 8-channel case, but the drive packs just the
channels whose address is nonzero, each with 1000 samples (verified
against real DX4 captures and the C# reference, which parses a 6-channel
setup with a 6-word stride).  With N active channels:
    Word 0..N-1:    Ch1[0], Ch2[0], ..., ChN[0]
    Word N..2N-1:   Ch1[1], Ch2[1], ..., ChN[1]
    ...
Only the first N × 1000 words of the 16000-byte upload are capture data;
the remainder is stale drive memory and must not be parsed.
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
    (0x0F32, "Position Error Low (select 0x0F32-0x0F35)"),
    (0x0F33, "Position Error Mid-Low"),
    (0x0F34, "Position Error Mid-High"),
    (0x0F35, "Position Error High"),
    (0x0000, "(Disabled)"),
]

SUPPORTED_DRIVE_TYPES = {
    41: "DX3",
    42: "DX4",
    43: "DX1",
    45: "DX5",
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
        self.drive_model = "DX3"  # "DX3", "DX5", or "DX1"
        self.channel_addresses: List[int] = [0] * NUM_CHANNELS
        self.active_channels: int = 0  # how many channels are in use
        self.sample_time: int = 1      # in units of 125 μs
        self.trigger_mode: int = 0
        self.trigger_value1: int = 0
        self.trigger_value2: int = 0
        self.ch1_data_type: int = 1    # Int16 by default
        self.display_names: List[str] = []

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
        if self.drive_model == "DX1":
            return self.sample_time * 62.5
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
        display_names: Optional[List[str]] = None,
        drive_model: Optional[str] = None,
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
            display_names: Custom display names corresponding to channels.
            drive_model: Optional manual override string ("DX3", "DX5", "DX1")

        Returns:
            Configuration summary dict.
        """
        if not channels:
            raise ValueError("At least one channel address is required")
        if len(channels) > NUM_CHANNELS:
            raise ValueError(f"Maximum {NUM_CHANNELS} channels supported")

        get_drive_type = getattr(self.connection, "GetAxisParameter_DRIVE_TYPE", None)
        detected_model = None
        if get_drive_type is not None:
            try:
                drive_type = int(get_drive_type(self.axis))
                if drive_type and drive_type not in SUPPORTED_DRIVE_TYPES:
                    raise RuntimeError(
                        "Drive Scope (SDO) currently supports Trio DX3/DX4/DX5/DX1 only; "
                        f"axis {self.axis} reports DRIVE_TYPE={drive_type}."
                    )
                if drive_type:
                    detected_model = SUPPORTED_DRIVE_TYPES[drive_type]
                    logger.info(
                        "Drive scope axis %d drive type: %s (%d)",
                        self.axis, detected_model, drive_type,
                    )
                else:
                    logger.warning(
                        "Drive scope axis %d reports DRIVE_TYPE=0; "
                        "cannot confirm drive protocol compatibility",
                        self.axis,
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                logger.debug("Could not read DRIVE_TYPE for axis %d: %s", self.axis, exc)

        if drive_model is not None:
            self.drive_model = drive_model
        elif detected_model is not None:
            self.drive_model = detected_model
        else:
            self.drive_model = "DX3"

        self.active_channels = len(channels)
        self.channel_addresses = list(channels) + [0] * (NUM_CHANNELS - len(channels))
        self.display_names = display_names if display_names else []
        self.sample_time = max(1, sample_time)
        self.trigger_mode = trigger_mode
        self.trigger_value1 = trigger_value1
        self.trigger_value2 = trigger_value2
        self.ch1_data_type = ch1_data_type

        sleep_time = 0.05

        if self.drive_model in ("DX5", "DX1"):
            # Disable Scope: Z_AScopeEnable = 0 (Index 0x2065, subindex 0, length 4 bytes)
            cmd = f"co_write_axis({self.axis}, $2065, 0, 4, -1, 0)"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

            # Set Length: Z_AScopeSamples = 8000 (Index 0x2068, subindex 0, length 4 bytes)
            cmd = f"co_write_axis({self.axis}, $2068, 0, 4, -1, 8000)"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

            # Set Steps: Z_AScopeSteps = sample_time (Index 0x2069, subindex 0, length 4 bytes)
            cmd = f"co_write_axis({self.axis}, $2069, 0, 4, -1, {self.sample_time})"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

            # Set Trigger Channel: Z_AScopeTrigChan (Index 0x206A, subindex 0, length 4)
            trig_chan = 0 if self.trigger_mode == 0 else 1
            cmd = f"co_write_axis({self.axis}, $206A, 0, 4, -1, {trig_chan})"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

            # Set Trigger Value: Z_AScopeTrigVal (Index 0x206B, subindex 0, length 8)
            cmd = f"co_write_axis({self.axis}, $206B, 0, 8, -1, {self.trigger_value1})"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

            # Set Trigger Up/Down: Z_AScopeTrigUpDown (Index 0x206C, subindex 0, length 4)
            trig_updown = 0 if self.trigger_mode in (1, 3, 5) else 1
            cmd = f"co_write_axis({self.axis}, $206C, 0, 4, -1, {trig_updown})"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

            # Set Pretrigger: Z_AScopeTrigPre (Index 0x206D, subindex 0, length 4)
            pretrigger = 100
            cmd = f"co_write_axis({self.axis}, $206D, 0, 4, -1, {pretrigger})"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

            # Calculate channel offset
            chOffset = 0
            if self.drive_model == "DX5":
                if self.axis % 2 != 0:
                    chOffset = 0x800

            # Write Setup channels Z_AScopeCh1 to Ch8 (Index 0x206E to 0x2075)
            objAdd_speed = 0x36df + chOffset
            objAdd_iqr = 0x36eb + chOffset
            objAdd_pos1 = 14064 + chOffset
            objAdd_pos2 = 14050 + chOffset
            objAdd_alarm = 14392

            channels_to_write = [
                (0x206E, objAdd_speed),
                (0x206F, objAdd_iqr),
                (0x2070, objAdd_pos1),
                (0x2071, objAdd_pos2),
                (0x2072, objAdd_alarm),
                (0x2073, 0),
                (0x2074, 0),
                (0x2075, 0),
            ]

            for index, objAdd in channels_to_write:
                cmd = f"co_write_axis({self.axis}, ${index:X}, 0, 4, -1, {objAdd})"
                self.connection.Execute(cmd)
                time.sleep(sleep_time)

            # Enable Scope: Z_AScopeEnable = 1
            cmd = f"co_write_axis({self.axis}, $2065, 0, 4, -1, 1)"
            self.connection.Execute(cmd)
            time.sleep(sleep_time)

        else:
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

        if self.drive_model in ("DX3", "DX4"):
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
        else:
            # DX5 / DX1
            # Re-arm by writing 0 and then 1 to $2065 Z_AScopeEnable
            cmd = f"co_write_axis({self.axis}, $2065, 0, 4, -1, 0)"
            self.connection.Execute(cmd)
            time.sleep(0.05)
            cmd = f"co_write_axis({self.axis}, $2065, 0, 4, -1, 1)"
            self.connection.Execute(cmd)
            self.is_capturing = True
            self.last_start_saw_sampling = True  # assume started

        logger.info(
            "Drive scope capture started; start status sequence=%s, saw_sampling=%s",
            self.last_start_status_sequence,
            self.last_start_saw_sampling,
        )

    def stop_capture(self):
        """Stop drive scope capture."""
        try:
            if self.drive_model in ("DX3", "DX4"):
                self._write_u16(CONTROL_INDEX, 0, 0)
            else:
                cmd = f"co_write_axis({self.axis}, $2065, 0, 4, -1, 0)"
                self.connection.Execute(cmd)
        except Exception as e:
            logger.warning("Failed to stop drive scope: %s", e)
        self.is_capturing = False
        logger.info("Drive scope capture stopped")

    def get_status(self) -> int:
        """
        Read capture status from 0x3680 bits 14-15 (DX3/DX4) or 0x2066 (DX5/DX1).
        """
        if self.drive_model in ("DX5", "DX1"):
            try:
                vr = self.vr_scratch
                self.connection.SetVrValue(vr, _VR_SENTINEL)
                cmd = f"co_read_axis({self.axis}, $2066, 0, 4, {vr})"
                self.connection.Execute(cmd)
                deadline = time.monotonic() + _SDO_TIMEOUT
                while time.monotonic() < deadline:
                    val = self.connection.GetVrValue(vr)
                    if val != _VR_SENTINEL:
                        val_int = int(val)
                        if val_int == 3:
                            return 2  # done
                        elif val_int in (1, 2):
                            return 1  # sampling
                        else:
                            return 0  # idle
                    time.sleep(0.01)
            except Exception as exc:
                logger.warning("Failed to read DX5/DX1 status: %s", exc)
            return 0

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
            logger.info("Adding device candidate %d (%s) for axis %d", value, source, self.axis)
            candidates.append((value, source))

        # 1. API physical position and slave address (highly specific)
        try:
            check_slaves = getattr(self.connection, "Ethercat_CheckNumberOfSlaves", None)
            get_slave_axis = getattr(self.connection, "Ethercat_GetSlaveAxis", None)
            get_slave_addr = getattr(self.connection, "Ethercat_GetSlaveAddress", None)
            if check_slaves is not None and get_slave_axis is not None:
                num_slaves = int(check_slaves(0))
                logger.debug("API: checking %d slaves on slot 0", num_slaves)
                for pos in range(max(0, num_slaves)):
                    mapped_axis = int(get_slave_axis(0, pos))
                    logger.debug("API: slave at position %d has axis %d (expected %d)", pos, mapped_axis, self.axis)
                    if mapped_axis == self.axis:
                        logger.info("API matched axis %d to slave position %d", self.axis, pos)
                        add(pos + 1, f"slave position {pos}")
                        if get_slave_addr is not None:
                            addr = int(get_slave_addr(0, pos))
                            if addr > 0:
                                add(addr, f"slave address {addr}")
        except Exception as exc:
            logger.warning("Could not resolve EtherCAT device via API for axis %d: %s", self.axis, exc)

        # 2. BASIC ETHERCAT functions physical position and address fallback (extremely robust)
        try:
            num_slaves = self._execute_ethercat_vr_function(3, "0", timeout=1.0)
            logger.debug("BASIC ETHERCAT: checking %d slaves on slot 0", num_slaves)
            for pos in range(max(0, num_slaves)):
                slave_axis = self._execute_ethercat_vr_function(5, f"0, {pos}", timeout=1.0)
                logger.debug("BASIC ETHERCAT: slave at position %d has axis %d (expected %d)", pos, slave_axis, self.axis)
                if slave_axis == self.axis:
                    logger.info("BASIC ETHERCAT matched axis %d to slave position %d", self.axis, pos)
                    add(pos + 1, f"slave position {pos} (via BASIC ETHERCAT)")
                    slave_address = self._execute_ethercat_vr_function(4, f"0, {pos}", timeout=1.0)
                    if slave_address > 0:
                        add(slave_address, f"slave address {slave_address} (via BASIC ETHERCAT)")
        except Exception as exc:
            logger.debug("Could not resolve EtherCAT device via ETHERCAT BASIC fallback for axis %d: %s", self.axis, exc)

        # 3. Axis SLOT_NUMBER parameter (configured station address)
        try:
            get_slot_number = getattr(self.connection, "GetAxisParameter_SLOT_NUMBER", None)
            if get_slot_number is not None:
                slot_number = int(get_slot_number(self.axis))
                if slot_number > 0:
                    logger.info("Axis SLOT_NUMBER parameter read: %d", slot_number)
                    add(slot_number, "axis SLOT_NUMBER")
        except Exception as exc:
            logger.warning("Could not read SLOT_NUMBER for axis %d: %s", self.axis, exc)

        # 4. Standard fallback: axis + 1
        fallback = self.axis + 1
        logger.info("Adding standard fallback candidate: %d", fallback)
        add(fallback, "axis+1 reference")

        logger.info("Drive scope FIFO device candidates for axis %d: %s", self.axis, candidates)
        return candidates

    def _execute_ethercat_vr_function(self, func_num: int, extra_args: str = "", timeout: float = 1.0) -> int:
        """Execute an ETHERCAT function that writes its output to a VR parameter.

        Assigning/writing the function value to a scratch VR gives us a numeric
        completion/error/progress code while still using the same BASIC command
        surface as the reference C# implementation.
        """
        vr = self.vr_scratch
        self.connection.SetVrValue(vr, _VR_SENTINEL)
        if extra_args:
            cmd = f"ETHERCAT({func_num}, {extra_args}, {vr})"
        else:
            cmd = f"ETHERCAT({func_num}, {vr})"
        logger.debug("Executing BASIC command: %s", cmd)
        self.connection.Execute(cmd)

        deadline = time.monotonic() + timeout
        poll_s = _SDO_POLL_MS / 1000.0
        while time.monotonic() < deadline:
            val = self.connection.GetVrValue(vr)
            if val != _VR_SENTINEL:
                logger.debug("BASIC command %s returned VR(%d)=%d", cmd, vr, int(val))
                return int(val)
            time.sleep(poll_s)

        raise TimeoutError(f"{cmd} did not return a value")

    def _wait_for_fifo_transfer(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Wait until the controller-side EC_COE_FIFO transfer is complete."""
        # ETHERCAT($161, ...) SDO transfers do not report progress via $142.
        # We sleep for a fixed duration of 2.0 seconds to allow the controller to
        # copy the SDO FIFO to the local file system (matching reference implementation).
        logger.info("Waiting 2.0 seconds for SDO FIFO transfer to complete on the controller...")
        if progress_callback:
            progress_callback(0.15, "FIFO transfer in progress...")
        time.sleep(2.0)
        if progress_callback:
            progress_callback(0.3, "FIFO transfer complete")

    def _start_fifo_transfer(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[int, str]:
        """Start EC_COE_FIFO transfer using the first working device candidate."""
        errors: List[str] = []

        for device, source in self._candidate_fifo_devices():
            self._delete_remote_fifo_file()
            if self.drive_model in ("DX5", "DX1"):
                cmd = f'ETHERCAT($141, 0, {device}, "C", "EC_COE_FIFO", "ASCOPE_data0", -1)'
            else:
                ethercat_args = f"$161, 0, {device}, $3687, 0, {EXPECTED_CAPTURE_BYTES}"
                cmd = f"ethercat({ethercat_args})"
            logger.debug("FIFO transfer candidate from %s: %s", source, cmd)

            try:
                # Match the C# reference: start as a BASIC command.
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
                if self.drive_model in ("DX5", "DX1"):
                    logger.info("Waiting 5.0 seconds for SDO FIFO file transfer on controller...")
                    time.sleep(5.0)
                else:
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
        """Return the 16000-byte capture payload from a downloaded FIFO file.

        The 0x3687 object payload starts at byte zero of the controller's
        EC_COE_FIFO file; the controller merely rounds the file up (typically
        to 0x8100 bytes) with padding after the payload.  This matches the
        working C# reference, which parses the downloaded file from byte 0,
        and was verified against real DX4 captures: with N active channels
        the stale-memory junk past the capture begins exactly at byte
        N × 1000 × 2 of the file, which is only consistent with the payload
        starting at offset 0.  (The remote file is deleted before every
        transfer, so the download never accumulates older captures.)
        """
        n_bytes = len(raw_bytes)
        if n_bytes < EXPECTED_CAPTURE_BYTES:
            logger.warning(
                "FIFO file has %d bytes; expected at least %d, padding capture",
                n_bytes, EXPECTED_CAPTURE_BYTES,
            )
            return raw_bytes + bytes(EXPECTED_CAPTURE_BYTES - n_bytes)

        if n_bytes > EXPECTED_CAPTURE_BYTES:
            logger.info(
                "FIFO file has %d bytes; using first %d (rest is container padding)",
                n_bytes, EXPECTED_CAPTURE_BYTES,
            )
        return raw_bytes[:EXPECTED_CAPTURE_BYTES]

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
        if self.drive_model in ("DX5", "DX1"):
            # No select/strip for DX5/DX1 binary file, run the converter tool on raw bytes
            file_path.write_bytes(raw_bytes)
            if progress_callback:
                progress_callback(0.9, "Running CSV converter...")
            result = self._convert_and_parse_dx5_data(str(file_path))
            if progress_callback:
                progress_callback(1.0, "Data download and parsing complete")
            return result
        else:
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

    def _convert_and_parse_dx5_data(self, local_bin_path: str) -> Dict[str, Any]:
        """Convert DX5/DX1 binary scope data to CSV using AScope2DataDx5.exe, and parse it."""
        import subprocess
        import sys

        bin_dir = pathlib.Path(local_bin_path).parent.resolve()
        csv_path = bin_dir / "data.csv"

        if csv_path.exists():
            try:
                csv_path.unlink()
            except OSError:
                pass

        exe_name = "AScope2DataDx5.exe"
        exe_candidates = [
            pathlib.Path.cwd() / exe_name,
            pathlib.Path(__file__).parent / exe_name,
            pathlib.Path(__file__).parent.parent / exe_name,
            pathlib.Path(sys.argv[0]).parent / exe_name,
        ]

        exe_path = None
        for cand in exe_candidates:
            if cand.exists():
                exe_path = cand
                break

        if exe_path is None:
            exe_path = pathlib.Path(exe_name)

        logger.info("Running converter: %s with args: %s data.csv", exe_path, local_bin_path)

        try:
            res = subprocess.run(
                [str(exe_path), str(pathlib.Path(local_bin_path).resolve()), "data.csv"],
                cwd=str(bin_dir),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Converter stdout: %s", res.stdout)
        except Exception as e:
            logger.error("Failed to run AScope2DataDx5.exe converter: %s", e)
            raise FileNotFoundError(
                f"Could not convert binary data to CSV. Please ensure {exe_name} is in "
                f"the application directory. Error: {e}"
            ) from e

        if not csv_path.exists():
            raise FileNotFoundError(f"Converter failed to output data.csv at {csv_path}")

        return self._parse_csv_file(str(csv_path))

    def _parse_csv_file(self, csv_path: str) -> Dict[str, Any]:
        """Parse the CSV file outputted by the converter tool."""
        logger.info("Parsing CSV file: %s", csv_path)

        times = []
        col1 = []
        col2 = []
        col3 = []
        col4 = []

        factor = 125e-6 if self.drive_model == "DX5" else 62.5e-6
        sample_period = self.sample_time * factor

        with open(csv_path, "r", encoding="utf-8") as f:
            line_num = 0
            for line in f:
                line_num += 1
                if line_num <= 8:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                fields = stripped.split("\t")
                if len(fields) < 5:
                    continue
                try:
                    v1 = float(fields[1])
                    v2 = float(fields[2])
                    v3 = float(fields[3])
                    v4 = float(fields[4])

                    j = len(times)
                    times.append(j * sample_period)
                    col1.append(v1)
                    col2.append(v2)
                    col3.append(v3)
                    col4.append(v4)
                except ValueError:
                    continue

        num_samples = len(times)
        logger.info("Parsed %d samples from CSV", num_samples)

        result = {
            'time': np.array(times, dtype=np.float64),
            'sample_period': sample_period,
            'num_samples': num_samples,
            'params': {},
        }

        temp_cols = [col1, col2, col3, col4]
        for idx in range(min(len(temp_cols), self.active_channels)):
            addr = self.channel_addresses[idx]
            if addr == 0:
                continue

            if addr in DRIVE_VARIABLES:
                name, desc, unit, dtype_code, dtype_str = DRIVE_VARIABLES[addr]
                display_name = f"{name} (0x{addr:04X})"
            else:
                display_name = f"Ch{idx+1} (0x{addr:04X})"

            result['params'][display_name] = np.array(temp_cols[idx], dtype=np.float64)

        return result

    def _parse_raw_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Parse binary data downloaded via EC_COE_FIFO.

        The data layout is interleaved across the ACTIVE channels only: the
        drive packs the channels with a nonzero address contiguously, 1000
        samples each, so the word stride per sample equals the number of
        active channels (matching the C# reference, which parses a 6-channel
        capture with a 6-word stride).  Everything past the first
        stride × 1000 words of the upload is stale drive memory, not samples,
        and is discarded.
        """
        n_bytes = len(raw_bytes)
        n_words = n_bytes // 2

        # Active channels: configure() packs them at the front of
        # channel_addresses; skip zeros defensively, keeping the original
        # index so display_names stays aligned.
        active = [
            (idx, addr)
            for idx, addr in enumerate(self.channel_addresses[:self.active_channels])
            if addr
        ]
        stride = len(active)

        logger.info(
            "Parsing %d bytes (%d words), %d active channels, "
            "stride=%d words/sample",
            n_bytes, n_words, stride, stride,
        )

        # Build time array
        time_array = np.arange(SAMPLES_PER_CHANNEL) * self.sample_period_sec

        result = {
            'time': time_array,
            'sample_period': self.sample_period_sec,
            'num_samples': SAMPLES_PER_CHANNEL,
            'params': {},
        }

        if stride == 0:
            logger.warning("No active drive scope channels; nothing to parse")
            result['num_samples'] = 0
            result['raw_words'] = np.zeros(0, dtype=np.uint16)
            return result

        # Convert bytes to uint16 array (little-endian)
        raw_words = np.frombuffer(raw_bytes[:n_words * 2], dtype=np.dtype('<u2'))

        # Useful capture data: stride × 1000 words; the rest of the
        # 16000-byte upload is stale drive memory and must not be parsed.
        expected_words = stride * SAMPLES_PER_CHANNEL
        if len(raw_words) < expected_words:
            logger.warning(
                "Got %d words, expected %d (%d ch × %d samples) — padding",
                len(raw_words), expected_words, stride, SAMPLES_PER_CHANNEL,
            )
            padded = np.zeros(expected_words, dtype=np.uint16)
            padded[:len(raw_words)] = raw_words
            raw_words = padded
        elif len(raw_words) > expected_words:
            logger.debug(
                "Got %d words from drive scope FIFO; using first %d, "
                "discarding %d trailing words of stale buffer memory",
                len(raw_words), expected_words, len(raw_words) - expected_words,
            )
            raw_words = raw_words[:expected_words]

        result['raw_words'] = raw_words

        # Reshape to (1000, stride) — each row is one sample across the
        # active channels
        data_2d = raw_words.reshape(SAMPLES_PER_CHANNEL, stride)

        # Extract each active channel with signed interpretation
        skip_channels = 0
        for col, (ch_idx, addr) in enumerate(active):
            if skip_channels > 0:
                skip_channels -= 1
                continue

            # No copy needed — the astype() calls below always allocate new
            # arrays, and data_2d is never mutated.
            raw_ch = data_2d[:, col]

            # Determine display name and data type
            if addr in DRIVE_VARIABLES:
                name, desc, unit, dtype_code, dtype_str = DRIVE_VARIABLES[addr]
                if self.display_names and ch_idx < len(self.display_names):
                    display_name = self.display_names[ch_idx]
                else:
                    display_name = f"{name} (0x{addr:04X})"
            else:
                if self.display_names and ch_idx < len(self.display_names):
                    display_name = self.display_names[ch_idx]
                else:
                    display_name = f"Ch{ch_idx+1} (0x{addr:04X})"
                dtype_str = "Int16"

            # Reconstruction Logic
            if dtype_str == "Int32" and col + 1 < stride:
                raw_high = data_2d[:, col+1]
                combined = (raw_high.astype(np.uint32) << 16) | raw_ch.astype(np.uint32)
                values = combined.astype(np.int32).astype(np.float64)
                skip_channels = 1
                display_name = display_name.replace("_L", "").replace("_L1", "")
            elif dtype_str == "Int64" and col + 3 < stride:
                raw_h1 = data_2d[:, col+1]
                raw_l2 = data_2d[:, col+2]
                raw_h2 = data_2d[:, col+3]
                combined = (
                    (raw_h2.astype(np.uint64) << 48) |
                    (raw_l2.astype(np.uint64) << 32) |
                    (raw_h1.astype(np.uint64) << 16) |
                    raw_ch.astype(np.uint64)
                )
                values = combined.astype(np.int64).astype(np.float64)
                skip_channels = 3
                display_name = display_name.replace("_L1", "")
            else:
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
