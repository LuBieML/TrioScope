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
import struct
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

# SDO read sentinel and timing
_VR_SENTINEL = -9999.0
_SDO_POLL_MS = 2      # fast poll for bulk reads (ms)
_SDO_TIMEOUT = 2.0    # seconds

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

        self.active_channels = len(channels)
        self.channel_addresses = list(channels) + [0] * (NUM_CHANNELS - len(channels))
        self.sample_time = max(1, sample_time)
        self.trigger_mode = trigger_mode
        self.trigger_value1 = trigger_value1
        self.trigger_value2 = trigger_value2
        self.ch1_data_type = ch1_data_type

        # Stop any running capture first (C# does this before configuring)
        stop_cmd = f"co_write_axis({self.axis}, $368b, 0, 6, -1, 0)"
        self.connection.Execute(stop_cmd)
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
            # Use $hex for channel addresses (sub 8-15), decimal for others
            if sub >= 8 and val != 0:
                val_str = f"${val:x}"
            else:
                val_str = str(val)
            cmd = f"co_write_axis({self.axis}, $368c, {sub}, 6, -1, {val_str})"
            self.connection.Execute(cmd)
            time.sleep(0.02)
            logger.debug("Setup 0x368C[%d] = %s via Execute", sub, val_str)

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
        """Start drive scope capture by writing 1 to 0x368B."""
        if not self.is_configured:
            raise RuntimeError("Drive scope not configured — call configure() first")
        cmd = f"co_write_axis({self.axis}, $368b, 0, 6, -1, 1)"
        self.connection.Execute(cmd)
        self.is_capturing = True
        logger.info("Drive scope capture started")

    def stop_capture(self):
        """Stop drive scope capture by writing 0 to 0x368B."""
        try:
            cmd = f"co_write_axis({self.axis}, $368b, 0, 6, -1, 0)"
            self.connection.Execute(cmd)
        except Exception as e:
            logger.warning("Failed to stop drive scope: %s", e)
        self.is_capturing = False
        logger.info("Drive scope capture stopped")

    def get_status(self) -> int:
        """
        Read capture status from 0x3680 bits 14-15.

        Uses co_read_axis via Execute (matching C# reference):
            co_read_axis(axis, $3680, 0, 4, vr)
        type 4 = Unsigned32

        Returns:
            0 = not in sampling status
            1 = sampling in progress
            2 = sampling done
        """
        vr = self.vr_scratch
        self.connection.SetVrValue(vr, _VR_SENTINEL)
        cmd = f"co_read_axis({self.axis}, $3680, 0, 4, {vr})"
        self.connection.Execute(cmd)

        deadline = time.monotonic() + _SDO_TIMEOUT
        while time.monotonic() < deadline:
            val = self.connection.GetVrValue(vr)
            if val != _VR_SENTINEL:
                raw = int(val)
                status = (raw >> 14) & 0x3
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

        # slave_position is 1-based in the ETHERCAT command (axis + 1)
        slave = self.axis + 1

        # Step 1: Initiate CoE FIFO file transfer on the controller
        # $161 = EC_COE_FIFO transfer function
        # 16000 bytes = 8000 words × 2 bytes/word
        cmd = f"ethercat($161, 0, {slave}, $3687, 0, 16000)"
        logger.debug("FIFO transfer cmd: %s", cmd)
        self.connection.Execute(cmd)

        if progress_callback:
            progress_callback(0.1, "Waiting for FIFO transfer...")

        # Wait for the transfer to complete on the controller
        time.sleep(2.0)

        if progress_callback:
            progress_callback(0.3, "Downloading file from controller...")

        # Step 2: Download the FIFO file to PC
        # Python API requires a progress callback: (ProgressInfo) -> None
        def _download_progress(info):
            logger.debug("DownloadFile progress: pos=%s", info.current_pos)

        try:
            self.connection.DownloadFile(local_filename, "EC_COE_FIFO", _download_progress)
        except Exception as e:
            logger.error("DownloadFile failed: %s", e)
            raise RuntimeError(f"Failed to download drive scope data: {e}") from e

        if progress_callback:
            progress_callback(0.8, "Parsing binary data...")

        # Step 3: Parse binary file
        import pathlib
        file_path = pathlib.Path(local_filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Downloaded file not found: {local_filename}")

        raw_bytes = file_path.read_bytes()
        elapsed = time.monotonic() - read_start
        logger.info(
            "FIFO download complete: %d bytes in %.1f s",
            len(raw_bytes), elapsed,
        )

        if progress_callback:
            progress_callback(1.0, "Data download complete")

        return self._parse_raw_bytes(raw_bytes)

    def _parse_raw_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Parse binary data downloaded via EC_COE_FIFO.

        The data layout is interleaved across ACTIVE channels only (not
        always 8).  Each sample row contains one 16-bit word per active
        channel, so the stride per sample = active_channels words.

        C# reference (Uapi.cs):
            int colonne = 6;  // words per sample row
            for (i = 0, j = 0; j < 1000; j++, i += colonne * 2)
        """
        n_bytes = len(raw_bytes)
        n_words = n_bytes // 2
        n_ch = self.active_channels

        logger.info(
            "Parsing %d bytes (%d words), %d active channels, "
            "stride=%d words/sample",
            n_bytes, n_words, n_ch, n_ch,
        )

        # Convert bytes to uint16 array (little-endian)
        raw_words = np.frombuffer(raw_bytes[:n_words * 2], dtype=np.dtype('<u2'))

        # Expected useful data: active_channels * 1000 words
        expected_words = n_ch * SAMPLES_PER_CHANNEL
        if len(raw_words) < expected_words:
            logger.warning(
                "Got %d words, expected %d (%d ch × %d samples) — padding",
                len(raw_words), expected_words, n_ch, SAMPLES_PER_CHANNEL,
            )
            padded = np.zeros(expected_words, dtype=np.uint16)
            padded[:len(raw_words)] = raw_words
            raw_words = padded

        # Take only the words we need (buffer may be larger)
        raw_words = raw_words[:expected_words]

        # Reshape to (1000, active_channels) — each row is one sample
        data_2d = raw_words.reshape(SAMPLES_PER_CHANNEL, n_ch)

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
