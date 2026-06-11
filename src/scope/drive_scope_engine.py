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

The implementation is split across focused modules:
    drive_scope_constants.py  SDO indices, variable tables, trigger modes
    drive_scope_coe.py        low-level CoE SDO read/write helpers
    drive_scope_config.py     configure() — capture setup writes
    drive_scope_transfer.py   EC_COE_FIFO transfer and file download
    drive_scope_parsing.py    binary/CSV payload decoding
"""

import logging
import time
from typing import Callable, List, Optional

try:
    import Trio_UnifiedApi as TUA
except ImportError:
    TUA = None

from .drive_scope_coe import (
    _SDO_TIMEOUT,
    _VR_SENTINEL,
    _fast_coe_read,
    _fast_coe_write,
    _get_u16,
)
from .drive_scope_config import DriveScopeConfigMixin
from .drive_scope_constants import (
    COMMON_DRIVE_VARIABLES,
    CONTROL_INDEX,
    DATA_INDEX,
    DATA_TYPES,
    DRIVE_VARIABLES,
    EXPECTED_CAPTURE_BYTES,
    NUM_CHANNELS,
    SAMPLE_TIME_UNIT_US,
    SAMPLES_PER_CHANNEL,
    SETUP_INDEX,
    STATUS_INDEX,
    SUPPORTED_DRIVE_TYPES,
    TOTAL_WORDS,
    TRIGGER_MODES,
)
from .drive_scope_parsing import DriveScopeParsingMixin
from .drive_scope_transfer import DriveScopeTransferMixin

logger = logging.getLogger(__name__)

__all__ = [
    "DriveScopeEngine",
    "COMMON_DRIVE_VARIABLES",
    "CONTROL_INDEX",
    "DATA_INDEX",
    "DATA_TYPES",
    "DRIVE_VARIABLES",
    "EXPECTED_CAPTURE_BYTES",
    "NUM_CHANNELS",
    "SAMPLE_TIME_UNIT_US",
    "SAMPLES_PER_CHANNEL",
    "SETUP_INDEX",
    "STATUS_INDEX",
    "SUPPORTED_DRIVE_TYPES",
    "TOTAL_WORDS",
    "TRIGGER_MODES",
]


class DriveScopeEngine(
    DriveScopeConfigMixin,
    DriveScopeTransferMixin,
    DriveScopeParsingMixin,
):
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
