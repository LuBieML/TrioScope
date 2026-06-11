"""
Drive scope configuration (mixin for DriveScopeEngine).

Writes the capture setup — channels, sample time, trigger — to the drive,
handling both the DX3/DX4 COMBO setup object (0x368C) and the DX5/DX1
Z_AScope* object family.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .drive_scope_constants import (
    CONTROL_INDEX,
    NUM_CHANNELS,
    SAMPLES_PER_CHANNEL,
    SETUP_INDEX,
    SUPPORTED_DRIVE_TYPES,
    TRIGGER_MODES,
)

logger = logging.getLogger(__name__)


class DriveScopeConfigMixin:
    """configure() implementation for DriveScopeEngine."""

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

        if self.drive_model in ("DX5", "DX1"):
            self._configure_dx5_dx1()
        else:
            self._configure_dx3_dx4()

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

    def _configure_dx5_dx1(self) -> None:
        """Write the Z_AScope* setup objects used by DX5/DX1 drives."""
        sleep_time = 0.05

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

    def _configure_dx3_dx4(self) -> None:
        """Write the 0x368C COMBO setup object used by DX3/DX4 drives."""
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
