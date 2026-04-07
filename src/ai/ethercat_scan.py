"""
EtherCAT network discovery for Trio controllers.

Scans all EtherCAT slots (0–3), enumerates online slaves, and returns
a structured list of discovered devices with axis mappings.
"""

import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import Trio_UnifiedApi as TUA

logger = logging.getLogger(__name__)

# EtherCAT slots available on Trio controllers
_MAX_SLOTS = 4


@dataclass
class EthercatSlave:
    """One device discovered on the EtherCAT bus."""
    slot: int
    position: int           # physical position on the bus (0-based)
    address: int            # configured station address
    axis: int               # Trio axis number mapped to this slave
    online: bool = True
    drive_type: int = 0     # raw DRIVE_TYPE axis parameter
    drive_status: int = 0   # raw DRIVE_STATUS axis parameter
    slot_number: int = 0    # SLOT_NUMBER axis parameter


@dataclass
class EthercatSlot:
    """One EtherCAT port/slot on the controller."""
    slot: int
    state: TUA.EthercatState = TUA.EthercatState.Initial
    num_slaves: int = 0
    slaves: list[EthercatSlave] = field(default_factory=list)

    @property
    def state_name(self) -> str:
        return {
            TUA.EthercatState.Initial: "Initial",
            TUA.EthercatState.PreOperational: "Pre-Operational",
            TUA.EthercatState.SafeOperational: "Safe-Operational",
            TUA.EthercatState.Operational: "Operational",
        }.get(self.state, f"Unknown ({self.state})")

    @property
    def is_operational(self) -> bool:
        return self.state == TUA.EthercatState.Operational


@dataclass
class EthercatNetwork:
    """Complete scan result for all slots."""
    slots: list[EthercatSlot] = field(default_factory=list)

    @property
    def all_slaves(self) -> list[EthercatSlave]:
        return [s for slot in self.slots for s in slot.slaves]

    @property
    def online_slaves(self) -> list[EthercatSlave]:
        return [s for s in self.all_slaves if s.online]

    @property
    def active_slots(self) -> list[EthercatSlot]:
        return [s for s in self.slots if s.num_slaves > 0]


def scan_network(
    connection: TUA.TrioConnection,
    conn_lock: Optional[threading.Lock] = None,
) -> EthercatNetwork:
    """
    Scan all EtherCAT slots and enumerate slaves.

    Returns an EthercatNetwork with all discovered devices.
    Safe to call at any time — failed queries are logged and skipped.

    Parameters
    ----------
    connection : active TUA.TrioConnection
    conn_lock  : optional lock to serialize access to the connection
    """
    lock = conn_lock or contextlib.nullcontext()
    network = EthercatNetwork()

    # Hold the lock for the entire scan to prevent watchdog heartbeats
    # from interleaving — rapid command alternation crashes the connection.
    with lock:
        for slot_idx in range(_MAX_SLOTS):
            slot = EthercatSlot(slot=slot_idx)

            # Check slot state
            try:
                slot.state = connection.Ethercat_GetState(slot_idx)
            except Exception:
                logger.debug("Slot %d: not available", slot_idx)
                network.slots.append(slot)
                continue

            # Count slaves
            try:
                slot.num_slaves = connection.Ethercat_CheckNumberOfSlaves(slot_idx)
            except Exception as exc:
                logger.debug("Slot %d: cannot count slaves — %s", slot_idx, exc)
                network.slots.append(slot)
                continue

            if slot.num_slaves == 0:
                network.slots.append(slot)
                continue

            logger.info(
                "Slot %d: state=%s, %d slave(s)",
                slot_idx, slot.state_name, slot.num_slaves,
            )

            # Enumerate each slave
            for pos in range(slot.num_slaves):
                slave = EthercatSlave(slot=slot_idx, position=pos, address=0, axis=-1)

                try:
                    slave.online = connection.Ethercat_CheckSlaveOnline(slot_idx, pos)
                except Exception:
                    slave.online = False

                try:
                    slave.address = connection.Ethercat_GetSlaveAddress(slot_idx, pos)
                except Exception:
                    pass

                try:
                    slave.axis = connection.Ethercat_GetSlaveAxis(slot_idx, pos)
                except Exception:
                    pass

                # If we got a valid axis, read drive parameters
                if slave.axis >= 0:
                    try:
                        slave.drive_type = connection.GetAxisParameter_DRIVE_TYPE(slave.axis)
                    except Exception:
                        pass
                    try:
                        slave.drive_status = connection.GetAxisParameter_DRIVE_STATUS(slave.axis)
                    except Exception:
                        pass
                    try:
                        slave.slot_number = connection.GetAxisParameter_SLOT_NUMBER(slave.axis)
                    except Exception:
                        pass

                logger.info(
                    "  Slave %d: addr=%d, axis=%d, online=%s, drive_type=%d",
                    pos, slave.address, slave.axis, slave.online, slave.drive_type,
                )
                slot.slaves.append(slave)

            network.slots.append(slot)

    return network
