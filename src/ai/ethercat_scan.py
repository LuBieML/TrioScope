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

# CoE Identity Object (0x1018) subindices
_IDENTITY_INDEX = 0x1018
_SUBIDX_VENDOR_ID = 1
_SUBIDX_PRODUCT_CODE = 2
_SUBIDX_REVISION = 3
_SUBIDX_SERIAL = 4

# ---------------------------------------------------------------------------
# EtherCAT Vendor ID lookup table
# Source: ETG (EtherCAT Technology Group) vendor registry.
# Extend as needed when encountering new devices.
# ---------------------------------------------------------------------------
VENDOR_NAMES: dict[int, str] = {
    0x00000001: "EtherCAT Technology Group",
    0x00000002: "Beckhoff Automation",
    0x00000004: "KEB Automation",
    0x0000000E: "Bosch Rexroth",
    0x00000022: "Lenze",
    0x00000044: "Wago",
    0x00000048: "B&R Industrial Automation",
    0x0000004C: "ifm electronic",
    0x0000006A: "Festo",
    0x00000083: "Omron",
    0x000000AB: "Trio Motion Technology",
    0x000002DE: "Trio Motion Technology",
    0x000000B9: "SEW-Eurodrive",
    0x000000C7: "Pilz",
    0x000000E4: "Hilscher",
    0x000000FB: "SMC Corporation",
    0x00000127: "Mitsubishi Electric",
    0x0000014E: "Baumer",
    0x00000195: "Sick",
    0x000001DD: "Delta Electronics",
    0x00000226: "Oriental Motor",
    0x0000029C: "Keyence",
    0x000002BE: "Sanyo Denki",
    0x00000539: "Yaskawa Electric",
    0x0000054D: "Panasonic",
    0x00000569: "Maxon Motor",
    0x000005A2: "Nanotec Electronic",
    0x00000659: "Schneider Electric",
    0x0000066F: "Inovance Technology",
    0x00000A13: "Elmo Motion Control",
    0x00100000: "Copley Controls",
}


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
    vendor_id: int = 0      # EtherCAT vendor ID from Identity Object 0x1018
    product_code: int = 0   # product code from Identity Object 0x1018
    revision: int = 0       # revision number from Identity Object 0x1018
    serial_number: int = 0  # serial number from Identity Object 0x1018

    @property
    def vendor_name(self) -> str:
        """Human-readable vendor name, or hex ID if unknown."""
        return VENDOR_NAMES.get(self.vendor_id, f"Unknown (0x{self.vendor_id:08X})")


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


def _read_identity_field(
    connection: TUA.TrioConnection,
    slot: int,
    position: int,
    subindex: int,
    vr_scratch: int = 900,
    timeout: float = 0.5,
) -> int:
    """Read one subindex of the Identity Object (0x1018) via SDO.

    Uses a shorter timeout than normal CoE reads since identity
    responses are immediate on present slaves.
    """
    import time
    _SENTINEL = -9999.0
    connection.SetVrValue(vr_scratch, _SENTINEL)
    connection.Ethercat_CoRead(
        slot, position, _IDENTITY_INDEX, subindex,
        TUA.Co_ObjectType.Unsigned32, vr_scratch,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = connection.GetVrValue(vr_scratch)
        if val != _SENTINEL:
            return int(val)
        time.sleep(0.05)
    raise TimeoutError(
        f"Identity read timed out — slot {slot}, slave {position}, sub {subindex}"
    )


def read_slave_vendor(
    connection: TUA.TrioConnection,
    slave: EthercatSlave,
    conn_lock: Optional[threading.Lock] = None,
) -> None:
    """Read vendor ID for a single slave via SDO (0x1018:1).

    Updates *slave.vendor_id* in place.  Designed to be called after
    the initial scan, outside the main scan lock.
    """
    lock = conn_lock or contextlib.nullcontext()
    with lock:
        try:
            slave.vendor_id = _read_identity_field(
                connection, slave.slot, slave.position, _SUBIDX_VENDOR_ID,
            )
        except Exception as exc:
            logger.debug("Slave %d: vendor read failed — %s", slave.position, exc)


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

    # Use per-call locking so the watchdog heartbeat can interleave
    # between commands.  Each API call is individually short.

    def _call(fn, *args, default=None):
        """Execute one API call under the lock, return *default* on failure."""
        try:
            with lock:
                return fn(*args)
        except Exception:
            return default

    for slot_idx in range(_MAX_SLOTS):
        slot = EthercatSlot(slot=slot_idx)

        state = _call(connection.Ethercat_GetState, slot_idx)
        if state is None:
            logger.debug("Slot %d: not available", slot_idx)
            network.slots.append(slot)
            continue
        slot.state = state

        n = _call(connection.Ethercat_CheckNumberOfSlaves, slot_idx, default=0)
        slot.num_slaves = int(n)

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

            raw_online = _call(connection.Ethercat_CheckSlaveOnline, slot_idx, pos,
                               default=False)
            slave.online = bool(raw_online)

            addr = _call(connection.Ethercat_GetSlaveAddress, slot_idx, pos, default=0)
            slave.address = int(addr)

            ax = _call(connection.Ethercat_GetSlaveAxis, slot_idx, pos, default=-1)
            slave.axis = int(ax)

            # Skip ghost slaves (configured but not physically present)
            if not slave.online and slave.address == 0:
                logger.debug("  Slave %d: skipping (not present)", pos)
                slot.slaves.append(slave)
                continue

            # If we got a valid axis, read drive parameters
            if slave.axis >= 0:
                dt = _call(connection.GetAxisParameter_DRIVE_TYPE, slave.axis, default=0)
                slave.drive_type = int(dt)
                ds = _call(connection.GetAxisParameter_DRIVE_STATUS, slave.axis, default=0)
                slave.drive_status = int(ds)
                sn = _call(connection.GetAxisParameter_SLOT_NUMBER, slave.axis, default=0)
                slave.slot_number = int(sn)

            logger.info(
                "  Slave %d: addr=%d, axis=%d, online=%s, drive_type=%d",
                pos, slave.address, slave.axis, slave.online, slave.drive_type,
            )
            slot.slaves.append(slave)

        network.slots.append(slot)

    # ----- Axis mapping fallback ----------------------------------------
    # If Ethercat_GetSlaveAxis didn't work (returns -1 for all), try to
    # map axes to slaves by probing controller axes directly.
    all_slaves = network.all_slaves
    unmapped = [s for s in all_slaves if s.axis < 0 and s.online]

    if unmapped:
        logger.debug("Attempting axis mapping for %d unmapped slaves", len(unmapped))
        addr_to_slave: dict[int, EthercatSlave] = {}
        for s in unmapped:
            if s.address > 0:
                addr_to_slave[s.address] = s

        consecutive_fails = 0
        for ax in range(32):
            dt = _call(connection.GetAxisParameter_DRIVE_TYPE, ax)
            if dt is None:
                consecutive_fails += 1
                if consecutive_fails >= 3:
                    break
                continue

            consecutive_fails = 0
            dt = int(dt)
            if dt == 0:
                continue

            sn_raw = _call(connection.GetAxisParameter_SLOT_NUMBER, ax, default=0)
            sn = int(sn_raw)

            if sn in addr_to_slave:
                slave = addr_to_slave.pop(sn)
                slave.axis = ax
                slave.drive_type = dt
                ds = _call(connection.GetAxisParameter_DRIVE_STATUS, ax, default=0)
                slave.drive_status = int(ds)
                slave.slot_number = sn
                logger.info(
                    "  Axis %d → slave addr %d (drive_type=%d)",
                    ax, sn, dt,
                )
                if not addr_to_slave:
                    break

    return network
