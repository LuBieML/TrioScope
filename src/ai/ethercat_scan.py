"""
EtherCAT network discovery for Trio controllers.

Scans all EtherCAT slots (0–3), enumerates online slaves, and returns
a structured list of discovered devices with axis mappings.
"""

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import Trio_UnifiedApi as TUA

from . import ethercat_devices
from .ethercat_devices import VENDOR_NAMES  # noqa: F401 — re-exported for compat

logger = logging.getLogger(__name__)

# EtherCAT slots available on Trio controllers
_MAX_SLOTS = 1

# CoE Device Type object (0x1000) — low word is the CiA profile number
_DEVICE_TYPE_INDEX = 0x1000

# CoE Identity Object (0x1018) subindices
_IDENTITY_INDEX = 0x1018
_SUBIDX_VENDOR_ID = 1
_SUBIDX_PRODUCT_CODE = 2
_SUBIDX_REVISION = 3
_SUBIDX_SERIAL = 4


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
    device_type: int = 0    # CoE Device Type object 0x1000 (CiA profile)

    @property
    def vendor_name(self) -> str:
        """Human-readable vendor name, or hex ID if unknown."""
        return ethercat_devices.vendor_name(self.vendor_id)

    @property
    def product_name(self) -> str:
        """Best-effort short product name (e.g. 'DX4', 'EK1100', 'Drive')."""
        return ethercat_devices.product_label(
            self.vendor_id, self.product_code,
            device_type=self.device_type, drive_type=self.drive_type,
        )

    @property
    def profile_name(self) -> str:
        """CiA device profile name derived from object 0x1000."""
        return ethercat_devices.device_profile_name(self.device_type)

    @property
    def revision_str(self) -> str:
        return ethercat_devices.revision_str(self.revision)


@dataclass
class EthercatSlot:
    """One EtherCAT port/slot on the controller."""
    slot: int
    state: int = 0
    num_slaves: int = 0
    slaves: list[EthercatSlave] = field(default_factory=list)

    @property
    def state_name(self) -> str:
        # Standard ESM: 1=Init, 2=PreOp, 4=SafeOp, 8=Op
        # TUA Enum: 0=Initial/Init, 1=PreOp/Init, 2=SafeOp/PreOp, 3=Op/SafeOp
        # Map both standard ESM and TUA enum values to clear names:
        val = int(self.state)
        if val in (0, 1):
            return "Initial"
        elif val == 2:
            return "Pre-Operational"
        elif val == 4:
            return "Safe-Operational"
        elif val in (3, 8):
            return "Operational"
        else:
            return f"Unknown ({val})"

    @property
    def is_operational(self) -> bool:
        return int(self.state) in (3, 8)


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


def _coe_read_u32(
    connection: TUA.TrioConnection,
    slot: int,
    position: int,
    index: int,
    subindex: int,
    vr_scratch: int = 900,
    timeout: float = 0.5,
) -> int:
    """Read one CoE object (Unsigned32) via SDO using slot + slave position.

    Uses a shorter timeout than normal CoE reads since identity
    responses are immediate on present slaves.
    """
    _SENTINEL = -9999.0
    connection.SetVrValue(vr_scratch, _SENTINEL)
    connection.Ethercat_CoRead(
        slot, position, index, subindex,
        TUA.Co_ObjectType.Unsigned32, vr_scratch,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = connection.GetVrValue(vr_scratch)
        if val != _SENTINEL:
            return int(val)
        time.sleep(0.05)
    raise TimeoutError(
        f"CoE read timed out — slot {slot}, slave {position}, "
        f"object 0x{index:04X}:{subindex}"
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
            slave.vendor_id = _coe_read_u32(
                connection, slave.slot, slave.position,
                _IDENTITY_INDEX, _SUBIDX_VENDOR_ID,
            )
        except Exception as exc:
            logger.debug("Slave %d: vendor read failed — %s", slave.position, exc)


def _read_slave_identity(connection: TUA.TrioConnection, slave: EthercatSlave,
                         lock) -> None:
    """Populate identity (0x1018) and device-type (0x1000) fields in place.

    Each field is read independently under the lock; failures are logged
    and leave the field at its default so a partial identity is still useful.
    """
    fields = (
        ("vendor_id", _IDENTITY_INDEX, _SUBIDX_VENDOR_ID),
        ("product_code", _IDENTITY_INDEX, _SUBIDX_PRODUCT_CODE),
        ("revision", _IDENTITY_INDEX, _SUBIDX_REVISION),
        ("serial_number", _IDENTITY_INDEX, _SUBIDX_SERIAL),
        ("device_type", _DEVICE_TYPE_INDEX, 0),
    )
    for attr, index, sub in fields:
        try:
            with lock:
                value = _coe_read_u32(
                    connection, slave.slot, slave.position, index, sub,
                )
            setattr(slave, attr, value)
        except Exception as exc:
            logger.debug(
                "Slave %d: %s read (0x%04X:%d) failed — %s",
                slave.position, attr, index, sub, exc,
            )
            # Identity object is mandatory; if the first read fails the
            # mailbox is probably not reachable — don't retry the rest.
            if attr == "vendor_id":
                break
        time.sleep(0.01)


def scan_network(
    connection: TUA.TrioConnection,
    conn_lock: Optional[threading.Lock] = None,
    read_identity: bool = True,
) -> EthercatNetwork:
    """
    Scan all EtherCAT slots and enumerate slaves.

    Returns an EthercatNetwork with all discovered devices.
    Safe to call at any time — failed queries are logged and skipped.

    Parameters
    ----------
    connection    : active TUA.TrioConnection
    conn_lock     : optional lock to serialize access to the connection
    read_identity : also read CoE Identity (0x1018) and Device Type (0x1000)
                    for each online slave (a few SDO round-trips per device)
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
        finally:
            time.sleep(0.01)

    for slot_idx in range(_MAX_SLOTS):
        slot = EthercatSlot(slot=slot_idx)

        # Use VR(901) to fetch the raw EtherCAT state to avoid enum ValueError conversion issues.
        # Since Ethercat_GetState_VR returns None on success, we execute it and check if the VR value changed.
        vr_scratch = 901
        _call(connection.SetVrValue, vr_scratch, -999.0)
        _call(connection.Ethercat_GetState_VR, slot_idx, vr_scratch)
        state_val = _call(connection.GetVrValue, vr_scratch, default=-999.0)
        if state_val != -999.0:
            state = int(state_val)
        else:
            state = None

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

            # Read CoE identity (vendor / product / revision / serial)
            if read_identity and slave.online:
                _read_slave_identity(connection, slave, lock)

            logger.info(
                "  Slave %d: addr=%d, axis=%d, online=%s, drive_type=%d, "
                "vendor=0x%08X, product=0x%08X",
                pos, slave.address, slave.axis, slave.online, slave.drive_type,
                slave.vendor_id, slave.product_code,
            )
            slot.slaves.append(slave)

        network.slots.append(slot)

    # ----- Axis mapping fallback ----------------------------------------
    # If Ethercat_GetSlaveAxis didn't work (returns -1 for all online slaves),
    # try to map axes to slaves by probing controller axes directly.
    all_slaves = network.all_slaves
    online_slaves = [s for s in all_slaves if s.online]
    has_mapped_axis = any(s.axis >= 0 for s in online_slaves)

    if online_slaves and not has_mapped_axis:
        unmapped = online_slaves
        logger.debug("Attempting axis mapping fallback for %d unmapped slaves", len(unmapped))
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
