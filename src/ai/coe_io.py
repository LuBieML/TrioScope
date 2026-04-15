"""
CoE (CANopen over EtherCAT) SDO read/write helpers for Trio DX3/DX4 drives.

Uses the Trio UnifiedAPI EtherCAT_CoRead/CoWrite family of calls.

Read flow:
    Ethercat_CoReadAxis(axis, index, subindex, type, vr_scratch)
    → value lands in VR[vr_scratch]
    → GetVrValue(vr_scratch) returns it as float → cast to int

Write flow:
    Ethercat_CoWriteAxis_Value(axis, index, subindex, type, int_value)

All DX3/DX4 Pn parameters are UINT16, so Co_ObjectType.Unsigned16 is the
default throughout.  The scratch VR defaults to VR 0; pass a different index
if that VR is in use by your application.
"""

import contextlib
import logging
import threading
import time
from typing import Optional

import Trio_UnifiedApi as TUA

from .drive_profile import DriveProfile, ETHERCAT_OBJECT_IDS, PARAM_DEFS, decode_pn100, encode_pn100

logger = logging.getLogger(__name__)

# Convenience alias used throughout this module — DX4 Pn objects are U32
_U32 = TUA.Co_ObjectType.Unsigned32

# Subindex for all simple (non-array) DX3/DX4 Pn objects
_SUBINDEX = 0x00

# Sentinel value written to the scratch VR before issuing a CoRead.
# Pn parameters are UINT16 (0–65535), so -9999.0 can never be a valid result.
_VR_SENTINEL = -9999.0

# Maximum time (seconds) to wait for the SDO response to land in the VR.
_SDO_TIMEOUT = 2.0

# Polling interval (seconds) between VR checks.
_SDO_POLL_INTERVAL = 0.05


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def coe_read_axis(
    connection: TUA.TrioConnection,
    axis: int,
    object_index: int,
    subindex: int = _SUBINDEX,
    obj_type: TUA.Co_ObjectType = _U32,
    vr_scratch: int = 900,
) -> int:
    """
    Read one CoE object from the drive on *axis* and return its value.

    Parameters
    ----------
    connection    : active TUA.TrioConnection
    axis          : Trio axis number (0-based)
    object_index  : CANopen object index  (e.g. 0x31CA for Pn102)
    subindex      : CANopen sub-index     (0x00 for all Pn parameters)
    obj_type      : Co_ObjectType enum value (default Unsigned16)
    vr_scratch    : VR index used as scratch space for the result

    Returns
    -------
    int  — the raw register value read from the drive
    """
    # Write a sentinel so we can detect when the SDO response arrives.
    connection.SetVrValue(vr_scratch, _VR_SENTINEL)

    # Verify sentinel was set
    check = connection.GetVrValue(vr_scratch)
    if check != _VR_SENTINEL:
        logger.warning(
            "VR %d sentinel verify failed: wrote %s, read back %s "
            "(VR may be in use by BASIC program)",
            vr_scratch, _VR_SENTINEL, check,
        )

    connection.Ethercat_CoReadAxis(axis, object_index, subindex, obj_type, vr_scratch)

    # Poll until the VR is no longer the sentinel (SDO response has landed).
    deadline = time.monotonic() + _SDO_TIMEOUT
    poll_count = 0
    last_val = _VR_SENTINEL
    while time.monotonic() < deadline:
        val = connection.GetVrValue(vr_scratch)
        last_val = val
        poll_count += 1
        if val != _VR_SENTINEL:
            logger.debug(
                "Axis %d obj 0x%04X: SDO response in %d polls, value=%s",
                axis, object_index, poll_count, val,
            )
            return int(val)
        time.sleep(_SDO_POLL_INTERVAL)

    logger.debug(
        "Axis %d obj 0x%04X: timed out after %d polls, last VR=%s",
        axis, object_index, poll_count, last_val,
    )
    raise TimeoutError(
        f"SDO read timed out after {_SDO_TIMEOUT}s — "
        f"axis {axis}, object 0x{object_index:04X}"
    )


def coe_read_slot(
    connection: TUA.TrioConnection,
    slot: int,
    slave_position: int,
    object_index: int,
    subindex: int = _SUBINDEX,
    obj_type: TUA.Co_ObjectType = _U32,
    vr_scratch: int = 900,
) -> int:
    """
    Read one CoE object using slot + slave position (not axis number).

    Uses Ethercat_CoRead(slot, slave_pos, index, subindex, type, vr).
    Useful when axis mapping is unknown or unreliable.
    """
    connection.SetVrValue(vr_scratch, _VR_SENTINEL)
    connection.Ethercat_CoRead(slot, slave_position, object_index, subindex, obj_type, vr_scratch)

    deadline = time.monotonic() + _SDO_TIMEOUT
    while time.monotonic() < deadline:
        val = connection.GetVrValue(vr_scratch)
        if val != _VR_SENTINEL:
            return int(val)
        time.sleep(_SDO_POLL_INTERVAL)

    raise TimeoutError(
        f"SDO read timed out after {_SDO_TIMEOUT}s — "
        f"slot {slot}, slave {slave_position}, object 0x{object_index:04X}"
    )


def coe_write_axis(
    connection: TUA.TrioConnection,
    axis: int,
    object_index: int,
    value: int,
    subindex: int = _SUBINDEX,
    obj_type: TUA.Co_ObjectType = _U32,
) -> None:
    """
    Write one CoE object to the drive on *axis*.

    Parameters
    ----------
    connection    : active TUA.TrioConnection
    axis          : Trio axis number (0-based)
    object_index  : CANopen object index  (e.g. 0x31CA for Pn102)
    value         : integer value to write
    subindex      : CANopen sub-index     (0x00 for all Pn parameters)
    obj_type      : Co_ObjectType enum value (default Unsigned16)
    """
    connection.Ethercat_CoWriteAxis_Value(axis, object_index, subindex, obj_type, int(value))


# ---------------------------------------------------------------------------
# Drive-profile helpers (Pn parameter level)
# ---------------------------------------------------------------------------

# Maps DriveProfile attr name → (object_index, Co_ObjectType)
# Excludes pn100 which is handled specially (nibble-packed sub-fields)
_PN_OBJECTS: dict[str, tuple[int, TUA.Co_ObjectType]] = {
    attr: (ETHERCAT_OBJECT_IDS[attr], _U32)
    for attr in ETHERCAT_OBJECT_IDS
    if attr != "pn100"
}

_PN100_INDEX = ETHERCAT_OBJECT_IDS["pn100"]


def read_drive_profile(
    connection: TUA.TrioConnection,
    axis: int,
    drive_type: str = "DX4",
    vr_scratch: int = 900,
    conn_lock: Optional[threading.Lock] = None,
) -> DriveProfile:
    """
    Read all known Pn parameters from the drive on *axis* and return a
    populated DriveProfile.

    Parameters that fail (e.g. drive offline) are left as None in the profile
    rather than raising; the error is logged at WARNING level.

    Parameters
    ----------
    connection  : active TUA.TrioConnection
    axis        : Trio axis number (0-based)
    drive_type  : "DX3" or "DX4" (stored in the returned profile)
    vr_scratch  : VR index used as scratch space for reads
    conn_lock   : optional lock to serialize access to the connection
    """
    lock = conn_lock or contextlib.nullcontext()
    profile = DriveProfile(drive_type=drive_type)

    # Hold lock for entire read batch — prevents watchdog interleaving
    with lock:
        # Read Pn100 (nibble-packed) and decode into sub-fields
        try:
            raw = coe_read_axis(connection, axis, _PN100_INDEX, vr_scratch=vr_scratch)
            fields = decode_pn100(raw)
            profile.pn100_tuning_mode = fields["tuning_mode"]
            profile.pn100_vibration = fields["vibration_suppression"]
            profile.pn100_damping = fields["damping"]
            logger.debug(
                "Axis %d  PN100 (0x%04X) = 0x%04X → mode=%d, vib=%d, damp=%d",
                axis, _PN100_INDEX, raw,
                fields["tuning_mode"], fields["vibration_suppression"], fields["damping"],
            )
        except Exception as exc:
            logger.warning("Axis %d  PN100 (0x%04X): read failed — %s", axis, _PN100_INDEX, exc)

        # Read remaining Pn parameters (simple values)
        for attr, (obj_index, obj_type) in _PN_OBJECTS.items():
            try:
                val = coe_read_axis(connection, axis, obj_index, vr_scratch=vr_scratch)
                setattr(profile, attr, val)
                logger.debug("Axis %d  %s (0x%04X) = %d", axis, attr.upper(), obj_index, val)
            except Exception as exc:
                logger.warning(
                    "Axis %d  %s (0x%04X): read failed — %s",
                    axis, attr.upper(), obj_index, exc,
                )

    if not profile.has_any_values():
        raise ConnectionError(
            f"All parameter reads failed for axis {axis} — drive may be offline or connection lost"
        )

    return profile


def write_drive_profile(
    connection: TUA.TrioConnection,
    axis: int,
    profile: DriveProfile,
    conn_lock: Optional[threading.Lock] = None,
) -> dict[str, Optional[Exception]]:
    """
    Write all non-None Pn parameters from *profile* to the drive on *axis*.

    Returns a dict mapping attribute name → Exception (or None on success)
    so the caller can report per-parameter failures without aborting the batch.

    Parameters
    ----------
    connection : active TUA.TrioConnection
    axis       : Trio axis number (0-based)
    profile    : DriveProfile whose Pn fields will be written
    conn_lock  : optional lock to serialize access to the connection
    """
    lock = conn_lock or contextlib.nullcontext()
    results: dict[str, Optional[Exception]] = {}

    # Hold lock for entire write batch — prevents watchdog interleaving
    with lock:
        # Write Pn100 (encode sub-fields into nibble-packed value)
        if profile.pn100_tuning_mode is not None:
            raw = encode_pn100(
                tuning_mode=profile.pn100_tuning_mode,
                vibration_suppression=profile.pn100_vibration or 0,
                damping=profile.pn100_damping or 0,
            )
            try:
                coe_write_axis(connection, axis, _PN100_INDEX, raw)
                logger.debug(
                    "Axis %d  PN100 (0x%04X) ← 0x%04X (mode=%d, vib=%d, damp=%d)",
                    axis, _PN100_INDEX, raw,
                    profile.pn100_tuning_mode, profile.pn100_vibration or 0, profile.pn100_damping or 0,
                )
                results["pn100"] = None
            except Exception as exc:
                logger.warning("Axis %d  PN100 (0x%04X): write failed — %s", axis, _PN100_INDEX, exc)
                results["pn100"] = exc

        # Write remaining Pn parameters (simple values)
        for attr, (obj_index, obj_type) in _PN_OBJECTS.items():
            val = getattr(profile, attr, None)
            if val is None:
                continue
            try:
                coe_write_axis(connection, axis, obj_index, val)
                logger.debug("Axis %d  %s (0x%04X) ← %d", axis, attr.upper(), obj_index, val)
                results[attr] = None
            except Exception as exc:
                logger.warning(
                    "Axis %d  %s (0x%04X): write failed — %s",
                    axis, attr.upper(), obj_index, exc,
                )
                results[attr] = exc

    return results


def read_single_pn(
    connection: TUA.TrioConnection,
    axis: int,
    pn_attr: str,
    vr_scratch: int = 900,
) -> int:
    """
    Read a single Pn parameter by attribute name (e.g. ``"pn102"``).

    Raises KeyError if *pn_attr* is not in ETHERCAT_OBJECT_IDS.
    Raises TUA.TrioConnectionException on comms failure.
    """
    obj_index = ETHERCAT_OBJECT_IDS[pn_attr.lower()]
    return coe_read_axis(connection, axis, obj_index, vr_scratch=vr_scratch)


def write_single_pn(
    connection: TUA.TrioConnection,
    axis: int,
    pn_attr: str,
    value: int,
) -> None:
    """
    Write a single Pn parameter by attribute name (e.g. ``"pn102"``).

    Raises KeyError if *pn_attr* is not in ETHERCAT_OBJECT_IDS.
    Raises TUA.TrioConnectionException on comms failure.
    """
    obj_index = ETHERCAT_OBJECT_IDS[pn_attr.lower()]
    coe_write_axis(connection, axis, obj_index, value)
