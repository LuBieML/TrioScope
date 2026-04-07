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

import logging
from typing import Optional

import Trio_UnifiedApi as TUA

from .drive_profile import DriveProfile, ETHERCAT_OBJECT_IDS, PARAM_DEFS

logger = logging.getLogger(__name__)

# Convenience alias used throughout this module
_U16 = TUA.Co_ObjectType.Unsigned16

# Subindex for all simple (non-array) DX3/DX4 Pn objects
_SUBINDEX = 0x00


# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def coe_read_axis(
    connection: TUA.TrioConnection,
    axis: int,
    object_index: int,
    subindex: int = _SUBINDEX,
    obj_type: TUA.Co_ObjectType = _U16,
    vr_scratch: int = 0,
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
    connection.Ethercat_CoReadAxis(axis, object_index, subindex, obj_type, vr_scratch)
    return int(connection.GetVrValue(vr_scratch))


def coe_write_axis(
    connection: TUA.TrioConnection,
    axis: int,
    object_index: int,
    value: int,
    subindex: int = _SUBINDEX,
    obj_type: TUA.Co_ObjectType = _U16,
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
_PN_OBJECTS: dict[str, tuple[int, TUA.Co_ObjectType]] = {
    attr: (ETHERCAT_OBJECT_IDS[attr], _U16)
    for attr in ETHERCAT_OBJECT_IDS
}


def read_drive_profile(
    connection: TUA.TrioConnection,
    axis: int,
    drive_type: str = "DX4",
    vr_scratch: int = 0,
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
    """
    profile = DriveProfile(drive_type=drive_type)

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

    return profile


def write_drive_profile(
    connection: TUA.TrioConnection,
    axis: int,
    profile: DriveProfile,
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
    """
    results: dict[str, Optional[Exception]] = {}

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
    vr_scratch: int = 0,
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
