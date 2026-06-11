"""
Low-level CoE SDO helpers for the drive scope.

Single-object reads/writes against the drive via the Trio Unified API,
using a scratch VR register for read-back with tight polling.
"""

import time

try:
    import Trio_UnifiedApi as TUA
except ImportError:
    TUA = None

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
