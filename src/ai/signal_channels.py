"""Axis-aware scope channel resolution.

Capture buffers key channels by display name — ``"MPOS(0)"``,
``"DRIVE_FE(3)"``, ``"IN Ch(2)"`` — or, for Drive Scope captures, by drive
variable label such as ``"Speed Feedback (rpm)"`` (no axis suffix; the axis
is global in drive mode).

Resolution is exact-name-first so that FE never falls back to FE_LATCH and
DEMAND_SPEED never silently binds to DEMAND_SPEED_NORMALISED, and
axis-aware so multi-axis captures cannot pair channels from different axes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# "MPOS(0)" / "IN Ch(2)" → (base, axis).  Only an all-digits parenthetical
# counts as an axis; "(rpm)" or "(0x0F10)" are unit/address annotations.
_AXIS_SUFFIX_RE = re.compile(r"^(.*?)\s*(?:ch)?\s*\((\d+)\)\s*$", re.IGNORECASE)
_ANNOTATION_SUFFIX_RE = re.compile(r"^(.*?)\s*\([^)]*\)\s*$")


def _flatten(name: str) -> str:
    return (name.lower().replace("_", "").replace(" ", "").replace("-", ""))


def split_channel_key(key: str) -> tuple[str, int | None]:
    """Return (normalized_base_name, axis) for a capture-buffer key.

    Axis is None for keys without an axis suffix (drive-mode labels).
    """
    stripped = key.strip()
    m = _AXIS_SUFFIX_RE.match(stripped)
    if m:
        return _flatten(m.group(1)), int(m.group(2))
    m = _ANNOTATION_SUFFIX_RE.match(stripped)
    if m:
        return _flatten(m.group(1)), None
    return _flatten(stripped), None


def detect_axes(params: dict) -> list[int]:
    """Sorted list of axis numbers that appear in the capture's channel keys."""
    axes = {
        axis
        for _, axis in (split_channel_key(k) for k in params)
        if axis is not None
    }
    return sorted(axes)


@dataclass(frozen=True)
class ChannelSpec:
    """Matching rule for one logical channel role.

    ``exact`` names are tried first, in order (earlier = preferred source).
    ``substrings`` are a fallback; a key containing any ``exclude`` term is
    never substring-matched (guards against FE_LIMIT, AXIS_MAX_TORQUE, …).
    """
    exact: tuple[str, ...]
    substrings: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


CHANNEL_SPECS: dict[str, ChannelSpec] = {
    "dpos": ChannelSpec(
        exact=("dpos",),
        substrings=("demandposition", "targetposition", "positiondemand"),
    ),
    "mpos": ChannelSpec(
        exact=("mpos",),
        substrings=("measuredposition", "actualposition", "positionactual"),
    ),
    "fe": ChannelSpec(
        exact=("drivefe", "fe"),
        substrings=("followingerror", "positiondeviation"),
        exclude=("limit", "latch", "range", "mode"),
    ),
    # Demand velocity already in the same units as measured velocity:
    # the app's virtual units/s channel, or drive-scope rpm command.
    "demand_vel_native": ChannelSpec(
        exact=("demandspeednormalised", "demandspeednormalized",
               "speedcommand", "spdcmdrpm"),
    ),
    # Controller DEMAND_SPEED — units/servocycle, needs 1/servo_period.
    "demand_vel_raw": ChannelSpec(
        exact=("demandspeed", "dspeed"),
        substrings=("demandvel", "velocitydemand"),
        exclude=("normalised", "normalized"),
    ),
    "measured_vel": ChannelSpec(
        exact=("mspeed", "mspeedf", "speedfeedback", "spdfbrpm"),
        substrings=("measuredvel", "actualvel", "vactual", "velocityactual"),
    ),
    "current": ChannelSpec(
        exact=("drivecurrent", "drivetorque", "dacout", "torquecommand", "tn"),
        substrings=("current", "torque"),
        exclude=("max", "limit", "postorque", "negtorque",
                 "controlword", "statusword", "currentpos"),
    ),
}


def resolve_channel(params: dict, spec: ChannelSpec,
                    axis: int | None = None) -> str | None:
    """Find the capture key for one channel role, or None.

    When ``axis`` is given, only keys carrying that axis suffix (or no axis
    suffix at all — drive-mode labels) are eligible.
    """
    entries = []
    for key in params:
        name, key_axis = split_channel_key(key)
        if axis is not None and key_axis is not None and key_axis != axis:
            continue
        entries.append((key, name))

    for target in spec.exact:
        for key, name in entries:
            if name == target:
                return key
    for sub in spec.substrings:
        for key, name in entries:
            if sub in name and not any(x in name for x in spec.exclude):
                return key
    return None


def resolve_channels(params: dict, axis: int | None = None) -> dict[str, str | None]:
    """Resolve every channel role for one axis. Returns {role: key_or_None}."""
    return {
        role: resolve_channel(params, spec, axis)
        for role, spec in CHANNEL_SPECS.items()
    }
