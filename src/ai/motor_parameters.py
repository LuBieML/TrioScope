"""DX motor metadata used by the inertia estimator.

The DX3 object dictionary exposes the connected motor's nameplate values as
Pn parameters.  DX4 drives use the same CoE object indices.  Keep the raw
register decoding Qt-free so it can be validated without drive hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


MOTOR_PARAMETER_OBJECT_IDS: dict[str, int] = {
    "rated_torque": 0x348E,   # Pn810, 0.01 Nm
    "rated_current": 0x3490,  # Pn812, 0.1 A
    "rotor_inertia": 0x34A3,  # Pn831, 1e-8 kg.m^2
    "encoder_bits": 0x34D4,   # Pn880, bits used by the drive program
}


@dataclass(frozen=True)
class DetectedMotorParameters:
    """Decoded motor values, with unavailable fields left unset."""

    rated_torque_nm: Optional[float] = None
    rated_current_a: Optional[float] = None
    rotor_inertia_units: Optional[int] = None
    encoder_resolution_counts: Optional[int] = None
    failures: Mapping[str, str] = field(default_factory=dict)

    @property
    def detected_fields(self) -> tuple[str, ...]:
        fields = []
        if self.rated_torque_nm is not None:
            fields.append("rated torque")
        if self.rated_current_a is not None:
            fields.append("rated current")
        if self.rotor_inertia_units is not None:
            fields.append("rotor inertia")
        if self.encoder_resolution_counts is not None:
            fields.append("encoder resolution")
        return tuple(fields)


def decode_motor_parameter_registers(
    raw: Mapping[str, int],
    failures: Optional[Mapping[str, str]] = None,
) -> DetectedMotorParameters:
    """Convert raw Pn810/Pn812/Pn831/Pn880 values to UI units.

    Values outside the ranges documented by the DX3 manual are rejected
    individually so a supported subset can still populate the UI.
    """

    problems = dict(failures or {})

    def valid(name: str, minimum: int, maximum: int) -> Optional[int]:
        if name not in raw:
            return None
        value = int(raw[name])
        if minimum <= value <= maximum:
            return value
        problems[name] = (
            f"raw value {value} is outside the documented "
            f"range {minimum}..{maximum}"
        )
        return None

    torque_raw = valid("rated_torque", 1, 10_000)
    current_raw = valid("rated_current", 1, 2_000)
    inertia_raw = valid("rotor_inertia", 1, 100_000)
    encoder_bits = valid("encoder_bits", 1, 24)

    return DetectedMotorParameters(
        rated_torque_nm=(torque_raw * 0.01 if torque_raw is not None else None),
        rated_current_a=(current_raw * 0.1 if current_raw is not None else None),
        rotor_inertia_units=inertia_raw,
        encoder_resolution_counts=(
            1 << encoder_bits if encoder_bits is not None else None
        ),
        failures=problems,
    )
