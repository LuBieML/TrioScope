"""Per-axis motion limits edited in the Axis setup tab."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict


@dataclass
class AxisParameterConfig:
    """Controller axis parameters persisted by and written from TrioScope."""

    axis: int = 0
    speed: float = 100.0
    units: float = 2097152.0
    accel: float = 3000.0
    decel: float = 3000.0
    fast_dec: float = 50000.0
    jerk: float = 100000.0
    fwd_in: int = -1
    rev_in: int = -1
    fe_limit: float = 10.0
    drive_fe_limit: int = 10
    fe_range: float = 10.0
    fs_limit: float = 1000.0
    rs_limit: float = -1000.0

    def validate(self) -> None:
        if isinstance(self.axis, bool) or not isinstance(self.axis, int):
            raise ValueError("Axis must be an integer between 0 and 25.")
        if not 0 <= self.axis <= 25:
            raise ValueError(f"Axis must be between 0 and 25 (got {self.axis}).")

        for field_name, value in asdict(self).items():
            if field_name == "axis":
                continue
            if field_name in ("fwd_in", "rev_in"):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{field_name} must be a whole-number input index.")
                if not -32768 <= value <= 32767:
                    raise ValueError(f"{field_name} must fit the UAPI int16 range.")
                continue
            if field_name == "drive_fe_limit":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("drive_fe_limit must be a whole number.")
                if not -(2**63) <= value <= 2**63 - 1:
                    raise ValueError("drive_fe_limit must fit the UAPI int64 range.")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be a finite number.")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxisParameterConfig":
        if not isinstance(data, dict):
            raise ValueError("Each axis configuration must be a JSON object.")

        defaults = cls()
        try:
            config = cls(
                axis=_parse_axis(data.get("axis", defaults.axis)),
                speed=float(data.get("speed", defaults.speed)),
                units=float(data.get("units", defaults.units)),
                accel=float(data.get("accel", defaults.accel)),
                decel=float(data.get("decel", defaults.decel)),
                fast_dec=float(data.get("fast_dec", defaults.fast_dec)),
                jerk=float(data.get("jerk", defaults.jerk)),
                fwd_in=_parse_int16(data.get("fwd_in", defaults.fwd_in), "fwd_in"),
                rev_in=_parse_int16(data.get("rev_in", defaults.rev_in), "rev_in"),
                fe_limit=float(data.get("fe_limit", defaults.fe_limit)),
                drive_fe_limit=_parse_int64(
                    data.get("drive_fe_limit", defaults.drive_fe_limit),
                    "drive_fe_limit",
                ),
                fe_range=float(data.get("fe_range", defaults.fe_range)),
                fs_limit=float(data.get("fs_limit", defaults.fs_limit)),
                rs_limit=float(data.get("rs_limit", defaults.rs_limit)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid axis parameter value: {exc}") from exc

        config.validate()
        return config


def _parse_int16(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a whole-number input index")
    parsed = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be a whole-number input index")
    if not -32768 <= parsed <= 32767:
        raise ValueError(f"{field_name} must fit the UAPI int16 range")
    return parsed


def _parse_int64(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a whole number")
    parsed = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be a whole number")
    if not -(2**63) <= parsed <= 2**63 - 1:
        raise ValueError(f"{field_name} must fit the UAPI int64 range")
    return parsed


def _parse_axis(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("axis must be an integer between 0 and 25")
    parsed = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("axis must be an integer between 0 and 25")
    return parsed
