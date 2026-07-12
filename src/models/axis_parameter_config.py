"""Per-axis motion limits edited in the Axis setup tab."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict


@dataclass
class AxisParameterConfig:
    """Controller axis parameters that can be saved before UAPI is available."""

    axis: int = 0
    speed: float = 100.0
    units: float = 2097152.0
    accel: float = 3000.0
    decel: float = 3000.0
    fast_dec: float = 50000.0
    jerk: float = 100000.0
    fwd_in: float = -1.0
    rev_in: float = -1.0
    fe_limit: float = 10.0

    def validate(self) -> None:
        if not 0 <= self.axis <= 25:
            raise ValueError(f"Axis must be between 0 and 25 (got {self.axis}).")

        for field_name, value in asdict(self).items():
            if field_name == "axis":
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
                axis=int(data.get("axis", defaults.axis)),
                speed=float(data.get("speed", defaults.speed)),
                units=float(data.get("units", defaults.units)),
                accel=float(data.get("accel", defaults.accel)),
                decel=float(data.get("decel", defaults.decel)),
                fast_dec=float(data.get("fast_dec", defaults.fast_dec)),
                jerk=float(data.get("jerk", defaults.jerk)),
                fwd_in=float(data.get("fwd_in", defaults.fwd_in)),
                rev_in=float(data.get("rev_in", defaults.rev_in)),
                fe_limit=float(data.get("fe_limit", defaults.fe_limit)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid axis parameter value: {exc}") from exc

        config.validate()
        return config
