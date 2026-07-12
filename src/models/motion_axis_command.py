"""Data model for one axis in a relative move scheme."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionAxisCommand:
    """One relative axis move, expressed in configured controller units."""

    axis: int
    speed: float = 100.0
    distance: float = 10.0

    def validate(self) -> None:
        if not 0 <= self.axis <= 25:
            raise ValueError("Axis must be between 0 and 25.")
        if self.speed <= 0:
            raise ValueError("Speed must be greater than zero.")

