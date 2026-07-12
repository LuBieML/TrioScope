"""UAPI transport for applying Trio controller axis parameters.

The dedicated setter names and value types come from
``Trio_UnifiedApi_CPP.pdf`` sections 6.3.3.545, 573, 601, 602, 619,
628, 658, 671 and 691.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

try:
    from ..models.axis_parameter_config import AxisParameterConfig
except ImportError:  # App runtime imports scope as a top-level package.
    from models.axis_parameter_config import AxisParameterConfig


@dataclass(frozen=True)
class AxisParameterSetter:
    field_name: str
    parameter_name: str
    method_name: str
    value_type: type


# UNITS is intentionally written first because SPEED/ACCEL/DECEL and related
# motion values are interpreted using the axis unit scaling.
AXIS_PARAMETER_SETTERS = (
    AxisParameterSetter("units", "UNITS", "SetAxisParameter_UNITS", float),
    AxisParameterSetter("speed", "SPEED", "SetAxisParameter_SPEED", float),
    AxisParameterSetter("accel", "ACCEL", "SetAxisParameter_ACCEL", float),
    AxisParameterSetter("decel", "DECEL", "SetAxisParameter_DECEL", float),
    AxisParameterSetter("fast_dec", "FASTDEC", "SetAxisParameter_FASTDEC", float),
    AxisParameterSetter("jerk", "JERK", "SetAxisParameter_JERK", float),
    AxisParameterSetter("fwd_in", "FWD_IN", "SetAxisParameter_FWD_IN", int),
    AxisParameterSetter("rev_in", "REV_IN", "SetAxisParameter_REV_IN", int),
    AxisParameterSetter("fe_limit", "FE_LIMIT", "SetAxisParameter_FE_LIMIT", float),
)


class AxisParameterWriteError(RuntimeError):
    """Adds the failed axis/parameter and partial-write count to UAPI errors."""

    def __init__(
        self,
        axis: Optional[int],
        parameter: str,
        completed: int,
        total: int,
        reason: str,
    ):
        self.axis = axis
        self.parameter = parameter
        self.completed = completed
        self.total = total
        location = f"axis {axis}, {parameter}" if axis is not None else parameter
        super().__init__(
            f"Failed to write {location} after {completed}/{total} values: {reason}"
        )


ProgressCallback = Callable[[int, int, int, str], None]


def write_axis_parameters(
    connection,
    configs: Iterable[AxisParameterConfig],
    connection_lock=None,
    progress_callback: Optional[ProgressCallback] = None,
) -> int:
    """Apply every configured value through the dedicated UAPI setters.

    The connection lock is acquired for one setter call at a time. This keeps
    hardware access serialized without starving TrioScope's watchdog.
    """

    items = list(configs)
    for config in items:
        config.validate()

    total = len(items) * len(AXIS_PARAMETER_SETTERS)
    if not items:
        return 0

    # Check SDK compatibility before touching the controller so an older
    # binding cannot leave an axis half configured merely due to a missing API.
    setters = {}
    for spec in AXIS_PARAMETER_SETTERS:
        setter = getattr(connection, spec.method_name, None)
        if not callable(setter):
            raise AxisParameterWriteError(
                None,
                spec.parameter_name,
                0,
                total,
                f"the connected UAPI binding has no {spec.method_name} method",
            )
        setters[spec.method_name] = setter

    completed = 0
    for config in items:
        for spec in AXIS_PARAMETER_SETTERS:
            value = spec.value_type(getattr(config, spec.field_name))
            try:
                if connection_lock is None:
                    setters[spec.method_name](config.axis, value)
                else:
                    with connection_lock:
                        setters[spec.method_name](config.axis, value)
            except Exception as exc:
                raise AxisParameterWriteError(
                    config.axis,
                    spec.parameter_name,
                    completed,
                    total,
                    str(exc) or type(exc).__name__,
                ) from exc

            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, config.axis, spec.parameter_name)

    return completed
