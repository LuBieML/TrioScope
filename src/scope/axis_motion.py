"""Qt-free Trio UAPI operations used by the Axis Motion window."""

from contextlib import nullcontext
from typing import Iterable, List, Optional

try:
    from ..models.motion_axis_command import MotionAxisCommand
except ImportError:  # App runtime imports scope as a top-level package.
    from models.motion_axis_command import MotionAxisCommand


CANCEL_ACTIVE_AND_BUFFERED = 2


class AxisMotionError(RuntimeError):
    """A UAPI motion operation failed with axis/method context."""

    def __init__(
        self,
        operation: str,
        method_name: str,
        cause: Exception,
        axis: Optional[int] = None,
    ) -> None:
        self.operation = operation
        self.method_name = method_name
        self.axis = axis
        self.cause = cause
        location = f" for axis {axis}" if axis is not None else ""
        super().__init__(
            f"{operation} failed{location} while calling {method_name}: {cause}"
        )


def _validated_axes(commands: Iterable[MotionAxisCommand]) -> List[int]:
    command_list = list(commands)
    for command in command_list:
        command.validate()
    axes = [command.axis for command in command_list]
    if not axes:
        raise ValueError("At least one axis is required.")
    if len(axes) != len(set(axes)):
        raise ValueError("Each axis can only appear once in a move scheme.")
    return axes


def _require_methods(connection, method_names: Iterable[str]) -> None:
    for method_name in method_names:
        method = getattr(connection, method_name, None)
        if not callable(method):
            raise AxisMotionError(
                "UAPI preflight",
                method_name,
                AttributeError(f"Connected SDK has no callable {method_name}"),
            )


def _call(
    connection,
    method_name: str,
    *args,
    operation: str,
    axis: Optional[int] = None,
    connection_lock=None,
):
    try:
        with connection_lock if connection_lock is not None else nullcontext():
            return getattr(connection, method_name)(*args)
    except Exception as exc:
        raise AxisMotionError(operation, method_name, exc, axis=axis) from exc


def set_axes_enabled(
    connection,
    commands: Iterable[MotionAxisCommand],
    enabled: bool,
    connection_lock=None,
) -> int:
    """Set controller WDOG, SERVO and AXIS_ENABLE for all involved axes.

    Enable order is WDOG -> SERVO -> AXIS_ENABLE. Disable is deliberately the
    reverse safety path: cancel motion, clear AXIS_ENABLE, clear SERVO, then
    clear the controller-wide WDOG output.
    """

    axes = _validated_axes(commands)
    enable_methods = (
        "SetSystemParameter_WDOG",
        "SetAxisParameter_SERVO",
        "SetAxisParameter_AXIS_ENABLE",
    )
    disable_methods = enable_methods + ("Cancel",)
    _require_methods(connection, enable_methods if enabled else disable_methods)

    if enabled:
        touched_axes: List[int] = []
        try:
            _call(
                connection,
                "SetSystemParameter_WDOG",
                True,
                operation="Enable axes",
                connection_lock=connection_lock,
            )
            for axis in axes:
                _call(
                    connection,
                    "SetAxisParameter_SERVO",
                    axis,
                    True,
                    operation="Enable axes",
                    axis=axis,
                    connection_lock=connection_lock,
                )
                touched_axes.append(axis)
                _call(
                    connection,
                    "SetAxisParameter_AXIS_ENABLE",
                    axis,
                    True,
                    operation="Enable axes",
                    axis=axis,
                    connection_lock=connection_lock,
                )
        except AxisMotionError:
            _best_effort_disable(connection, touched_axes, connection_lock)
            raise
        return len(axes)

    for axis in axes:
        _call(
            connection,
            "Cancel",
            CANCEL_ACTIVE_AND_BUFFERED,
            axis,
            operation="Disable axes",
            axis=axis,
            connection_lock=connection_lock,
        )
        _call(
            connection,
            "SetAxisParameter_AXIS_ENABLE",
            axis,
            False,
            operation="Disable axes",
            axis=axis,
            connection_lock=connection_lock,
        )
        _call(
            connection,
            "SetAxisParameter_SERVO",
            axis,
            False,
            operation="Disable axes",
            axis=axis,
            connection_lock=connection_lock,
        )
    _call(
        connection,
        "SetSystemParameter_WDOG",
        False,
        operation="Disable axes",
        connection_lock=connection_lock,
    )
    return len(axes)


def execute_relative_moves(
    connection,
    commands: Iterable[MotionAxisCommand],
    connection_lock=None,
) -> int:
    """Prepare every motion profile, then issue the UAPI ``MoveRel`` calls.

    Tuning-workspace moves carry an explicit acceleration.  For those moves
    ACCEL and DECEL are both set immediately before MOVE so the displayed
    profile is the profile the controller actually executes.  Older callers
    may omit acceleration and retain the historical SPEED-only behaviour.
    """

    command_list = list(commands)
    _validated_axes(command_list)
    required_methods = ["SetAxisParameter_SPEED", "MoveRel"]
    if any(command.acceleration is not None for command in command_list):
        required_methods.extend(("SetAxisParameter_ACCEL", "SetAxisParameter_DECEL"))
    _require_methods(connection, required_methods)

    # Configure every profile before starting any axis, minimizing skew
    # between subsequent controller-side MOVE commands.
    for command in command_list:
        _call(
            connection,
            "SetAxisParameter_SPEED",
            command.axis,
            float(command.speed),
            operation="Start move",
            axis=command.axis,
            connection_lock=connection_lock,
        )
        if command.acceleration is not None:
            for parameter, method_name in (
                ("ACCEL", "SetAxisParameter_ACCEL"),
                ("DECEL", "SetAxisParameter_DECEL"),
            ):
                _call(
                    connection,
                    method_name,
                    command.axis,
                    float(command.acceleration),
                    operation=f"Set move {parameter}",
                    axis=command.axis,
                    connection_lock=connection_lock,
                )
    for command in command_list:
        _call(
            connection,
            "MoveRel",
            float(command.distance),
            command.axis,
            operation="Start move",
            axis=command.axis,
            connection_lock=connection_lock,
        )
    return len(command_list)


def cancel_axis_moves(
    connection,
    axes: Iterable[int],
    connection_lock=None,
) -> int:
    """Cancel active and buffered MOVE commands for only the involved axes."""

    axis_list = list(axes)
    if not axis_list:
        return 0
    _require_methods(connection, ("Cancel",))
    for axis in axis_list:
        _call(
            connection,
            "Cancel",
            CANCEL_ACTIVE_AND_BUFFERED,
            axis,
            operation="Stop move",
            axis=axis,
            connection_lock=connection_lock,
        )
    return len(axis_list)


def axes_are_idle(connection, axes: Iterable[int], connection_lock=None) -> bool:
    """Return whether all involved axes report the UAPI IDLE parameter."""

    axis_list = list(axes)
    _require_methods(connection, ("GetAxisParameter_IDLE",))
    return all(
        bool(
            _call(
                connection,
                "GetAxisParameter_IDLE",
                axis,
                operation="Monitor move",
                axis=axis,
                connection_lock=connection_lock,
            )
        )
        for axis in axis_list
    )


def _best_effort_disable(connection, axes: Iterable[int], connection_lock=None) -> None:
    """Make a failed enable as safe as possible without masking its error."""

    for axis in reversed(list(axes)):
        for method_name, value in (
            ("SetAxisParameter_AXIS_ENABLE", False),
            ("SetAxisParameter_SERVO", False),
        ):
            try:
                with connection_lock if connection_lock is not None else nullcontext():
                    getattr(connection, method_name)(axis, value)
            except Exception:
                pass
    try:
        with connection_lock if connection_lock is not None else nullcontext():
            connection.SetSystemParameter_WDOG(False)
    except Exception:
        pass
