from src.models.motion_axis_command import MotionAxisCommand
from src.scope.axis_motion import (
    CANCEL_ACTIVE_AND_BUFFERED,
    axes_are_idle,
    cancel_axis_moves,
    execute_relative_moves,
    set_axes_enabled,
)


class _RecordingMotionConnection:
    def __init__(self):
        self.calls = []
        self.idle = {}

    def SetSystemParameter_WDOG(self, enabled):
        self.calls.append(("WDOG", enabled))

    def SetAxisParameter_SERVO(self, axis, enabled):
        self.calls.append(("SERVO", axis, enabled))

    def SetAxisParameter_AXIS_ENABLE(self, axis, enabled):
        self.calls.append(("AXIS_ENABLE", axis, enabled))

    def SetAxisParameter_SPEED(self, axis, value):
        self.calls.append(("SPEED", axis, value))

    def SetAxisParameter_ACCEL(self, axis, value):
        self.calls.append(("ACCEL", axis, value))

    def SetAxisParameter_DECEL(self, axis, value):
        self.calls.append(("DECEL", axis, value))

    def MoveRel(self, distance, axis):
        self.calls.append(("MOVE", distance, axis))

    def Cancel(self, mode, axis):
        self.calls.append(("CANCEL", mode, axis))

    def GetAxisParameter_IDLE(self, axis):
        self.calls.append(("IDLE", axis))
        return self.idle.get(axis, False)


class _TrackingLock:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, exc_type, exc, traceback):
        return False


COMMANDS = [
    MotionAxisCommand(axis=1, speed=120.0, distance=25.0),
    MotionAxisCommand(axis=4, speed=60.0, distance=-8.0),
]


def test_enable_and_disable_use_safe_uapi_order():
    connection = _RecordingMotionConnection()
    lock = _TrackingLock()

    assert set_axes_enabled(connection, COMMANDS, True, lock) == 2
    assert connection.calls == [
        ("WDOG", True),
        ("SERVO", 1, True),
        ("AXIS_ENABLE", 1, True),
        ("SERVO", 4, True),
        ("AXIS_ENABLE", 4, True),
    ]

    connection.calls.clear()
    assert set_axes_enabled(connection, COMMANDS, False, lock) == 2
    assert connection.calls == [
        ("CANCEL", CANCEL_ACTIVE_AND_BUFFERED, 1),
        ("AXIS_ENABLE", 1, False),
        ("SERVO", 1, False),
        ("CANCEL", CANCEL_ACTIVE_AND_BUFFERED, 4),
        ("AXIS_ENABLE", 4, False),
        ("SERVO", 4, False),
        ("WDOG", False),
    ]
    assert lock.entries == 12


def test_move_sets_all_speeds_before_executing_relative_moves():
    connection = _RecordingMotionConnection()

    assert execute_relative_moves(connection, COMMANDS) == 2
    assert connection.calls == [
        ("SPEED", 1, 120.0),
        ("SPEED", 4, 60.0),
        ("MOVE", 25.0, 1),
        ("MOVE", -8.0, 4),
    ]


def test_tuning_move_sets_speed_accel_and_decel_before_move():
    connection = _RecordingMotionConnection()
    command = MotionAxisCommand(
        axis=3,
        speed=75.0,
        distance=-12.0,
        acceleration=600.0,
    )

    assert execute_relative_moves(connection, [command]) == 1
    assert connection.calls == [
        ("SPEED", 3, 75.0),
        ("ACCEL", 3, 600.0),
        ("DECEL", 3, 600.0),
        ("MOVE", -12.0, 3),
    ]


def test_stop_and_idle_are_scoped_to_involved_axes():
    connection = _RecordingMotionConnection()
    connection.idle = {1: True, 4: True}

    assert cancel_axis_moves(connection, [1, 4]) == 2
    assert axes_are_idle(connection, [1, 4])
    assert connection.calls == [
        ("CANCEL", CANCEL_ACTIVE_AND_BUFFERED, 1),
        ("CANCEL", CANCEL_ACTIVE_AND_BUFFERED, 4),
        ("IDLE", 1),
        ("IDLE", 4),
    ]
