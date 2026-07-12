import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import QApplication, QFrame, QWidget

from src.ui.motion_controller import MotionController
from src.ui.motion_window import AxisMotionWindow, MotionAxisCommand


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_motion_window_adds_unique_axes_and_preserves_values(qt_app):
    window = AxisMotionWindow()
    window.set_commands(
        [
            MotionAxisCommand(axis=2, speed=125.5, distance=80.0),
            MotionAxisCommand(axis=7, speed=30.0, distance=-12.25),
        ]
    )

    assert window.commands() == [
        MotionAxisCommand(axis=2, speed=125.5, distance=80.0),
        MotionAxisCommand(axis=7, speed=30.0, distance=12.25),
    ]
    assert len(window.rows_container.findChildren(QFrame, "motionAxisRow")) == 2
    assert window.add_axis()
    assert [command.axis for command in window.commands()] == [2, 7, 0]
    assert not window.add_axis(MotionAxisCommand(axis=7))


def test_motion_window_remove_and_duplicate_validation(qt_app):
    window = AxisMotionWindow()
    window.set_commands([MotionAxisCommand(axis=1), MotionAxisCommand(axis=4)])

    assert window.remove_axis(1)
    assert not window.remove_axis(9)
    assert [command.axis for command in window.commands()] == [4]

    with pytest.raises(ValueError, match="only appear once"):
        window.set_commands([MotionAxisCommand(axis=3), MotionAxisCommand(axis=3)])


def test_start_and_stop_emit_future_uapi_hooks(qt_app):
    window = AxisMotionWindow()
    window.set_commands(
        [
            MotionAxisCommand(axis=0, speed=100.0, distance=25.0),
            MotionAxisCommand(axis=3, speed=60.0, distance=5.0),
        ]
    )
    starts = []
    stops = []
    enables = []
    window.enableRequested.connect(lambda enabled, commands: enables.append((enabled, commands)))
    window.startRequested.connect(starts.append)
    window.stopRequested.connect(lambda: stops.append(True))

    window.set_connection_available(True)
    assert window.btn_enable.isEnabled()
    assert all(not row.negative_button.isEnabled() for row in window._rows)
    assert all(not row.positive_button.isEnabled() for row in window._rows)

    window.btn_enable.click()
    assert enables == [(True, window.commands())]
    assert all(not row.speed_edit.isEnabled() for row in window._rows)
    assert all(not row.distance_edit.isEnabled() for row in window._rows)
    window.complete_enable(True)
    assert window.btn_enable.isChecked()
    assert all(row.negative_button.isEnabled() for row in window._rows)
    assert all(row.positive_button.isEnabled() for row in window._rows)

    axis_0_row, axis_3_row = window._rows
    assert not axis_3_row.axis_combo.isEnabled()
    assert not axis_3_row.remove_button.isEnabled()
    assert axis_3_row.speed_edit.isEnabled()
    assert axis_3_row.distance_edit.isEnabled()
    axis_3_row.speed_edit.setValue(75.0)
    axis_3_row.distance_edit.setValue(7.5)
    axis_3_row.negative_button.click()

    assert starts == [[MotionAxisCommand(axis=3, speed=75.0, distance=-7.5)]]
    assert axis_0_row.negative_button.isEnabled()
    assert axis_0_row.positive_button.isEnabled()
    assert not axis_3_row.negative_button.isEnabled()
    assert not axis_3_row.positive_button.isEnabled()
    assert window.btn_stop.isEnabled()

    axis_0_row.positive_button.click()
    assert starts[-1] == [MotionAxisCommand(axis=0, speed=100.0, distance=25.0)]
    assert not axis_0_row.negative_button.isEnabled()
    assert not axis_0_row.positive_button.isEnabled()

    window.btn_stop.click()

    assert stops == [True]
    window.complete_stop()
    assert all(row.negative_button.isEnabled() for row in window._rows)
    assert all(row.positive_button.isEnabled() for row in window._rows)
    assert not window.btn_stop.isEnabled()

    window.btn_enable.click()
    assert enables[-1] == (False, window.commands())
    window.complete_enable(False)
    assert not window.btn_enable.isChecked()
    assert all(not row.negative_button.isEnabled() for row in window._rows)
    assert all(not row.positive_button.isEnabled() for row in window._rows)


def test_motion_command_rejects_invalid_axis_and_speed():
    with pytest.raises(ValueError, match="between 0 and 25"):
        MotionAxisCommand(axis=26).validate()
    with pytest.raises(ValueError, match="greater than zero"):
        MotionAxisCommand(axis=0, speed=0).validate()


class _AsyncMotionConnection:
    def __init__(self):
        self.calls = []

    def SetSystemParameter_WDOG(self, enabled):
        self.calls.append(("WDOG", enabled))

    def SetAxisParameter_SERVO(self, axis, enabled):
        self.calls.append(("SERVO", axis, enabled))

    def SetAxisParameter_AXIS_ENABLE(self, axis, enabled):
        self.calls.append(("AXIS_ENABLE", axis, enabled))

    def SetAxisParameter_SPEED(self, axis, speed):
        self.calls.append(("SPEED", axis, speed))

    def MoveRel(self, distance, axis):
        self.calls.append(("MOVE", distance, axis))

    def Cancel(self, mode, axis):
        self.calls.append(("CANCEL", mode, axis))

    def GetAxisParameter_IDLE(self, axis):
        self.calls.append(("IDLE", axis))
        return True


class _BlockingEnableConnection(_AsyncMotionConnection):
    def __init__(self):
        super().__init__()
        self.enable_started = threading.Event()
        self.release_enable = threading.Event()

    def SetSystemParameter_WDOG(self, enabled):
        super().SetSystemParameter_WDOG(enabled)
        if enabled:
            self.enable_started.set()
            self.release_enable.wait(timeout=2.0)


def _wait_for(qt_app, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_action_controller_runs_enable_move_and_disable_uapi(qt_app):
    host = QWidget()
    host.trio_connected = True
    host.trio_connection = _AsyncMotionConnection()
    host._conn_lock = threading.Lock()
    host._motion_window = AxisMotionWindow(host)
    host._motion_window.set_commands(
        [
            MotionAxisCommand(axis=0, speed=100.0, distance=10.0),
            MotionAxisCommand(axis=2, speed=45.0, distance=6.0),
        ]
    )
    controller = MotionController(host)

    host._motion_window.enableRequested.connect(controller._on_motion_enable_requested)
    host._motion_window.startRequested.connect(controller._on_motion_start_requested)
    host._motion_window.stopRequested.connect(controller._on_motion_stop_requested)
    host._motion_window.set_connection_available(True)

    host._motion_window.btn_enable.click()
    axis_0_row, axis_2_row = host._motion_window._rows
    assert _wait_for(qt_app, lambda: axis_2_row.negative_button.isEnabled())

    axis_2_row.negative_button.click()
    assert _wait_for(
        qt_app,
        lambda: axis_2_row.negative_button.isEnabled()
        and ("MOVE", -6.0, 2) in host.trio_connection.calls,
    )
    assert ("MOVE", 10.0, 0) not in host.trio_connection.calls

    axis_0_row.positive_button.click()
    assert _wait_for(
        qt_app,
        lambda: axis_0_row.positive_button.isEnabled()
        and ("MOVE", 10.0, 0) in host.trio_connection.calls,
    )

    host._motion_window.btn_enable.click()
    assert _wait_for(qt_app, lambda: not host._motion_window.btn_enable.isChecked())
    assert host.trio_connection.calls[-1] == ("WDOG", False)


def test_closing_during_enable_finishes_with_hardware_disabled(qt_app):
    host = QWidget()
    host.trio_connected = True
    host.trio_connection = _BlockingEnableConnection()
    host._conn_lock = threading.Lock()
    host._motion_window = AxisMotionWindow(host)
    controller = MotionController(host)
    host._motion_window.enableRequested.connect(controller._on_motion_enable_requested)
    host._motion_window.set_connection_available(True)

    host._motion_window.show()
    host._motion_window.btn_enable.click()
    assert host.trio_connection.enable_started.wait(timeout=1.0)

    host._motion_window.close()
    host.trio_connection.release_enable.set()

    assert _wait_for(
        qt_app,
        lambda: not host._motion_enabled_commands
        and host.trio_connection.calls[-1:] == [("WDOG", False)],
    )
