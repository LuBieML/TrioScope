from PySide6.QtCore import QObject

# Importing the application entry point adds src/ to sys.path, matching the
# runtime import convention used by scope_app.py.
import scope_app  # noqa: F401
from ui.capture_controller import CaptureController


class FakeConnection:
    def __init__(self, positions):
        self._positions = iter(positions)

    def GetSystemParameter_SCOPE_POS(self):
        return next(self._positions)


class FakeScopeEngine:
    def __init__(self, positions):
        self.connection = FakeConnection(positions)
        self.arm_calls = 0
        self.trigger_calls = []
        self.is_capturing = False

    def arm_capture(self):
        self.arm_calls += 1

    def trigger_capture(self, auto_retrigger=False):
        self.trigger_calls.append(auto_retrigger)
        self.is_capturing = True


def make_controller(positions):
    window = QObject()
    window.scope_engine = FakeScopeEngine(positions)
    window.is_running = True
    window.trio_connected = True
    return CaptureController(window), window.scope_engine


def test_external_trigger_enables_auto_retrigger_for_continuous_mode():
    controller, engine = make_controller([0, 1])

    assert controller._arm_and_wait_for_external_trigger(auto_retrigger=True)

    assert engine.arm_calls == 1
    assert engine.trigger_calls == [True]
    assert engine.is_capturing


def test_external_trigger_remains_single_shot_by_default():
    controller, engine = make_controller([0, 1])

    assert controller._arm_and_wait_for_external_trigger()

    assert engine.arm_calls == 1
    assert engine.trigger_calls == []
    assert engine.is_capturing
