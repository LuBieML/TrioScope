import pytest
from PySide6.QtWidgets import QApplication

from src.scope.parameters import MAX_CONTROLLER_AXIS
from src.ui.trace_control import TraceControl


@pytest.fixture
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_trace_control_initial_axis_range(qt_app):
    trace = TraceControl(0)
    assert trace.axis_spin.minimum() == 0
    assert trace.axis_spin.maximum() == MAX_CONTROLLER_AXIS
    assert trace.axis_spin.maximum() >= 255
    assert trace.axis_label.text() == "Axis"


def test_trace_control_allows_axis_higher_than_15(qt_app):
    trace = TraceControl(0)
    for axis in (16, 24, 32, 63, 127, 255):
        trace.axis_spin.setValue(axis)
        assert trace.axis_spin.value() == axis


def test_trace_control_axis_buttons_step_past_15(qt_app):
    trace = TraceControl(0)
    trace.axis_spin.setValue(15)
    assert trace.axis_spin.value() == 15

    trace.btn_ax_up.click()
    assert trace.axis_spin.value() == 16

    trace.btn_ax_up.click()
    assert trace.axis_spin.value() == 17

    trace.btn_ax_down.click()
    assert trace.axis_spin.value() == 16


def test_trace_control_param_change_preserves_extended_axis_range(qt_app):
    trace = TraceControl(0)

    # Change to a channel parameter
    trace.param_combo.setCurrentText("AIN")
    assert trace.is_channel_parameter()
    assert trace.axis_label.text() == "Ch"
    assert trace.axis_spin.maximum() == 1024

    # Change back to an axis parameter
    trace.param_combo.setCurrentText("MPOS")
    assert not trace.is_channel_parameter()
    assert trace.axis_label.text() == "Axis"
    assert trace.axis_spin.maximum() == MAX_CONTROLLER_AXIS

    # Verify we can set axis higher than 15 after parameter change
    trace.axis_spin.setValue(32)
    assert trace.axis_spin.value() == 32


def test_trace_control_generates_correct_parameter_string_for_high_axis(qt_app):
    trace = TraceControl(0)
    trace.param_combo.setCurrentText("MPOS")
    trace.axis_spin.setValue(16)
    assert trace.get_parameter_string() == "MPOS AXIS(16)"
    assert trace.get_display_name() == "MPOS(16)"

    trace.param_combo.setCurrentText("FE")
    trace.axis_spin.setValue(64)
    assert trace.get_parameter_string() == "FE AXIS(64)"
    assert trace.get_display_name() == "FE(64)"


def test_main_window_drive_axis_spin_allows_high_axis(qt_app):
    from src.ui.main_window import ParameterScopeOscilloscope
    win = ParameterScopeOscilloscope()
    assert win.drv_axis_spin.minimum() == 0
    assert win.drv_axis_spin.maximum() == MAX_CONTROLLER_AXIS
    assert win.drv_axis_spin.maximum() >= 255
    win.drv_axis_spin.setValue(16)
    assert win.drv_axis_spin.value() == 16
    win.close()

