import numpy as np
import pytest

from src.scope.cursor_statistics import calculate_cursor_range_statistics
from src.ui.plot_renderer import PlotRenderer
from src.ui.main_window_actions import MainWindowActions


class _Trace:
    def is_fft(self):
        return False

    def get_display_name(self):
        return "CURRENT(0)"

    def get_color(self):
        return "#03DAC6"


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _CursorReadoutStub:
    _get_nearest_index = PlotRenderer._get_nearest_index
    _update_cursor_readout = PlotRenderer._update_cursor_readout

    def __init__(self):
        self._cursors_enabled = True
        self._y_cursors_enabled = False
        self.plot_mode = "time"
        self._cursor_pos = {"c1": 0.1, "c2": 0.3}
        self.accumulated_data = {
            "time": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            "params": {"CURRENT(0)": np.array([0.0, 10.0, 20.0, 30.0, 40.0])},
        }
        self.cursor_readout_label = _Label()

    def get_enabled_traces(self):
        return [_Trace()]


def test_cursor_range_counts_inclusive_samples_and_calculates_means():
    time_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    parameters = {
        "A(0)": np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
        "B(0)": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
    }

    result = calculate_cursor_range_statistics(
        time_values, parameters, cursor_1=0.1, cursor_2=0.3
    )

    assert result.sample_count == 3
    assert result.means == {"A(0)": 20.0, "B(0)": 3.0}


def test_cursor_range_is_order_independent_and_ignores_non_finite_values():
    result = calculate_cursor_range_statistics(
        [0.0, 0.1, 0.2, 0.3],
        {"FE(0)": [1.0, np.nan, 5.0, np.inf]},
        cursor_1=0.3,
        cursor_2=0.1,
    )

    assert result.sample_count == 3
    assert result.means["FE(0)"] == pytest.approx(5.0)


def test_cursor_range_outside_capture_has_no_samples():
    result = calculate_cursor_range_statistics(
        [0.0, 0.1, 0.2], {"A(0)": [1.0, 2.0, 3.0]}, 1.0, 2.0
    )

    assert result.sample_count == 0
    assert result.means["A(0)"] is None


def test_cursor_readout_displays_sample_count_and_trace_average():
    readout = _CursorReadoutStub()

    readout._update_cursor_readout()

    assert "n = 3 samples" in readout.cursor_readout_label.text
    assert "AVG CURRENT(0):</span> 20.0000" in readout.cursor_readout_label.text


class _YCursorReadoutStub:
    _update_cursor_readout = PlotRenderer._update_cursor_readout
    _y_trace_entries = PlotRenderer._y_trace_entries

    def __init__(self):
        self._cursors_enabled = False
        self._y_cursors_enabled = True
        self.plot_mode = "time"
        self.plot_items = {}
        self.accumulated_data = None
        self.cursor_readout_label = _Label()
        self.trace = _Trace()
        self.plot_items[id(self.trace)] = object()
        self._y_cursor_pos = {
            id(self.trace): {"y1": -100.0, "y2": -50.0}
        }

    def get_enabled_traces(self):
        return [self.trace]


def test_y_cursor_readout_displays_range_for_any_trace_type():
    readout = _YCursorReadoutStub()

    readout._update_cursor_readout()

    assert "CURRENT(0):</span><br>" in readout.cursor_readout_label.text
    assert "Y1" in readout.cursor_readout_label.text
    assert "-100.0000" in readout.cursor_readout_label.text
    assert "Y2</span> -50.0000" in readout.cursor_readout_label.text
    assert "ΔY</span> 50.0000" in readout.cursor_readout_label.text


class _TuningCursorProviderStub:
    _get_cursor_statistics_for_tuning = (
        MainWindowActions._get_cursor_statistics_for_tuning
    )

    def __init__(self):
        self._cursors_enabled = True
        self._cursor_pos = {"c1": 0.1, "c2": 0.3}
        self.accumulated_data = {
            "time": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            "params": {
                "DRIVE_CURRENT(2)": np.array([0.0, 100.0, 200.0, 300.0, 0.0]),
                "DRIVE_CURRENT(3)": np.array([0.0, 10.0, 20.0, 30.0, 0.0]),
            },
        }


def test_tuning_cursor_provider_returns_selected_axis_average():
    provider = _TuningCursorProviderStub()

    result = provider._get_cursor_statistics_for_tuning(3)

    assert result == {
        "source": "DRIVE_CURRENT(3)",
        "average": 20.0,
        "sample_count": 3,
    }


def test_tuning_cursor_provider_accepts_drive_scope_iq_actual():
    provider = _TuningCursorProviderStub()
    provider.accumulated_data["params"] = {
        "IQ (0x0F21)": np.array([0.0, 20.0, 40.0, 60.0, 0.0])
    }

    result = provider._get_cursor_statistics_for_tuning(0)

    assert result["source"] == "IQ (0x0F21)"
    assert result["average"] == pytest.approx(40.0)
