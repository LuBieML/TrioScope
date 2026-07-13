import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.scope.fft_analysis import (
    amplitude_spectrum,
    estimate_sample_period,
    hann_window,
)
from src.ui.plot_renderer import PlotRenderer


def test_even_length_nyquist_bin_is_not_doubled():
    sample_count = 1024
    time_arr = np.arange(sample_count, dtype=float) / 1000.0
    values = np.cos(np.pi * np.arange(sample_count, dtype=float))

    freqs, magnitude = amplitude_spectrum(time_arr, values)

    assert freqs is not None
    assert magnitude is not None
    np.testing.assert_allclose(freqs[-1], 500.0)
    np.testing.assert_allclose(magnitude[-1], 1.0, rtol=1e-6)


def test_odd_length_spectrum_preserves_exact_frequency_axis():
    sample_count = 1001
    sample_period = 0.001
    time_arr = np.arange(sample_count, dtype=float) * sample_period
    values = np.sin(2.0 * np.pi * 50.0 * time_arr)

    freqs, magnitude = amplitude_spectrum(time_arr, values)

    assert freqs is not None
    assert magnitude is not None
    np.testing.assert_allclose(
        freqs,
        np.fft.rfftfreq(sample_count, d=sample_period),
    )
    assert len(freqs) == len(magnitude)


def test_nonuniform_timestamps_are_rejected():
    time_arr = np.array([0.0, 0.001, 0.002, 0.004, 0.005])
    values = np.arange(len(time_arr), dtype=float)

    assert estimate_sample_period(time_arr) is None
    assert amplitude_spectrum(time_arr, values) == (None, None)


def test_marked_rollover_gap_keeps_rolling_fft_history():
    first_time = np.arange(10, dtype=float) * 0.001
    second_time = 0.020 + np.arange(10, dtype=float) * 0.001
    time_arr = np.concatenate((first_time, second_time))
    values = np.sin(2.0 * np.pi * 100.0 * time_arr)

    assert amplitude_spectrum(time_arr, values) == (None, None)

    freqs, magnitude = amplitude_spectrum(
        time_arr,
        values,
        segment_breaks=[10],
    )

    assert freqs is not None
    assert magnitude is not None
    np.testing.assert_allclose(freqs, np.fft.rfftfreq(20, d=0.001))
    assert int(np.argmax(magnitude)) == 2


def test_marked_rollover_gap_is_ignored_only_for_sample_period_estimation():
    time_arr = np.concatenate((
        np.arange(10, dtype=float) * 0.001,
        0.020 + np.arange(5, dtype=float) * 0.001,
    ))

    assert estimate_sample_period(time_arr) is None
    np.testing.assert_allclose(
        estimate_sample_period(time_arr, segment_breaks=[10]),
        0.001,
    )


def test_small_timestamp_quantisation_is_accepted():
    time_arr = np.array([0.0, 0.000187, 0.000375, 0.000562, 0.000750])

    np.testing.assert_allclose(
        estimate_sample_period(time_arr),
        0.0001875,
        rtol=0.01,
    )


def test_two_point_hann_falls_back_but_spectrum_requires_four_samples():
    window, window_sum = hann_window(2)

    np.testing.assert_array_equal(window, np.ones(2))
    assert window_sum == 2.0
    assert amplitude_spectrum(
        np.array([0.0, 0.001]),
        np.array([1.0, -1.0]),
    ) == (None, None)


def test_max_samples_uses_a_bounded_tail_with_matching_frequency_bins():
    time_arr = np.arange(100, dtype=float) * 0.001
    values = np.sin(2.0 * np.pi * 20.0 * time_arr)

    freqs, magnitude = amplitude_spectrum(time_arr, values, max_samples=17)

    assert freqs is not None
    assert magnitude is not None
    np.testing.assert_allclose(freqs, np.fft.rfftfreq(17, d=0.001))
    assert len(magnitude) == 9


def test_plot_renderer_reset_invalidates_all_fft_state():
    renderer = PlotRenderer(QObject())
    renderer._fft_cache = {1: {"magnitude": np.ones(3)}}
    renderer._fft_window_cache = (4, np.ones(4))
    renderer._fft_peak_cache = {1: (50.0, 1.0)}
    renderer._last_freqs = np.arange(3)
    renderer._fft_dirty = False

    renderer._reset_fft_state()

    assert renderer._fft_cache == {}
    assert renderer._fft_window_cache == (0, None)
    assert renderer._fft_peak_cache == {}
    assert renderer._last_freqs is None
    assert renderer._fft_dirty is True


def test_companion_fft_keeps_odd_length_frequency_axis():
    renderer = PlotRenderer(QObject())
    sample_count = 1001
    time_arr = np.arange(sample_count, dtype=float) * 0.001
    data = {
        "time": time_arr,
        "params": {"FE(0)": np.sin(2.0 * np.pi * 50.0 * time_arr)},
    }

    class Trace:
        @staticmethod
        def get_display_name():
            return "FE(0)"

    freqs, magnitude = renderer._compute_fft_payload_for_trace(Trace(), data)

    assert freqs is not None
    assert magnitude is not None
    np.testing.assert_allclose(freqs, np.fft.rfftfreq(sample_count, d=0.001))
