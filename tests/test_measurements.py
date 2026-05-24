import unittest

import numpy as np

from src.scope.measurements import (
    compute_capture_summary,
    compute_trace_measurement,
)


class MeasurementTests(unittest.TestCase):
    def test_capture_summary_reports_timing(self):
        time_arr = np.array([0.0, 0.001, 0.002, 0.003])

        summary = compute_capture_summary(time_arr, segment_breaks=[2])

        self.assertEqual(summary.samples, 4)
        self.assertAlmostEqual(summary.duration_s, 0.003)
        self.assertAlmostEqual(summary.dt_ms, 1.0)
        self.assertAlmostEqual(summary.sample_rate_hz, 1000.0)
        self.assertAlmostEqual(summary.nyquist_hz, 500.0)
        self.assertEqual(summary.segment_count, 1)

    def test_trace_measurement_reports_scalar_stats(self):
        time_arr = np.array([0.0, 0.5, 1.0])
        values = np.array([1.0, 3.0, 5.0])

        measurement = compute_trace_measurement("MPOS(0)", time_arr, values)

        self.assertEqual(measurement.name, "MPOS(0)")
        self.assertEqual(measurement.samples, 3)
        self.assertEqual(measurement.latest, 5.0)
        self.assertEqual(measurement.minimum, 1.0)
        self.assertEqual(measurement.maximum, 5.0)
        self.assertEqual(measurement.mean, 3.0)
        self.assertAlmostEqual(measurement.rms, np.sqrt(35.0 / 3.0))
        self.assertEqual(measurement.peak_to_peak, 4.0)
        self.assertAlmostEqual(measurement.slope_per_s, 4.0)

    def test_trace_measurement_detects_dominant_frequency(self):
        fs = 1000.0
        time_arr = np.arange(0.0, 1.0, 1.0 / fs)
        values = 2.0 * np.sin(2.0 * np.pi * 50.0 * time_arr)

        measurement = compute_trace_measurement("FE(0)", time_arr, values)

        self.assertAlmostEqual(measurement.dominant_freq_hz, 50.0, delta=1.0)
        self.assertGreater(measurement.dominant_magnitude, 1.0)

    def test_constant_trace_has_no_dominant_frequency(self):
        time_arr = np.arange(10, dtype=float) * 0.001
        values = np.ones(10)

        measurement = compute_trace_measurement("DAC_OUT(0)", time_arr, values)

        self.assertIsNone(measurement.dominant_freq_hz)
        self.assertIsNone(measurement.dominant_magnitude)


if __name__ == "__main__":
    unittest.main()
