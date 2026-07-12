"""
Classical control-theory helpers for the Ziegler-Nichols tuning workflow.

Scope-capture loop analysis lives in the shared SignalMetrics engine
(``signal_metrics.py``) — this module keeps only the standalone utilities
used around a manual oscillation test:

  - :meth:`ClassicalTuner.detect_oscillation` — dominant frequency of a
    sustained-oscillation capture (source for the ZN ultimate period Tu).
  - :meth:`ClassicalTuner.bandwidth_calculate` — closed-loop bandwidth
    estimates from the drive's Pn gain values.
"""

from __future__ import annotations

import math

import numpy as np


class ClassicalTuner:
    """Standalone helpers for the manual ZN tuning workflow."""

    # --------------------------------------------------------- bandwidth calc
    @staticmethod
    def bandwidth_calculate(
        pn102: float, pn103: float, pn104: float
    ) -> dict[str, float]:
        """Estimate closed-loop bandwidths from gain parameters.

        pn102: speed loop gain (rad/s)
        pn103: speed loop integral time (×0.1 ms)
        pn104: position loop gain (1/s)
        """
        ti_s = pn103 * 0.1e-3
        speed_bw_hz = pn102 / (2 * math.pi)
        pos_bw_hz = pn104 / (2 * math.pi)
        integral_freq_hz = 1.0 / (2 * math.pi * ti_s) if ti_s > 0 else 0.0
        return {
            "speed_loop_bw_hz": round(speed_bw_hz, 1),
            "position_loop_bw_hz": round(pos_bw_hz, 1),
            "integral_freq_hz": round(integral_freq_hz, 1),
            "speed_to_position_ratio": (
                round(speed_bw_hz / pos_bw_hz, 1) if pos_bw_hz > 0 else float("inf")
            ),
        }

    # ------------------------------------------------------- oscillation det
    @staticmethod
    def detect_oscillation(
        time_arr: np.ndarray, signal: np.ndarray, min_freq: float = 5.0
    ) -> dict:
        """Detect dominant oscillation frequency via FFT.

        Returns dict with freq_hz, amplitude, and is_oscillating flag.
        """
        n = len(signal)
        if n < 64:
            return {"freq_hz": 0.0, "amplitude": 0.0, "is_oscillating": False}

        dt = float(np.median(np.diff(time_arr)))
        fs = 1.0 / dt if dt > 0 else 0.0
        if fs <= 0:
            return {"freq_hz": 0.0, "amplitude": 0.0, "is_oscillating": False}

        x = signal - np.mean(signal)
        window = np.hanning(n)
        spectrum = np.fft.rfft(x * window)
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        mag = np.abs(spectrum) * 2.0 / n / np.mean(window)

        valid = freqs >= min_freq
        if not np.any(valid):
            return {"freq_hz": 0.0, "amplitude": 0.0, "is_oscillating": False}

        mag_v = mag[valid]
        freqs_v = freqs[valid]
        noise_floor = float(np.median(mag_v))
        idx = int(np.argmax(mag_v))
        peak_freq = float(freqs_v[idx])
        peak_amp = float(mag_v[idx])
        is_osc = peak_amp > 5.0 * noise_floor

        return {
            "freq_hz": round(peak_freq, 1),
            "amplitude": round(peak_amp, 6),
            "noise_floor": round(noise_floor, 6),
            "is_oscillating": bool(is_osc),
        }
