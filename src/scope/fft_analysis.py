"""Shared FFT helpers for scope plots, measurements, and reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np


MIN_FFT_SAMPLES: Final = 4
SAMPLE_PERIOD_RTOL: Final = 0.01


def estimate_sample_period(
    time_arr: np.ndarray,
    *,
    relative_tolerance: float = SAMPLE_PERIOD_RTOL,
    segment_breaks: Sequence[int] | None = None,
) -> float | None:
    """Return the median sample period when timestamps are finite and uniform.

    A small tolerance admits timestamp quantisation from CSV export while
    rejecting data that require resampling before a conventional FFT is valid.
    Differences crossing explicit capture boundaries are excluded so continuous
    scope chunks can share one rolling FFT sample grid.
    """
    times = np.asarray(time_arr, dtype=float)
    if times.size < 2 or not np.all(np.isfinite(times)):
        return None

    diffs = np.diff(times)
    if segment_breaks:
        keep = np.ones(len(diffs), dtype=bool)
        for boundary in segment_breaks:
            boundary = int(boundary)
            if 0 < boundary < len(times):
                keep[boundary - 1] = False
        diffs = diffs[keep]
    if diffs.size == 0:
        return None
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
        return None

    dt = float(np.median(diffs))
    if not np.isfinite(dt) or dt <= 0:
        return None

    tolerance = max(abs(dt) * relative_tolerance, np.finfo(float).eps * 8)
    if np.any(np.abs(diffs - dt) > tolerance):
        return None
    return dt


def hann_window(sample_count: int) -> tuple[np.ndarray, float]:
    """Return a Hann window and coherent gain denominator.

    NumPy's two-point Hann window is all zero, so use a rectangular fallback
    for degenerate sizes even though normal scope FFTs require four samples.
    """
    window = np.hanning(sample_count)
    window_sum = float(np.sum(window))
    if not np.isfinite(window_sum) or window_sum <= 0:
        window = np.ones(sample_count, dtype=float)
        window_sum = float(sample_count)
    return window, window_sum


def one_sided_amplitude(
    values: np.ndarray,
    window: np.ndarray | None = None,
    window_sum: float | None = None,
) -> np.ndarray | None:
    """Return a correctly scaled, DC-removed one-sided amplitude spectrum."""
    samples = np.asarray(values, dtype=float)
    n = samples.size
    if n < MIN_FFT_SAMPLES or not np.all(np.isfinite(samples)):
        return None

    if window is None or len(window) != n:
        window, window_sum = hann_window(n)
    elif window_sum is None:
        window_sum = float(np.sum(window))
    if not np.isfinite(window_sum) or window_sum <= 0:
        window, window_sum = hann_window(n)

    centered = samples - float(np.mean(samples))
    magnitude = np.abs(np.fft.rfft(centered * window)) * 2.0 / window_sum
    magnitude[0] = 0.0
    if n % 2 == 0:
        # DC and Nyquist have no negative-frequency partner in an rFFT.
        magnitude[-1] *= 0.5
    return magnitude


def amplitude_spectrum(
    time_arr: np.ndarray,
    values: np.ndarray,
    *,
    max_samples: int | None = None,
    segment_breaks: Sequence[int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute a validated Hann-windowed rolling one-sided amplitude spectrum."""
    n = min(len(time_arr), len(values))
    if n < MIN_FFT_SAMPLES:
        return None, None

    times = np.asarray(time_arr[:n], dtype=float)
    samples = np.asarray(values[:n], dtype=float)
    first_sample = 0
    if max_samples is not None and n > max_samples:
        first_sample = n - max_samples
        times = times[-max_samples:]
        samples = samples[-max_samples:]
        n = len(samples)

    relative_breaks = [
        int(boundary) - first_sample
        for boundary in segment_breaks or ()
        if first_sample < int(boundary) < first_sample + n
    ]
    dt = estimate_sample_period(times, segment_breaks=relative_breaks)
    if dt is None:
        return None, None
    magnitude = one_sided_amplitude(samples)
    if magnitude is None:
        return None, None
    return np.fft.rfftfreq(n, d=dt), magnitude
