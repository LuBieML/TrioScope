"""Numeric measurements for TrioScope captures.

The functions in this module are UI-independent so the same calculations can
be used by dock widgets, reports, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CaptureSummary:
    samples: int
    duration_s: float | None
    sample_rate_hz: float | None
    dt_ms: float | None
    nyquist_hz: float | None
    segment_count: int


@dataclass(frozen=True)
class TraceMeasurement:
    name: str
    samples: int
    latest: float | None
    minimum: float | None
    maximum: float | None
    mean: float | None
    rms: float | None
    peak_to_peak: float | None
    std: float | None
    slope_per_s: float | None
    dominant_freq_hz: float | None
    dominant_magnitude: float | None


def _positive_dt(time_arr: np.ndarray) -> float | None:
    if len(time_arr) < 2:
        return None
    diffs = np.diff(time_arr.astype(float, copy=False))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def compute_capture_summary(
    time_arr: np.ndarray,
    segment_breaks: list[int] | tuple[int, ...] | None = None,
) -> CaptureSummary:
    """Return timing-level metrics for a capture or selected capture window."""
    samples = int(len(time_arr))
    dt = _positive_dt(time_arr)
    duration = None
    sample_rate = None
    nyquist = None

    if samples >= 2:
        t0 = float(time_arr[0])
        t1 = float(time_arr[-1])
        if np.isfinite(t0) and np.isfinite(t1):
            duration = max(0.0, t1 - t0)
    if dt and dt > 0:
        sample_rate = 1.0 / dt
        nyquist = sample_rate / 2.0

    return CaptureSummary(
        samples=samples,
        duration_s=duration,
        sample_rate_hz=sample_rate,
        dt_ms=dt * 1000.0 if dt else None,
        nyquist_hz=nyquist,
        segment_count=len(segment_breaks or []),
    )


def compute_trace_measurement(
    name: str,
    time_arr: np.ndarray,
    values: np.ndarray,
    *,
    fft_max_samples: int = 16384,
) -> TraceMeasurement:
    """Return scalar measurements for one trace."""
    n = min(len(time_arr), len(values))
    if n == 0:
        return TraceMeasurement(name, 0, *([None] * 10))

    t = time_arr[:n].astype(float, copy=False)
    y = values[:n].astype(float, copy=False)
    finite = np.isfinite(t) & np.isfinite(y)
    if not np.any(finite):
        return TraceMeasurement(name, 0, *([None] * 10))

    t = t[finite]
    y = y[finite]
    samples = int(y.size)

    latest = float(y[-1])
    minimum = float(np.min(y))
    maximum = float(np.max(y))
    mean = float(np.mean(y))
    rms = float(np.sqrt(np.mean(y * y)))
    peak_to_peak = maximum - minimum
    std = float(np.std(y))

    slope = None
    if samples >= 2:
        duration = float(t[-1] - t[0])
        if duration > 0:
            slope = float((y[-1] - y[0]) / duration)

    peak_freq, peak_mag = _dominant_frequency(t, y, fft_max_samples)

    return TraceMeasurement(
        name=name,
        samples=samples,
        latest=latest,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        rms=rms,
        peak_to_peak=peak_to_peak,
        std=std,
        slope_per_s=slope,
        dominant_freq_hz=peak_freq,
        dominant_magnitude=peak_mag,
    )


def compute_trace_measurements(
    time_arr: np.ndarray,
    params: Mapping[str, np.ndarray],
    *,
    fft_max_samples: int = 16384,
) -> list[TraceMeasurement]:
    """Compute trace measurements in parameter insertion order."""
    return [
        compute_trace_measurement(
            name,
            time_arr,
            values,
            fft_max_samples=fft_max_samples,
        )
        for name, values in params.items()
    ]


def _dominant_frequency(
    time_arr: np.ndarray,
    values: np.ndarray,
    fft_max_samples: int,
) -> tuple[float | None, float | None]:
    n = min(len(time_arr), len(values))
    if n < 4:
        return None, None

    if n > fft_max_samples:
        time_arr = time_arr[-fft_max_samples:]
        values = values[-fft_max_samples:]
        n = len(values)

    dt = _positive_dt(time_arr)
    if not dt or dt <= 0:
        return None, None

    y = values.astype(float, copy=False)
    if not np.all(np.isfinite(y)):
        finite = np.isfinite(y)
        y = y[finite]
        n = len(y)
        if n < 4:
            return None, None

    centered = y - float(np.mean(y))
    if float(np.max(np.abs(centered))) <= 0.0:
        return None, None

    window = np.hanning(n)
    window_sum = float(np.sum(window))
    if window_sum <= 0:
        window = np.ones(n)
        window_sum = float(n)

    fft_vals = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(n, d=dt)
    magnitude = np.abs(fft_vals) * 2.0 / window_sum
    if magnitude.size <= 1:
        return None, None

    magnitude[0] = 0.0
    idx = int(np.argmax(magnitude))
    peak_mag = float(magnitude[idx])
    if peak_mag <= 0.0:
        return None, None
    return float(freqs[idx]), peak_mag
