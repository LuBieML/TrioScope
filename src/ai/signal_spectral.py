"""Spectral analysis of cruise-phase signals.

FFTs run on the longest *contiguous* cruise run — never on a concatenation
of disjoint segments, whose splice discontinuities leak broadband energy
and fake or bury peaks.
"""

from __future__ import annotations

import numpy as np

from .signal_constants import (
    NOISE_FLOOR_SIGMA, MIN_OSCILLATION_HZ, MIN_CRUISE_DURATION_S,
    MIN_CYCLES_FOR_PEAK,
)
from .signal_phases import contiguous_runs


def _longest_run(mask: np.ndarray,
                 bounds: list[tuple[int, int]]) -> tuple[int, int] | None:
    runs = contiguous_runs(mask, bounds)
    if not runs:
        return None
    return max(runs, key=lambda r: r[1] - r[0])


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove mean + linear trend (cruise FE often carries a slope)."""
    idx = np.arange(len(x), dtype=np.float64)
    slope, intercept = np.polyfit(idx, x, 1)
    return x - (slope * idx + intercept)


def fft_peaks(signal: np.ndarray, cruise_mask: np.ndarray, fs: float,
              bounds: list[tuple[int, int]], top_n: int = 3) -> dict:
    """Dominant oscillation peaks in the longest contiguous cruise run."""
    run = _longest_run(cruise_mask, bounds)
    n_runs = len(contiguous_runs(cruise_mask, bounds))
    if run is None or fs <= 0:
        return {
            "note": "no cruise samples for oscillation analysis",
            "has_significant_oscillation": False,
            "cruise_duration_s": 0.0,
            "insufficient_duration": True,
        }

    start, stop = run
    n = stop - start
    duration = n / fs

    if duration < MIN_CRUISE_DURATION_S or n < 64:
        return {
            "note": (f"insufficient cruise duration for oscillation analysis "
                     f"({duration:.2f} s < {MIN_CRUISE_DURATION_S} s contiguous)"),
            "has_significant_oscillation": False,
            "cruise_duration_s": round(duration, 3),
            "n_cruise_runs": n_runs,
            "insufficient_duration": True,
        }

    x = _detrend(signal[start:stop].astype(np.float64))
    window = np.hanning(n)
    spectrum = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(spectrum) * 2.0 / n / np.mean(window)

    valid = (freqs >= MIN_OSCILLATION_HZ) & (
        freqs >= MIN_CYCLES_FOR_PEAK / duration)
    if not np.any(valid):
        return {
            "note": f"no frequency bins above {MIN_OSCILLATION_HZ} Hz floor",
            "has_significant_oscillation": False,
            "cruise_duration_s": round(duration, 3),
            "n_cruise_runs": n_runs,
            "analysis_band_hz": f"{MIN_OSCILLATION_HZ} to {round(fs / 2, 1)}",
        }

    mag_v = mag[valid]
    freqs_v = freqs[valid]
    noise_floor = float(np.median(mag_v))
    threshold = max(noise_floor * NOISE_FLOOR_SIGMA, 1e-12)

    peaks: list[tuple[float, float]] = []
    for i in range(1, len(mag_v) - 1):
        if (mag_v[i] > mag_v[i - 1] and mag_v[i] > mag_v[i + 1]
                and mag_v[i] > threshold):
            peaks.append((float(freqs_v[i]), float(mag_v[i])))
    peaks.sort(key=lambda p: -p[1])

    return {
        "analysis_band_hz": f"{MIN_OSCILLATION_HZ} to {round(fs / 2, 1)}",
        "noise_floor": round(noise_floor, 6),
        "peaks": [
            {"freq_hz": round(f, 1), "amplitude": round(a, 6)}
            for f, a in peaks[:top_n]
        ],
        "dominant_hz": round(peaks[0][0], 1) if peaks else None,
        "has_significant_oscillation": bool(peaks),
        "cruise_duration_s": round(duration, 3),
        "n_cruise_runs": n_runs,
    }


def cross_phase(a: np.ndarray, b: np.ndarray, cruise_mask: np.ndarray,
                fs: float, bounds: list[tuple[int, int]]) -> dict | None:
    """Phase of ``a`` relative to ``b`` at their strongest shared frequency.

    Used for current-vs-velocity: ~+90° = mechanical resonance, ~0° = loop
    instability. Runs on the longest contiguous cruise run.
    """
    run = _longest_run(cruise_mask, bounds)
    if run is None or fs <= 0:
        return None
    start, stop = run
    n = stop - start
    duration = n / fs
    if duration < MIN_CRUISE_DURATION_S or n < 64:
        return None

    a_run = _detrend(a[start:stop].astype(np.float64))
    b_run = _detrend(b[start:stop].astype(np.float64))
    window = np.hanning(n)
    spec_a = np.fft.rfft(a_run * window)
    spec_b = np.fft.rfft(b_run * window)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    cross = spec_a * np.conj(spec_b)

    # Coherence proxy: both signals must exceed 5x their median magnitude
    mag_a = np.abs(spec_a)
    mag_b = np.abs(spec_b)
    coherent = (mag_a > 5.0 * np.median(mag_a)) & (
        mag_b > 5.0 * np.median(mag_b))

    valid = ((freqs >= MIN_OSCILLATION_HZ)
             & (freqs >= MIN_CYCLES_FOR_PEAK / duration)
             & coherent)
    if not np.any(valid):
        return {
            "note": "no coherent oscillation detected in cruise segments",
            "dominant_freq_hz": None,
            "analysis_band_hz": f"{MIN_OSCILLATION_HZ} to {round(fs / 2, 1)}",
            "cruise_duration_s": round(duration, 3),
        }

    weight = mag_a * mag_b
    idx_rel = int(np.argmax(weight[valid]))
    idx = np.where(valid)[0][idx_rel]
    phase_deg = float(np.degrees(np.angle(cross[idx])))

    if 60 < phase_deg < 120:
        interp = "~+90° (current leads velocity) → MECHANICAL RESONANCE → notch filter"
    elif -30 < phase_deg < 30:
        interp = "~0° (in-phase) → LOOP INSTABILITY → reduce Pn102 or position gain"
    elif -120 < phase_deg < -60:
        interp = "~-90° (velocity leads current) → unusual, check sign conventions"
    else:
        interp = f"{phase_deg:.0f}° → intermediate, pattern unclear"

    return {
        "analysis_band_hz": f"{MIN_OSCILLATION_HZ} to {round(fs / 2, 1)}",
        "dominant_freq_hz": round(float(freqs[idx]), 1),
        "phase_deg": round(phase_deg, 1),
        "interpretation": interp,
        "cruise_duration_s": round(duration, 3),
    }
