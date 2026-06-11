"""
Spectral analysis on cruise segments: FFT peak detection and the
current-vs-velocity cross-spectrum phase used to separate mechanical
resonance from loop instability.
"""

from __future__ import annotations

import numpy as np

from .signal_constants import (
    MIN_CRUISE_DURATION_S,
    MIN_CYCLES_FOR_PEAK,
    MIN_OSCILLATION_HZ,
    NOISE_FLOOR_SIGMA,
)


def fft_peaks(signal: np.ndarray, cruise_mask: np.ndarray, fs: float,
              top_n: int = 3) -> dict:
    cruise_signal = signal[cruise_mask]
    n_cruise = len(cruise_signal)
    cruise_duration = float(cruise_mask.sum()) / fs

    if cruise_duration < MIN_CRUISE_DURATION_S:
        return {
            "note": (f"insufficient cruise duration for oscillation analysis "
                     f"({cruise_duration:.2f} s < {MIN_CRUISE_DURATION_S} s)"),
            "has_significant_oscillation": False,
        }

    if n_cruise < 64 or fs <= 0:
        return {"note": "signal too short for FFT",
                "has_significant_oscillation": False}

    x = cruise_signal - np.mean(cruise_signal)
    window = np.hanning(n_cruise)
    X = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(n_cruise, 1.0 / fs)
    mag = np.abs(X) * 2.0 / n_cruise / np.mean(window)

    # Gate: MIN_OSCILLATION_HZ and MIN_CYCLES_FOR_PEAK
    valid = (freqs >= MIN_OSCILLATION_HZ) & (
        freqs >= MIN_CYCLES_FOR_PEAK / cruise_duration)
    if not np.any(valid):
        return {
            "note": f"no frequency bins above {MIN_OSCILLATION_HZ} Hz floor",
            "has_significant_oscillation": False,
            "cruise_duration_s": round(cruise_duration, 3),
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
        "cruise_duration_s": round(cruise_duration, 3),
    }


def cross_phase(a: np.ndarray, b: np.ndarray, cruise_mask: np.ndarray,
                fs: float) -> dict | None:
    a_cruise = a[cruise_mask]
    b_cruise = b[cruise_mask]
    n = len(a_cruise)
    cruise_duration = float(cruise_mask.sum()) / fs

    if cruise_duration < MIN_CRUISE_DURATION_S or n < 64:
        return None

    window = np.hanning(n)
    A = np.fft.rfft(a_cruise * window)
    B = np.fft.rfft(b_cruise * window)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    cross = A * np.conj(B)

    # Coherence proxy: both signals must exceed 5× their median magnitude
    mag_A = np.abs(A)
    mag_B = np.abs(B)
    coherent = (mag_A > 5.0 * np.median(mag_A)) & (
        mag_B > 5.0 * np.median(mag_B))

    valid = ((freqs >= MIN_OSCILLATION_HZ)
             & (freqs >= MIN_CYCLES_FOR_PEAK / cruise_duration)
             & coherent)
    if not np.any(valid):
        return {
            "note": "no coherent oscillation detected in cruise segments",
            "dominant_freq_hz": None,
            "analysis_band_hz": f"{MIN_OSCILLATION_HZ} to {round(fs / 2, 1)}",
            "cruise_duration_s": round(cruise_duration, 3),
        }

    weight = mag_A * mag_B
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
        "cruise_duration_s": round(cruise_duration, 3),
    }
