"""Spectral analysis of cruise-phase signals.

Welch-style analysis: hann windows are taken from *every* contiguous
cruise run (never across a splice — discontinuities leak broadband energy
and fake or bury peaks) and their spectra averaged. Repeated moves in one
capture therefore average together, cutting variance, and a long cruise
contributes several overlapping windows.

Peak frequencies are refined by parabolic interpolation of the peak bin
and its neighbours. Peaks in the drive's valid notch range can therefore
be targeted to a fraction of the bin width rather than 1/duration.

The current-vs-velocity phase test uses real magnitude-squared coherence
when at least two windows are available (a single window has coherence
identically 1, which proves nothing — in that case a magnitude proxy is
used and reported as such).
"""

from __future__ import annotations

import numpy as np

from .drive_profile import MIN_NOTCH_FILTER_HZ
from .signal_constants import (
    NOISE_FLOOR_SIGMA, MIN_OSCILLATION_HZ, MIN_CRUISE_DURATION_S,
    MIN_CYCLES_FOR_PEAK, MIN_FFT_SAMPLES, WELCH_MAX_NPERSEG, MIN_COHERENCE,
)
from .signal_phases import contiguous_runs


def _usable_runs(mask: np.ndarray,
                 bounds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [r for r in contiguous_runs(mask, bounds)
            if r[1] - r[0] >= MIN_FFT_SAMPLES]


def _pick_nperseg(runs: list[tuple[int, int]]) -> int:
    """Welch window length: ~the longest run, capped, floored.

    0.9x the longest run so that repeated moves with near-equal cruise
    lengths (the common capture pattern) all contribute a window.
    """
    longest = max(stop - start for start, stop in runs)
    return max(MIN_FFT_SAMPLES, min(WELCH_MAX_NPERSEG, int(0.9 * longest)))


def _detrend(x: np.ndarray) -> np.ndarray:
    """Remove mean + linear trend (cruise FE often carries a slope)."""
    idx = np.arange(len(x), dtype=np.float64)
    slope, intercept = np.polyfit(idx, x, 1)
    return x - (slope * idx + intercept)


def _welch_ffts(signals: list[np.ndarray], runs: list[tuple[int, int]],
                nperseg: int) -> list[list[np.ndarray]]:
    """Windowed FFT segments for each signal, from identical positions.

    Returns one list of rfft arrays per input signal; the k-th segment of
    every signal covers the same samples (required for cross-spectra).
    """
    window = np.hanning(nperseg)
    step = max(1, nperseg // 2)
    out: list[list[np.ndarray]] = [[] for _ in signals]
    for start, stop in runs:
        run_len = stop - start
        if run_len < nperseg:
            continue
        for pos in range(0, run_len - nperseg + 1, step):
            for sig_idx, signal in enumerate(signals):
                chunk = _detrend(
                    signal[start + pos:start + pos + nperseg].astype(np.float64))
                out[sig_idx].append(np.fft.rfft(chunk * window))
    return out


def _interpolate_peak(freqs: np.ndarray, mag: np.ndarray,
                      i: int) -> tuple[float, float]:
    """Sub-bin peak frequency/amplitude via a parabola through 3 bins."""
    if i <= 0 or i >= len(mag) - 1:
        return float(freqs[i]), float(mag[i])
    y0, y1, y2 = float(mag[i - 1]), float(mag[i]), float(mag[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(freqs[i]), y1
    delta = 0.5 * (y0 - y2) / denom
    delta = max(-0.5, min(0.5, delta))
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    peak_freq = float(freqs[i]) + delta * df
    peak_amp = y1 - 0.25 * (y0 - y2) * delta
    return peak_freq, float(peak_amp)


def _insufficient(duration: float, n_runs: int) -> dict:
    return {
        "note": (f"insufficient cruise duration for oscillation analysis "
                 f"({duration:.2f} s < {MIN_CRUISE_DURATION_S} s usable)"),
        "has_significant_oscillation": False,
        "cruise_duration_s": round(duration, 3),
        "n_cruise_runs": n_runs,
        "insufficient_duration": True,
    }


def fft_peaks(signal: np.ndarray, cruise_mask: np.ndarray, fs: float,
              bounds: list[tuple[int, int]], top_n: int = 3) -> dict:
    """Dominant oscillation peaks, Welch-averaged over all cruise runs."""
    if fs <= 0:
        return _insufficient(0.0, 0)
    runs = _usable_runs(cruise_mask, bounds)
    if not runs:
        return _insufficient(0.0, 0)

    duration = sum(stop - start for start, stop in runs) / fs
    if duration < MIN_CRUISE_DURATION_S:
        return _insufficient(duration, len(runs))

    nperseg = _pick_nperseg(runs)
    (segments,) = _welch_ffts([signal], runs, nperseg)
    n_averages = len(segments)

    window = np.hanning(nperseg)
    scale = 2.0 / nperseg / float(np.mean(window))
    mag = np.mean([np.abs(s) for s in segments], axis=0) * scale
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    df = fs / nperseg

    valid = (freqs >= MIN_OSCILLATION_HZ) & (freqs >= MIN_CYCLES_FOR_PEAK * df)
    if not np.any(valid):
        return {
            "note": f"no frequency bins above {MIN_OSCILLATION_HZ} Hz floor",
            "has_significant_oscillation": False,
            "cruise_duration_s": round(duration, 3),
            "n_cruise_runs": len(runs),
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
            peaks.append(_interpolate_peak(freqs_v, mag_v, i))
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
        "n_cruise_runs": len(runs),
        "n_averages": n_averages,
        "resolution_hz": round(df, 2),
        "note": "peak frequencies parabolically interpolated below bin width",
    }


def cross_phase(a: np.ndarray, b: np.ndarray, cruise_mask: np.ndarray,
                fs: float, bounds: list[tuple[int, int]]) -> dict | None:
    """Phase of ``a`` relative to ``b`` at their strongest shared frequency.

    Used for current-vs-velocity: ~+90° = mechanical resonance, ~0° = loop
    instability. With ≥2 Welch windows the claim is gated on real
    magnitude-squared coherence; with a single window (coherence would be
    identically 1) a magnitude proxy is used and reported.
    """
    if fs <= 0:
        return None
    runs = _usable_runs(cruise_mask, bounds)
    if not runs:
        return None
    duration = sum(stop - start for start, stop in runs) / fs
    if duration < MIN_CRUISE_DURATION_S:
        return None

    nperseg = _pick_nperseg(runs)
    segs_a, segs_b = _welch_ffts([a, b], runs, nperseg)
    n_averages = len(segs_a)
    if n_averages == 0:
        return None

    spec_a = np.stack(segs_a)
    spec_b = np.stack(segs_b)
    s_aa = np.mean(np.abs(spec_a) ** 2, axis=0)
    s_bb = np.mean(np.abs(spec_b) ** 2, axis=0)
    s_ab = np.mean(spec_a * np.conj(spec_b), axis=0)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    df = fs / nperseg

    # A genuine shared oscillation needs power in BOTH signals — this
    # prunes phase-lucky noise bins before any coherence decision.
    # (auto-spectra are power: 5x in magnitude = 25x in power)
    powered = ((s_aa > 25.0 * np.median(s_aa))
               & (s_bb > 25.0 * np.median(s_bb)))

    coherence: np.ndarray | None = None
    if n_averages >= 2:
        coh_arr = np.abs(s_ab) ** 2 / np.maximum(s_aa * s_bb, 1e-30)
        # Under independence, P(coh > c) = (1-c)^(n-1): with few averages
        # a fixed 0.7 gate passes noise, so require significance at 5%
        # per bin, floored at MIN_COHERENCE once n is large enough.
        significance = 1.0 - 0.05 ** (1.0 / (n_averages - 1))
        threshold = max(MIN_COHERENCE, significance)
        coherent = (coh_arr >= threshold) & powered
        coherence = coh_arr
        method = (f"welch coherence ({n_averages} averages, "
                  f"gate {threshold:.2f})")
    else:
        # Single window: coherence is identically 1 — fall back to the
        # magnitude proxy (both signals well above their own floors).
        coherent = powered
        method = "magnitude proxy (single window — coherence unavailable)"

    # The Nyquist (and DC) rfft bins are real-valued — their "phase" cannot
    # decorrelate across windows, so they fake coherence. Exclude them.
    valid = ((freqs >= MIN_OSCILLATION_HZ)
             & (freqs >= MIN_CYCLES_FOR_PEAK * df)
             & (freqs < fs / 2)
             & coherent)
    if not np.any(valid):
        return {
            "note": "no coherent oscillation detected in cruise segments",
            "dominant_freq_hz": None,
            "analysis_band_hz": f"{MIN_OSCILLATION_HZ} to {round(fs / 2, 1)}",
            "cruise_duration_s": round(duration, 3),
            "n_averages": n_averages,
            "method": method,
        }

    weight = np.abs(s_ab)
    idx_rel = int(np.argmax(weight[valid]))
    idx = np.where(valid)[0][idx_rel]
    phase_deg = float(np.degrees(np.angle(s_ab[idx])))

    peak_freq = float(freqs[idx])
    if 60 < phase_deg < 120 and peak_freq >= MIN_NOTCH_FILTER_HZ:
        interp = "~+90° (current leads velocity) → MECHANICAL RESONANCE → notch filter"
    elif 60 < phase_deg < 120:
        interp = ("~+90° (current leads velocity) → MECHANICAL RESONANCE "
                  f"→ below {MIN_NOTCH_FILTER_HZ:g} Hz notch-filter range")
    elif -30 < phase_deg < 30:
        interp = "~0° (in-phase) → LOOP INSTABILITY → reduce Pn102 or position gain"
    elif -120 < phase_deg < -60:
        interp = "~-90° (velocity leads current) → unusual, check sign conventions"
    else:
        interp = f"{phase_deg:.0f}° → intermediate, pattern unclear"

    return {
        "analysis_band_hz": f"{MIN_OSCILLATION_HZ} to {round(fs / 2, 1)}",
        "dominant_freq_hz": round(peak_freq, 1),
        "phase_deg": round(phase_deg, 1),
        "interpretation": interp,
        "coherence": (round(float(coherence[idx]), 2)
                      if coherence is not None else None),
        "cruise_duration_s": round(duration, 3),
        "n_averages": n_averages,
        "method": method,
    }
