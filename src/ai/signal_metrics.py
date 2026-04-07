"""
Deterministic signal metrics computed from scope data.
These are calculated locally (no API call) and fed to the LLM as structured context,
keeping token usage low and analysis accurate.
"""

import numpy as np
from typing import Optional


class SignalMetrics:
    """Compute standard signal analysis metrics from scope capture data."""

    @staticmethod
    def compute_all(time_arr: np.ndarray, params: dict[str, np.ndarray],
                    sample_period: Optional[float] = None) -> dict:
        """
        Compute a full metrics summary for all parameters in the capture.

        Returns a dict suitable for JSON serialization and LLM consumption:
        {
            "capture_info": { ... },
            "parameters": {
                "MPOS(0)": { "stats": {...}, "dynamics": {...}, "frequency": {...} },
                ...
            }
        }
        """
        if sample_period is None and len(time_arr) > 1:
            sample_period = float(time_arr[1] - time_arr[0])

        n_samples = len(time_arr)
        duration = float(time_arr[-1] - time_arr[0]) if n_samples > 1 else 0.0

        result = {
            "capture_info": {
                "num_samples": n_samples,
                "duration_s": round(duration, 6),
                "sample_period_s": round(sample_period, 6) if sample_period else None,
                "sample_rate_hz": round(1.0 / sample_period, 1) if sample_period and sample_period > 0 else None,
                "parameters": list(params.keys()),
            },
            "parameters": {},
        }

        for name, values in params.items():
            if len(values) == 0:
                continue
            result["parameters"][name] = SignalMetrics._analyze_parameter(
                time_arr, values, sample_period
            )

        return result

    @staticmethod
    def _analyze_parameter(time_arr: np.ndarray, values: np.ndarray,
                           sample_period: Optional[float]) -> dict:
        """Analyze a single parameter trace."""
        n = len(values)
        metrics = {}

        # Basic statistics
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        v_mean = float(np.mean(values))
        v_std = float(np.std(values))
        v_range = v_max - v_min

        metrics["stats"] = {
            "min": round(v_min, 6),
            "max": round(v_max, 6),
            "mean": round(v_mean, 6),
            "std": round(v_std, 6),
            "range": round(v_range, 6),
            "start_value": round(float(values[0]), 6),
            "end_value": round(float(values[-1]), 6),
        }

        # Dynamics — velocity/acceleration of the signal itself
        if n > 1 and sample_period and sample_period > 0:
            dt = sample_period
            velocity = np.diff(values) / dt
            metrics["dynamics"] = {
                "max_rate_of_change": round(float(np.max(np.abs(velocity))), 6),
                "mean_rate_of_change": round(float(np.mean(np.abs(velocity))), 6),
                "is_monotonic_increasing": bool(np.all(velocity >= -1e-9)),
                "is_monotonic_decreasing": bool(np.all(velocity <= 1e-9)),
                "is_stationary": bool(v_std < 1e-6 * max(abs(v_mean), 1.0)),
            }

            # Settling detection: find if signal settles within a band
            if v_range > 0:
                settle_band = v_range * 0.02  # 2% band
                final_value = float(values[-1])
                settled = np.abs(values - final_value) <= settle_band
                # Find first index where it stays settled until end
                settled_from_end = np.flip(settled)
                settled_count = 0
                for s in settled_from_end:
                    if s:
                        settled_count += 1
                    else:
                        break
                if settled_count > n * 0.1:  # at least 10% of samples
                    settle_idx = n - settled_count
                    metrics["dynamics"]["settling_time_s"] = round(
                        float(time_arr[settle_idx] - time_arr[0]), 6
                    )
                    metrics["dynamics"]["settled_value"] = round(final_value, 6)

            # Overshoot detection (for step-like signals)
            if v_range > 0:
                final_val = float(values[-1])
                start_val = float(values[0])
                step_size = final_val - start_val
                if abs(step_size) > v_range * 0.1:  # meaningful step
                    if step_size > 0:
                        overshoot = (v_max - final_val) / abs(step_size) * 100
                    else:
                        overshoot = (final_val - v_min) / abs(step_size) * 100
                    if overshoot > 1.0:  # > 1% overshoot is notable
                        metrics["dynamics"]["overshoot_percent"] = round(overshoot, 2)

        # Frequency analysis — dominant frequency via FFT
        if n > 8 and sample_period and sample_period > 0:
            centered = values - v_mean
            window = np.hanning(n)
            windowed = centered * window
            fft_vals = np.fft.rfft(windowed)
            magnitude = np.abs(fft_vals) * 2.0 / np.sum(window)
            freqs = np.fft.rfftfreq(n, d=sample_period)

            # Skip DC bin
            if len(magnitude) > 1:
                mag_no_dc = magnitude[1:]
                freqs_no_dc = freqs[1:]

                if len(mag_no_dc) > 0:
                    peak_idx = np.argmax(mag_no_dc)
                    peak_freq = float(freqs_no_dc[peak_idx])
                    peak_mag = float(mag_no_dc[peak_idx])

                    # RMS of spectrum for SNR estimate
                    rms_mag = float(np.sqrt(np.mean(mag_no_dc**2)))

                    metrics["frequency"] = {
                        "dominant_freq_hz": round(peak_freq, 3),
                        "dominant_amplitude": round(peak_mag, 6),
                        "spectral_rms": round(rms_mag, 6),
                        "nyquist_freq_hz": round(float(freqs[-1]), 3),
                    }

                    # Find top 3 peaks
                    if len(mag_no_dc) > 10:
                        top_indices = np.argsort(mag_no_dc)[-3:][::-1]
                        metrics["frequency"]["top_peaks_hz"] = [
                            round(float(freqs_no_dc[i]), 3) for i in top_indices
                        ]

        # Anomaly detection — spikes, clipping, sudden jumps
        anomalies = []
        if n > 2 and sample_period:
            # Spike detection: points > 4 sigma from local mean
            if v_std > 1e-9:
                z_scores = np.abs(values - v_mean) / v_std
                spike_mask = z_scores > 4.0
                spike_count = int(np.sum(spike_mask))
                if spike_count > 0:
                    spike_times = time_arr[spike_mask]
                    anomalies.append({
                        "type": "spikes",
                        "count": spike_count,
                        "first_at_s": round(float(spike_times[0]), 6),
                    })

            # Clipping detection: sustained min or max values
            at_min = values == v_min
            at_max = values == v_max
            min_runs = SignalMetrics._count_sustained_runs(at_min, threshold=5)
            max_runs = SignalMetrics._count_sustained_runs(at_max, threshold=5)
            if min_runs > 0 or max_runs > 0:
                anomalies.append({
                    "type": "clipping",
                    "at_min_runs": min_runs,
                    "at_max_runs": max_runs,
                })

            # Step/jump detection: large instantaneous changes
            if n > 1:
                diffs = np.abs(np.diff(values))
                median_diff = float(np.median(diffs))
                if median_diff > 0:
                    jump_mask = diffs > median_diff * 20
                    jump_count = int(np.sum(jump_mask))
                    if jump_count > 0:
                        jump_indices = np.where(jump_mask)[0]
                        anomalies.append({
                            "type": "sudden_jumps",
                            "count": jump_count,
                            "at_times_s": [
                                round(float(time_arr[i]), 6)
                                for i in jump_indices[:5]  # first 5
                            ],
                        })

        if anomalies:
            metrics["anomalies"] = anomalies

        return metrics

    @staticmethod
    def _count_sustained_runs(mask: np.ndarray, threshold: int = 5) -> int:
        """Count runs of True values longer than threshold."""
        if not np.any(mask):
            return 0
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        # Handle edge cases
        if mask[0]:
            starts = np.concatenate(([0], starts))
        if mask[-1]:
            ends = np.concatenate((ends, [len(mask)]))
        run_lengths = ends - starts
        return int(np.sum(run_lengths >= threshold))

    @staticmethod
    def format_for_llm(metrics: dict) -> str:
        """
        Format metrics dict as a concise text summary for the LLM system prompt.
        This keeps token count low while providing structured data.
        """
        lines = []
        info = metrics["capture_info"]
        lines.append(f"Capture: {info['num_samples']} samples, "
                     f"{info['duration_s']}s duration, "
                     f"{info['sample_rate_hz']} Hz sample rate")
        lines.append(f"Parameters: {', '.join(info['parameters'])}")
        lines.append("")

        for name, pm in metrics["parameters"].items():
            lines.append(f"--- {name} ---")
            s = pm["stats"]
            lines.append(f"  Range: [{s['min']}, {s['max']}], Mean: {s['mean']}, "
                         f"Std: {s['std']}, Start: {s['start_value']}, End: {s['end_value']}")

            if "dynamics" in pm:
                d = pm["dynamics"]
                parts = [f"MaxRate: {d['max_rate_of_change']}"]
                if d.get("is_stationary"):
                    parts.append("STATIONARY")
                if "settling_time_s" in d:
                    parts.append(f"SettlingTime: {d['settling_time_s']}s")
                if "overshoot_percent" in d:
                    parts.append(f"Overshoot: {d['overshoot_percent']}%")
                lines.append(f"  Dynamics: {', '.join(parts)}")

            if "frequency" in pm:
                f = pm["frequency"]
                lines.append(f"  Dominant freq: {f['dominant_freq_hz']} Hz "
                             f"(amplitude {f['dominant_amplitude']})")
                if "top_peaks_hz" in f:
                    lines.append(f"  Top peaks: {f['top_peaks_hz']} Hz")

            if "anomalies" in pm:
                for a in pm["anomalies"]:
                    if a["type"] == "spikes":
                        lines.append(f"  ANOMALY: {a['count']} spikes detected, "
                                     f"first at {a['first_at_s']}s")
                    elif a["type"] == "clipping":
                        lines.append(f"  ANOMALY: Signal clipping detected "
                                     f"(min_runs={a['at_min_runs']}, max_runs={a['at_max_runs']})")
                    elif a["type"] == "sudden_jumps":
                        lines.append(f"  ANOMALY: {a['count']} sudden jumps at "
                                     f"{a['at_times_s']}s")
            lines.append("")

        return "\n".join(lines)
