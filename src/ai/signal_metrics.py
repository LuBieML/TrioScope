"""
Signal metrics for ScopeEngine AI analysis.

The LLM cannot read raw numeric arrays — it pattern-matches named scalars
against the rules in its system prompt. This module turns a full-rate scope
capture into a compact, structured report of named metrics the LLM can
reason over directly.

Every number the LLM is allowed to trust comes from here.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# --- Tuning constants -------------------------------------------------------
EPS_VEL_FRAC = 0.02   # 2% of peak demand velocity → "moving"
EPS_ACC_FRAC = 0.10   # 10% of peak demand accel → "constant velocity"
SETTLE_MS = 200       # settle window after each move end
NOISE_FLOOR_SIGMA = 5.0  # FFT peak threshold above median noise floor (raised for noise rejection)
SATURATION_FRAC = 0.95   # |current| > 95% of observed peak = near-saturation
MIN_OSCILLATION_HZ = 5.0 # position loop bandwidth floor
MIN_CRUISE_DURATION_S = 0.3   # need at least 300 ms of cruise for FFT
MIN_CYCLES_FOR_PEAK = 3       # peak must fit ≥3 cycles in analyzed window
MIN_COHERENCE = 0.7           # cross-phase coherence threshold (proxy used instead)


def _find_channel(params: dict, *keywords: str) -> str | None:
    """Case-insensitive fuzzy channel name match."""
    for key in params:
        k = key.lower().replace("_", "").replace(" ", "").replace("-", "")
        for kw in keywords:
            if kw in k:
                return key
    return None


@dataclass
class PhaseStats:
    n: int = 0
    mean: float = 0.0
    std: float = 0.0
    rms: float = 0.0
    vmin: float = 0.0
    vmax: float = 0.0

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PhaseStats":
        if arr.size == 0:
            return cls()
        return cls(
            n=int(arr.size),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            rms=float(np.sqrt(np.mean(arr ** 2))),
            vmin=float(np.min(arr)),
            vmax=float(np.max(arr)),
        )

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "rms": round(self.rms, 4),
            "peak_abs": round(max(abs(self.vmin), abs(self.vmax)), 4),
        }


class SignalMetrics:
    """Compute and format a structured metrics report from a scope capture."""

    # ---------------------------------------------------------------- public
    @classmethod
    def compute_all(cls, time_arr: np.ndarray, params: dict) -> dict:
        result: dict = {
            "capture": {},
            "channels_detected": {},
            "phases": {},
            "fe": {},
            "velocity": {},
            "current": {},
            "oscillation": {},
            "asymmetry": {},
            "settle": {},
            "data_sufficiency": "OK",
            "warnings": [],
        }

        n = len(time_arr)
        if n < 32:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append(f"capture too short ({n} samples, need >=32)")
            return result

        dt = float(np.median(np.diff(time_arr)))
        fs = 1.0 / dt if dt > 0 else 0.0
        duration = float(time_arr[-1] - time_arr[0])
        result["capture"] = {
            "duration_s": round(duration, 4),
            "n_samples": n,
            "sample_rate_hz": round(fs, 1),
            "dt_ms": round(dt * 1000, 3),
            "nyquist_hz": round(fs / 2, 1),
        }

        # --- Channel detection ---
        ch_dpos = _find_channel(params, "dpos", "demandposition", "targetposition")
        ch_mpos = _find_channel(params, "mpos", "measuredposition", "actualposition")
        ch_fe = _find_channel(params, "fe", "followingerror")
        ch_dvel = _find_channel(params, "demandspeed", "demandvel", "dspeed")
        ch_mvel = _find_channel(params, "mspeed", "measuredvel", "actualvel", "vactual")
        ch_cur = _find_channel(params, "current", "torque", "dacout")

        result["channels_detected"] = {
            "dpos": ch_dpos, "mpos": ch_mpos, "fe": ch_fe,
            "demand_vel": ch_dvel, "measured_vel": ch_mvel, "current": ch_cur,
        }

        dpos = params.get(ch_dpos) if ch_dpos else None
        mpos = params.get(ch_mpos) if ch_mpos else None
        fe = params.get(ch_fe) if ch_fe else None
        mvel = params.get(ch_mvel) if ch_mvel else None
        cur = params.get(ch_cur) if ch_cur else None

        # Derive FE from DPOS/MPOS if not captured directly
        if fe is None and dpos is not None and mpos is not None:
            fe = dpos - mpos
            result["warnings"].append("FE derived from DPOS-MPOS (not captured directly)")

        # Derive demand velocity
        if dpos is not None:
            dvel = np.gradient(dpos, time_arr)
        elif ch_dvel:
            dvel = params[ch_dvel]
        else:
            dvel = None
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append(
                "no DPOS or demand-velocity channel — cannot segment motion phases"
            )
            return result

        # Data sufficiency: is anything actually moving?
        v_peak = float(np.max(np.abs(dvel)))
        if v_peak < 1e-9:
            result["data_sufficiency"] = "INSUFFICIENT"
            result["warnings"].append("no motion detected in demand velocity (idle capture)")
            return result

        # --- Phase segmentation ---
        phases = cls._segment_phases(time_arr, dvel, dt)
        result["phases"] = {
            "n_moves": int(phases["n_moves"]),
            "n_reversals": int(phases["n_reversals"]),
            "idle_pct": round(100 * phases["idle"].sum() / n, 1),
            "accel_pct": round(100 * phases["accel"].sum() / n, 1),
            "cruise_pct": round(100 * phases["cruise"].sum() / n, 1),
            "decel_pct": round(100 * phases["decel"].sum() / n, 1),
            "settle_pct": round(100 * phases["settle"].sum() / n, 1),
            "reversal_pct": round(100 * phases["reversal"].sum() / n, 1),
            "peak_demand_velocity": round(v_peak, 4),
        }

        # --- FE per-phase + cruise slope vs velocity ---
        if fe is not None:
            result["fe"] = cls._analyze_fe(fe, dvel, phases)
        else:
            result["warnings"].append("no FE channel (and no DPOS+MPOS to derive from)")

        # --- Velocity tracking error ---
        if mvel is not None:
            result["velocity"] = cls._analyze_velocity(mvel, dvel, phases)

        # --- Current / torque ---
        if cur is not None:
            result["current"] = cls._analyze_current(cur, phases, dt)

        # --- Oscillation (FFT, cruise-only) ---
        cruise_mask = phases["cruise"]
        if fe is not None:
            result["oscillation"]["fe"] = cls._fft_peaks(fe, cruise_mask, fs)
        if mvel is not None:
            vel_err = mvel - dvel
            result["oscillation"]["velocity_error"] = cls._fft_peaks(
                vel_err, cruise_mask, fs)
        if cur is not None:
            result["oscillation"]["current"] = cls._fft_peaks(
                cur - np.mean(cur), cruise_mask, fs)

        # --- Phase relationship between current and velocity ---
        if cur is not None and mvel is not None:
            result["oscillation"]["current_vs_velocity_phase"] = cls._cross_phase(
                cur - np.mean(cur), mvel - np.mean(mvel), cruise_mask, fs
            )

        # --- Asymmetry (+ vs - direction) ---
        if fe is not None:
            result["asymmetry"] = cls._analyze_asymmetry(fe, dvel, phases)

        # --- Settling ---
        if fe is not None:
            result["settle"] = cls._analyze_settling(fe, phases, dt)

        # --- New warnings ---
        # Insufficient cruise duration for oscillation
        for sig_name in ("fe", "velocity_error", "current"):
            fft_res = result["oscillation"].get(sig_name, {})
            if ("insufficient cruise duration" in fft_res.get("note", "")
                    and not any("oscillation analysis skipped" in w
                                for w in result["warnings"])):
                dur = fft_res["note"].split("(")[1].split(" s")[0]
                result["warnings"].append(
                    f"oscillation analysis skipped: insufficient cruise duration "
                    f"({dur}s)")
                break

        # Multi-move capture
        n_moves = phases["n_moves"]
        if n_moves > 1:
            result["warnings"].append(
                f"multi-move capture detected ({n_moves} moves) — per-move stats "
                f"aggregated; check reversal transient stats for stiction/backlash "
                f"signatures")

        # Large FE spikes at reversals
        fe_data = result.get("fe", {})
        if "reversal" in fe_data and "cruise" in fe_data:
            rev_peak = fe_data["reversal"]["peak_abs"]
            cruise_peak = fe_data["cruise"]["peak_abs"]
            if cruise_peak > 0 and rev_peak > 5 * cruise_peak:
                result["warnings"].append(
                    f"large FE spikes at reversals ({rev_peak} vs cruise "
                    f"{cruise_peak}) — likely stiction, backlash, or triangle-wave "
                    f"acceleration discontinuity")

        return result

    # ---------------------------------------------------------------- phases
    @staticmethod
    def _segment_phases(t: np.ndarray, dvel: np.ndarray, dt: float) -> dict:
        n = len(dvel)
        dacc = np.gradient(dvel, t)

        v_max = float(np.max(np.abs(dvel)))
        a_max = float(np.max(np.abs(dacc)))
        v_thresh = EPS_VEL_FRAC * v_max if v_max > 0 else 1e-9
        a_thresh = EPS_ACC_FRAC * a_max if a_max > 0 else 1e-9

        # --- Reversal detection: zero-crossings of demand velocity ---
        signs = np.sign(dvel)
        sign_diff = signs[:-1] != signs[1:]
        above_thresh = (np.abs(dvel[:-1]) > v_thresh) | (np.abs(dvel[1:]) > v_thresh)
        reversal_indices = np.where(sign_diff & above_thresh)[0]

        reversal_half_width = max(1, int(0.080 / dt))
        reversal = np.zeros(n, dtype=bool)
        for idx in reversal_indices:
            lo = max(0, idx - reversal_half_width + 1)
            hi = min(n, idx + 1 + reversal_half_width)
            reversal[lo:hi] = True

        # --- Phase masks (reversal excluded from everything) ---
        kinematic_idle = np.abs(dvel) <= v_thresh
        moving = ~kinematic_idle & ~reversal

        accel = moving & (dacc > a_thresh)
        decel = moving & (dacc < -a_thresh)
        cruise = moving & ~accel & ~decel
        idle = kinematic_idle & ~reversal

        # --- Settle (moving → ~moving edges, reversal wins on overlap) ---
        settle = np.zeros(n, dtype=bool)
        settle_samples = max(1, int(SETTLE_MS * 1e-3 / dt))
        transitions = np.where(moving[:-1] & ~moving[1:])[0]
        for idx in transitions:
            end = min(n, idx + 1 + settle_samples)
            settle[idx + 1:end] = True
        settle = settle & ~reversal
        idle = idle & ~settle

        # --- Identify individual moves as contiguous runs of `moving` ---
        changes = np.diff(moving.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0]
        if moving[0]:
            starts = np.concatenate(([0], starts))
        if moving[-1]:
            ends = np.concatenate((ends, [n - 1]))
        moves = list(zip(starts.tolist(), ends.tolist()))

        return {
            "idle": idle, "accel": accel, "cruise": cruise,
            "decel": decel, "settle": settle, "moving": moving,
            "reversal": reversal, "moves": moves,
            "n_moves": len(moves), "n_reversals": len(reversal_indices),
            "transitions": transitions,
        }

    # ---------------------------------------------------------------- FE
    @staticmethod
    def _analyze_fe(fe: np.ndarray, dvel: np.ndarray, phases: dict) -> dict:
        result: dict = {}
        for phase_name in ("idle", "accel", "cruise", "decel", "settle", "reversal"):
            mask = phases[phase_name]
            if mask.sum() > 0:
                result[phase_name] = PhaseStats.from_array(fe[mask]).as_dict()

        # Cruise FE vs velocity linear fit — THE key VFF diagnostic
        cruise = phases["cruise"]
        if cruise.sum() > 20:
            v = dvel[cruise]
            f = fe[cruise]
            if np.std(v) > 1e-9:
                slope, intercept = np.polyfit(v, f, 1)
                residual = f - (slope * v + intercept)
                signal = slope * np.mean(np.abs(v))
                noise = np.std(residual)
                result["cruise_fe_vs_velocity"] = {
                    "slope": round(float(slope), 6),
                    "intercept": round(float(intercept), 4),
                    "proportional_to_velocity": bool(abs(signal) > 2 * noise),
                    "note": (
                        "slope>0 with proportional_to_velocity=true → FE scales with "
                        "speed → insufficient VFF_GAIN/Pn112"
                    ),
                }
        return result

    # ---------------------------------------------------------------- velocity
    @staticmethod
    def _analyze_velocity(mvel: np.ndarray, dvel: np.ndarray, phases: dict) -> dict:
        result: dict = {}
        err = mvel - dvel
        for phase_name in ("accel", "cruise", "decel", "reversal"):
            mask = phases[phase_name]
            if mask.sum() > 5:
                result[phase_name + "_err"] = PhaseStats.from_array(err[mask]).as_dict()

        # Per-move velocity overshoot (replaces velocity_overshoot_accel_peak)
        accel_mask = phases["accel"]
        moves = phases.get("moves", [])
        overshoot_values: list[float] = []
        for start, end in moves:
            move_range = np.zeros(len(mvel), dtype=bool)
            move_range[start:end + 1] = True
            move_accel = accel_mask & move_range
            if move_accel.sum() > 0:
                signed_err = err[move_accel] * np.sign(dvel[move_accel])
                overshoot_values.append(round(float(np.max(signed_err)), 4))
        result["velocity_overshoot_per_move"] = {
            "per_move": overshoot_values,
            "max": round(float(max(overshoot_values)), 4) if overshoot_values else 0.0,
            "n_moves": len(moves),
        }

        cruise_mask = phases["cruise"]
        if cruise_mask.sum() > 0:
            ratio = float(np.mean(np.abs(mvel[cruise_mask])) /
                          max(np.mean(np.abs(dvel[cruise_mask])), 1e-9))
            result["cruise_velocity_reach_ratio"] = round(ratio, 3)
        return result

    # ---------------------------------------------------------------- current
    @staticmethod
    def _analyze_current(cur: np.ndarray, phases: dict, dt: float) -> dict:
        peak = float(np.max(np.abs(cur)))
        if peak < 1e-9:
            return {"note": "no current signal detected"}

        sat_thresh = SATURATION_FRAC * peak
        min_run_samples = max(1, int(0.010 / dt))
        result: dict = {"observed_peak": round(peak, 4)}
        for phase_name in ("accel", "cruise", "decel", "idle", "reversal"):
            mask = phases[phase_name]
            if mask.sum() > 0:
                # Sustained saturation: only count runs >= 10 ms
                sat_raw = np.abs(cur[mask]) > sat_thresh
                padded = np.concatenate(([False], sat_raw, [False]))
                diffs = np.diff(padded.astype(np.int8))
                run_starts = np.where(diffs == 1)[0]
                run_ends = np.where(diffs == -1)[0]
                run_lengths = run_ends - run_starts
                sustained = int(np.sum(run_lengths[run_lengths >= min_run_samples]))
                sat_pct = 100 * sustained / mask.sum()

                stats = PhaseStats.from_array(cur[mask])
                entry = stats.as_dict()
                entry["saturation_pct"] = round(float(sat_pct), 1)
                result[phase_name] = entry
        result["saturation_note"] = (
            "saturation_pct = % of samples in sustained runs (>=10 ms) within 5% of "
            "observed capture peak; confirm against drive rated current before "
            "concluding torque-limited"
        )

        # Bimodality guard (belt-and-braces, per-move segmentation should
        # make this unreachable in practice)
        cruise_mask = phases["cruise"]
        if cruise_mask.sum() > 20:
            c = cur[cruise_mask]
            median_abs = float(np.median(np.abs(c)))
            mean_abs = float(abs(np.mean(c)))
            if median_abs > 3 * mean_abs and median_abs > 0:
                result["cruise_bimodal_warning"] = (
                    f"cruise current appears bimodal (median|x|={median_abs:.1f} vs "
                    f"|mean|={mean_abs:.1f}) — likely multiple moves with direction "
                    f"reversals pooled into one phase. std is NOT oscillation."
                )

        return result

    # ---------------------------------------------------------------- FFT
    @staticmethod
    def _fft_peaks(signal: np.ndarray, cruise_mask: np.ndarray, fs: float,
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

    # ---------------------------------------------------------------- cross phase
    @staticmethod
    def _cross_phase(a: np.ndarray, b: np.ndarray, cruise_mask: np.ndarray,
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

    # ---------------------------------------------------------------- asymmetry
    @staticmethod
    def _analyze_asymmetry(fe: np.ndarray, dvel: np.ndarray, phases: dict) -> dict:
        cruise = phases["cruise"]
        if cruise.sum() < 10:
            return {}
        v = dvel[cruise]
        f = fe[cruise]
        pos = v > 0
        neg = v < 0
        if pos.sum() < 5 or neg.sum() < 5:
            return {"note": "insufficient bidirectional cruise data"}

        pos_mean = float(np.mean(f[pos]))
        neg_mean = float(np.mean(f[neg]))
        denom = max(abs(pos_mean), abs(neg_mean), 1e-9)
        ratio = abs(pos_mean - neg_mean) / denom
        return {
            "cruise_fe_pos_dir_mean": round(pos_mean, 4),
            "cruise_fe_neg_dir_mean": round(neg_mean, 4),
            "asymmetry_ratio": round(ratio, 3),
            "significant": bool(ratio > 0.2),
            "note": "significant=true → friction/stiction, backlash, or gravity load",
        }

    # ---------------------------------------------------------------- settling
    @staticmethod
    def _analyze_settling(fe: np.ndarray, phases: dict, dt: float) -> dict:
        settle_mask = phases["settle"]
        if settle_mask.sum() < 5:
            return {}
        fe_s = fe[settle_mask]
        tail = fe_s[-max(1, len(fe_s) // 4):]
        steady = float(np.mean(tail))
        signs = np.sign(fe_s - steady)
        zc = int(np.sum(np.diff(signs) != 0))
        return {
            "fe_at_settle_start": round(float(fe_s[0]), 4),
            "fe_steady_state": round(steady, 4),
            "fe_peak_during_settle": round(float(np.max(np.abs(fe_s))), 4),
            "zero_crossings": zc,
            "ringing": bool(zc > 3),
            "steady_state_offset_nonzero": bool(abs(steady) > 2 * float(np.std(tail))),
            "note": (
                "ringing=true → underdamped (↑D_GAIN or ↓P_GAIN); "
                "steady_state_offset_nonzero=true → insufficient integral action"
            ),
        }

    # ---------------------------------------------------------------- format
    @staticmethod
    def format_for_llm(metrics: dict) -> str:
        """Format the metrics dict as a compact named-scalar block for the LLM."""
        lines: list[str] = []

        lines.append(f"DATA SUFFICIENCY: {metrics.get('data_sufficiency', 'UNKNOWN')}")

        cap = metrics.get("capture", {})
        if cap:
            lines.append("\n## Capture")
            for k, v in cap.items():
                lines.append(f"  {k}: {v}")

        ch = metrics.get("channels_detected", {})
        if ch:
            lines.append("\n## Channels detected")
            for k, v in ch.items():
                lines.append(f"  {k}: {v or '(MISSING)'}")

        phases = metrics.get("phases", {})
        if phases:
            lines.append("\n## Phase segmentation")
            for k, v in phases.items():
                lines.append(f"  {k}: {v}")

        fe = metrics.get("fe", {})
        if fe:
            lines.append("\n## Following error per phase")
            for phase in ("idle", "accel", "cruise", "decel", "settle"):
                if phase in fe:
                    s = fe[phase]
                    lines.append(
                        f"  {phase}: mean={s['mean']} std={s['std']} "
                        f"rms={s['rms']} peak_abs={s['peak_abs']}"
                    )
            if "cruise_fe_vs_velocity" in fe:
                c = fe["cruise_fe_vs_velocity"]
                lines.append(
                    f"  cruise_fe_vs_velocity: slope={c['slope']} "
                    f"intercept={c['intercept']} "
                    f"proportional_to_velocity={c['proportional_to_velocity']}"
                )
                lines.append(f"    note: {c['note']}")

        vel = metrics.get("velocity", {})
        if vel:
            lines.append("\n## Velocity tracking (measured - demand)")
            for k, v in vel.items():
                if k == "velocity_overshoot_per_move":
                    lines.append(
                        f"  velocity_overshoot_per_move: n_moves={v['n_moves']} "
                        f"max={v['max']}")
                else:
                    lines.append(f"  {k}: {v}")

        cur = metrics.get("current", {})
        if cur:
            lines.append("\n## Drive current / torque")
            for k, v in cur.items():
                if k == "saturation_note":
                    lines.append(f"  note: {v}")
                elif k == "cruise_bimodal_warning":
                    lines.append(f"  WARNING: {v}")
                else:
                    lines.append(f"  {k}: {v}")

        osc = metrics.get("oscillation", {})
        if osc:
            lines.append("\n## Oscillation (FFT, cruise-only, hann-windowed)")
            for sig in ("fe", "velocity_error", "current"):
                if sig in osc:
                    lines.append(f"  {sig}: {osc[sig]}")
            if "current_vs_velocity_phase" in osc:
                cvp = osc["current_vs_velocity_phase"]
                if cvp:
                    lines.append(f"  current_vs_velocity_phase: {cvp}")

        asym = metrics.get("asymmetry", {})
        if asym:
            lines.append("\n## Directional asymmetry")
            for k, v in asym.items():
                lines.append(f"  {k}: {v}")

        settle = metrics.get("settle", {})
        if settle:
            lines.append("\n## Settling (first 200 ms after move end)")
            for k, v in settle.items():
                lines.append(f"  {k}: {v}")

        # Reversal transients
        fe_data = metrics.get("fe", {})
        cur_data = metrics.get("current", {})
        vel_data = metrics.get("velocity", {})
        has_reversal = ("reversal" in fe_data or "reversal" in cur_data
                        or "reversal_err" in vel_data)
        if has_reversal:
            n_rev = metrics.get("phases", {}).get("n_reversals", 0)
            lines.append(f"\n## Reversal transients ({n_rev} reversals detected)")
            if "reversal" in fe_data:
                s = fe_data["reversal"]
                lines.append(
                    f"  fe: mean={s['mean']} std={s['std']} "
                    f"rms={s['rms']} peak_abs={s['peak_abs']}")
            if "reversal_err" in vel_data:
                s = vel_data["reversal_err"]
                lines.append(
                    f"  velocity_err: mean={s['mean']} std={s['std']} "
                    f"rms={s['rms']} peak_abs={s['peak_abs']}")
            if "reversal" in cur_data:
                s = cur_data["reversal"]
                lines.append(
                    f"  current: mean={s['mean']} std={s['std']} "
                    f"rms={s['rms']} peak_abs={s['peak_abs']}")

        warnings = metrics.get("warnings", [])
        if warnings:
            lines.append("\n## Warnings")
            for w in warnings:
                lines.append(f"  - {w}")

        return "\n".join(lines)