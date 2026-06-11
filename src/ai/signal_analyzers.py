"""
Per-signal analyzers: following error, velocity tracking, current,
directional asymmetry, and settling, each computed per motion phase.
"""

from __future__ import annotations

import numpy as np

from .signal_constants import SATURATION_FRAC
from .signal_phases import PhaseStats


def analyze_fe(fe: np.ndarray, dvel: np.ndarray, phases: dict) -> dict:
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


def analyze_velocity(mvel: np.ndarray, dvel: np.ndarray, phases: dict) -> dict:
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


def analyze_current(cur: np.ndarray, phases: dict, dt: float) -> dict:
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


def analyze_asymmetry(fe: np.ndarray, dvel: np.ndarray, phases: dict) -> dict:
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


def analyze_settling(fe: np.ndarray, phases: dict, dt: float) -> dict:
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
