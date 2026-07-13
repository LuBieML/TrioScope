"""Per-phase and per-move analyzers for scope captures.

All functions are pure numpy — no Qt, no hardware. They consume the phase
masks / windows produced by :mod:`signal_phases` and return plain dicts
ready for the metrics report and the tuner panel cards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .signal_constants import (
    SATURATION_FRAC, SETTLE_BAND_SIGMA, RINGING_CROSSINGS_MAX,
    MAX_PER_MOVE_REPORTED,
)


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


# ---------------------------------------------------------------- tolerance
def estimate_settle_band(fe: np.ndarray, phases: dict,
                         user_band: float | None = None) -> tuple[float, str]:
    """Tolerance band (± user units) for settling / ringing decisions.

    A user-supplied band wins. Otherwise the band is 4x a robust noise
    sigma estimated from first differences (immune to slow drift), taken
    from idle samples when enough exist.
    """
    if user_band is not None and user_band > 0:
        return float(user_band), "user"

    def _diff_sigma(x: np.ndarray) -> float:
        d = np.diff(x)
        d = d[np.isfinite(d)]
        if d.size >= 10:
            return 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)
        return float(np.std(x)) if x.size else 0.0

    idle = phases.get("idle")
    if idle is not None and idle.sum() >= 50:
        source = fe[idle]
        # Difference-based sigma alone collapses on correlated or quantized
        # FE (tiny sample-to-sample steps, real slow wander) and produces a
        # microscopic band that nothing ever "settles" into. The idle-value
        # spread is the floor: FE cannot be judged below its own idle wander.
        value_sigma = 1.4826 * float(
            np.median(np.abs(source - np.median(source))))
        sigma = max(_diff_sigma(source), value_sigma)
    else:
        # No usable idle stretch — differences only (values include motion).
        sigma = _diff_sigma(fe)
    band = max(SETTLE_BAND_SIGMA * sigma, 1e-12)
    return band, "auto (4x noise)"


# ---------------------------------------------------------------- FE
def analyze_fe(fe: np.ndarray, dvel: np.ndarray, phases: dict,
               velocity_units_known: bool = True) -> dict:
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
            entry = {
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 4),
                "proportional_to_velocity": bool(abs(signal) > 2 * noise),
                "note": (
                    "slope>0 with proportional_to_velocity=true → FE scales with "
                    "speed → insufficient VFF_GAIN/Pn112"
                ),
            }
            if not velocity_units_known:
                entry["units_note"] = (
                    "slope basis is raw demand-velocity units (servo period "
                    "unknown); the proportionality flag is still valid"
                )
            result["cruise_fe_vs_velocity"] = entry
    return result


# ---------------------------------------------------------------- velocity
def analyze_velocity(mvel: np.ndarray, dvel: np.ndarray, phases: dict) -> dict:
    """Velocity tracking metrics. ``mvel`` and ``dvel`` must share units."""
    result: dict = {}
    err = mvel - dvel
    for phase_name in ("accel", "cruise", "decel", "reversal"):
        mask = phases[phase_name]
        if mask.sum() > 5:
            result[phase_name + "_err"] = PhaseStats.from_array(err[mask]).as_dict()

    # Per-move velocity overshoot, normalised to that move's peak demand
    accel_mask = phases["accel"]
    moves = phases.get("moves", [])
    per_move: list[float] = []
    per_move_pct: list[float] = []
    for start, stop in moves:
        move_slice = slice(start, stop)
        move_accel = accel_mask[move_slice]
        if move_accel.sum() == 0:
            continue
        seg_err = err[move_slice][move_accel]
        seg_dvel = dvel[move_slice][move_accel]
        signed_err = seg_err * np.sign(seg_dvel)
        peak = float(np.max(signed_err))
        move_vpeak = float(np.max(np.abs(dvel[move_slice])))
        per_move.append(round(peak, 4))
        if move_vpeak > 1e-9:
            per_move_pct.append(round(100.0 * peak / move_vpeak, 2))
    result["velocity_overshoot_per_move"] = {
        "per_move": per_move[:MAX_PER_MOVE_REPORTED],
        "max": round(max(per_move), 4) if per_move else 0.0,
        "max_pct": round(max(per_move_pct), 2) if per_move_pct else 0.0,
        "n_moves": len(moves),
    }

    cruise_mask = phases["cruise"]
    if cruise_mask.sum() > 0:
        ratio = float(np.mean(np.abs(mvel[cruise_mask])) /
                      max(np.mean(np.abs(dvel[cruise_mask])), 1e-9))
        result["cruise_velocity_reach_ratio"] = round(ratio, 3)
    return result


# ---------------------------------------------------------------- current
def analyze_current(cur: np.ndarray, phases: dict, dt: float) -> dict:
    peak = float(np.max(np.abs(cur)))
    if peak < 1e-9:
        return {"note": "no current signal detected"}

    sat_thresh = SATURATION_FRAC * peak
    min_run_samples = max(1, int(0.010 / dt)) if dt > 0 else 1
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

            entry = PhaseStats.from_array(cur[mask]).as_dict()
            entry["saturation_pct"] = round(float(sat_pct), 1)
            result[phase_name] = entry
    result["saturation_note"] = (
        "saturation_pct = % of samples in sustained runs (>=10 ms) within 5% of "
        "observed capture peak; confirm against drive rated current before "
        "concluding torque-limited"
    )

    # Bimodality guard — multiple pooled moves with direction reversals
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


# ---------------------------------------------------------------- asymmetry
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
    # Compare magnitudes, not signed values: viscous/VFF-lag FE is +kv one
    # way and -kv the other (symmetric, NOT an asymmetry), while friction /
    # gravity produce different |FE| magnitudes per direction.
    denom = max(abs(pos_mean), abs(neg_mean), 1e-9)
    ratio = abs(abs(pos_mean) - abs(neg_mean)) / denom
    return {
        "cruise_fe_pos_dir_mean": round(pos_mean, 4),
        "cruise_fe_neg_dir_mean": round(neg_mean, 4),
        "asymmetry_ratio": round(ratio, 3),
        "significant": bool(ratio > 0.2),
        "note": "significant=true → friction/stiction, backlash, or gravity load",
    }


# ---------------------------------------------------------------- settling
def _hysteresis_flips(resid: np.ndarray, band: float) -> tuple[int, np.ndarray]:
    """Count sign flips of ``resid`` that exceed ±band (noise-immune).

    Returns (n_flips, sample_indices_of_flips).
    """
    quantized = np.zeros(len(resid), dtype=np.int8)
    quantized[resid > band] = 1
    quantized[resid < -band] = -1
    nz_idx = np.where(quantized != 0)[0]
    if nz_idx.size < 2:
        return 0, np.empty(0, dtype=int)
    nz = quantized[nz_idx]
    flip_pos = np.where(nz[:-1] != nz[1:])[0]
    return int(flip_pos.size), nz_idx[flip_pos + 1]


def _ringdown_damping(resid: np.ndarray, band: float) -> float | None:
    """Damping ratio from the decay of successive |residual| peaks.

    |resid| peaks are half a period apart, so the log decrement is twice
    the mean log ratio of consecutive peaks. Returns None when there is no
    measurable ringdown (fewer than 3 peaks above the band).
    """
    r = np.abs(resid)
    peak_amps = [
        float(r[i]) for i in range(1, len(r) - 1)
        if r[i] > band and r[i] >= r[i - 1] and r[i] > r[i + 1]
    ]
    if len(peak_amps) < 3:
        return None
    peak_amps = peak_amps[:6]
    ratios = [p0 / p1 for p0, p1 in zip(peak_amps, peak_amps[1:]) if p1 > 0]
    if not ratios:
        return None
    delta = 2.0 * float(np.mean(np.log(ratios)))
    if delta <= 0:
        return 0.0  # non-decaying → undamped / marginally stable
    return float(delta / math.sqrt(4 * math.pi ** 2 + delta ** 2))


def analyze_settling(fe: np.ndarray, t: np.ndarray, phases: dict,
                     band: float, band_source: str) -> dict:
    """Per-move settle analysis against an absolute ±band tolerance.

    Each settle window is analyzed independently (windows are already
    clipped at the next move / segment splice), then aggregated to the
    worst case across moves.
    """
    windows = [w for w in phases.get("settle_windows", []) if w[1] - w[0] >= 5]
    if not windows:
        return {}

    per_move: list[dict] = []
    for start, stop in windows:
        w = fe[start:stop]
        tw = t[start:stop]
        t0 = float(tw[0])
        tail = w[-max(1, len(w) // 4):]
        steady = float(np.mean(tail))
        resid = w - steady
        peak = float(np.max(np.abs(w)))

        # Time to enter ±band and stay inside for the rest of the window
        outside = np.abs(w) > band
        if not outside.any():
            time_to_band_ms: float | None = 0.0
            settled = True
        else:
            last_out = int(np.where(outside)[0][-1])
            if last_out == len(w) - 1:
                time_to_band_ms = None
                settled = False
            else:
                time_to_band_ms = (float(tw[last_out + 1]) - t0) * 1000.0
                settled = True

        crossings, flip_idx = _hysteresis_flips(resid, band)

        natural_freq: float | None = None
        if crossings >= 4:  # at least two full periods — fewer is noise
            flip_times = tw[flip_idx]
            half_periods = np.diff(flip_times)
            if half_periods.size and np.median(half_periods) > 0:
                natural_freq = 1.0 / (2.0 * float(np.median(half_periods)))

        per_move.append({
            "time_to_band_ms": (round(time_to_band_ms, 1)
                                if time_to_band_ms is not None else None),
            "settled": settled,
            "fe_peak": round(peak, 4),
            "fe_steady_state": round(steady, 4),
            "zero_crossings": crossings,
            "damping_ratio": _ringdown_damping(resid, band),
            "natural_freq_hz": (round(natural_freq, 1)
                                if natural_freq else None),
            "window_ms": round((float(tw[-1]) - t0) * 1000.0, 1),
        })

    settled_all = all(m["settled"] for m in per_move)
    settle_times = [m["time_to_band_ms"] for m in per_move
                    if m["time_to_band_ms"] is not None]
    worst_steady = max(per_move, key=lambda m: abs(m["fe_steady_state"]))
    zero_crossings = max(m["zero_crossings"] for m in per_move)
    dampings = [m["damping_ratio"] for m in per_move
                if m["damping_ratio"] is not None]
    freqs = [m["natural_freq_hz"] for m in per_move
             if m["natural_freq_hz"] is not None]

    return {
        "band": round(band, 6),
        "band_source": band_source,
        "n_windows": len(per_move),
        "time_to_band_ms": (round(max(settle_times), 1)
                            if settled_all and settle_times else None),
        "settled_within_window": settled_all,
        "fe_peak_during_settle": round(
            max(m["fe_peak"] for m in per_move), 4),
        "fe_steady_state": worst_steady["fe_steady_state"],
        "zero_crossings": zero_crossings,
        "ringing": bool(zero_crossings > RINGING_CROSSINGS_MAX),
        "steady_state_offset_nonzero": bool(
            abs(worst_steady["fe_steady_state"]) > band),
        "damping_ratio": (round(float(np.median(dampings)), 3)
                          if dampings else None),
        "natural_freq_hz": (round(float(np.median(freqs)), 1)
                            if freqs else None),
        "per_move": per_move[:MAX_PER_MOVE_REPORTED],
        "note": (
            "time_to_band_ms = time after move end until |FE| stays within "
            "±band; ringing counts only excursions beyond ±band"
        ),
    }


# ---------------------------------------------------------------- health
def build_health(result: dict) -> dict:
    """Loop health verdicts + human-readable issue lists from the metrics."""
    osc = result.get("oscillation", {})

    velocity_issues: list[str] = []
    vel = result.get("velocity", {})
    v_health: bool | None = None
    if vel:
        ratio = vel.get("cruise_velocity_reach_ratio")
        if ratio is not None:
            if ratio < 0.90:
                velocity_issues.append(
                    f"Velocity not reaching demand (ratio {ratio:.3f})")
            elif ratio > 1.10:
                velocity_issues.append(
                    f"Velocity exceeding demand during cruise (ratio {ratio:.3f})")
        overshoot = vel.get("velocity_overshoot_per_move", {})
        if overshoot.get("max_pct", 0.0) > 15.0:
            velocity_issues.append(
                f"Velocity overshoot {overshoot['max_pct']:.1f}% during accel (>15%)")
        vel_osc = osc.get("velocity_error", {})
        if vel_osc.get("has_significant_oscillation"):
            velocity_issues.append(
                f"Velocity-error oscillation at {vel_osc.get('dominant_hz')} Hz")
        v_health = not velocity_issues

    position_issues: list[str] = []
    settle = result.get("settle", {})
    p_health: bool | None = None
    if settle:
        band = settle.get("band", 0.0)
        if not settle.get("settled_within_window", True):
            position_issues.append(
                f"FE not settled within ±{band:g} in the post-move window")
        if settle.get("ringing"):
            position_issues.append(
                f"Post-move ringing ({settle['zero_crossings']} band crossings)")
        if settle.get("steady_state_offset_nonzero"):
            position_issues.append(
                f"Steady-state FE offset {settle['fe_steady_state']:g} exceeds "
                f"±{band:g} — insufficient integral action")
        fe_osc = osc.get("fe", {})
        if fe_osc.get("has_significant_oscillation"):
            position_issues.append(
                f"FE oscillation at {fe_osc.get('dominant_hz')} Hz")
        p_health = not position_issues

    return {
        "velocity": v_health,
        "position": p_health,
        "velocity_issues": velocity_issues,
        "position_issues": position_issues,
    }
