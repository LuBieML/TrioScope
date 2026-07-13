"""Segment- and move-aware phase segmentation for scope captures.

Continuous-mode captures accumulate several trigger events into one buffer
with ``segment_breaks`` marking the splice points. The segments are separate
real-world recordings: position may jump across a splice, so gradients,
reversal detection, move runs, and settle windows must never cross one.
"""

from __future__ import annotations

import numpy as np

from .signal_constants import (
    EPS_VEL_FRAC, EPS_ACC_FRAC, SETTLE_MS, REVERSAL_HALF_WIDTH_S,
)


def segment_bounds(n: int, segment_breaks=None) -> list[tuple[int, int]]:
    """[(start, stop), ...] half-open segment slices covering 0..n."""
    breaks = sorted({int(b) for b in (segment_breaks or ()) if 0 < int(b) < n})
    edges = [0, *breaks, n]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def per_segment_gradient(y: np.ndarray, t: np.ndarray,
                         bounds: list[tuple[int, int]]) -> np.ndarray:
    """np.gradient computed independently per segment (no splice spikes)."""
    g = np.zeros(len(y), dtype=np.float64)
    for a, b in bounds:
        if b - a >= 2:
            g[a:b] = np.gradient(y[a:b], t[a:b])
    return g


def median_dt(t: np.ndarray, bounds: list[tuple[int, int]]) -> float:
    """Median sample interval, ignoring cross-segment time steps."""
    diffs = [np.diff(t[a:b]) for a, b in bounds if b - a >= 2]
    if not diffs:
        return 0.0
    d = np.concatenate(diffs)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else 0.0


def contiguous_runs(mask: np.ndarray,
                    bounds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """True-runs of ``mask`` as (start, stop) half-open slices.

    Runs are split at segment boundaries so no run spans a splice.
    """
    runs: list[tuple[int, int]] = []
    for a, b in bounds:
        m = mask[a:b]
        if not m.any():
            continue
        d = np.diff(m.astype(np.int8))
        starts = np.where(d == 1)[0] + 1
        stops = np.where(d == -1)[0] + 1
        if m[0]:
            starts = np.concatenate(([0], starts))
        if m[-1]:
            stops = np.concatenate((stops, [b - a]))
        runs.extend((a + int(s), a + int(e)) for s, e in zip(starts, stops))
    return runs


def segment_phases(t: np.ndarray, dvel: np.ndarray, dt: float,
                   bounds: list[tuple[int, int]]) -> dict:
    """Classify every sample into motion phases from the demand velocity.

    Returns masks (idle/accel/cruise/decel/settle/moving/reversal), the
    per-move runs and per-move settle windows (half-open slices), and
    counts. Settle windows are clipped at the segment end and at the next
    move start, so back-to-back moves never leak accel samples into
    settle statistics.
    """
    n = len(dvel)
    dacc = per_segment_gradient(dvel, t, bounds)

    v_max = float(np.max(np.abs(dvel)))
    a_max = float(np.max(np.abs(dacc)))
    v_thresh = EPS_VEL_FRAC * v_max if v_max > 0 else 1e-9
    a_thresh = EPS_ACC_FRAC * a_max if a_max > 0 else 1e-9

    # --- Reversals: true direction changes of the demand, per segment ---
    # Only samples moving meaningfully (above threshold) define direction,
    # and consecutive opposite directions must be close together. A move
    # that merely starts or stops at rest is NOT a reversal — treating the
    # 0→v / v→0 edges as reversals blankets ±80 ms of every move boundary
    # and eats short moves entirely.
    reversal = np.zeros(n, dtype=bool)
    n_reversals = 0
    half_width = max(1, int(round(REVERSAL_HALF_WIDTH_S / dt))) if dt > 0 else 1
    max_gap = 2 * half_width
    for a, b in bounds:
        v = dvel[a:b]
        if len(v) < 2:
            continue
        above_idx = np.nonzero(np.abs(v) > v_thresh)[0]
        if above_idx.size < 2:
            continue
        directions = np.sign(v[above_idx])
        flips = np.nonzero(directions[:-1] * directions[1:] < 0)[0]
        for f in flips:
            i_prev = int(above_idx[f])
            i_next = int(above_idx[f + 1])
            if i_next - i_prev > max_gap:
                continue  # a stop between opposite moves, not a reversal
            n_reversals += 1
            center = (i_prev + i_next) // 2
            lo = max(a, a + center - half_width + 1)
            hi = min(b, a + center + 1 + half_width)
            reversal[lo:hi] = True

    # --- Phase masks (reversal excluded from everything) ---
    kinematic_idle = np.abs(dvel) <= v_thresh
    moving = ~kinematic_idle & ~reversal
    accel = moving & (dacc > a_thresh)
    decel = moving & (dacc < -a_thresh)
    cruise = moving & ~accel & ~decel
    idle = kinematic_idle & ~reversal

    moves = contiguous_runs(moving, bounds)

    # --- Settle windows: after each move end, clipped so they never
    # include the next move or cross a segment splice ---
    settle = np.zeros(n, dtype=bool)
    settle_samples = max(1, int(round(SETTLE_MS * 1e-3 / dt))) if dt > 0 else 1
    for i, (_, move_stop) in enumerate(moves):
        seg_stop = next(b for a, b in bounds if a <= move_stop - 1 < b)
        stop = min(move_stop + settle_samples, seg_stop)
        for next_start, _ in moves[i + 1:]:
            if next_start >= move_stop:
                stop = min(stop, next_start)
                break
        if stop > move_stop:
            settle[move_stop:stop] = True
    settle &= ~reversal & ~moving
    idle &= ~settle

    settle_windows = contiguous_runs(settle, bounds)

    return {
        "idle": idle, "accel": accel, "cruise": cruise, "decel": decel,
        "settle": settle, "moving": moving, "reversal": reversal,
        "moves": moves, "settle_windows": settle_windows,
        "n_moves": len(moves), "n_reversals": n_reversals,
    }
