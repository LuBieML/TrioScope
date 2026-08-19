"""Qt-free geometry helpers for reducing long 3-D paths to drawable vertex sets.

A multi-hour capture holds tens of millions of samples, far more than can be
uploaded to the GPU every frame. Almost all of that data is redundant: a machine
path spends most of its samples on straight travel and dwell, where hundreds of
consecutive samples sit on the same line. Simplifying by *geometry* rather than
by sample index removes exactly those samples and keeps every corner, so the
drawn path is unchanged to the eye while the vertex count collapses.
"""

from __future__ import annotations

import numpy as np

# Give up on Douglas-Peucker when fewer than this fraction of interior
# points sit within epsilon of their neighbour chord -- the run is noise
# and the algorithm would keep almost every sample.
MIN_COLLINEAR_FRACTION = 0.2
# Skip the collinearity screen on short runs; the DP loop is cheap there.
SCREEN_THRESHOLD = 512
# Cap the number of split vertices at n // MAX_KEPT_FRACTION. Above that
# the run is kept whole so the epsilon guarantee still holds.
MAX_KEPT_FRACTION = 32
MIN_SPLIT_BUDGET = 32


def neighbour_chord_distances(points) -> np.ndarray:
    """Distance of each interior point from the chord joining its neighbours.

    A cheap, fully vectorised proxy for how much of a run is redundant.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.empty(0)

    before = pts[:-2]
    middle = pts[1:-1]
    after = pts[2:]
    chords = after - before
    offsets = middle - before
    chord_sq = np.einsum("ij,ij->i", chords, chords)
    safe = np.where(chord_sq > 0, chord_sq, 1.0)
    t = np.clip(np.einsum("ij,ij->i", offsets, chords) / safe, 0.0, 1.0)
    perp = offsets - t[:, None] * chords
    return np.sqrt(np.einsum("ij,ij->i", perp, perp))


def simplify_indices(points, epsilon: float) -> np.ndarray:
    """Return indices of the vertices to keep (Douglas-Peucker).

    Every discarded point lies within ``epsilon`` of the retained polyline, so
    the reduction is invisible at any zoom level where ``epsilon`` is below one
    pixel. The first and last vertex are always kept, which lets adjacent chunks
    be simplified independently and still join without a seam.

    Implemented with an explicit stack (not recursion) so a pathological path
    cannot overflow the interpreter stack, and with the per-segment distance
    search vectorised so the Python-level loop runs once per *kept* vertex
    rather than once per sample.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3 or epsilon <= 0:
        return np.arange(n)

    if n > SCREEN_THRESHOLD:
        near = neighbour_chord_distances(pts) <= epsilon
        if near.mean() < MIN_COLLINEAR_FRACTION:
            return np.arange(n)

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[n - 1] = True
    eps_sq = float(epsilon) * float(epsilon)
    budget = max(MIN_SPLIT_BUDGET, n // MAX_KEPT_FRACTION)
    kept = 2
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        anchor = pts[start]
        chord = pts[end] - anchor
        offsets = pts[start + 1:end] - anchor
        chord_sq = float(chord @ chord)
        if chord_sq > 0.0:
            t = np.clip((offsets @ chord) / chord_sq, 0.0, 1.0)
            perp = offsets - t[:, None] * chord
        else:
            perp = offsets
        dist_sq = np.einsum("ij,ij->i", perp, perp)
        worst = int(np.argmax(dist_sq))
        if dist_sq[worst] > eps_sq:
            kept += 1
            if kept > budget:
                return np.arange(n)
            split = start + 1 + worst
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))
    return np.flatnonzero(keep)


def decimate_to_budget(points, max_vertices: int) -> np.ndarray:
    """Force a run down to ``max_vertices``, keeping the most significant ones.

    A fallback for paths :func:`simplify_indices` cannot compress. It repeatedly
    drops the vertices that deviate least from the chord through their
    neighbours, which is fully vectorised -- unlike Douglas-Peucker, whose cost
    scales with the number of vertices *kept* and so is unusable when the result
    is still large.

    Unlike :func:`simplify_indices` this gives no epsilon guarantee: error can
    accumulate across passes. It exists to bound memory on a path that cannot be
    simplified honestly, and callers should report when they resort to it.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= max_vertices or max_vertices < 2:
        return np.arange(n)

    keep = np.arange(n)
    while len(keep) > max_vertices:
        current = pts[keep]
        distances = neighbour_chord_distances(current)
        if len(distances) == 0:
            return keep
        excess = len(keep) - max_vertices
        wanted = min(len(distances) - 1, max(1, excess * 2))
        threshold = np.partition(distances, wanted)[wanted]
        drop = distances <= threshold
        # Never drop the endpoints of the current run.
        drop = drop & ~np.concatenate([[False], drop[:-1]])
        if not drop.any():
            return keep
        mask = np.ones(len(current), dtype=bool)
        mask[1:-1] = ~drop
        keep = keep[mask]
    return keep


def block_bounds(points, block_size: int) -> np.ndarray:
    """Axis-aligned bounds of each consecutive ``block_size`` run of points.

    Returns an ``(m, 2, 3)`` array of per-block ``(min, max)`` corners, used to
    cull most of a long path before doing any per-point work.
    """
    pts = np.asarray(points)
    n = len(pts)
    if n == 0:
        return np.empty((0, 2, 3), dtype=np.float32)

    full = n // block_size
    bounds = []
    if full:
        square = pts[:full * block_size].reshape(full, block_size, 3)
        bounds.append(np.stack([square.min(axis=1), square.max(axis=1)], axis=1))
    if n % block_size:
        rest = pts[full * block_size:]
        bounds.append(np.stack([rest.min(axis=0), rest.max(axis=0)], axis=0)[None, :, :])
    return np.concatenate(bounds).astype(np.float32, copy=False)


def aabb_corners(bounds) -> np.ndarray:
    """Expand ``(m, 2, 3)`` min/max bounds into their ``(m, 8, 3)`` corners."""
    box = np.asarray(bounds, dtype=np.float32)
    if len(box) == 0:
        return np.empty((0, 8, 3), dtype=np.float32)

    picks = np.array(
        [[(corner >> axis) & 1 for axis in range(3)] for corner in range(8)],
        dtype=np.intp,
    )
    blocks = np.arange(len(box), dtype=np.intp).reshape(-1, 1, 1)
    axes = np.arange(3, dtype=np.intp).reshape(1, 1, 3)
    return box[blocks, picks.reshape(1, 8, 3), axes]
