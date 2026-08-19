"""Small, Qt-free helpers for snapping path hover readouts to samples."""

from __future__ import annotations

import numpy as np


PATH_HOVER_DISTANCE_PX = 14.0


def nearest_xy_point_index(
    x_values,
    y_values,
    mouse_x: float,
    mouse_y: float,
    x_units_per_pixel: float,
    y_units_per_pixel: float,
    max_distance_px: float = PATH_HOVER_DISTANCE_PX,
) -> int | None:
    """Return the sample nearest the mouse in screen-space, if close enough."""
    x = np.asarray(x_values)
    y = np.asarray(y_values)
    if x.ndim != 1 or y.ndim != 1 or len(x) == 0 or len(x) != len(y):
        return None
    if x_units_per_pixel <= 0 or y_units_per_pixel <= 0:
        return None

    dx = (x - mouse_x) / x_units_per_pixel
    dy = (y - mouse_y) / y_units_per_pixel
    distances_sq = dx * dx + dy * dy
    finite = np.isfinite(distances_sq)
    if not finite.any():
        return None

    candidates = np.where(finite, distances_sq, np.inf)
    index = int(np.argmin(candidates))
    if candidates[index] > max_distance_px * max_distance_px:
        return None
    return index


def block_screen_distances_sq(projected_corners, mouse_x: float, mouse_y: float) -> np.ndarray:
    """Squared screen distance from the cursor to each block's bounding box.

    Takes the projected ``(m, 8, 2)`` screen positions of each block's bounding
    box corners. A box lying entirely in front of the camera projects inside the
    convex hull of its projected corners, so this distance is a lower bound on
    the distance to any sample in the block -- a block farther than the hover
    radius cannot contain a hit and can be skipped without touching its samples.
    A block with any corner behind the camera has no valid screen bounds, and is
    reported at distance zero so it is never wrongly skipped.
    """
    corners = np.asarray(projected_corners, dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (8, 2) or len(corners) == 0:
        return np.zeros(len(corners), dtype=float)

    finite = np.isfinite(corners).all(axis=2)
    unprojectable = ~finite.all(axis=1)
    safe = np.where(finite[:, :, None], corners, [mouse_x, mouse_y])
    x_min = safe[:, :, 0].min(axis=1)
    x_max = safe[:, :, 0].max(axis=1)
    y_min = safe[:, :, 1].min(axis=1)
    y_max = safe[:, :, 1].max(axis=1)
    zero = np.zeros(len(corners))
    dx = np.maximum.reduce([x_min - mouse_x, zero, mouse_x - x_max])
    dy = np.maximum.reduce([y_min - mouse_y, zero, mouse_y - y_max])
    return np.where(unprojectable, 0.0, dx * dx + dy * dy)


def nearest_projected_point_index(
    projected_points,
    mouse_x: float,
    mouse_y: float,
    max_distance_px: float = PATH_HOVER_DISTANCE_PX,
) -> int | None:
    """Return the nearest valid Nx2 projected point within a pixel radius."""
    points = np.asarray(projected_points)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return None

    distances_sq = (
        (points[:, 0] - mouse_x) ** 2
        + (points[:, 1] - mouse_y) ** 2
    )
    finite = np.isfinite(distances_sq)
    if not finite.any():
        return None

    candidates = np.where(finite, distances_sq, np.inf)
    index = int(np.argmin(candidates))
    if candidates[index] > max_distance_px * max_distance_px:
        return None
    return index
