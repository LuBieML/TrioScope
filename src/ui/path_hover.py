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
