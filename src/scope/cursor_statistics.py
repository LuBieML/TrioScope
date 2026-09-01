"""Qt-free statistics for samples selected by the plot cursors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class CursorRangeStatistics:
    """Sample count and per-signal means inside an inclusive time range."""

    sample_count: int
    means: dict[str, Optional[float]]


def calculate_cursor_range_statistics(
    time_values,
    parameters: Mapping[str, object],
    cursor_1: float,
    cursor_2: float,
) -> CursorRangeStatistics:
    """Calculate arithmetic means for samples between two cursor positions.

    Both cursor bounds are inclusive. Cursor order does not matter. Non-finite
    signal values are excluded from the mean so one missing sample does not
    hide the statistics for the rest of the selected range.
    """

    time_arr = np.asarray(time_values)
    names = list(parameters)
    empty_means = {name: None for name in names}
    if time_arr.ndim != 1 or time_arr.size == 0:
        return CursorRangeStatistics(0, empty_means)

    range_start, range_end = sorted((float(cursor_1), float(cursor_2)))
    start = int(np.searchsorted(time_arr, range_start, side="left"))
    stop = int(np.searchsorted(time_arr, range_end, side="right"))
    start = max(0, min(start, len(time_arr)))
    stop = max(start, min(stop, len(time_arr)))

    means: dict[str, Optional[float]] = {}
    for name, raw_values in parameters.items():
        values = np.asarray(raw_values)
        usable_stop = min(stop, len(values)) if values.ndim == 1 else start
        if usable_stop <= start:
            means[name] = None
            continue
        selected = np.asarray(values[start:usable_stop], dtype=float)
        finite = selected[np.isfinite(selected)]
        means[name] = float(np.mean(finite)) if finite.size else None

    return CursorRangeStatistics(stop - start, means)

