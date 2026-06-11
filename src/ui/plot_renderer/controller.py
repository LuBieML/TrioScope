"""PlotRenderer class composition and controller-local render caches."""

from ui.window_controller import WindowBackedController

from .compare_windows import CompareWindowsMixin
from .cursors import CursorsMixin
from .hover_overlays import HoverOverlaysMixin
from .layout import PlotLayoutMixin
from .rendering import RenderingMixin
from .traces import TraceManagementMixin


class PlotRenderer(
    PlotLayoutMixin,
    HoverOverlaysMixin,
    CompareWindowsMixin,
    CursorsMixin,
    TraceManagementMixin,
    RenderingMixin,
    WindowBackedController,
):
    """Owns subplot layout, curve rendering, cursors, hover, and compare windows.

    Shared UI state lives on the main window (WindowBackedController proxies
    attribute access); only the render caches below are controller-local.
    """

    _local_attrs = WindowBackedController._local_attrs | frozenset({
        "_fft_cache", "_fft_window_cache", "_fft_dirty", "_fft_peak_cache",
        "_fft_max_samples", "_last_data_len", "_stats_cache", "_ref_set",
        "_stats_pos_cache", "_last_render_data_len", "_stats_reposition_scheduled",
        "_pending_stats_vbs", "_pending_stats_vb_refs", "_detail_update_scheduled",
        "_pending_detail_vbs", "_pending_detail_vb_refs", "_hover_vlines",
        "_hover_labels", "_last_freqs", "_hover_pending_pos",
        "_hover_update_scheduled",
    })

    def __init__(self, window):
        super().__init__(window)
        self._fft_cache = {}
        self._fft_window_cache = (0, None)
        self._fft_dirty = True
        self._fft_peak_cache = {}
        self._fft_max_samples = 16384
        self._last_data_len = 0
        self._stats_cache = {}
        self._ref_set = {}
        self._stats_pos_cache = {}
        self._last_render_data_len = 0
        self._stats_reposition_scheduled = False
        self._pending_stats_vbs = set()
        self._pending_stats_vb_refs = {}
        self._detail_update_scheduled = False
        self._pending_detail_vbs = set()
        self._pending_detail_vb_refs = {}
        self._hover_vlines = {}
        self._hover_labels = {}
        self._last_freqs = None
        self._hover_pending_pos = None
        self._hover_update_scheduled = False
