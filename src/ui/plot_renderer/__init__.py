"""
Plot renderer package.

PlotRenderer drives all plotting for the main window. Implementation
modules:

    controller.py       PlotRenderer class composition + render caches
    layout.py           subplot creation, plot modes, axis linking
    rendering.py        curve data updates (time/FFT/XY/3D), clear/fit
    cursors.py          C1/C2 measurement cursors and readout
    hover_overlays.py   hover crosshair, stats text, dot-detail debounce
    compare_windows.py  pop-out trace and compare windows
    traces.py           TraceControl lifecycle and reference pinning
"""

from .controller import PlotRenderer

__all__ = ["PlotRenderer"]
