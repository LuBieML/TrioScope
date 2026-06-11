"""Offscreen construction smoke test for the full main window.

Guards the controller composition (connection/capture/plot/actions
packages) and the window-proxy binding: every bound method must resolve
and the window must build, render, and switch modes without a live
controller connection.
"""

import sys
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

# Create QApplication if it doesn't exist (needed for Qt widgets)
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from src.ui.main_window import ParameterScopeOscilloscope
from src.ui.main_window_bindings import (
    ACTION_METHODS, CAPTURE_METHODS, CONNECTION_METHODS, PLOT_METHODS,
)


class MainWindowSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window = ParameterScopeOscilloscope()

    @classmethod
    def tearDownClass(cls):
        cls.window.deleteLater()

    def test_all_bound_methods_resolve(self):
        for name in (CONNECTION_METHODS + CAPTURE_METHODS
                     + PLOT_METHODS + ACTION_METHODS):
            self.assertTrue(callable(getattr(self.window, name)), name)

    def test_render_and_trace_ops(self):
        w = self.window
        trace_name = w.traces[0].get_display_name()
        w.accumulated_data = {
            'time': np.linspace(0.0, 1.0, 100),
            'num_samples': 100,
            'params': {trace_name: np.sin(np.linspace(0.0, 6.0, 100))},
            'segment_breaks': [],
        }
        w._recreate_subplots()
        w._render_plots()
        self.assertTrue(w.plot_items)

        n_before = len(w.traces)
        w.add_trace()
        self.assertEqual(len(w.traces), n_before + 1)
        w.on_trace_changed()

    def test_cursors_and_plot_modes(self):
        w = self.window
        w._toggle_cursors(True)
        self.assertTrue(w._cursors_enabled)
        w._toggle_cursors(False)
        self.assertFalse(w._cursors_enabled)

        w._on_plot_mode_changed(1)  # XY
        self.assertEqual(w.plot_mode, 'xy')
        w._on_plot_mode_changed(0)  # time
        self.assertEqual(w.plot_mode, 'time')


if __name__ == "__main__":
    unittest.main()
