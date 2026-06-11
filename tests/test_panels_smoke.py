"""Construction smoke tests for the AI dock panels.

These panels have no behavioural unit tests; building them offscreen
catches import errors, missing attributes, and broken signal wiring
introduced by refactors.
"""

import sys
import unittest

from PySide6.QtWidgets import QApplication

# Create QApplication if it doesn't exist (needed for Qt widgets)
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from src.ai.analysis_panel import AIAnalysisPanel
from src.ai.drive_profile_editor import DriveProfileEditor
from src.ai.tuner_panel import TunerPanel


class TunerPanelSmokeTests(unittest.TestCase):
    def test_constructs_and_exposes_profile_api(self):
        panel = TunerPanel()
        self.assertEqual(panel.get_all_profiles(), {})

        panel.set_all_profiles({0: {"drive_type": "DX4", "pn102": 500}})
        profiles = panel.get_all_profiles()
        self.assertIn(0, profiles)
        self.assertEqual(profiles[0]["drive_type"], "DX4")
        self.assertEqual(profiles[0]["pn102"], 500)

        panel.set_connection(None)
        panel.deleteLater()

    def test_analyze_without_provider_sets_status(self):
        panel = TunerPanel()
        panel._on_analyze()
        self.assertIn("No data provider", panel._status_label.text())
        panel.deleteLater()

    def test_analyze_without_capture_sets_status(self):
        panel = TunerPanel()
        panel.set_data_provider(lambda: None)
        panel._on_analyze()
        self.assertIn("No captured data", panel._status_label.text())
        panel.deleteLater()


class AIAnalysisPanelSmokeTests(unittest.TestCase):
    def test_constructs_and_round_trips_profiles(self):
        panel = AIAnalysisPanel()
        panel.set_all_profiles({1: {"drive_type": "DX3", "pn104": 40}})
        profiles = panel.get_all_profiles()
        self.assertEqual(profiles[1]["drive_type"], "DX3")
        self.assertEqual(profiles[1]["pn104"], 40)

        panel.set_connection(None)
        self.assertIsNone(panel._get_scope_context())
        panel.deleteLater()

    def test_drive_context_empty_without_profile(self):
        panel = AIAnalysisPanel()
        self.assertEqual(panel._get_drive_context(), "")
        panel.deleteLater()


class DriveProfileEditorTests(unittest.TestCase):
    def test_axis_switch_keeps_profiles_separate(self):
        editor = DriveProfileEditor()
        editor.set_all_profiles({
            0: {"drive_type": "DX4", "pn102": 111},
            1: {"drive_type": "DX3", "pn102": 222},
        })

        editor._axis_combo.setCurrentIndex(0)
        self.assertEqual(editor.current_axis(), 0)
        self.assertEqual(editor.current_profile().pn102, 111)

        editor._axis_combo.setCurrentIndex(1)
        self.assertEqual(editor.current_axis(), 1)
        self.assertEqual(editor.current_profile().pn102, 222)
        editor.deleteLater()

    def test_buttons_disabled_without_connection(self):
        editor = DriveProfileEditor(autowrite=True)
        self.assertFalse(editor._read_btn.isEnabled())
        self.assertFalse(editor._write_btn.isEnabled())
        self.assertIsNotNone(editor._autowrite_chk)
        self.assertFalse(editor._autowrite_chk.isEnabled())
        editor.deleteLater()


if __name__ == "__main__":
    unittest.main()
