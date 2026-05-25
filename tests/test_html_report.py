import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.reports.html_report import build_html_report, write_html_report


class HtmlReportTests(unittest.TestCase):
    def test_report_contains_core_sections_and_escaped_notes(self):
        fs = 1000.0
        time_arr = np.arange(0.0, 1.0, 1.0 / fs)
        params = {
            "MPOS(0)": np.sin(2.0 * np.pi * 20.0 * time_arr),
            "FE(0)": 0.1 * np.sin(2.0 * np.pi * 50.0 * time_arr),
        }

        html = build_html_report(
            time_arr=time_arr,
            params=params,
            trace_order=["FE(0)", "MPOS(0)"],
            trace_colors={"FE(0)": "#ff0000", "MPOS(0)": "#00ff00"},
            trace_fft_flags={"FE(0)": True},
            controller_metadata={"Controller IP": "192.168.0.245"},
            drive_metadata={"Drive Model": "DX4"},
            drive_profiles={0: {"drive_type": "DX4", "pn102": 600}},
            user_notes="<script>alert('x')</script>\nAxis 0 checked",
            segment_breaks=[500],
        )

        self.assertIn("TrioScope Commissioning Report", html)
        self.assertIn("Controller Metadata", html)
        self.assertIn("Controller IP", html)
        self.assertIn("Drive Metadata", html)
        self.assertIn("Drive Parameters", html)
        self.assertIn("Pn102 Speed Loop Gain", html)
        self.assertIn("Measurement Table", html)
        self.assertIn("Plots and FFT Peaks", html)
        self.assertIn("FE(0)", html)
        self.assertIn("MPOS(0)", html)
        self.assertIn("&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert", html)
        self.assertLess(html.find("FE(0)"), html.find("MPOS(0)"))

    def test_write_html_report_creates_parent_and_adds_extension(self):
        time_arr = np.array([0.0, 0.001, 0.002, 0.003])
        params = {"MPOS(0)": np.array([0.0, 1.0, 0.0, -1.0])}

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "report"
            written = write_html_report(target, time_arr=time_arr, params=params)

            self.assertEqual(written.suffix, ".html")
            self.assertTrue(written.exists())
            self.assertIn("MPOS(0)", written.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

