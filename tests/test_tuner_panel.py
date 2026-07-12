"""Offscreen smoke tests for the Servo Loop Analyser panel."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import QApplication

from src.ai.tuner_panel import TunerPanel
from src.ai.zn_calculator import zn_pi_table

FS = 1000.0


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _capture(axis=0):
    """A clean single-move capture as the provider 4-tuple."""
    n_pre, n_acc, n_cr, n_post = 300, 200, 600, 800
    dvel = np.concatenate([
        np.zeros(n_pre),
        np.linspace(0.0, 100.0, n_acc, endpoint=False),
        np.full(n_cr, 100.0),
        np.linspace(100.0, 0.0, n_acc, endpoint=False),
        np.zeros(n_post),
    ])
    t = np.arange(len(dvel)) / FS
    dpos = np.cumsum(dvel) / FS
    rng = np.random.default_rng(1)
    params = {
        f"DPOS({axis})": dpos,
        f"DRIVE_FE({axis})": rng.normal(0, 0.002, len(t)),
        f"MSPEED({axis})": dvel + rng.normal(0, 0.05, len(t)),
    }
    return t, params, 0.001, []


def test_analyze_populates_cards_and_status(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    panel._on_analyze()

    assert "Axis 0" in panel._status_label.text()
    metrics = panel.last_metrics()
    assert metrics is not None
    assert metrics["data_sufficiency"] == "OK"
    assert metrics["health"]["position"] is True
    # Cards got populated (rows exist beyond the reset placeholders)
    assert panel._pos_card.rows.count() > 0
    assert panel._vel_card.rows.count() > 0
    assert panel._fe_card.rows.count() > 0


def test_analyze_uses_only_captured_axis_when_unambiguous(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(lambda: _capture(axis=3))
    # Profile combo sits at axis 0, but only axis 3 was captured
    panel._on_analyze()
    assert "Axis 3" in panel._status_label.text()
    assert panel.last_metrics()["channels_detected"]["dpos"] == "DPOS(3)"


def test_analyze_demands_choice_for_multi_axis_capture(qt_app):
    def provider():
        t, params, sp, breaks = _capture(axis=1)
        _, params2, _, _ = _capture(axis=2)
        params.update(params2)
        return t, params, sp, breaks

    panel = TunerPanel()
    panel.set_data_provider(provider)
    panel._on_analyze()  # combo at axis 0, capture has axes 1 and 2
    assert panel.last_metrics() is None
    assert "pick one" in panel._status_label.text()


def test_analyze_without_capture_shows_hint(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(lambda: (None, None, None, None))
    panel._on_analyze()
    assert "run a capture" in panel._status_label.text()


def test_old_three_tuple_provider_still_works(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(lambda: _capture()[:3])
    panel._on_analyze()
    assert panel.last_metrics()["data_sufficiency"] == "OK"


def test_zn_pi_table_classical_values():
    rows = zn_pi_table(ku=500.0, tu_s=0.010)
    classical = rows[0]
    assert classical["method"] == "Classical ZN"
    assert classical["kp"] == pytest.approx(225.0)
    assert classical["pn103"] == pytest.approx(83.3, abs=0.1)
    # Invalid inputs produce empty rows, not crashes
    assert zn_pi_table(0.0, 0.010)[0]["kp"] is None
