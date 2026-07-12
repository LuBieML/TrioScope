"""Offscreen smoke tests for the Servo Loop Analyser panel."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import QApplication, QSpinBox

from src.ai.tuner_panel import TunerPanel
from src.ai.tuning_history import KPI_DEFS
from src.ai.zn_calculator import zn_pi_table

FS = 1000.0


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _capture(axis=0, fe_sigma=0.002, seed=1):
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
    rng = np.random.default_rng(seed)
    params = {
        f"DPOS({axis})": dpos,
        f"DRIVE_FE({axis})": rng.normal(0, fe_sigma, len(t)),
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
    metrics = panel.last_metrics()
    assert metrics is not None
    assert metrics["channels_detected"]["dpos"] == "DPOS(3)"


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
    metrics = panel.last_metrics()
    assert metrics is not None
    assert metrics["data_sufficiency"] == "OK"


def test_zn_pi_table_classical_values():
    rows = zn_pi_table(ku=500.0, tu_s=0.010)
    classical = rows[0]
    assert classical["method"] == "Classical ZN"
    assert classical["kp"] == pytest.approx(225.0)
    assert classical["pn103"] == pytest.approx(83.3, abs=0.1)
    # Invalid inputs produce empty rows, not crashes
    assert zn_pi_table(0.0, 0.010)[0]["kp"] is None


# ---------------------------------------------------------------------------
# Tuning history integration
# ---------------------------------------------------------------------------
_CHANGES_COL = 2 + len(KPI_DEFS)


def test_history_records_each_analyze(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    panel._on_analyze()
    panel._on_analyze()

    assert len(panel._history) == 2
    assert panel._history_card._table.rowCount() == 2
    assert "Run 2" in panel._status_label.text()
    # Insufficient captures must not be recorded
    panel.set_data_provider(lambda: (None, None, None, None))
    panel._on_analyze()
    assert len(panel._history) == 2


def test_history_marks_worse_run_red(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(lambda: _capture(fe_sigma=0.002))
    panel._on_analyze()
    panel.set_data_provider(lambda: _capture(fe_sigma=0.05, seed=2))
    panel._on_analyze()

    table = panel._history_card._table
    cruise_col = 2 + next(
        i for i, kpi in enumerate(KPI_DEFS) if kpi.key == "fe_cruise_rms")
    newest_cell = table.item(0, cruise_col)
    assert newest_cell is not None
    assert "▲" in newest_cell.text()                      # value increased
    assert newest_cell.foreground().color().name() == "#e74c3c"  # RED = worse


def test_history_shows_pn_changes_between_runs(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    assert panel._drive_combo is not None
    panel._drive_combo.setCurrentText("DX4")   # loads defaults (Pn102 = 500)
    panel._on_analyze()
    pn102_spin = panel._param_widgets["pn102"]
    assert isinstance(pn102_spin, QSpinBox)
    pn102_spin.setValue(600)
    panel._on_analyze()

    changes_cell = panel._history_card._table.item(0, _CHANGES_COL)
    assert changes_cell is not None
    assert "Pn102 500→600" in changes_cell.text()
    baseline_cell = panel._history_card._table.item(1, _CHANGES_COL)
    assert baseline_cell is not None
    assert baseline_cell.text() == "baseline"


def test_history_row_click_recalls_run(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    panel._on_analyze()
    panel._on_analyze()

    panel._history_card._on_row_clicked(1, 0)  # older run (newest first)
    assert "Viewing run" in panel._status_label.text()
    metrics = panel.last_metrics()
    assert metrics is not None
    assert metrics["data_sufficiency"] == "OK"


def test_history_csv_export_content(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    panel._on_analyze()
    csv_text = panel._history.to_csv()
    assert "timestamp" in csv_text.splitlines()[0]
    assert "score" in csv_text.splitlines()[0]
    assert len(csv_text.strip().splitlines()) == 2


# ---------------------------------------------------------------------------
# Offline recommendations
# ---------------------------------------------------------------------------
def _card_texts(card):
    texts = []
    for i in range(card.rows.count()):
        widget = card.rows.itemAt(i).widget()
        if widget is not None:
            texts.append(widget.text())
    return texts


def test_recommendations_card_reports_well_tuned_axis(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    panel._on_analyze()

    assert "/10" in panel._rec_card.status_lbl.text()
    assert any("well tuned" in t for t in _card_texts(panel._rec_card))


def test_recommendations_card_flags_vff_defect(qt_app):
    def provider():
        t, params, sp, breaks = _capture()
        dvel = np.gradient(params["DPOS(0)"], t)
        params["DRIVE_FE(0)"] = params["DRIVE_FE(0)"] + 0.002 * dvel
        # Second slower move gives the FE-vs-velocity fit its speed spread
        params2 = {k: v for k, v in params.items()}
        dvel2 = dvel * 0.5
        dpos2 = np.cumsum(dvel2) / FS
        t2 = t[-1] + (np.arange(len(t)) + 1) / FS
        full_t = np.concatenate([t, t2])
        full = {
            "DPOS(0)": np.concatenate([params["DPOS(0)"],
                                       params["DPOS(0)"][-1] + dpos2]),
            "DRIVE_FE(0)": np.concatenate([
                params["DRIVE_FE(0)"],
                params2["DRIVE_FE(0)"] * 0 + 0.002 * dvel2]),
        }
        return full_t, full, sp, breaks

    panel = TunerPanel()
    panel.set_data_provider(provider)
    panel._on_analyze()

    texts = " ".join(_card_texts(panel._rec_card))
    assert "VFF_GAIN" in texts or "Pn112" in texts


def test_history_table_tracks_score_column(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    panel._on_analyze()

    table = panel._history_card._table
    header = table.horizontalHeaderItem(2)
    assert header is not None and "Score" in header.text()
    score_cell = table.item(0, 2)
    assert score_cell is not None
    assert float(score_cell.text().split()[0]) >= 8.0
