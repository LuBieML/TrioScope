"""Offscreen smoke tests for the Servo Loop Analyser panel."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import QApplication, QMainWindow, QSpinBox

from src.ai.tuner_panel import TunerPanel
from src.ai.tuner_theme import RED
from src.ai.tuning_history import KPI_DEFS
from src.ai.zn_calculator import zn_pi_table
from src.ai.inertia_estimator import RAW_CURRENT
from src.ai.motor_parameters import DetectedMotorParameters
from src.models.motion_axis_command import MotionAxisCommand

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


def test_tuner_is_a_separate_resizable_window(qt_app):
    main_window = QMainWindow()
    main_window.resize(900, 600)
    original_size = main_window.size()

    panel = TunerPanel(main_window)
    panel.show()
    qt_app.processEvents()

    assert panel.isWindow()
    assert panel.parent() is main_window
    assert main_window.size() == original_size
    assert panel.size().width() >= 1000
    assert panel.minimumWidth() < panel.width()
    panel.hide()


def test_workspace_is_grouped_into_task_tabs(qt_app):
    panel = TunerPanel()

    assert panel._workspace_tabs.count() == 3
    assert [panel._workspace_tabs.tabText(i) for i in range(3)] == [
        "TUNE & ANALYZE",
        "MOTION & INERTIA",
        "HISTORY",
    ]
    assert panel._workspace_tabs.currentWidget() is panel._analysis_tab

    panel.focus_motion()

    assert panel._workspace_tabs.currentWidget() is panel._motion_tab
    assert panel.motion_panel.minimumWidth() >= 330
    assert panel.inertia_card.minimumWidth() >= 400


def test_tuner_close_hides_without_losing_state(qt_app):
    panel = TunerPanel()
    panel.set_data_provider(_capture)
    panel._on_analyze()
    metrics = panel.last_metrics()
    panel.show()
    qt_app.processEvents()

    panel.close()
    qt_app.processEvents()

    assert not panel.isVisible()
    assert panel.last_metrics() is metrics


def test_tuner_combines_drive_profile_and_single_axis_motion(qt_app):
    panel = TunerPanel()

    assert panel.motion_panel is not None
    panel._axis_combo.setCurrentText("4")
    assert panel.motion_panel.axis == 4

    panel.motion_panel.distance_edit.setValue(80.0)
    panel.motion_panel.speed_edit.setValue(125.0)
    panel.motion_panel.acceleration_edit.setValue(750.0)
    assert panel.motion_panel.commands() == [
        MotionAxisCommand(
            axis=4,
            distance=80.0,
            speed=125.0,
            acceleration=750.0,
        )
    ]

    panel.motion_panel.axis_combo.setCurrentIndex(2)
    assert panel._current_axis() == 2


def test_motor_acceleration_is_derived_from_motion_and_encoder_resolution(qt_app):
    panel = TunerPanel()
    panel.set_axis_units_provider(lambda axis: 400.0 if axis == 0 else 800.0)
    panel.inertia_card.encoder_resolution_edit.setValue(4000.0)
    panel.motion_panel.acceleration_edit.setValue(500.0)

    assert panel.inertia_card.motor_acceleration_edit.isReadOnly()
    assert panel.inertia_card.axis_units_edit.value() == pytest.approx(400.0)
    assert panel.inertia_card.motor_acceleration_edit.value() == pytest.approx(50.0)
    assert "ACCEL 500" in panel.inertia_card.motion_conversion_label.text()
    assert "UNITS 400" in panel.inertia_card.motion_conversion_label.text()

    panel.inertia_card.encoder_resolution_edit.setValue(2000.0)

    assert panel.inertia_card.motor_acceleration_edit.value() == pytest.approx(100.0)


def test_inertia_card_uses_cursor_average_and_tracks_tuning_axis(qt_app):
    panel = TunerPanel()
    panel.set_cursor_statistics_provider(
        lambda axis: {
            "source": f"DRIVE_CURRENT({axis})",
            "average": 425.5,
            "sample_count": 80,
        }
    )
    panel._axis_combo.setCurrentText("3")

    panel.inertia_card.btn_capture_acceleration.click()

    assert panel.inertia_card.acceleration_average_edit.value() == pytest.approx(425.5)
    assert panel.inertia_card.signal_combo.currentData() == RAW_CURRENT
    assert "80 samples" in panel.inertia_card.status_label.text()
    assert "Axis 3" in panel.inertia_card.status_label.text()


def test_inertia_card_only_exposes_acceleration_deceleration_method(qt_app):
    panel = TunerPanel()
    card = panel.inertia_card

    assert not hasattr(card, "method_combo")
    assert card.deceleration_average_edit.isEnabled()
    assert card.btn_capture_deceleration.isEnabled()


def test_detected_motor_parameters_populate_inertia_fields(qt_app):
    panel = TunerPanel()
    card = panel.inertia_card
    parameters = DetectedMotorParameters(
        rated_torque_nm=0.15,
        rated_current_a=0.9,
        rotor_inertia_units=230,
        encoder_resolution_counts=8_388_608,
    )

    card.apply_detected_motor_parameters(parameters)

    assert card.rated_torque_edit.value() == pytest.approx(0.15)
    assert card.rated_current_edit.value() == pytest.approx(0.9)
    assert card.motor_inertia_edit.value() == pytest.approx(230)
    assert card.encoder_resolution_edit.value() == pytest.approx(8_388_608)
    assert "read rated torque" in card.status_label.text()


def test_motor_read_button_tracks_connection_and_selected_axis(qt_app):
    panel = TunerPanel()
    requested = []
    panel.inertia_card.readMotorRequested.connect(requested.append)

    assert not panel.inertia_card.btn_read_motor.isEnabled()
    panel.set_connection(object())
    panel._axis_combo.setCurrentText("3")
    panel.inertia_card.btn_read_motor.click()

    assert requested == [3]
    panel.set_connection(None)
    assert not panel.inertia_card.btn_read_motor.isEnabled()


def test_inertia_estimate_can_be_applied_to_current_drive_profile(qt_app):
    panel = TunerPanel()
    panel.set_axis_units_provider(lambda axis: 1.0)
    panel._drive_combo.setCurrentText("DX4")
    card = panel.inertia_card
    card.signal_combo.setCurrentIndex(card.signal_combo.findData("raw_torque"))
    card.acceleration_average_edit.setValue(250.0)
    card.steady_average_edit.setValue(50.0)
    card.deceleration_average_edit.setValue(-150.0)
    card.encoder_resolution_edit.setValue(100.0)
    panel.motion_panel.acceleration_edit.setValue(1000.0)
    card.rated_torque_edit.setValue(2.0)
    card.motor_inertia_edit.setValue(100_000.0)

    estimate = card.last_estimate
    assert estimate is not None
    assert (
        estimate.total_inertia_kgm2 - estimate.load_inertia_kgm2
    ) == pytest.approx(0.001)
    expected = int(round(estimate.pn106_value))
    assert card.btn_apply_pn106.isEnabled()
    assert card.pn106_label.text() == f"Pn106  {expected}"
    assert "%" not in card.pn106_label.text()
    assert "Load / motor inertia" in card.inertia_ratio_label.text()
    assert ": 1" in card.inertia_ratio_label.text()

    card.btn_apply_pn106.click()

    assert panel._param_widgets["pn106"].value() == expected
    assert f"Pn106 set to raw value {expected}" in panel._status_label.text()


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
    assert newest_cell.foreground().color().name() == RED  # theme error = worse


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
    # No DX3/DX4 profile is configured → the card must say so instead of
    # silently omitting proposals and tuning-mode gating
    assert "No drive Pn profile attached" in texts


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
