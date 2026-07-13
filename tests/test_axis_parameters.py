import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import QApplication

from src.models.axis_parameter_config import AxisParameterConfig
from src.scope.axis_parameter_writer import (
    AXIS_PARAMETER_SETTERS,
    AxisParameterWriteError,
    write_axis_parameters,
)
from src.storage.axis_config_io import load_axis_config, save_axis_config
from src.ui.axis_parameters_tab import AxisParametersTab


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_axis_parameter_json_round_trip(tmp_path):
    path = tmp_path / "axes.json"
    configs = [
        AxisParameterConfig(axis=0),
        AxisParameterConfig(
            axis=5,
            speed=420.5,
            units=1048576.0,
            accel=5000.0,
            decel=4500.0,
            fast_dec=60000.0,
            jerk=110000.0,
            fwd_in=12,
            rev_in=13,
            fe_limit=2.25,
            drive_fe_limit=8,
            fe_range=7.5,
            fs_limit=950.0,
            rs_limit=-850.0,
        ),
    ]

    save_axis_config(str(path), configs)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["format"] == "TrioScope axis parameters"
    assert payload["version"] == 1
    assert payload["axis_parameters"][1]["fast_dec"] == 60000.0
    assert payload["axis_parameters"][1]["drive_fe_limit"] == 8
    assert payload["axis_parameters"][1]["rs_limit"] == -850.0
    assert load_axis_config(str(path)) == configs


def test_axis_parameter_defaults_and_legacy_json_migration(tmp_path):
    defaults = AxisParameterConfig()
    assert defaults.drive_fe_limit == 10
    assert defaults.fe_range == 10.0
    assert defaults.fs_limit == 1000.0
    assert defaults.rs_limit == -1000.0

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"axis_parameters": [{"axis": 3}]}), encoding="utf-8")

    assert load_axis_config(str(path)) == [AxisParameterConfig(axis=3)]


def test_axis_parameter_json_rejects_duplicate_axes(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps({"axis_parameters": [{"axis": 3}, {"axis": 3}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configured more than once"):
        load_axis_config(str(path))


def test_axis_tab_add_copy_remove_and_file_actions(qt_app, tmp_path):
    tab = AxisParametersTab()
    assert [tab.table.columnWidth(column) for column in range(14)] == [
        54,
        76,
        94,
        78,
        78,
        82,
        82,
        68,
        68,
        72,
        88,
        76,
        76,
        76,
    ]
    assert tab.table.columnWidth(14) == 170
    assert tab.table.columnWidth(15) == 70
    tab.set_configurations(
        [
            AxisParameterConfig(axis=1, speed=200.0, units=1000.0),
            AxisParameterConfig(axis=9, speed=25.0, units=500.0),
        ]
    )

    tab.copy_axis_values(source_axis=1, target_axis=9)
    copied = {config.axis: config for config in tab.configurations()}
    assert copied[9].speed == 200.0
    assert copied[9].units == 1000.0
    assert copied[9].drive_fe_limit == 10
    assert copied[9].fs_limit == 1000.0

    assert tab.add_axis()
    assert [config.axis for config in tab.configurations()] == [1, 9, 0]

    path = tmp_path / "from_tab.json"
    assert tab.save_config(str(path))

    restored = AxisParametersTab()
    assert restored.load_config(str(path))
    assert restored.configurations() == tab.configurations()


class _RecordingConnection:
    def __init__(self, fail_method=None, missing_method=None):
        self.calls = []
        self.fail_method = fail_method
        self.missing_method = missing_method

    def __getattr__(self, name):
        valid_methods = {spec.method_name for spec in AXIS_PARAMETER_SETTERS}
        if name not in valid_methods or name == self.missing_method:
            raise AttributeError(name)

        def _setter(axis, value):
            if name == self.fail_method:
                raise RuntimeError("controller rejected value")
            self.calls.append((name, axis, value))

        return _setter


class _TrackingLock:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_uapi_writer_uses_manual_defined_methods_types_and_order():
    connection = _RecordingConnection()
    lock = _TrackingLock()
    progress = []
    config = AxisParameterConfig(
        axis=4,
        speed=250.5,
        units=1048576.0,
        accel=4000.0,
        decel=3500.0,
        fast_dec=45000.0,
        jerk=90000.0,
        fwd_in=12,
        rev_in=-1,
        fe_limit=3.5,
        drive_fe_limit=9,
        fe_range=8.5,
        fs_limit=1200.0,
        rs_limit=-1100.0,
    )

    count = write_axis_parameters(
        connection,
        [config],
        connection_lock=lock,
        progress_callback=lambda *args: progress.append(args),
    )

    assert count == 13
    assert lock.entries == 13
    assert [call[0] for call in connection.calls] == [
        "SetAxisParameter_UNITS",
        "SetAxisParameter_SPEED",
        "SetAxisParameter_ACCEL",
        "SetAxisParameter_DECEL",
        "SetAxisParameter_FASTDEC",
        "SetAxisParameter_JERK",
        "SetAxisParameter_FWD_IN",
        "SetAxisParameter_REV_IN",
        "SetAxisParameter_FE_LIMIT",
        "SetAxisParameter_DRIVE_FE_LIMIT",
        "SetAxisParameter_FE_RANGE",
        "SetAxisParameter_FS_LIMIT",
        "SetAxisParameter_RS_LIMIT",
    ]
    assert all(call[1] == 4 for call in connection.calls)
    assert isinstance(connection.calls[6][2], int)
    assert isinstance(connection.calls[7][2], int)
    assert isinstance(connection.calls[9][2], int)
    assert all(isinstance(connection.calls[i][2], float) for i in (0, 1, 2, 3, 4, 5, 8))
    assert all(isinstance(connection.calls[i][2], float) for i in (10, 11, 12))
    assert progress[-1] == (13, 13, 4, "RS_LIMIT")


def test_uapi_writer_reports_partial_failure_context():
    connection = _RecordingConnection(fail_method="SetAxisParameter_FASTDEC")

    with pytest.raises(AxisParameterWriteError) as caught:
        write_axis_parameters(connection, [AxisParameterConfig(axis=2)])

    error = caught.value
    assert error.axis == 2
    assert error.parameter == "FASTDEC"
    assert error.completed == 4
    assert error.total == 13
    assert "controller rejected value" in str(error)


def test_uapi_writer_preflights_binding_before_any_write():
    connection = _RecordingConnection(missing_method="SetAxisParameter_RS_LIMIT")

    with pytest.raises(AxisParameterWriteError, match="SetAxisParameter_RS_LIMIT"):
        write_axis_parameters(connection, [AxisParameterConfig(axis=0)])

    assert connection.calls == []


def test_axis_input_parameters_require_uapi_int16_values():
    with pytest.raises(ValueError, match="whole-number"):
        AxisParameterConfig(axis=0, fwd_in=1.5).validate()
    with pytest.raises(ValueError, match="int16"):
        AxisParameterConfig(axis=0, rev_in=40000).validate()


def test_drive_fe_limit_requires_uapi_int64_value():
    with pytest.raises(ValueError, match="whole number"):
        AxisParameterConfig(axis=0, drive_fe_limit=1.5).validate()
    with pytest.raises(ValueError, match="int64"):
        AxisParameterConfig(axis=0, drive_fe_limit=2**63).validate()


def test_send_button_tracks_connection_and_calls_uapi_writer(qt_app):
    tab = AxisParametersTab()
    tab.add_axis(AxisParameterConfig(axis=2))
    assert not tab.btn_send.isEnabled()

    connection = _RecordingConnection()
    lock = _TrackingLock()
    tab.set_connection(connection, lock)
    assert tab.btn_send.isEnabled()

    assert tab._write_axis_parameters(tab.configurations()) == 13
    assert len(connection.calls) == 13
    assert lock.entries == 13
