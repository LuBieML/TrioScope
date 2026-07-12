import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import QApplication

from src.models.axis_parameter_config import AxisParameterConfig
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
            fwd_in=12.0,
            rev_in=13.0,
            fe_limit=2.25,
        ),
    ]

    save_axis_config(str(path), configs)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["format"] == "TrioScope axis parameters"
    assert payload["version"] == 1
    assert payload["axis_parameters"][1]["fast_dec"] == 60000.0
    assert load_axis_config(str(path)) == configs


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

    assert tab.add_axis()
    assert [config.axis for config in tab.configurations()] == [1, 9, 0]

    path = tmp_path / "from_tab.json"
    assert tab.save_config(str(path))

    restored = AxisParametersTab()
    assert restored.load_config(str(path))
    assert restored.configurations() == tab.configurations()


def test_send_button_tracks_connection_but_transport_is_placeholder(qt_app):
    tab = AxisParametersTab()
    tab.add_axis(AxisParameterConfig(axis=2))
    assert not tab.btn_send.isEnabled()

    tab.set_connection(object())
    assert tab.btn_send.isEnabled()

    with pytest.raises(NotImplementedError):
        tab._write_axis_parameters(tab.configurations())
