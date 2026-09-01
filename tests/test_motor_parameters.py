"""DX motor metadata decoding and CoE read tests."""

import pytest

from src.ai import coe_io
from src.ai.motor_parameters import (
    MOTOR_PARAMETER_OBJECT_IDS,
    decode_motor_parameter_registers,
)


def test_dx_motor_registers_decode_to_inertia_ui_units():
    result = decode_motor_parameter_registers(
        {
            "rated_torque": 15,
            "rated_current": 9,
            "rotor_inertia": 230,
            "encoder_bits": 23,
        }
    )

    assert result.rated_torque_nm == pytest.approx(0.15)
    assert result.rated_current_a == pytest.approx(0.9)
    assert result.rotor_inertia_units == 230
    assert result.encoder_resolution_counts == 8_388_608
    assert result.detected_fields == (
        "rated torque",
        "rated current",
        "rotor inertia",
        "encoder resolution",
    )
    assert not result.failures


def test_invalid_motor_register_does_not_replace_manual_value():
    result = decode_motor_parameter_registers(
        {
            "rated_torque": 15,
            "rated_current": 0,
            "rotor_inertia": 230,
            "encoder_bits": 31,
        }
    )

    assert result.rated_torque_nm == pytest.approx(0.15)
    assert result.rated_current_a is None
    assert result.encoder_resolution_counts is None
    assert set(result.failures) == {"rated_current", "encoder_bits"}


def test_motor_parameter_reader_returns_supported_subset(monkeypatch):
    values = {
        MOTOR_PARAMETER_OBJECT_IDS["rated_torque"]: 15,
        MOTOR_PARAMETER_OBJECT_IDS["rated_current"]: 9,
        MOTOR_PARAMETER_OBJECT_IDS["rotor_inertia"]: 230,
    }

    def fake_read(_connection, axis, object_index, **_kwargs):
        assert axis == 4
        if object_index == MOTOR_PARAMETER_OBJECT_IDS["encoder_bits"]:
            raise TimeoutError("unsupported object")
        return values[object_index]

    monkeypatch.setattr(coe_io, "coe_read_axis", fake_read)

    result = coe_io.read_motor_parameters(object(), axis=4)

    assert result.rated_torque_nm == pytest.approx(0.15)
    assert result.rated_current_a == pytest.approx(0.9)
    assert result.rotor_inertia_units == 230
    assert result.encoder_resolution_counts is None
    assert "encoder_bits" in result.failures


def test_motor_parameter_reader_rejects_total_failure(monkeypatch):
    def fail_read(*_args, **_kwargs):
        raise TimeoutError("no response")

    monkeypatch.setattr(coe_io, "coe_read_axis", fail_read)

    with pytest.raises(ConnectionError, match="No motor parameters"):
        coe_io.read_motor_parameters(object(), axis=1)
