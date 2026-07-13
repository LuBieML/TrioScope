"""Analyzer-aware test-stroke calculator used by the Axis Motion window."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

try:
    from ..ai.analysis_move_profile import (
        AnalysisMoveProfile,
        RECOMMENDED_CRUISE_DURATION_S,
        RECOMMENDED_DWELL_DURATION_S,
        RECOMMENDED_IDLE_DURATION_S,
        calculate_analysis_move,
    )
except ImportError:  # App runtime imports ui as a top-level package.
    from ai.analysis_move_profile import (
        AnalysisMoveProfile,
        RECOMMENDED_CRUISE_DURATION_S,
        RECOMMENDED_DWELL_DURATION_S,
        RECOMMENDED_IDLE_DURATION_S,
        calculate_analysis_move,
    )


class _CalculatorValueSpinBox(QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:
        return format(value, ".12g")


class TestMoveCalculator(QFrame):
    """Calculate and apply speed/distance values for an analysis move."""

    applyRequested = Signal(int, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("motionTestCalculator")
        self._profile: AnalysisMoveProfile | None = None
        self._build_ui()
        self._update_result()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 14, 14)
        layout.setSpacing(18)

        intro = QVBoxLayout()
        intro.setSpacing(3)
        eyebrow = QLabel("ANALYZER-READY MOVE")
        eyebrow.setObjectName("motionCalcEyebrow")
        intro.addWidget(eyebrow)
        title = QLabel("Test stroke calculator")
        title.setObjectName("motionCalcTitle")
        intro.addWidget(title)
        description = QLabel(
            "Enter the speed and the ACCEL value the controller will use. "
            f"The stroke includes {RECOMMENDED_CRUISE_DURATION_S:g} s of "
            "clean cruise for signal analysis."
        )
        description.setObjectName("motionCalcDescription")
        description.setWordWrap(True)
        intro.addWidget(description)
        intro.addStretch()
        layout.addLayout(intro, 2)

        inputs = QGridLayout()
        inputs.setHorizontalSpacing(10)
        inputs.setVerticalSpacing(6)
        speed_label = self._input_label("TEST SPEED")
        self.speed_edit = self._value_editor(100.0, "Desired constant speed.")
        speed_label.setBuddy(self.speed_edit)
        inputs.addWidget(speed_label, 0, 0)
        inputs.addWidget(self.speed_edit, 1, 0)

        accel_label = self._input_label("AXIS ACCEL")
        self.acceleration_edit = self._value_editor(
            500.0, "Use the same value as the axis ACCEL setting."
        )
        accel_label.setBuddy(self.acceleration_edit)
        inputs.addWidget(accel_label, 0, 1)
        inputs.addWidget(self.acceleration_edit, 1, 1)

        target_label = self._input_label("APPLY TO")
        self.axis_combo = QComboBox()
        self.axis_combo.setObjectName("motionCalcAxis")
        self.axis_combo.setAccessibleName("Axis row to receive calculated values")
        target_label.setBuddy(self.axis_combo)
        inputs.addWidget(target_label, 2, 0, 1, 2)
        inputs.addWidget(self.axis_combo, 3, 0, 1, 2)
        layout.addLayout(inputs, 2)

        readout = QFrame()
        readout.setObjectName("motionCalcReadout")
        readout_layout = QVBoxLayout(readout)
        readout_layout.setContentsMargins(14, 10, 14, 10)
        readout_layout.setSpacing(2)
        readout_label = QLabel("RECOMMENDED DISTANCE")
        readout_label.setObjectName("motionCalcReadoutLabel")
        readout_layout.addWidget(readout_label)

        value_row = QHBoxLayout()
        value_row.setSpacing(7)
        self.distance_label = QLabel("—")
        self.distance_label.setObjectName("motionCalcDistance")
        self.distance_label.setAccessibleName("Recommended test distance")
        value_row.addWidget(self.distance_label)
        units = QLabel("axis units")
        units.setObjectName("motionCalcUnits")
        value_row.addWidget(units, 1)
        readout_layout.addLayout(value_row)

        self.detail_label = QLabel()
        self.detail_label.setObjectName("motionCalcDetail")
        self.detail_label.setWordWrap(True)
        readout_layout.addWidget(self.detail_label)

        self.apply_button = QPushButton("Apply speed + distance")
        self.apply_button.setObjectName("motionCalcApplyButton")
        self.apply_button.setToolTip(
            "Copies speed and recommended distance to the selected row. "
            "The controller ACCEL value is not changed."
        )
        self.apply_button.clicked.connect(self._request_apply)
        readout_layout.addWidget(self.apply_button)
        layout.addWidget(readout, 3)

        self.speed_edit.valueChanged.connect(self._update_result)
        self.acceleration_edit.valueChanged.connect(self._update_result)

    @staticmethod
    def _input_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("motionCalcInputLabel")
        return label

    @staticmethod
    def _value_editor(value: float, tooltip: str) -> QDoubleSpinBox:
        editor = _CalculatorValueSpinBox()
        editor.setObjectName("motionCalcInput")
        editor.setRange(0.001, 1_000_000_000_000.0)
        editor.setDecimals(6)
        editor.setSingleStep(1.0)
        editor.setValue(value)
        editor.setKeyboardTracking(False)
        editor.setToolTip(tooltip)
        editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        editor.setMinimumWidth(130)
        return editor

    @property
    def profile(self) -> AnalysisMoveProfile:
        assert self._profile is not None
        return self._profile

    def set_axes(self, axes: Iterable[int]) -> None:
        previous = self.axis_combo.currentData()
        self.axis_combo.blockSignals(True)
        self.axis_combo.clear()
        for axis in axes:
            self.axis_combo.addItem(f"Axis {int(axis)}", int(axis))
        if previous is not None:
            index = self.axis_combo.findData(previous)
            if index >= 0:
                self.axis_combo.setCurrentIndex(index)
        self.axis_combo.blockSignals(False)
        self._refresh_apply_state()

    def set_editable(self, editable: bool) -> None:
        self.speed_edit.setEnabled(editable)
        self.acceleration_edit.setEnabled(editable)
        self.axis_combo.setEnabled(editable and self.axis_combo.count() > 0)
        self.apply_button.setEnabled(
            editable and self.axis_combo.count() > 0 and self._profile is not None
        )

    def _update_result(self, _value: float | None = None) -> None:
        self._profile = calculate_analysis_move(
            self.speed_edit.value(), self.acceleration_edit.value()
        )
        p = self._profile
        self.distance_label.setText(format(p.recommended_distance, ".8g"))
        self.detail_label.setText(
            f"{p.cruise_duration_s:g} s cruise  ·  "
            f"{p.acceleration_duration_s:.3g} s ramp each way  ·  "
            f"{p.total_move_duration_s:.3g} s move\n"
            f"Capture ≥{p.minimum_capture_duration_s:.3g} s with "
            f"{RECOMMENDED_IDLE_DURATION_S:g} s idle before and "
            f"{RECOMMENDED_DWELL_DURATION_S:g} s after"
        )
        self._refresh_apply_state()

    def _refresh_apply_state(self) -> None:
        enabled = self.speed_edit.isEnabled() and self.axis_combo.count() > 0
        self.axis_combo.setEnabled(enabled)
        self.apply_button.setEnabled(enabled and self._profile is not None)

    def _request_apply(self) -> None:
        axis = self.axis_combo.currentData()
        if axis is None or self._profile is None:
            return
        self.applyRequested.emit(
            int(axis),
            self._profile.speed,
            self._profile.recommended_distance,
            self._profile.acceleration,
        )
