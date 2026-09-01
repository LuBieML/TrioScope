"""Inertia estimation card for the Servo Tuning Workspace."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .inertia_estimator import (
    ACCEL_MINUS_STEADY,
    ACCEL_VS_DECEL,
    CURRENT_AMPS,
    RAW_CURRENT,
    RAW_TORQUE,
    TORQUE_NM,
    DX_ENCODER_COUNTS_PER_REV,
    InertiaEstimate,
    acceleration_to_motor_rev_s2,
    estimate_inertia,
    motor_inertia_units_to_kgm2,
)
from .motor_parameters import DetectedMotorParameters
from .tuner_theme import (
    ACCENT,
    BG_CARD,
    BG_PANEL,
    BORDER,
    CYAN,
    GREEN,
    GROUP_STYLE,
    RED,
    TEXT,
    TEXT_BRIGHT,
    TEXT_DIM,
)


class _EngineeringSpinBox(QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:
        return format(value, ".10g")


class InertiaCalculatorCard(QGroupBox):
    """Calculate motor-side load inertia and a per-drive Pn106 value."""

    applyPn106Requested = Signal(int)
    readMotorRequested = Signal(int)

    _SIGNAL_OPTIONS = (
        ("DX torque · 0.1% rated (recommended)", RAW_TORQUE),
        ("DX current · 0.1% rated", RAW_CURRENT),
        ("Current · A", CURRENT_AMPS),
        ("Torque · Nm", TORQUE_NM),
    )
    _METHOD_OPTIONS = (
        ("Acceleration − steady", ACCEL_MINUS_STEADY),
        ("Acceleration ↔ deceleration (recommended)", ACCEL_VS_DECEL),
    )

    def __init__(self, parent=None):
        super().__init__("Inertia estimate", parent)
        self.setObjectName("inertiaCalculator")
        self.setMinimumWidth(400)
        self.setMaximumWidth(620)
        self.setStyleSheet(GROUP_STYLE + self._local_style())
        self._axis = 0
        self._cursor_provider: Optional[Callable[[int], dict]] = None
        self._last_estimate: Optional[InertiaEstimate] = None
        self._axis_acceleration_units_s2: Optional[float] = None
        self._axis_units_counts: Optional[float] = None
        self._motion_source_error = ""
        self._motor_read_available = False
        self._build_ui()
        self._update_method_controls()
        self._update_signal_controls()
        self._recalculate()

    @staticmethod
    def _local_style() -> str:
        return f"""
            QLabel#inertiaStep {{
                color: {ACCENT}; font-family: Consolas; font-size: 7pt;
                font-weight: bold; letter-spacing: 1px; padding-top: 3px;
            }}
            QLabel#inertiaHint, QLabel#inertiaStatus {{
                color: {TEXT_DIM}; font-size: 8pt;
            }}
            QLabel#inertiaStatus[error="true"] {{ color: {RED}; }}
            QLabel#inertiaStatus[good="true"] {{ color: {GREEN}; }}
            QComboBox#inertiaChoice, QDoubleSpinBox#inertiaValue,
            QSpinBox#inertiaCount {{
                background: {BG_PANEL}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 3px;
                padding: 2px 4px; min-height: 19px;
            }}
            QPushButton#inertiaCapture {{
                background: {BG_PANEL}; color: {CYAN};
                border: 1px solid {BORDER}; border-radius: 3px;
                padding: 3px 5px; font-size: 7pt;
            }}
            QPushButton#inertiaReadMotor {{
                background: {BG_PANEL}; color: {CYAN};
                border: 1px solid {BORDER}; border-radius: 3px;
                padding: 3px 8px; font-size: 7pt; font-weight: bold;
            }}
            QFrame#inertiaResult {{
                background: {BG_CARD}; border: 1px solid {BORDER};
                border-radius: 4px;
            }}
            QLabel#inertiaResultLabel {{ color: {TEXT_DIM}; font-size: 7pt; }}
            QLabel#inertiaPn106 {{
                color: {TEXT_BRIGHT}; font-family: Consolas;
                font-size: 16pt; font-weight: bold;
            }}
            QLabel#inertiaDetails {{
                color: {TEXT}; font-family: Consolas; font-size: 8pt;
            }}
            QPushButton#inertiaApply {{
                background: {ACCENT}; color: #191919; border: none;
                border-radius: 3px; padding: 5px 8px; font-weight: bold;
            }}
            QPushButton:disabled {{ background: #454545; color: #777; border-color: #505050; }}
        """

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 7, 8, 8)
        outer.setSpacing(6)

        hint = QLabel(
            "Estimate inertia from a short controlled move when drive inertia "
            "identification cannot be used. Results are motor-side."
        )
        hint.setObjectName("inertiaHint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        outer.addWidget(self._step_label("1 · SELECT THE MEASUREMENT METHOD"))
        self.signal_combo = QComboBox()
        self.signal_combo.setObjectName("inertiaChoice")
        for label, value in self._SIGNAL_OPTIONS:
            self.signal_combo.addItem(label, value)
        self.signal_combo.currentIndexChanged.connect(self._update_signal_controls)
        outer.addWidget(self.signal_combo)

        self.method_combo = QComboBox()
        self.method_combo.setObjectName("inertiaChoice")
        for label, value in self._METHOD_OPTIONS:
            self.method_combo.addItem(label, value)
        self.method_combo.currentIndexChanged.connect(self._update_method_controls)
        outer.addWidget(self.method_combo)

        outer.addWidget(self._step_label("2 · CAPTURE PHASE AVERAGES"))
        measurement_grid = QGridLayout()
        measurement_grid.setContentsMargins(0, 0, 0, 0)
        measurement_grid.setHorizontalSpacing(5)
        measurement_grid.setVerticalSpacing(4)
        measurement_grid.addWidget(self._field_label("Phase"), 0, 0)
        self.measurement_unit_label = self._field_label("Average")
        measurement_grid.addWidget(self.measurement_unit_label, 0, 1)

        self.acceleration_average_edit = self._signed_editor(
            "Average signed torque/current during constant acceleration."
        )
        self.steady_average_edit = self._signed_editor(
            "Average signed torque/current at steady speed in the same direction."
        )
        self.deceleration_average_edit = self._signed_editor(
            "Average signed torque/current during constant deceleration."
        )
        self.btn_capture_acceleration = self._capture_button("acceleration")
        self.btn_capture_steady = self._capture_button("steady")
        self.btn_capture_deceleration = self._capture_button("deceleration")

        measurement_rows = (
            ("Acceleration", self.acceleration_average_edit, self.btn_capture_acceleration),
            ("Steady speed", self.steady_average_edit, self.btn_capture_steady),
            ("Deceleration", self.deceleration_average_edit, self.btn_capture_deceleration),
        )
        for row_index, (label, editor, button) in enumerate(measurement_rows, 1):
            measurement_grid.addWidget(QLabel(label), row_index, 0)
            measurement_grid.addWidget(editor, row_index, 1)
            measurement_grid.addWidget(button, row_index, 2)
        outer.addLayout(measurement_grid)

        outer.addWidget(self._step_label("3 · ENTER TEST MOTION DATA"))
        motion_grid = QGridLayout()
        motion_grid.setContentsMargins(0, 0, 0, 0)
        motion_grid.setHorizontalSpacing(5)
        motion_grid.setVerticalSpacing(4)

        self.axis_units_edit = self._positive_editor(
            "Controller UNITS value for the selected axis in feedback "
            "counts per user unit.",
            decimals=6,
        )
        self.axis_units_edit.setReadOnly(True)
        self.axis_units_edit.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.axis_units_edit.setFocusPolicy(Qt.NoFocus)
        self.axis_units_edit.setSpecialValueText("Needs Axis setup")
        motion_grid.addWidget(QLabel("Axis scaling (UNITS)"), 0, 0)
        motion_grid.addWidget(self.axis_units_edit, 0, 1)
        axis_units_label = QLabel("counts/user unit")
        axis_units_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
        motion_grid.addWidget(axis_units_label, 0, 2)

        self.encoder_resolution_edit = self._positive_editor(
            "Motor encoder resolution in controller motion units per "
            "revolution.",
            decimals=0,
        )
        self.encoder_resolution_edit.setValue(DX_ENCODER_COUNTS_PER_REV)
        motion_grid.addWidget(QLabel("Motor encoder resolution"), 1, 0)
        motion_grid.addWidget(self.encoder_resolution_edit, 1, 1)
        encoder_unit = QLabel("counts/rev")
        encoder_unit.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
        motion_grid.addWidget(encoder_unit, 1, 2)

        self.motor_acceleration_edit = self._positive_editor(
            "Calculated from Test motion acceleration, controller UNITS and "
            "motor encoder resolution.",
            decimals=10,
        )
        self.motor_acceleration_edit.setReadOnly(True)
        self.motor_acceleration_edit.setButtonSymbols(
            QAbstractSpinBox.NoButtons
        )
        self.motor_acceleration_edit.setFocusPolicy(Qt.NoFocus)
        self.motor_acceleration_edit.setSpecialValueText("Required")
        motion_grid.addWidget(QLabel("Calculated acceleration"), 2, 0)
        motion_grid.addWidget(self.motor_acceleration_edit, 2, 1)
        acceleration_unit = QLabel("rev/s²")
        acceleration_unit.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 7pt;"
        )
        motion_grid.addWidget(acceleration_unit, 2, 2)
        self.motion_conversion_label = QLabel()
        self.motion_conversion_label.setObjectName("inertiaHint")
        self.motion_conversion_label.setWordWrap(True)
        motion_grid.addWidget(self.motion_conversion_label, 3, 0, 1, 3)
        outer.addLayout(motion_grid)

        motor_heading = QHBoxLayout()
        motor_heading.setContentsMargins(0, 0, 0, 0)
        motor_heading.addWidget(self._step_label("4 · ENTER MOTOR NAMEPLATE DATA"))
        motor_heading.addStretch()
        self.btn_read_motor = QPushButton("READ FROM DRIVE")
        self.btn_read_motor.setObjectName("inertiaReadMotor")
        self.btn_read_motor.setEnabled(False)
        self.btn_read_motor.setToolTip(
            "Read Pn810 rated torque, Pn812 rated current, Pn831 rotor "
            "inertia and Pn880 encoder resolution from the connected DX drive."
        )
        self.btn_read_motor.clicked.connect(
            lambda: self.readMotorRequested.emit(self._axis)
        )
        motor_heading.addWidget(self.btn_read_motor)
        outer.addLayout(motor_heading)
        motor_grid = QGridLayout()
        motor_grid.setContentsMargins(0, 0, 0, 0)
        motor_grid.setHorizontalSpacing(5)
        motor_grid.setVerticalSpacing(4)

        self.rated_torque_edit = self._positive_editor("Motor rated torque in Nm.")
        self.rated_current_edit = self._positive_editor("Motor rated current in A.")
        self.motor_inertia_edit = self._positive_editor(
            "Enter the motor-table value directly. One unit equals "
            "1e-8 kg·m²; for example, enter 230 for 2.30e-6 kg·m².",
            decimals=3,
        )
        self.motor_count_edit = QSpinBox()
        self.motor_count_edit.setObjectName("inertiaCount")
        self.motor_count_edit.setRange(1, 32)
        self.motor_count_edit.setValue(1)
        self.motor_count_edit.setToolTip(
            "Number of identical gantry motors contributing equal torque."
        )
        self.motor_count_edit.valueChanged.connect(self._recalculate)

        motor_rows = (
            ("Rated torque", self.rated_torque_edit, "Nm"),
            ("Rated current", self.rated_current_edit, "A"),
            ("Rotor inertia", self.motor_inertia_edit, "1e-8 kg·m²"),
            ("Equal motors", self.motor_count_edit, ""),
        )
        for row_index, (label, editor, unit) in enumerate(motor_rows):
            motor_grid.addWidget(QLabel(label), row_index, 0)
            motor_grid.addWidget(editor, row_index, 1)
            if unit:
                unit_label = QLabel(unit)
                unit_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
                motor_grid.addWidget(unit_label, row_index, 2)
        outer.addLayout(motor_grid)

        result = QFrame()
        result.setObjectName("inertiaResult")
        result_layout = QVBoxLayout(result)
        result_layout.setContentsMargins(8, 7, 8, 7)
        result_layout.setSpacing(2)
        result_label = QLabel("ESTIMATED LOAD INERTIA RATIO")
        result_label.setObjectName("inertiaResultLabel")
        result_layout.addWidget(result_label)
        self.pn106_label = QLabel("Pn106  —")
        self.pn106_label.setObjectName("inertiaPn106")
        result_layout.addWidget(self.pn106_label)
        self.result_details = QLabel("Enter test-motion and motor data to calculate.")
        self.result_details.setObjectName("inertiaDetails")
        self.result_details.setWordWrap(True)
        result_layout.addWidget(self.result_details)
        outer.addWidget(result)

        self.btn_apply_pn106 = QPushButton("Apply estimate to Pn106")
        self.btn_apply_pn106.setObjectName("inertiaApply")
        self.btn_apply_pn106.setEnabled(False)
        self.btn_apply_pn106.setToolTip(
            "Copy the rounded estimate into the current drive profile."
        )
        self.btn_apply_pn106.clicked.connect(self._request_apply)
        outer.addWidget(self.btn_apply_pn106)

        self.status_label = QLabel()
        self.status_label.setObjectName("inertiaStatus")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        for editor in (
            self.acceleration_average_edit,
            self.steady_average_edit,
            self.deceleration_average_edit,
            self.motor_acceleration_edit,
            self.rated_torque_edit,
            self.rated_current_edit,
            self.motor_inertia_edit,
        ):
            editor.valueChanged.connect(self._recalculate)
        self.encoder_resolution_edit.valueChanged.connect(
            self._update_derived_acceleration
        )
        self._update_derived_acceleration()

    @staticmethod
    def _step_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("inertiaStep")
        return label

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
        return label

    @staticmethod
    def _signed_editor(tooltip: str) -> QDoubleSpinBox:
        editor = _EngineeringSpinBox()
        editor.setObjectName("inertiaValue")
        editor.setRange(-1_000_000_000.0, 1_000_000_000.0)
        editor.setDecimals(6)
        editor.setKeyboardTracking(False)
        editor.setToolTip(tooltip)
        return editor

    @staticmethod
    def _positive_editor(tooltip: str, decimals: int = 6) -> QDoubleSpinBox:
        editor = _EngineeringSpinBox()
        editor.setObjectName("inertiaValue")
        editor.setRange(0.0, 1_000_000_000.0)
        editor.setDecimals(decimals)
        editor.setKeyboardTracking(False)
        editor.setSpecialValueText("Required")
        editor.setToolTip(tooltip)
        return editor

    def _capture_button(self, target: str) -> QPushButton:
        button = QPushButton("Use AVG")
        button.setObjectName("inertiaCapture")
        button.setToolTip(f"Use the current C1-C2 average as the {target} value.")
        button.clicked.connect(lambda _=False, name=target: self._capture_average(name))
        return button

    def set_axis(self, axis: int) -> None:
        self._axis = int(axis)

    def set_motor_read_available(self, available: bool) -> None:
        """Enable motor detection when a controller connection is available."""
        self._motor_read_available = bool(available)
        self.btn_read_motor.setEnabled(self._motor_read_available)

    def set_motor_read_busy(self, busy: bool) -> None:
        self.btn_read_motor.setText("READING..." if busy else "READ FROM DRIVE")
        self.btn_read_motor.setEnabled(self._motor_read_available and not busy)

    def apply_detected_motor_parameters(
        self, parameters: DetectedMotorParameters
    ) -> None:
        """Populate successful drive reads while preserving manual fallbacks."""
        values = (
            (self.rated_torque_edit, parameters.rated_torque_nm),
            (self.rated_current_edit, parameters.rated_current_a),
            (self.motor_inertia_edit, parameters.rotor_inertia_units),
            (
                self.encoder_resolution_edit,
                parameters.encoder_resolution_counts,
            ),
        )
        for editor, value in values:
            if value is None:
                continue
            editor.blockSignals(True)
            editor.setValue(float(value))
            editor.blockSignals(False)

        self._update_derived_acceleration()
        self._recalculate()
        found = ", ".join(parameters.detected_fields)
        missing = ", ".join(parameters.failures)
        message = f"Axis {self._axis}: read {found} from the DX drive."
        if missing:
            message += f" Manual value retained for: {missing}."
        self._set_status(message, good=not parameters.failures)

    def show_motor_read_error(self, message: str) -> None:
        self._set_status(message, error=True)

    def set_test_motion_acceleration(
        self,
        acceleration_units_s2: float,
        axis_units_counts: Optional[float],
        source_error: str = "",
    ) -> None:
        """Update the derived motor acceleration from the motion setup."""
        self._axis_acceleration_units_s2 = float(acceleration_units_s2)
        self._axis_units_counts = (
            float(axis_units_counts) if axis_units_counts is not None else None
        )
        self._motion_source_error = source_error
        self._update_derived_acceleration()

    def set_cursor_provider(self, provider: Callable[[int], dict]) -> None:
        self._cursor_provider = provider

    @property
    def last_estimate(self) -> Optional[InertiaEstimate]:
        return self._last_estimate

    def _capture_average(self, target: str) -> None:
        if self._cursor_provider is None:
            self._set_status("Cursor measurements are unavailable.", error=True)
            return
        try:
            selection = self._cursor_provider(self._axis)
            average = float(selection["average"])
            source = str(selection["source"])
            sample_count = int(selection["sample_count"])
        except Exception as exc:
            self._set_status(str(exc), error=True)
            return

        editors = {
            "acceleration": self.acceleration_average_edit,
            "steady": self.steady_average_edit,
            "deceleration": self.deceleration_average_edit,
        }
        editors[target].setValue(average)
        self._select_signal_mode_for_source(source)
        self._set_status(
            f"Axis {self._axis} {target}: {average:.6g} from {source} "
            f"({sample_count} samples)."
        )

    def _select_signal_mode_for_source(self, source: str) -> None:
        flat = source.lower().replace("_", "").replace(" ", "")
        mode = RAW_TORQUE if ("torque" in flat or flat.startswith("tn")) else RAW_CURRENT
        index = self.signal_combo.findData(mode)
        if index >= 0:
            self.signal_combo.setCurrentIndex(index)

    def _update_method_controls(self) -> None:
        recommended = self.method_combo.currentData() == ACCEL_VS_DECEL
        self.deceleration_average_edit.setEnabled(recommended)
        self.btn_capture_deceleration.setEnabled(recommended)
        self._recalculate()

    def _update_signal_controls(self) -> None:
        mode = self.signal_combo.currentData()
        uses_current = mode in {RAW_CURRENT, CURRENT_AMPS}
        self.rated_current_edit.setEnabled(uses_current)
        self.rated_current_edit.setSpecialValueText(
            "Required" if mode == CURRENT_AMPS else "Optional"
        )
        units = {
            RAW_TORQUE: "AVERAGE · 0.1% Tn",
            RAW_CURRENT: "AVERAGE · 0.1% In",
            CURRENT_AMPS: "AVERAGE · A",
            TORQUE_NM: "AVERAGE · Nm",
        }
        self.measurement_unit_label.setText(units.get(mode, "AVERAGE"))
        self._recalculate()

    def _update_derived_acceleration(self, _value: float = 0.0) -> None:
        acceleration = self._axis_acceleration_units_s2
        axis_units = self._axis_units_counts
        self.axis_units_edit.setValue(axis_units or 0.0)
        if acceleration is None or axis_units is None:
            self.motor_acceleration_edit.setValue(0.0)
            self.motion_conversion_label.setText(
                self._motion_source_error
                or "Configure the selected axis in Axis setup to provide UNITS."
            )
            return
        try:
            motor_rev_s2 = acceleration_to_motor_rev_s2(
                acceleration,
                axis_units,
                self.encoder_resolution_edit.value(),
            )
        except ValueError as exc:
            self.motor_acceleration_edit.setValue(0.0)
            self.motion_conversion_label.setText(str(exc))
            return
        self.motor_acceleration_edit.setValue(motor_rev_s2)
        self.motion_conversion_label.setText(
            f"ACCEL {acceleration:.6g} × UNITS {axis_units:.6g} ÷ "
            f"encoder {self.encoder_resolution_edit.value():.6g}."
        )

    def _recalculate(self) -> None:
        if not hasattr(self, "status_label"):
            return
        try:
            estimate = estimate_inertia(
                acceleration_average=self.acceleration_average_edit.value(),
                steady_average=self.steady_average_edit.value(),
                deceleration_average=self.deceleration_average_edit.value(),
                motor_acceleration_rpm_s=(
                    self.motor_acceleration_edit.value() * 60.0
                ),
                rated_torque_nm=self.rated_torque_edit.value(),
                rated_current_a=self.rated_current_edit.value(),
                motor_inertia_kgm2=motor_inertia_units_to_kgm2(
                    self.motor_inertia_edit.value()
                ),
                motor_count=self.motor_count_edit.value(),
                signal_mode=str(self.signal_combo.currentData()),
                method=str(self.method_combo.currentData()),
            )
        except ValueError as exc:
            self._last_estimate = None
            self.pn106_label.setText("Pn106  —")
            self.result_details.setText(
                "Enter the required phase, test-motion and motor values."
            )
            self.btn_apply_pn106.setEnabled(False)
            self._set_status(str(exc))
            return

        self._last_estimate = estimate
        rounded_pn106 = int(round(estimate.pn106_percent))
        self.pn106_label.setText(f"Pn106  {rounded_pn106} %")
        mode = self.signal_combo.currentData()
        if mode == RAW_CURRENT:
            raw_current = abs(estimate.signal_delta)
            amps = (
                f" = {estimate.acceleration_current_a:.6g} A"
                if estimate.acceleration_current_a is not None
                else ""
            )
            current_line = (
                f"Iacc {raw_current:.6g} ×0.1% In "
                f"({raw_current / 10.0:.6g}% In){amps}  ·  "
            )
        elif estimate.acceleration_current_a is not None:
            current_line = f"Iacc {estimate.acceleration_current_a:.6g} A  ·  "
        elif mode == RAW_TORQUE:
            current_line = (
                f"Δtorque {abs(estimate.signal_delta):.6g} ×0.1% Tn  ·  "
            )
        else:
            current_line = ""
        self.result_details.setText(
            f"{current_line}Tacc {estimate.torque_per_motor_nm:.6g} Nm/motor\n"
            f"Jtotal {estimate.total_inertia_kgm2:.8g} kg·m²  ·  "
            f"Jload {estimate.load_inertia_kgm2:.8g} kg·m²"
        )

        valid_pn106 = 0 <= rounded_pn106 <= 9999
        self.btn_apply_pn106.setEnabled(valid_pn106)
        if estimate.load_inertia_kgm2 < 0:
            self._set_status(
                "Estimated total inertia is below the entered rotor inertia. "
                "Check signal scaling, acceleration and phase selection.",
                error=True,
            )
        elif not valid_pn106:
            self._set_status(
                "The estimate is outside the drive's Pn106 range (0–9999%).",
                error=True,
            )
        elif (
            estimate.symmetry_error_percent is not None
            and estimate.symmetry_error_percent > 20.0
        ):
            self._set_status(
                f"Acceleration/deceleration mismatch is "
                f"{estimate.symmetry_error_percent:.1f}%. Repeat the windows "
                "at comparable speed.",
                error=True,
            )
        else:
            symmetry = (
                f" Phase symmetry {estimate.symmetry_error_percent:.1f}%."
                if estimate.symmetry_error_percent is not None
                else ""
            )
            self._set_status(
                f"Estimate ready for {self.motor_count_edit.value()} equal "
                f"motor{'s' if self.motor_count_edit.value() != 1 else ''}."
                f"{symmetry}",
                good=True,
            )

    def _request_apply(self) -> None:
        if self._last_estimate is None:
            return
        value = int(round(self._last_estimate.pn106_percent))
        if 0 <= value <= 9999:
            self.applyPn106Requested.emit(value)

    def _set_status(
        self, text: str, error: bool = False, good: bool = False
    ) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("error", bool(error))
        self.status_label.setProperty("good", bool(good))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
