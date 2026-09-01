"""Compact single-axis motion controls embedded in the tuning workspace."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

try:
    from ..models.motion_axis_command import MotionAxisCommand
except ImportError:  # App runtime imports ai as a top-level package.
    from models.motion_axis_command import MotionAxisCommand

from .tuner_theme import (
    ACCENT,
    BG_CARD,
    BG_PANEL,
    BORDER,
    GREEN,
    GROUP_STYLE,
    RED,
    TEXT,
    TEXT_DIM,
)


class _MotionSpinBox(QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:
        return format(value, ".12g")


class TuningMotionPanel(QGroupBox):
    """A focused prepare-enable-move workflow for one tuning axis."""

    axisChanged = Signal(int)
    enableRequested = Signal(bool, object)
    startRequested = Signal(object)
    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__("Test motion", parent)
        self.setObjectName("tuningMotionPanel")
        self.setMinimumWidth(330)
        self.setMaximumWidth(520)
        self.setStyleSheet(GROUP_STYLE + self._local_style())
        self._connection_available = False
        self._armed = False
        self._busy = False
        self._moving_axes: set[int] = set()
        self._build_ui()
        self._refresh_controls()
        self._set_status("Connect to a controller to test motion.")

    @staticmethod
    def _local_style() -> str:
        return f"""
            QLabel#motionWorkflow {{
                color: {TEXT_DIM}; background: {BG_CARD};
                border: 1px solid {BORDER}; border-radius: 3px;
                padding: 5px 7px; font-family: Consolas; font-size: 7pt;
            }}
            QLabel#motionHint {{ color: {TEXT_DIM}; font-size: 8pt; }}
            QLabel#motionState {{ color: {TEXT_DIM}; font-size: 8pt; }}
            QLabel#motionState[error="true"] {{ color: {RED}; }}
            QComboBox#motionAxis, QDoubleSpinBox#motionValue {{
                background: {BG_PANEL}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 3px;
                padding: 3px 5px; min-height: 20px;
            }}
            QPushButton#motionEnable {{
                background: {BG_PANEL}; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 3px;
                padding: 5px 8px; font-weight: 600;
            }}
            QPushButton#motionEnable:checked {{
                background: {GREEN}; color: white; border-color: {GREEN};
            }}
            QPushButton#motionMove {{
                background: {ACCENT}; color: #191919; border: none;
                border-radius: 3px; padding: 6px 7px; font-weight: 700;
            }}
            QPushButton#motionStop {{
                background: #653437; color: white; border: 1px solid {RED};
                border-radius: 3px; padding: 6px 7px; font-weight: 700;
            }}
            QPushButton:disabled {{ background: #454545; color: #777; border-color: #505050; }}
        """

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 7, 8, 8)
        outer.setSpacing(7)

        workflow = QLabel("1  SET PROFILE   →   2  ENABLE   →   3  MOVE")
        workflow.setObjectName("motionWorkflow")
        workflow.setAlignment(Qt.AlignCenter)
        outer.addWidget(workflow)

        hint = QLabel(
            "The selected axis is shared with the drive profile. Speed, ACCEL "
            "and DECEL are written immediately before every relative move."
        )
        hint.setObjectName("motionHint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)

        self.axis_combo = QComboBox()
        self.axis_combo.setObjectName("motionAxis")
        self.axis_combo.setAccessibleName("Tuning motion axis")
        for axis in range(16):
            self.axis_combo.addItem(f"Axis {axis}", axis)
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        form.addRow("Axis", self.axis_combo)

        self.distance_edit = self._value_editor(
            100.0, "Relative travel distance in the configured axis units."
        )
        form.addRow("Distance", self.distance_edit)

        self.speed_edit = self._value_editor(
            100.0, "SPEED written to the controller before this move."
        )
        form.addRow("Speed", self.speed_edit)

        self.acceleration_edit = self._value_editor(
            500.0,
            "ACCEL and DECEL written to the controller before this move.",
        )
        form.addRow("Acceleration", self.acceleration_edit)
        outer.addLayout(form)

        self.btn_enable = QPushButton("Enable axis")
        self.btn_enable.setObjectName("motionEnable")
        self.btn_enable.setCheckable(True)
        self.btn_enable.setToolTip(
            "Enable controller WDOG, SERVO and AXIS_ENABLE for the selected axis."
        )
        self.btn_enable.toggled.connect(self.request_enable)
        outer.addWidget(self.btn_enable)

        move_row = QHBoxLayout()
        move_row.setSpacing(5)
        self.btn_negative = QPushButton("←  Move −")
        self.btn_negative.setObjectName("motionMove")
        self.btn_negative.setToolTip("Move the entered distance in the negative direction.")
        self.btn_negative.clicked.connect(lambda: self.request_start(-1))
        move_row.addWidget(self.btn_negative, 1)

        self.btn_positive = QPushButton("Move +  →")
        self.btn_positive.setObjectName("motionMove")
        self.btn_positive.setToolTip("Move the entered distance in the positive direction.")
        self.btn_positive.clicked.connect(lambda: self.request_start(1))
        move_row.addWidget(self.btn_positive, 1)
        outer.addLayout(move_row)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("motionStop")
        self.btn_stop.setToolTip("Cancel active and buffered moves on this axis.")
        self.btn_stop.clicked.connect(self.request_stop)
        outer.addWidget(self.btn_stop)

        self.status_label = QLabel()
        self.status_label.setObjectName("motionState")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

    @staticmethod
    def _value_editor(value: float, tooltip: str) -> QDoubleSpinBox:
        editor = _MotionSpinBox()
        editor.setObjectName("motionValue")
        editor.setRange(0.001, 1_000_000_000_000.0)
        editor.setDecimals(6)
        editor.setSingleStep(1.0)
        editor.setValue(value)
        editor.setKeyboardTracking(False)
        editor.setToolTip(tooltip)
        return editor

    @property
    def axis(self) -> int:
        return int(self.axis_combo.currentData())

    @property
    def axis_locked(self) -> bool:
        return self._armed or self._busy

    def set_axis(self, axis: int) -> bool:
        """Synchronize the tuning axis, unless changing an armed axis."""
        index = self.axis_combo.findData(int(axis))
        if index < 0:
            return False
        if self.axis_locked and axis != self.axis:
            self._set_status("Disable the current axis before selecting another axis.", True)
            return False
        if index != self.axis_combo.currentIndex():
            self.axis_combo.blockSignals(True)
            self.axis_combo.setCurrentIndex(index)
            self.axis_combo.blockSignals(False)
        return True

    def commands(self) -> list[MotionAxisCommand]:
        command = MotionAxisCommand(
            axis=self.axis,
            speed=float(self.speed_edit.value()),
            distance=float(self.distance_edit.value()),
            acceleration=float(self.acceleration_edit.value()),
        )
        command.validate()
        return [command]

    def request_enable(self, enabled: bool) -> None:
        if enabled and not self._connection_available:
            self._set_enable_checked(False)
            self._set_status("Connect to a controller before enabling the axis.", True)
            return
        self._busy = True
        self._refresh_controls()
        self._set_status(f"{'Enabling' if enabled else 'Disabling'} axis {self.axis}…")
        self.enableRequested.emit(bool(enabled), self.commands())

    def complete_enable(self, requested_enabled: bool, error=None) -> None:
        self._busy = False
        if error is not None:
            # If disabling fails, retain the armed state because hardware state
            # is uncertain and the operator must retry or disconnect safely.
            self._armed = not requested_enabled
            self._set_enable_checked(self._armed)
            self._set_status(str(error), True)
        else:
            self._armed = bool(requested_enabled)
            if not self._armed:
                self._moving_axes.clear()
            self._set_enable_checked(self._armed)
            self._set_status(
                f"Axis {self.axis} enabled. Choose a move direction."
                if self._armed
                else f"Axis {self.axis} disabled."
            )
        self._refresh_controls()

    def request_start(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 or 1.")
        if not self._armed:
            self._set_status("Enable the axis before starting a move.", True)
            return
        if self._busy or self.axis in self._moving_axes:
            return
        base = self.commands()[0]
        command = MotionAxisCommand(
            axis=base.axis,
            speed=base.speed,
            distance=direction * abs(base.distance),
            acceleration=base.acceleration,
        )
        self._moving_axes.add(command.axis)
        self._refresh_controls()
        self.startRequested.emit([command])
        self._set_status(
            f"Axis {command.axis} moving. SPEED, ACCEL and DECEL were prepared first."
        )

    def complete_move(self, axes, error=None) -> None:
        completed_axes = set(axes)
        self._moving_axes.difference_update(completed_axes)
        self._refresh_controls()
        if error is not None:
            self._set_status(str(error), True)
        else:
            self._set_status(f"Axis {self.axis} move complete. Axis remains enabled.")

    def request_stop(self) -> None:
        if not self._armed or self._busy:
            return
        self._busy = True
        self._refresh_controls()
        self.stopRequested.emit()
        self._set_status(f"Stopping axis {self.axis}…")

    def complete_stop(self, error=None) -> None:
        self._moving_axes.clear()
        self._busy = False
        self._refresh_controls()
        self._set_status(
            str(error) if error is not None else "Move stopped. Axis remains enabled.",
            error is not None,
        )

    def set_connection_available(self, connected: bool) -> None:
        self._connection_available = bool(connected)
        if not connected:
            self._armed = False
            self._busy = False
            self._moving_axes.clear()
            self._set_enable_checked(False)
            self._set_status("Controller disconnected. Motion is unavailable.", True)
        elif not self._armed:
            self._set_status("Controller connected. Set the profile, then enable the axis.")
        self._refresh_controls()

    def request_safe_disable(self) -> None:
        if self._armed or (self._busy and self.btn_enable.isChecked()):
            self._set_enable_checked(False)
            self.request_enable(False)

    def focus_first_control(self) -> None:
        self.axis_combo.setFocus(Qt.ShortcutFocusReason)

    def _on_axis_changed(self) -> None:
        self.axisChanged.emit(self.axis)
        self._set_status(f"Axis {self.axis} selected for tuning and test motion.")

    def _refresh_controls(self) -> None:
        profile_editable = not self._busy
        self.axis_combo.setEnabled(not self._armed and not self._busy)
        self.distance_edit.setEnabled(profile_editable)
        self.speed_edit.setEnabled(profile_editable)
        self.acceleration_edit.setEnabled(profile_editable)
        can_move = (
            self._connection_available
            and self._armed
            and not self._busy
            and self.axis not in self._moving_axes
        )
        self.btn_negative.setEnabled(can_move)
        self.btn_positive.setEnabled(can_move)
        self.btn_enable.setEnabled(self._connection_available and not self._busy)
        self.btn_stop.setEnabled(
            self._connection_available
            and self._armed
            and bool(self._moving_axes)
            and not self._busy
        )

    def _set_enable_checked(self, checked: bool) -> None:
        self.btn_enable.blockSignals(True)
        self.btn_enable.setChecked(checked)
        self.btn_enable.setText("Axis enabled" if checked else "Enable axis")
        self.btn_enable.blockSignals(False)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("error", bool(error))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
