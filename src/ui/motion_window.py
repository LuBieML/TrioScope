"""Modeless editor for coordinated controller-axis move requests."""

from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import MOTION_WINDOW_STYLESHEET
try:
    from ..models.motion_axis_command import MotionAxisCommand
except ImportError:  # App runtime imports ui as a top-level package.
    from models.motion_axis_command import MotionAxisCommand


class _MotionValueSpinBox(QDoubleSpinBox):
    """Wide-range motion input that avoids unnecessary trailing zeroes."""

    def textFromValue(self, value: float) -> str:
        return format(value, ".12g")


@dataclass
class _MotionRow:
    frame: QFrame
    index_label: QLabel
    axis_combo: QComboBox
    speed_edit: QDoubleSpinBox
    distance_edit: QDoubleSpinBox
    negative_button: QPushButton
    positive_button: QPushButton
    remove_button: QPushButton


class AxisMotionWindow(QMainWindow):
    """Build and dispatch a multi-axis move scheme."""

    enableRequested = Signal(bool, object)
    startRequested = Signal(object)
    stopRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Axis Motion")
        self.setMinimumSize(680, 410)
        self.resize(760, 500)
        self._rows: List[_MotionRow] = []
        self._connection_available = False
        self._armed = False
        self._moving_axes = set()
        self._busy = False
        self._build_ui()
        self.add_axis()
        self._set_status("Connect to a controller to enable motion.")

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("motionRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        title = QLabel("Axis motion")
        title.setObjectName("motionTitle")
        subtitle = QLabel(
            "Configure independent axis moves. Enter speed and distance magnitudes, then choose direction."
        )
        subtitle.setObjectName("motionSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack, 1)

        self.btn_add = QPushButton("+  Add axis")
        self.btn_add.setObjectName("motionAddButton")
        self.btn_add.setToolTip("Add another axis to this move scheme.")
        self.btn_add.clicked.connect(lambda: self.add_axis())
        header.addWidget(self.btn_add)
        layout.addLayout(header)

        accent_rule = QFrame()
        accent_rule.setObjectName("motionAccentRule")
        accent_rule.setFixedHeight(2)
        layout.addWidget(accent_rule)

        column_header = QHBoxLayout()
        column_header.setContentsMargins(13, 0, 13, 0)
        column_header.setSpacing(10)
        column_header.addWidget(self._column_label("ROW", 48))
        column_header.addWidget(self._column_label("AXIS", 100))
        column_header.addWidget(self._column_label("SPEED", 190), 1)
        column_header.addWidget(self._column_label("DISTANCE", 190), 1)
        column_header.addWidget(self._column_label("DIRECTION", 90))
        column_header.addSpacing(74)
        layout.addLayout(column_header)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("motionScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.rows_container = QWidget()
        self.rows_container.setObjectName("motionRows")
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(8)
        self.rows_layout.addStretch()
        self.scroll.setWidget(self.rows_container)
        layout.addWidget(self.scroll, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.status_label = QLabel("Connect to a controller to enable motion.")
        self.status_label.setObjectName("motionStatus")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)

        self.btn_enable = QPushButton("Enable")
        self.btn_enable.setObjectName("motionEnableButton")
        self.btn_enable.setCheckable(True)
        self.btn_enable.setMinimumWidth(96)
        self.btn_enable.setToolTip(
            "Toggle controller WDOG, SERVO and AXIS_ENABLE for every listed axis."
        )
        self.btn_enable.setEnabled(False)
        self.btn_enable.toggled.connect(self.request_enable)
        footer.addWidget(self.btn_enable)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("motionStopButton")
        self.btn_stop.setMinimumWidth(96)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.request_stop)
        footer.addWidget(self.btn_stop)

        layout.addLayout(footer)

        self.setStyleSheet(MOTION_WINDOW_STYLESHEET)

    @staticmethod
    def _column_label(text: str, width: int) -> QLabel:
        label = QLabel(text)
        label.setObjectName("motionColumnLabel")
        label.setFixedWidth(width)
        return label

    def add_axis(self, command: Optional[MotionAxisCommand] = None) -> bool:
        used_axes = {int(row.axis_combo.currentData()) for row in self._rows}
        if command is None:
            axis = next((value for value in range(26) if value not in used_axes), None)
            if axis is None:
                self._set_status("All controller axes (0–25) are already in this move.", True)
                return False
            command = MotionAxisCommand(axis=axis)
        else:
            command.validate()
            if command.axis in used_axes:
                self._set_status(f"Axis {command.axis} is already in this move.", True)
                return False

        frame = QFrame()
        frame.setObjectName("motionAxisRow")
        row_layout = QHBoxLayout(frame)
        row_layout.setContentsMargins(12, 10, 12, 10)
        row_layout.setSpacing(10)

        index_label = QLabel()
        index_label.setObjectName("motionMoveIndex")
        index_label.setAlignment(Qt.AlignCenter)
        index_label.setFixedWidth(48)
        row_layout.addWidget(index_label)

        axis_combo = QComboBox()
        axis_combo.setObjectName("motionAxisCombo")
        axis_combo.setFixedWidth(100)
        for axis_number in range(26):
            axis_combo.addItem(f"Axis {axis_number}", axis_number)
        axis_combo.setCurrentIndex(command.axis)
        axis_combo.setProperty("previousAxis", command.axis)
        row_layout.addWidget(axis_combo)

        speed_edit = self._make_value_editor(
            minimum=0.001,
            maximum=1_000_000_000_000.0,
            value=command.speed,
            prefix="",
            tooltip="Positive commanded speed for this axis.",
        )
        row_layout.addWidget(speed_edit, 1)

        distance_edit = self._make_value_editor(
            minimum=0.0,
            maximum=1_000_000_000_000.0,
            value=abs(command.distance),
            prefix="",
            tooltip="Relative move distance magnitude. Choose direction with an arrow button.",
        )
        row_layout.addWidget(distance_edit, 1)

        negative_button = QPushButton("←")
        negative_button.setObjectName("motionRowStartButton")
        negative_button.setFixedWidth(40)
        negative_button.setToolTip("Move by the entered distance in the negative direction.")
        row_layout.addWidget(negative_button)

        positive_button = QPushButton("→")
        positive_button.setObjectName("motionRowStartButton")
        positive_button.setFixedWidth(40)
        positive_button.setToolTip("Move by the entered distance in the positive direction.")
        row_layout.addWidget(positive_button)

        remove_button = QPushButton("Remove")
        remove_button.setObjectName("motionRemoveButton")
        remove_button.setFixedWidth(74)
        remove_button.setToolTip("Remove this axis from the move scheme.")
        row_layout.addWidget(remove_button)

        row = _MotionRow(
            frame,
            index_label,
            axis_combo,
            speed_edit,
            distance_edit,
            negative_button,
            positive_button,
            remove_button,
        )
        self._rows.append(row)
        self.rows_layout.insertWidget(self.rows_layout.count() - 1, frame)
        axis_combo.currentIndexChanged.connect(lambda _=0, r=row: self._on_axis_changed(r))
        speed_edit.valueChanged.connect(self._mark_edited)
        distance_edit.valueChanged.connect(self._mark_edited)
        negative_button.clicked.connect(
            lambda _=False, r=row: self.request_start(r, -1)
        )
        positive_button.clicked.connect(
            lambda _=False, r=row: self.request_start(r, 1)
        )
        remove_button.clicked.connect(lambda _=False, r=row: self._remove_row(r))

        self._refresh_row_indices()
        self._refresh_controls()
        self._set_status(f"Axis {command.axis} added to the move scheme.")
        return True

    @staticmethod
    def _make_value_editor(
        minimum: float,
        maximum: float,
        value: float,
        prefix: str,
        tooltip: str,
    ) -> QDoubleSpinBox:
        editor = _MotionValueSpinBox()
        editor.setRange(minimum, maximum)
        editor.setDecimals(6)
        editor.setSingleStep(1.0)
        editor.setValue(value)
        editor.setPrefix(prefix)
        editor.setKeyboardTracking(False)
        editor.setToolTip(tooltip)
        editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        editor.setMinimumWidth(140)
        return editor

    def commands(self) -> List[MotionAxisCommand]:
        commands = [
            MotionAxisCommand(
                axis=int(row.axis_combo.currentData()),
                speed=float(row.speed_edit.value()),
                distance=float(row.distance_edit.value()),
            )
            for row in self._rows
        ]
        for command in commands:
            command.validate()
        return commands

    def set_commands(self, commands: List[MotionAxisCommand]) -> None:
        axes = [command.axis for command in commands]
        if len(axes) != len(set(axes)):
            raise ValueError("Each axis can only appear once in a move scheme.")
        for command in commands:
            command.validate()

        for row in self._rows:
            self.rows_layout.removeWidget(row.frame)
            row.frame.setParent(None)
            row.frame.deleteLater()
        self._rows.clear()
        for command in commands:
            self.add_axis(command)
        self._refresh_row_indices()
        self._refresh_controls()
        self._set_status(
            f"Loaded {len(commands)} axis move{'s' if len(commands) != 1 else ''}."
        )

    def remove_axis(self, axis: int) -> bool:
        row = next(
            (candidate for candidate in self._rows if candidate.axis_combo.currentData() == axis),
            None,
        )
        if row is None:
            return False
        self._remove_row(row)
        return True

    def request_enable(self, enabled: bool) -> None:
        if enabled and not self._connection_available:
            self._set_enable_checked(False)
            self._set_status("Connect to a controller before enabling axes.", True)
            return
        commands = self.commands()
        if enabled and not commands:
            self._set_enable_checked(False)
            self._set_status("Add at least one axis before enabling motion.", True)
            return

        self._busy = True
        self._refresh_controls()
        action = "Enabling" if enabled else "Disabling"
        axes = ", ".join(str(command.axis) for command in commands)
        self._set_status(f"{action} axes {axes}…")
        self.enableRequested.emit(enabled, commands)

    def complete_enable(self, requested_enabled: bool, error=None) -> None:
        """Apply the asynchronous UAPI enable/disable result on the UI thread."""
        self._busy = False
        if error is not None:
            # A failed disable leaves the hardware state uncertain, so keep the
            # window armed and make the operator retry or disconnect safely.
            self._armed = not requested_enabled
            self._set_enable_checked(self._armed)
            self._set_status(str(error), True)
        else:
            self._armed = requested_enabled
            if not self._armed:
                self._moving_axes.clear()
            self._set_enable_checked(self._armed)
            axes = ", ".join(str(command.axis) for command in self.commands())
            self._set_status(
                f"Axes {axes} enabled. Ready to move."
                if self._armed
                else f"Axes {axes} disabled."
            )
        self._refresh_controls()

    def request_start(self, row: _MotionRow, direction: int) -> None:
        command = self._command_for_row(row, direction)
        if not self._armed:
            self._set_status("Enable the involved axes before starting a move.", True)
            return
        if command.axis in self._moving_axes or self._busy:
            return
        self._moving_axes.add(command.axis)
        self._refresh_controls()
        self.startRequested.emit([command])
        self._set_status(f"MOVE sent to axis {command.axis}. Monitoring motion…")

    def request_stop(self) -> None:
        if not self._armed or self._busy:
            return
        self._busy = True
        self._refresh_controls()
        self.stopRequested.emit()
        self._set_status("Stopping involved axes…")

    def complete_move(self, axes, error=None) -> None:
        completed_axes = set(axes)
        self._moving_axes.difference_update(completed_axes)
        self._refresh_controls()
        axes_text = ", ".join(str(axis) for axis in sorted(completed_axes))
        if error is not None:
            self._set_status(str(error), True)
        else:
            noun = "Axis" if len(completed_axes) == 1 else "Axes"
            self._set_status(f"{noun} {axes_text} move complete. Axes remain enabled.")

    def complete_stop(self, error=None) -> None:
        self._moving_axes.clear()
        self._busy = False
        self._refresh_controls()
        if error is not None:
            self._set_status(str(error), True)
        else:
            self._set_status("Move stopped. Axes remain enabled.")

    def set_connection_available(self, connected: bool) -> None:
        self._connection_available = connected
        if not connected:
            self._armed = False
            self._moving_axes.clear()
            self._busy = False
            self._set_enable_checked(False)
            self._set_status("Controller disconnected. Motion is unavailable.", True)
        elif not self._armed:
            self._set_status("Controller connected. Enable the involved axes to move.")
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        structure_editable = not self._armed and not self._busy
        values_editable = not self._busy
        for row in self._rows:
            row.axis_combo.setEnabled(structure_editable)
            row.speed_edit.setEnabled(values_editable)
            row.distance_edit.setEnabled(values_editable)
            row.remove_button.setEnabled(structure_editable)
            axis = int(row.axis_combo.currentData())
            can_start = (
                self._connection_available
                and self._armed
                and axis not in self._moving_axes
                and not self._busy
            )
            row.negative_button.setEnabled(can_start)
            row.positive_button.setEnabled(can_start)
        self.btn_add.setEnabled(structure_editable and len(self._rows) < 26)
        self.btn_enable.setEnabled(
            self._connection_available and bool(self._rows) and not self._busy
        )
        self.btn_stop.setEnabled(
            self._connection_available
            and self._armed
            and bool(self._moving_axes)
            and not self._busy
        )

    @staticmethod
    def _command_for_row(row: _MotionRow, direction: int) -> MotionAxisCommand:
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 or 1.")
        command = MotionAxisCommand(
            axis=int(row.axis_combo.currentData()),
            speed=float(row.speed_edit.value()),
            distance=direction * abs(float(row.distance_edit.value())),
        )
        command.validate()
        return command

    def _set_enable_checked(self, checked: bool) -> None:
        self.btn_enable.blockSignals(True)
        self.btn_enable.setChecked(checked)
        self.btn_enable.setText("Enabled" if checked else "Enable")
        self.btn_enable.blockSignals(False)

    def _on_axis_changed(self, row: _MotionRow) -> None:
        new_axis = int(row.axis_combo.currentData())
        previous_axis = int(row.axis_combo.property("previousAxis"))
        if any(
            other is not row and int(other.axis_combo.currentData()) == new_axis
            for other in self._rows
        ):
            row.axis_combo.blockSignals(True)
            row.axis_combo.setCurrentIndex(previous_axis)
            row.axis_combo.blockSignals(False)
            self._set_status(f"Axis {new_axis} is already in this move.", True)
            return
        row.axis_combo.setProperty("previousAxis", new_axis)
        self._set_status(f"Axis {previous_axis} changed to axis {new_axis}.")

    def _remove_row(self, row: _MotionRow) -> None:
        axis = int(row.axis_combo.currentData())
        self._rows.remove(row)
        self.rows_layout.removeWidget(row.frame)
        row.frame.setParent(None)
        row.frame.deleteLater()
        self._refresh_row_indices()
        self._refresh_controls()
        if self._rows:
            self._set_status(f"Axis {axis} removed from the move scheme.")
        else:
            self._set_status("No axes in this move. Add an axis to continue.")

    def _refresh_row_indices(self) -> None:
        for index, row in enumerate(self._rows, start=1):
            row.index_label.setText(f"{index:02d}")

    def _mark_edited(self, _value: float) -> None:
        if not self._armed and not self._busy:
            self._set_status("Move scheme edited. Ready to start.")

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("error", error)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def closeEvent(self, event) -> None:
        if self._armed or (self._busy and self.btn_enable.isChecked()):
            self.enableRequested.emit(False, self.commands())
            self._armed = False
            self._moving_axes.clear()
        super().closeEvent(event)
