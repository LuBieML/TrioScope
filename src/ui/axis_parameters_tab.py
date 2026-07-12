"""Dynamic editor for controller axis motion parameters."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from ..models.axis_parameter_config import AxisParameterConfig
    from ..storage.axis_config_io import load_axis_config, save_axis_config
except ImportError:  # App runtime imports ui as a top-level package.
    from models.axis_parameter_config import AxisParameterConfig
    from storage.axis_config_io import load_axis_config, save_axis_config


logger = logging.getLogger(__name__)


PARAMETER_COLUMNS = (
    ("speed", "Speed", "Maximum commanded axis speed."),
    ("units", "Units", "Controller UNITS scaling for this axis."),
    ("accel", "Accel", "Acceleration used for axis moves."),
    ("decel", "Decel", "Normal deceleration used for axis moves."),
    ("fast_dec", "FastDec", "Fast deceleration used for stops."),
    ("jerk", "Jerk", "Jerk limit used for S-curve motion."),
    ("fwd_in", "Fwd In", "Forward travel input assignment; -1 disables it."),
    ("rev_in", "Rev In", "Reverse travel input assignment; -1 disables it."),
    ("fe_limit", "FE Limit", "Following-error limit for the axis."),
)


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Wide-range numeric editor without forced trailing zeroes."""

    def textFromValue(self, value: float) -> str:
        return format(value, ".12g")


@dataclass
class _AxisRow:
    axis_combo: QComboBox
    editors: Dict[str, CompactDoubleSpinBox]
    copy_combo: QComboBox
    copy_button: QPushButton
    remove_button: QPushButton


class AxisParametersTab(QWidget):
    """A horizontal commissioning grid for axes 0 through 25."""

    configurationChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[_AxisRow] = []
        self._connection = None
        self._connection_lock = None
        self._build_ui()

    def minimumSizeHint(self) -> QSize:
        return QSize(760, 420)

    def _build_ui(self) -> None:
        self.setObjectName("axisParametersTab")
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetNoConstraint)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        title = QLabel("Axis setup")
        title.setObjectName("axisSetupTitle")
        subtitle = QLabel(
            "Per-axis speed, acceleration and travel limits used by controller motion."
        )
        subtitle.setObjectName("axisSetupSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack)
        header.addStretch()

        self.btn_add = QPushButton("+  Add axis")
        self.btn_add.setObjectName("axisAddButton")
        self.btn_add.clicked.connect(lambda: self.add_axis())
        header.addWidget(self.btn_add)

        self.btn_load = QPushButton("Load config")
        self.btn_load.clicked.connect(lambda: self.load_config())
        header.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save config")
        self.btn_save.clicked.connect(lambda: self.save_config())
        header.addWidget(self.btn_save)

        self.btn_send = QPushButton("Send to controller")
        self.btn_send.setObjectName("accent")
        self.btn_send.setEnabled(False)
        self.btn_send.setToolTip("Connect to a Trio controller to enable this action.")
        self.btn_send.clicked.connect(self.send_to_controller)
        header.addWidget(self.btn_send)
        layout.addLayout(header)

        accent_rule = QFrame()
        accent_rule.setObjectName("axisAccentRule")
        accent_rule.setFixedHeight(2)
        layout.addWidget(accent_rule)

        self.table = QTableWidget(0, 1 + len(PARAMETER_COLUMNS) + 2)
        self.table.setObjectName("axisParametersTable")
        self.table.setHorizontalHeaderLabels(
            ["Axis"]
            + [label for _, label, _ in PARAMETER_COLUMNS]
            + ["Copy values", ""]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(48)
        # The grid is deliberately wider than a typical laptop window. Ignore
        # its width hint so Qt uses the horizontal scrollbar instead of forcing
        # the whole application to the sum of all column widths.
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

        horizontal = self.table.horizontalHeader()
        horizontal.setSectionResizeMode(QHeaderView.Fixed)
        horizontal.setStretchLastSection(False)
        self.table.setColumnWidth(0, 92)
        for column, (_, _, tooltip) in enumerate(PARAMETER_COLUMNS, start=1):
            self.table.setColumnWidth(column, 126 if column != 2 else 146)
            item = self.table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tooltip)
        self.table.setColumnWidth(1 + len(PARAMETER_COLUMNS), 210)
        self.table.setColumnWidth(2 + len(PARAMETER_COLUMNS), 76)
        layout.addWidget(self.table, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("Add an axis to begin.")
        self.status_label.setObjectName("axisSetupStatus")
        footer.addWidget(self.status_label)
        footer.addStretch()
        self.connection_label = QLabel("●  Controller offline")
        self.connection_label.setObjectName("axisConnectionStatus")
        footer.addWidget(self.connection_label)
        layout.addLayout(footer)

        self.setStyleSheet(
            """
            QWidget#axisParametersTab { background-color: #24272b; }
            QLabel#axisSetupTitle { color: #f0f2f4; font-size: 18pt; font-weight: 600; }
            QLabel#axisSetupSubtitle { color: #9aa3ad; font-size: 9pt; }
            QFrame#axisAccentRule { background-color: #f3a712; border: none; }
            QPushButton#axisAddButton { color: #f3b63f; font-weight: 600; }
            QTableWidget#axisParametersTable {
                background-color: #24272b;
                alternate-background-color: #292d32;
                color: #d7dce1;
                border: 1px solid #383e45;
                selection-background-color: #3b3b30;
                selection-color: #ffffff;
            }
            QTableWidget#axisParametersTable QHeaderView::section {
                background-color: #30353b;
                color: #f3b63f;
                border: none;
                border-right: 1px solid #3c4249;
                border-bottom: 1px solid #f3a712;
                padding: 8px 6px;
                font-weight: 600;
            }
            QTableWidget#axisParametersTable QDoubleSpinBox,
            QTableWidget#axisParametersTable QComboBox {
                background-color: #202328;
                color: #e1e5e9;
                border: 1px solid #414851;
                border-radius: 3px;
                padding: 4px 6px;
                font-family: 'Consolas';
            }
            QLabel#axisSetupStatus { color: #9aa3ad; }
            QLabel#axisConnectionStatus { color: #d06b64; font-weight: 600; }
            """
        )

    def add_axis(self, config: Optional[AxisParameterConfig] = None) -> bool:
        used_axes = {row.axis_combo.currentData() for row in self._rows}
        if config is None:
            axis = next((candidate for candidate in range(26) if candidate not in used_axes), None)
            if axis is None:
                self._set_status("All controller axes (0–25) are already configured.", error=True)
                return False
            config = AxisParameterConfig(axis=axis)
        else:
            config.validate()
            if config.axis in used_axes:
                self._set_status(f"Axis {config.axis} is already configured.", error=True)
                return False

        table_row = self.table.rowCount()
        self.table.insertRow(table_row)

        axis_combo = QComboBox()
        for axis_number in range(26):
            axis_combo.addItem(str(axis_number), axis_number)
        axis_combo.setCurrentIndex(config.axis)
        axis_combo.setProperty("previousAxis", config.axis)
        self.table.setCellWidget(table_row, 0, axis_combo)

        editors: Dict[str, CompactDoubleSpinBox] = {}
        for column, (field_name, _, tooltip) in enumerate(PARAMETER_COLUMNS, start=1):
            editor = CompactDoubleSpinBox()
            editor.setRange(-1_000_000_000_000.0, 1_000_000_000_000.0)
            editor.setDecimals(6)
            editor.setSingleStep(1.0)
            editor.setKeyboardTracking(False)
            editor.setValue(float(getattr(config, field_name)))
            editor.setToolTip(tooltip)
            editor.valueChanged.connect(self.configurationChanged)
            editors[field_name] = editor
            self.table.setCellWidget(table_row, column, editor)

        copy_container = QWidget()
        copy_layout = QHBoxLayout(copy_container)
        copy_layout.setContentsMargins(2, 2, 2, 2)
        copy_layout.setSpacing(4)
        copy_combo = QComboBox()
        copy_combo.setToolTip("Choose another configured axis as the source.")
        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy every parameter value from the selected axis.")
        copy_layout.addWidget(copy_combo, 1)
        copy_layout.addWidget(copy_button)
        self.table.setCellWidget(table_row, 1 + len(PARAMETER_COLUMNS), copy_container)

        remove_button = QPushButton("Remove")
        remove_button.setToolTip("Remove this axis from the configuration.")
        self.table.setCellWidget(table_row, 2 + len(PARAMETER_COLUMNS), remove_button)

        row = _AxisRow(axis_combo, editors, copy_combo, copy_button, remove_button)
        self._rows.append(row)
        axis_combo.currentIndexChanged.connect(lambda _=0, r=row: self._on_axis_changed(r))
        copy_button.clicked.connect(lambda _=False, r=row: self._copy_from_selected(r))
        remove_button.clicked.connect(lambda _=False, r=row: self._remove_row(r))

        self._refresh_copy_sources()
        self.btn_add.setEnabled(len(self._rows) < 26)
        self._set_status(f"Axis {config.axis} added.")
        self.configurationChanged.emit()
        return True

    def configurations(self) -> List[AxisParameterConfig]:
        configs = []
        for row in self._rows:
            values = {name: editor.value() for name, editor in row.editors.items()}
            config = AxisParameterConfig(axis=int(row.axis_combo.currentData()), **values)
            config.validate()
            configs.append(config)
        return configs

    def set_configurations(self, configs: List[AxisParameterConfig]) -> None:
        axes = [config.axis for config in configs]
        if len(axes) != len(set(axes)):
            raise ValueError("Each axis can only be configured once.")

        self.table.setRowCount(0)
        self._rows.clear()
        for config in configs:
            self.add_axis(config)
        self.btn_add.setEnabled(len(self._rows) < 26)
        noun = "axis" if len(configs) == 1 else "axes"
        self._set_status(f"Loaded {len(configs)} configured {noun}.")
        self.configurationChanged.emit()

    def copy_axis_values(self, source_axis: int, target_axis: int) -> None:
        source = self._row_for_axis(source_axis)
        target = self._row_for_axis(target_axis)
        if source is None or target is None:
            raise ValueError("Both source and target axes must already be configured.")
        if source is target:
            raise ValueError("Source and target axes must be different.")

        for field_name, editor in target.editors.items():
            editor.setValue(source.editors[field_name].value())
        self._set_status(f"Copied all values from axis {source_axis} to axis {target_axis}.")
        self.configurationChanged.emit()

    def save_config(self, path: Optional[str] = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save axis configuration",
                "axis_parameters.json",
                "JSON files (*.json);;All files (*)",
            )
        if not path:
            return False
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_axis_config(path, self.configurations())
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save configuration", str(exc))
            self._set_status("Configuration was not saved.", error=True)
            return False
        self._set_status(f"Saved {len(self._rows)} axes to {path}.")
        return True

    def load_config(self, path: Optional[str] = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load axis configuration",
                "",
                "JSON files (*.json);;All files (*)",
            )
        if not path:
            return False
        try:
            configs = load_axis_config(path)
            self.set_configurations(configs)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not load configuration", str(exc))
            self._set_status("Configuration was not loaded.", error=True)
            return False
        self._set_status(f"Loaded {len(configs)} axes from {path}.")
        return True

    def set_connection(self, connection, connection_lock=None) -> None:
        self._connection = connection
        self._connection_lock = connection_lock
        connected = connection is not None
        self.btn_send.setEnabled(connected)
        if connected:
            self.connection_label.setText("●  Controller connected")
            self.connection_label.setStyleSheet("color: #62c980; font-weight: 600;")
            self.btn_send.setToolTip("Send every configured row to the connected controller.")
        else:
            self.connection_label.setText("●  Controller offline")
            self.connection_label.setStyleSheet("color: #d06b64; font-weight: 600;")
            self.btn_send.setToolTip("Connect to a Trio controller to enable this action.")

    def send_to_controller(self) -> None:
        if self._connection is None:
            self._set_status("Connect to a controller before sending values.", error=True)
            return
        configs = self.configurations()
        try:
            self._write_axis_parameters(configs)
        except NotImplementedError:
            QMessageBox.information(
                self,
                "Controller write placeholder",
                "The axis configuration is ready. The UAPI parameter-write mapping "
                "will be connected here in a future update; no controller values were changed.",
            )
            self._set_status("UAPI write placeholder reached; no values were sent.")

    def _write_axis_parameters(self, configs: List[AxisParameterConfig]) -> None:
        """Future UAPI boundary; deliberately performs no controller writes yet."""
        logger.info(
            "Axis parameter UAPI placeholder called for axes %s",
            [config.axis for config in configs],
        )
        raise NotImplementedError

    def _on_axis_changed(self, row: _AxisRow) -> None:
        new_axis = int(row.axis_combo.currentData())
        previous_axis = int(row.axis_combo.property("previousAxis"))
        if any(other is not row and other.axis_combo.currentData() == new_axis for other in self._rows):
            row.axis_combo.blockSignals(True)
            row.axis_combo.setCurrentIndex(previous_axis)
            row.axis_combo.blockSignals(False)
            self._set_status(f"Axis {new_axis} is already configured.", error=True)
            return
        row.axis_combo.setProperty("previousAxis", new_axis)
        self._refresh_copy_sources()
        self._set_status(f"Axis {previous_axis} changed to axis {new_axis}.")
        self.configurationChanged.emit()

    def _copy_from_selected(self, row: _AxisRow) -> None:
        source_axis = row.copy_combo.currentData()
        if source_axis is None:
            return
        self.copy_axis_values(int(source_axis), int(row.axis_combo.currentData()))

    def _remove_row(self, row: _AxisRow) -> None:
        index = self._rows.index(row)
        axis = int(row.axis_combo.currentData())
        self._rows.pop(index)
        self.table.removeRow(index)
        self._refresh_copy_sources()
        self.btn_add.setEnabled(True)
        self._set_status(f"Axis {axis} removed.")
        self.configurationChanged.emit()

    def _row_for_axis(self, axis: int) -> Optional[_AxisRow]:
        return next(
            (row for row in self._rows if int(row.axis_combo.currentData()) == axis),
            None,
        )

    def _refresh_copy_sources(self) -> None:
        for row in self._rows:
            previous = row.copy_combo.currentData()
            source_axes = sorted(
                int(other.axis_combo.currentData())
                for other in self._rows
                if other is not row
            )
            row.copy_combo.blockSignals(True)
            row.copy_combo.clear()
            if source_axes:
                for axis in source_axes:
                    row.copy_combo.addItem(f"Axis {axis}", axis)
                if previous in source_axes:
                    row.copy_combo.setCurrentIndex(source_axes.index(previous))
            else:
                row.copy_combo.addItem("No source axis", None)
            row.copy_combo.blockSignals(False)
            row.copy_button.setEnabled(bool(source_axes))

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #d06b64;" if error else "color: #9aa3ad;")
