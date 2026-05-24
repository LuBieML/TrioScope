from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox,
    QCompleter, QPushButton, QLabel, QSpinBox, QSizePolicy, QColorDialog
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPen

from ui.theme import TRACE_COLORS
from scope.parameters import SCOPE_PARAMETERS, CHANNEL_PARAMETERS_SET, _VIRTUAL_PARAM_MAP
from scope.drive_scope_engine import COMMON_DRIVE_VARIABLES, DRIVE_VARIABLES


class TraceControl(QFrame):
    """Individual trace control (like one channel on an oscilloscope)"""

    changed = Signal()

    def __init__(self, trace_number, parent=None):
        super().__init__(parent)
        self.trace_number = trace_number
        self.color = TRACE_COLORS[trace_number % len(TRACE_COLORS)]

        # Colored border
        self.setStyleSheet(f"""
            TraceControl {{
                background-color: #353536;
                border: 2px solid {self.color};
                border-radius: 4px;
            }}
        """)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(3)

        # Row 0: Enable checkbox + parameter dropdown + delete button
        row0 = QHBoxLayout()
        row0.setSpacing(4)

        self.chk_enable = QCheckBox(f"Trace {trace_number + 1}")
        self.chk_enable.setStyleSheet(f"color: {self.color}; font-weight: bold;")
        self.chk_enable.toggled.connect(lambda: self.changed.emit())
        row0.addWidget(self.chk_enable)

        self.param_combo = QComboBox()
        self.param_combo.setEditable(True)
        self.param_combo.setInsertPolicy(QComboBox.NoInsert)
        self.param_combo.addItems(SCOPE_PARAMETERS)

        # Searchable dropdown: QCompleter with substring matching
        p_completer = QCompleter(SCOPE_PARAMETERS, self.param_combo)
        p_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        p_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        p_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        p_completer.popup().setStyleSheet(
            "QAbstractItemView {"
            "  background-color: #3a3a3a; color: #d4d4d4;"
            "  selection-background-color: #FFA500; selection-color: #000;"
            "  font-size: 9pt; border: 1px solid #666;"
            "}"
        )
        self.param_combo.setCompleter(p_completer)

        self.param_combo.setCurrentText("MPOS")
        self.param_combo.setMaxVisibleItems(20)
        self.param_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.param_combo.currentIndexChanged.connect(self._on_param_changed)
        row0.addWidget(self.param_combo, 1)

        # "Show all" button for parameter combo
        self.btn_show_params = QPushButton("\u25bc")
        self.btn_show_params.setFixedSize(20, 22)
        self.btn_show_params.setToolTip("Show all parameters")
        self.btn_show_params.setStyleSheet(
            "QPushButton { background-color: #4b4a4a; color: #ccc;"
            " border: 1px solid #606060; border-radius: 2px;"
            " font-size: 8pt; padding: 0px; }"
            "QPushButton:hover { background-color: #5a5a5a; }"
            "QPushButton:pressed { background-color: #666; }"
        )
        self.btn_show_params.clicked.connect(self._show_all_params)
        row0.addWidget(self.btn_show_params)

        # Drive variable combo (hidden by default)
        self.drive_var_combo = QComboBox()
        self.drive_var_combo.setEditable(True)
        self.drive_var_combo.setInsertPolicy(QComboBox.NoInsert)
        for addr, label in COMMON_DRIVE_VARIABLES:
            self.drive_var_combo.addItem(label, addr)

        drive_labels = [label for _, label in COMMON_DRIVE_VARIABLES]
        d_completer = QCompleter(drive_labels, self.drive_var_combo)
        d_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        d_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        d_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        d_completer.popup().setStyleSheet(
            "QAbstractItemView {"
            "  background-color: #3a3a3a; color: #d4d4d4;"
            "  selection-background-color: #FFA500; selection-color: #000;"
            "  font-size: 9pt; border: 1px solid #666;"
            "}"
        )
        self.drive_var_combo.setCompleter(d_completer)

        self.drive_var_combo.setMaxVisibleItems(20)
        self.drive_var_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.drive_var_combo.currentIndexChanged.connect(lambda: self.changed.emit())
        self.drive_var_combo.setVisible(False)
        row0.addWidget(self.drive_var_combo, 1)

        # "Show all" button for drive variable combo
        self.btn_show_drive_vars = QPushButton("\u25bc")
        self.btn_show_drive_vars.setFixedSize(20, 22)
        self.btn_show_drive_vars.setToolTip("Show all drive variables")
        self.btn_show_drive_vars.setStyleSheet(
            "QPushButton { background-color: #4b4a4a; color: #ccc;"
            " border: 1px solid #606060; border-radius: 2px;"
            " font-size: 8pt; padding: 0px; }"
            "QPushButton:hover { background-color: #5a5a5a; }"
            "QPushButton:pressed { background-color: #666; }"
        )
        self.btn_show_drive_vars.clicked.connect(self._show_all_drive_vars)
        self.btn_show_drive_vars.setVisible(False)
        row0.addWidget(self.btn_show_drive_vars)

        self.btn_popout = QPushButton("\u2197")
        self.btn_popout.setFixedSize(28, 22)
        self.btn_popout.setToolTip("Open this trace in a separate window")
        self.btn_popout.setEnabled(False)
        self.btn_popout.setStyleSheet("""
            QPushButton {
                background-color: #4b4a4a;
                color: #d4d4d4;
                border: 1px solid #606060;
                border-radius: 2px;
                font-size: 10pt;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover { background-color: #5a5a5a; }
            QPushButton:disabled {
                color: #666;
                background-color: #3a3a3a;
                border-color: #4a4a4a;
            }
        """)
        self.chk_enable.toggled.connect(self.btn_popout.setEnabled)
        row0.addWidget(self.btn_popout)

        self._drive_mode = False

        self.btn_delete = QPushButton("X")
        self.btn_delete.setFixedWidth(34)
        self.btn_delete.setStyleSheet("color: #ff4d4d; font-weight: bold; font-size: 10pt;")
        self.btn_delete.clicked.connect(self._on_delete)
        row0.addWidget(self.btn_delete)

        vbox.addLayout(row0)

        # Row 1: Axis selector + value display + FFT button
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        self.axis_label = QLabel("Axis")
        row1.addWidget(self.axis_label)
        self.axis_spin = QSpinBox()
        self.axis_spin.setRange(0, 15)
        self.axis_spin.setFixedWidth(28)
        self.axis_spin.setStyleSheet(
            "QSpinBox::up-button { width: 0; } QSpinBox::down-button { width: 0; }"
        )
        self.axis_spin.valueChanged.connect(lambda: self.changed.emit())
        row1.addWidget(self.axis_spin)

        _arrow_style = ("QPushButton { background-color: #4b4a4a; color: #ccc; "
                        "border: 1px solid #606060; border-radius: 2px; "
                        "font-size: 7pt; padding: 0px; }"
                        "QPushButton:pressed { background-color: #666; }")
        self.btn_ax_down = QPushButton("\u25bc")
        self.btn_ax_down.setFixedSize(18, 12)
        self.btn_ax_down.setStyleSheet(_arrow_style)
        self.btn_ax_down.clicked.connect(lambda: self.axis_spin.setValue(
            max(self.axis_spin.minimum(), self.axis_spin.value() - 1)))
        self.btn_ax_up = QPushButton("\u25b2")
        self.btn_ax_up.setFixedSize(18, 12)
        self.btn_ax_up.setStyleSheet(_arrow_style)
        self.btn_ax_up.clicked.connect(lambda: self.axis_spin.setValue(
            min(self.axis_spin.maximum(), self.axis_spin.value() + 1)))

        ax_arrows = QVBoxLayout()
        ax_arrows.setSpacing(1)
        ax_arrows.setContentsMargins(0, 0, 0, 0)
        ax_arrows.addWidget(self.btn_ax_up)
        ax_arrows.addWidget(self.btn_ax_down)
        row1.addLayout(ax_arrows)

        self.value_label = QLabel("0.0000")
        self.value_label.setObjectName("value_display")
        self.value_label.setStyleSheet(
            f"color: {self.color}; background-color: #2e2e2e; "
            f"font-family: Consolas; font-size: 9pt; font-weight: bold;"
        )
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setFixedWidth(140)
        row1.addWidget(self.value_label)

        self.btn_fft = QPushButton("FFT")
        self.btn_fft.setCheckable(True)
        self.btn_fft.setFixedSize(36, 22)
        self.btn_fft.setToolTip("Toggle FFT spectrum display for this trace")
        self.btn_fft.setStyleSheet("""
            QPushButton {
                background-color: #4b4a4a;
                color: #888;
                border: 1px solid #606060;
                border-radius: 2px;
                font-size: 8pt;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:checked {
                background-color: #8B4513;
                color: #FFA500;
                border: 1px solid #FFA500;
            }
        """)
        self.btn_fft.toggled.connect(lambda: self.changed.emit())
        row1.addWidget(self.btn_fft)

        self.btn_pin = QPushButton("PIN")
        self.btn_pin.setCheckable(True)
        self.btn_pin.setFixedSize(36, 22)
        self.btn_pin.setToolTip("Pin current trace as reference for comparison")
        self.btn_pin.setStyleSheet("""
            QPushButton {
                background-color: #4b4a4a;
                color: #888;
                border: 1px solid #606060;
                border-radius: 2px;
                font-size: 8pt;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:checked {
                background-color: #1a4a1a;
                color: #66FF66;
                border: 1px solid #66FF66;
            }
        """)
        row1.addWidget(self.btn_pin)

        # Default reference color — dimmed version of trace color
        qc = QColor(self.color)
        self.ref_color = QColor(
            (qc.red() + 128) // 2,
            (qc.green() + 128) // 2,
            (qc.blue() + 128) // 2,
        ).name()

        self.btn_ref_color = QPushButton()
        self.btn_ref_color.setFixedSize(22, 22)
        self.btn_ref_color.setToolTip("Choose reference trace color")
        self._update_ref_color_swatch()
        self.btn_ref_color.clicked.connect(self._pick_ref_color)
        row1.addWidget(self.btn_ref_color)

        # Reference (pinned) data: {'time': np.array, 'values': np.array} or None
        self.ref_data = None

        vbox.addLayout(row1)

    def _on_param_changed(self, index=None):
        """Update axis/channel label and range based on selected parameter."""
        is_ch = self.param_combo.currentText().strip() in CHANNEL_PARAMETERS_SET
        if is_ch:
            self.axis_label.setText("Ch")
            self.axis_spin.setRange(0, 1024)
            self.axis_spin.setFixedWidth(40)
        else:
            self.axis_label.setText("Axis")
            self.axis_spin.setRange(0, 15)
            self.axis_spin.setFixedWidth(28)
        self.changed.emit()

    def is_channel_parameter(self):
        """Return True if the currently selected parameter is a channel-type."""
        return self.param_combo.currentText().strip() in CHANNEL_PARAMETERS_SET

    def _on_delete(self):
        self.setParent(None)
        self.deleteLater()
        self.changed.emit()

    def _show_all_params(self):
        """Show full parameter dropdown without clearing the selected value."""
        self.param_combo.showPopup()

    def _show_all_drive_vars(self):
        """Show full drive variable dropdown without clearing the selected value."""
        self.drive_var_combo.showPopup()

    def is_enabled(self):
        return self.chk_enable.isChecked()

    def get_parameter_string(self):
        param = self.param_combo.currentText().strip()
        if not param:
            return ""
        # Virtual params capture their underlying Trio parameter from the controller
        trio_param = _VIRTUAL_PARAM_MAP.get(param, param)
        idx = self.axis_spin.value()
        if trio_param in CHANNEL_PARAMETERS_SET:
            return f"{trio_param}({idx})"
        return f"{trio_param} AXIS({idx})"

    def get_display_name(self):
        if self._drive_mode:
            return self.get_drive_display_name()
        param = self.param_combo.currentText().strip()
        if not param:
            return f"Trace {self.trace_number + 1} (no parameter)"
        idx = self.axis_spin.value()
        if param in CHANNEL_PARAMETERS_SET:
            return f"{param} Ch({idx})"
        return f"{param}({idx})"

    def update_value(self, value):
        self.value_label.setText(f"{value:>10.4f}")

    def is_fft(self):
        return self.btn_fft.isChecked()

    def set_fft(self, enabled):
        self.btn_fft.setChecked(enabled)

    def get_color(self):
        return self.color

    def is_pinned(self):
        return self.btn_pin.isChecked()

    def has_ref_data(self):
        return self.ref_data is not None

    def _update_ref_color_swatch(self):
        self.btn_ref_color.setStyleSheet(
            f"QPushButton {{ background-color: {self.ref_color};"
            f" border: 1px solid #606060; border-radius: 2px; }}"
            f"QPushButton:hover {{ border: 1px solid #ffffff; }}"
        )

    def _pick_ref_color(self):
        color = QColorDialog.getColor(
            QColor(self.ref_color), self, "Reference Trace Color")
        if color.isValid():
            self.ref_color = color.name()
            self._update_ref_color_swatch()
            self.changed.emit()

    def set_drive_mode(self, enabled: bool):
        """Switch between controller parameter and drive variable selection."""
        self._drive_mode = enabled
        self.param_combo.setVisible(not enabled)
        self.btn_show_params.setVisible(not enabled)
        self.drive_var_combo.setVisible(enabled)
        self.btn_show_drive_vars.setVisible(enabled)
        # Hide axis selector in drive mode (axis set globally)
        self.axis_label.setVisible(not enabled)
        self.axis_spin.setVisible(not enabled)
        self.btn_ax_down.setVisible(not enabled)
        self.btn_ax_up.setVisible(not enabled)

    def is_drive_mode(self):
        return self._drive_mode

    def get_drive_variable_address(self) -> int:
        """Return the selected drive variable address (0x0F10, etc.)."""
        return self.drive_var_combo.currentData()

    def get_drive_display_name(self) -> str:
        """Return display name for the drive variable."""
        addr = self.get_drive_variable_address()
        if addr and addr in DRIVE_VARIABLES:
            name = DRIVE_VARIABLES[addr][0]
            return f"{name} (0x{addr:04X})"
        return self.drive_var_combo.currentText()
