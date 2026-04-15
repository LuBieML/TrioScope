"""
Servo Loop Analyser Panel — dockable Qt widget for scope-based loop diagnostics.

Combines:
  - Drive profile editor (axis selector, Pn parameter spinboxes, CoE Read/Write)
  - Velocity loop metrics card
  - Position loop metrics card (including following-error during motion)
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Callable

import numpy as np

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFormLayout, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import (
    QColor, QPainter, QBrush, QRadialGradient,
)

from .classical_tuner import (
    ClassicalTuner, StepResponseMetrics, VelocityLoopMetrics,
)
from .drive_profile import (
    DriveProfile, DRIVE_TYPES, PARAM_DEFS, COMBO_ATTRS,
    TUNING_MODE_LABELS, TUNING_MODE_VALUES,
    VIBRATION_SUPPRESSION_LABELS, VIBRATION_SUPPRESSION_VALUES,
    DAMPING_LABELS, DAMPING_VALUES,
)
from .coe_io import read_drive_profile, write_drive_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_BG_DARK      = "#1a1a22"
_BG_CARD      = "#22222c"
_BG_PANEL     = "#2a2a36"
_BORDER       = "#3a3a4a"
_BORDER_LIGHT = "#4b4b5a"
_TEXT          = "#d4d4d4"
_TEXT_DIM      = "#888899"
_TEXT_BRIGHT   = "#f0f0f5"
_ACCENT        = "#FFA500"
_CYAN          = "#00d4aa"
_GREEN         = "#2ecc71"
_AMBER         = "#f39c12"
_RED           = "#e74c3c"


def _health_color(healthy: bool | None) -> str:
    if healthy is None:
        return _TEXT_DIM
    return _GREEN if healthy else _RED


# ---------------------------------------------------------------------------
# Metric row widget
# ---------------------------------------------------------------------------
def _metric_label(name: str, value: str, unit: str = "",
                  color: str = _CYAN) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    lbl_name = QLabel(name)
    lbl_name.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 8pt;")
    lbl_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    row.addWidget(lbl_name)
    lbl_val = QLabel(value)
    lbl_val.setStyleSheet(
        f"color: {color}; font-family: Consolas; font-size: 9pt; font-weight: bold;"
    )
    lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    row.addWidget(lbl_val)
    if unit:
        lbl_unit = QLabel(unit)
        lbl_unit.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 7pt;")
        lbl_unit.setFixedWidth(28)
        row.addWidget(lbl_unit)
    return row


def _separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet(f"color: {_BORDER};")
    sep.setFixedHeight(1)
    return sep


def _card_frame() -> QFrame:
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame {{ background-color: {_BG_CARD}; border: 1px solid {_BORDER};"
        f" border-radius: 6px; }}"
    )
    return frame


# ---------------------------------------------------------------------------
# Health indicator dot
# ---------------------------------------------------------------------------
class _HealthDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(_TEXT_DIM)
        self.setFixedSize(14, 14)

    def set_healthy(self, healthy: bool | None):
        if healthy is None:
            self._color = QColor(_TEXT_DIM)
        elif healthy:
            self._color = QColor(_GREEN)
        else:
            self._color = QColor(_RED)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        glow = QRadialGradient(7, 7, 9)
        glow.setColorAt(0, QColor(self._color.red(), self._color.green(),
                                   self._color.blue(), 80))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 14, 14)
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(3, 3, 8, 8)
        painter.end()


# ---------------------------------------------------------------------------
# Thread-safe signals for CoE operations
# ---------------------------------------------------------------------------
class _CoESignals(QObject):
    coe_read_done = Signal(int, object, str)
    coe_write_done = Signal(int, object, str)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
class TunerPanel(QDockWidget):
    """Dockable servo loop analyser panel with drive profile editor."""

    analysis_complete = Signal()

    def __init__(self, parent=None):
        super().__init__("Servo Loop Analyser", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(560)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # --- State ---
        self._data_provider: Callable | None = None
        self._connection = None
        self._conn_lock: threading.Lock | None = None
        self._pos_metrics: StepResponseMetrics | None = None
        self._vel_metrics: VelocityLoopMetrics | None = None

        self._profiles: dict[int, DriveProfile] = {}
        self._param_widgets: dict[str, QWidget] = {}
        self._param_frame: QFrame | None = None
        self._axis_combo: QComboBox | None = None
        self._drive_combo: QComboBox | None = None
        self._read_btn: QPushButton | None = None
        self._write_btn: QPushButton | None = None
        self._autowrite_chk: QCheckBox | None = None
        self._autowrite_busy: bool = False
        self._autowrite_timer = QTimer(self)
        self._autowrite_timer.setSingleShot(True)
        self._autowrite_timer.setInterval(400)
        self._autowrite_timer.timeout.connect(self._trigger_autowrite)

        self._coe_signals = _CoESignals()
        self._coe_signals.coe_read_done.connect(self._on_coe_read_done)
        self._coe_signals.coe_write_done.connect(self._on_coe_write_done)

        self._build_ui()

    # ================================================================
    # Public API
    # ================================================================

    def set_data_provider(self, provider: Callable):
        self._data_provider = provider

    def set_connection(self, connection, conn_lock=None):
        self._connection = connection
        self._conn_lock = conn_lock
        self._update_drive_buttons()

    def get_all_profiles(self) -> dict[int, dict]:
        return {axis: p.to_dict() for axis, p in self._profiles.items()}

    def set_all_profiles(self, profiles: dict[int, dict]):
        self._profiles = {
            int(axis): DriveProfile.from_dict(d)
            for axis, d in profiles.items()
        }
        self._on_axis_changed()

    # ================================================================
    # UI construction
    # ================================================================

    def _build_ui(self):
        container = QWidget()
        container.setStyleSheet(
            f"QWidget {{ background-color: {_BG_DARK}; color: {_TEXT}; }}"
        )
        root = QVBoxLayout(container)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header ──────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(6)

        title = QLabel("SERVO LOOP ANALYSER")
        title.setStyleSheet(
            f"color: {_ACCENT}; font-family: Consolas; font-size: 11pt;"
            f" font-weight: bold; letter-spacing: 3px;"
        )
        header.addWidget(title)
        header.addStretch()

        self._btn_analyze = QPushButton("ANALYZE")
        self._btn_analyze.setFixedHeight(30)
        self._btn_analyze.setFixedWidth(110)
        self._btn_analyze.setCursor(Qt.PointingHandCursor)
        self._btn_analyze.setStyleSheet(f"""
            QPushButton {{
                background-color: {_ACCENT};
                color: #000;
                font-family: Consolas;
                font-size: 9pt;
                font-weight: bold;
                letter-spacing: 2px;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{ background-color: #ffb52e; }}
            QPushButton:pressed {{ background-color: #e09000; }}
            QPushButton:disabled {{ background-color: #4a4a4a; color: #777; }}
        """)
        self._btn_analyze.clicked.connect(self._on_analyze)
        header.addWidget(self._btn_analyze)

        root.addLayout(header)

        # ── Thin accent line ────────────────────────────────────────
        accent_line = QFrame()
        accent_line.setFixedHeight(2)
        accent_line.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f" stop:0 {_ACCENT}, stop:0.5 {_ACCENT}44, stop:1 transparent);"
        )
        root.addWidget(accent_line)

        # ── Status ──────────────────────────────────────────────────
        self._status_label = QLabel("Capture scope data, then click ANALYZE")
        self._status_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 8pt; padding: 2px 0;"
        )
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # ── Two-column scrollable content area ──────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {_BG_DARK}; border: none; }}"
            f"QScrollBar:vertical {{ background: {_BG_DARK}; width: 8px; }}"
            f"QScrollBar::handle:vertical {{ background: #555; border-radius: 4px;"
            f" min-height: 20px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{"
            f" height: 0; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet(f"background-color: {_BG_DARK};")
        self._scroll_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        columns = QHBoxLayout(self._scroll_content)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(8)

        # ── Left column: Drive Profile ──────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.setContentsMargins(0, 0, 0, 0)

        self._drive_card = self._build_drive_profile_section()
        left_col.addWidget(self._drive_card)

        self._zn_card = self._build_zn_card()
        left_col.addWidget(self._zn_card)

        left_col.addStretch()
        columns.addLayout(left_col, 1)

        # ── Right column: Analysis cards ────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.setContentsMargins(0, 0, 0, 0)

        self._vel_card = self._build_vel_card()
        right_col.addWidget(self._vel_card)

        self._pos_card = self._build_pos_card()
        right_col.addWidget(self._pos_card)

        right_col.addStretch()
        columns.addLayout(right_col, 1)

        scroll.setWidget(self._scroll_content)
        root.addWidget(scroll, 1)

        self.setWidget(container)
        self._reset_display()

    # ================================================================
    # Drive profile section
    # ================================================================

    def _build_drive_profile_section(self) -> QGroupBox:
        group = QGroupBox("Drive Profile")
        group.setMaximumWidth(300)
        group.setStyleSheet(
            f"QGroupBox {{ color: {_TEXT_DIM}; font-size: 8pt;"
            f" border: 1px solid {_BORDER}; border-radius: 4px;"
            f" margin-top: 8px; padding-top: 6px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px;"
            f" padding: 0 4px; color: {_TEXT}; }}"
        )
        outer = QVBoxLayout(group)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        selector_row = QHBoxLayout()
        selector_row.setSpacing(4)

        selector_row.addWidget(QLabel("Axis:"))
        self._axis_combo = QComboBox()
        self._axis_combo.setFixedWidth(50)
        for i in range(16):
            self._axis_combo.addItem(str(i))
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        selector_row.addWidget(self._axis_combo)

        selector_row.addWidget(QLabel("Drive:"))
        self._drive_combo = QComboBox()
        self._drive_combo.addItems(DRIVE_TYPES)
        self._drive_combo.setFixedWidth(80)
        self._drive_combo.currentTextChanged.connect(self._on_drive_type_changed)
        selector_row.addWidget(self._drive_combo)

        selector_row.addStretch()

        outer.addLayout(selector_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(4)

        self._read_btn = QPushButton("Read")
        self._read_btn.setFixedHeight(22)
        self._read_btn.setFixedWidth(55)
        self._read_btn.setEnabled(False)
        self._read_btn.setToolTip(
            "Read Pn parameters from the drive via EtherCAT CoE SDO."
        )
        self._read_btn.clicked.connect(self._on_read_from_drive)
        button_row.addWidget(self._read_btn)

        self._write_btn = QPushButton("Write")
        self._write_btn.setFixedHeight(22)
        self._write_btn.setFixedWidth(55)
        self._write_btn.setEnabled(False)
        self._write_btn.setToolTip(
            "Write ALL Pn parameters to the drive via EtherCAT CoE SDO."
        )
        self._write_btn.clicked.connect(self._on_write_to_drive)
        button_row.addWidget(self._write_btn)

        self._autowrite_chk = QCheckBox("Auto")
        self._autowrite_chk.setStyleSheet(
            f"QCheckBox {{ color: {_TEXT_DIM}; font-size: 8pt; }}"
        )
        self._autowrite_chk.setToolTip(
            "⚠ WARNING — Auto-write to drive\n\n"
            "When checked, any change to a Pn parameter is sent to the drive\n"
            "automatically (after a short debounce) WITHOUT a confirmation\n"
            "prompt. This speeds up interactive tuning but lets you push bad\n"
            "values to a live servo instantly.\n\n"
            "Only enable on a safe setup \n"
            "and keep E-stop within reach. Uncheck to require the Write button\n"
            "and confirmation dialog."
        )
        self._autowrite_chk.setEnabled(False)
        button_row.addWidget(self._autowrite_chk)
        button_row.addStretch()

        outer.addLayout(button_row)

        self._param_frame = QFrame()
        self._param_frame.setVisible(False)
        self._param_frame.setMaximumWidth(280)
        self._param_frame.setStyleSheet(
            f"QFrame {{ border: none; background: transparent; }}"
        )
        param_layout = QFormLayout(self._param_frame)
        param_layout.setContentsMargins(0, 2, 0, 0)
        param_layout.setSpacing(2)
        param_layout.setLabelAlignment(Qt.AlignLeft)
        param_layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        label_style = f"color: {_TEXT}; font-size: 8pt; border: none;"
        combo_style = (
            f"QComboBox {{ background: {_BG_PANEL}; color: {_TEXT};"
            f" border: 1px solid {_BORDER}; border-radius: 2px;"
            f" padding: 1px 3px; font-size: 8pt; }}"
        )
        arrow_style = (
            f"QPushButton {{ background-color: {_BG_PANEL}; color: {_TEXT_DIM};"
            f" border: 1px solid {_BORDER}; border-radius: 2px;"
            f" font-size: 7pt; padding: 0px; }}"
            f"QPushButton:pressed {{ background-color: {_BORDER_LIGHT}; }}"
        )

        for entry in PARAM_DEFS:
            attr, pn_code, label, unit, min_v, max_v, default, tooltip = entry

            row_label = QLabel(f"{pn_code} {label}:")
            row_label.setStyleSheet(label_style)
            row_label.setToolTip(tooltip)

            if attr in COMBO_ATTRS:
                combo_options = {
                    "pn100_tuning_mode": TUNING_MODE_LABELS,
                    "pn100_vibration": VIBRATION_SUPPRESSION_LABELS,
                    "pn100_damping": DAMPING_LABELS,
                }
                w = QComboBox()
                w.setFixedWidth(150)
                w.setStyleSheet(combo_style)
                w.addItems(combo_options.get(attr, []))
                w.setToolTip(tooltip)
                w.currentIndexChanged.connect(self._on_param_changed)
                self._param_widgets[attr] = w
                param_layout.addRow(row_label, w)
            else:
                spin = QSpinBox()
                spin.setRange(min_v, max_v)
                spin.setValue(default)
                spin.setToolTip(tooltip)
                spin.setFixedWidth(60)
                spin.setStyleSheet(
                    f"QSpinBox {{ background: {_BG_PANEL}; color: {_TEXT};"
                    f" border: 1px solid {_BORDER}; border-radius: 2px;"
                    f" padding: 1px 3px; font-size: 8pt; }}"
                    f"QSpinBox::up-button {{ width: 0; border: none; }}"
                    f"QSpinBox::down-button {{ width: 0; border: none; }}"
                )
                spin.valueChanged.connect(self._on_param_changed)

                btn_up = QPushButton("\u25b2")
                btn_up.setFixedSize(18, 12)
                btn_up.setStyleSheet(arrow_style)
                btn_up.clicked.connect(
                    lambda _, s=spin, mx=max_v: s.setValue(min(mx, s.value() + 1))
                )

                btn_down = QPushButton("\u25bc")
                btn_down.setFixedSize(18, 12)
                btn_down.setStyleSheet(arrow_style)
                btn_down.clicked.connect(
                    lambda _, s=spin, mn=min_v: s.setValue(max(mn, s.value() - 1))
                )

                arrows = QVBoxLayout()
                arrows.setSpacing(1)
                arrows.setContentsMargins(0, 0, 0, 0)
                arrows.addWidget(btn_up)
                arrows.addWidget(btn_down)

                unit_lbl = QLabel(unit)
                unit_lbl.setFixedWidth(50)
                unit_lbl.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 8pt;")

                field_row = QHBoxLayout()
                field_row.setSpacing(2)
                field_row.setContentsMargins(0, 0, 0, 0)
                field_row.addWidget(spin)
                field_row.addLayout(arrows)
                field_row.addWidget(unit_lbl)
                field_row.addStretch()

                field_container = QWidget()
                field_container.setStyleSheet("border: none; background: transparent;")
                field_container.setLayout(field_row)

                self._param_widgets[attr] = spin
                param_layout.addRow(row_label, field_container)

        outer.addWidget(self._param_frame)
        return group

    # ── Drive profile UI callbacks ──────────────────────────────────

    def _current_axis(self) -> int:
        return int(self._axis_combo.currentText())

    def _on_axis_changed(self):
        axis = self._current_axis()
        profile = self._profiles.get(axis, DriveProfile())
        self._load_profile_to_ui(profile)

    def _on_drive_type_changed(self, drive_type: str):
        is_trio_drive = drive_type in ("DX3", "DX4")
        self._param_frame.setVisible(is_trio_drive)
        self._update_drive_buttons()
        axis = self._current_axis()
        existing = self._profiles.get(axis)
        if is_trio_drive and (existing is None or not existing.has_drive_params()):
            self._set_ui_to_defaults()
        self._save_ui_to_profile()

    def _on_param_changed(self):
        self._save_ui_to_profile()
        if (
            self._autowrite_chk is not None
            and self._autowrite_chk.isChecked()
            and self._autowrite_chk.isEnabled()
        ):
            self._autowrite_timer.start()

    def _trigger_autowrite(self):
        if self._autowrite_busy:
            self._autowrite_timer.start()
            return
        if self._connection is None:
            return
        if self._autowrite_chk is None or not self._autowrite_chk.isChecked():
            return
        axis = self._current_axis()
        profile = self._profiles.get(axis)
        if profile is None or not profile.has_drive_params():
            return

        connection = self._connection
        conn_lock = self._conn_lock
        self._autowrite_busy = True
        self._write_btn.setEnabled(False)
        self._read_btn.setEnabled(False)
        self._write_btn.setText("Auto\u2026")

        def _do_write():
            try:
                results = write_drive_profile(
                    connection, axis=axis, profile=profile, conn_lock=conn_lock,
                )
                self._coe_signals.coe_write_done.emit(axis, results, "")
            except Exception as exc:
                logger.error("Axis %d: auto-write drive profile failed — %s", axis, exc)
                self._coe_signals.coe_write_done.emit(axis, {}, str(exc))

        threading.Thread(target=_do_write, name="TunerCoEAutoWrite", daemon=True).start()

    def _update_drive_buttons(self):
        if self._read_btn is None or self._write_btn is None or self._drive_combo is None:
            return
        drive_type = self._drive_combo.currentText()
        enabled = self._connection is not None and drive_type in ("DX3", "DX4")
        self._read_btn.setEnabled(enabled)
        self._write_btn.setEnabled(enabled)
        if self._autowrite_chk is not None:
            self._autowrite_chk.setEnabled(enabled)
            if not enabled:
                self._autowrite_chk.setChecked(False)

    def _load_profile_to_ui(self, profile: DriveProfile):
        self._drive_combo.blockSignals(True)
        drive_idx = DRIVE_TYPES.index(profile.drive_type) if profile.drive_type in DRIVE_TYPES else 0
        self._drive_combo.setCurrentIndex(drive_idx)
        self._drive_combo.blockSignals(False)

        is_trio = profile.has_drive_params()
        self._param_frame.setVisible(is_trio)
        self._update_drive_buttons()

        if is_trio:
            for entry in PARAM_DEFS:
                attr = entry[0]
                default = entry[6]
                val = getattr(profile, attr, None)
                w = self._param_widgets.get(attr)
                if w is None:
                    continue
                w.blockSignals(True)
                if attr in COMBO_ATTRS:
                    combo_values = {
                        "pn100_tuning_mode": TUNING_MODE_VALUES,
                        "pn100_vibration": VIBRATION_SUPPRESSION_VALUES,
                        "pn100_damping": DAMPING_VALUES,
                    }
                    values = combo_values.get(attr, [])
                    idx = values.index(val) if val in values else 0
                    w.setCurrentIndex(idx)
                else:
                    w.setValue(val if val is not None else default)
                w.blockSignals(False)

    def _set_ui_to_defaults(self):
        for entry in PARAM_DEFS:
            attr, _, _, _, _, _, default, _ = entry
            w = self._param_widgets.get(attr)
            if w is None:
                continue
            w.blockSignals(True)
            if attr in COMBO_ATTRS:
                w.setCurrentIndex(0)
            else:
                w.setValue(default)
            w.blockSignals(False)

    def _save_ui_to_profile(self):
        axis = self._current_axis()
        drive_type = self._drive_combo.currentText()
        profile = DriveProfile(drive_type=drive_type)
        if profile.has_drive_params():
            for entry in PARAM_DEFS:
                attr = entry[0]
                w = self._param_widgets.get(attr)
                if w is None:
                    continue
                if attr in COMBO_ATTRS:
                    combo_values = {
                        "pn100_tuning_mode": TUNING_MODE_VALUES,
                        "pn100_vibration": VIBRATION_SUPPRESSION_VALUES,
                        "pn100_damping": DAMPING_VALUES,
                    }
                    values = combo_values.get(attr, [])
                    setattr(profile, attr, values[w.currentIndex()] if values else 0)
                else:
                    setattr(profile, attr, w.value())
        self._profiles[axis] = profile

    # ── CoE Read / Write ────────────────────────────────────────────

    def _on_read_from_drive(self):
        if self._connection is None:
            return
        axis = self._current_axis()
        drive_type = self._drive_combo.currentText()
        connection = self._connection
        self._read_btn.setEnabled(False)
        self._read_btn.setText("Reading\u2026")
        conn_lock = self._conn_lock

        def _do_read():
            try:
                profile = read_drive_profile(
                    connection, axis=axis, drive_type=drive_type, conn_lock=conn_lock,
                )
                self._coe_signals.coe_read_done.emit(axis, profile, "")
            except Exception as exc:
                logger.error("Axis %d: read drive profile failed — %s", axis, exc)
                self._coe_signals.coe_read_done.emit(
                    axis, DriveProfile(drive_type=drive_type), str(exc),
                )

        threading.Thread(target=_do_read, name="TunerCoERead", daemon=True).start()

    def _on_coe_read_done(self, axis: int, profile: DriveProfile, error: str):
        self._read_btn.setText("Read")
        self._update_drive_buttons()
        if error:
            QMessageBox.warning(
                self, "CoE Read Error",
                f"Failed to read drive parameters from axis {axis}:\n{error}",
            )
            return
        self._profiles[axis] = profile
        if axis == self._current_axis():
            self._load_profile_to_ui(profile)
        logger.info("Axis %d: read drive profile OK — %s", axis, profile.to_dict())

    def _on_write_to_drive(self):
        if self._connection is None:
            return
        axis = self._current_axis()
        reply = QMessageBox.question(
            self, "Write to Drive",
            f"Write current Pn parameters to axis {axis} drive?\n\n"
            "This will overwrite the drive's tuning parameters.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_ui_to_profile()
        profile = self._profiles.get(axis)
        if profile is None or not profile.has_drive_params():
            return
        connection = self._connection
        self._write_btn.setEnabled(False)
        self._read_btn.setEnabled(False)
        self._write_btn.setText("Writing\u2026")
        conn_lock = self._conn_lock

        def _do_write():
            try:
                results = write_drive_profile(
                    connection, axis=axis, profile=profile, conn_lock=conn_lock,
                )
                self._coe_signals.coe_write_done.emit(axis, results, "")
            except Exception as exc:
                logger.error("Axis %d: write drive profile failed — %s", axis, exc)
                self._coe_signals.coe_write_done.emit(axis, {}, str(exc))

        threading.Thread(target=_do_write, name="TunerCoEWrite", daemon=True).start()

    def _on_coe_write_done(self, axis: int, results: dict, error: str):
        was_autowrite = self._autowrite_busy
        self._autowrite_busy = False
        self._write_btn.setText("Write")
        self._update_drive_buttons()
        if error:
            if was_autowrite and self._autowrite_chk is not None:
                self._autowrite_chk.setChecked(False)
            QMessageBox.warning(
                self, "CoE Write Error",
                f"Failed to write drive parameters to axis {axis}:\n{error}",
            )
        else:
            failures = {k: v for k, v in results.items() if v is not None}
            if failures:
                detail = "\n".join(f"  {k}: {v}" for k, v in failures.items())
                QMessageBox.warning(
                    self, "CoE Write Partial",
                    f"Some parameters failed to write on axis {axis}:\n{detail}",
                )
            else:
                n = len(results)
                logger.info("Axis %d: wrote %d parameters OK", axis, n)

    # ================================================================
    # Ziegler-Nichols PI calculator
    # ================================================================

    # (Ku multiplier, Tu multiplier for Ti). Ti = Tu * ti_mult.
    _ZN_PI_METHODS: tuple[tuple[str, float, float], ...] = (
        ("Classical ZN",     0.45,  1.0 / 1.2),   # Kp=0.45 Ku, Ti=Tu/1.2
        ("Tyreus-Luyben",    1.0 / 3.2, 2.2),     # Kp=Ku/3.2, Ti=2.2 Tu  (conservative)
        ("Ciancone-Marlin",  0.303, 1.74),        # robust / low overshoot
    )

    def _build_zn_card(self) -> QGroupBox:
        group = QGroupBox("Ziegler-Nichols PI Calculator")
        group.setMaximumWidth(300)
        group.setStyleSheet(
            f"QGroupBox {{ color: {_TEXT_DIM}; font-size: 8pt;"
            f" border: 1px solid {_BORDER}; border-radius: 4px;"
            f" margin-top: 8px; padding-top: 6px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px;"
            f" padding: 0 4px; color: {_TEXT}; }}"
        )
        outer = QVBoxLayout(group)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        hint = QLabel(
            "Speed loop: raise gain until sustained oscillation.\n"
            "Enter Ku (gain at onset) and Tu (period)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 7pt;")
        outer.addWidget(hint)

        spin_style = (
            f"QDoubleSpinBox {{ background: {_BG_PANEL}; color: {_TEXT};"
            f" border: 1px solid {_BORDER}; border-radius: 2px;"
            f" padding: 1px 3px; font-size: 8pt; }}"
        )

        inputs = QFormLayout()
        inputs.setContentsMargins(0, 2, 0, 2)
        inputs.setSpacing(3)
        inputs.setLabelAlignment(Qt.AlignLeft)

        lbl_style = f"color: {_TEXT}; font-size: 8pt;"

        self._zn_ku = QDoubleSpinBox()
        self._zn_ku.setRange(0.0, 100000.0)
        self._zn_ku.setDecimals(2)
        self._zn_ku.setValue(500.0)
        self._zn_ku.setSuffix("  rad/s")
        self._zn_ku.setFixedWidth(130)
        self._zn_ku.setStyleSheet(spin_style)
        self._zn_ku.setToolTip("Ultimate gain — Pn102 value where the loop just begins sustained oscillation.")
        self._zn_ku.valueChanged.connect(self._recalc_zn)
        ku_lbl = QLabel("Ku (ultimate gain):")
        ku_lbl.setStyleSheet(lbl_style)
        inputs.addRow(ku_lbl, self._zn_ku)

        self._zn_tu = QDoubleSpinBox()
        self._zn_tu.setRange(0.0, 100000.0)
        self._zn_tu.setDecimals(2)
        self._zn_tu.setValue(10.0)
        self._zn_tu.setSuffix("  ms")
        self._zn_tu.setFixedWidth(130)
        self._zn_tu.setStyleSheet(spin_style)
        self._zn_tu.setToolTip("Ultimate period — period of the sustained oscillation, in milliseconds.")
        self._zn_tu.valueChanged.connect(self._recalc_zn)
        tu_lbl = QLabel("Tu (period):")
        tu_lbl.setStyleSheet(lbl_style)
        inputs.addRow(tu_lbl, self._zn_tu)

        outer.addLayout(inputs)
        outer.addWidget(_separator())

        # Results grid: Method | Kp (→Pn102) | Ti ms (→Pn103)
        self._zn_results_grid = QGridLayout()
        self._zn_results_grid.setHorizontalSpacing(6)
        self._zn_results_grid.setVerticalSpacing(2)
        self._zn_results_grid.setContentsMargins(0, 2, 0, 0)

        hdr_style = (
            f"color: {_TEXT_DIM}; font-family: Consolas; font-size: 7pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        for col, text in enumerate(("METHOD", "Kp (Pn102)", "Ti (Pn103)")):
            h = QLabel(text)
            h.setStyleSheet(hdr_style)
            self._zn_results_grid.addWidget(h, 0, col)

        self._zn_result_labels: list[tuple[QLabel, QLabel]] = []
        for row, (name, _, _) in enumerate(self._ZN_PI_METHODS, start=1):
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 8pt;")
            self._zn_results_grid.addWidget(name_lbl, row, 0)

            kp_lbl = QLabel("--")
            kp_lbl.setStyleSheet(
                f"color: {_CYAN}; font-family: Consolas; font-size: 8pt;"
                f" font-weight: bold;"
            )
            kp_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._zn_results_grid.addWidget(kp_lbl, row, 1)

            ti_lbl = QLabel("--")
            ti_lbl.setStyleSheet(
                f"color: {_ACCENT}; font-family: Consolas; font-size: 8pt;"
                f" font-weight: bold;"
            )
            ti_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._zn_results_grid.addWidget(ti_lbl, row, 2)

            self._zn_result_labels.append((kp_lbl, ti_lbl))

        outer.addLayout(self._zn_results_grid)

        note = QLabel(
            "Kp shown in rad/s → write to Pn102.\n"
            "Ti shown in ×0.1 ms units → write to Pn103."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 7pt; padding-top: 4px;")
        outer.addWidget(note)

        self._recalc_zn()
        return group

    def _recalc_zn(self):
        ku = float(self._zn_ku.value())
        tu_ms = float(self._zn_tu.value())
        tu_s = tu_ms * 1e-3

        for (kp_lbl, ti_lbl), (_, kp_mult, ti_mult) in zip(
            self._zn_result_labels, self._ZN_PI_METHODS,
        ):
            if ku <= 0 or tu_s <= 0:
                kp_lbl.setText("--")
                ti_lbl.setText("--")
                continue
            kp = ku * kp_mult
            ti_s = tu_s * ti_mult
            # Pn103 is in units of 0.1 ms
            pn103 = ti_s / 0.1e-3
            kp_lbl.setText(f"{kp:.1f}")
            ti_lbl.setText(f"{pn103:.0f}")

    # ================================================================
    # Analysis card builders
    # ================================================================

    def _build_vel_card(self) -> QFrame:
        card = _card_frame()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        self._vel_dot = _HealthDot()
        hdr.addWidget(self._vel_dot)
        lbl = QLabel("VELOCITY LOOP")
        lbl.setStyleSheet(
            f"color: {_TEXT_BRIGHT}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        hdr.addWidget(lbl)
        hdr.addStretch()
        self._vel_status_lbl = QLabel("--")
        self._vel_status_lbl.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 8pt; font-style: italic;"
        )
        hdr.addWidget(self._vel_status_lbl)
        lay.addLayout(hdr)
        lay.addWidget(_separator())

        self._vel_metrics_layout = QVBoxLayout()
        self._vel_metrics_layout.setSpacing(2)
        lay.addLayout(self._vel_metrics_layout)

        self._vel_issues_label = QLabel("")
        self._vel_issues_label.setWordWrap(True)
        self._vel_issues_label.setStyleSheet(
            f"color: {_RED}; font-size: 8pt; padding: 2px 0 0 0;"
        )
        self._vel_issues_label.hide()
        lay.addWidget(self._vel_issues_label)

        return card

    def _build_pos_card(self) -> QFrame:
        card = _card_frame()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        self._pos_dot = _HealthDot()
        hdr.addWidget(self._pos_dot)
        lbl = QLabel("POSITION LOOP")
        lbl.setStyleSheet(
            f"color: {_TEXT_BRIGHT}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        hdr.addWidget(lbl)
        hdr.addStretch()
        lay.addLayout(hdr)
        lay.addWidget(_separator())

        self._pos_metrics_layout = QVBoxLayout()
        self._pos_metrics_layout.setSpacing(2)
        lay.addLayout(self._pos_metrics_layout)

        return card

    # ================================================================
    # Layout helpers
    # ================================================================

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
            sub = item.layout()
            if sub:
                TunerPanel._clear_layout(sub)

    def _add_metric_row(self, layout, name, value, unit="", color=_CYAN):
        layout.addLayout(_metric_label(name, value, unit, color))

    # ================================================================
    # Reset / populate display
    # ================================================================

    def _reset_display(self):
        self._vel_dot.set_healthy(None)
        self._vel_status_lbl.setText("--")
        self._vel_issues_label.hide()
        self._pos_dot.set_healthy(None)
        self._clear_layout(self._vel_metrics_layout)
        self._clear_layout(self._pos_metrics_layout)

        self._add_metric_row(self._vel_metrics_layout, "Accel overshoot", "--", "%")
        self._add_metric_row(self._vel_metrics_layout, "Cruise tracking", "--")
        self._add_metric_row(self._vel_metrics_layout, "Settle time", "--", "ms")
        self._add_metric_row(self._vel_metrics_layout, "Oscillations", "--")

        self._add_metric_row(self._pos_metrics_layout, "Overshoot", "--", "%")
        self._add_metric_row(self._pos_metrics_layout, "Settling time", "--", "ms")
        self._add_metric_row(self._pos_metrics_layout, "Rise time", "--", "ms")
        self._add_metric_row(self._pos_metrics_layout, "Oscillations", "--")
        self._add_metric_row(self._pos_metrics_layout, "Steady-state err", "--")
        self._add_metric_row(self._pos_metrics_layout, "Drive FE (peak)", "--", "u")
        self._add_metric_row(self._pos_metrics_layout, "Drive FE (cruise)", "--", "u")
        self._add_metric_row(self._pos_metrics_layout, "Damping ratio", "--")

    def _populate_vel(self, vm: VelocityLoopMetrics | None):
        self._clear_layout(self._vel_metrics_layout)
        if vm is None:
            self._vel_dot.set_healthy(None)
            self._vel_status_lbl.setText("No MSPEED data")
            self._vel_issues_label.hide()
            self._add_metric_row(
                self._vel_metrics_layout, "Status",
                "No velocity data", color=_TEXT_DIM,
            )
            return

        self._vel_dot.set_healthy(vm.is_healthy)
        self._vel_status_lbl.setText(
            "No issues" if vm.is_healthy else "Issues detected"
        )
        self._vel_status_lbl.setStyleSheet(
            f"color: {_health_color(vm.is_healthy)}; font-size: 8pt;"
            f" font-style: italic;"
        )

        ov_color = _CYAN if vm.accel_overshoot_pct <= 15 else _RED
        self._add_metric_row(
            self._vel_metrics_layout, "Accel overshoot",
            f"{vm.accel_overshoot_pct:.1f}", "%", ov_color,
        )

        ratio = vm.cruise_tracking_ratio
        r_color = _CYAN if 0.90 <= ratio <= 1.10 else _AMBER
        self._add_metric_row(
            self._vel_metrics_layout, "Cruise tracking",
            f"{ratio:.3f}", "", r_color,
        )

        st_color = _CYAN if vm.accel_settle_time_ms <= 100 else _AMBER
        self._add_metric_row(
            self._vel_metrics_layout, "Settle time",
            f"{vm.accel_settle_time_ms:.0f}", "ms", st_color,
        )

        osc_color = _CYAN if vm.accel_oscillation_count <= 3 else _RED
        self._add_metric_row(
            self._vel_metrics_layout, "Oscillations",
            str(vm.accel_oscillation_count), "", osc_color,
        )

        self._add_metric_row(
            self._vel_metrics_layout, "Cruise vel. std",
            f"{vm.cruise_velocity_std:.3f}", "", _CYAN,
        )

        if vm.issues:
            self._vel_issues_label.setText(
                "\n".join(f"\u26a0 {iss}" for iss in vm.issues)
            )
            self._vel_issues_label.show()
        else:
            self._vel_issues_label.hide()

    def _populate_pos(self, pm: StepResponseMetrics):
        self._clear_layout(self._pos_metrics_layout)

        is_empty = (
            pm.overshoot_pct == 0 and pm.oscillation_count == 0
            and pm.settling_time_ms == 0
        )
        if is_empty:
            self._pos_dot.set_healthy(None)
            self._add_metric_row(
                self._pos_metrics_layout, "Status",
                "No analysable move", color=_TEXT_DIM,
            )
            return

        # Simple health indicator from raw metrics (no scoring/verdict).
        healthy = (
            pm.overshoot_pct <= 15
            and pm.oscillation_count <= 3
            and pm.settling_time_ms <= 500
            and pm.drive_fe_peak_pct <= 1.5
        )
        self._pos_dot.set_healthy(healthy)

        ov_color = _CYAN if pm.overshoot_pct <= 5 else (
            _AMBER if pm.overshoot_pct <= 15 else _RED
        )
        self._add_metric_row(
            self._pos_metrics_layout, "Overshoot",
            f"{pm.overshoot_pct:.1f}", "%", ov_color,
        )

        st_color = _CYAN if pm.settling_time_ms <= 200 else (
            _AMBER if pm.settling_time_ms <= 500 else _RED
        )
        self._add_metric_row(
            self._pos_metrics_layout, "Settling time",
            f"{pm.settling_time_ms:.0f}", "ms", st_color,
        )

        self._add_metric_row(
            self._pos_metrics_layout, "Rise time",
            f"{pm.rise_time_ms:.0f}", "ms",
        )

        osc_color = _CYAN if pm.oscillation_count <= 1 else (
            _AMBER if pm.oscillation_count <= 3 else _RED
        )
        self._add_metric_row(
            self._pos_metrics_layout, "Oscillations",
            str(pm.oscillation_count), "", osc_color,
        )

        ss_pct = pm.steady_state_error * 100
        ss_color = _CYAN if ss_pct <= 2 else _AMBER
        self._add_metric_row(
            self._pos_metrics_layout, "Steady-state err",
            f"{ss_pct:.2f}", "%", ss_color,
        )

        fe_color = _CYAN if pm.drive_fe_peak_pct <= 0.2 else (
            _AMBER if pm.drive_fe_peak_pct <= 0.5 else _RED
        )
        self._add_metric_row(
            self._pos_metrics_layout, "Drive FE (peak)",
            f"{pm.drive_fe_peak:.4g}", "u", fe_color,
        )

        fe_cruise_color = _CYAN if pm.drive_fe_cruise_mean_pct <= 0.2 else (
            _AMBER if pm.drive_fe_cruise_mean_pct <= 0.5 else _RED
        )
        self._add_metric_row(
            self._pos_metrics_layout, "Drive FE (cruise)",
            f"{pm.drive_fe_cruise_mean:.4g}", "u", fe_cruise_color,
        )

        self._add_metric_row(
            self._pos_metrics_layout, "Damping ratio",
            f"{pm.damping_ratio:.3f}", "",
        )

        if pm.natural_freq_est_hz > 0:
            self._add_metric_row(
                self._pos_metrics_layout, "Natural freq",
                f"{pm.natural_freq_est_hz:.1f}", "Hz",
            )

    # ================================================================
    # Analysis entry point
    # ================================================================

    def _on_analyze(self):
        if not self._data_provider:
            self._status_label.setText("No data provider connected")
            self._status_label.setStyleSheet(f"color: {_RED}; font-size: 8pt;")
            return

        provider_result = self._data_provider()
        if provider_result is None:
            time_arr, params, servo_period_sec = None, None, None
        elif len(provider_result) == 3:
            time_arr, params, servo_period_sec = provider_result
        else:
            time_arr, params = provider_result
            servo_period_sec = None
        if time_arr is None or params is None:
            self._status_label.setText(
                "No captured data available \u2014 run a capture first"
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        if len(time_arr) < 20:
            self._status_label.setText("Capture too short for analysis")
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        from .signal_metrics import _find_channel

        ch_dpos = _find_channel(params, "dpos", "demandposition", "targetposition")
        ch_mpos = _find_channel(params, "mpos", "measuredposition", "actualposition")
        ch_mvel = _find_channel(
            params, "mspeed", "measuredvel", "actualvel", "vactual",
        )
        ch_dvel = _find_channel(params, "demandspeed", "demandvel", "dspeed")
        ch_fe = _find_channel(params, "drivefe", "fe", "followingerror")

        dpos = params.get(ch_dpos) if ch_dpos else None
        mpos = params.get(ch_mpos) if ch_mpos else None
        mvel = params.get(ch_mvel) if ch_mvel else None
        dvel_raw = params.get(ch_dvel) if ch_dvel else None
        drive_fe_raw = params.get(ch_fe) if ch_fe else None

        if dpos is None or mpos is None:
            self._status_label.setText(
                "Need DPOS and MPOS channels for analysis. "
                "Capture demand position and measured position."
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        if dvel_raw is None:
            self._status_label.setText(
                "DEMAND_SPEED not captured \u2014 velocity-loop analysis skipped. "
                "Add DEMAND_SPEED to the scope channel list and re-capture."
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        if not servo_period_sec or servo_period_sec <= 0:
            self._status_label.setText(
                "Servo period unknown \u2014 cannot scale DEMAND_SPEED. "
                "Reconnect to the controller and re-capture."
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        command = np.asarray(dpos, dtype=np.float64)
        response = np.asarray(mpos, dtype=np.float64)
        velocity = np.asarray(mvel, dtype=np.float64) if mvel is not None else None
        demand_velocity = np.asarray(dvel_raw, dtype=np.float64) / float(servo_period_sec)
        drive_fe = (
            np.asarray(drive_fe_raw, dtype=np.float64)
            if drive_fe_raw is not None else None
        )
        time_np = np.asarray(time_arr, dtype=np.float64)

        self._status_label.setText("Analyzing\u2026")
        self._status_label.setStyleSheet(f"color: {_ACCENT}; font-size: 8pt;")

        try:
            pos_m, vel_m = ClassicalTuner.analyze_step_response(
                time_np, response, command, velocity, demand_velocity,
                drive_fe=drive_fe,
            )
        except Exception as exc:
            logger.exception("Step response analysis failed")
            self._status_label.setText(f"Analysis error: {exc}")
            self._status_label.setStyleSheet(f"color: {_RED}; font-size: 8pt;")
            return

        self._pos_metrics = pos_m
        self._vel_metrics = vel_m

        self._populate_vel(vel_m)
        self._populate_pos(pos_m)

        n_samples = len(time_np)
        dur_s = float(time_np[-1] - time_np[0])
        channels_used = []
        if ch_dpos:
            channels_used.append("DPOS")
        if ch_mpos:
            channels_used.append("MPOS")
        if ch_mvel:
            channels_used.append("MSPEED")
        if ch_dvel:
            channels_used.append("DEMAND_SPEED")
        ch_str = " + ".join(channels_used)

        self._status_label.setText(
            f"Analyzed {n_samples} samples ({dur_s:.2f}s) | {ch_str}"
        )
        self._status_label.setStyleSheet(f"color: {_GREEN}; font-size: 8pt;")

        self.analysis_complete.emit()
