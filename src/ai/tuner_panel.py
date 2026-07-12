"""
Servo Loop Analyser Panel — dockable Qt widget for scope-based loop diagnostics.

Combines:
  - Drive profile editor (axis selector, Pn parameter spinboxes, CoE Read/Write)
  - Ziegler-Nichols PI calculator (zn_calculator.py)
  - Velocity / position / FE-diagnostics cards (loop_cards.py) fed by the
    shared SignalMetrics engine — the same analysis the AI panel uses.

ANALYZE resolves channels for the selected axis only, honours continuous-mode
segment breaks, and judges settling against a tolerance band (auto-derived
from the capture's noise floor, or set explicitly in the Settle tolerance box).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFormLayout, QGroupBox,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer

from .signal_metrics import SignalMetrics
from .signal_channels import detect_axes
from .loop_cards import FePhaseCard, PositionLoopCard, VelocityLoopCard
from .zn_calculator import ZNCalculatorCard
from .history_card import HistoryCard
from .tuning_history import TuningHistory, TuningRun, make_run
from .tuner_theme import (
    ACCENT, AMBER, BG_DARK, BG_PANEL, BORDER, BORDER_LIGHT, GREEN, GROUP_STYLE,
    RED, TEXT, TEXT_DIM,
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
        self._last_metrics: dict | None = None
        self._history = TuningHistory()

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
        """provider() → (time, params[, servo_period_sec[, segment_breaks]])."""
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

    def last_metrics(self) -> dict | None:
        """The full SignalMetrics dict from the most recent ANALYZE."""
        return self._last_metrics

    # ================================================================
    # UI construction
    # ================================================================

    def _build_ui(self):
        container = QWidget()
        container.setStyleSheet(
            f"QWidget {{ background-color: {BG_DARK}; color: {TEXT}; }}"
        )
        root = QVBoxLayout(container)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header ──────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(6)

        title = QLabel("SERVO LOOP ANALYSER")
        title.setStyleSheet(
            f"color: {ACCENT}; font-family: Consolas; font-size: 11pt;"
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
                background-color: {ACCENT};
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
            f" stop:0 {ACCENT}, stop:0.5 {ACCENT}44, stop:1 transparent);"
        )
        root.addWidget(accent_line)

        # ── Status + settle tolerance ───────────────────────────────
        status_row = QHBoxLayout()
        status_row.setSpacing(6)

        self._status_label = QLabel("Capture scope data, then click ANALYZE")
        self._status_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 8pt; padding: 2px 0;"
        )
        self._status_label.setWordWrap(True)
        status_row.addWidget(self._status_label, 1)

        tol_lbl = QLabel("Settle tol ±")
        tol_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8pt;")
        status_row.addWidget(tol_lbl)

        self._band_spin = QDoubleSpinBox()
        self._band_spin.setRange(0.0, 1e9)
        self._band_spin.setDecimals(4)
        self._band_spin.setValue(0.0)
        self._band_spin.setSpecialValueText("Auto")
        self._band_spin.setFixedWidth(90)
        self._band_spin.setStyleSheet(
            f"QDoubleSpinBox {{ background: {BG_PANEL}; color: {TEXT};"
            f" border: 1px solid {BORDER}; border-radius: 2px;"
            f" padding: 1px 3px; font-size: 8pt; }}"
        )
        self._band_spin.setToolTip(
            "Settling tolerance band in user units (e.g. the machine's "
            "in-position window).\nSettle time = time after move end until "
            "|FE| stays within ±band.\nAuto (0) derives the band from the "
            "capture's measured noise floor."
        )
        status_row.addWidget(self._band_spin)

        root.addLayout(status_row)

        # ── Two-column scrollable content area ──────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {BG_DARK}; border: none; }}"
            f"QScrollBar:vertical {{ background: {BG_DARK}; width: 8px; }}"
            f"QScrollBar::handle:vertical {{ background: #555; border-radius: 4px;"
            f" min-height: 20px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{"
            f" height: 0; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet(f"background-color: {BG_DARK};")
        self._scroll_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        content = QVBoxLayout(self._scroll_content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(8)
        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(8)
        content.addLayout(columns)

        # ── Left column: Drive Profile + ZN ─────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.setContentsMargins(0, 0, 0, 0)

        self._drive_card = self._build_drive_profile_section()
        left_col.addWidget(self._drive_card)

        self._zn_card = ZNCalculatorCard()
        left_col.addWidget(self._zn_card)

        left_col.addStretch()
        columns.addLayout(left_col, 1)

        # ── Right column: Analysis cards ────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.setContentsMargins(0, 0, 0, 0)

        self._vel_card = VelocityLoopCard()
        right_col.addWidget(self._vel_card)

        self._pos_card = PositionLoopCard()
        right_col.addWidget(self._pos_card)

        self._fe_card = FePhaseCard()
        right_col.addWidget(self._fe_card)

        right_col.addStretch()
        columns.addLayout(right_col, 1)

        # ── Full-width tuning history under both columns ────────────
        self._history_card = HistoryCard()
        self._history_card.run_selected.connect(self._on_history_run_selected)
        content.addWidget(self._history_card)

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
        group.setStyleSheet(GROUP_STYLE)
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
        self._axis_combo.setToolTip(
            "Axis for the drive profile AND for scope analysis — ANALYZE "
            "only uses channels captured for this axis."
        )
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
            f"QCheckBox {{ color: {TEXT_DIM}; font-size: 8pt; }}"
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

        label_style = f"color: {TEXT}; font-size: 8pt; border: none;"
        combo_style = (
            f"QComboBox {{ background: {BG_PANEL}; color: {TEXT};"
            f" border: 1px solid {BORDER}; border-radius: 2px;"
            f" padding: 1px 3px; font-size: 8pt; }}"
        )
        arrow_style = (
            f"QPushButton {{ background-color: {BG_PANEL}; color: {TEXT_DIM};"
            f" border: 1px solid {BORDER}; border-radius: 2px;"
            f" font-size: 7pt; padding: 0px; }}"
            f"QPushButton:pressed {{ background-color: {BORDER_LIGHT}; }}"
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
                    f"QSpinBox {{ background: {BG_PANEL}; color: {TEXT};"
                    f" border: 1px solid {BORDER}; border-radius: 2px;"
                    f" padding: 1px 3px; font-size: 8pt; }}"
                    f"QSpinBox::up-button {{ width: 0; border: none; }}"
                    f"QSpinBox::down-button {{ width: 0; border: none; }}"
                )
                spin.valueChanged.connect(self._on_param_changed)

                btn_up = QPushButton("▲")
                btn_up.setFixedSize(18, 12)
                btn_up.setStyleSheet(arrow_style)
                btn_up.clicked.connect(
                    lambda _, s=spin, mx=max_v: s.setValue(min(mx, s.value() + 1))
                )

                btn_down = QPushButton("▼")
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
                unit_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8pt;")

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
        self._write_btn.setText("Auto…")

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
        self._read_btn.setText("Reading…")
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
        self._write_btn.setText("Writing…")
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
    # Analysis entry point
    # ================================================================

    def _reset_display(self):
        self._vel_card.reset()
        self._pos_card.reset()
        self._fe_card.reset()

    def _set_status(self, text: str, color: str):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 8pt;")

    def _fetch_capture(self):
        """Unpack the provider result → (time, params, servo_period, breaks)."""
        result = self._data_provider()
        time_arr = params = servo_period = None
        breaks = None
        if result is not None:
            seq = tuple(result)
            if len(seq) >= 2:
                time_arr, params = seq[0], seq[1]
            if len(seq) >= 3:
                servo_period = seq[2]
            if len(seq) >= 4:
                breaks = seq[3]
        return time_arr, params, servo_period, breaks

    def _select_analysis_axis(self, params: dict) -> int | None:
        """Analysis axis: the profile axis, or the only captured axis.

        Returns None (with a status message) when the capture holds several
        axes and none of them is the selected one.
        """
        selected = self._current_axis()
        captured = detect_axes(params)
        if not captured or selected in captured:
            return selected
        if len(captured) == 1:
            return captured[0]
        self._set_status(
            f"Capture contains axes {captured} — pick one in the Axis box "
            f"(currently {selected})", AMBER)
        return None

    def _on_analyze(self):
        if not self._data_provider:
            self._set_status("No data provider connected", RED)
            return

        time_arr, params, servo_period, breaks = self._fetch_capture()
        if time_arr is None or params is None or len(time_arr) == 0:
            self._set_status(
                "No captured data available — run a capture first", AMBER)
            return

        axis = self._select_analysis_axis(params)
        if axis is None:
            return

        band = float(self._band_spin.value()) or None

        self._set_status("Analyzing…", ACCENT)
        try:
            metrics = SignalMetrics.compute_all(
                time_arr, params,
                axis=axis,
                servo_period_sec=servo_period,
                segment_breaks=breaks,
                settle_band=band,
            )
        except Exception as exc:
            logger.exception("Scope analysis failed")
            self._set_status(f"Analysis error: {exc}", RED)
            return

        self._last_metrics = metrics
        self._vel_card.populate(metrics)
        self._pos_card.populate(metrics)
        self._fe_card.populate(metrics)

        warnings = metrics.get("warnings", [])
        if metrics.get("data_sufficiency") != "OK":
            reason = warnings[-1] if warnings else "insufficient data"
            self._set_status(f"Insufficient data: {reason}", AMBER)
            self.analysis_complete.emit()
            return

        # Record this run in the tuning history with the axis Pn snapshot
        profile = self._profiles.get(axis)
        snapshot = (profile.to_dict()
                    if profile is not None and profile.has_drive_params()
                    else None)
        self._history.add(make_run(metrics, axis, snapshot))
        self._history_card.refresh(self._history)

        cap = metrics.get("capture", {})
        phases = metrics.get("phases", {})
        parts = [
            f"Run {len(self._history)}",
            f"Axis {axis}",
            f"{cap.get('n_samples', 0)} samples ({cap.get('duration_s', 0):.2f}s)",
            f"{phases.get('n_moves', 0)} move(s)",
        ]
        if cap.get("n_segments", 1) > 1:
            parts.append(f"{cap['n_segments']} segments")
        if "settle_band" in cap:
            parts.append(
                f"band ±{cap['settle_band']:g} ({cap.get('settle_band_source')})")
        self._set_status(" | ".join(parts), GREEN)

        channels = metrics.get("channels_detected", {})
        tooltip_lines = ["Channels used:"]
        tooltip_lines += [f"  {k}: {v or '—'}" for k, v in channels.items()]
        if warnings:
            tooltip_lines.append("Warnings:")
            tooltip_lines += [f"  • {w}" for w in warnings]
        self._status_label.setToolTip("\n".join(tooltip_lines))

        self.analysis_complete.emit()

    def _on_history_run_selected(self, run: TuningRun):
        """Recall a previous run's full analysis into the metric cards."""
        if not run.full_metrics:
            return
        self._last_metrics = run.full_metrics
        self._vel_card.populate(run.full_metrics)
        self._pos_card.populate(run.full_metrics)
        self._fe_card.populate(run.full_metrics)
        self._set_status(
            f"Viewing run {run.timestamp} (axis {run.axis}) — "
            f"press ANALYZE for a new run", ACCENT)
