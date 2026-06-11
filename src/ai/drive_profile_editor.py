"""
Shared drive profile editor widget.

A QGroupBox with the per-axis Trio drive profile UI used by both the
Servo Loop Analyser (TunerPanel) and the AI Analysis panel:

  - axis + drive type selectors
  - Pn parameter spinboxes / combos (from drive_profile.PARAM_DEFS)
  - CoE Read / Write buttons (EtherCAT SDO, run on worker threads)
  - optional debounced auto-write of parameter edits

Profiles are stored per axis as DriveProfile objects and exposed via
get_all_profiles()/set_all_profiles() for QSettings persistence.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtWidgets import (
    QGroupBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QMessageBox, QComboBox, QSpinBox,
    QCheckBox, QFormLayout,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer

from .drive_profile import (
    DriveProfile, DRIVE_TYPES, PARAM_DEFS, COMBO_ATTRS,
    TUNING_MODE_LABELS, TUNING_MODE_VALUES,
    VIBRATION_SUPPRESSION_LABELS, VIBRATION_SUPPRESSION_VALUES,
    DAMPING_LABELS, DAMPING_VALUES,
)
from .coe_io import read_drive_profile, write_drive_profile
from .tuner_theme import _BG_PANEL, _BORDER, _BORDER_LIGHT, _TEXT, _TEXT_DIM

logger = logging.getLogger(__name__)

_COMBO_LABELS = {
    "pn100_tuning_mode": TUNING_MODE_LABELS,
    "pn100_vibration": VIBRATION_SUPPRESSION_LABELS,
    "pn100_damping": DAMPING_LABELS,
}
_COMBO_VALUES = {
    "pn100_tuning_mode": TUNING_MODE_VALUES,
    "pn100_vibration": VIBRATION_SUPPRESSION_VALUES,
    "pn100_damping": DAMPING_VALUES,
}


class _CoESignals(QObject):
    """Thread-safe relay for CoE worker thread results."""
    coe_read_done = Signal(int, object, str)   # axis, DriveProfile, error_msg
    coe_write_done = Signal(int, object, str)  # axis, results_dict, error_msg


class DriveProfileEditor(QGroupBox):
    """Per-axis drive profile editor with CoE Read/Write support."""

    profiles_changed = Signal()

    def __init__(self, parent=None, *, autowrite: bool = False,
                 max_width: int | None = 300):
        super().__init__("Drive Profile", parent)
        if max_width is not None:
            self.setMaximumWidth(max_width)

        self._connection = None
        self._conn_lock: threading.Lock | None = None
        self._profiles: dict[int, DriveProfile] = {}
        self._param_widgets: dict[str, QWidget] = {}
        self._param_frame: QFrame | None = None
        self._axis_combo: QComboBox | None = None
        self._drive_combo: QComboBox | None = None
        self._read_btn: QPushButton | None = None
        self._write_btn: QPushButton | None = None

        self._autowrite_enabled = autowrite
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

    def set_connection(self, connection, conn_lock=None):
        """Provide the active TUA.TrioConnection (None disables Read/Write)."""
        self._connection = connection
        self._conn_lock = conn_lock
        self._update_drive_buttons()

    def get_all_profiles(self) -> dict[int, dict]:
        """Return all per-axis profiles as plain dicts (for QSettings persistence)."""
        return {axis: p.to_dict() for axis, p in self._profiles.items()}

    def set_all_profiles(self, profiles: dict[int, dict]):
        """Restore per-axis profiles from plain dicts (loaded from QSettings)."""
        self._profiles = {
            int(axis): DriveProfile.from_dict(d)
            for axis, d in profiles.items()
        }
        self._on_axis_changed()

    def current_axis(self) -> int:
        return int(self._axis_combo.currentText())

    def current_profile(self) -> DriveProfile | None:
        """Return the profile for the currently selected axis, if any."""
        return self._profiles.get(self.current_axis())

    # ================================================================
    # UI construction
    # ================================================================

    def _build_ui(self):
        self.setStyleSheet(
            f"QGroupBox {{ color: {_TEXT_DIM}; font-size: 8pt;"
            f" border: 1px solid {_BORDER}; border-radius: 4px;"
            f" margin-top: 8px; padding-top: 6px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px;"
            f" padding: 0 4px; color: {_TEXT}; }}"
        )
        outer = QVBoxLayout(self)
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

        if self._autowrite_enabled:
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
                w = QComboBox()
                w.setFixedWidth(150)
                w.setStyleSheet(combo_style)
                w.addItems(_COMBO_LABELS.get(attr, []))
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

    # ================================================================
    # UI callbacks
    # ================================================================

    def _on_axis_changed(self):
        axis = self.current_axis()
        profile = self._profiles.get(axis, DriveProfile())
        self._load_profile_to_ui(profile)

    def _on_drive_type_changed(self, drive_type: str):
        is_trio_drive = drive_type in ("DX3", "DX4")
        self._param_frame.setVisible(is_trio_drive)
        self._update_drive_buttons()
        axis = self.current_axis()
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
                    values = _COMBO_VALUES.get(attr, [])
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
        axis = self.current_axis()
        drive_type = self._drive_combo.currentText()
        profile = DriveProfile(drive_type=drive_type)
        if profile.has_drive_params():
            for entry in PARAM_DEFS:
                attr = entry[0]
                w = self._param_widgets.get(attr)
                if w is None:
                    continue
                if attr in COMBO_ATTRS:
                    values = _COMBO_VALUES.get(attr, [])
                    setattr(profile, attr, values[w.currentIndex()] if values else 0)
                else:
                    setattr(profile, attr, w.value())
        self._profiles[axis] = profile
        self.profiles_changed.emit()

    # ================================================================
    # CoE Read / Write
    # ================================================================

    def _on_read_from_drive(self):
        if self._connection is None:
            return
        axis = self.current_axis()
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

        threading.Thread(target=_do_read, name="CoERead", daemon=True).start()

    def _on_coe_read_done(self, axis: int, profile: DriveProfile, error: str):
        self._read_btn.setText("Read")
        self._update_drive_buttons()
        if error:
            QMessageBox.warning(
                self, "CoE Read Error",
                f"Failed to read drive parameters from axis {axis}:\n{error}",
            )
            return
        # Always cache the profile for the axis that was read…
        self._profiles[axis] = profile
        # …but only touch the UI if the user is still viewing that axis —
        # otherwise we would clobber the axis they switched to mid-read.
        if axis == self.current_axis():
            self._load_profile_to_ui(profile)
        self.profiles_changed.emit()
        logger.info("Axis %d: read drive profile OK — %s", axis, profile.to_dict())

    def _on_write_to_drive(self):
        if self._connection is None:
            return
        axis = self.current_axis()
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

        threading.Thread(target=_do_write, name="CoEWrite", daemon=True).start()

    def _trigger_autowrite(self):
        if self._autowrite_busy:
            self._autowrite_timer.start()
            return
        if self._connection is None:
            return
        if self._autowrite_chk is None or not self._autowrite_chk.isChecked():
            return
        axis = self.current_axis()
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

        threading.Thread(target=_do_write, name="CoEAutoWrite", daemon=True).start()

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
