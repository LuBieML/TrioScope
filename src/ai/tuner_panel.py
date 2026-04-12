"""
Classical Tuner Panel — dockable Qt widget for servo loop analysis.

Combines:
  - Drive profile editor (axis selector, Pn parameter spinboxes, CoE Read/Write)
  - Tuning score gauge (custom-painted arc, 0–10)
  - Velocity loop health card
  - Position loop metrics card
  - Proposed corrections with current → new values
  - Apply to Profile / Write to Drive action buttons
  - Cascade-aware warnings
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional, Callable

import numpy as np

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy, QGridLayout,
    QGraphicsDropShadowEffect, QMessageBox, QComboBox, QSpinBox,
    QFormLayout, QGroupBox,
)
from PySide6.QtCore import Qt, Signal, QObject, QRectF, QPointF, QTimer
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QConicalGradient,
    QLinearGradient, QRadialGradient, QPainterPath, QFontMetrics,
)

from .classical_tuner import (
    ClassicalTuner, StepResponseMetrics, VelocityLoopMetrics, TuningResult,
)
from .drive_profile import (
    DriveProfile, DRIVE_TYPES, PARAM_DEFS, COMBO_ATTRS,
    TUNING_MODE_LABELS, TUNING_MODE_VALUES,
    VIBRATION_SUPPRESSION_LABELS, VIBRATION_SUPPRESSION_VALUES,
    DAMPING_LABELS, DAMPING_VALUES,
)
from .signal_metrics import SignalMetrics
from .coe_io import read_drive_profile, write_drive_profile, write_single_pn

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
_ACCENT        = "#FFA500"      # orange — matches app accent
_CYAN          = "#00d4aa"      # oscilloscope readout
_GREEN         = "#2ecc71"
_AMBER         = "#f39c12"
_RED           = "#e74c3c"
_BLUE_MUTED    = "#5b7fb5"

# Gauge arc colours
_GAUGE_RED     = QColor(231, 76, 60)
_GAUGE_AMBER   = QColor(243, 156, 18)
_GAUGE_GREEN   = QColor(46, 204, 113)
_GAUGE_BG      = QColor(42, 42, 54)


def _health_color(healthy: bool | None) -> str:
    if healthy is None:
        return _TEXT_DIM
    return _GREEN if healthy else _RED


def _score_color(score: float) -> str:
    if score >= 8:
        return _GREEN
    elif score >= 5:
        return _AMBER
    return _RED


# ---------------------------------------------------------------------------
# Custom gauge widget
# ---------------------------------------------------------------------------
class _ScoreGauge(QWidget):
    """Custom-painted arc gauge showing tuning score 0–10."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._score: float = 0.0
        self._verdict: str = "No data"
        self._animating_to: float = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._animate_step)
        self.setMinimumSize(200, 170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_score(self, score: float, verdict: str):
        self._verdict = verdict
        self._animating_to = max(0.0, min(10.0, score))
        if not self._timer.isActive():
            self._timer.start()

    def reset(self):
        self._score = 0.0
        self._verdict = "No data"
        self._animating_to = 0.0
        self.update()

    def _animate_step(self):
        diff = self._animating_to - self._score
        if abs(diff) < 0.02:
            self._score = self._animating_to
            self._timer.stop()
        else:
            self._score += diff * 0.12
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2

        # Arc spans 200° → -20°, so its endpoints dip sin(20°)·r below cy.
        # Reserve space below for the verdict label, and above for top padding.
        verdict_reserve = 26
        top_pad = 6
        sin20 = math.sin(math.radians(20))
        max_r_h = max(30.0, (h - verdict_reserve - top_pad) / (1.0 + sin20))
        max_r_w = max(30.0, (w - 24) / 2.0)
        radius = min(max_r_h, max_r_w)
        gauge_size = radius * 2
        cy = top_pad + radius

        arc_rect = QRectF(cx - radius, cy - radius, gauge_size, gauge_size)
        pen_width = max(8, radius * 0.12)

        start_angle = 200 * 16
        span_angle = -220 * 16

        bg_pen = QPen(QColor(_GAUGE_BG), pen_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(arc_rect, start_angle, span_angle)

        if self._score > 0.01:
            fraction = self._score / 10.0
            value_span = int(-220 * 16 * fraction)

            if self._score < 4:
                arc_color = _GAUGE_RED
            elif self._score < 7:
                t = (self._score - 4) / 3.0
                arc_color = QColor(
                    int(_GAUGE_RED.red() + t * (_GAUGE_AMBER.red() - _GAUGE_RED.red())),
                    int(_GAUGE_RED.green() + t * (_GAUGE_AMBER.green() - _GAUGE_RED.green())),
                    int(_GAUGE_RED.blue() + t * (_GAUGE_AMBER.blue() - _GAUGE_RED.blue())),
                )
            else:
                t = (self._score - 7) / 3.0
                arc_color = QColor(
                    int(_GAUGE_AMBER.red() + t * (_GAUGE_GREEN.red() - _GAUGE_AMBER.red())),
                    int(_GAUGE_AMBER.green() + t * (_GAUGE_GREEN.green() - _GAUGE_AMBER.green())),
                    int(_GAUGE_AMBER.blue() + t * (_GAUGE_GREEN.blue() - _GAUGE_AMBER.blue())),
                )

            glow_pen = QPen(
                QColor(arc_color.red(), arc_color.green(), arc_color.blue(), 50),
                pen_width + 6, Qt.SolidLine, Qt.RoundCap,
            )
            painter.setPen(glow_pen)
            painter.drawArc(arc_rect, start_angle, value_span)

            arc_pen = QPen(arc_color, pen_width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(arc_pen)
            painter.drawArc(arc_rect, start_angle, value_span)

        score_font = QFont("Consolas", max(18, int(radius * 0.35)), QFont.Bold)
        painter.setFont(score_font)
        score_str = f"{self._score:.1f}"
        painter.setPen(QColor(_TEXT_BRIGHT))

        fm = QFontMetrics(score_font)
        text_rect = fm.boundingRect(score_str)
        text_y = cy - radius * 0.10
        painter.drawText(QPointF(cx - text_rect.width() / 2, text_y), score_str)

        sub_font = QFont("Consolas", max(8, int(radius * 0.14)))
        painter.setFont(sub_font)
        painter.setPen(QColor(_TEXT_DIM))
        painter.drawText(QPointF(cx + text_rect.width() / 2 + 2, text_y), "/10")

        verdict_font = QFont("Segoe UI", max(8, int(radius * 0.13)))
        painter.setFont(verdict_font)
        painter.setPen(QColor(_score_color(self._score)))
        vfm = QFontMetrics(verdict_font)
        vw = vfm.horizontalAdvance(self._verdict)
        painter.drawText(QPointF(cx - vw / 2, h - 8), self._verdict)

        tick_pen = QPen(QColor(_BORDER_LIGHT), 1)
        painter.setPen(tick_pen)
        small_font = QFont("Consolas", max(6, int(radius * 0.09)))
        painter.setFont(small_font)
        for i in range(11):
            angle_deg = 200 - (i / 10.0) * 220
            angle_rad = math.radians(angle_deg)
            inner_r = radius - pen_width / 2 - 4
            outer_r = radius - pen_width / 2 - 10
            x1 = cx + inner_r * math.cos(angle_rad)
            y1 = cy - inner_r * math.sin(angle_rad)
            x2 = cx + outer_r * math.cos(angle_rad)
            y2 = cy - outer_r * math.sin(angle_rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            if i % 2 == 0:
                lbl = str(i)
                lw = QFontMetrics(small_font).horizontalAdvance(lbl)
                label_r = outer_r - 8
                lx = cx + label_r * math.cos(angle_rad) - lw / 2
                ly = cy - label_r * math.sin(angle_rad) + 3
                painter.drawText(QPointF(lx, ly), lbl)

        painter.end()


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
    coe_read_done = Signal(int, object, str)   # axis, DriveProfile, error_msg
    coe_write_done = Signal(int, object, str)  # axis, results_dict, error_msg


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
class TunerPanel(QDockWidget):
    """Dockable servo tuner panel with drive profile editor and analysis."""

    analysis_complete = Signal()
    _correction_write_done = Signal(str)  # error message ("" on success)

    def __init__(self, parent=None):
        super().__init__("Servo Tuner", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(560)
        # Prevent the dock from forcing the main window to resize
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # --- State ---
        self._data_provider: Callable | None = None
        self._connection = None
        self._conn_lock: threading.Lock | None = None
        self._pos_metrics: StepResponseMetrics | None = None
        self._vel_metrics: VelocityLoopMetrics | None = None
        self._tuning_result: TuningResult | None = None

        # Per-axis drive profiles: {axis_int: DriveProfile}
        self._profiles: dict[int, DriveProfile] = {}
        self._param_widgets: dict[str, QWidget] = {}  # attr → spinbox/combo
        self._param_frame: QFrame | None = None
        self._axis_combo: QComboBox | None = None
        self._drive_combo: QComboBox | None = None
        self._read_btn: QPushButton | None = None
        self._write_btn: QPushButton | None = None

        # Signals for CoE background ops
        self._coe_signals = _CoESignals()
        self._coe_signals.coe_read_done.connect(self._on_coe_read_done)
        self._coe_signals.coe_write_done.connect(self._on_coe_write_done)
        self._correction_write_done.connect(self._on_correction_write_done)

        self._build_ui()

    # ================================================================
    # Public API
    # ================================================================

    def set_data_provider(self, provider: Callable):
        """Set callback → (time_arr, params_dict)."""
        self._data_provider = provider

    def set_connection(self, connection, conn_lock=None):
        """Provide active TUA.TrioConnection for CoE reads/writes."""
        self._connection = connection
        self._conn_lock = conn_lock
        self._update_drive_buttons()
        self._update_action_buttons()

    def get_all_profiles(self) -> dict[int, dict]:
        """Return all per-axis profiles as plain dicts for QSettings persistence."""
        return {axis: p.to_dict() for axis, p in self._profiles.items()}

    def set_all_profiles(self, profiles: dict[int, dict]):
        """Restore per-axis profiles from plain dicts loaded from QSettings."""
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

        title = QLabel("SERVO TUNER")
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
            QPushButton:hover {{
                background-color: #ffb52e;
            }}
            QPushButton:pressed {{
                background-color: #e09000;
            }}
            QPushButton:disabled {{
                background-color: #4a4a4a;
                color: #777;
            }}
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

        # ── Left column: Drive Profile + Gauge ──────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.setContentsMargins(0, 0, 0, 0)

        self._drive_card = self._build_drive_profile_section()
        left_col.addWidget(self._drive_card)

        self._gauge = _ScoreGauge()
        left_col.addWidget(self._gauge)

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

        self._grade_card = self._build_grade_card()
        right_col.addWidget(self._grade_card)

        self._corrections_card = self._build_corrections_card()
        right_col.addWidget(self._corrections_card)

        self._warnings_card = self._build_warnings_card()
        right_col.addWidget(self._warnings_card)

        right_col.addStretch()
        columns.addLayout(right_col, 1)

        scroll.setWidget(self._scroll_content)
        root.addWidget(scroll, 1)

        # ── Persistent action bar (always visible) ──────────────────
        action_sep = QFrame()
        action_sep.setFixedHeight(1)
        action_sep.setStyleSheet(f"background-color: {_BORDER};")
        root.addWidget(action_sep)

        _action_btn_style = (
            "QPushButton {{"
            "  background-color: {bg}; color: {fg};"
            "  font-family: Consolas; font-size: 8pt; font-weight: bold;"
            "  letter-spacing: 1px; border: 1px solid {border};"
            "  border-radius: 4px; padding: 6px 10px;"
            "}}"
            "QPushButton:hover {{ background-color: {hover}; }}"
            "QPushButton:pressed {{ background-color: {press}; }}"
            "QPushButton:disabled {{ background-color: #3a3a3a; color: #555;"
            "  border-color: #4a4a4a; }}"
        )

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.setContentsMargins(0, 4, 0, 0)

        self._btn_apply = QPushButton("APPLY TO PROFILE")
        self._btn_apply.setCursor(Qt.PointingHandCursor)
        self._btn_apply.setToolTip(
            "Update the drive profile spinboxes above with the proposed values.\n"
            "Does NOT write to the physical drive."
        )
        self._btn_apply.setStyleSheet(
            _action_btn_style.format(
                bg=_BLUE_MUTED, fg="#fff", border="#7a9fd0",
                hover="#6b90c5", press="#4a70a5",
            )
        )
        self._btn_apply.clicked.connect(self._on_apply_to_profile)
        self._btn_apply.setEnabled(False)
        btn_row.addWidget(self._btn_apply)

        self._btn_correction_write = QPushButton("WRITE CORRECTIONS")
        self._btn_correction_write.setCursor(Qt.PointingHandCursor)
        self._btn_correction_write.setToolTip(
            "Write ONLY the proposed corrections to the physical drive\n"
            "via EtherCAT CoE SDO. Requires active controller connection."
        )
        self._btn_correction_write.setStyleSheet(
            _action_btn_style.format(
                bg="#2e8b3e", fg="#fff", border="#3aad4a",
                hover="#38a548", press="#267a34",
            )
        )
        self._btn_correction_write.clicked.connect(self._on_write_corrections)
        self._btn_correction_write.setEnabled(False)
        btn_row.addWidget(self._btn_correction_write)

        root.addLayout(btn_row)

        self.setWidget(container)
        self._reset_display()

    # ================================================================
    # Drive profile section
    # ================================================================

    def _build_drive_profile_section(self) -> QGroupBox:
        """Build the drive profile configurator group."""
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

        # ── Axis + Drive type row ──────────────────────────────────
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

        # Read from Drive
        self._read_btn = QPushButton("Read")
        self._read_btn.setFixedHeight(22)
        self._read_btn.setFixedWidth(55)
        self._read_btn.setEnabled(False)
        self._read_btn.setToolTip(
            "Read Pn parameters from the drive via EtherCAT CoE SDO."
        )
        self._read_btn.clicked.connect(self._on_read_from_drive)
        selector_row.addWidget(self._read_btn)

        # Write to Drive
        self._write_btn = QPushButton("Write")
        self._write_btn.setFixedHeight(22)
        self._write_btn.setFixedWidth(55)
        self._write_btn.setEnabled(False)
        self._write_btn.setToolTip(
            "Write ALL Pn parameters to the drive via EtherCAT CoE SDO."
        )
        self._write_btn.clicked.connect(self._on_write_to_drive)
        selector_row.addWidget(self._write_btn)

        outer.addLayout(selector_row)

        # ── Parameter fields (shown only for DX3 / DX4) ───────────
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

    def _update_drive_buttons(self):
        if self._read_btn is None or self._write_btn is None or self._drive_combo is None:
            return
        drive_type = self._drive_combo.currentText()
        enabled = self._connection is not None and drive_type in ("DX3", "DX4")
        self._read_btn.setEnabled(enabled)
        self._write_btn.setEnabled(enabled)

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
        self._write_btn.setText("Write")
        self._update_drive_buttons()
        if error:
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

    def _build_grade_card(self) -> QFrame:
        card = _card_frame()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        lbl = QLabel("ASSESSMENT")
        lbl.setStyleSheet(
            f"color: {_TEXT_BRIGHT}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        lay.addWidget(lbl)
        lay.addWidget(_separator())

        self._grade_details_label = QLabel("")
        self._grade_details_label.setWordWrap(True)
        self._grade_details_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 8pt; line-height: 1.4;"
        )
        lay.addWidget(self._grade_details_label)

        return card

    def _build_corrections_card(self) -> QFrame:
        card = _card_frame()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        lbl = QLabel("CORRECTIONS")
        lbl.setStyleSheet(
            f"color: {_TEXT_BRIGHT}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        hdr.addWidget(lbl)
        hdr.addStretch()

        self._confidence_lbl = QLabel("")
        self._confidence_lbl.setStyleSheet(f"font-size: 7pt;")
        hdr.addWidget(self._confidence_lbl)
        lay.addLayout(hdr)
        lay.addWidget(_separator())

        self._corrections_layout = QVBoxLayout()
        self._corrections_layout.setSpacing(4)
        lay.addLayout(self._corrections_layout)

        self._no_corrections_lbl = QLabel("No corrections needed")
        self._no_corrections_lbl.setStyleSheet(
            f"color: {_GREEN}; font-size: 8pt; font-style: italic;"
        )
        lay.addWidget(self._no_corrections_lbl)

        self._reasons_label = QLabel("")
        self._reasons_label.setWordWrap(True)
        self._reasons_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 8pt; padding-top: 4px;"
        )
        lay.addWidget(self._reasons_label)

        return card

    def _build_warnings_card(self) -> QFrame:
        card = _card_frame()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        lbl = QLabel("WARNINGS")
        lbl.setStyleSheet(
            f"color: {_AMBER}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        lay.addWidget(lbl)
        lay.addWidget(_separator())

        self._warnings_label = QLabel("")
        self._warnings_label.setWordWrap(True)
        self._warnings_label.setStyleSheet(
            f"color: {_AMBER}; font-size: 8pt; line-height: 1.4;"
        )
        lay.addWidget(self._warnings_label)

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
        self._gauge.reset()
        self._vel_dot.set_healthy(None)
        self._vel_status_lbl.setText("--")
        self._vel_issues_label.hide()
        self._pos_dot.set_healthy(None)
        self._clear_layout(self._vel_metrics_layout)
        self._clear_layout(self._pos_metrics_layout)
        self._clear_layout(self._corrections_layout)
        self._grade_details_label.setText("")
        self._no_corrections_lbl.show()
        self._reasons_label.setText("")
        self._confidence_lbl.setText("")
        self._warnings_label.setText("")
        self._warnings_card.hide()
        self._grade_card.hide()
        self._corrections_card.hide()
        self._btn_apply.setEnabled(False)
        self._btn_correction_write.setEnabled(False)

        # Placeholder rows
        self._add_metric_row(self._vel_metrics_layout, "Accel overshoot", "--", "%")
        self._add_metric_row(self._vel_metrics_layout, "Cruise tracking", "--")
        self._add_metric_row(self._vel_metrics_layout, "Settle time", "--", "ms")
        self._add_metric_row(self._vel_metrics_layout, "Oscillations", "--")

        self._add_metric_row(self._pos_metrics_layout, "Overshoot", "--", "%")
        self._add_metric_row(self._pos_metrics_layout, "Settling time", "--", "ms")
        self._add_metric_row(self._pos_metrics_layout, "Rise time", "--", "ms")
        self._add_metric_row(self._pos_metrics_layout, "Oscillations", "--")
        self._add_metric_row(self._pos_metrics_layout, "Steady-state err", "--")
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
        self._vel_status_lbl.setText("Healthy" if vm.is_healthy else "Issues detected")
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

        score, _, _ = pm.grade()
        self._pos_dot.set_healthy(score >= 5)

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

        self._add_metric_row(
            self._pos_metrics_layout, "Damping ratio",
            f"{pm.damping_ratio:.3f}", "",
        )

        if pm.natural_freq_est_hz > 0:
            self._add_metric_row(
                self._pos_metrics_layout, "Natural freq",
                f"{pm.natural_freq_est_hz:.1f}", "Hz",
            )

    def _populate_grade(self, pm: StepResponseMetrics):
        score, verdict, details = pm.grade()

        is_empty = (
            pm.overshoot_pct == 0 and pm.oscillation_count == 0
            and pm.settling_time_ms == 0
        )
        if is_empty:
            self._gauge.set_score(0, "No data")
            self._grade_card.hide()
            return

        self._gauge.set_score(score, verdict)
        self._grade_card.show()

        lines = []
        for d in details:
            if "excellent" in d.lower() or "well damped" in d.lower():
                lines.append(f'<span style="color:{_GREEN}">\u2713 {d}</span>')
            elif "good" in d.lower():
                lines.append(f'<span style="color:{_CYAN}">\u2713 {d}</span>')
            else:
                lines.append(f'<span style="color:{_AMBER}">\u2022 {d}</span>')
        self._grade_details_label.setText("<br>".join(lines))

    def _populate_corrections(self, result: TuningResult | None,
                              profile: DriveProfile | None):
        self._clear_layout(self._corrections_layout)

        if result is None:
            self._corrections_card.hide()
            return

        self._corrections_card.show()

        delta = result.to_profile_delta()
        has_changes = len(delta) > 0

        conf = result.confidence
        conf_colors = {"high": _GREEN, "medium": _AMBER, "low": _RED}
        conf_c = conf_colors.get(conf, _TEXT_DIM)
        self._confidence_lbl.setText(f"confidence: {conf}")
        self._confidence_lbl.setStyleSheet(
            f"color: {conf_c}; font-size: 7pt; font-family: Consolas;"
        )

        if has_changes:
            self._no_corrections_lbl.hide()
            pn_labels = {
                "pn101": "Pn101  Servo Rigidity",
                "pn102": "Pn102  Speed Loop Gain",
                "pn103": "Pn103  Speed Loop Ti",
                "pn104": "Pn104  Position Loop Gain",
                "pn106": "Pn106  Load Inertia",
                "pn112": "Pn112  Speed Feedforward",
                "pn113": "Pn113  Speed FF Filter",
                "pn114": "Pn114  Torque Feedforward",
                "pn115": "Pn115  Torque FF Filter",
                "pn135": "Pn135  Speed Filter",
            }
            for attr, new_val in delta.items():
                current_val = getattr(profile, attr, None) if profile else None

                row_frame = QFrame()
                row_frame.setStyleSheet(
                    f"QFrame {{ background-color: {_BG_PANEL};"
                    f" border: 1px solid {_BORDER}; border-radius: 4px; }}"
                )
                row_lay = QVBoxLayout(row_frame)
                row_lay.setContentsMargins(8, 6, 8, 6)
                row_lay.setSpacing(2)

                name_lbl = QLabel(pn_labels.get(attr, attr))
                name_lbl.setStyleSheet(
                    f"color: {_TEXT}; font-family: Consolas; font-size: 8pt;"
                    f" border: none;"
                )
                row_lay.addWidget(name_lbl)

                val_row = QHBoxLayout()
                val_row.setSpacing(4)

                if current_val is not None:
                    cur_lbl = QLabel(str(current_val))
                    cur_lbl.setStyleSheet(
                        f"color: {_TEXT_DIM}; font-family: Consolas;"
                        f" font-size: 10pt; border: none;"
                    )
                    val_row.addWidget(cur_lbl)

                    arrow = QLabel("\u2192")
                    arrow.setStyleSheet(
                        f"color: {_ACCENT}; font-size: 12pt; border: none;"
                    )
                    val_row.addWidget(arrow)

                new_lbl = QLabel(str(new_val))
                new_lbl.setStyleSheet(
                    f"color: {_ACCENT}; font-family: Consolas; font-size: 12pt;"
                    f" font-weight: bold; border: none;"
                )
                val_row.addWidget(new_lbl)
                val_row.addStretch()

                if current_val is not None and current_val != 0:
                    pct_change = ((new_val - current_val) / current_val) * 100
                    sign = "+" if pct_change > 0 else ""
                    badge_color = _RED if pct_change < 0 else _GREEN
                    badge = QLabel(f"{sign}{pct_change:.0f}%")
                    badge.setStyleSheet(
                        f"color: {badge_color}; font-family: Consolas;"
                        f" font-size: 8pt; font-weight: bold;"
                        f" background-color: {badge_color}22;"
                        f" border: 1px solid {badge_color}44;"
                        f" border-radius: 3px; padding: 1px 4px;"
                    )
                    val_row.addWidget(badge)

                row_lay.addLayout(val_row)
                self._corrections_layout.addWidget(row_frame)
        else:
            self._no_corrections_lbl.show()

        corrections_text = result.diagnostics.get("corrections", [])
        if corrections_text:
            reason_lines = [f"\u2022 {r}" for r in corrections_text]
            self._reasons_label.setText("\n".join(reason_lines))
        else:
            self._reasons_label.setText("")

    def _populate_warnings(self, result: TuningResult | None):
        if result is None or not result.warnings:
            self._warnings_card.hide()
            return
        self._warnings_card.show()
        lines = [f"\u26a0 {w}" for w in result.warnings]
        self._warnings_label.setText("\n".join(lines))

    # ================================================================
    # Action button state
    # ================================================================

    def _update_action_buttons(self):
        has_delta = (
            self._tuning_result is not None
            and len(self._tuning_result.to_profile_delta()) > 0
        )
        self._btn_apply.setEnabled(has_delta)
        self._btn_correction_write.setEnabled(has_delta and self._connection is not None)

    # ================================================================
    # Apply / Write corrections
    # ================================================================

    def _on_apply_to_profile(self):
        """Push proposed corrections into the drive profile spinboxes."""
        if self._tuning_result is None:
            return
        delta = self._tuning_result.to_profile_delta()
        if not delta:
            return

        axis = self._current_axis()
        profile = self._profiles.get(axis)
        if profile is None:
            return

        for attr, value in delta.items():
            setattr(profile, attr, value)

        if axis == self._current_axis():
            self._load_profile_to_ui(profile)

        names = ", ".join(f"Pn{k[2:]}" for k in delta)
        self._status_label.setText(f"Applied to profile: {names}")
        self._status_label.setStyleSheet(f"color: {_GREEN}; font-size: 8pt;")

    def _on_write_corrections(self):
        """Write only the proposed corrections to the drive via CoE SDO."""
        if self._tuning_result is None or self._connection is None:
            return
        delta = self._tuning_result.to_profile_delta()
        if not delta:
            return

        axis = self._current_axis()
        names = ", ".join(f"Pn{k[2:]}={v}" for k, v in delta.items())
        reply = QMessageBox.question(
            self, "Write Corrections to Drive",
            f"Write proposed corrections to axis {axis}?\n\n{names}\n\n"
            "This will overwrite these drive parameters.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._btn_correction_write.setEnabled(False)
        self._btn_correction_write.setText("Writing\u2026")
        connection = self._connection
        conn_lock = self._conn_lock

        def _do_write():
            errors = []
            for attr, value in delta.items():
                try:
                    if conn_lock:
                        with conn_lock:
                            write_single_pn(connection, axis, attr, value)
                    else:
                        write_single_pn(connection, axis, attr, value)
                except Exception as exc:
                    errors.append(f"Pn{attr[2:]}: {exc}")
            self._correction_write_done.emit("\n".join(errors) if errors else "")

        threading.Thread(
            target=_do_write, name="TunerCorrectionWrite", daemon=True,
        ).start()

    def _on_correction_write_done(self, error: str):
        self._btn_correction_write.setText("WRITE CORRECTIONS")
        self._update_action_buttons()

        if error:
            QMessageBox.warning(
                self, "CoE Write Error",
                f"Some parameters failed to write:\n{error}",
            )
        else:
            delta = self._tuning_result.to_profile_delta() if self._tuning_result else {}
            names = ", ".join(f"Pn{k[2:]}" for k in delta)
            self._status_label.setText(f"Written to drive: {names}")
            self._status_label.setStyleSheet(f"color: {_GREEN}; font-size: 8pt;")
            # Sync profile spinboxes
            self._on_apply_to_profile()

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

        dpos = params.get(ch_dpos) if ch_dpos else None
        mpos = params.get(ch_mpos) if ch_mpos else None
        mvel = params.get(ch_mvel) if ch_mvel else None
        dvel_raw = params.get(ch_dvel) if ch_dvel else None

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
        # DEMAND_SPEED is captured as user-units per servocycle; scale to units/second.
        demand_velocity = np.asarray(dvel_raw, dtype=np.float64) / float(servo_period_sec)
        time_np = np.asarray(time_arr, dtype=np.float64)

        self._status_label.setText("Analyzing\u2026")
        self._status_label.setStyleSheet(f"color: {_ACCENT}; font-size: 8pt;")

        try:
            pos_m, vel_m = ClassicalTuner.analyze_step_response(
                time_np, response, command, velocity, demand_velocity,
            )
        except Exception as exc:
            logger.exception("Step response analysis failed")
            self._status_label.setText(f"Analysis error: {exc}")
            self._status_label.setStyleSheet(f"color: {_RED}; font-size: 8pt;")
            return

        self._pos_metrics = pos_m
        self._vel_metrics = vel_m

        # Get drive profile for corrections
        axis = self._current_axis()
        profile = self._profiles.get(axis)

        result: TuningResult | None = None
        if profile is not None and profile.has_drive_params():
            try:
                result = ClassicalTuner.suggest_corrections(pos_m, vel_m, profile)
            except Exception as exc:
                logger.exception("Correction suggestion failed")
                result = TuningResult(
                    warnings=[f"Correction engine error: {exc}"],
                    confidence="low",
                )
        self._tuning_result = result

        # Populate UI
        self._populate_vel(vel_m)
        self._populate_pos(pos_m)
        self._populate_grade(pos_m)
        self._populate_corrections(result, profile)
        self._populate_warnings(result)

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

        self._update_action_buttons()
        self.analysis_complete.emit()
