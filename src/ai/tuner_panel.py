"""
Classical Tuner Panel — dockable Qt widget for real-time servo loop analysis.

Runs ClassicalTuner on the current scope capture and displays:
  - Tuning score gauge (custom-painted arc, 0–10)
  - Velocity loop health card
  - Position loop metrics card
  - Proposed corrections with current → new values
  - Cascade-aware warnings
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Callable

import numpy as np

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy, QGridLayout,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QTimer
from PySide6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QConicalGradient,
    QLinearGradient, QRadialGradient, QPainterPath, QFontMetrics,
)

from .classical_tuner import (
    ClassicalTuner, StepResponseMetrics, VelocityLoopMetrics, TuningResult,
)
from .drive_profile import DriveProfile
from .signal_metrics import SignalMetrics

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
        self.setMinimumSize(200, 140)
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
        # Gauge sits in upper portion
        gauge_size = min(w - 24, (h - 36) * 2)
        gauge_size = max(gauge_size, 100)
        radius = gauge_size / 2
        cy = h - 32  # arc centre near bottom

        arc_rect = QRectF(cx - radius, cy - radius, gauge_size, gauge_size)
        pen_width = max(8, radius * 0.12)

        # --- Background arc (220° sweep from 160° to -40°) ---
        start_angle = 200 * 16   # Qt uses 1/16th degrees
        span_angle = -220 * 16

        bg_pen = QPen(QColor(_GAUGE_BG), pen_width, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(bg_pen)
        painter.drawArc(arc_rect, start_angle, span_angle)

        # --- Coloured arc (proportional to score) ---
        if self._score > 0.01:
            fraction = self._score / 10.0
            value_span = int(-220 * 16 * fraction)

            # Gradient colour based on score
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

            # Glow effect — draw wider translucent arc behind
            glow_pen = QPen(
                QColor(arc_color.red(), arc_color.green(), arc_color.blue(), 50),
                pen_width + 6, Qt.SolidLine, Qt.RoundCap,
            )
            painter.setPen(glow_pen)
            painter.drawArc(arc_rect, start_angle, value_span)

            arc_pen = QPen(arc_color, pen_width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(arc_pen)
            painter.drawArc(arc_rect, start_angle, value_span)

        # --- Score text ---
        score_font = QFont("Consolas", max(18, int(radius * 0.35)), QFont.Bold)
        painter.setFont(score_font)
        score_str = f"{self._score:.1f}"
        painter.setPen(QColor(_TEXT_BRIGHT))

        fm = QFontMetrics(score_font)
        text_rect = fm.boundingRect(score_str)
        text_y = cy - radius * 0.10
        painter.drawText(
            QPointF(cx - text_rect.width() / 2, text_y),
            score_str,
        )

        # "/10" subscript
        sub_font = QFont("Consolas", max(8, int(radius * 0.14)))
        painter.setFont(sub_font)
        painter.setPen(QColor(_TEXT_DIM))
        painter.drawText(
            QPointF(cx + text_rect.width() / 2 + 2, text_y),
            "/10",
        )

        # --- Verdict text below ---
        verdict_font = QFont("Segoe UI", max(8, int(radius * 0.13)))
        painter.setFont(verdict_font)
        painter.setPen(QColor(_score_color(self._score)))
        vfm = QFontMetrics(verdict_font)
        vw = vfm.horizontalAdvance(self._verdict)
        painter.drawText(QPointF(cx - vw / 2, h - 8), self._verdict)

        # --- Tick marks ---
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
    """Build a single metric readout row: name ... value unit."""
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
    """Small painted circle with glow for health status."""

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
        # Glow
        glow = QRadialGradient(7, 7, 9)
        glow.setColorAt(0, QColor(self._color.red(), self._color.green(),
                                   self._color.blue(), 80))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 14, 14)
        # Dot
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(3, 3, 8, 8)
        painter.end()


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
class TunerPanel(QDockWidget):
    """Dockable servo tuner panel with visual metrics and correction display."""

    analysis_complete = Signal()

    def __init__(self, parent=None):
        super().__init__("Classical Tuner", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(300)

        self._data_provider: Callable | None = None
        self._profile_provider: Callable | None = None
        self._pos_metrics: StepResponseMetrics | None = None
        self._vel_metrics: VelocityLoopMetrics | None = None
        self._tuning_result: TuningResult | None = None

        self._build_ui()

    # ---------------------------------------------------------------- public
    def set_data_provider(self, provider: Callable):
        """Set callback → (time_arr, params_dict)."""
        self._data_provider = provider

    def set_profile_provider(self, provider: Callable):
        """Set callback → DriveProfile for the active axis."""
        self._profile_provider = provider

    # ---------------------------------------------------------------- UI
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

        # ── Scrollable content area ─────────────────────────────────
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
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(8)

        # -- Score gauge
        self._gauge = _ScoreGauge()
        self._scroll_layout.addWidget(self._gauge)

        # -- Velocity loop card
        self._vel_card = self._build_vel_card()
        self._scroll_layout.addWidget(self._vel_card)

        # -- Position loop card
        self._pos_card = self._build_pos_card()
        self._scroll_layout.addWidget(self._pos_card)

        # -- Grade details card
        self._grade_card = self._build_grade_card()
        self._scroll_layout.addWidget(self._grade_card)

        # -- Corrections card
        self._corrections_card = self._build_corrections_card()
        self._scroll_layout.addWidget(self._corrections_card)

        # -- Warnings card
        self._warnings_card = self._build_warnings_card()
        self._scroll_layout.addWidget(self._warnings_card)

        self._scroll_layout.addStretch()

        scroll.setWidget(self._scroll_content)
        root.addWidget(scroll, 1)

        self.setWidget(container)
        self._reset_display()

    # ── Card builders ───────────────────────────────────────────────

    def _build_vel_card(self) -> QFrame:
        card = _card_frame()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        # Header row
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

        # Metrics grid
        self._vel_metrics_layout = QVBoxLayout()
        self._vel_metrics_layout.setSpacing(2)
        lay.addLayout(self._vel_metrics_layout)

        # Issue list
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

        # Reasons
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

    # ── Helpers to populate metric rows ─────────────────────────────

    @staticmethod
    def _clear_layout(layout: QVBoxLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
            sub = item.layout()
            if sub:
                TunerPanel._clear_layout_recursive(sub)

    @staticmethod
    def _clear_layout_recursive(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
            sub = item.layout()
            if sub:
                TunerPanel._clear_layout_recursive(sub)

    def _add_metric_row(self, layout: QVBoxLayout, name: str, value: str,
                        unit: str = "", color: str = _CYAN):
        layout.addLayout(_metric_label(name, value, unit, color))

    # ── Reset / populate ────────────────────────────────────────────

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

        # Determine overall health from grade
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
            # Colour-code each line based on content
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

        # Confidence badge
        conf = result.confidence
        conf_colors = {"high": _GREEN, "medium": _AMBER, "low": _RED}
        conf_c = conf_colors.get(conf, _TEXT_DIM)
        self._confidence_lbl.setText(f"confidence: {conf}")
        self._confidence_lbl.setStyleSheet(
            f"color: {conf_c}; font-size: 7pt; font-family: Consolas;"
        )

        if has_changes:
            self._no_corrections_lbl.hide()
            # Build parameter change rows
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

                # Param name
                name_lbl = QLabel(pn_labels.get(attr, attr))
                name_lbl.setStyleSheet(
                    f"color: {_TEXT}; font-family: Consolas; font-size: 8pt;"
                    f" border: none;"
                )
                row_lay.addWidget(name_lbl)

                # Current → New
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

                # Delta badge
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

        # Reasons
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

    # ── Analysis entry point ────────────────────────────────────────

    def _on_analyze(self):
        if not self._data_provider:
            self._status_label.setText("No data provider connected")
            self._status_label.setStyleSheet(f"color: {_RED}; font-size: 8pt;")
            return

        time_arr, params = self._data_provider()
        if time_arr is None or params is None:
            self._status_label.setText("No captured data available — run a capture first")
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        if len(time_arr) < 20:
            self._status_label.setText("Capture too short for analysis")
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        # Find channels using SignalMetrics' fuzzy match
        from .signal_metrics import _find_channel

        ch_dpos = _find_channel(params, "dpos", "demandposition", "targetposition")
        ch_mpos = _find_channel(params, "mpos", "measuredposition", "actualposition")
        ch_mvel = _find_channel(params, "mspeed", "measuredvel", "actualvel", "vactual")

        dpos = params.get(ch_dpos) if ch_dpos else None
        mpos = params.get(ch_mpos) if ch_mpos else None
        mvel = params.get(ch_mvel) if ch_mvel else None

        if dpos is None or mpos is None:
            self._status_label.setText(
                "Need DPOS and MPOS channels for analysis. "
                "Capture demand position and measured position."
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        command = np.asarray(dpos, dtype=np.float64)
        response = np.asarray(mpos, dtype=np.float64)
        velocity = np.asarray(mvel, dtype=np.float64) if mvel is not None else None
        time_np = np.asarray(time_arr, dtype=np.float64)

        # Run analysis
        self._status_label.setText("Analyzing...")
        self._status_label.setStyleSheet(f"color: {_ACCENT}; font-size: 8pt;")

        try:
            pos_m, vel_m = ClassicalTuner.analyze_step_response(
                time_np, response, command, velocity,
            )
        except Exception as exc:
            logger.exception("Step response analysis failed")
            self._status_label.setText(f"Analysis error: {exc}")
            self._status_label.setStyleSheet(f"color: {_RED}; font-size: 8pt;")
            return

        self._pos_metrics = pos_m
        self._vel_metrics = vel_m

        # Get drive profile for corrections
        profile: DriveProfile | None = None
        if self._profile_provider:
            try:
                profile = self._profile_provider()
            except Exception:
                pass

        # Run correction suggestion
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
        ch_str = " + ".join(channels_used)

        self._status_label.setText(
            f"Analyzed {n_samples} samples ({dur_s:.2f}s) | {ch_str}"
        )
        self._status_label.setStyleSheet(f"color: {_GREEN}; font-size: 8pt;")

        self.analysis_complete.emit()
