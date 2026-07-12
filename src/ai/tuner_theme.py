"""Shared colours and small widgets for the Servo Loop Analyser panel."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QRadialGradient

BG_DARK      = "#1a1a22"
BG_CARD      = "#22222c"
BG_PANEL     = "#2a2a36"
BORDER       = "#3a3a4a"
BORDER_LIGHT = "#4b4b5a"
TEXT         = "#d4d4d4"
TEXT_DIM     = "#888899"
TEXT_BRIGHT  = "#f0f0f5"
ACCENT       = "#FFA500"
CYAN         = "#00d4aa"
GREEN        = "#2ecc71"
AMBER        = "#f39c12"
RED          = "#e74c3c"

GROUP_STYLE = (
    f"QGroupBox {{ color: {TEXT_DIM}; font-size: 8pt;"
    f" border: 1px solid {BORDER}; border-radius: 4px;"
    f" margin-top: 8px; padding-top: 6px; }}"
    f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px;"
    f" padding: 0 4px; color: {TEXT}; }}"
)

CARD_STYLE = (
    f"QFrame {{ background-color: {BG_CARD}; border: 1px solid {BORDER};"
    f" border-radius: 6px; }}"
)


def health_color(healthy: bool | None) -> str:
    if healthy is None:
        return TEXT_DIM
    return GREEN if healthy else RED


def metric_label(name: str, value: str, unit: str = "",
                 color: str = CYAN) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(4)
    lbl_name = QLabel(name)
    lbl_name.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8pt;")
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
        lbl_unit.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
        lbl_unit.setFixedWidth(28)
        row.addWidget(lbl_unit)
    return row


def separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet(f"color: {BORDER};")
    sep.setFixedHeight(1)
    return sep


def card_frame() -> QFrame:
    frame = QFrame()
    frame.setStyleSheet(CARD_STYLE)
    return frame


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w:
            w.deleteLater()
        sub = item.layout()
        if sub:
            clear_layout(sub)


class HealthDot(QWidget):
    """Small glowing status dot: green / red / dim-grey (unknown)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(TEXT_DIM)
        self.setFixedSize(14, 14)

    def set_healthy(self, healthy: bool | None):
        if healthy is None:
            self._color = QColor(TEXT_DIM)
        elif healthy:
            self._color = QColor(GREEN)
        else:
            self._color = QColor(RED)
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
