"""
Shared colour palette and small widget helpers for the Servo Loop
Analyser panel family (tuner panel, drive profile editor, loop cards).
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QRadialGradient

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


def clear_layout(layout):
    """Recursively delete every widget and sub-layout in a layout."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w:
            w.deleteLater()
        sub = item.layout()
        if sub:
            clear_layout(sub)


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
