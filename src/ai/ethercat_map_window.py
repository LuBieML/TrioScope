"""
EtherCAT Network Map window — visual topology of discovered slaves.

Shows the controller on the left with EtherCAT slot ports, connected slaves
drawn as a horizontal bus chain.  Each slave box shows axis, address, status,
and drive type.
"""

import logging
import threading
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget,
    QScrollArea, QFrame, QSizePolicy, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush, QPainterPath

import Trio_UnifiedApi as TUA

from .ethercat_scan import scan_network, EthercatNetwork, EthercatSlot, EthercatSlave

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_CLR_BG          = QColor("#2e2e2e")
_CLR_CARD_BG     = QColor("#3a3a3a")
_CLR_CARD_BORDER = QColor("#606060")
_CLR_TEXT         = QColor("#d4d4d4")
_CLR_TEXT_DIM     = QColor("#888888")
_CLR_ACCENT      = QColor("#FFA500")   # orange — matches app theme
_CLR_GREEN        = QColor("#00cc00")
_CLR_RED          = QColor("#f14c4c")
_CLR_BLUE         = QColor("#4a9eff")
_CLR_CONTROLLER   = QColor("#2e8b3e")
_CLR_BUS_LINE     = QColor("#FFA500")


# ---------------------------------------------------------------------------
# Helper: state → colour
# ---------------------------------------------------------------------------
def _state_colour(state) -> QColor:
    if state == TUA.EthercatState.Operational:
        return _CLR_GREEN
    if state == TUA.EthercatState.SafeOperational:
        return _CLR_ACCENT
    if state == TUA.EthercatState.PreOperational:
        return QColor("#cccc00")
    return _CLR_RED


def _drive_type_label(raw: int) -> str:
    """Best-effort decode of DRIVE_TYPE axis parameter."""
    known = {0: "Unknown", 41: "DX3", 42: "DX4"}
    return known.get(raw, f"Type {raw}")


# ---------------------------------------------------------------------------
# Slave card widget
# ---------------------------------------------------------------------------
class _SlaveCard(QFrame):
    """Visual card for one EtherCAT slave."""

    def __init__(self, slave: EthercatSlave, parent=None):
        super().__init__(parent)
        self.slave = slave
        self.setFixedSize(160, 130)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet(
            f"background-color: {_CLR_CARD_BG.name()};"
            f" border: 2px solid {(_CLR_GREEN if slave.online else _CLR_RED).name()};"
            f" border-radius: 6px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Header — position badge
        hdr = QHBoxLayout()
        pos_label = QLabel(f"#{slave.position}")
        pos_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        pos_label.setStyleSheet(f"color: {_CLR_ACCENT.name()}; border: none;")
        hdr.addWidget(pos_label)
        hdr.addStretch()
        dot = QLabel("\u25cf")
        dot.setStyleSheet(
            f"color: {(_CLR_GREEN if slave.online else _CLR_RED).name()};"
            " font-size: 12pt; border: none;"
        )
        dot.setToolTip("Online" if slave.online else "Offline")
        hdr.addWidget(dot)
        layout.addLayout(hdr)

        # Axis
        axis_lbl = QLabel(f"Axis {slave.axis}" if slave.axis >= 0 else "No axis")
        axis_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        axis_lbl.setStyleSheet(f"color: {_CLR_TEXT.name()}; border: none;")
        axis_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(axis_lbl)

        # Drive type
        dt = _drive_type_label(slave.drive_type)
        dt_lbl = QLabel(dt)
        dt_lbl.setFont(QFont("Segoe UI", 9))
        dt_lbl.setStyleSheet(f"color: {_CLR_BLUE.name()}; border: none;")
        dt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(dt_lbl)

        # Address / status row
        info = QLabel(f"Addr {slave.address}  |  Status 0x{slave.drive_status:04X}")
        info.setFont(QFont("Segoe UI", 7))
        info.setStyleSheet(f"color: {_CLR_TEXT_DIM.name()}; border: none;")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        layout.addStretch()


# ---------------------------------------------------------------------------
# Slot row — controller port + bus chain of slaves
# ---------------------------------------------------------------------------
class _SlotRow(QWidget):
    """Horizontal strip: slot label → bus line → slave cards."""

    def __init__(self, ecat_slot: EthercatSlot, parent=None):
        super().__init__(parent)
        self.ecat_slot = ecat_slot

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # Slot header card
        slot_card = QFrame()
        slot_card.setFixedSize(110, 130)
        slot_card.setStyleSheet(
            f"background-color: {_CLR_CONTROLLER.name()};"
            f" border: 2px solid {_state_colour(ecat_slot.state).name()};"
            " border-radius: 6px;"
        )
        sc_layout = QVBoxLayout(slot_card)
        sc_layout.setContentsMargins(6, 6, 6, 6)
        sc_layout.setSpacing(2)

        title = QLabel(f"Slot {ecat_slot.slot}")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color: white; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sc_layout.addWidget(title)

        state_lbl = QLabel(ecat_slot.state_name)
        state_lbl.setFont(QFont("Segoe UI", 8))
        state_lbl.setStyleSheet(
            f"color: {_state_colour(ecat_slot.state).name()}; border: none;"
        )
        state_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sc_layout.addWidget(state_lbl)

        count_lbl = QLabel(f"{ecat_slot.num_slaves} slave(s)")
        count_lbl.setFont(QFont("Segoe UI", 8))
        count_lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM.name()}; border: none;")
        count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sc_layout.addWidget(count_lbl)

        sc_layout.addStretch()
        layout.addWidget(slot_card)

        # Bus connector + slave cards
        if ecat_slot.slaves:
            for slave in ecat_slot.slaves:
                # Connector line
                connector = QWidget()
                connector.setFixedSize(30, 130)
                connector.setStyleSheet("background: transparent;")
                layout.addWidget(connector)

                card = _SlaveCard(slave)
                layout.addWidget(card)

        layout.addStretch()

    def paintEvent(self, event):
        """Draw the bus line connecting slot port to slaves."""
        super().paintEvent(event)
        if not self.ecat_slot.slaves:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(_CLR_BUS_LINE, 3)
        painter.setPen(pen)

        # Horizontal line through the middle of all cards
        y_mid = 65  # middle of 130px cards
        # Start from right edge of slot card
        x_start = 110
        # End at right edge of last slave card
        n = len(self.ecat_slot.slaves)
        x_end = 110 + n * (30 + 160)
        painter.drawLine(x_start, y_mid, x_end, y_mid)

        # Small vertical ticks at each slave entry
        for i in range(n):
            x_tick = 110 + i * (30 + 160) + 30
            painter.drawLine(x_tick, y_mid - 8, x_tick, y_mid + 8)

        painter.end()


# ---------------------------------------------------------------------------
# Signal bridge for thread → GUI
# ---------------------------------------------------------------------------
class _ScanSignals(QObject):
    finished = Signal(object)   # EthercatNetwork
    error = Signal(str)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class EthercatMapWindow(QDialog):
    """Scrollable window showing the discovered EtherCAT topology."""

    def __init__(self, connection: TUA.TrioConnection, parent=None):
        super().__init__(parent)
        self._connection = connection
        self._network: Optional[EthercatNetwork] = None
        self._signals = _ScanSignals()
        self._signals.finished.connect(self._on_scan_finished)
        self._signals.error.connect(self._on_scan_error)

        self.setWindowTitle("EtherCAT Network Map")
        self.resize(900, 500)
        self.setMinimumSize(600, 300)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_scan = QPushButton("\u27f3  Scan Network")
        self._btn_scan.setFixedHeight(30)
        self._btn_scan.clicked.connect(self._start_scan)
        toolbar.addWidget(self._btn_scan)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888888;")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        toolbar.addWidget(self._summary_label)
        root.addLayout(toolbar)

        # Scrollable content area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._content_layout.setSpacing(12)
        self._scroll.setWidget(self._content)
        root.addWidget(self._scroll, 1)

        # Kick off initial scan
        QTimer.singleShot(100, self._start_scan)

    # ----- scanning --------------------------------------------------------

    def _start_scan(self):
        if not self._connection:
            self._status_label.setText("No connection")
            return

        self._btn_scan.setEnabled(False)
        self._status_label.setText("Scanning...")

        def _worker():
            try:
                net = scan_network(self._connection)
                self._signals.finished.emit(net)
            except Exception as exc:
                logger.exception("EtherCAT scan failed")
                self._signals.error.emit(str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_scan_finished(self, network: EthercatNetwork):
        self._network = network
        self._btn_scan.setEnabled(True)
        self._status_label.setText("")
        self._rebuild_map()

    def _on_scan_error(self, msg: str):
        self._btn_scan.setEnabled(True)
        self._status_label.setText(f"Scan error: {msg}")

    # ----- map rendering ---------------------------------------------------

    def _rebuild_map(self):
        """Rebuild the visual map from the current network scan."""
        # Clear old content
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        net = self._network
        if not net:
            return

        total_slaves = len(net.all_slaves)
        online = len(net.online_slaves)
        active = len(net.active_slots)
        self._summary_label.setText(
            f"{active} active slot(s)  |  {online}/{total_slaves} slave(s) online"
        )

        if total_slaves == 0:
            empty = QLabel("No EtherCAT slaves detected.\n\n"
                           "Check that the EtherCAT network is started\n"
                           "and drives are powered on.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("color: #888888; font-size: 11pt; padding: 40px;")
            self._content_layout.addWidget(empty)
            return

        # Only show slots that have slaves (or are operational)
        for ecat_slot in net.slots:
            if ecat_slot.num_slaves == 0 and not ecat_slot.is_operational:
                continue
            row = _SlotRow(ecat_slot)
            self._content_layout.addWidget(row)

        self._content_layout.addStretch()

    # ----- public API ------------------------------------------------------

    def get_network(self) -> Optional[EthercatNetwork]:
        """Return the last scan result."""
        return self._network
