"""
EtherCAT Network Map window — visual topology of discovered slaves.

Diagram-style layout inspired by Trio Motion Perfect:
  Address row  →  device strip with bus line  →  axis row
"""

import logging
import threading
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget,
    QScrollArea, QFrame, QSizePolicy, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QRect
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush

import Trio_UnifiedApi as TUA

from .ethercat_scan import scan_network, EthercatNetwork, EthercatSlot, EthercatSlave

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_CLR_BG          = QColor("#2e2e2e")
_CLR_CARD_BG     = QColor("#3a3a3a")
_CLR_CARD_BORDER = QColor("#555555")
_CLR_TEXT         = QColor("#d4d4d4")
_CLR_TEXT_DIM     = QColor("#888888")
_CLR_ACCENT       = QColor("#FFA500")
_CLR_GREEN        = QColor("#00cc00")
_CLR_RED          = QColor("#f14c4c")
_CLR_BLUE         = QColor("#4a9eff")
_CLR_CONTROLLER   = QColor("#2e8b3e")
_CLR_BUS_LINE     = QColor("#FFA500")

# Layout constants
_DEV_W      = 48   # device block width
_DEV_H      = 60   # device block height
_DEV_GAP    = 8    # gap between device blocks
_LABEL_H    = 16   # row height for address / axis labels
_BUS_Y_OFF  = 6    # bus line offset above device blocks
_MARG       = 8    # outer margin


def _state_colour(state) -> QColor:
    if state == TUA.EthercatState.Operational:
        return _CLR_GREEN
    if state == TUA.EthercatState.SafeOperational:
        return _CLR_ACCENT
    if state == TUA.EthercatState.PreOperational:
        return QColor("#cccc00")
    return _CLR_RED


def _drive_type_label(raw: int) -> str:
    known = {0: "", 41: "DX3", 42: "DX4"}
    return known.get(raw, f"T{raw}")


# ---------------------------------------------------------------------------
# Diagram widget — one per EtherCAT slot
# ---------------------------------------------------------------------------
class _SlotDiagram(QWidget):
    """Custom-painted diagram for one EtherCAT slot, resembling Motion Perfect."""

    def __init__(self, ecat_slot: EthercatSlot, parent=None):
        super().__init__(parent)
        self.ecat_slot = ecat_slot

        # Filter ghost slaves
        self.devices: list[EthercatSlave] = [
            s for s in ecat_slot.slaves if s.online or s.address != 0
        ]

        n = len(self.devices)
        # Total width: margin + master label + devices + margin
        self._master_w = 70
        self._total_w = _MARG + self._master_w + _DEV_GAP + n * (_DEV_W + _DEV_GAP) + _MARG
        # Total height: top label + bus line area + device blocks + bottom label + margins
        self._total_h = _MARG + _LABEL_H + _BUS_Y_OFF + _DEV_H + _LABEL_H + _MARG

        self.setMinimumSize(self._total_w, self._total_h)
        self.setFixedHeight(self._total_h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Build tooltip map: device index → tooltip text
        self._tooltips: list[tuple[QRect, str]] = []
        for i, dev in enumerate(self.devices):
            x = self._dev_x(i)
            y = _MARG + _LABEL_H + _BUS_Y_OFF
            rect = QRect(x, y, _DEV_W, _DEV_H)
            lines = [f"Position: #{dev.position}"]
            lines.append(f"Address: {dev.address}")
            if dev.axis >= 0:
                lines.append(f"Axis: {dev.axis}")
            if dev.vendor_id:
                lines.append(f"Vendor: {dev.vendor_name}")
            if dev.drive_type:
                lines.append(f"Drive: {_drive_type_label(dev.drive_type)}")
            lines.append(f"Online: {'Yes' if dev.online else 'No'}")
            if dev.drive_status:
                lines.append(f"Status: 0x{dev.drive_status:04X}")
            self._tooltips.append((rect, "\n".join(lines)))

        self.setMouseTracking(True)

    def _dev_x(self, i: int) -> int:
        """X position of device block i."""
        return _MARG + self._master_w + _DEV_GAP + i * (_DEV_W + _DEV_GAP)

    def event(self, ev):
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.Type.ToolTip:
            pos = ev.pos()
            for rect, tip in self._tooltips:
                if rect.contains(pos):
                    from PySide6.QtWidgets import QToolTip
                    QToolTip.showText(ev.globalPos(), tip, self, rect)
                    return True
            from PySide6.QtWidgets import QToolTip
            QToolTip.hideText()
            return True
        return super().event(ev)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        n = len(self.devices)
        y_addr = _MARG                            # address label row
        y_bus  = _MARG + _LABEL_H                 # bus line y
        y_dev  = y_bus + _BUS_Y_OFF               # device block top
        y_axis = y_dev + _DEV_H + 2               # axis label row

        slot = self.ecat_slot
        font_sm = QFont("Segoe UI", 7)
        font_md = QFont("Segoe UI", 8, QFont.Weight.Bold)
        font_lg = QFont("Segoe UI", 9, QFont.Weight.Bold)

        # ── Master state box ──────────────────────────────────
        mx = _MARG
        p.setPen(QPen(_state_colour(slot.state), 1))
        p.setBrush(QBrush(_CLR_CONTROLLER))
        p.drawRoundedRect(mx, y_dev, self._master_w, _DEV_H, 4, 4)

        p.setPen(Qt.GlobalColor.white)
        p.setFont(font_md)
        p.drawText(QRect(mx, y_dev, self._master_w, _DEV_H // 2),
                   Qt.AlignmentFlag.AlignCenter, f"Slot {slot.slot}")
        p.setFont(font_sm)
        p.setPen(_state_colour(slot.state))
        p.drawText(QRect(mx, y_dev + _DEV_H // 2, self._master_w, _DEV_H // 2),
                   Qt.AlignmentFlag.AlignCenter, slot.state_name)

        # "Master state:" label above
        p.setPen(_CLR_GREEN)
        p.setFont(font_sm)
        p.drawText(QRect(mx, y_addr, self._master_w, _LABEL_H),
                   Qt.AlignmentFlag.AlignCenter, "Master state:")

        if n == 0:
            p.end()
            return

        # ── Bus line ──────────────────────────────────────────
        bus_y = y_bus + _BUS_Y_OFF // 2
        x_bus_start = _MARG + self._master_w
        x_bus_end = self._dev_x(n - 1) + _DEV_W
        p.setPen(QPen(_CLR_BUS_LINE, 2))
        p.drawLine(x_bus_start, bus_y, x_bus_end, bus_y)

        # ── "Address:" label ──────────────────────────────────
        p.setPen(_CLR_GREEN)
        p.setFont(font_sm)
        addr_lbl_x = _MARG
        p.drawText(QRect(addr_lbl_x, y_addr, self._master_w, _LABEL_H),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "")

        # ── "Axis:" label ─────────────────────────────────────
        has_axes = any(d.axis >= 0 for d in self.devices)
        if has_axes:
            p.setPen(_CLR_TEXT_DIM)
            p.setFont(font_sm)
            p.drawText(QRect(mx, y_axis, self._master_w, _LABEL_H),
                       Qt.AlignmentFlag.AlignCenter, "Axis:")

        # ── Device blocks ─────────────────────────────────────
        for i, dev in enumerate(self.devices):
            x = self._dev_x(i)

            # Drop line from bus to device
            p.setPen(QPen(_CLR_BUS_LINE, 2))
            p.drawLine(x + _DEV_W // 2, bus_y, x + _DEV_W // 2, y_dev)

            # Device rectangle
            border_clr = _CLR_GREEN if dev.online else _CLR_RED
            p.setPen(QPen(border_clr, 1))
            p.setBrush(QBrush(_CLR_CARD_BG))
            p.drawRoundedRect(x, y_dev, _DEV_W, _DEV_H, 3, 3)

            # Online indicator bar at top of device
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(border_clr))
            p.drawRect(x + 1, y_dev + 1, _DEV_W - 2, 3)

            # Drive type / vendor icon text inside device
            p.setPen(_CLR_TEXT)
            p.setFont(font_md)
            dt = _drive_type_label(dev.drive_type)
            if dt:
                p.drawText(QRect(x, y_dev + 8, _DEV_W, 20),
                           Qt.AlignmentFlag.AlignCenter, dt)

            # Vendor short name
            if dev.vendor_id:
                p.setPen(_CLR_ACCENT)
                p.setFont(font_sm)
                # Shorten long vendor names
                vn = dev.vendor_name
                if len(vn) > 8:
                    vn = vn.split()[0]  # first word only
                p.drawText(QRect(x, y_dev + 26, _DEV_W, 14),
                           Qt.AlignmentFlag.AlignCenter, vn)

            # Position number at bottom of device
            p.setPen(_CLR_TEXT_DIM)
            p.setFont(font_sm)
            p.drawText(QRect(x, y_dev + _DEV_H - 16, _DEV_W, 14),
                       Qt.AlignmentFlag.AlignCenter, f"#{dev.position}")

            # ── Address label above ───────────────────────────
            p.setPen(_CLR_TEXT)
            p.setFont(font_md)
            p.drawText(QRect(x, y_addr, _DEV_W, _LABEL_H),
                       Qt.AlignmentFlag.AlignCenter, str(dev.address))

            # ── Axis label below ──────────────────────────────
            if dev.axis >= 0:
                p.setPen(_CLR_BLUE)
                p.setFont(font_lg)
                p.drawText(QRect(x, y_axis, _DEV_W, _LABEL_H),
                           Qt.AlignmentFlag.AlignCenter, str(dev.axis))

        p.end()


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

    def __init__(self, connection: TUA.TrioConnection, parent=None, conn_lock=None):
        super().__init__(parent)
        self._connection = connection
        self._conn_lock = conn_lock
        self._network: Optional[EthercatNetwork] = None
        self._signals = _ScanSignals()
        self._signals.finished.connect(self._on_scan_finished)
        self._signals.error.connect(self._on_scan_error)

        self.setWindowTitle("EtherCAT Network Map")
        self.resize(750, 220)
        self.setMinimumSize(400, 160)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_scan = QPushButton("\u27f3  Scan Network")
        self._btn_scan.setFixedHeight(26)
        self._btn_scan.clicked.connect(self._start_scan)
        toolbar.addWidget(self._btn_scan)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888888;")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #FFA500; font-weight: bold; font-size: 8pt;")
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
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
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

        conn_lock = self._conn_lock

        def _worker():
            try:
                net = scan_network(self._connection, conn_lock=conn_lock)
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
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        net = self._network
        if not net:
            return

        present = [s for s in net.all_slaves if s.online or s.address != 0]
        online = len([s for s in present if s.online])
        active = len(net.active_slots)
        self._summary_label.setText(
            f"{active} active slot(s)  |  {len(present)} device(s)  |  {online} online"
        )

        if not present:
            empty = QLabel("No EtherCAT slaves detected.\n\n"
                           "Check that the EtherCAT network is started\n"
                           "and drives are powered on.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("color: #888888; font-size: 10pt; padding: 20px;")
            self._content_layout.addWidget(empty)
            return

        for ecat_slot in net.slots:
            if ecat_slot.num_slaves == 0 and not ecat_slot.is_operational:
                continue
            diagram = _SlotDiagram(ecat_slot)
            self._content_layout.addWidget(diagram)

        self._content_layout.addStretch()

    # ----- public API ------------------------------------------------------

    def get_network(self) -> Optional[EthercatNetwork]:
        """Return the last scan result."""
        return self._network
