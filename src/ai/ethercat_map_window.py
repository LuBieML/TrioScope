"""
EtherCAT Network Map window — visual topology of discovered slaves.

Diagram-style layout inspired by Trio Motion Perfect:
  Address row  →  device strip with bus line  →  axis row

Devices are clickable; the details panel on the right shows the full
CoE identity (vendor, product, revision, serial, profile) and offers
an internet lookup for the selected device.
"""

import logging
import threading
import webbrowser
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget,
    QScrollArea, QSizePolicy, QGroupBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QRect
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush

import Trio_UnifiedApi as TUA

from . import ethercat_devices
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
_CLR_SELECTED     = QColor("#FFA500")

# Layout constants
_DEV_W      = 84   # device block width
_DEV_H      = 72   # device block height
_DEV_GAP    = 10   # gap between device blocks
_LABEL_H    = 16   # row height for address / axis labels
_BUS_Y_OFF  = 6    # bus line offset above device blocks
_MARG       = 8    # outer margin


def _state_colour(state) -> QColor:
    try:
        val = int(state)
    except (ValueError, TypeError):
        val = 0
    if val in (3, 8):
        return _CLR_GREEN
    if val == 4:
        return _CLR_ACCENT
    if val == 2:
        return QColor("#cccc00")
    return _CLR_RED


# ---------------------------------------------------------------------------
# Diagram widget — one per EtherCAT slot
# ---------------------------------------------------------------------------
class _SlotDiagram(QWidget):
    """Custom-painted diagram for one EtherCAT slot, resembling Motion Perfect."""

    deviceSelected = Signal(object)   # EthercatSlave or None

    def __init__(self, ecat_slot: EthercatSlot, parent=None):
        super().__init__(parent)
        self.ecat_slot = ecat_slot
        self._selected: Optional[int] = None

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
        self._dev_rects: list[QRect] = []
        self._tooltips: list[tuple[QRect, str]] = []
        for i, dev in enumerate(self.devices):
            x = self._dev_x(i)
            y = _MARG + _LABEL_H + _BUS_Y_OFF
            rect = QRect(x, y, _DEV_W, _DEV_H)
            self._dev_rects.append(rect)
            lines = [f"Position: #{dev.position}"]
            lines.append(f"Address: {dev.address}")
            if dev.axis >= 0:
                lines.append(f"Axis: {dev.axis}")
            if dev.product_name:
                lines.append(f"Device: {dev.product_name}")
            if dev.vendor_id:
                lines.append(f"Vendor: {dev.vendor_name}")
            lines.append(f"Online: {'Yes' if dev.online else 'No'}")
            if dev.drive_status:
                lines.append(f"Status: 0x{dev.drive_status:04X}")
            lines.append("Click for full details")
            self._tooltips.append((rect, "\n".join(lines)))

        self.setMouseTracking(True)

    def _dev_x(self, i: int) -> int:
        """X position of device block i."""
        return _MARG + self._master_w + _DEV_GAP + i * (_DEV_W + _DEV_GAP)

    # ----- selection -------------------------------------------------------

    def mousePressEvent(self, ev):
        pos = ev.position().toPoint()
        for i, rect in enumerate(self._dev_rects):
            if rect.contains(pos):
                self._selected = i
                self.update()
                self.deviceSelected.emit(self.devices[i])
                return
        # Click on empty space clears selection
        if self._selected is not None:
            self._selected = None
            self.update()
            self.deviceSelected.emit(None)
        super().mousePressEvent(ev)

    def clear_selection(self):
        if self._selected is not None:
            self._selected = None
            self.update()

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
            pen_w = 1
            if i == self._selected:
                border_clr = _CLR_SELECTED
                pen_w = 2
            p.setPen(QPen(border_clr, pen_w))
            p.setBrush(QBrush(_CLR_CARD_BG))
            p.drawRoundedRect(x, y_dev, _DEV_W, _DEV_H, 3, 3)

            # Online indicator bar at top of device
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_CLR_GREEN if dev.online else _CLR_RED))
            p.drawRect(x + 1, y_dev + 1, _DEV_W - 2, 3)

            # Product / device name inside device
            p.setPen(_CLR_TEXT)
            p.setFont(font_md)
            name = dev.product_name
            if name:
                p.drawText(QRect(x + 2, y_dev + 8, _DEV_W - 4, 18),
                           Qt.AlignmentFlag.AlignCenter, name)

            # Vendor short name
            if dev.vendor_id:
                p.setPen(_CLR_ACCENT)
                p.setFont(font_sm)
                # Shorten long vendor names
                vn = dev.vendor_name
                if len(vn) > 12:
                    vn = vn.split()[0]  # first word only
                p.drawText(QRect(x + 2, y_dev + 26, _DEV_W - 4, 14),
                           Qt.AlignmentFlag.AlignCenter, vn)

            # Revision (small, dimmed)
            if dev.revision_str:
                p.setPen(_CLR_TEXT_DIM)
                p.setFont(font_sm)
                p.drawText(QRect(x + 2, y_dev + 40, _DEV_W - 4, 14),
                           Qt.AlignmentFlag.AlignCenter, f"rev {dev.revision_str}")

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
# Details panel for the selected device
# ---------------------------------------------------------------------------
class _DetailsPanel(QGroupBox):
    """Shows the full CoE identity of the selected slave."""

    _FIELDS = [
        ("Device",      "device"),
        ("Vendor",      "vendor"),
        ("Product code", "product"),
        ("Revision",    "revision"),
        ("Serial no.",  "serial"),
        ("Profile",     "profile"),
        ("Position",    "position"),
        ("Address",     "address"),
        ("Axis",        "axis"),
        ("Drive status", "status"),
        ("Online",      "online"),
    ]

    def __init__(self, parent=None):
        super().__init__("Device details", parent)
        self.setFixedWidth(260)
        self._slave: Optional[EthercatSlave] = None

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(4)

        self._values: dict[str, QLabel] = {}
        for row, (title, key) in enumerate(self._FIELDS):
            lbl = QLabel(title + ":")
            lbl.setStyleSheet("color: #888888; font-size: 8pt;")
            val = QLabel("—")
            val.setStyleSheet("color: #d4d4d4; font-size: 8pt;")
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            val.setWordWrap(True)
            grid.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(val, row, 1)
            self._values[key] = val
        layout.addLayout(grid)
        layout.addStretch()

        self._btn_search = QPushButton("\U0001F310  Search device online")
        self._btn_search.setToolTip(
            "Open a web search for this device (vendor + product) in your browser"
        )
        self._btn_search.setEnabled(False)
        self._btn_search.clicked.connect(self._search_online)
        layout.addWidget(self._btn_search)

    def set_slave(self, slave: Optional[EthercatSlave]):
        self._slave = slave
        if slave is None:
            for val in self._values.values():
                val.setText("—")
            self._btn_search.setEnabled(False)
            return

        v = self._values
        v["device"].setText(slave.product_name or "Unknown")
        if slave.vendor_id:
            v["vendor"].setText(f"{slave.vendor_name}\n(0x{slave.vendor_id:08X})")
        else:
            v["vendor"].setText("—")
        v["product"].setText(f"0x{slave.product_code:08X}" if slave.product_code else "—")
        v["revision"].setText(slave.revision_str or "—")
        v["serial"].setText(str(slave.serial_number) if slave.serial_number else "—")
        v["profile"].setText(slave.profile_name or "—")
        v["position"].setText(f"#{slave.position} (slot {slave.slot})")
        v["address"].setText(str(slave.address))
        v["axis"].setText(str(slave.axis) if slave.axis >= 0 else "not mapped")
        v["status"].setText(f"0x{slave.drive_status:04X}" if slave.drive_status else "—")
        v["online"].setText("Yes" if slave.online else "No")
        self._btn_search.setEnabled(True)

    def _search_online(self):
        if self._slave is None:
            return
        url = ethercat_devices.web_search_url(
            self._slave.vendor_id, self._slave.product_code,
            drive_type=self._slave.drive_type,
        )
        webbrowser.open(url)


# ---------------------------------------------------------------------------
# Signal bridge for thread → GUI
# ---------------------------------------------------------------------------
class _ScanSignals(QObject):
    finished = Signal(object)   # EthercatNetwork
    error = Signal(str)
    etg_done = Signal(int)      # number of vendors fetched
    etg_error = Signal(str)


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
        self._signals.etg_done.connect(self._on_etg_done)
        self._signals.etg_error.connect(self._on_etg_error)

        self.setWindowTitle("EtherCAT Network Map")
        self.resize(950, 320)
        self.setMinimumSize(500, 200)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_scan = QPushButton("⟳  Scan Network")
        self._btn_scan.setFixedHeight(26)
        self._btn_scan.clicked.connect(self._start_scan)
        toolbar.addWidget(self._btn_scan)

        self._btn_etg = QPushButton("\U0001F310  Update Vendor Names")
        self._btn_etg.setFixedHeight(26)
        self._btn_etg.setToolTip(
            "Download the official EtherCAT vendor-ID registry from\n"
            "ethercat.org (requires internet). The list is cached locally."
        )
        self._btn_etg.clicked.connect(self._start_etg_update)
        toolbar.addWidget(self._btn_etg)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888888;")
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #FFA500; font-weight: bold; font-size: 8pt;")
        toolbar.addWidget(self._summary_label)
        root.addLayout(toolbar)

        # Content row: scrollable diagrams + details panel
        content_row = QHBoxLayout()
        content_row.setSpacing(6)

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
        content_row.addWidget(self._scroll, 1)

        self._details = _DetailsPanel()
        content_row.addWidget(self._details)
        root.addLayout(content_row, 1)

        self._diagrams: list[_SlotDiagram] = []

        # Kick off initial scan
        QTimer.singleShot(100, self._start_scan)

    # ----- scanning --------------------------------------------------------

    def _start_scan(self):
        if not self._connection:
            self._status_label.setText("No connection")
            return

        self._btn_scan.setEnabled(False)
        self._status_label.setText("Scanning (reading device identities)...")

        conn_lock = self._conn_lock

        # Stop connection watchdog during network scan to prevent timeout/disconnect
        parent = self.parent()
        if parent and hasattr(parent, "_stop_watchdog"):
            try:
                parent._stop_watchdog()
            except Exception as exc:
                logger.debug("Failed to stop watchdog: %s", exc)

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

        # Restart connection watchdog
        parent = self.parent()
        if parent and hasattr(parent, "_start_watchdog"):
            try:
                parent._start_watchdog()
            except Exception as exc:
                logger.debug("Failed to start watchdog: %s", exc)

    def _on_scan_error(self, msg: str):
        self._btn_scan.setEnabled(True)
        self._status_label.setText(f"Scan error: {msg}")

        # Restart connection watchdog
        parent = self.parent()
        if parent and hasattr(parent, "_start_watchdog"):
            try:
                parent._start_watchdog()
            except Exception as exc:
                logger.debug("Failed to start watchdog: %s", exc)

    def closeEvent(self, event):
        # Ensure watchdog is running when the window is closed
        parent = self.parent()
        if parent and hasattr(parent, "_start_watchdog"):
            try:
                parent._start_watchdog()
            except Exception as exc:
                logger.debug("Failed to start watchdog on close: %s", exc)
        super().closeEvent(event)

    # ----- ETG vendor registry update --------------------------------------

    def _start_etg_update(self):
        """Download the official vendor-ID registry from ethercat.org."""
        self._btn_etg.setEnabled(False)
        self._status_label.setText("Downloading ETG vendor registry...")

        def _worker():
            try:
                vendors = ethercat_devices.fetch_etg_vendors()
                self._signals.etg_done.emit(len(vendors))
            except Exception as exc:
                logger.warning("ETG vendor fetch failed: %s", exc)
                self._signals.etg_error.emit(str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_etg_done(self, count: int):
        self._btn_etg.setEnabled(True)
        self._status_label.setText(f"Vendor registry updated ({count} vendors)")
        # Re-render so unknown vendor IDs pick up their new names
        if self._network:
            self._rebuild_map()

    def _on_etg_error(self, msg: str):
        self._btn_etg.setEnabled(True)
        self._status_label.setText(f"Vendor registry update failed: {msg}")

    # ----- map rendering ---------------------------------------------------

    def _rebuild_map(self):
        """Rebuild the visual map from the current network scan."""
        self._diagrams.clear()
        self._details.set_slave(None)
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
            diagram.deviceSelected.connect(self._on_device_selected)
            self._diagrams.append(diagram)
            self._content_layout.addWidget(diagram)

        self._content_layout.addStretch()

    def _on_device_selected(self, slave: Optional[EthercatSlave]):
        # Only one device selected across all slot diagrams
        sender = self.sender()
        for diagram in self._diagrams:
            if diagram is not sender:
                diagram.clear_selection()
        self._details.set_slave(slave)

    # ----- public API ------------------------------------------------------

    def get_network(self) -> Optional[EthercatNetwork]:
        """Return the last scan result."""
        return self._network
