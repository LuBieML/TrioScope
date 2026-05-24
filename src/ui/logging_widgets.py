import logging
from PySide6.QtWidgets import QDialog, QVBoxLayout, QPlainTextEdit, QPushButton, QHBoxLayout
from PySide6.QtCore import Slot, Qt, QMetaObject, Q_ARG

class _LogWindow(QDialog):
    """Scrollable log viewer — opened when the user clicks the log bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log")
        self.resize(860, 380)
        self.setStyleSheet(
            "QDialog { background-color: #1a1a1a; color: #d4d4d4; }"
            "QPlainTextEdit { background-color: #1a1a1a; color: #d4d4d4;"
            " font-family: Consolas, monospace; font-size: 8pt;"
            " border: none; }"
            "QPushButton { background-color: #4b4a4a; color: #d4d4d4;"
            " border: 1px solid #606060; border-radius: 3px; padding: 3px 10px; }"
            "QPushButton:hover { background-color: #5a5a5a; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(2000)
        layout.addWidget(self._view, 1)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(70)
        btn_clear.clicked.connect(self._view.clear)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        layout.addLayout(btn_row)

    @Slot(str)
    def append(self, html: str):
        """Append a pre-formatted HTML line (called from main thread via invokeMethod)."""
        self._view.appendHtml(html)
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())


class _LogBarHandler(logging.Handler):
    """Logging handler that updates the status bar label and feeds the log window."""

    _COLOURS = {
        logging.DEBUG:    "#888888",
        logging.INFO:     "#aaaaaa",
        logging.WARNING:  "#FFB74D",
        logging.ERROR:    "#f14c4c",
        logging.CRITICAL: "#ff0000",
    }

    def __init__(self, label, log_window: _LogWindow):
        super().__init__()
        self._label = label
        self._window = log_window
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                             datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            msg = self.format(record)
            colour = self._COLOURS.get(record.levelno, "#aaaaaa")
            bar_style = (
                f"background-color: #1a1a1a; color: {colour}; font-size: 8pt;"
                " border-top: 1px solid #444;"
            )
            # Short version for the bar (no timestamp)
            bar_msg = f"{record.levelname}  {record.getMessage()}"
            html_line = f'<span style="color:{colour};">{msg}</span>'

            QMetaObject.invokeMethod(
                self._label, "setText",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, bar_msg),
            )
            QMetaObject.invokeMethod(
                self._label, "setStyleSheet",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, bar_style),
            )
            QMetaObject.invokeMethod(
                self._window, "append",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, html_line),
            )
        except Exception:
            pass
