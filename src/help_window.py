"""
Help window — markdown-rendered user manual viewer.

Loads the documentation from docs/help/*.md and presents it in a two-pane
dialog: a table of contents on the left, the rendered page on the right.
Internal links between markdown files are followed within the viewer.

Works in both development checkouts and PyInstaller frozen builds; the
help directory is resolved relative to either sys._MEIPASS (frozen) or
the project root (source).
"""

import sys
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QSize
from PySide6.QtGui import (
    QFont, QTextDocument, QPalette, QColor,
    QTextCursor, QTextCharFormat, QBrush,
)
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QTextBrowser, QSplitter, QWidget, QPushButton, QLabel,
)

LINK_COLOR = QColor("#FFC857")  # bright amber, high contrast on dark bg

logger = logging.getLogger(__name__)


# Ordered table of contents — (filename, display title)
HELP_PAGES = [
    ("index.md",                 "Overview"),
    ("01_getting_started.md",    "1. Getting Started"),
    ("02_capture_modes.md",      "2. Capture Modes"),
    ("03_traces.md",             "3. Traces & Parameters"),
    ("04_plot_modes.md",         "4. Plot Modes"),
    ("05_navigation.md",         "5. Navigation & Cursors"),
    ("06_fft.md",                "6. FFT Analysis"),
    ("07_export_import.md",      "7. Export / Import"),
    ("08_ai_analysis.md",        "8. AI Analysis Panel"),
    ("09_ethercat_map.md",       "9. EtherCAT Map"),
    ("10_settings.md",           "10. Settings"),
    ("11_shortcuts.md",          "11. Keyboard & Mouse"),
    ("12_troubleshooting.md",    "12. Troubleshooting"),
    ("about.md",                 "About TrioScope"),
]


HELP_STYLESHEET = """
QDialog {
    background-color: #2e2e2e;
    color: #d4d4d4;
}
QListWidget {
    background-color: #353536;
    color: #d4d4d4;
    border: 1px solid #4b4a4a;
    border-radius: 3px;
    font-size: 9pt;
    outline: 0;
    padding: 4px;
}
QListWidget::item {
    padding: 6px 8px;
    border-radius: 3px;
}
QListWidget::item:selected {
    background-color: #FFA500;
    color: #000000;
}
QListWidget::item:hover {
    background-color: #4b4a4a;
}
QTextBrowser {
    background-color: #1e1e1e;
    color: #d4d4d4;
    border: 1px solid #4b4a4a;
    border-radius: 3px;
    padding: 12px 18px;
    font-family: 'Segoe UI';
    font-size: 10pt;
    selection-background-color: #FFA500;
    selection-color: #000000;
}
QPushButton {
    background-color: #4b4a4a;
    color: #d4d4d4;
    border: 1px solid #606060;
    border-radius: 3px;
    padding: 5px 14px;
    min-width: 70px;
}
QPushButton:hover { background-color: #5a5a5a; }
QPushButton:pressed { background-color: #666666; }
QPushButton:disabled { color: #666666; background-color: #3a3a3a; }
QLabel#help_title {
    color: #FFA500;
    font-size: 11pt;
    font-weight: bold;
    padding: 4px 6px;
}
QSplitter::handle {
    background-color: #353536;
}
"""

# CSS injected into rendered markdown so headings, code, tables match the
# dark theme. QTextDocument supports a subset of CSS via setDefaultStyleSheet.
DOCUMENT_CSS = """
body { color: #d4d4d4; font-family: 'Segoe UI'; font-size: 10pt; }
h1 { color: #FFA500; font-size: 18pt; margin-top: 0; padding-bottom: 6px; }
h2 { color: #03DAC6; font-size: 14pt; margin-top: 18px; }
h3 { color: #64B5F6; font-size: 12pt; margin-top: 14px; }
h4 { color: #d4d4d4; font-size: 11pt; }
p  { color: #d4d4d4; line-height: 140%; }
a  { color: #FFC857; text-decoration: underline; font-weight: bold; }
a:hover { color: #FFE082; text-decoration: underline; }
a:visited { color: #FFC857; }
code { background-color: #2e2e2e; color: #03DAC6; font-family: 'Consolas', monospace;
       padding: 1px 4px; border-radius: 2px; }
pre  { background-color: #0A0A0A; color: #d4d4d4; padding: 8px; border: 1px solid #4b4a4a;
       border-radius: 3px; font-family: 'Consolas', monospace; }
table { border-collapse: collapse; margin: 8px 0; }
th { background-color: #353536; color: #FFA500; border: 1px solid #4b4a4a;
     padding: 4px 10px; text-align: left; }
td { border: 1px solid #4b4a4a; padding: 4px 10px; color: #d4d4d4; }
blockquote { border-left: 3px solid #FFA500; margin-left: 0; padding-left: 12px;
             color: #BBBBBB; background-color: #2a2a2a; }
hr { border: 0; border-top: 1px solid #4b4a4a; }
ul, ol { color: #d4d4d4; }
li { margin: 3px 0; }
"""


def _resolve_help_dir() -> Path:
    """Locate the docs/help directory in both source and frozen builds."""
    if getattr(sys, "frozen", False):
        # PyInstaller --onedir / --onefile: assets unpacked to _MEIPASS
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates = [base / "docs" / "help", Path(sys.executable).parent / "docs" / "help"]
    else:
        # Source checkout: src/help_window.py → project root is two levels up
        project_root = Path(__file__).resolve().parent.parent
        candidates = [project_root / "docs" / "help"]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Fall back to the first candidate even if missing — we render an error page
    return candidates[0]


class HelpWindow(QDialog):
    """Dark-themed two-pane markdown manual viewer."""

    def __init__(self, parent=None, start_page: str = "index.md"):
        super().__init__(parent)
        self.setWindowTitle("TrioScope — Help")
        self.setMinimumSize(900, 650)
        self.resize(1100, 750)
        self.setStyleSheet(HELP_STYLESHEET)

        self._help_dir = _resolve_help_dir()
        self._history: list[str] = []
        self._history_pos: int = -1
        self._navigating_history: bool = False

        self._build_ui()
        self._populate_toc()
        self.show_page(start_page, push_history=True)

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Top toolbar: back / forward / home + page title
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_back = QPushButton("\u25c0  Back")
        self.btn_back.clicked.connect(self._go_back)
        self.btn_back.setEnabled(False)
        toolbar.addWidget(self.btn_back)

        self.btn_forward = QPushButton("Forward  \u25b6")
        self.btn_forward.clicked.connect(self._go_forward)
        self.btn_forward.setEnabled(False)
        toolbar.addWidget(self.btn_forward)

        self.btn_home = QPushButton("\u2302 Home")
        self.btn_home.clicked.connect(lambda: self.show_page("index.md", push_history=True))
        toolbar.addWidget(self.btn_home)

        toolbar.addSpacing(10)
        self.title_label = QLabel("")
        self.title_label.setObjectName("help_title")
        toolbar.addWidget(self.title_label, 1)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        toolbar.addWidget(btn_close)

        root.addLayout(toolbar)

        # Main splitter: TOC on left, content on right
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        self.toc_list = QListWidget()
        self.toc_list.setMinimumWidth(200)
        self.toc_list.setMaximumWidth(280)
        self.toc_list.itemClicked.connect(self._on_toc_clicked)
        splitter.addWidget(self.toc_list)

        self.text_browser = QTextBrowser()
        self.text_browser.setOpenLinks(False)            # we handle links ourselves
        self.text_browser.setOpenExternalLinks(False)
        self.text_browser.anchorClicked.connect(self._on_anchor_clicked)
        # Resolve relative resources from the help directory
        self.text_browser.setSearchPaths([str(self._help_dir)])
        # Apply dark CSS to the document
        self.text_browser.document().setDefaultStyleSheet(DOCUMENT_CSS)
        # QTextBrowser draws hyperlinks using the widget palette as a
        # fallback — set Link/LinkVisited too so any anchors that don't
        # carry an explicit char-format colour pick up the bright amber.
        link_palette = self.text_browser.palette()
        link_palette.setColor(QPalette.Link, LINK_COLOR)
        link_palette.setColor(QPalette.LinkVisited, LINK_COLOR)
        self.text_browser.setPalette(link_palette)
        splitter.addWidget(self.text_browser)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 880])
        root.addWidget(splitter, 1)

    def _populate_toc(self):
        for filename, title in HELP_PAGES:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, filename)
            self.toc_list.addItem(item)

    # ──────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────

    def show_page(self, filename: str, push_history: bool = True):
        """Render a markdown page by filename (relative to docs/help)."""
        if not filename:
            return
        path = self._help_dir / filename

        if not path.exists():
            self._render_error(
                f"Help page not found: {filename}\n\n"
                f"Looked in: {self._help_dir}"
            )
            return

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            self._render_error(f"Failed to read {filename}: {e}")
            return

        # QTextDocument.setMarkdown supports CommonMark + GFM tables
        self.text_browser.document().setDefaultStyleSheet(DOCUMENT_CSS)
        self.text_browser.setMarkdown(text)
        # setMarkdown bakes a default link colour into each anchor's char
        # format, overriding both CSS and palette. Walk the document and
        # rewrite every anchor fragment so links are clearly visible.
        self._recolor_links()
        self.text_browser.verticalScrollBar().setValue(0)

        # Update title and TOC selection
        title = self._title_for(filename)
        self.title_label.setText(title)
        self._select_toc(filename)

        # History
        if push_history and not self._navigating_history:
            # Truncate forward history beyond current position
            self._history = self._history[: self._history_pos + 1]
            self._history.append(filename)
            self._history_pos = len(self._history) - 1
        self._update_nav_buttons()

    def _render_error(self, message: str):
        self.text_browser.document().setDefaultStyleSheet(DOCUMENT_CSS)
        self.text_browser.setMarkdown(
            f"# Error\n\n```\n{message}\n```\n\n"
            "Please reinstall the application or report this issue."
        )
        self._recolor_links()
        self.title_label.setText("Error")

    def _recolor_links(self):
        """Force every anchor in the document to use LINK_COLOR.

        QTextDocument.setMarkdown applies a hard-coded blue foreground to
        each anchor's QTextCharFormat — neither setDefaultStyleSheet (CSS)
        nor QPalette.Link can override that. The only reliable fix is to
        iterate every text fragment, find anchors, and merge a new char
        format with our chosen colour, weight, and underline.
        """
        doc = self.text_browser.document()
        cursor = QTextCursor(doc)
        cursor.beginEditBlock()
        try:
            link_format = QTextCharFormat()
            link_format.setForeground(QBrush(LINK_COLOR))
            link_format.setFontUnderline(True)
            link_format.setFontWeight(QFont.Bold)

            block = doc.begin()
            while block.isValid():
                it = block.begin()
                while not it.atEnd():
                    fragment = it.fragment()
                    if fragment.isValid() and fragment.charFormat().isAnchor():
                        start = fragment.position()
                        length = fragment.length()
                        cursor.setPosition(start)
                        cursor.setPosition(start + length, QTextCursor.KeepAnchor)
                        cursor.mergeCharFormat(link_format)
                    it += 1
                block = block.next()
        finally:
            cursor.endEditBlock()

    def _title_for(self, filename: str) -> str:
        for fname, title in HELP_PAGES:
            if fname == filename:
                return title
        return filename

    def _select_toc(self, filename: str):
        for i in range(self.toc_list.count()):
            item = self.toc_list.item(i)
            if item.data(Qt.UserRole) == filename:
                self.toc_list.blockSignals(True)
                self.toc_list.setCurrentRow(i)
                self.toc_list.blockSignals(False)
                return

    def _on_toc_clicked(self, item: QListWidgetItem):
        filename = item.data(Qt.UserRole)
        self.show_page(filename, push_history=True)

    def _on_anchor_clicked(self, url: QUrl):
        """Handle clicks on links inside the rendered markdown."""
        target = url.toString()

        # External http(s) links — open in the system browser
        if target.startswith(("http://", "https://", "mailto:")):
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(url)
            return

        # In-page anchor (#section)
        if target.startswith("#"):
            self.text_browser.scrollToAnchor(target.lstrip("#"))
            return

        # Relative markdown link — strip any anchor fragment and load
        path_part = url.path() or target
        if "#" in path_part:
            path_part = path_part.split("#", 1)[0]

        # Some links arrive as "01_getting_started.md", others as "./01_..."
        path_part = path_part.lstrip("./")

        if path_part.endswith(".md"):
            self.show_page(path_part, push_history=True)
            return

        # Unknown link type — ignore silently
        logger.debug("Unhandled help link: %s", target)

    def _go_back(self):
        if self._history_pos <= 0:
            return
        self._history_pos -= 1
        self._navigating_history = True
        self.show_page(self._history[self._history_pos], push_history=False)
        self._navigating_history = False
        self._update_nav_buttons()

    def _go_forward(self):
        if self._history_pos >= len(self._history) - 1:
            return
        self._history_pos += 1
        self._navigating_history = True
        self.show_page(self._history[self._history_pos], push_history=False)
        self._navigating_history = False
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        self.btn_back.setEnabled(self._history_pos > 0)
        self.btn_forward.setEnabled(self._history_pos < len(self._history) - 1)
