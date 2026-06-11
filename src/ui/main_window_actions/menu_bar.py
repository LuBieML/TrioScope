"""Menu bar construction plus the Help and About actions."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMessageBox

from version import __version__

try:
    from help_window import HelpWindow
except ImportError:
    HelpWindow = None


class MenuBarMixin:
    """File / View / Help menu bar and the help/about handlers."""

    def _create_menu_bar(self):
        """Create the top menu bar with File / View / Help menus."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #353536;
                color: #d4d4d4;
                border-bottom: 1px solid #4b4a4a;
                padding: 2px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 10px;
                border-radius: 3px;
            }
            QMenuBar::item:selected {
                background-color: #FFA500;
                color: #000000;
            }
            QMenu {
                background-color: #353536;
                color: #d4d4d4;
                border: 1px solid #4b4a4a;
                padding: 4px;
            }
            QMenu::item {
                padding: 5px 22px 5px 16px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #FFA500;
                color: #000000;
            }
            QMenu::separator {
                height: 1px;
                background-color: #4b4a4a;
                margin: 4px 6px;
            }
        """)

        # ── File menu ──────────────────────────────────────────────
        file_menu = menubar.addMenu("&File")

        act_export = QAction("&Export CSV...", self)
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.window.export_to_csv)
        file_menu.addAction(act_export)

        act_report = QAction("HTML &Report...", self)
        act_report.setShortcut(QKeySequence("Ctrl+R"))
        act_report.triggered.connect(self.window.export_html_report)
        file_menu.addAction(act_report)

        act_import = QAction("&Import CSV...", self)
        act_import.setShortcut(QKeySequence("Ctrl+O"))
        act_import.triggered.connect(self.window.import_from_csv)
        file_menu.addAction(act_import)

        file_menu.addSeparator()

        act_settings = QAction("&Settings...", self)
        act_settings.setShortcut(QKeySequence("Ctrl+,"))
        act_settings.triggered.connect(self.window.open_settings)
        file_menu.addAction(act_settings)

        file_menu.addSeparator()

        act_screenshot = QAction("Take &Screenshot", self)
        act_screenshot.triggered.connect(self.window.take_screenshot)
        file_menu.addAction(act_screenshot)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.window.close)
        file_menu.addAction(act_quit)

        # ── View menu ──────────────────────────────────────────────
        view_menu = menubar.addMenu("&View")

        act_tuner = QAction("&Servo Tuner", self)
        act_tuner.setShortcut(QKeySequence("Ctrl+T"))
        act_tuner.triggered.connect(self.window._toggle_tuner_panel)
        view_menu.addAction(act_tuner)

        act_measurements = QAction("&Measurements", self)
        act_measurements.triggered.connect(self.window._toggle_measurement_panel)
        view_menu.addAction(act_measurements)

        act_ecat = QAction("&EtherCAT Map", self)
        act_ecat.setShortcut(QKeySequence("Ctrl+M"))
        act_ecat.triggered.connect(self.window._open_ethercat_map)
        view_menu.addAction(act_ecat)

        view_menu.addSeparator()

        # Profiles submenu
        self._profiles_menu = view_menu.addMenu("📋 &Profiles")
        self._rebuild_profiles_menu()

        # ── Help menu ──────────────────────────────────────────────
        help_menu = menubar.addMenu("&Help")

        act_manual = QAction("&User Manual", self)
        act_manual.setShortcut(QKeySequence.HelpContents)  # F1
        act_manual.triggered.connect(lambda: self._show_help("index.md"))
        help_menu.addAction(act_manual)

        act_started = QAction("&Getting Started", self)
        act_started.triggered.connect(lambda: self._show_help("01_getting_started.md"))
        help_menu.addAction(act_started)

        act_capture = QAction("Capture &Modes", self)
        act_capture.triggered.connect(lambda: self._show_help("02_capture_modes.md"))
        help_menu.addAction(act_capture)

        act_traces = QAction("&Traces && Parameters", self)
        act_traces.triggered.connect(lambda: self._show_help("03_traces.md"))
        help_menu.addAction(act_traces)

        act_plotmodes = QAction("&Plot Modes", self)
        act_plotmodes.triggered.connect(lambda: self._show_help("04_plot_modes.md"))
        help_menu.addAction(act_plotmodes)

        act_nav = QAction("&Navigation && Cursors", self)
        act_nav.triggered.connect(lambda: self._show_help("05_navigation.md"))
        help_menu.addAction(act_nav)

        act_fft = QAction("&FFT Analysis", self)
        act_fft.triggered.connect(lambda: self._show_help("06_fft.md"))
        help_menu.addAction(act_fft)

        help_menu.addSeparator()

        act_shortcuts = QAction("&Keyboard && Mouse Reference", self)
        act_shortcuts.triggered.connect(lambda: self._show_help("11_shortcuts.md"))
        help_menu.addAction(act_shortcuts)

        act_trouble = QAction("Trou&bleshooting", self)
        act_trouble.triggered.connect(lambda: self._show_help("12_troubleshooting.md"))
        help_menu.addAction(act_trouble)

        help_menu.addSeparator()

        act_about = QAction("&About TrioScope", self)
        act_about.triggered.connect(self.window._show_about)
        help_menu.addAction(act_about)

    def _show_help(self, page: str = "index.md"):
        """Open the help window at the given markdown page."""
        if HelpWindow is None:
            QMessageBox.warning(
                self.window, "Help",
                "Help module not available. Reinstall the application or check that "
                "src/help_window.py and docs/help/ are present.")
            return

        if self._help_window is None:
            self._help_window = HelpWindow(self.window, start_page=page)
            self._help_window.setAttribute(Qt.WA_DeleteOnClose)
            self._help_window.destroyed.connect(lambda: setattr(self, "_help_window", None))
            self._help_window.show()
        else:
            self._help_window.show_page(page, push_history=True)
            self._help_window.raise_()
            self._help_window.activateWindow()

    def _show_about(self):
        """Show the About dialog."""
        QMessageBox.about(
            self.window, "About TrioScope",
            "<h2>TrioScope</h2>"
            "<p>An oscilloscope-style data capture and analysis tool for "
            "Trio Motion Controllers and Trio DX-series servo drives.</p>"
            f"<p><b>Version: {__version__}</b></p>"
            "<p>Real-time multi-trace plotting, FFT, XY/XYZ/XYZW path views, "
            "AI-powered tuning analysis, and EtherCAT diagnostics.</p>"
            "<p>Built with PySide6, pyqtgraph, and Trio_UnifiedApi.</p>"
            "<br><p><b>Legal & Licenses:</b><br>"
            "This application utilizes third-party open source software.<br>"
            "See <b>THIRD_PARTY_LICENSES.txt</b> in the installation directory for full <br>"
            "license texts (LGPLv3, MIT, BSD) and copyright attributions.</p>"
            "<p><a href='#'>Help → User Manual</a> for full documentation.</p>"
        )
