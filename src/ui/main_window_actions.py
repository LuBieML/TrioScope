import logging
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout,
)
import pyqtgraph as pg

from version import __version__
try:
    from ai.tuner_panel import TunerPanel
except ImportError:
    TunerPanel = None
try:
    from ai.ethercat_map_window import EthercatMapWindow
except ImportError:
    EthercatMapWindow = None
try:
    from help_window import HelpWindow
except ImportError:
    HelpWindow = None

from models.app_settings import AppSettings
from models.trace_config import TraceConfig
from reports.html_report import write_html_report
from scope.drive_scope_engine import TRIGGER_MODES
from storage.csv_io import CSVStorage
from storage.profiles import ProfileStore
from storage.settings_store import SettingsStore
from ui.measurement_panel import MeasurementPanel
from ui.profile_dialog import _ProfileManagerDialog
from ui.theme import DARK_STYLESHEET, TRACE_COLORS
from ui.window_controller import WindowBackedController

logger = logging.getLogger(__name__)


class MainWindowActions(WindowBackedController):
    def take_screenshot(self):
        """Take a screenshot of the main application window and save as PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Save Screenshot", f"scope_screenshot_{datetime.now():%Y%m%d_%H%M%S}.png",
            "PNG Files (*.png)"
        )
        if not path:
            return

        try:
            # useOpenGL=True means self.grab() will leave plots blank.
            # grabWindow takes an OS-level grab of the widget's bounds.
            pixmap = self.screen().grabWindow(self.winId())
            pixmap.save(path, "PNG")
            self.status_label.setText(f"Screenshot saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self.window, "Screenshot Error", str(e))

    def export_to_csv(self):
        if self.accumulated_data is None:
            QMessageBox.warning(self.window, "No Data", "No data to export")
            return

        # Warn if any currently enabled traces have no captured data
        captured = set(self.accumulated_data['params'].keys())
        missing = [
            t.get_display_name()
            for t in self.get_enabled_traces()
            if t.get_display_name() not in captured
        ]
        if missing:
            QMessageBox.warning(
                self.window, "Missing Channels",
                "The following enabled traces have no captured data and will not be exported:\n\n"
                + "\n".join(f"  • {m}" for m in missing)
                + "\n\nRe-run the capture with all desired traces enabled."
            )

        path, _ = QFileDialog.getSaveFileName(
            self.window, "Export CSV", f"scope_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            data = self.accumulated_data
            CSVStorage.export_data(path, data['time'], data['params'])
            self.status_label.setText(f"Exported {len(data['params'])} channel(s) to {path}")
        except Exception as e:
            QMessageBox.critical(self.window, "Export Error", str(e))

    def export_html_report(self):
        """Create a self-contained HTML commissioning report."""
        if self.accumulated_data is None:
            QMessageBox.warning(self.window, "No Data", "No capture data to report")
            return

        request = self._show_html_report_dialog()
        if request is None:
            return
        path, notes = request

        try:
            # Pull the latest buffered samples into accumulated_data before export.
            self._on_update_timer()

            data = self.accumulated_data
            if data is None or len(data.get('time', [])) == 0:
                QMessageBox.warning(self.window, "No Data", "No capture data to report")
                return

            trace_order, trace_colors, trace_fft_flags = self._report_trace_context()
            report_path = write_html_report(
                path,
                time_arr=data['time'],
                params=data['params'],
                trace_order=trace_order,
                trace_colors=trace_colors,
                trace_fft_flags=trace_fft_flags,
                controller_metadata=self._report_controller_metadata(),
                drive_metadata=self._report_drive_metadata(),
                drive_profiles=self._report_drive_profiles(),
                user_notes=notes,
                segment_breaks=data.get('segment_breaks', []),
            )
            self.status_label.setText(f"HTML report saved to {report_path}")
            QMessageBox.information(
                self.window,
                "HTML Report",
                f"Report created:\n{report_path}",
            )
        except Exception as e:
            logger.exception("HTML report export failed")
            QMessageBox.critical(self.window, "Report Error", str(e))

    def _show_html_report_dialog(self):
        """Ask for report path and optional notes."""
        dlg = QDialog(self.window)
        dlg.setWindowTitle("HTML Report")
        dlg.setMinimumWidth(560)
        dlg.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        path_label = QLabel("Output file:")
        path_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(path_label)

        path_row = QHBoxLayout()
        default_dir = Path.cwd() / "reports"
        default_path = default_dir / f"trioscope_report_{datetime.now():%Y%m%d_%H%M%S}.html"
        path_edit = QLineEdit(str(default_path))
        path_row.addWidget(path_edit, 1)
        btn_browse = QPushButton("Browse...")
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        notes_label = QLabel("User notes:")
        notes_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(notes_label)

        notes_edit = QPlainTextEdit()
        notes_edit.setPlaceholderText(
            "Customer, machine serial, axis, commissioning result, or support context")
        notes_edit.setFixedHeight(140)
        layout.addWidget(notes_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_create = QPushButton("Create Report")
        btn_create.setObjectName("accent")
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_create)
        layout.addLayout(btn_row)

        result = {}

        def browse():
            selected, _ = QFileDialog.getSaveFileName(
                dlg,
                "Save HTML Report",
                path_edit.text(),
                "HTML Files (*.html);;All Files (*)",
            )
            if selected:
                path_edit.setText(selected)

        def create():
            selected = path_edit.text().strip()
            if not selected:
                QMessageBox.warning(dlg, "HTML Report", "Choose an output file.")
                return
            result["path"] = selected
            result["notes"] = notes_edit.toPlainText()
            dlg.accept()

        btn_browse.clicked.connect(browse)
        btn_cancel.clicked.connect(dlg.reject)
        btn_create.clicked.connect(create)

        if dlg.exec() != QDialog.Accepted:
            return None
        return result["path"], result["notes"]

    def _report_trace_context(self):
        """Return ordered trace names, colors, and FFT flags for report export."""
        if self.accumulated_data is None:
            return [], {}, {}

        params = self.accumulated_data.get('params', {})
        ordered = []
        colors = {}
        fft_flags = {}

        for trace in self.get_enabled_traces():
            name = trace.get_display_name()
            if name in params and name not in ordered:
                ordered.append(name)
                colors[name] = trace.get_color()
                fft_flags[name] = trace.is_fft()

        for idx, name in enumerate(params.keys()):
            if name in ordered:
                continue
            ordered.append(name)
            colors[name] = TRACE_COLORS[idx % len(TRACE_COLORS)]
            fft_flags[name] = False

        return ordered, colors, fft_flags

    def _report_controller_metadata(self):
        """Build controller and capture metadata for report export."""
        metadata = {
            "Application": f"TrioScope v{__version__}",
            "Capture Source": "Drive Scope" if self.capture_source == 'drive' else "Controller SCOPE",
            "Controller IP": self.ip_edit.text().strip(),
            "Connection": "Connected" if self.trio_connected else "Disconnected/imported",
            "Plot Mode": self.plot_mode,
            "Configured Sample Period": self.period_edit.text().strip(),
            "Configured Duration": self.duration_edit.text().strip() + " s",
            "Capture Mode": "Single" if self.radio_single.isChecked() else "Continuous",
            "External Trigger": "Yes" if self.external_trigger_chk.isChecked() else "No",
            "Table Start Mode": "End of TABLE" if self.use_end_of_table else "Manual",
            "Manual Table Start": self.table_start_edit.text().strip(),
            "Window Duration": f"{self.window_duration:g} s",
            "Lock X Axis": "Yes" if self.lock_x_axis else "No",
        }
        if self.scope_engine is not None:
            servo_period = getattr(self.scope_engine, "servo_period_sec", None)
            if servo_period:
                metadata["Servo Period"] = f"{servo_period * 1000:.4f} ms"
            for attr, label in [
                ("period_cycles", "SCOPE Period Cycles"),
                ("tsize", "SCOPE Table Size"),
                ("table_start", "SCOPE Table Start"),
                ("table_end", "SCOPE Table End"),
                ("num_params", "SCOPE Parameter Count"),
            ]:
                value = getattr(self.scope_engine, attr, None)
                if value is not None:
                    metadata[label] = value
        return metadata

    def _report_drive_metadata(self):
        """Build drive-scope and drive-profile metadata for report export."""
        profiles = self._report_drive_profiles()
        metadata = {
            "Configured Drive Profiles": len(profiles),
        }

        if self.capture_source == 'drive':
            metadata.update({
                "Drive Scope Axis": self.drv_axis_spin.value(),
                "Drive Scope Trigger": self.drv_trigger_combo.currentText(),
                "Drive Scope Duration Setting": self.drv_sample_edit.text().strip() + " s",
                "Trigger Value 1": self.drv_trig_val1_edit.text().strip(),
                "Trigger Value 2": self.drv_trig_val2_edit.text().strip(),
            })
            try:
                metadata["Drive Scope Sample Time Units"] = self._get_drive_sample_time_units()
            except Exception:
                pass

        if self.drive_scope_engine is not None:
            for attr, label in [
                ("drive_model", "Drive Model"),
                ("axis", "Drive Axis"),
                ("active_channels", "Drive Scope Channels"),
                ("sample_time", "Drive Sample Time Units"),
                ("sample_period_us", "Drive Sample Period us"),
                ("capture_duration_sec", "Drive Capture Duration s"),
                ("trigger_value1", "Drive Trigger Value 1"),
                ("trigger_value2", "Drive Trigger Value 2"),
                ("display_names", "Drive Scope Signals"),
            ]:
                value = getattr(self.drive_scope_engine, attr, None)
                if value is not None:
                    metadata[label] = value
            trigger_mode = getattr(self.drive_scope_engine, "trigger_mode", None)
            if trigger_mode is not None:
                metadata["Drive Trigger Mode"] = TRIGGER_MODES.get(trigger_mode, trigger_mode)

        return metadata

    def _report_drive_profiles(self):
        """Return the newest available drive profile dicts."""
        if self._tuner_panel is not None:
            try:
                return self._tuner_panel.get_all_profiles()
            except Exception:
                logger.exception("Could not read drive profiles from tuner panel")
        return SettingsStore().load().drive_profiles

    def import_from_csv(self):
        """Import data from a previously exported CSV file.
        Reconstructs traces from the column headers and loads all data."""
        if self.is_running:
            QMessageBox.warning(self.window, "Running", "Stop capture before importing")
            return

        path, _ = QFileDialog.getOpenFileName(
            self.window, "Import CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            time_arr, params, traces_data = CSVStorage.import_data(path)

            # --- Reconfigure traces to match imported columns ---
            # Remove all existing traces
            for t in list(self.traces):
                t.setParent(None)
                t.deleteLater()
            self.traces.clear()

            for param, axis in traces_data:
                self.add_trace()
                trace = self.traces[-1]
                trace.chk_enable.setChecked(True)
                trace.param_combo.setCurrentText((param or "MPOS").strip() or "MPOS")
                trace.axis_spin.setValue(axis)

            # --- Load data ---
            self.accumulated_data = {
                'time': time_arr,
                'num_samples': len(time_arr),
                'params': params,
                'segment_breaks': [],
            }
            self.total_samples = len(time_arr)
            self.sample_counter_label.setText(f"Samples: {self.total_samples}")

            # Also populate chunk buffers so further captures can append
            with self._data_lock:
                n_import = len(time_arr)
                if n_import > self._buffer_capacity:
                    self._buffer_capacity = max(100_000, n_import * 2)
                    self._time_buffer = np.empty(self._buffer_capacity, dtype=np.float64)
                    self._param_buffers = {}
                self._time_buffer[:n_import] = time_arr
                for k, v in params.items():
                    buf = np.empty(self._buffer_capacity, dtype=v.dtype)
                    buf[:n_import] = v
                    self._param_buffers[k] = buf
                self._buffer_len = n_import
                self._segment_breaks = []
            self._last_consumed_state = None
            self._virtual_buffers = {}

            # Ensure we're not in auto-scroll/running state
            self.auto_scroll = False
            self._update_auto_scroll_button()

            self._recreate_subplots()

            # Set X range to full data extent before rendering
            t_min, t_max = float(time_arr[0]), float(time_arr[-1])
            padding = (t_max - t_min) * 0.02
            for pi in self.plot_items.values():
                pi.setXRange(t_min - padding, t_max + padding, padding=0)

            self._render_plots()
            self._sync_measurement_panel(force=True)

            self.status_label.setText(
                f"Imported {len(time_arr)} samples, {len(params)} params from {Path(path).name}")

        except Exception as e:
            QMessageBox.critical(self.window, "Import Error", str(e))

    def _get_profile_names(self):
        """Return a list of all saved profile names."""
        return ProfileStore().get_profile_names()

    def _save_profile(self, name):
        """Save the current trace configuration as a named profile."""
        enabled_traces = [
            TraceConfig(
                param=t.param_combo.currentText().strip() or "MPOS",
                axis=t.axis_spin.value(),
                enabled=t.chk_enable.isChecked(),
                fft=t.is_fft()
            )
            for t in self.traces if t.parent() is not None
        ]
        ProfileStore().save_profile(name, enabled_traces)
        self._rebuild_profiles_menu()
        logger.info(f"Profile '{name}' saved with {len(enabled_traces)} trace(s)")

    def _load_profile(self, name):
        """Load a named profile, replacing all current traces."""
        traces_config = ProfileStore().load_profile(name)
        if not traces_config:
            logger.warning(f"Profile '{name}' is empty or not found")
            return

        # Remove all existing traces
        for t in list(self.traces):
            t.setParent(None)
            t.deleteLater()
        self.traces.clear()

        # Recreate traces from profile
        for t_cfg in traces_config:
            self.add_trace()
            t = self.traces[-1]
            t.param_combo.setCurrentText((t_cfg.param or "MPOS").strip() or "MPOS")
            t.axis_spin.setValue(t_cfg.axis)
            t.chk_enable.setChecked(t_cfg.enabled)
            t.set_fft(t_cfg.fft)

        self.on_trace_changed()
        logger.info(f"Profile '{name}' loaded with {len(traces_config)} trace(s)")

    def _delete_profile(self, name):
        """Delete a saved profile."""
        ProfileStore().delete_profile(name)
        self._rebuild_profiles_menu()
        logger.info(f"Profile '{name}' deleted")

    def _rename_profile(self, old_name, new_name):
        """Rename a saved profile."""
        if old_name == new_name:
            return
        ProfileStore().rename_profile(old_name, new_name)
        self._rebuild_profiles_menu()
        logger.info(f"Profile renamed: '{old_name}' → '{new_name}'")

    def _rebuild_profiles_menu(self):
        """Refresh the View → Profiles submenu with current profile names."""
        if not hasattr(self, '_profiles_menu') or self._profiles_menu is None:
            return
        self._profiles_menu.clear()
        names = self._get_profile_names()

        if names:
            for name in names:
                act = self._profiles_menu.addAction(name)
                act.triggered.connect(lambda checked, n=name: self._load_profile(n))
            self._profiles_menu.addSeparator()

        act_manage = self._profiles_menu.addAction("\u2699 Manage Profiles…")
        act_manage.triggered.connect(self.window._show_manage_profiles_dialog)

    def _show_save_profile_dialog(self):
        """Show a dialog to save the current traces as a named profile."""
        if not self.traces:
            QMessageBox.information(self.window, "Save Profile",
                                   "Add at least one trace before saving a profile.")
            return

        existing = self._get_profile_names()

        dlg = QDialog(self.window)
        dlg.setWindowTitle("Save Profile")
        dlg.setMinimumWidth(320)
        dlg.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        lbl = QLabel("Profile name:")
        lbl.setStyleSheet("font-weight: bold; font-size: 10pt;")
        layout.addWidget(lbl)

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g. Position Tuning")
        name_edit.setStyleSheet(
            "padding: 6px; font-size: 10pt; background-color: #4b4a4a;"
            " color: #d4d4d4; border: 1px solid #606060; border-radius: 3px;")
        layout.addWidget(name_edit)

        if existing:
            hint = QLabel(f"Existing profiles: {', '.join(existing)}")
            hint.setStyleSheet("color: #888; font-size: 8pt;")
            hint.setWordWrap(True)
            layout.addWidget(hint)

        # Preview of what will be saved
        preview_lbl = QLabel("Traces to save:")
        preview_lbl.setStyleSheet("color: #aaa; font-size: 9pt; margin-top: 6px;")
        layout.addWidget(preview_lbl)
        for t in self.traces:
            if t.parent() is not None:
                status = "\u2713" if t.chk_enable.isChecked() else "\u2717"
                fft_tag = " [FFT]" if t.is_fft() else ""
                trace_text = f"  {status}  {t.get_display_name()}{fft_tag}"
                trace_lbl = QLabel(trace_text)
                trace_lbl.setStyleSheet(f"color: {t.color}; font-size: 9pt; font-family: Consolas;")
                layout.addWidget(trace_lbl)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_save = QPushButton("Save")
        btn_save.setObjectName("accent")
        btn_save.setFixedWidth(90)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedWidth(90)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)

        def do_save():
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "Save Profile", "Please enter a profile name.")
                return
            if name in existing:
                reply = QMessageBox.question(
                    dlg, "Overwrite Profile",
                    f"Profile '{name}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return
            self._save_profile(name)
            dlg.accept()

        btn_save.clicked.connect(do_save)
        btn_cancel.clicked.connect(dlg.reject)
        name_edit.returnPressed.connect(do_save)

        dlg.exec()

    def _show_manage_profiles_dialog(self):
        """Show the profile manager dialog for load/rename/delete."""
        dlg = _ProfileManagerDialog(self.window)
        dlg.exec()

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

        act_motion = QAction("Axis &Motion...", self)
        act_motion.setShortcut(QKeySequence("Ctrl+Shift+M"))
        act_motion.triggered.connect(self.window._open_motion_window)
        view_menu.addAction(act_motion)

        view_menu.addSeparator()

        # Profiles submenu
        self._profiles_menu = view_menu.addMenu("\U0001f4cb &Profiles")
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

    def _toggle_tuner_panel(self):
        """Show/hide the servo tuner dock panel."""
        if TunerPanel is None:
            QMessageBox.warning(self.window, "Servo Tuner",
                                "Tuner module not available. Check src/ai/ is present.")
            return

        if self._tuner_panel is None:
            self._tuner_panel = TunerPanel(self.window)
            self._tuner_panel.set_data_provider(self._get_scope_data_for_ai)
            if self.trio_connected and self.trio_connection:
                self._tuner_panel.set_connection(self.trio_connection, self._conn_lock)
            # Restore saved per-axis drive profiles
            app_settings = SettingsStore().load()
            if app_settings.drive_profiles:
                self._tuner_panel.set_all_profiles(app_settings.drive_profiles)
            # Preserve window geometry — adding the dock triggers a deferred
            # layout pass that inflates the main window's minimumSizeHint.
            saved_size = self.size()
            self.setFixedWidth(saved_size.width())
            self.addDockWidget(Qt.RightDockWidgetArea, self._tuner_panel)
            # Release the width lock after the layout settles
            QTimer.singleShot(0, lambda: self.setMaximumWidth(16777215))
        else:
            self._tuner_panel.setVisible(not self._tuner_panel.isVisible())

    def _toggle_measurement_panel(self):
        """Show/hide the live measurements window."""
        if self._measurement_panel is None:
            self._measurement_panel = MeasurementPanel(self.window)
            self._sync_measurement_panel(force=True)
            self._measurement_panel.show()
            self._measurement_panel.raise_()
            self._measurement_panel.activateWindow()
        else:
            if self._measurement_panel.isVisible():
                self._measurement_panel.hide()
            else:
                self._sync_measurement_panel(force=True)
                self._measurement_panel.show()
                self._measurement_panel.raise_()
                self._measurement_panel.activateWindow()

    def _sync_measurement_panel(self, force=False):
        """Push the latest capture buffer into the measurements window."""
        if self._measurement_panel is None:
            return
        if not self._measurement_panel.isVisible() and not force:
            return
        if self.accumulated_data is None:
            self._measurement_panel.clear()
            return

        trace_names = [
            t.get_display_name()
            for t in self.get_enabled_traces()
            if t.get_display_name() in self.accumulated_data.get('params', {})
        ]
        cursor_window = None
        if self._cursors_enabled and self.plot_mode == 'time':
            cursor_window = (
                float(self._cursor_pos['c1']),
                float(self._cursor_pos['c2']),
            )
        self._measurement_panel.set_capture_data(
            self.accumulated_data.get('time'),
            self.accumulated_data.get('params'),
            trace_names=trace_names,
            cursor_window=cursor_window,
            segment_breaks=self.accumulated_data.get('segment_breaks', []),
        )
        if force:
            self._measurement_panel.refresh_now()

    def _open_ethercat_map(self):
        """Open the EtherCAT network map window."""
        if EthercatMapWindow is None:
            QMessageBox.warning(self.window, "EtherCAT Map",
                                "EtherCAT map module not available. Check src/ai/ is present.")
            return

        if not self.trio_connected or not self.trio_connection:
            QMessageBox.warning(self.window, "EtherCAT Map",
                                "Connect to a Trio controller first.")
            return

        if self._ethercat_map is None or not self._ethercat_map.isVisible():
            self._ethercat_map = EthercatMapWindow(
                self.trio_connection, parent=self.window, conn_lock=self._conn_lock)
            self._ethercat_map.show()
        else:
            self._ethercat_map.raise_()
            self._ethercat_map.activateWindow()

    def _get_scope_data_for_ai(self):
        """Data provider callback for the analysis panels.

        Returns (time_arr, params_dict, servo_period_sec, segment_breaks).
        servo_period_sec scales DEMAND_SPEED (captured as units/servocycle)
        into units/second; segment_breaks marks continuous-mode splice
        points so the analysis never computes gradients or FFTs across them.
        """
        if self.accumulated_data is None:
            return None, None, None, None
        time_arr = self.accumulated_data.get('time')
        params = self.accumulated_data.get('params')
        if time_arr is None or len(time_arr) == 0:
            return None, None, None, None
        servo_period_sec = None
        if self.scope_engine is not None:
            servo_period_sec = self.scope_engine.servo_period_sec
        segment_breaks = self.accumulated_data.get('segment_breaks', [])
        return time_arr, params, servo_period_sec, segment_breaks

    def open_settings(self):
        if self._settings_window is not None:
            try:
                self._settings_window.raise_()
                self._settings_window.activateWindow()
                return
            except RuntimeError:
                self._settings_window = None

        dlg = QDialog(self.window)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumSize(340, 700)
        dlg.setStyleSheet(DARK_STYLESHEET)
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg.destroyed.connect(lambda: setattr(self, '_settings_window', None))
        self._settings_window = dlg

        main_layout = QVBoxLayout(dlg)

        # Display section
        display_group = QGroupBox("Display")
        display_layout = QFormLayout(display_group)

        window_dur_edit = QLineEdit(str(self.window_duration))
        display_layout.addRow("Scroll window (s):", window_dur_edit)

        lock_x_chk = QCheckBox("Lock X-Axis across subplots")
        lock_x_chk.setChecked(self.lock_x_axis)
        display_layout.addRow(lock_x_chk)
        main_layout.addWidget(display_group)

        # Capture section
        capture_group = QGroupBox("Capture")
        capture_layout = QFormLayout(capture_group)

        use_end_chk = QCheckBox("Use end of TABLE")
        use_end_chk.setChecked(self.use_end_of_table)
        capture_layout.addRow(use_end_chk)

        table_start_edit = QLineEdit(self.table_start_edit.text())
        table_start_edit.setEnabled(not self.use_end_of_table)
        use_end_chk.toggled.connect(lambda checked: table_start_edit.setEnabled(not checked))
        capture_layout.addRow("Table Start:", table_start_edit)
        main_layout.addWidget(capture_group)

        # Plot Style section
        style_group = QGroupBox("Plot Style")
        style_layout = QFormLayout(style_group)

        line_w_edit = QLineEdit(str(self.line_width))
        style_layout.addRow("Line width:", line_w_edit)

        grid_a_edit = QLineEdit(str(self.grid_alpha))
        style_layout.addRow("Grid opacity (0-1):", grid_a_edit)

        plot_bg_edit = QLineEdit(self.plot_bg_color)
        style_layout.addRow("Plot background:", plot_bg_edit)
        main_layout.addWidget(style_group)

        # Buttons
        btn_layout = QHBoxLayout()

        def apply_settings():
            try:
                self.window_duration = float(window_dur_edit.text())
                self.lock_x_axis = lock_x_chk.isChecked()
                self.chk_lock_x.setChecked(self.lock_x_axis)
                self.use_end_of_table = use_end_chk.isChecked()
                self.table_start_edit.setText(table_start_edit.text())
                self.line_width = float(line_w_edit.text())
                self.grid_alpha = max(0.0, min(1.0, float(grid_a_edit.text())))
                self.plot_bg_color = plot_bg_edit.text()
                self._apply_plot_settings()
                self.status_label.setText("Settings applied")
            except ValueError as e:
                QMessageBox.critical(dlg, "Invalid value", str(e))

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(apply_settings)
        btn_layout.addWidget(btn_apply)

        btn_ok = QPushButton("OK")
        btn_ok.setObjectName("accent")
        btn_ok.clicked.connect(lambda: (apply_settings(), dlg.close()))
        btn_layout.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.close)
        btn_layout.addWidget(btn_cancel)
        main_layout.addLayout(btn_layout)

        dlg.show()

    def _load_settings(self):
        """Restore saved settings from QSettings."""
        app_settings = SettingsStore().load()

        # Connection
        self.ip_edit.setText(app_settings.connection.ip)

        # Configuration
        self.period_edit.setText(app_settings.capture.sample_period)
        self.duration_edit.setText(app_settings.capture.duration)
        self.table_start_edit.setText(app_settings.capture.table_start)
        self.use_end_of_table = app_settings.capture.use_end_of_table
        if app_settings.capture.capture_mode == "single":
            self.radio_single.setChecked(True)
        else:
            self.radio_continuous.setChecked(True)
        self.external_trigger_chk.setChecked(app_settings.capture.external_trigger)

        # Display / plot settings
        self.plot_mode = app_settings.display.plot_mode
        mode_index = {'time': 0, 'xy': 1, 'xyz': 2, 'xyzw': 3}.get(self.plot_mode, 0)
        self.plot_mode_combo.setCurrentIndex(mode_index)
        self.window_duration = app_settings.display.window_duration
        self.lock_x_axis = app_settings.display.lock_x_axis
        self.chk_lock_x.setChecked(self.lock_x_axis)
        self.path_view_scale = max(
            0.25, min(4.0, app_settings.display.path_view_scale)
        )
        self.path_view_scale_slider.setValue(round(self.path_view_scale * 100))
        self.line_width = app_settings.plot.line_width
        self.grid_alpha = app_settings.plot.grid_alpha
        self.plot_bg_color = app_settings.plot.bg_color

        # Traces
        if not app_settings.traces:
            app_settings.traces.append(TraceConfig(param="MPOS", axis=0, enabled=True, fft=False))

        # Recreate traces in UI
        for i, trace_config in enumerate(app_settings.traces):
            if i >= len(self.traces):
                self.add_trace()
            t = self.traces[i]
            t.param_combo.setCurrentText((trace_config.param or "MPOS").strip() or "MPOS")
            t.axis_spin.setValue(trace_config.axis)
            t.chk_enable.setChecked(trace_config.enabled)
            t.set_fft(trace_config.fft)

        # Remove extra UI trace controls if there are any
        while len(self.traces) > len(app_settings.traces):
            t = self.traces.pop()
            t.setParent(None)
            t.deleteLater()

        self.axis_parameters_tab.set_configurations(app_settings.axis_parameters)

    def _save_settings(self):
        """Persist current settings to QSettings."""
        app_settings = AppSettings()

        # Connection
        app_settings.connection.ip = self.ip_edit.text()

        # Configuration
        app_settings.capture.sample_period = self.period_edit.text()
        app_settings.capture.duration = self.duration_edit.text()
        app_settings.capture.table_start = self.table_start_edit.text()
        app_settings.capture.use_end_of_table = self.use_end_of_table
        app_settings.capture.capture_mode = "single" if self.radio_single.isChecked() else "continuous"
        app_settings.capture.external_trigger = self.external_trigger_chk.isChecked()

        # Display / plot settings
        app_settings.display.plot_mode = self.plot_mode
        app_settings.display.window_duration = self.window_duration
        app_settings.display.lock_x_axis = self.lock_x_axis
        app_settings.display.path_view_scale = self.path_view_scale
        app_settings.plot.line_width = self.line_width
        app_settings.plot.grid_alpha = self.grid_alpha
        app_settings.plot.bg_color = self.plot_bg_color

        # Traces
        for t in self.traces:
            app_settings.traces.append(TraceConfig(
                param=t.param_combo.currentText().strip() or "MPOS",
                axis=t.axis_spin.value(),
                enabled=t.chk_enable.isChecked(),
                fft=t.is_fft()
            ))

        # Per-axis drive profiles (from tuner panel if open)
        if self._tuner_panel is not None:
            app_settings.drive_profiles = self._tuner_panel.get_all_profiles()
        else:
            # We should load the existing ones from settings so we don't overwrite them with empty
            existing = SettingsStore().load()
            app_settings.drive_profiles = existing.drive_profiles

        app_settings.axis_parameters = self.axis_parameters_tab.configurations()

        SettingsStore().save(app_settings)

    def closeEvent(self, event):
        self._save_settings()
        self._shutting_down = True
        self.is_running = False
        self._update_timer.stop()
        self._stop_watchdog()
        self._disable_motion_axes_before_disconnect()
        if self._measurement_panel is not None:
            self._measurement_panel.hide()
        if self._motion_window is not None:
            self._motion_window.close()
        for compare_window in list(self._compare_windows):
            compare_window.close()
        self._compare_windows.clear()
        if self.trio_connected and self.trio_connection:
            # Close with 5s timeout — don't block app exit on dead socket
            close_done = threading.Event()
            def _close():
                try:
                    self.trio_connection.CloseConnection()
                except Exception:
                    pass
                finally:
                    close_done.set()
            threading.Thread(target=_close, daemon=True).start()
            close_done.wait(timeout=5.0)
        self.trio_connection = None
        self.trio_connected = False
        event.accept()

