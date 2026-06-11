"""Settings dialog, QSettings persistence, and application shutdown."""

import logging
import threading

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout,
)

from models.app_settings import AppSettings
from models.trace_config import TraceConfig
from storage.settings_store import SettingsStore
from ui.theme import DARK_STYLESHEET

logger = logging.getLogger(__name__)


class SettingsMixin:
    """Settings dialog, load/save persistence, and closeEvent handling."""

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
        mode_index = {'time': 0, 'xy': 1, 'xyz': 2}.get(self.plot_mode, 0)
        self.plot_mode_combo.setCurrentIndex(mode_index)
        self.window_duration = app_settings.display.window_duration
        self.lock_x_axis = app_settings.display.lock_x_axis
        self.chk_lock_x.setChecked(self.lock_x_axis)
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

        SettingsStore().save(app_settings)

    def closeEvent(self, event):
        self._save_settings()
        self._shutting_down = True
        self.is_running = False
        self._update_timer.stop()
        self._stop_watchdog()
        if self._measurement_panel is not None:
            self._measurement_panel.hide()
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
