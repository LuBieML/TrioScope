"""Screenshot, CSV export/import, and HTML report export actions."""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from version import __version__
from reports.html_report import write_html_report
from scope.drive_scope_engine import TRIGGER_MODES
from storage.csv_io import CSVStorage
from storage.settings_store import SettingsStore
from ui.theme import DARK_STYLESHEET, TRACE_COLORS

logger = logging.getLogger(__name__)


class ExportImportMixin:
    """File-oriented actions: screenshot, CSV export/import, HTML report."""

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
