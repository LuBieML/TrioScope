"""Dock panel and tool window toggles (tuner, measurements, EtherCAT map)."""

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMessageBox

try:
    from ai.tuner_panel import TunerPanel
except ImportError:
    TunerPanel = None
try:
    from ai.ethercat_map_window import EthercatMapWindow
except ImportError:
    EthercatMapWindow = None

from storage.settings_store import SettingsStore
from ui.measurement_panel import MeasurementPanel

logger = logging.getLogger(__name__)


class PanelsMixin:
    """Show/hide handlers for the auxiliary panels and windows."""

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
        """Data provider callback for AI panel.

        Returns (time_arr, params_dict, servo_period_sec). servo_period_sec
        is needed to scale DEMAND_SPEED (captured as units/servocycle) into
        units/second for velocity-loop analysis.
        """
        if self.accumulated_data is None:
            return None, None, None
        time_arr = self.accumulated_data.get('time')
        params = self.accumulated_data.get('params')
        if time_arr is None or len(time_arr) == 0:
            return None, None, None
        servo_period_sec = None
        if self.scope_engine is not None:
            servo_period_sec = self.scope_engine.servo_period_sec
        return time_arr, params, servo_period_sec
