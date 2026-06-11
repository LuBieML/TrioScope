"""Capture-source switching and drive scope UI helpers."""

import logging

logger = logging.getLogger(__name__)


class CaptureSourceUiMixin:
    """Source selector handling shared by CaptureController."""

    def _on_source_changed(self, index):
        """Toggle between Controller SCOPE and Drive Scope UI."""
        is_drive = (index == 1)
        self.capture_source = 'drive' if is_drive else 'controller'

        # Controller SCOPE widgets
        for w in (self.ctrl_period_label, self.period_edit, self.ctrl_period_unit,
                  self.ctrl_duration_label, self.duration_edit, self.ctrl_duration_unit,
                  self.ctrl_mode_label, self.ctrl_mode_widget, self.external_trigger_chk):
            w.setVisible(not is_drive)

        # Drive Scope widgets
        for w in (self.drv_sample_label, self.drv_sample_edit, self.drv_sample_unit,
                  self.drv_trigger_label, self.drv_trigger_combo,
                  self.drv_axis_label, self.drv_axis_spin, self.drv_info_label):
            w.setVisible(is_drive)

        if is_drive:
            self._update_drive_info_label()
            self._on_drive_trigger_changed()  # show/hide trigger value inputs
            # Switch trace controls to drive variable mode
            for trace in self.traces:
                trace.set_drive_mode(True)
        else:
            self.drv_info_label.setText("")
            # Hide trigger value inputs when switching away from drive mode
            self.drv_trig_val_label.setVisible(False)
            self.drv_trig_val1_edit.setVisible(False)
            self.drv_trig_val2_edit.setVisible(False)
            for trace in self.traces:
                trace.set_drive_mode(False)

    def _get_drive_sample_time_units(self) -> int:
        """Convert capture duration (seconds) to drive sample_time units (×125 μs).

        sample_time_units = duration_s / (1000 × 125 μs)
        e.g. 1.0 s → 1.0 / 0.125 = 8 units → 8 × 125 μs = 1 ms per sample
        """
        try:
            duration_s = float(self.drv_sample_edit.text())
        except ValueError:
            return 8  # default → 1 ms/sample → 1 s capture
        # duration_s = 1000_samples × sample_time_units × 125e-6
        # sample_time_units = duration_s / (1000 × 125e-6) = duration_s / 0.125
        units = max(1, round(duration_s / 0.125))
        return units

    def _update_drive_info_label(self):
        """Update the drive scope info label and resolution display."""
        units = self._get_drive_sample_time_units()
        period_us = units * 125
        # Update resolution next to the "s" unit label
        if period_us >= 1000:
            res_str = f"{period_us / 1000:.2f} ms"
        else:
            res_str = f"{period_us} μs"
        self.drv_sample_unit.setText(f"s  (res: {res_str})")

    def _on_drive_trigger_changed(self):
        """Show/hide trigger value inputs based on selected trigger mode."""
        mode = self.drv_trigger_combo.currentData()
        # Modes needing a threshold: 1=Rising, 2=Falling, 3=Greater, 4=Less
        needs_value1 = mode in (1, 2, 3, 4, 5, 6)
        # Window modes need two thresholds: 5=Inside, 6=Outside
        needs_value2 = mode in (5, 6)

        is_drive = (self.capture_source == 'drive')
        self.drv_trig_val_label.setVisible(is_drive and needs_value1)
        self.drv_trig_val1_edit.setVisible(is_drive and needs_value1)
        self.drv_trig_val2_edit.setVisible(is_drive and needs_value2)
