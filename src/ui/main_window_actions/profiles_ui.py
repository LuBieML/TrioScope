"""Trace profile actions: save/load/rename/delete and their dialogs."""

import logging

from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout,
)

from models.trace_config import TraceConfig
from storage.profiles import ProfileStore
from ui.profile_dialog import _ProfileManagerDialog
from ui.theme import DARK_STYLESHEET

logger = logging.getLogger(__name__)


class TraceProfilesMixin:
    """Named trace-configuration profiles (View → Profiles)."""

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

        act_manage = self._profiles_menu.addAction("⚙ Manage Profiles…")
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
                status = "✓" if t.chk_enable.isChecked() else "✗"
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
