from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QGroupBox, QWidget, QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, QSettings

from ui.theme import DARK_STYLESHEET, TRACE_COLORS

class _ProfileManagerDialog(QDialog):
    """Dialog for managing saved trace profiles: load, rename, delete."""

    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self._app = app
        self.setWindowTitle("Manage Profiles")
        self.setMinimumSize(460, 380)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Saved Profiles")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(8)

        # Profile list
        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background-color: #2e2e2e; border: 1px solid #4b4a4a;"
            " border-radius: 3px; font-size: 10pt; }"
            "QListWidget::item { padding: 5px 8px; }"
            "QListWidget::item:selected { background-color: #FFA500; color: #000; }"
        )
        self._list.currentRowChanged.connect(self._on_selection_changed)
        body.addWidget(self._list, 1)

        # Buttons column
        btn_col = QVBoxLayout()
        btn_col.setSpacing(6)

        self.btn_load = QPushButton("▶ Load")
        self.btn_load.setToolTip("Load selected profile (replaces current traces)")
        self.btn_load.clicked.connect(self._on_load)
        btn_col.addWidget(self.btn_load)

        self.btn_rename = QPushButton("✏ Rename")
        self.btn_rename.setToolTip("Rename the selected profile")
        self.btn_rename.clicked.connect(self._on_rename)
        btn_col.addWidget(self.btn_rename)

        self.btn_delete = QPushButton("🗑 Delete")
        self.btn_delete.setToolTip("Delete the selected profile")
        self.btn_delete.clicked.connect(self._on_delete)
        btn_col.addWidget(self.btn_delete)

        btn_col.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_col.addWidget(btn_close)

        body.addLayout(btn_col)
        layout.addLayout(body, 1)

        # Preview area
        preview_frame = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 12, 8, 8)
        preview_layout.setSpacing(2)
        self._preview_container = QWidget()
        self._preview_layout = QVBoxLayout(self._preview_container)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_layout.setSpacing(1)
        self._preview_layout.setAlignment(Qt.AlignTop)
        preview_layout.addWidget(self._preview_container)
        layout.addWidget(preview_frame)

        self._refresh_list()
        self._on_selection_changed(-1)

    def _refresh_list(self):
        self._list.clear()
        for name in self._app._get_profile_names():
            self._list.addItem(name)

    def _selected_name(self):
        item = self._list.currentItem()
        return item.text() if item else None

    def _on_selection_changed(self, row):
        has_sel = row >= 0
        self.btn_load.setEnabled(has_sel)
        self.btn_rename.setEnabled(has_sel)
        self.btn_delete.setEnabled(has_sel)
        self._update_preview()

    def _update_preview(self):
        # Clear old preview
        while self._preview_layout.count():
            child = self._preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        name = self._selected_name()
        if not name:
            lbl = QLabel("Select a profile to preview")
            lbl.setStyleSheet("color: #666; font-style: italic;")
            self._preview_layout.addWidget(lbl)
            return

        s = QSettings("TrioScope", "ParameterScope")
        count = int(s.value(f"profiles/data/{name}/count", 0))
        if count == 0:
            lbl = QLabel("(empty profile)")
            lbl.setStyleSheet("color: #666;")
            self._preview_layout.addWidget(lbl)
            return

        for i in range(count):
            param = str(s.value(f"profiles/data/{name}/{i}/param", "?"))
            axis = int(s.value(f"profiles/data/{name}/{i}/axis", 0))
            enabled = s.value(f"profiles/data/{name}/{i}/enabled", "true") == "true"
            fft = s.value(f"profiles/data/{name}/{i}/fft", "false") == "true"

            status = "✓" if enabled else "✗"
            fft_tag = " [FFT]" if fft else ""
            color = TRACE_COLORS[i % len(TRACE_COLORS)]
            text = f"  {status}  {param}({axis}){fft_tag}"
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {color}; font-family: Consolas; font-size: 9pt;"
            )
            self._preview_layout.addWidget(lbl)

    def _on_load(self):
        name = self._selected_name()
        if name:
            self._app._load_profile(name)
            self.close()

    def _on_rename(self):
        old_name = self._selected_name()
        if not old_name:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Profile", "New name:", text=old_name)
        new_name = new_name.strip() if ok else ""
        if not new_name or new_name == old_name:
            return
        existing = self._app._get_profile_names()
        if new_name in existing:
            QMessageBox.warning(self, "Rename",
                                f"A profile named '{new_name}' already exists.")
            return
        self._app._rename_profile(old_name, new_name)
        self._refresh_list()

    def _on_delete(self):
        name = self._selected_name()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete profile '{name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._app._delete_profile(name)
            self._refresh_list()
            self._on_selection_changed(-1)
