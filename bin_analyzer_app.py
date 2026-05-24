#!/usr/bin/env python3
"""
TrioScope Scope Binary File Analyzer
Standalone GUI application to load, parse, analyze, and plot drive scope binary files
(drive_scope.bin and drive_scope_fifo_raw.bin) according to design documents.
"""

import sys
import os
import pathlib
import hashlib
import datetime
import csv
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QFrame, QScrollArea,
    QFileDialog, QMessageBox, QGridLayout, QSplitter, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QGroupBox
)
from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QFont, QColor, QIcon, QKeySequence, QAction
import pyqtgraph as pg

# Configure pyqtgraph
pg.setConfigOptions(
    background='#0A0A0A',
    foreground='#d4d4d4',
    antialias=True
)

# ── DESIGN SYSTEM TOKENS ────────────────────────────────────────────────
TRACE_COLORS = [
    '#03DAC6',  # Teal
    '#FFB74D',  # Orange
    '#64B5F6',  # Blue
    '#F06292',  # Pink
    '#FFF176',  # Yellow
    '#E57373',  # Red
    '#81C784',  # Green
    '#BA68C8',  # Purple
]

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1E1E1E;
    color: #D4D4D4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 9pt;
}
QGroupBox {
    background-color: #252526;
    border: 1px solid #3F3F46;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #03DAC6;
}
QPushButton {
    background-color: #2D2D30;
    color: #D4D4D4;
    border: 1px solid #3F3F46;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #3F3F46;
    border-color: #52525B;
}
QPushButton:pressed {
    background-color: #1E1E1E;
}
QPushButton#primary {
    background-color: #03DAC6;
    color: #121212;
    border: 1px solid #01B3A0;
    font-weight: bold;
}
QPushButton#primary:hover {
    background-color: #33E0D1;
}
QPushButton#primary:pressed {
    background-color: #019989;
}
QComboBox, QLineEdit {
    background-color: #2D2D30;
    color: #D4D4D4;
    border: 1px solid #3F3F46;
    border-radius: 4px;
    padding: 4px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #2D2D30;
    color: #D4D4D4;
    selection-background-color: #03DAC6;
    selection-color: #121212;
}
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3F3F46;
    border-radius: 3px;
    background-color: #1E1E1E;
}
QCheckBox::indicator:checked {
    background-color: #03DAC6;
    border-color: #03DAC6;
}
QPlainTextEdit {
    background-color: #181818;
    color: #D4D4D4;
    border: 1px solid #2D2D30;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9.5pt;
    line-height: 140%;
}
QTableWidget {
    background-color: #1E1E1E;
    gridline-color: #2D2D30;
    border: 1px solid #2D2D30;
}
QHeaderView::section {
    background-color: #252526;
    color: #D4D4D4;
    border: 1px solid #2D2D30;
    padding: 4px;
    font-weight: bold;
}
QScrollBar:vertical {
    background-color: #1E1E1E;
    width: 10px;
}
QScrollBar::handle:vertical {
    background-color: #3F3F46;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #52525B;
}
"""

# ── DRIVE VARIABLE DEFINITIONS (from Docx Object 0x368C Sub 8-15) ────────
DRIVE_VARIABLES = {
    0x0000: ("Disabled", "Channel Disabled", "", "Uint16"),
    0x0F10: ("SPD_FB_RPM", "Speed feedback", "rpm", "Int16"),
    0x0F11: ("SPD_CMD_RPM", "Speed command", "rpm", "Int16"),
    0x0F13: ("TN", "Torque command %", "%Tn", "Int16"),
    0x0F16: ("CURRENT_POS_L1", "Current pos low 16b", "pulse", "Int64"),
    0x0F17: ("CURRENT_POS_H1", "Current pos mid-low 16b", "pulse", "Int64"),
    0x0F18: ("CURRENT_POS_L2", "Current pos mid-high 16b", "pulse", "Int64"),
    0x0F19: ("CURRENT_POS_H2", "Current pos high 16b", "pulse", "Int64"),
    0x0F1C: ("IU", "Phase U current", "0.1%rated", "Int16"),
    0x0F1D: ("IV", "Phase V current", "0.1%rated", "Int16"),
    0x0F1E: ("ID_REF", "Id reference", "0.1%rated", "Int16"),
    0x0F1F: ("ID", "Id actual", "0.1%rated", "Int16"),
    0x0F20: ("IQ_REF", "Iq reference", "0.1%rated", "Int16"),
    0x0F21: ("IQ", "Iq actual", "0.1%rated", "Int16"),
    0x0F22: ("UD", "Ud voltage", "V", "Uint16"),
    0x0F23: ("UQ", "Uq voltage", "V", "Uint16"),
    0x0F2A: ("EST_SPD_L", "Observer speed low 16b", "0.1rpm", "Int32"),
    0x0F2B: ("EST_SPD_H", "Observer speed high 16b", "0.1rpm", "Int32"),
    0x0F2C: ("EST_TORQ_PER", "Observer torque", "0.1%rated", "Int16"),
    0x0F2D: ("FF_SPEED", "Speed feedforward", "rpm", "Int16"),
    0x0F2E: ("FF_TORQUE", "Torque feedforward", "0.1%rated", "Uint16"),
    0x0F2F: ("PGERR_SPEED", "Pos cmd speed", "rpm", "Int16"),
}


class TrioScopeAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrioScope - Standalone Scope Binary File Analyzer")
        self.resize(1500, 950)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # State variables
        self.current_file_path = None
        self.raw_data = None
        self.file_type = ""
        self.md5_hash = ""
        self.file_size = 0
        self.mod_time = ""
        
        # Channel configurations: list of dicts holding user settings per channel
        # e.g., {'selected_address': 0x0F10, 'force_type': 'Auto', 'visible': True, 'color': '#03DAC6'}
        self.channel_configs = []
        default_var_keys = [0x0F10, 0x0F13, 0x0F21, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000]
        # Let's map some common defaults
        initial_defaults = [0x0F10, 0x0F11, 0x0F13, 0x0F21, 0x0F22, 0x0F2C, 0x0000, 0x0000]
        
        for i in range(8):
            self.channel_configs.append({
                'address': initial_defaults[i],
                'force_type': 'Auto',
                'visible': True,
                'color': TRACE_COLORS[i]
            })

        self.init_ui()
        
        # Enable Drag and Drop
        self.setAcceptDrops(True)
        
        # Auto-load project default files if they exist
        self.auto_load_default()

    def init_ui(self):
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ── TOP BAR (File selection & general settings) ──────────────────
        top_bar = QFrame()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)
        
        title_label = QLabel("TrioScope Analyzer")
        title_label.setStyleSheet("font-size: 15pt; font-weight: bold; color: #03DAC6;")
        top_layout.addWidget(title_label)
        
        # File selector button
        self.btn_open = QPushButton("📂 Open Scope Bin...")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.select_file)
        top_layout.addWidget(self.btn_open)
        
        # Currently loaded file path label
        self.lbl_filepath = QLabel("No file loaded. Drag & drop a .bin file here or click Open.")
        self.lbl_filepath.setStyleSheet("font-style: italic; color: #8E8E93;")
        top_layout.addWidget(self.lbl_filepath, 1)
        
        # X-Axis Time Configuration
        top_layout.addWidget(QLabel("Sample Time Multiplier:"))
        self.txt_sample_time = QLineEdit("8")
        self.txt_sample_time.setFixedWidth(40)
        self.txt_sample_time.setToolTip("Sub-index 7 of Setup Object 0x368C. Default is 8.")
        self.txt_sample_time.textChanged.connect(self.update_analysis)
        top_layout.addWidget(self.txt_sample_time)
        
        top_layout.addWidget(QLabel("Time Unit (μs):"))
        self.txt_time_unit = QLineEdit("125")
        self.txt_time_unit.setFixedWidth(40)
        self.txt_time_unit.setToolTip("Basic time unit (125 μs).")
        self.txt_time_unit.textChanged.connect(self.update_analysis)
        top_layout.addWidget(self.txt_time_unit)
        
        # Overlay vs Stacked Plots mode
        self.chk_stacked_plots = QCheckBox("Stacked Subplots")
        self.chk_stacked_plots.setChecked(True)
        self.chk_stacked_plots.toggled.connect(self.update_plots)
        top_layout.addWidget(self.chk_stacked_plots)
        
        # Export Actions
        self.btn_export_csv = QPushButton("📥 Export CSV...")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setEnabled(False)
        top_layout.addWidget(self.btn_export_csv)
        
        self.btn_export_report = QPushButton("📄 Save Report...")
        self.btn_export_report.clicked.connect(self.export_report)
        self.btn_export_report.setEnabled(False)
        top_layout.addWidget(self.btn_export_report)
        
        main_layout.addWidget(top_bar)
        
        # ── MAIN SPLITTER (Left: Report, Center: Plot, Right: Channel Controls)
        splitter = QSplitter(Qt.Horizontal)
        
        # 1. Left Panel (Metadata & Report)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        meta_group = QGroupBox("File Metadata")
        meta_grid = QGridLayout(meta_group)
        meta_grid.setSpacing(6)
        meta_grid.setContentsMargins(8, 8, 8, 8)
        
        meta_grid.addWidget(QLabel("File Size:"), 0, 0)
        self.lbl_size = QLabel("-")
        self.lbl_size.setStyleSheet("font-weight: bold;")
        meta_grid.addWidget(self.lbl_size, 0, 1)
        
        meta_grid.addWidget(QLabel("MD5 Hash:"), 1, 0)
        self.lbl_md5 = QLabel("-")
        self.lbl_md5.setStyleSheet("font-family: Consolas; font-size: 8pt;")
        meta_grid.addWidget(self.lbl_md5, 1, 1)
        
        meta_grid.addWidget(QLabel("File Type:"), 2, 0)
        self.lbl_type = QLabel("-")
        self.lbl_type.setStyleSheet("color: #03DAC6; font-weight: bold;")
        meta_grid.addWidget(self.lbl_type, 2, 1)
        
        meta_grid.addWidget(QLabel("Modified:"), 3, 0)
        self.lbl_modified = QLabel("-")
        meta_grid.addWidget(self.lbl_modified, 3, 1)
        
        left_layout.addWidget(meta_group)
        
        report_group = QGroupBox("Scope Analysis Report")
        report_layout = QVBoxLayout(report_group)
        report_layout.setContentsMargins(4, 8, 4, 4)
        
        self.txt_report = QPlainTextEdit()
        self.txt_report.setReadOnly(True)
        report_layout.addWidget(self.txt_report)
        
        left_layout.addWidget(report_group, 1)
        
        splitter.addWidget(left_widget)
        splitter.setStretchFactor(0, 2)
        
        # 2. Center Panel (PyQtGraph plots)
        self.plot_layout_widget = pg.GraphicsLayoutWidget()
        self.plot_layout_widget.setMinimumWidth(600)
        splitter.addWidget(self.plot_layout_widget)
        splitter.setStretchFactor(1, 6)
        
        # 3. Right Panel (Channel mappings and checkboxes)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(400)
        
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(10)
        
        ch_group = QGroupBox("Interleaved Channels Mapping")
        self.ch_group_layout = QVBoxLayout(ch_group)
        self.ch_group_layout.setSpacing(12)
        self.ch_group_layout.setContentsMargins(6, 12, 6, 6)
        
        # Generate 8 channel mapping controls
        self.channel_ui_widgets = []
        for i in range(8):
            ch_box = self.create_channel_mapping_widget(i)
            self.ch_group_layout.addWidget(ch_box)
            self.channel_ui_widgets.append(ch_box)
            
        right_layout.addWidget(ch_group)
        right_layout.addStretch()
        right_scroll.setWidget(right_container)
        
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(2, 2)
        
        main_layout.addWidget(splitter, 1)

    def create_channel_mapping_widget(self, ch_idx):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ border: 1px solid #2D2D30; border-radius: 4px; background-color: #202021; }}")
        
        grid = QGridLayout(frame)
        grid.setSpacing(6)
        grid.setContentsMargins(6, 6, 6, 6)
        
        # Checkbox & Color Indicator
        chk = QCheckBox()
        chk.setChecked(self.channel_configs[ch_idx]['visible'])
        chk.stateChanged.connect(lambda state, idx=ch_idx: self.on_channel_toggled(idx, state))
        grid.addWidget(chk, 0, 0)
        
        color_dot = QLabel("█")
        color_dot.setStyleSheet(f"color: {self.channel_configs[ch_idx]['color']}; font-size: 12pt;")
        grid.addWidget(color_dot, 0, 1)
        
        lbl_title = QLabel(f"Channel {ch_idx+1}")
        lbl_title.setStyleSheet("font-weight: bold; color: #FFF;")
        grid.addWidget(lbl_title, 0, 2)
        
        # Variable address selection dropdown
        cmb = QComboBox()
        cmb.setMinimumWidth(160)
        for addr, (name, desc, unit, dtype) in DRIVE_VARIABLES.items():
            if addr == 0x0000:
                cmb.addItem("0x0000: Disabled", addr)
            else:
                cmb.addItem(f"0x{addr:04X}: {name} ({unit})", addr)
        
        # Set current index based on initial default configs
        default_addr = self.channel_configs[ch_idx]['address']
        index = cmb.findData(default_addr)
        if index >= 0:
            cmb.setCurrentIndex(index)
            
        cmb.currentIndexChanged.connect(lambda val, idx=ch_idx: self.on_variable_changed(idx, cmb))
        grid.addWidget(cmb, 0, 3)
        
        # Signed/Unsigned selection override
        cmb_type = QComboBox()
        cmb_type.addItems(["Auto", "Force Signed (Int16)", "Force Unsigned (Uint16)"])
        cmb_type.setFixedWidth(100)
        cmb_type.currentIndexChanged.connect(lambda val, idx=ch_idx: self.on_type_override_changed(idx, cmb_type))
        grid.addWidget(cmb_type, 0, 4)
        
        # Statistic Display Labels
        lbl_stats = QLabel("No file loaded")
        lbl_stats.setStyleSheet("font-size: 8.5pt; color: #8E8E93; font-family: Consolas;")
        grid.addWidget(lbl_stats, 1, 0, 1, 5)
        
        # Keep track of controls in config
        self.channel_configs[ch_idx]['chk_widget'] = chk
        self.channel_configs[ch_idx]['cmb_widget'] = cmb
        self.channel_configs[ch_idx]['type_widget'] = cmb_type
        self.channel_configs[ch_idx]['stats_label'] = lbl_stats
        
        return frame

    # ── EVENT HANDLERS ──────────────────────────────────────────────────
    def on_channel_toggled(self, ch_idx, state):
        self.channel_configs[ch_idx]['visible'] = (state == Qt.Checked.value)
        self.update_plots()

    def on_variable_changed(self, ch_idx, cmb):
        addr = cmb.currentData()
        self.channel_configs[ch_idx]['address'] = addr
        self.update_analysis()

    def on_type_override_changed(self, ch_idx, cmb_type):
        text = cmb_type.currentText()
        self.channel_configs[ch_idx]['force_type'] = text
        self.update_analysis()

    # Drag and Drop handlers
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.bin'):
                self.load_file(file_path)
                break

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Scope Binary File", "", "Binary Files (*.bin);;All Files (*)"
        )
        if file_path:
            self.load_file(file_path)

    def auto_load_default(self):
        # Check standard project locations for default drive scope files
        candidates = [
            "./drive_scope_fifo_raw.bin",
            "./drive_scope.bin",
            "../drive_scope_fifo_raw.bin",
            "../drive_scope.bin"
        ]
        # Resolve absolute paths in current workspace
        for c in candidates:
            p = pathlib.Path(c).resolve()
            if p.exists():
                self.load_file(str(p))
                break

    # ── LOADING AND PARSING ─────────────────────────────────────────────
    def load_file(self, file_path):
        try:
            path = pathlib.Path(file_path)
            self.current_file_path = file_path
            
            data = path.read_bytes()
            self.raw_data = data
            self.file_size = len(data)
            self.md5_hash = hashlib.md5(data).hexdigest()
            
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            self.mod_time = mtime.strftime("%Y-%m-%d %H:%M:%S")
            
            # Identify file layout
            if self.file_size == 33024:
                self.file_type = "Raw Controller FIFO File (33,024 bytes)"
            elif self.file_size == 16000:
                self.file_type = "Composed Scope Payload (16,000 bytes)"
            else:
                self.file_type = f"Non-standard Binary File ({self.file_size:,} bytes)"
                
            self.lbl_filepath.setText(f"File: {path.name}  ({path.parent})")
            
            self.lbl_size.setText(f"{self.file_size:,} bytes")
            self.lbl_md5.setText(self.md5_hash)
            self.lbl_type.setText(self.file_type)
            self.lbl_modified.setText(self.mod_time)
            
            self.btn_export_csv.setEnabled(True)
            self.btn_export_report.setEnabled(True)
            
            self.update_analysis()
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"An error occurred while loading file:\n{str(e)}")

    def parse_data(self):
        if self.raw_data is None:
            return None
        
        # Resolve raw bytes to 16,000-byte scope payload
        if len(self.raw_data) == 33024:
            # 256-byte header, followed by 16000-byte payload
            payload = self.raw_data[256:256+16000]
        else:
            payload = self.raw_data[:16000]
            if len(payload) < 16000:
                payload = payload + bytes(16000 - len(payload))
                
        # 16-bit little-endian words
        raw_words = np.frombuffer(payload, dtype=np.dtype('<u2'))
        
        # Reshape to (1000 samples, 8 channels)
        data_2d = raw_words.reshape(1000, 8)
        return data_2d

    # ── STATISTICAL ANALYSIS & REPORT GENERATION ────────────────────────
    def update_analysis(self):
        data_2d = self.parse_data()
        if data_2d is None:
            return
        
        # Read time parameters
        try:
            sample_time = int(self.txt_sample_time.text())
        except ValueError:
            sample_time = 8
            
        try:
            time_unit = float(self.txt_time_unit.text())
        except ValueError:
            time_unit = 125.0
            
        sample_period_sec = (sample_time * time_unit) / 1_000_000.0
        
        # Parse each channel based on data type mapping and override
        self.parsed_channels = {}
        report_sections = []
        
        report_sections.append(f"# TrioScope Scope Analysis Report")
        report_sections.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"Source File: {self.current_file_path}")
        report_sections.append(f"File Type: {self.file_type}")
        report_sections.append(f"File MD5: {self.md5_hash}")
        report_sections.append(f"File Modified Date: {self.mod_time}")
        report_sections.append("-" * 70)
        report_sections.append(f"Configuration settings:")
        report_sections.append(f"  - Sample period: {sample_period_sec*1000:.3f} ms ({sample_time} units × {time_unit} μs)")
        report_sections.append(f"  - Capture duration: {sample_period_sec * 1000:.3f} seconds (1,000 samples)")
        report_sections.append(f"  - Interleaved structure: 8 channels, 1000 samples per channel")
        report_sections.append("-" * 70)
        
        active_channels_count = 0
        skip_channels = 0
        
        for ch_idx in range(8):
            config = self.channel_configs[ch_idx]
            addr = config['address']
            
            if skip_channels > 0:
                skip_channels -= 1
                config['stats_label'].setText(f"Merged into Channel {ch_idx - skip_channels}")
                config['stats_label'].setStyleSheet("font-size: 8.5pt; color: #8E8E93; font-family: Consolas;")
                self.parsed_channels[ch_idx] = {
                    'is_active': False,
                    'name': 'Merged',
                    'desc': 'Merged multi-word',
                    'unit': '',
                    'type_str': ''
                }
                continue
                
            raw_ch = data_2d[:, ch_idx]
            
            # Determine mapping
            mapping_name = "Disabled"
            mapping_desc = "Channel Disabled"
            mapping_unit = ""
            default_dtype = "Uint16"
            
            if addr in DRIVE_VARIABLES:
                mapping_name, mapping_desc, mapping_unit, default_dtype = DRIVE_VARIABLES[addr]
                
            # Determine data type interpretation
            force_type = config['force_type']
            
            # Reconstruction Logic
            if default_dtype == "Int32" and ch_idx + 1 < 8:
                raw_high = data_2d[:, ch_idx+1]
                combined = (raw_high.astype(np.uint32) << 16) | raw_ch.astype(np.uint32)
                ch_data = combined.astype(np.int32).astype(np.float64)
                type_str = "Int32 (Signed, 2 Words)"
                skip_channels = 1
                mapping_name = mapping_name.replace("_L", "").replace("_L1", "")
            elif default_dtype == "Int64" and ch_idx + 3 < 8:
                raw_h1 = data_2d[:, ch_idx+1]
                raw_l2 = data_2d[:, ch_idx+2]
                raw_h2 = data_2d[:, ch_idx+3]
                combined = (
                    (raw_h2.astype(np.uint64) << 48) |
                    (raw_l2.astype(np.uint64) << 32) |
                    (raw_h1.astype(np.uint64) << 16) |
                    raw_ch.astype(np.uint64)
                )
                ch_data = combined.astype(np.int64).astype(np.float64)
                type_str = "Int64 (Signed, 4 Words)"
                skip_channels = 3
                mapping_name = mapping_name.replace("_L1", "")
            else:
                # 16-bit logic
                is_signed = False
                if force_type == "Force Signed (Int16)":
                    is_signed = True
                elif force_type == "Force Unsigned (Uint16)":
                    is_signed = False
                else: # Auto
                    if default_dtype in ("Int16", "Int32", "Int64"):
                        is_signed = True
                        
                if is_signed:
                    ch_data = raw_ch.astype(np.int16).astype(np.float64)
                    type_str = "Int16 (Signed)"
                else:
                    ch_data = raw_ch.astype(np.float64)
                    type_str = "Uint16 (Unsigned)"
                
            # Calculate stats
            nonzero_count = int(np.count_nonzero(ch_data))
            is_active = nonzero_count > 0
            
            if is_active:
                active_channels_count += 1
                ch_min = float(ch_data.min())
                ch_max = float(ch_data.max())
                ch_mean = float(ch_data.mean())
                ch_std = float(ch_data.std())
                ch_p2p = ch_max - ch_min
                
                # Check for dominant frequency (FFT)
                # Remove DC offset first
                ch_detrend = ch_data - ch_mean
                fft_mags = np.abs(np.fft.rfft(ch_detrend))
                fft_freqs = np.fft.rfftfreq(1000, d=sample_period_sec)
                
                # Find dominant peak (ignore very low frequencies/DC leakage)
                peak_idx = np.argmax(fft_mags[1:]) + 1
                peak_freq = fft_freqs[peak_idx]
                peak_mag = fft_mags[peak_idx]
                
                stats_str = (
                    f"Min: {ch_min:+.1f} | Max: {ch_max:+.1f} | Mean: {ch_mean:+.1f} | P2P: {ch_p2p:.1f}"
                )
                
                # Save parsed channel data for plotting
                self.parsed_channels[ch_idx] = {
                    'data': ch_data,
                    'min': ch_min,
                    'max': ch_max,
                    'mean': ch_mean,
                    'std': ch_std,
                    'p2p': ch_p2p,
                    'peak_freq': peak_freq,
                    'peak_mag': peak_mag,
                    'name': mapping_name,
                    'desc': mapping_desc,
                    'unit': mapping_unit,
                    'is_active': True,
                    'type_str': type_str
                }
            else:
                stats_str = "Disabled / All zeros"
                self.parsed_channels[ch_idx] = {
                    'data': ch_data,
                    'is_active': False,
                    'name': mapping_name,
                    'desc': mapping_desc,
                    'unit': mapping_unit,
                    'type_str': type_str
                }
                
            # Update UI labels
            config['stats_label'].setText(stats_str)
            if is_active:
                config['stats_label'].setStyleSheet("font-size: 8.5pt; color: #03DAC6; font-family: Consolas;")
            else:
                config['stats_label'].setStyleSheet("font-size: 8.5pt; color: #8E8E93; font-family: Consolas;")
                
            # Generate Report Text for this channel
            ch_header = f"Channel {ch_idx+1}: {mapping_name} (Address: 0x{addr:04X})"
            report_sections.append(ch_header)
            report_sections.append(f"  Description : {mapping_desc}")
            report_sections.append(f"  Interpretation: {type_str} | Active: {is_active}")
            
            if is_active:
                report_sections.append(f"  Non-zero samples : {nonzero_count} / 1000")
                report_sections.append(f"  Minimum value    : {ch_min:+.3f} {mapping_unit}")
                report_sections.append(f"  Maximum value    : {ch_max:+.3f} {mapping_unit}")
                report_sections.append(f"  Peak-to-Peak     : {ch_p2p:.3f} {mapping_unit}")
                report_sections.append(f"  Mean average     : {ch_mean:.3f} {mapping_unit}")
                report_sections.append(f"  Std Deviation    : {ch_std:.3f} {mapping_unit}")
                if peak_mag > 10.0:  # Only report peak frequency if there is meaningful magnitude
                    report_sections.append(f"  Dominant Frequency: {peak_freq:.2f} Hz (vibration spectral peak)")
            else:
                report_sections.append("  Status: Contains no capture data (all samples are zero).")
            report_sections.append("")
            
        report_sections.append("=" * 70)
        report_sections.append(f"SUMMARY: Captured {active_channels_count} active scope channels out of 8.")
        if active_channels_count > 0:
            report_sections.append("Significant signals detected. Check frequency peaks for mechanical resonance.")
        report_sections.append("=" * 70)
        
        # Display report in the text edit
        self.txt_report.setPlainText("\n".join(report_sections))
        
        # Trigger plot update
        self.update_plots()

    # ── PLOTTING WAVEFORMS ──────────────────────────────────────────────
    def update_plots(self):
        # Clear existing plot layout items
        self.plot_layout_widget.clear()
        self.plot_layout_widget.ci.currentRow = 0
        self.plot_layout_widget.ci.currentCol = 0
        
        if self.raw_data is None:
            # Add an empty plot placeholder
            p = self.plot_layout_widget.addPlot(title="Waveform Viewer (No data loaded)")
            p.showGrid(x=True, y=True, alpha=0.3)
            return

        # Prepare X time base array
        try:
            sample_time = int(self.txt_sample_time.text())
        except ValueError:
            sample_time = 8
            
        try:
            time_unit = float(self.txt_time_unit.text())
        except ValueError:
            time_unit = 125.0
            
        sample_period_sec = (sample_time * time_unit) / 1_000_000.0
        time_array = np.arange(1000) * sample_period_sec
        
        # Collect visible, active channels
        visible_channels = []
        for ch_idx in range(8):
            config = self.channel_configs[ch_idx]
            parsed = self.parsed_channels[ch_idx]
            
            if config['visible'] and parsed['is_active']:
                visible_channels.append((ch_idx, config, parsed))
                
        if not visible_channels:
            p = self.plot_layout_widget.addPlot(title="Select active channels on the right to plot")
            p.showGrid(x=True, y=True, alpha=0.3)
            return
            
        # Draw plots depending on Stacked/Overlay mode
        stacked_mode = self.chk_stacked_plots.isChecked()
        
        if stacked_mode:
            # Create a stack of subplots (linked X axes)
            first_plot = None
            for idx, (ch_idx, config, parsed) in enumerate(visible_channels):
                # Add plot row
                p = self.plot_layout_widget.addPlot(row=idx, col=0)
                p.showGrid(x=True, y=True, alpha=0.3)
                
                # Title & labels
                p.setLabel('left', parsed['name'], units=parsed['unit'], color=config['color'])
                p.getAxis('left').setPen(pg.mkPen(config['color']))
                p.getAxis('left').setTextPen(pg.mkPen(config['color']))
                
                if idx == len(visible_channels) - 1:
                    p.setLabel('bottom', 'Time', units='s', color='#d4d4d4')
                else:
                    p.getAxis('bottom').setStyle(showValues=False)
                    
                # Link X-axes
                if first_plot is None:
                    first_plot = p
                else:
                    p.setXLink(first_plot)
                    
                # Plot data
                pen = pg.mkPen(config['color'], width=2)
                p.plot(time_array, parsed['data'], pen=pen)
                
                # Add a horizontal line at 0 for reference
                p.addLine(y=0, pen=pg.mkPen('#3F3F46', style=Qt.DashLine))
                
                # Add text label with stats in the plot corner
                stats_text = f"Min: {parsed['min']:.1f}  Max: {parsed['max']:.1f}  P2P: {parsed['p2p']:.1f}"
                text_item = pg.TextItem(stats_text, color='#A1A1AA', anchor=(0, 0))
                p.addItem(text_item)
                text_item.setPos(0, parsed['max'] + (parsed['p2p'] * 0.05) if parsed['p2p'] > 0 else 1)
        else:
            # Overlay mode (Single shared plot)
            p = self.plot_layout_widget.addPlot(title="Overlay Waves")
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel('bottom', 'Time', units='s', color='#d4d4d4')
            p.setLabel('left', 'Value', color='#d4d4d4')
            
            # Add legend
            p.addLegend(offset=(10, 10))
            
            for ch_idx, config, parsed in visible_channels:
                pen = pg.mkPen(config['color'], width=1.5)
                p.plot(time_array, parsed['data'], pen=pen, name=f"Ch{ch_idx+1}: {parsed['name']}")
                
            p.addLine(y=0, pen=pg.mkPen('#3F3F46', style=Qt.DashLine))

    # ── EXPORTING DATA ──────────────────────────────────────────────────
    def export_csv(self):
        data_2d = self.parse_data()
        if data_2d is None:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Decoded Data to CSV", "scope_decoded_data.csv", "CSV Files (*.csv)"
        )
        if not file_path:
            return
            
        try:
            # Read time parameters
            try:
                sample_time = int(self.txt_sample_time.text())
            except ValueError:
                sample_time = 8
            try:
                time_unit = float(self.txt_time_unit.text())
            except ValueError:
                time_unit = 125.0
            sample_period_sec = (sample_time * time_unit) / 1_000_000.0
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header row
                header = ["Time(s)"]
                for i in range(8):
                    parsed = self.parsed_channels[i]
                    suffix = f"_{parsed['type_str'].split()[0]}"
                    header.append(f"Ch{i+1}_{parsed['name']}{suffix}")
                writer.writerow(header)
                
                # Write data
                for idx in range(1000):
                    t = idx * sample_period_sec
                    row = [f"{t:.6f}"]
                    for i in range(8):
                        row.append(f"{self.parsed_channels[i]['data'][idx]}")
                    writer.writerow(row)
                    
            QMessageBox.information(self, "Export Complete", f"Data exported successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"An error occurred while exporting CSV:\n{str(e)}")

    def export_report(self):
        if not self.txt_report.toPlainText():
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Scope Analysis Report", "scope_analysis_report.md", "Markdown Files (*.md);;Text Files (*.txt)"
        )
        if not file_path:
            return
            
        try:
            pathlib.Path(file_path).write_text(self.txt_report.toPlainText(), encoding='utf-8')
            QMessageBox.information(self, "Report Saved", f"Analysis report saved successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving report:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = TrioScopeAnalyzer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
