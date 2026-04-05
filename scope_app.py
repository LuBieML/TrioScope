#!/usr/bin/env python3
"""
Parameter Scope with Oscilloscope-Style UI
PySide6 + pyqtgraph implementation for GPU-accelerated real-time plotting.
"""

import sys
import time
import re
import threading
import logging
import csv
import numpy as np
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QFrame,
    QScrollArea, QRadioButton, QButtonGroup, QLineEdit, QGroupBox,
    QDialog, QFileDialog, QMessageBox, QGridLayout,
    QFormLayout, QSizePolicy, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QRectF, QSettings
from PySide6.QtGui import QFont, QColor, QPen, QBrush

import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Configure pyqtgraph before creating any widgets
pg.setConfigOptions(
    background='#0A0A0A',
    foreground='#d4d4d4',
    antialias=True,         # Smooth lines
    useOpenGL=True,         # GPU acceleration
)

# Add src to path for scope engine
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

try:
    import Trio_UnifiedApi as TUA
    from scope.scope_engine import ScopeEngine, ScopeParameterParser
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure Trio_UnifiedApi is installed and scope_engine.py is in src/scope/")

try:
    from ai.analysis_panel import AIAnalysisPanel
except ImportError:
    AIAnalysisPanel = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPE_PARAMETERS = [
    "ACCEL", "ACCEL_FACTOR", "ADDAX_AXIS", "AFF_GAIN", "ATYPE",
    "AXIS_A_OUTPUT", "AXIS_ACCEL", "AXIS_B_OUTPUT", "AXIS_BLENDING",
    "AXIS_D_OUTPUT", "AXIS_DEBUG_A", "AXIS_DEBUG_B", "AXIS_DEBUG_C",
    "AXIS_DEBUG_D", "AXIS_DECEL", "AXIS_DIRECTION", "AXIS_DPOS",
    "AXIS_ENABLE", "AXIS_ENABLE_OVERRIDE", "AXIS_ERROR_COUNT",
    "AXIS_FASTDEC", "AXIS_FS_LIMIT", "AXIS_MAX_ACC", "AXIS_MAX_JERK",
    "AXIS_MAX_JOG_ACC", "AXIS_MAX_JOG_SPEED", "AXIS_MAX_SPEED",
    "AXIS_MAX_TORQUE", "AXIS_MODE", "AXIS_RS_LIMIT", "AXIS_SPEED",
    "AXIS_UNITS", "AXIS_Z_OUTPUT", "AXISSTATUS",
    "BACKLASH_DIST", "CHANGE_DIR_LAST", "CLOSE_WIN", "CLOSE_WINB",
    "CLUTCH_RATE", "CORNER_FACTOR", "CORNER_MODE", "CORNER_RADIUS",
    "CORNER_STATE", "CREEP",
    "D_GAIN", "D_ZONE_MAX", "D_ZONE_MIN", "DAC", "DAC_OUT", "DAC_SCALE",
    "DATUM_IN", "DECEL", "DECEL_ANGLE", "DEMAND_EDGES", "DEMAND_SPEED",
    "DISTANCE_TO_SYNC", "DPOS",
    "DRIVE_BRAKE_OUTPUT", "DRIVE_CONTROL", "DRIVE_CONTROLWORD",
    "DRIVE_CURRENT", "DRIVE_CW_MODE", "DRIVE_ENABLE", "DRIVE_FE",
    "DRIVE_FE_LIMIT", "DRIVE_INDEX", "DRIVE_INPUTS", "DRIVE_MODE",
    "DRIVE_MONITOR", "DRIVE_NEG_TORQUE", "DRIVE_PARAMETER",
    "DRIVE_POS_TORQUE", "DRIVE_PROFILE", "DRIVE_REP_DIST",
    "DRIVE_SET_VAL", "DRIVE_STATUS", "DRIVE_TORQUE", "DRIVE_TYPE",
    "DRIVE_VALUE", "DRIVE_VELOCITY",
    "ENCODER", "ENCODER_BITS", "ENCODER_CONTROL", "ENCODER_FILTER",
    "ENCODER_ID", "ENCODER_STATUS", "ENCODER_TURNS",
    "END_DIR_LAST", "END_VELOCITY", "ENDMOVE", "ENDMOVE_BUFFER",
    "ENDMOVE_SPEED",
    "FAST_JOG", "FASTDEC", "FASTJERK", "FE", "FE_LATCH", "FE_LIMIT",
    "FE_LIMIT_MODE", "FE_RANGE", "FEED_OVERRIDE", "FHOLD_IN", "FHSPEED",
    "FORCE_ACCEL", "FORCE_DECEL", "FORCE_DWELL", "FORCE_JERK",
    "FORCE_RAMP", "FORCE_SPEED", "FRAME_REP_DIST", "FRAME_SCALE",
    "FS_LIMIT", "FULL_SP_RADIUS", "FWD_IN", "FWD_JOG", "FWD_START",
    "I_GAIN", "IDLE", "IN_POS", "IN_POS_DIST", "IN_POS_SPEED",
    "INTERP_FACTOR", "INVERT_STEP", "JERK", "JERK_FACTOR", "JOGSPEED",
    "LINK_AXIS", "LOADED", "LOOKAHEAD_FACTOR",
    "MARK", "MARKB", "MERGE", "MICROSTEP", "MOVE_COUNT", "MOVE_COUNT_INC",
    "MOVE_ENDMOVE_SPEED", "MOVE_FORCE_ACCEL", "MOVE_FORCE_DECEL",
    "MOVE_FORCE_RAMP", "MOVE_FORCE_SPEED", "MOVE_PA", "MOVE_PA_CONT",
    "MOVE_PA_IDLE", "MOVE_PB", "MOVE_PB_CONT", "MOVE_PB_IDLE",
    "MOVE_STARTMOVE_SPEED", "MOVELINK_MODIFY", "MOVES_BUFFERED",
    "MPOS", "MSPEED", "MSPEED_FILTER", "MSPEEDF", "MTYPE",
    "NEG_OFFSET", "NTYPE",
    "OFFPOS", "OPEN_WIN", "OPEN_WINB", "OUTLIMIT", "OV_GAIN",
    "P_GAIN", "POS_DELAY", "POS_OFFSET", "POSI_SEQ_DELAY",
    "POSI_SEQ_MODE", "PP_STEP", "PS_ENCODER", "PWM_CYCLE", "PWM_MARK",
    "RAISE_ANGLE", "REG_INPUTS", "REG_POS", "REG_POSB",
    "REGIST_CONTROL", "REGIST_DELAY", "REGIST_SPEED", "REGIST_SPEEDB",
    "REMAIN", "REMAIN_TIME", "REP_DIST", "REP_OPTION", "REV_IN",
    "REV_JOG", "REV_START",
    "ROBOT_DPOS", "ROBOT_FE", "ROBOT_FE_LATCH", "ROBOT_FE_LIMIT",
    "ROBOT_FE_RANGE", "ROBOT_FS_LIMIT", "ROBOT_RS_LIMIT",
    "ROBOT_SP_MODE", "ROBOT_UNITS", "ROBOTSTATUS", "RS_LIMIT",
    "S_REF", "S_REF_OUT", "SERVO", "SPEED", "SPEED_FACTOR", "SPEED_SIGN",
    "SRAMP", "START_DIR_LAST", "STARTMOVE_SPEED", "STOP_ANGLE",
    "STOPPING_DISTANCE", "SYNC_AXIS", "SYNC_CONTROL", "SYNC_DPOS",
    "SYNC_DWELL", "SYNC_FLIGHT", "SYNC_PAUSE", "SYNC_TIME",
    "SYNC_TIMER", "SYNC_WITHDRAW",
    "T_REF", "T_REF_OUT", "TABLE_POINTER", "TANG_DIRECTION",
    "TCP_DPOS", "TCP_FS_LIMIT", "TCP_RS_LIMIT", "TCP_UNITS",
    "TRIOPCTESTVARIAB",
    "UNITS",
    "V_LIMIT", "VECTOR_BUFFERED", "VERIFY", "VFF_GAIN",
    "VP_ACCEL", "VP_DEMAND_ACCEL", "VP_DEMAND_DECEL", "VP_DEMAND_JERK",
    "VP_DEMAND_SPEED", "VP_ERROR", "VP_JERK", "VP_MODE", "VP_OPTIONS",
    "VP_POSITION", "VP_SPEED",
    "WORLD_ACCEL", "WORLD_DECEL", "WORLD_DPOS", "WORLD_FASTDEC",
    "WORLD_FS_LIMIT", "WORLD_JERK", "WORLD_JOGSPEED", "WORLD_RS_LIMIT",
    "WORLD_SPEED", "WORLD_UNITS",
]

TRACE_COLORS = [
    '#03DAC6',  # Teal
    '#FFB74D',  # Orange
    '#64B5F6',  # Blue
    '#F06292',  # Pink
    '#FFF176',  # Yellow
    '#E57373',  # Red
    '#81C784',  # Green
    '#BA68C8',  # Purple
    '#4DD0E1',  # Cyan
    '#AED581',  # Light Green
]

CURSOR_COLORS = {
    'c1': '#FFD700',  # Gold
    'c2': '#00CED1',  # Dark Turquoise
}

# Dark theme stylesheet
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #2e2e2e;
    color: #d4d4d4;
    font-family: 'Segoe UI';
    font-size: 9pt;
}
QGroupBox {
    background-color: #353536;
    border: 1px solid #4b4a4a;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #d4d4d4;
}
QPushButton {
    background-color: #4b4a4a;
    color: #d4d4d4;
    border: 1px solid #606060;
    border-radius: 3px;
    padding: 5px 10px;
    font-size: 9pt;
}
QPushButton:hover { background-color: #5a5a5a; }
QPushButton:pressed { background-color: #666666; }
QPushButton:disabled { color: #666666; background-color: #3a3a3a; border-color: #4a4a4a; }
QPushButton#accent {
    background-color: #2e8b3e;
    color: #ffffff;
    font-weight: bold;
    border: 1px solid #3aad4a;
}
QPushButton#accent:hover { background-color: #38a548; }
QPushButton#accent:pressed { background-color: #267a34; }
QLineEdit, QComboBox, QSpinBox {
    background-color: #4b4a4a;
    color: #d4d4d4;
    border: 1px solid #4b4a4a;
    border-radius: 2px;
    padding: 3px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #4b4a4a;
    color: #d4d4d4;
    selection-background-color: #FFA500;
    selection-color: #000000;
}
QCheckBox {
    color: #d4d4d4;
    spacing: 5px;
}
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #666;
    border-radius: 2px;
    background-color: #2e2e2e;
}
QCheckBox::indicator:checked {
    background-color: #FFA500;
    border-color: #FFA500;
}
QRadioButton {
    color: #d4d4d4;
    spacing: 5px;
}
QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1px solid #666;
    border-radius: 7px;
    background-color: #2e2e2e;
}
QRadioButton::indicator:checked {
    background-color: #FFA500;
    border-color: #FFA500;
}
QScrollArea {
    border: none;
    background-color: #353536;
}
QScrollBar:vertical {
    background-color: #2e2e2e;
    width: 10px;
}
QScrollBar::handle:vertical {
    background-color: #555;
    border-radius: 5px;
    min-height: 20px;
}
QLabel#status_dot {
    font-size: 16pt;
}
QLabel#value_display {
    background-color: #2e2e2e;
    border-radius: 2px;
    padding: 3px 5px;
    font-family: 'Consolas';
    font-size: 10pt;
    font-weight: bold;
}
"""


class ScopeViewBox(pg.ViewBox):
    """Custom ViewBox with oscilloscope-style mouse controls.

    Controls:
        Left-drag        → Pan (X and Y)
        Scroll wheel     → Zoom X (time axis)
        Ctrl + scroll    → Zoom Y (value axis)
        Right-drag       → Rubber-band zoom to region
        Double-click     → Reset view / re-enable auto-scroll
        Middle-click     → Context menu
    """

    # Signal emitted on double-click so the main app can re-enable auto-scroll
    doubleClicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Left-drag = pan (not rubber band)
        self.setMouseMode(pg.ViewBox.PanMode)
        # Disable default right-click context menu (we use right-drag for zoom)
        self.menu = None
        # Rubber band rectangle overlay
        self._rb_rect = None
        # When True, wheel zooms both axes uniformly (for XY mode)
        self.uniform_zoom = False

    def wheelEvent(self, ev, axis=None):
        """Scroll wheel: zoom Y by default, zoom X with Ctrl held.
        In uniform_zoom mode (XY plots), scroll zooms both axes together."""
        if self.uniform_zoom:
            # Zoom both axes together — no axis restriction
            super().wheelEvent(ev, axis=None)
            return
        modifiers = ev.modifiers() if hasattr(ev, 'modifiers') else Qt.NoModifier
        if modifiers == Qt.ControlModifier:
            # Ctrl + scroll → zoom X only (time axis)
            super().wheelEvent(ev, axis=0)
        else:
            # Plain scroll → zoom Y only
            super().wheelEvent(ev, axis=1)

    def mouseDragEvent(self, ev, axis=None):
        """Left-drag = pan, Right-drag = rubber-band zoom."""
        if ev.button() == Qt.RightButton:
            ev.accept()
            if ev.isStart():
                # Create rubber band rectangle
                self._rb_rect = pg.QtWidgets.QGraphicsRectItem(self.childGroup)
                pen = QPen(QColor('#FFA500'), 1, Qt.DashLine)
                pen.setCosmetic(True)
                self._rb_rect.setPen(pen)
                self._rb_rect.setBrush(QBrush(QColor(255, 165, 0, 40)))

            if ev.isFinish():
                # Remove rectangle
                if self._rb_rect is not None:
                    self._rb_rect.setParentItem(None)
                    self._rb_rect = None

                # Zoom to the dragged region
                r = pg.Point(ev.buttonDownPos()) - pg.Point(ev.pos())
                start = self.mapToView(ev.buttonDownPos())
                end = self.mapToView(ev.pos())
                x0, x1 = sorted([start.x(), end.x()])
                y0, y1 = sorted([start.y(), end.y()])
                if abs(r.x()) > 5 or abs(r.y()) > 5:
                    self.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
            else:
                # Update rubber band rectangle during drag
                if self._rb_rect is not None:
                    start = self.mapToView(ev.buttonDownPos())
                    end = self.mapToView(ev.pos())
                    r = QRectF(start, end).normalized()
                    self._rb_rect.setRect(r)
        else:
            # Left-drag: default pan behavior
            super().mouseDragEvent(ev, axis)

    def mouseDoubleClickEvent(self, ev):
        """Double-click: reset view to auto-range and re-enable auto-scroll."""
        if ev.button() == Qt.LeftButton:
            self.enableAutoRange()
            self.doubleClicked.emit()
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)


class TraceControl(QFrame):
    """Individual trace control (like one channel on an oscilloscope)"""

    changed = Signal()

    def __init__(self, trace_number, parent=None):
        super().__init__(parent)
        self.trace_number = trace_number
        self.color = TRACE_COLORS[trace_number % len(TRACE_COLORS)]

        # Colored border
        self.setStyleSheet(f"""
            TraceControl {{
                background-color: #353536;
                border: 2px solid {self.color};
                border-radius: 4px;
            }}
        """)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(3)

        # Row 0: Enable checkbox + parameter dropdown + delete button
        row0 = QHBoxLayout()
        row0.setSpacing(4)

        self.chk_enable = QCheckBox(f"Trace {trace_number + 1}")
        self.chk_enable.setStyleSheet(f"color: {self.color}; font-weight: bold;")
        self.chk_enable.toggled.connect(lambda: self.changed.emit())
        row0.addWidget(self.chk_enable)

        self.param_combo = QComboBox()
        self.param_combo.addItems(SCOPE_PARAMETERS)
        self.param_combo.setCurrentText("MPOS")
        self.param_combo.setMaxVisibleItems(20)
        self.param_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.param_combo.currentTextChanged.connect(lambda: self.changed.emit())
        row0.addWidget(self.param_combo, 1)

        self.btn_delete = QPushButton("\u2715")
        self.btn_delete.setFixedWidth(24)
        self.btn_delete.clicked.connect(self._on_delete)
        row0.addWidget(self.btn_delete)

        vbox.addLayout(row0)

        # Row 1: Axis selector + value display + FFT button
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        row1.addWidget(QLabel("Ax"))
        self.axis_spin = QSpinBox()
        self.axis_spin.setRange(0, 15)
        self.axis_spin.setFixedWidth(28)
        self.axis_spin.setStyleSheet(
            "QSpinBox::up-button { width: 0; } QSpinBox::down-button { width: 0; }"
        )
        self.axis_spin.valueChanged.connect(lambda: self.changed.emit())
        row1.addWidget(self.axis_spin)

        _arrow_style = ("QPushButton { background-color: #4b4a4a; color: #ccc; "
                        "border: 1px solid #606060; border-radius: 2px; "
                        "font-size: 7pt; padding: 0px; }"
                        "QPushButton:pressed { background-color: #666; }")
        btn_ax_down = QPushButton("\u25bc")
        btn_ax_down.setFixedSize(18, 12)
        btn_ax_down.setStyleSheet(_arrow_style)
        btn_ax_down.clicked.connect(lambda: self.axis_spin.setValue(max(0, self.axis_spin.value() - 1)))
        btn_ax_up = QPushButton("\u25b2")
        btn_ax_up.setFixedSize(18, 12)
        btn_ax_up.setStyleSheet(_arrow_style)
        btn_ax_up.clicked.connect(lambda: self.axis_spin.setValue(min(15, self.axis_spin.value() + 1)))

        ax_arrows = QVBoxLayout()
        ax_arrows.setSpacing(1)
        ax_arrows.setContentsMargins(0, 0, 0, 0)
        ax_arrows.addWidget(btn_ax_up)
        ax_arrows.addWidget(btn_ax_down)
        row1.addLayout(ax_arrows)

        self.value_label = QLabel("0.0000")
        self.value_label.setObjectName("value_display")
        self.value_label.setStyleSheet(
            f"color: {self.color}; background-color: #2e2e2e; "
            f"font-family: Consolas; font-size: 9pt; font-weight: bold;"
        )
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setFixedWidth(113)
        row1.addWidget(self.value_label)

        self.btn_fft = QPushButton("FFT")
        self.btn_fft.setCheckable(True)
        self.btn_fft.setFixedSize(36, 22)
        self.btn_fft.setToolTip("Toggle FFT spectrum display for this trace")
        self.btn_fft.setStyleSheet("""
            QPushButton {
                background-color: #4b4a4a;
                color: #888;
                border: 1px solid #606060;
                border-radius: 2px;
                font-size: 8pt;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:checked {
                background-color: #8B4513;
                color: #FFA500;
                border: 1px solid #FFA500;
            }
        """)
        self.btn_fft.toggled.connect(lambda: self.changed.emit())
        row1.addWidget(self.btn_fft)

        vbox.addLayout(row1)

    def _on_delete(self):
        self.setParent(None)
        self.deleteLater()
        self.changed.emit()

    def is_enabled(self):
        return self.chk_enable.isChecked()

    def get_parameter_string(self):
        return f"{self.param_combo.currentText()} AXIS({self.axis_spin.value()})"

    def get_display_name(self):
        return f"{self.param_combo.currentText()}({self.axis_spin.value()})"

    def update_value(self, value):
        self.value_label.setText(f"{value:>10.4f}")

    def is_fft(self):
        return self.btn_fft.isChecked()

    def set_fft(self, enabled):
        self.btn_fft.setChecked(enabled)

    def get_color(self):
        return self.color



class ParameterScopeOscilloscope(QMainWindow):
    """Main application with oscilloscope-style UI — pyqtgraph version"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parameter Scope - Oscilloscope Mode")
        self.resize(1400, 900)

        # Trio connection
        self.trio_connection = None
        self.trio_connected = False
        self.scope_engine = None

        # Connection management (matching gcode parser pattern)
        self._max_connection_attempts = 3
        self._connection_timeout_seconds = [5, 10, 15]  # Escalating timeouts
        self._disconnect_cooldown_seconds = 1.0
        self._disconnect_cooldown_end = 0.0
        self._state_lock = threading.Lock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None

        # Capture state
        self.is_running = False
        self.scope_thread = None
        self._shutting_down = False

        # Data storage — accumulated across all captures
        self.accumulated_data = None
        self.total_samples = 0

        # Thread-safe data buffer
        self._data_lock = threading.Lock()
        self._time_chunks = []
        self._param_chunks = {}
        self._segment_breaks = []  # sample indices where capture restarted

        # Scrolling window settings
        self.window_duration = 5.0
        self.auto_scroll = True
        self._xy_auto_range = True
        self.lock_x_axis = True

        # Plot settings
        self.grid_alpha = 0.3
        self.line_width = 1.8
        self.plot_bg_color = '#0A0A0A'
        self.plot_mode = 'time'  # 'time', 'xy', 'xyz', 'xyzw'

        # 3D view state
        self.gl_widget = None
        self.gl_line_item = None
        self.gl_cursor_item = None
        self.gl_colorbar_items = []  # color bar legend items for XYZW mode
        self._gl_line_segments = None

        # Trace controls
        self.traces = []
        self.max_traces = 10

        # Plot items and curves
        self.plot_items = {}    # {key: PlotItem}
        self.curves = {}        # {display_name: PlotDataItem}
        self.stats_texts = {}   # {trace_id: pg.TextItem}

        # Cursor / measurement tool
        self._cursors_enabled = False
        self._cursor_lines_c1 = {}   # {plot_key: InfiniteLine}
        self._cursor_lines_c2 = {}
        self._cursor_pos = {'c1': 0.0, 'c2': 0.0}
        self._cursor_updating = False  # prevent recursive signal loops

        # Settings window
        self._settings_window = None

        # AI Analysis
        self._ai_panel = None

        # FFT performance caches
        self._fft_cache = {}        # {trace_id: {'key': tuple, 'magnitude': array}}
        self._fft_window_cache = (0, None)  # (n_fft, hanning_window)
        self._fft_dirty = True      # set True when cursor moves; timer picks it up
        self._fft_peak_cache = {}   # {trace_id: (peak_freq, peak_mag)}
        self._fft_max_samples = 16384  # cap FFT size when cursors disabled
        self._last_data_len = 0     # track data growth to skip redundant setData
        self._stats_cache = {}      # {trace_id: (v_min_str, v_max_str)}

        self._create_ui()
        self._load_settings()

        # Update timer — drives plot refresh at ~30fps
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_update_timer)
        self._update_timer.setInterval(33)

    def _create_ui(self):
        """Create main UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # === LEFT PANEL (fixed width) ===
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(360)
        left_panel.setStyleSheet("background-color: #353536;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # -- Connection --
        conn_group = QGroupBox("Connection")
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("IP:"))
        self.ip_edit = QLineEdit("192.168.0.245")
        self.ip_edit.setFixedWidth(100)
        conn_layout.addWidget(self.ip_edit)
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        conn_layout.addWidget(self.btn_connect)
        self.status_dot = QLabel("\u25cf")
        self.status_dot.setObjectName("status_dot")
        self.status_dot.setStyleSheet("color: #f14c4c; font-size: 16pt;")
        conn_layout.addWidget(self.status_dot)
        left_layout.addWidget(conn_group)

        # -- Configuration --
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Sample Period:"), 0, 0)
        self.period_edit = QLineEdit("1")
        self.period_edit.setFixedWidth(60)
        config_layout.addWidget(self.period_edit, 0, 1)
        config_layout.addWidget(QLabel("servocycles"), 0, 2)

        config_layout.addWidget(QLabel("Duration:"), 1, 0)
        self.duration_edit = QLineEdit("5.0")
        self.duration_edit.setFixedWidth(60)
        config_layout.addWidget(self.duration_edit, 1, 1)
        config_layout.addWidget(QLabel("seconds"), 1, 2)

        config_layout.addWidget(QLabel("Capture Mode:"), 2, 0)
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.radio_single = QRadioButton("Single")
        self.radio_continuous = QRadioButton("Continuous")
        self.radio_continuous.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_continuous)
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_continuous)
        config_layout.addWidget(mode_widget, 2, 1, 1, 2)

        # Plot mode selector
        config_layout.addWidget(QLabel("Plot Mode:"), 3, 0)
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(["Time", "XY (2D path)", "XYZ (3D path)", "XYZW (4D path)"])
        self.plot_mode_combo.setToolTip(
            "Time: standard time-based oscilloscope\n"
            "XY: Trace 1→X, Trace 2→Y (2D CNC path)\n"
            "XYZ: Trace 1→X, Trace 2→Y, Trace 3→Z (3D path)\n"
            "XYZW: Trace 1→X, Trace 2→Y, Trace 3→Z, Trace 4→Color (4D path)\n"
            "Use the FFT button on each trace for per-trace spectrum analysis"
        )
        self.plot_mode_combo.currentIndexChanged.connect(self._on_plot_mode_changed)
        config_layout.addWidget(self.plot_mode_combo, 3, 1, 1, 2)

        self.path_info_label = QLabel("")
        self.path_info_label.setStyleSheet("color: #FFA500; font-size: 8pt;")
        config_layout.addWidget(self.path_info_label, 4, 0, 1, 3)

        # Table start (hidden, managed via settings dialog)
        self.table_start_edit = QLineEdit("0")
        self.table_usage_label = QLabel("")
        self.use_end_of_table = True  # default: use end of TABLE

        left_layout.addWidget(config_group)

        # -- Traces header --
        traces_header = QWidget()
        traces_header.setStyleSheet("background-color: #353536;")
        th_layout = QHBoxLayout(traces_header)
        th_layout.setContentsMargins(5, 0, 5, 0)
        th_label = QLabel("Traces")
        th_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        th_layout.addWidget(th_label)
        th_layout.addStretch()
        btn_add = QPushButton("+ Add New Trace")
        btn_add.clicked.connect(self.add_trace)
        th_layout.addWidget(btn_add)
        left_layout.addWidget(traces_header)

        # -- Scrollable traces area --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.traces_container = QWidget()
        self.traces_layout = QVBoxLayout(self.traces_container)
        self.traces_layout.setContentsMargins(2, 2, 2, 2)
        self.traces_layout.setAlignment(Qt.AlignTop)
        self.traces_layout.setSpacing(6)
        scroll.setWidget(self.traces_container)
        left_layout.addWidget(scroll, 1)

        # -- Control buttons --
        ctrl_grid = QGridLayout()
        ctrl_grid.setContentsMargins(0, 0, 0, 0)
        ctrl_grid.setSpacing(4)

        # Row 0: RUN / STOP (full width, prominent)
        self.btn_run = QPushButton("\u25b6  RUN")
        self.btn_run.setObjectName("accent")
        self.btn_run.setFixedHeight(32)
        self.btn_run.clicked.connect(self.start_capture)
        ctrl_grid.addWidget(self.btn_run, 0, 0)

        self.btn_stop = QPushButton("\u25a0  STOP")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_capture)
        ctrl_grid.addWidget(self.btn_stop, 0, 1)

        # Row 1: Clear / Settings
        btn_clear = QPushButton("\u239a Clear")
        btn_clear.clicked.connect(self.clear_data)
        ctrl_grid.addWidget(btn_clear, 1, 0)

        btn_settings = QPushButton("\u2699 Settings")
        btn_settings.clicked.connect(self.open_settings)
        ctrl_grid.addWidget(btn_settings, 1, 1)

        # Row 2: Export / Import
        btn_export = QPushButton("\u2913 Export CSV")
        btn_export.clicked.connect(self.export_to_csv)
        ctrl_grid.addWidget(btn_export, 2, 0)

        btn_import = QPushButton("\u2912 Import CSV")
        btn_import.clicked.connect(self.import_from_csv)
        ctrl_grid.addWidget(btn_import, 2, 1)

        # Row 3: AI Analysis
        btn_ai = QPushButton("\u2728 AI Analysis")
        btn_ai.clicked.connect(self._toggle_ai_panel)
        ctrl_grid.addWidget(btn_ai, 3, 0, 1, 2)

        left_layout.addLayout(ctrl_grid)

        main_layout.addWidget(left_panel)

        # === RIGHT PANEL ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # 2D Plot area — vertical splitter so scope heights are draggable
        self.plot_splitter = QSplitter(Qt.Vertical)
        self.plot_splitter.setHandleWidth(5)
        self.plot_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #353536; }"
        )
        right_layout.addWidget(self.plot_splitter, 1)

        # 3D Plot area (hidden by default)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('#0A0A0A')
        self.gl_widget.setCameraPosition(distance=50)
        self.gl_widget.hide()
        right_layout.addWidget(self.gl_widget, 1)

        # Create initial empty plot
        self._recreate_subplots()

        # -- Cursor readout panel (hidden until cursors toggled on) --
        self.cursor_readout = QFrame()
        self.cursor_readout.setFixedHeight(78)
        self.cursor_readout.setStyleSheet(
            "QFrame { background-color: #1a1a2e; border: 1px solid #4b4a4a;"
            " border-radius: 4px; }"
        )
        readout_inner = QVBoxLayout(self.cursor_readout)
        readout_inner.setContentsMargins(10, 4, 10, 4)
        readout_inner.setSpacing(0)
        self.cursor_readout_label = QLabel("")
        self.cursor_readout_label.setStyleSheet(
            "color: #d4d4d4; font-family: Consolas; font-size: 9pt;"
            " background: transparent; border: none;"
        )
        self.cursor_readout_label.setTextFormat(Qt.RichText)
        readout_inner.addWidget(self.cursor_readout_label)
        self.cursor_readout.hide()
        right_layout.addWidget(self.cursor_readout)

        # -- Status bar --
        status_frame = QWidget()
        status_frame.setStyleSheet("background-color: #353536;")
        status_frame.setFixedHeight(30)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 0, 5, 0)

        self.status_label = QLabel("Not connected")
        status_layout.addWidget(self.status_label)

        self.btn_auto_scroll = QPushButton("\u25b6 Auto-scroll ON")
        self.btn_auto_scroll.setFixedWidth(140)
        self.btn_auto_scroll.clicked.connect(self.toggle_auto_scroll)
        self.btn_auto_scroll.setVisible(False)
        status_layout.addWidget(self.btn_auto_scroll)

        self.chk_lock_x = QCheckBox("Lock X-Axis")
        self.chk_lock_x.setChecked(True)
        self.chk_lock_x.toggled.connect(self._on_lock_x_changed)
        status_layout.addWidget(self.chk_lock_x)

        self.btn_cursors = QPushButton("\u2295 Cursors")
        self.btn_cursors.setFixedWidth(100)
        self.btn_cursors.setCheckable(True)
        self.btn_cursors.toggled.connect(self._toggle_cursors)
        status_layout.addWidget(self.btn_cursors)

        status_layout.addStretch()

        self.progress_label = QLabel("")
        status_layout.addWidget(self.progress_label)

        self.sample_counter_label = QLabel("Samples: 0")
        status_layout.addWidget(self.sample_counter_label)

        right_layout.addWidget(status_frame)

        main_layout.addWidget(right_panel, 1)  # stretch factor 1 → plot expands

    # ─── Plot management ────────────────────────────────────────────

    def _create_scope_plot(self):
        """Create a PlotItem inside its own GraphicsLayoutWidget and add to the splitter."""
        vb = ScopeViewBox()
        vb.doubleClicked.connect(self._on_plot_double_click)
        pi = pg.PlotItem(viewBox=vb)
        pw = pg.GraphicsLayoutWidget()
        pw.setBackground('#0A0A0A')
        pw.addItem(pi, row=0, col=0)
        self.plot_splitter.addWidget(pw)
        return pi

    def _on_plot_double_click(self):
        """Re-enable auto-scroll on double-click during capture."""
        if self.is_running:
            self.auto_scroll = True
            self._xy_auto_range = True
            self._update_auto_scroll_button()

    def _recreate_subplots(self):
        """Recreate subplots — one row per enabled trace for independent Y-scales.
        Each trace gets its own left Y-axis, color-coded. X-axes are linked.
        In XY mode, a single plot shows trace1 vs trace2."""
        # Remove all plot widgets from the splitter
        while self.plot_splitter.count():
            w = self.plot_splitter.widget(0)
            w.setParent(None)
            w.deleteLater()
        self.plot_items = {}
        self.curves = {}
        self.stats_texts = {}
        self._cursor_lines_c1.clear()
        self._cursor_lines_c2.clear()
        self._xy_auto_range = True

        enabled_traces = self.get_enabled_traces()

        if not enabled_traces:
            pi = self._create_scope_plot()
            self._configure_plot(pi, show_xlabel=True)
            self.plot_items['empty'] = pi
            return

        # XY mode: single 2D plot with first two traces as X and Y
        if self.plot_mode == 'xy':
            self._update_path_info_label()
            if len(enabled_traces) < 2:
                pi = self._create_scope_plot()
                self._configure_plot(pi, show_xlabel=True)
                self.plot_items['empty'] = pi
                return

            pi = self._create_scope_plot()
            vb = pi.getViewBox()
            vb.setBackgroundColor(self.plot_bg_color)
            pi.showGrid(x=True, y=True, alpha=self.grid_alpha)
            vb.uniform_zoom = True
            vb.setAspectLocked(True)
            x_trace = enabled_traces[0]
            y_trace = enabled_traces[1]
            pi.setLabel('bottom', x_trace.get_display_name(),
                        color=x_trace.get_color())
            pi.setLabel('left', y_trace.get_display_name(),
                        color=y_trace.get_color())
            pi.getAxis('bottom').setPen(pg.mkPen(x_trace.get_color()))
            pi.getAxis('bottom').setTextPen(pg.mkPen(x_trace.get_color()))
            pi.getAxis('left').setPen(pg.mkPen(y_trace.get_color()))
            pi.getAxis('left').setTextPen(pg.mkPen(y_trace.get_color()))
            pi.disableAutoRange()
            vb.sigRangeChangedManually.connect(self._on_manual_range_change)
            vb.sigRangeChangedManually.connect(self._on_xy_manual_zoom)
            self.plot_items['xy'] = pi
            return

        # XYZ/XYZW mode: 3D OpenGL view
        if self.plot_mode in ('xyz', 'xyzw'):
            self._update_path_info_label()
            self._setup_3d_view()
            return

        num_subplots = len(enabled_traces)

        for row, trace in enumerate(enabled_traces):
            pi = self._create_scope_plot()
            is_last = (row == num_subplots - 1)
            if trace.is_fft():
                self._configure_fft_plot(pi, show_xlabel=is_last)
            else:
                self._configure_plot(pi, show_xlabel=is_last)

            # Color-code left Y-axis to match trace
            color = trace.get_color()
            pi.getAxis('left').setPen(pg.mkPen(color))
            pi.getAxis('left').setTextPen(pg.mkPen(color))

            self.plot_items[id(trace)] = pi

        # Link X-axes for synchronized scrolling (partitioned by time/FFT)
        self._update_x_links()

        # Re-add cursor lines if cursors are active
        if self._cursors_enabled:
            self._add_cursors_to_plots()

    def _configure_plot(self, plot_item, show_xlabel=True):
        """Configure a PlotItem with standard settings"""
        vb = plot_item.getViewBox()
        vb.setBackgroundColor(self.plot_bg_color)
        plot_item.showGrid(x=True, y=True, alpha=self.grid_alpha)
        if show_xlabel:
            plot_item.setLabel('bottom', 'Time (seconds)', color='#d4d4d4')
        else:
            plot_item.setLabel('bottom', '')
        plot_item.setLabel('left', '')

        # Y auto-range follows visible data
        plot_item.enableAutoRange(axis='y', enable=True)
        plot_item.setAutoVisible(y=True)

        # Disable auto-scroll when user manually interacts
        vb.sigRangeChangedManually.connect(self._on_manual_range_change)

        # Reposition stats text and update dot visibility when view range changes
        vb.sigRangeChanged.connect(self._reposition_stats_texts)
        vb.sigRangeChanged.connect(self._update_curve_detail)

    def _configure_fft_plot(self, plot_item, show_xlabel=True):
        """Configure a PlotItem for FFT spectrum display."""
        vb = plot_item.getViewBox()
        vb.setBackgroundColor(self.plot_bg_color)
        plot_item.showGrid(x=True, y=True, alpha=self.grid_alpha)
        if show_xlabel:
            plot_item.setLabel('bottom', 'Frequency (Hz)', color='#d4d4d4')
        else:
            plot_item.setLabel('bottom', '')
        plot_item.setLabel('left', 'Magnitude', color='#d4d4d4')
        plot_item.enableAutoRange(axis='y', enable=True)
        plot_item.setAutoVisible(y=True)
        vb.sigRangeChanged.connect(self._reposition_stats_texts)

    def _on_manual_range_change(self, _changes):
        """When user manually pans/zooms, disable auto-scroll"""
        if self.is_running and self.auto_scroll:
            self.auto_scroll = False
            self._update_auto_scroll_button()

    def _reposition_stats_texts(self, vb):
        """Reposition stats text items to top-right of the visible area."""
        for trace_id, pi in self.plot_items.items():
            if trace_id in self.stats_texts and pi.getViewBox() is vb:
                view_range = vb.viewRange()
                self.stats_texts[trace_id].setPos(view_range[0][1], view_range[1][1])
                break

    def _update_curve_detail(self, vb):
        """Show/hide sample dots and adjust downsampling based on zoom level."""
        if self.plot_mode in ('xy', 'xyz'):
            return
        view_range = vb.viewRange()
        visible_span = view_range[0][1] - view_range[0][0]
        for trace_id, pi in self.plot_items.items():
            if pi.getViewBox() is not vb or trace_id not in self.curves:
                continue
            # Skip FFT traces — dot detail only applies to time-domain
            trace_obj = next((t for t in self.traces if id(t) == trace_id), None)
            if trace_obj is not None and trace_obj.is_fft():
                continue
            curve = self.curves[trace_id]
            xData = curve.xData
            if xData is None or len(xData) < 2:
                continue
            sample_dt = xData[1] - xData[0]
            if sample_dt <= 0:
                continue
            visible_points = visible_span / sample_dt
            want_dots = visible_points <= 2000
            # Track state to avoid redundant symbol toggling (each call triggers repaint)
            had_dots = getattr(curve, '_has_dots', False)
            if want_dots and not had_dots:
                curve.setDownsampling(ds=1)
                color = curve.opts['pen'].color()
                curve.setSymbol('o')
                curve.setSymbolSize(4)
                curve.setSymbolBrush(color)
                curve.setSymbolPen(None)
                curve._has_dots = True
            elif not want_dots and had_dots:
                curve.setSymbol(None)
                curve.setDownsampling(auto=True, method='peak')
                curve._has_dots = False

    def _on_xy_manual_zoom(self, _changes):
        """When user manually pans/zooms in XY mode, stop auto-fitting."""
        self._xy_auto_range = False
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()

    # ─── Cursor / measurement tool ───────────────────────────────

    def _toggle_cursors(self, checked):
        """Toggle cursor measurement mode on/off."""
        self._cursors_enabled = checked
        if checked:
            self._init_cursor_positions()
            self._add_cursors_to_plots()
            has_time = any(not t.is_fft() for t in self.get_enabled_traces())
            if self.plot_mode == 'time' and has_time:
                self.cursor_readout.show()
            self._update_cursor_readout()
            self.btn_cursors.setStyleSheet(
                "background-color: #3a3a5c; border: 1px solid #FFD700;"
            )
        else:
            self._remove_cursors_from_plots()
            self.cursor_readout.hide()
            self.btn_cursors.setStyleSheet("")
        # Re-render FFT traces with/without cursor window
        if any(t.is_fft() for t in self.get_enabled_traces()) and self.accumulated_data is not None:
            self.curves = {}
            self.stats_texts = {}
            self._recreate_subplots()
            self._render_plots()

    def _init_cursor_positions(self):
        """Set initial cursor positions to 1/3 and 2/3 of the visible range."""
        if self.accumulated_data is not None and len(self.accumulated_data['time']) > 0:
            t = self.accumulated_data['time']
            # Use visible range if available, otherwise full data range
            # Use first time-domain plot for visible range
            first_pi = None
            for tr in self.get_enabled_traces():
                if not tr.is_fft() and id(tr) in self.plot_items:
                    first_pi = self.plot_items[id(tr)]
                    break
            if first_pi and self.plot_mode == 'time':
                vr = first_pi.getViewBox().viewRange()
                t_min, t_max = vr[0]
            else:
                t_min, t_max = float(t[0]), float(t[-1])
            span = t_max - t_min
            self._cursor_pos['c1'] = t_min + span * 0.33
            self._cursor_pos['c2'] = t_min + span * 0.67
        else:
            self._cursor_pos['c1'] = 0.0
            self._cursor_pos['c2'] = 1.0

    def _add_cursors_to_plots(self):
        """Add draggable cursor lines to time-domain subplots."""
        self._remove_cursors_from_plots()
        if self.plot_mode in ('xy', 'xyz'):
            return
        # Only add cursors to time-domain traces (not FFT subplots)
        fft_plot_keys = {id(t) for t in self.get_enabled_traces() if t.is_fft()}
        for plot_key, pi in self.plot_items.items():
            if plot_key in fft_plot_keys:
                continue
            for cid, color, store in [
                ('c1', CURSOR_COLORS['c1'], self._cursor_lines_c1),
                ('c2', CURSOR_COLORS['c2'], self._cursor_lines_c2),
            ]:
                line = pg.InfiniteLine(
                    pos=self._cursor_pos[cid],
                    angle=90,
                    movable=True,
                    pen=pg.mkPen(color, width=1.5, style=Qt.DashLine),
                    hoverPen=pg.mkPen(color, width=2.5),
                    label=cid.upper(),
                    labelOpts={
                        'position': 0.95,
                        'color': color,
                        'fill': pg.mkBrush('#2B2B2BBB'),
                        'movable': True,
                    },
                )
                line.setZValue(1000)
                # Tag line so the callback knows which cursor it is
                line._cursor_id = cid
                line.sigPositionChanged.connect(self._on_cursor_line_moved)
                pi.addItem(line)
                store[plot_key] = line

    def _remove_cursors_from_plots(self):
        """Remove all cursor lines from plots."""
        for store in (self._cursor_lines_c1, self._cursor_lines_c2):
            for plot_key, line in store.items():
                if plot_key in self.plot_items:
                    try:
                        self.plot_items[plot_key].removeItem(line)
                    except Exception:
                        pass
            store.clear()

    def _on_cursor_line_moved(self, line):
        """Called when any cursor line is dragged — sync all lines of same cursor."""
        if self._cursor_updating:
            return
        cid = line._cursor_id
        new_x = line.value()
        self._cursor_pos[cid] = new_x
        self._cursor_updating = True
        try:
            store = self._cursor_lines_c1 if cid == 'c1' else self._cursor_lines_c2
            for pk, other_line in store.items():
                if other_line is not line:
                    other_line.setValue(new_x)
        finally:
            self._cursor_updating = False
        self._update_cursor_readout()
        # Mark FFT dirty so the next timer tick re-renders (avoid per-pixel recompute)
        if any(t.is_fft() for t in self.get_enabled_traces()):
            self._fft_dirty = True
            # If capture is stopped (timer not running), render directly
            if not self._update_timer.isActive() and self.accumulated_data is not None:
                self._render_plots()

    def _get_value_at_time(self, param_name, t):
        """Interpolate parameter value at time t from accumulated data."""
        if self.accumulated_data is None:
            return None
        time_arr = self.accumulated_data['time']
        if len(time_arr) == 0 or param_name not in self.accumulated_data['params']:
            return None
        values = self.accumulated_data['params'][param_name]
        # Clamp to data range
        if t <= time_arr[0]:
            return float(values[0])
        if t >= time_arr[-1]:
            return float(values[-1])
        # Find nearest sample (use searchsorted for efficiency)
        idx = np.searchsorted(time_arr, t)
        # Pick the closer of the two neighboring samples
        if idx > 0 and (idx >= len(time_arr) or
                        abs(time_arr[idx - 1] - t) <= abs(time_arr[idx] - t)):
            idx = idx - 1
        return float(values[idx])

    def _update_cursor_readout(self):
        """Update the cursor readout panel with current cursor values."""
        if not self._cursors_enabled or self.plot_mode in ('xy', 'xyz'):
            self.cursor_readout_label.setText("")
            return

        # Only show readout for time-domain traces
        enabled_traces = [t for t in self.get_enabled_traces() if not t.is_fft()]
        if not enabled_traces:
            self.cursor_readout_label.setText("")
            return
        t1 = self._cursor_pos['c1']
        t2 = self._cursor_pos['c2']
        dt = t2 - t1

        # Build HTML table for readout
        param_cells_c1 = []
        param_cells_c2 = []
        param_cells_delta = []

        for trace in enabled_traces:
            pname = trace.get_display_name()
            color = trace.get_color()
            v1 = self._get_value_at_time(pname, t1)
            v2 = self._get_value_at_time(pname, t2)
            v1_str = f"{v1:.4f}" if v1 is not None else "---"
            v2_str = f"{v2:.4f}" if v2 is not None else "---"
            if v1 is not None and v2 is not None:
                dv = v2 - v1
                dv_str = f"{dv:+.4f}"
            else:
                dv_str = "---"
            param_cells_c1.append(
                f'<td style="padding: 0 12px;">'
                f'<span style="color:{color};">{pname}:</span> {v1_str}</td>'
            )
            param_cells_c2.append(
                f'<td style="padding: 0 12px;">'
                f'<span style="color:{color};">{pname}:</span> {v2_str}</td>'
            )
            param_cells_delta.append(
                f'<td style="padding: 0 12px;">'
                f'<span style="color:{color};">\u0394{pname}:</span> {dv_str}</td>'
            )

        # Frequency from delta-t
        if abs(dt) > 1e-9:
            freq = 1.0 / abs(dt)
            freq_str = f"{freq:.2f} Hz"
        else:
            freq_str = "--- Hz"

        c1_color = CURSOR_COLORS['c1']
        c2_color = CURSOR_COLORS['c2']

        html = (
            '<table cellspacing="0" cellpadding="1" style="font-family: Consolas; font-size: 9pt;">'
            f'<tr><td style="color:{c1_color}; font-weight:bold; padding-right:8px;">C1</td>'
            f'<td style="padding: 0 12px;">t = {t1:.6f} s</td>'
            f'{"".join(param_cells_c1)}</tr>'
            f'<tr><td style="color:{c2_color}; font-weight:bold; padding-right:8px;">C2</td>'
            f'<td style="padding: 0 12px;">t = {t2:.6f} s</td>'
            f'{"".join(param_cells_c2)}</tr>'
            f'<tr><td style="color:#FFA500; font-weight:bold; padding-right:8px;">\u0394</td>'
            f'<td style="padding: 0 12px;">\u0394t = {dt:+.6f} s</td>'
            f'{"".join(param_cells_delta)}'
            f'<td style="padding: 0 12px; color:#FFA500;">f = {freq_str}</td></tr>'
            '</table>'
        )
        self.cursor_readout_label.setText(html)

    def _setup_3d_view(self):
        """Set up 3D OpenGL view with grid and axes for XYZ/XYZW path mode."""
        # Clear previous items
        for item in self.gl_widget.items[:]:
            self.gl_widget.removeItem(item)
        self.gl_line_item = None
        self.gl_cursor_item = None
        self._gl_line_segments = None
        self.gl_colorbar_items = []

        # Add ground grid (will be resized dynamically to fit data)
        self.gl_grid_item = gl.GLGridItem()
        self.gl_grid_item.setSize(100, 100)
        self.gl_grid_item.setSpacing(10, 10)
        self.gl_grid_item.setColor((255, 255, 255, 40))
        self.gl_widget.addItem(self.gl_grid_item)

        # Add axis lines (R=X, G=Y, B=Z)
        axis_len = 50
        for direction, color in [
            ([axis_len, 0, 0], (1, 0.2, 0.2, 0.8)),   # X red
            ([0, axis_len, 0], (0.2, 1, 0.2, 0.8)),    # Y green
            ([0, 0, axis_len], (0.4, 0.4, 1, 0.8)),    # Z blue
        ]:
            pts = np.array([[0, 0, 0], direction], dtype=np.float32)
            line = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True)
            self.gl_widget.addItem(line)

        # Add axis labels
        enabled = self.get_enabled_traces()
        if len(enabled) >= 3:
            labels = [
                (enabled[0].get_display_name(), [axis_len + 3, 0, 0]),
                (enabled[1].get_display_name(), [0, axis_len + 3, 0]),
                (enabled[2].get_display_name(), [0, 0, axis_len + 3]),
            ]
            for text, pos in labels:
                font = QFont('Segoe UI', 8)
                txt = gl.GLTextItem(pos=pos, text=text, color=(212, 212, 212, 200), font=font)
                self.gl_widget.addItem(txt)

        # XYZW mode: add a color bar legend in 3D space
        if self.plot_mode == 'xyzw' and len(enabled) >= 4:
            self._build_3d_colorbar(axis_len, enabled[3].get_display_name())

    def _build_3d_colorbar(self, axis_len, w_label):
        """Build a vertical color bar in 3D space showing the turbo colormap."""
        self.gl_colorbar_items = []
        cmap = pg.colormap.get('turbo')
        num_segments = 30
        bar_x = axis_len + 10  # offset to the right of the scene
        bar_height = axis_len * 0.8
        seg_h = bar_height / num_segments

        for i in range(num_segments):
            t = i / (num_segments - 1)
            color = cmap.map([t], mode='float')[0]
            z_base = i * seg_h
            pts = np.array([
                [bar_x, 0, z_base],
                [bar_x, 0, z_base + seg_h]
            ], dtype=np.float32)
            seg = gl.GLLinePlotItem(pos=pts, color=tuple(color), width=12, antialias=True)
            self.gl_widget.addItem(seg)
            self.gl_colorbar_items.append(seg)

        # Min / Max labels (placeholders, updated with real data later)
        font = QFont('Segoe UI', 7)
        self._cb_min_label = gl.GLTextItem(
            pos=[bar_x + 2, 0, -2], text="min",
            color=(212, 212, 212, 200), font=font)
        self.gl_widget.addItem(self._cb_min_label)
        self.gl_colorbar_items.append(self._cb_min_label)

        self._cb_max_label = gl.GLTextItem(
            pos=[bar_x + 2, 0, bar_height + 1], text="max",
            color=(212, 212, 212, 200), font=font)
        self.gl_widget.addItem(self._cb_max_label)
        self.gl_colorbar_items.append(self._cb_max_label)

        # W parameter name label
        title_font = QFont('Segoe UI', 8)
        self._cb_title_label = gl.GLTextItem(
            pos=[bar_x - 2, 0, bar_height + 5], text=w_label,
            color=(255, 165, 0, 220), font=title_font)
        self.gl_widget.addItem(self._cb_title_label)
        self.gl_colorbar_items.append(self._cb_title_label)

    def _update_colorbar_range(self, w_min, w_max):
        """Update the color bar min/max labels with actual data values."""
        if hasattr(self, '_cb_min_label') and self._cb_min_label is not None:
            self._cb_min_label.setData(text=f"{w_min:.2f}")
        if hasattr(self, '_cb_max_label') and self._cb_max_label is not None:
            self._cb_max_label.setData(text=f"{w_max:.2f}")

    def _update_x_links(self):
        """Link or unlink X-axes across subplots, partitioned by time/FFT."""
        # Unlink everything first
        for pi in self.plot_items.values():
            pi.setXLink(None)

        if not self.lock_x_axis:
            return

        # Partition into time and FFT groups
        time_plots = []
        fft_plots = []
        for trace in self.get_enabled_traces():
            tid = id(trace)
            if tid in self.plot_items:
                if trace.is_fft():
                    fft_plots.append(self.plot_items[tid])
                else:
                    time_plots.append(self.plot_items[tid])

        # Link within each group
        for group in (time_plots, fft_plots):
            if len(group) > 1:
                for pi in group[1:]:
                    pi.setXLink(group[0])

    def _on_lock_x_changed(self, checked):
        self.lock_x_axis = checked
        self._update_x_links()

    def _on_plot_mode_changed(self, index):
        modes = ['time', 'xy', 'xyz', 'xyzw']
        self.plot_mode = modes[index]
        self._update_path_info_label()
        self.curves = {}
        self.stats_texts = {}

        # Show/hide 2D vs 3D widgets
        if self.plot_mode in ('xyz', 'xyzw'):
            self.plot_splitter.hide()
            self.gl_widget.show()
        else:
            self.gl_widget.hide()
            self.plot_splitter.show()

        # Cursor readout visible when any trace is in time mode
        if self._cursors_enabled:
            has_time = self.plot_mode == 'time' and any(
                not t.is_fft() for t in self.get_enabled_traces())
            self.cursor_readout.setVisible(has_time)

        self._recreate_subplots()

        # Re-render static data in new mode (e.g. switching to FFT after capture)
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()

    def _update_path_info_label(self):
        """Update path mode info label showing axis assignments."""
        if self.plot_mode == 'time':
            fft_count = sum(1 for t in self.get_enabled_traces() if t.is_fft())
            if fft_count > 0:
                self.path_info_label.setText(
                    f"FFT: {fft_count} trace(s) in spectrum mode")
            else:
                self.path_info_label.setText("")
            return
        enabled = self.get_enabled_traces()
        if self.plot_mode == 'xy':
            if len(enabled) < 2:
                self.path_info_label.setText("Enable at least 2 traces for XY")
            else:
                self.path_info_label.setText(
                    f"X: {enabled[0].get_display_name()}  |  "
                    f"Y: {enabled[1].get_display_name()}")
        elif self.plot_mode == 'xyz':
            if len(enabled) < 3:
                self.path_info_label.setText("Enable at least 3 traces for XYZ")
            else:
                self.path_info_label.setText(
                    f"X: {enabled[0].get_display_name()}  |  "
                    f"Y: {enabled[1].get_display_name()}  |  "
                    f"Z: {enabled[2].get_display_name()}")
        elif self.plot_mode == 'xyzw':
            if len(enabled) < 4:
                self.path_info_label.setText("Enable at least 4 traces for XYZW")
            else:
                self.path_info_label.setText(
                    f"X: {enabled[0].get_display_name()}  |  "
                    f"Y: {enabled[1].get_display_name()}  |  "
                    f"Z: {enabled[2].get_display_name()}  |  "
                    f"W(color): {enabled[3].get_display_name()}")

    # ─── Trace management ───────────────────────────────────────────

    def add_trace(self):
        if len(self.traces) >= self.max_traces:
            QMessageBox.warning(self, "Maximum Traces", f"Maximum {self.max_traces} traces allowed")
            return

        trace = TraceControl(len(self.traces), parent=self.traces_container)
        trace.changed.connect(self.on_trace_changed)
        self.traces_layout.addWidget(trace)
        self.traces.append(trace)

        if len(self.traces) == 1:
            trace.chk_enable.setChecked(True)

    def on_trace_changed(self):
        # Remove destroyed traces
        self.traces = [t for t in self.traces if t.parent() is not None]
        self.curves = {}
        self.stats_texts = {}
        self._fft_cache = {}
        self._fft_peak_cache = {}
        self._stats_cache = {}
        self._update_path_info_label()
        self._recreate_subplots()

        # Re-render captured data when scope is stopped (e.g. toggling FFT)
        if not self.is_running and self.accumulated_data is not None:
            self._render_plots()

    def get_enabled_traces(self):
        return [t for t in self.traces if t.is_enabled()]

    # ─── Connection ─────────────────────────────────────────────────

    def _on_connect_clicked(self):
        if self.trio_connected:
            self.disconnect()
        else:
            self.connect()

    def _event_handler(self, et, ival, sval):
        """Handle Trio API events — ignore during shutdown."""
        if not self.trio_connection or self._shutting_down:
            return
        if et == TUA.EventType.Error or et == TUA.EventType.Warning:
            ival_repr = hex(ival) if isinstance(ival, int) else ival
            logger.error(f"Trio event: ({ival_repr}) {sval}")

    def _watchdog_loop(self):
        """Heartbeat loop — polls VR(66) every 0.5s, 5s timeout on dead socket."""
        while not self._watchdog_stop.wait(0.5):
            if not (self.trio_connection and self.trio_connected):
                continue
            try:
                heartbeat_done = threading.Event()
                heartbeat_error = []

                def _heartbeat():
                    try:
                        self.trio_connection.SetVrValue(66, 1)
                    except Exception as e:
                        heartbeat_error.append(e)
                    finally:
                        heartbeat_done.set()

                t = threading.Thread(target=_heartbeat, name="ScopeWatchdog", daemon=True)
                t.start()
                if not heartbeat_done.wait(timeout=5.0):
                    logger.warning("Watchdog heartbeat timed out after 5s")
                    self._mark_connection_lost()
                    break
                if heartbeat_error:
                    raise heartbeat_error[0]
            except Exception as exc:
                if 'Disconnected' in str(exc) or 'No connection' in str(exc):
                    logger.warning(f"Watchdog detected connection loss: {exc}")
                    self._mark_connection_lost()
                    break
                logger.debug(f"Watchdog heartbeat error: {exc}")

    def _start_watchdog(self):
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="ScopeWatchdog", daemon=True)
        self._watchdog_thread.start()

    def _stop_watchdog(self):
        if not self._watchdog_thread:
            return
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=1.0)
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()

    def _mark_connection_lost(self):
        """Called by watchdog when connection is lost — thread-safe."""
        with self._state_lock:
            if not self.trio_connected:
                return
            self.trio_connected = False
        self._watchdog_stop.set()
        self._disconnect_cooldown_end = time.monotonic() + self._disconnect_cooldown_seconds
        # Schedule UI update on main thread
        QTimer.singleShot(0, self._on_connection_lost_ui)

    def _on_connection_lost_ui(self):
        """Update UI after connection loss — runs on main thread."""
        self.trio_connected = False
        self.trio_connection = None
        self.scope_engine = None
        self.status_dot.setStyleSheet("color: #f14c4c; font-size: 16pt;")
        self.status_label.setText("Connection lost")
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)
        if self.is_running:
            self.stop_capture()

    def _attempt_connection_with_timeout(self, conn, timeout_seconds):
        """Open connection with timeout. Returns True/False."""
        connection_opened = threading.Event()
        connection_error = []

        def _open():
            try:
                conn.OpenConnection()
                connection_opened.set()
            except Exception as e:
                connection_error.append(e)
                connection_opened.set()

        thread = threading.Thread(target=_open, name="ScopeConnOpen", daemon=True)
        thread.start()

        elapsed = 0.0
        poll_interval = 0.5
        while elapsed < timeout_seconds:
            remaining = min(poll_interval, timeout_seconds - elapsed)
            if connection_opened.wait(timeout=remaining):
                if connection_error:
                    raise connection_error[0]
                return True
            elapsed += remaining

        logger.warning(f"Connection attempt timed out after {timeout_seconds}s")
        return False

    def _cleanup_connection_async(self, conn):
        """Close a connection in a fire-and-forget thread (avoids blocking on dead socket)."""
        if conn is None:
            return
        def _close():
            try:
                conn.CloseConnection()
            except Exception:
                pass
        threading.Thread(target=_close, name="ScopeCloseCleanup", daemon=True).start()

    def connect(self):
        ip = self.ip_edit.text()

        # Check disconnect cooldown
        cooldown_remaining = max(0.0, self._disconnect_cooldown_end - time.monotonic())
        if cooldown_remaining > 0:
            self.status_label.setText(f"Please wait {cooldown_remaining:.1f}s before reconnecting")
            return

        self.btn_connect.setEnabled(False)
        self.status_label.setText(f"Connecting to {ip}...")

        def _connect_worker():
            """Retry loop with escalating timeouts — runs in background thread."""
            for attempt in range(self._max_connection_attempts):
                timeout = self._connection_timeout_seconds[attempt]
                attempt_label = f"Attempt {attempt + 1}/{self._max_connection_attempts}"
                self._pending_connect_progress = f"{attempt_label} (timeout: {timeout}s)"

                try:
                    conn = TUA.TrioConnectionTCP(self._event_handler, ip)
                    succeeded = self._attempt_connection_with_timeout(conn, timeout)

                    if not succeeded:
                        # Timeout — clean up and retry
                        self._cleanup_connection_async(conn)
                        if attempt < self._max_connection_attempts - 1:
                            time.sleep(1.0)
                            continue
                        else:
                            self._pending_connect_result = (
                                None, None, None, ip,
                                TimeoutError(f"Connection timed out after {self._max_connection_attempts} attempts"))
                            return

                    # Verify connection with VR probe
                    try:
                        conn.SetVrValue(66, 0)
                        conn.SetVrValue(66, 1)
                    except Exception as probe_err:
                        logger.warning(f"Connection probe failed: {probe_err}")
                        self._cleanup_connection_async(conn)
                        if attempt < self._max_connection_attempts - 1:
                            time.sleep(1.0)
                            continue
                        else:
                            self._pending_connect_result = (
                                None, None, None, ip,
                                ConnectionError(f"Connection verification failed: {probe_err}"))
                            return

                    # Connection verified — initialize scope engine
                    engine = ScopeEngine(conn)
                    servo_period = engine.read_servo_period()
                    engine.read_table_size()
                    self._pending_connect_result = (conn, engine, servo_period, ip, None)
                    return

                except TUA.TrioConnectionError as e:
                    logger.error(f"TrioConnectionError attempt {attempt + 1}: {e}")
                    if attempt < self._max_connection_attempts - 1:
                        time.sleep(1.0)
                        continue
                    self._pending_connect_result = (None, None, None, ip, e)
                    return

                except Exception as e:
                    logger.error(f"Unexpected error attempt {attempt + 1}: {e}")
                    if attempt < self._max_connection_attempts - 1:
                        time.sleep(1.0)
                        continue
                    self._pending_connect_result = (None, None, None, ip, e)
                    return

        self._pending_connect_result = None
        self._pending_connect_progress = ""
        threading.Thread(target=_connect_worker, daemon=True).start()

        # Poll for result without blocking UI
        def _check_connect():
            if self._pending_connect_result is None:
                # Update status with progress
                if self._pending_connect_progress:
                    self.status_label.setText(f"Connecting to {ip}... {self._pending_connect_progress}")
                QTimer.singleShot(100, _check_connect)
                return
            conn, engine, servo_period, ip_addr, err = self._pending_connect_result
            self._pending_connect_result = None
            if err is not None:
                self.btn_connect.setEnabled(True)
                self.status_label.setText("Connection failed")
                QMessageBox.critical(self, "Connection Error", str(err))
                logger.exception("Connection failed")
            else:
                self.trio_connection = conn
                self.trio_connected = True
                self.scope_engine = engine
                self._start_watchdog()
                self.status_dot.setStyleSheet("color: #00cc00; font-size: 16pt;")
                self.status_label.setText(f"Connected to {ip_addr} (Servo: {servo_period*1000:.1f}ms)")
                self.table_usage_label.setText(f"TABLE size: {engine.tsize}")
                self.btn_connect.setText("Disconnect")
                self.btn_connect.setEnabled(True)

        QTimer.singleShot(100, _check_connect)

    def disconnect(self):
        """Disconnect with proper cleanup — matching gcode parser pattern."""
        self.btn_connect.setEnabled(False)
        self.status_label.setText("Disconnecting...")

        if self.is_running:
            self.stop_capture()

        self._stop_watchdog()
        self._shutting_down = True

        if self.trio_connection:
            # Close with 5s timeout to avoid hanging on dead socket
            close_done = threading.Event()

            def _close_thread():
                try:
                    self.trio_connection.CloseConnection()
                except Exception:
                    pass
                finally:
                    close_done.set()

            t = threading.Thread(target=_close_thread, name="ScopeCloseConn", daemon=True)
            t.start()
            if not close_done.wait(timeout=5.0):
                logger.warning("CloseConnection() timed out after 5s — abandoning")

        self.trio_connection = None
        self.trio_connected = False
        self.scope_engine = None
        self._shutting_down = False
        self._disconnect_cooldown_end = time.monotonic() + self._disconnect_cooldown_seconds

        self.status_dot.setStyleSheet("color: #f14c4c; font-size: 16pt;")
        self.status_label.setText("Disconnected")
        self.table_usage_label.setText("")
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)

    # ─── Capture control ────────────────────────────────────────────

    def start_capture(self):
        if not self.trio_connected:
            QMessageBox.critical(self, "Error", "Not connected")
            return

        enabled_traces = self.get_enabled_traces()
        if not enabled_traces:
            QMessageBox.warning(self, "No Traces", "Enable at least one trace")
            return

        # Deduplicate parameters — Trio SCOPE supports max 8 unique params
        seen = {}
        unique_params = []
        unique_display = []
        for t in enabled_traces:
            ps = t.get_parameter_string()
            if ps not in seen:
                seen[ps] = t.get_display_name()
                unique_params.append(ps)
                unique_display.append(t.get_display_name())

        if len(unique_params) > 8:
            QMessageBox.warning(self, "Too Many Parameters",
                                "Trio SCOPE supports max 8 unique parameters.\n"
                                f"You have {len(unique_params)} unique parameters enabled.\n"
                                "Use duplicate parameters across traces to stay within the limit.")
            return

        # Rebuild subplots and clear old curves
        self.curves = {}
        self.stats_texts = {}
        self._recreate_subplots()

        try:
            period_cycles = int(self.period_edit.text())
            duration_sec = float(self.duration_edit.text())
            num_params = len(unique_params)

            # Calculate table_start
            if self.use_end_of_table:
                sample_period_sec = period_cycles * self.scope_engine.servo_period_sec
                total_samples = int(duration_sec / sample_period_sec)
                total_entries = total_samples * num_params
                table_start = max(0, self.scope_engine.tsize - total_entries)
            else:
                try:
                    table_start = int(self.table_start_edit.text())
                except ValueError:
                    table_start = 0

            if self.radio_continuous.isChecked():
                sample_period_sec = period_cycles * self.scope_engine.servo_period_sec
                available = self.scope_engine.tsize - table_start
                max_samples = available // num_params
                config_duration = max_samples * sample_period_sec
            else:
                config_duration = duration_sec

            self.scope_engine.configure(
                unique_params, unique_display, period_cycles, config_duration, table_start)

            # Clear data
            self.accumulated_data = None
            self.total_samples = 0
            with self._data_lock:
                self._time_chunks = []
                self._param_chunks = {}
                self._segment_breaks = []

            # Update UI
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.is_running = True
            self.auto_scroll = True
            self._update_auto_scroll_button()

            # Start update timer
            self._update_timer.start()

            # Start capture thread
            if self.radio_single.isChecked():
                self.scope_thread = threading.Thread(target=self._scope_single_shot_thread, daemon=True)
            else:
                self.scope_thread = threading.Thread(target=self._scope_continuous_thread, daemon=True)
            self.scope_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Start Error", str(e))
            logger.exception("Start capture failed")

    def _scope_single_shot_thread(self):
        """Single-shot capture — background thread"""
        try:
            period_cycles = int(self.period_edit.text())
            self.scope_engine.start_capture()
            capture_start_time = time.time()
            duration_sec = float(self.duration_edit.text())
            last_sample_idx = 0

            while self.is_running and (time.time() - capture_start_time) < duration_sec:
                batch_data, last_sample_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)
                if batch_data and batch_data['num_samples'] > 0:
                    self._push_data(batch_data)

                elapsed = time.time() - capture_start_time
                pct = (elapsed / duration_sec) * 100
                self._pending_progress = f"Progress: {pct:.1f}%"
                time.sleep(0.010)

            if not self.is_running:
                self.scope_engine.stop_capture()
                return

            # Wait for capture completion
            timeout = 10
            wait_start = time.time()
            while not self.scope_engine.is_capture_complete():
                if time.time() - wait_start > timeout:
                    break
                time.sleep(0.02)

            # Final read — only fetch samples not yet streamed
            final_batch, last_sample_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)
            self.scope_engine.stop_capture()
            if final_batch and final_batch['num_samples'] > 0:
                self._push_data(final_batch)
            self._pending_status = f"Captured {last_sample_idx} samples"

        except Exception as e:
            self._pending_status = f"Error: {e}"
            logger.exception("Single-shot error")
        finally:
            self.is_running = False
            self._pending_stop_ui = True

    def _scope_continuous_thread(self):
        """Continuous capture — background thread.

        Uses TRIGGER(1) for auto-retrigger: the controller automatically
        restarts capture when the buffer fills, eliminating PC-side
        stop/restart gaps and timing compensation.
        """
        try:
            samples_per_param = (
                (self.scope_engine.table_end - self.scope_engine.table_start + 1)
                // self.scope_engine.num_params
            )
            sample_period = self.scope_engine.period_cycles * self.scope_engine.servo_period_sec

            self.scope_engine.start_capture(auto_retrigger=True)
            time.sleep(0.05)
            last_sample_idx = 0
            sample_offset = 0  # cumulative sample offset across wraps
            self._pending_status = "Capturing (continuous)..."

            while self.is_running and self.trio_connected:
                batch_data, new_idx = self.scope_engine.read_new_data(last_sample_idx, max_samples=0)

                if batch_data and batch_data['num_samples'] > 0:
                    # Shift time by cumulative offset from previous scans
                    time_shift = sample_offset * sample_period
                    if time_shift > 0:
                        batch_data['time'] = batch_data['time'] + time_shift
                    self._push_data(batch_data)
                    last_sample_idx = new_idx
                else:
                    # Detect auto-retrigger wrap: SCOPE_POS resets to 0
                    try:
                        scope_pos = self.scope_engine.connection.GetSystemParameter_SCOPE_POS()
                        if scope_pos < last_sample_idx and last_sample_idx > 0:
                            sample_offset += samples_per_param
                            last_sample_idx = 0
                            continue
                    except Exception:
                        pass

                time.sleep(0.010)

            if not self.is_running:
                try:
                    self.scope_engine.stop_capture()
                except Exception:
                    pass

        except Exception as e:
            self._pending_status = f"Error: {e}"
            logger.exception("Continuous error")
        finally:
            self.is_running = False
            self._pending_stop_ui = True

    def _push_data(self, data):
        """Thread-safe: push new data chunk from capture thread"""
        with self._data_lock:
            self._time_chunks.append(data['time'])
            for param_name, values in data['params'].items():
                if param_name not in self._param_chunks:
                    self._param_chunks[param_name] = []
                self._param_chunks[param_name].append(values)

    def _push_segment_break(self):
        """Record current sample count as a segment boundary (capture restart)."""
        with self._data_lock:
            total = sum(len(c) for c in self._time_chunks)
            self._segment_breaks.append(total)

    _pending_progress = ""
    _pending_status = ""
    _pending_stop_ui = False

    def _on_update_timer(self):
        """Main-thread timer: consolidate data and update plots at ~30fps"""
        # Handle pending UI updates from background thread
        if self._pending_progress:
            self.progress_label.setText(self._pending_progress)
            self._pending_progress = ""
        if self._pending_status:
            self.status_label.setText(self._pending_status)
            self._pending_status = ""
        if self._pending_stop_ui:
            self._pending_stop_ui = False
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._update_timer.stop()
            # Final render: show all captured data so panning works immediately
            if self.auto_scroll:
                self.auto_scroll = False
                self._update_auto_scroll_button()
            self._fit_all_data()

        # Consolidate data chunks under lock
        with self._data_lock:
            if not self._time_chunks:
                return
            # Only concatenate if new chunks arrived (more than the 1 consolidated chunk)
            if len(self._time_chunks) == 1:
                all_time = self._time_chunks[0]
                all_params = {k: v[0] for k, v in self._param_chunks.items()}
            else:
                all_time = np.concatenate(self._time_chunks)
                all_params = {}
                for param_name, chunks in self._param_chunks.items():
                    all_params[param_name] = np.concatenate(chunks)
                # Reset to single consolidated chunk
                self._time_chunks = [all_time]
                self._param_chunks = {k: [v] for k, v in all_params.items()}
            seg_breaks = list(self._segment_breaks)

        self.accumulated_data = {
            'time': all_time,
            'num_samples': len(all_time),
            'params': all_params,
            'segment_breaks': seg_breaks,
        }
        self.total_samples = len(all_time)
        self.sample_counter_label.setText(f"Samples: {self.total_samples}")

        # Update plots
        self._render_plots()

        # Update trace value labels
        for trace in self.get_enabled_traces():
            param_name = trace.get_display_name()
            if param_name in all_params and len(all_params[param_name]) > 0:
                trace.update_value(all_params[param_name][-1])

    def _render_plots(self):
        """Update all plot curves with current accumulated data"""
        if self.accumulated_data is None:
            return

        plot_data = self.accumulated_data
        time_arr = plot_data['time']
        if len(time_arr) == 0:
            return

        enabled_traces = self.get_enabled_traces()

        # ── XY Mode ──
        if self.plot_mode == 'xy' and len(enabled_traces) >= 2 and 'xy' in self.plot_items:
            pi = self.plot_items['xy']
            x_name = enabled_traces[0].get_display_name()
            y_name = enabled_traces[1].get_display_name()

            if x_name not in plot_data['params'] or y_name not in plot_data['params']:
                return

            x_vals = plot_data['params'][x_name]
            y_vals = plot_data['params'][y_name]

            # Downsample for rendering if too many points
            max_xy_points = 8000
            n = len(x_vals)
            if n > max_xy_points:
                # Keep every Nth point, but always include the last point
                step = n // max_xy_points
                idx = np.arange(0, n, step)
                if idx[-1] != n - 1:
                    idx = np.append(idx, n - 1)
                x_plot = x_vals[idx]
                y_plot = y_vals[idx]
            else:
                x_plot = x_vals
                y_plot = y_vals

            if 'xy_path' not in self.curves:
                pen = pg.mkPen('#03DAC6', width=self.line_width)
                self.curves['xy_path'] = pi.plot(pen=pen)
            if 'xy_cursor' not in self.curves:
                self.curves['xy_cursor'] = pi.plot(
                    symbol='o', symbolSize=8,
                    symbolBrush='#FF5555', symbolPen=None, pen=None)

            vb = pi.getViewBox()
            if self._xy_auto_range:
                # Update data then fit view to all data
                self.curves['xy_path'].setData(x_plot, y_plot)
                self.curves['xy_cursor'].setData([x_vals[-1]], [y_vals[-1]])
                margin = 0.05
                x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
                y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
                # Make X and Y spans equal so the view is symmetric
                x_span = x_max - x_min
                y_span = y_max - y_min
                max_span = max(x_span, y_span, 1.0)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                half = max_span / 2 * (1 + margin)
                pi.setXRange(x_center - half, x_center + half, padding=0)
                pi.setYRange(y_center - half, y_center + half, padding=0)
            else:
                # User has zoomed/panned — update data without touching the range
                # Block ViewBox auto-range signals so setData cannot move the view
                vb.blockSignals(True)
                try:
                    self.curves['xy_path'].setData(x_plot, y_plot)
                    self.curves['xy_cursor'].setData([x_vals[-1]], [y_vals[-1]])
                finally:
                    vb.blockSignals(False)
            return

        # ── XYZ Mode ──
        if self.plot_mode == 'xyz' and len(enabled_traces) >= 3:
            x_name = enabled_traces[0].get_display_name()
            y_name = enabled_traces[1].get_display_name()
            z_name = enabled_traces[2].get_display_name()

            if not all(n in plot_data['params'] for n in (x_name, y_name, z_name)):
                return

            x_vals = plot_data['params'][x_name]
            y_vals = plot_data['params'][y_name]
            z_vals = plot_data['params'][z_name]

            # Downsample for rendering if too many points
            max_xyz_points = 8000
            n = len(x_vals)
            if n > max_xyz_points:
                step = n // max_xyz_points
                idx = np.arange(0, n, step)
                if idx[-1] != n - 1:
                    idx = np.append(idx, n - 1)
                pts = np.column_stack([x_vals[idx], y_vals[idx], z_vals[idx]]).astype(np.float32)
            else:
                pts = np.column_stack([x_vals, y_vals, z_vals]).astype(np.float32)

            # Single continuous line
            if self.gl_line_item is None:
                self.gl_line_item = gl.GLLinePlotItem(
                    pos=pts, color=(0.012, 0.855, 0.776, 1.0),
                    width=self.line_width, antialias=True)
                self.gl_widget.addItem(self.gl_line_item)
            else:
                self.gl_line_item.setData(pos=pts)

            # Dynamically resize grid to cover data + origin (0,0,0), with margin
            if self.gl_grid_item is not None and len(x_vals) > 0:
                x_min = min(x_vals.min(), 0)
                x_max = max(x_vals.max(), 0)
                y_min = min(y_vals.min(), 0)
                y_max = max(y_vals.max(), 0)
                margin = max(x_max - x_min, y_max - y_min, 10) * 0.2
                x_min -= margin; x_max += margin
                y_min -= margin; y_max += margin
                sx = x_max - x_min
                sy = y_max - y_min
                spacing = max(1, round(max(sx, sy) / 15, -int(np.floor(np.log10(max(max(sx, sy) / 15, 0.1))))))
                cx = (x_max + x_min) / 2
                cy = (y_max + y_min) / 2
                self.gl_grid_item.setSize(sx, sy)
                self.gl_grid_item.setSpacing(spacing, spacing)
                self.gl_grid_item.resetTransform()
                self.gl_grid_item.translate(cx, cy, 0)

            # Current position sphere
            if len(x_vals) > 0:
                if self.gl_cursor_item is None:
                    md = gl.MeshData.sphere(rows=10, cols=10, radius=0.5)
                    self.gl_cursor_item = gl.GLMeshItem(
                        meshdata=md, smooth=True,
                        color=(1.0, 0.33, 0.33, 1.0), shader='balloon')
                    self.gl_widget.addItem(self.gl_cursor_item)
                self.gl_cursor_item.resetTransform()
                self.gl_cursor_item.translate(x_vals[-1], y_vals[-1], z_vals[-1])
            return

        # ── XYZW Mode (4D: color-mapped W) ──
        if self.plot_mode == 'xyzw' and len(enabled_traces) >= 4:
            x_name = enabled_traces[0].get_display_name()
            y_name = enabled_traces[1].get_display_name()
            z_name = enabled_traces[2].get_display_name()
            w_name = enabled_traces[3].get_display_name()

            if not all(n in plot_data['params'] for n in (x_name, y_name, z_name, w_name)):
                return

            x_vals = plot_data['params'][x_name]
            y_vals = plot_data['params'][y_name]
            z_vals = plot_data['params'][z_name]
            w_vals = plot_data['params'][w_name]

            # Downsample for rendering if too many points
            max_xyz_points = 8000
            n = len(x_vals)
            if n > max_xyz_points:
                step = n // max_xyz_points
                idx = np.arange(0, n, step)
                if idx[-1] != n - 1:
                    idx = np.append(idx, n - 1)
                x_ds, y_ds, z_ds, w_ds = x_vals[idx], y_vals[idx], z_vals[idx], w_vals[idx]
            else:
                x_ds, y_ds, z_ds, w_ds = x_vals, y_vals, z_vals, w_vals

            pts = np.column_stack([x_ds, y_ds, z_ds]).astype(np.float32)

            # Normalize W values to [0, 1] for color mapping
            w_min, w_max = float(w_ds.min()), float(w_ds.max())
            if w_max - w_min > 1e-12:
                w_norm = (w_ds - w_min) / (w_max - w_min)
            else:
                w_norm = np.full_like(w_ds, 0.5)

            # Map to turbo colormap (Nx4 RGBA float array)
            cmap = pg.colormap.get('turbo')
            colors = cmap.map(w_norm, mode='float')

            # GLLinePlotItem needs per-segment approach; use scatter for per-point color
            # and a line with averaged segment colors
            if self.gl_line_item is None:
                self.gl_line_item = gl.GLScatterPlotItem(
                    pos=pts, color=colors, size=2.5, pxMode=True)
                self.gl_widget.addItem(self.gl_line_item)
                # Also add a thin connecting line using segment-averaged colors
                if len(pts) > 1:
                    seg_colors = (colors[:-1] + colors[1:]) / 2.0
                    self._gl_line_segments = gl.GLLinePlotItem(
                        pos=pts, color=seg_colors, width=self.line_width, antialias=True)
                    self.gl_widget.addItem(self._gl_line_segments)
            else:
                self.gl_line_item.setData(pos=pts, color=colors, size=2.5)
                if hasattr(self, '_gl_line_segments') and self._gl_line_segments is not None and len(pts) > 1:
                    seg_colors = (colors[:-1] + colors[1:]) / 2.0
                    self._gl_line_segments.setData(pos=pts, color=seg_colors)

            # Dynamically resize grid
            if self.gl_grid_item is not None and len(x_vals) > 0:
                x_lo = min(x_vals.min(), 0)
                x_hi = max(x_vals.max(), 0)
                y_lo = min(y_vals.min(), 0)
                y_hi = max(y_vals.max(), 0)
                gm = max(x_hi - x_lo, y_hi - y_lo, 10) * 0.2
                x_lo -= gm; x_hi += gm
                y_lo -= gm; y_hi += gm
                sx = x_hi - x_lo
                sy = y_hi - y_lo
                spacing = max(1, round(max(sx, sy) / 15,
                              -int(np.floor(np.log10(max(max(sx, sy) / 15, 0.1))))))
                cx = (x_hi + x_lo) / 2
                cy = (y_hi + y_lo) / 2
                self.gl_grid_item.setSize(sx, sy)
                self.gl_grid_item.setSpacing(spacing, spacing)
                self.gl_grid_item.resetTransform()
                self.gl_grid_item.translate(cx, cy, 0)

            # Current position sphere
            if len(x_vals) > 0:
                if self.gl_cursor_item is None:
                    md = gl.MeshData.sphere(rows=10, cols=10, radius=0.5)
                    self.gl_cursor_item = gl.GLMeshItem(
                        meshdata=md, smooth=True,
                        color=(1.0, 0.33, 0.33, 1.0), shader='balloon')
                    self.gl_widget.addItem(self.gl_cursor_item)
                self.gl_cursor_item.resetTransform()
                self.gl_cursor_item.translate(x_vals[-1], y_vals[-1], z_vals[-1])

            # Update color bar range labels
            self._update_colorbar_range(w_min, w_max)
            return

        # ── Per-trace rendering (time or FFT per trace) ──

        # Auto-scroll: apply only to the first time-domain plot
        if self.auto_scroll and self.is_running:
            max_time = time_arr[-1]
            min_time = max(0, max_time - self.window_duration)
            for trace in enabled_traces:
                if not trace.is_fft() and id(trace) in self.plot_items:
                    self.plot_items[id(trace)].setXRange(min_time, max_time, padding=0)
                    break

        # Precompute FFT shared data if any trace needs it
        has_fft_traces = any(t.is_fft() for t in enabled_traces)
        freqs = None
        fft_time = time_arr
        fft_params = plot_data['params']
        fft_cursor_key = None
        if has_fft_traces and len(time_arr) >= 2:
            sample_dt = float(time_arr[1] - time_arr[0])
            if sample_dt > 0:
                # Windowed FFT: if cursors are enabled, use C1\u2013C2 time window
                if self._cursors_enabled:
                    t1 = min(self._cursor_pos['c1'], self._cursor_pos['c2'])
                    t2 = max(self._cursor_pos['c1'], self._cursor_pos['c2'])
                    mask = (time_arr >= t1) & (time_arr <= t2)
                    if np.sum(mask) >= 2:
                        fft_time = time_arr[mask]
                        fft_params = {k: v[mask] for k, v in plot_data['params'].items()}
                        duration = float(fft_time[-1] - fft_time[0])
                        freq_res = 1.0 / duration if duration > 0 else 0
                        self.path_info_label.setText(
                            f"FFT window: {t1:.3f}s \u2192 {t2:.3f}s "
                            f"({duration:.3f}s, {len(fft_time)} pts, "
                            f"\u0394f={freq_res:.2f} Hz)")
                    else:
                        self.path_info_label.setText(
                            "FFT: cursor window too narrow \u2014 using full data")
                else:
                    # Cap FFT size to last N samples for performance
                    if len(fft_time) > self._fft_max_samples:
                        fft_time = fft_time[-self._fft_max_samples:]
                        fft_params = {k: v[-self._fft_max_samples:] for k, v in plot_data['params'].items()}
                    self.path_info_label.setText(
                        f"FFT: last {len(fft_time)} pts (enable cursors to window)")
                n_fft = len(fft_time)
                freqs = np.fft.rfftfreq(n_fft, d=sample_dt)
                # Cache Hanning window — reuse if size unchanged
                if self._fft_window_cache[0] != n_fft:
                    self._fft_window_cache = (n_fft, np.hanning(n_fft))
                fft_cursor_key = (round(self._cursor_pos['c1'], 6),
                                  round(self._cursor_pos['c2'], 6)) if self._cursors_enabled else None
        self._fft_dirty = False

        for trace in enabled_traces:
            trace_id = id(trace)
            if trace_id not in self.plot_items:
                continue

            pi = self.plot_items[trace_id]
            param_name = trace.get_display_name()
            color = trace.get_color()

            if trace.is_fft():
                # ── FFT rendering for this trace ──
                if freqs is None or param_name not in fft_params:
                    continue
                values = fft_params[param_name]
                n_fft = len(fft_time)

                # Check FFT cache — skip recompute if data unchanged
                cache_key = (n_fft, len(time_arr), fft_cursor_key)
                cached = self._fft_cache.get(trace_id)
                if cached and cached['key'] == cache_key:
                    magnitude = cached['magnitude']
                else:
                    # Compute single-sided amplitude spectrum
                    centered = values - np.mean(values)
                    window = self._fft_window_cache[1]
                    windowed = centered * window
                    fft_vals = np.fft.rfft(windowed)
                    window_sum = np.sum(window)
                    magnitude = np.abs(fft_vals) * 2.0 / window_sum
                    magnitude[0] /= 2.0
                    self._fft_cache[trace_id] = {
                        'key': cache_key,
                        'magnitude': magnitude,
                    }

                if trace_id not in self.curves:
                    if pi.legend is None:
                        pi.addLegend(
                            offset=(10, 5),
                            brush=pg.mkBrush('#2B2B2BBB'),
                            pen=pg.mkPen('#606060'),
                            labelTextColor='#d4d4d4',
                            labelTextSize='9pt',
                        )
                    pen = pg.mkPen(color, width=self.line_width)
                    curve = pi.plot(name=param_name, pen=pen)
                    curve.setClipToView(True)
                    curve.setDownsampling(auto=True, method='peak')
                    self.curves[trace_id] = curve

                self.curves[trace_id].setData(freqs, magnitude)

                # Peak frequency annotation (throttled — only update when values change)
                if len(magnitude) > 1:
                    peak_idx = np.argmax(magnitude[1:]) + 1
                    peak_freq = round(float(freqs[peak_idx]), 2)
                    peak_mag = round(float(magnitude[peak_idx]), 4)
                    prev_peak = self._fft_peak_cache.get(trace_id)
                    if prev_peak != (peak_freq, peak_mag):
                        self._fft_peak_cache[trace_id] = (peak_freq, peak_mag)
                        stats_html = (
                            f'<span style="font-family: Segoe UI; font-size: 8pt;">'
                            f'<span style="color: #FFA500;">Peak: {peak_freq:.2f} Hz</span><br>'
                            f'<span style="color: #99FF99;">Mag: {peak_mag:.4f}</span>'
                            f'</span>'
                        )
                        if trace_id not in self.stats_texts:
                            txt = pg.TextItem(anchor=(1, 0))
                            txt.setHtml(stats_html)
                            pi.getViewBox().addItem(txt, ignoreBounds=True)
                            self.stats_texts[trace_id] = txt
                        else:
                            self.stats_texts[trace_id].setHtml(stats_html)
                    if trace_id in self.stats_texts:
                        vb = pi.getViewBox()
                        view_range = vb.viewRange()
                        self.stats_texts[trace_id].setPos(view_range[0][1], view_range[1][1])
            else:
                # ── Time-domain rendering for this trace ──
                if param_name not in plot_data['params']:
                    continue

                values = plot_data['params'][param_name]

                if trace_id not in self.curves:
                    if pi.legend is None:
                        pi.addLegend(
                            offset=(10, 5),
                            brush=pg.mkBrush('#2B2B2BBB'),
                            pen=pg.mkPen('#606060'),
                            labelTextColor='#d4d4d4',
                            labelTextSize='9pt'
                        )
                    pen = pg.mkPen(color, width=self.line_width)
                    curve = pi.plot(name=param_name, pen=pen)
                    curve.setClipToView(True)
                    curve.setDownsampling(auto=True, method='peak')
                    self.curves[trace_id] = curve

                self.curves[trace_id].setData(time_arr, values)

                # Update min/max stats text (throttled — only when display string changes)
                v_min_s = f"{float(np.min(values)):.4f}"
                v_max_s = f"{float(np.max(values)):.4f}"
                prev_stats = self._stats_cache.get(trace_id)
                if prev_stats != (v_min_s, v_max_s):
                    self._stats_cache[trace_id] = (v_min_s, v_max_s)
                    stats_html = (
                        f'<span style="font-family: Segoe UI; font-size: 8pt;">'
                        f'<span style="color: #FF9999;">Min: {v_min_s}</span><br>'
                        f'<span style="color: #99FF99;">Max: {v_max_s}</span>'
                        f'</span>'
                    )
                    if trace_id not in self.stats_texts:
                        txt = pg.TextItem(anchor=(1, 0))
                        txt.setHtml(stats_html)
                        pi.getViewBox().addItem(txt, ignoreBounds=True)
                        self.stats_texts[trace_id] = txt
                    else:
                        self.stats_texts[trace_id].setHtml(stats_html)
                if trace_id in self.stats_texts:
                    vb = pi.getViewBox()
                    view_range = vb.viewRange()
                    self.stats_texts[trace_id].setPos(view_range[0][1], view_range[1][1])

        # Update cursor readout if cursors are active (time-domain traces only)
        if self._cursors_enabled:
            self._update_cursor_readout()

    # ─── Controls ───────────────────────────────────────────────────

    def stop_capture(self):
        self.is_running = False
        if self.scope_engine:
            try:
                self.scope_engine.stop_capture()
            except Exception:
                pass
        self.status_label.setText("Stopped")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_auto_scroll.setVisible(False)

    def toggle_auto_scroll(self):
        self.auto_scroll = not self.auto_scroll
        self._update_auto_scroll_button()

    def _update_auto_scroll_button(self):
        if self.auto_scroll:
            self.btn_auto_scroll.setText("\u25b6 Auto-scroll ON")
            self.btn_auto_scroll.setVisible(False if self.is_running else False)
        else:
            self.btn_auto_scroll.setText("\U0001f512 Auto-scroll OFF")
            self.btn_auto_scroll.setVisible(self.is_running)

    def _fit_all_data(self):
        """Set view range to show all captured data."""
        if self.accumulated_data is None:
            return
        time_arr = self.accumulated_data['time']
        if len(time_arr) == 0:
            return
        if self.plot_mode in ('xy', 'xyz'):
            return
        # Fit only time-domain traces
        for trace in self.get_enabled_traces():
            if not trace.is_fft() and id(trace) in self.plot_items:
                self.plot_items[id(trace)].setXRange(
                    float(time_arr[0]), float(time_arr[-1]), padding=0.02)
                break

    def clear_data(self):
        self.accumulated_data = None
        self.total_samples = 0
        with self._data_lock:
            self._time_chunks = []
            self._param_chunks = {}
            self._segment_breaks = []
        self.curves = {}
        self.stats_texts = {}
        self.gl_line_item = None
        self.gl_cursor_item = None
        self._recreate_subplots()
        self.sample_counter_label.setText("Samples: 0")
        self.status_label.setText("Data cleared")
        if self._cursors_enabled:
            self._update_cursor_readout()

    def export_to_csv(self):
        if self.accumulated_data is None:
            QMessageBox.warning(self, "No Data", "No data to export")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"scope_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            data = self.accumulated_data
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                param_names = list(data['params'].keys())
                writer.writerow(['Time'] + param_names)
                for i in range(len(data['time'])):
                    row = [round(data['time'][i], 6)] + [data['params'][p][i] for p in param_names]
                    writer.writerow(row)
            self.status_label.setText(f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def import_from_csv(self):
        """Import data from a previously exported CSV file.
        Reconstructs traces from the column headers and loads all data."""
        if self.is_running:
            QMessageBox.warning(self, "Running", "Stop capture before importing")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)

                if not header or header[0] != 'Time':
                    QMessageBox.critical(self, "Import Error",
                                         "Invalid CSV format — expected 'Time' as first column")
                    return

                param_names = header[1:]
                if not param_names:
                    QMessageBox.critical(self, "Import Error", "No parameter columns found")
                    return

                # Read all data rows
                rows = list(reader)

            if not rows:
                QMessageBox.warning(self, "Import", "CSV file contains no data rows")
                return

            time_arr = np.array([float(row[0]) for row in rows])
            params = {}
            for col_idx, pname in enumerate(param_names, start=1):
                params[pname] = np.array([float(row[col_idx]) for row in rows])

            # --- Reconfigure traces to match imported columns ---
            # Remove all existing traces
            for t in list(self.traces):
                t.setParent(None)
                t.deleteLater()
            self.traces.clear()

            # Parse column names like "MPOS(0)" → param="MPOS", axis=0
            param_pattern = re.compile(r'^(.+)\((\d+)\)$')

            for pname in param_names:
                m = param_pattern.match(pname)
                self.add_trace()
                trace = self.traces[-1]
                trace.chk_enable.setChecked(True)
                if m:
                    trace.param_combo.setCurrentText(m.group(1))
                    trace.axis_spin.setValue(int(m.group(2)))
                else:
                    # Fallback: use full name as parameter, axis 0
                    trace.param_combo.setCurrentText(pname)
                    trace.axis_spin.setValue(0)

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
                self._time_chunks = [time_arr]
                self._param_chunks = {k: [v] for k, v in params.items()}
                self._segment_breaks = []

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

            self.status_label.setText(
                f"Imported {len(time_arr)} samples, {len(param_names)} params from {Path(path).name}")

        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    # ─── AI Analysis ──────────────────────────────────────────────

    def _toggle_ai_panel(self):
        """Show/hide the AI analysis dock panel."""
        if AIAnalysisPanel is None:
            QMessageBox.warning(self, "AI Analysis",
                                "AI module not available. Check src/ai/ is present.")
            return

        if self._ai_panel is None:
            self._ai_panel = AIAnalysisPanel(self)
            self._ai_panel.set_data_provider(self._get_scope_data_for_ai)
            # Restore saved API key and model
            s = QSettings("TrioScope", "ParameterScope")
            api_key = s.value("ai/api_key", "")
            model = s.value("ai/model", "openai/gpt-4.1-mini")
            if api_key:
                self._ai_panel.set_api_key(api_key)
            self._ai_panel.set_model(model)
            self.addDockWidget(Qt.RightDockWidgetArea, self._ai_panel)
        else:
            self._ai_panel.setVisible(not self._ai_panel.isVisible())

    def _get_scope_data_for_ai(self):
        """Data provider callback for AI panel. Returns (time_arr, params_dict)."""
        if self.accumulated_data is None:
            return None, None
        time_arr = self.accumulated_data.get('time')
        params = self.accumulated_data.get('params')
        if time_arr is None or len(time_arr) == 0:
            return None, None
        return time_arr, params

    def open_settings(self):
        if self._settings_window is not None:
            try:
                self._settings_window.raise_()
                self._settings_window.activateWindow()
                return
            except RuntimeError:
                self._settings_window = None

        dlg = QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setFixedSize(300, 520)
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

        # AI Analysis section
        ai_group = QGroupBox("AI Analysis (NanoGPT)")
        ai_layout = QFormLayout(ai_group)

        s_ai = QSettings("TrioScope", "ParameterScope")
        ai_key_edit = QLineEdit(s_ai.value("ai/api_key", ""))
        ai_key_edit.setEchoMode(QLineEdit.Password)
        ai_key_edit.setPlaceholderText("Enter NanoGPT API key")
        ai_layout.addRow("API Key:", ai_key_edit)

        ai_model_edit = QComboBox()
        if AIAnalysisPanel is not None:
            from ai.nanogpt_client import NanoGPTClient
            ai_model_edit.addItems(NanoGPTClient.AVAILABLE_MODELS)
        ai_model_edit.setCurrentText(s_ai.value("ai/model", "openai/gpt-4.1-mini"))
        ai_model_edit.setEditable(True)
        ai_layout.addRow("Model:", ai_model_edit)
        main_layout.addWidget(ai_group)

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
                # AI settings
                s_save = QSettings("TrioScope", "ParameterScope")
                s_save.setValue("ai/api_key", ai_key_edit.text().strip())
                s_save.setValue("ai/model", ai_model_edit.currentText().strip())
                if self._ai_panel is not None:
                    self._ai_panel.set_api_key(ai_key_edit.text().strip())
                    self._ai_panel.set_model(ai_model_edit.currentText().strip())
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

    def _apply_plot_settings(self):
        """Apply settings to all plots"""
        for pi in self.plot_items.values():
            pi.getViewBox().setBackgroundColor(self.plot_bg_color)
            pi.showGrid(x=True, y=True, alpha=self.grid_alpha)
        # Update line widths
        for curve in self.curves.values():
            pen = curve.opts.get('pen')
            if pen:
                color = pen.color()
                curve.setPen(pg.mkPen(color, width=self.line_width))
        self._update_x_links()

    # ─── Settings persistence ──────────────────────────────────────

    def _load_settings(self):
        """Restore saved settings from QSettings."""
        s = QSettings("TrioScope", "ParameterScope")

        # Connection
        self.ip_edit.setText(s.value("connection/ip", "192.168.0.245"))

        # Configuration
        self.period_edit.setText(s.value("config/sample_period", "1"))
        self.duration_edit.setText(s.value("config/duration", "5.0"))
        self.table_start_edit.setText(s.value("config/table_start", "0"))
        self.use_end_of_table = s.value("config/use_end_of_table", "true") == "true"
        if s.value("config/capture_mode", "continuous") == "single":
            self.radio_single.setChecked(True)
        else:
            self.radio_continuous.setChecked(True)

        # Display / plot settings
        self.plot_mode = s.value("display/plot_mode", "time")
        # Migration: old 'fft' global mode → 'time' with per-trace FFT
        migrate_global_fft = (self.plot_mode == 'fft')
        if migrate_global_fft:
            self.plot_mode = 'time'
        mode_index = {'time': 0, 'xy': 1, 'xyz': 2}.get(self.plot_mode, 0)
        self.plot_mode_combo.setCurrentIndex(mode_index)
        self.window_duration = float(s.value("display/window_duration", 5.0))
        self.lock_x_axis = s.value("display/lock_x_axis", "true") == "true"
        self.chk_lock_x.setChecked(self.lock_x_axis)
        self.line_width = float(s.value("plot/line_width", 1.8))
        self.grid_alpha = float(s.value("plot/grid_alpha", 0.3))
        self.plot_bg_color = s.value("plot/bg_color", "#0A0A0A")

        # Traces
        num_traces = int(s.value("traces/count", 1))
        for i in range(num_traces):
            if i >= len(self.traces):
                self.add_trace()
            t = self.traces[i]
            param = s.value(f"traces/{i}/param", "MPOS")
            axis = int(s.value(f"traces/{i}/axis", 0))
            enabled = s.value(f"traces/{i}/enabled", "true") == "true"
            t.param_combo.setCurrentText(param)
            t.axis_spin.setValue(axis)
            t.chk_enable.setChecked(enabled)
            fft = s.value(f"traces/{i}/fft", "false") == "true" or migrate_global_fft
            t.set_fft(fft)

        # If no traces were saved, add default
        if not self.traces:
            self.add_trace()

    def _save_settings(self):
        """Persist current settings to QSettings."""
        s = QSettings("TrioScope", "ParameterScope")

        # Connection
        s.setValue("connection/ip", self.ip_edit.text())

        # Configuration
        s.setValue("config/sample_period", self.period_edit.text())
        s.setValue("config/duration", self.duration_edit.text())
        s.setValue("config/table_start", self.table_start_edit.text())
        s.setValue("config/use_end_of_table", "true" if self.use_end_of_table else "false")
        s.setValue("config/capture_mode",
                   "single" if self.radio_single.isChecked() else "continuous")

        # Display / plot settings
        s.setValue("display/plot_mode", self.plot_mode)
        s.setValue("display/window_duration", self.window_duration)
        s.setValue("display/lock_x_axis", "true" if self.lock_x_axis else "false")
        s.setValue("plot/line_width", self.line_width)
        s.setValue("plot/grid_alpha", self.grid_alpha)
        s.setValue("plot/bg_color", self.plot_bg_color)

        # Traces
        s.setValue("traces/count", len(self.traces))
        for i, t in enumerate(self.traces):
            s.setValue(f"traces/{i}/param", t.param_combo.currentText())
            s.setValue(f"traces/{i}/axis", t.axis_spin.value())
            s.setValue(f"traces/{i}/enabled",
                       "true" if t.chk_enable.isChecked() else "false")
            s.setValue(f"traces/{i}/fft",
                       "true" if t.is_fft() else "false")

    # ─── Cleanup ────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._save_settings()
        self._shutting_down = True
        self.is_running = False
        self._update_timer.stop()
        self._stop_watchdog()
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


def main():
    # Enable OpenGL multisampling for proper antialiasing
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setSamples(8)  # 4x MSAA
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = ParameterScopeOscilloscope()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
