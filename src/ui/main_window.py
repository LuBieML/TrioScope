#!/usr/bin/env python3
"""
Parameter Scope with Oscilloscope-Style UI
PySide6 + pyqtgraph implementation for GPU-accelerated real-time plotting.
"""

import sys
import threading
import logging
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QFrame,
    QScrollArea, QRadioButton, QButtonGroup, QLineEdit, QGroupBox,
    QGridLayout, QSlider,
)
from PySide6.QtCore import Qt, QTimer

import pyqtgraph as pg

# Configure pyqtgraph before creating any widgets
pg.setConfigOptions(
    background='#0A0A0A',
    foreground='#d4d4d4',
    antialias=True,       # AA + width>1 is a Qt software-path; skip for pan/zoom speed
    useOpenGL=True,        # Single shared GraphicsLayoutWidget = 1 GL surface, no context switches
)

# Add src to path when this module is imported directly.
src_path = Path(__file__).resolve().parents[1]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from version import __version__
    from scope.drive_scope_engine import TRIGGER_MODES
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure Trio_UnifiedApi is installed and scope modules are available.")

# Setup detailed logging to a file in the workspace
log_file = "trio_scope.log"
try:
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    ))
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Ensure standard handlers default to INFO (so they aren't flooded with DEBUG)
    for handler in root_logger.handlers:
        if handler != file_handler:
            handler.setLevel(logging.INFO)
except Exception as e:
    print(f"Failed to initialize file logger: {e}", file=sys.stderr)

logger = logging.getLogger(__name__)
logger.info("=========================================")
logger.info("TrioScope app starting. Version: %s", getattr(sys.modules.get('version'), '__version__', 'unknown'))
logger.info("Current working directory: %s", Path.cwd())
logger.info("Python interpreter: %s", sys.executable)
logger.info("Python version: %s", sys.version)
logger.info("=========================================")


from ui.logging_widgets import _LogWindow, _LogBarHandler
from ui.path_3d_view import Path3DView
from ui.theme import DARK_STYLESHEET
from ui.main_window_bindings import bind_main_window_controllers



class ParameterScopeOscilloscope(QMainWindow):
    """Main application with oscilloscope-style UI — pyqtgraph version"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"TrioScope v{__version__} - Parameter Scope")
        self.resize(1400, 900)

        # Trio connection
        self.trio_connection = None
        self.trio_connected = False
        self.scope_engine = None
        self.drive_scope_engine = None
        self.capture_source = 'controller'  # 'controller' or 'drive'

        # Connection management (matching gcode parser pattern)
        self._max_connection_attempts = 3
        self._connection_timeout_seconds = [5, 10, 15]  # Escalating timeouts
        self._disconnect_cooldown_seconds = 1.0
        self._disconnect_cooldown_end = 0.0
        self._state_lock = threading.Lock()
        self._conn_lock = threading.Lock()  # serialize all Trio API calls across threads
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
        self._buffer_capacity = 100_000
        self._time_buffer = np.empty(self._buffer_capacity, dtype=np.float64)
        self._param_buffers = {}
        self._buffer_len = 0
        self._segment_breaks = []  # sample indices where capture restarted
        # (buffer_len, segment_count) last consumed by _on_update_timer —
        # lets the tick skip all work when no new samples have arrived
        self._last_consumed_state = None
        # Incrementally-filled buffers for virtual derived channels
        # {dst_key: (processed_len, np.ndarray)}
        self._virtual_buffers = {}

        # Scrolling window settings
        self.window_duration = 5.0
        self.auto_scroll = True
        self._xy_auto_range = True
        self.lock_x_axis = True

        # Plot settings
        self.grid_alpha = 0.3
        self.line_width = 1  # int width uses Qt's fast cosmetic pen path
        self.plot_bg_color = '#0A0A0A'
        self.plot_mode = 'time'  # 'time', 'xy', 'xyz', 'xyzw'
        self.path_view_scale = 1.0

        # 3D view widget
        self.gl_widget = None

        # Trace controls
        self.traces = []
        self.max_traces = 10

        # Plot items and curves
        self.plot_items = {}    # {key: PlotItem}
        self.curves = {}        # {display_name: PlotDataItem}
        self.ref_curves = {}    # {trace_id: PlotDataItem} — pinned reference traces
        self.stats_texts = {}   # {trace_id: pg.TextItem}

        # Cursor / measurement tool
        self._cursors_enabled = False
        self._cursor_lines_c1 = {}   # {plot_key: InfiniteLine}
        self._cursor_lines_c2 = {}
        self._cursor_pos = {'c1': 0.0, 'c2': 0.0}
        self._cursor_updating = False  # prevent recursive signal loops

        # Settings window
        self._settings_window = None

        # Classical Tuner
        self._tuner_panel = None

        # Measurement panel
        self._measurement_panel = None

        # EtherCAT map window
        self._ethercat_map = None

        # Help window (lazy)
        self._help_window = None

        # Compare/single-trace companion windows.
        self._compare_windows = []
        self._compare_window_counter = 0
        self._trace_window_counter = 0

        bind_main_window_controllers(self)

        self._create_ui()
        self._load_settings()

        # Update timer — drives plot refresh at ~30fps
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_update_timer)
        self._update_timer.setInterval(33)

        # Signal connections
        self.connection_controller.sig_connect_progress.connect(self._on_connect_progress)
        self.connection_controller.sig_connect_result.connect(self._on_connect_result)
        self.capture_controller.sig_capture_progress.connect(self._on_capture_progress)
        self.capture_controller.sig_capture_status.connect(self._on_capture_status)
        self.capture_controller.sig_capture_stopped.connect(self._on_capture_stopped)

        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self.do_connect)

    def _create_ui(self):
        """Create main UI"""
        # Top menu bar (File / View / Help)
        self._create_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        content = QWidget()
        main_layout = QHBoxLayout(content)
        main_layout.setContentsMargins(5, 5, 5, 5)
        outer_layout.addWidget(content, 1)

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

        # Capture source selector: Controller SCOPE or Drive Scope (SDO)
        self._source_label = QLabel("Source:")
        config_layout.addWidget(self._source_label, 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Controller SCOPE", "Drive Scope (SDO)"])
        self.source_combo.setToolTip(
            "Controller SCOPE: captures Trio axis parameters at servo rate\n"
            "Drive Scope (SDO): captures internal drive variables at 125μs rate"
        )
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        config_layout.addWidget(self.source_combo, 0, 1, 1, 2)

        # -- Controller SCOPE config widgets --
        self.ctrl_period_label = QLabel("Sample Period:")
        config_layout.addWidget(self.ctrl_period_label, 1, 0)
        self.period_edit = QLineEdit("1")
        self.period_edit.setFixedWidth(60)
        config_layout.addWidget(self.period_edit, 1, 1)
        self.ctrl_period_unit = QLabel("servocycles")
        config_layout.addWidget(self.ctrl_period_unit, 1, 2)

        self.ctrl_duration_label = QLabel("Duration:")
        config_layout.addWidget(self.ctrl_duration_label, 2, 0)
        self.duration_edit = QLineEdit("5.0")
        self.duration_edit.setFixedWidth(60)
        config_layout.addWidget(self.duration_edit, 2, 1)
        self.ctrl_duration_unit = QLabel("seconds")
        config_layout.addWidget(self.ctrl_duration_unit, 2, 2)

        self.ctrl_mode_label = QLabel("Capture Mode:")
        config_layout.addWidget(self.ctrl_mode_label, 3, 0)
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
        self.ctrl_mode_widget = mode_widget
        config_layout.addWidget(mode_widget, 3, 1, 1, 2)

        self.external_trigger_chk = QCheckBox("External TRIGGER")
        self.external_trigger_chk.setToolTip(
            "Arm SCOPE and wait for a TRIGGER command from the Trio controller program"
        )
        config_layout.addWidget(self.external_trigger_chk, 4, 1, 1, 2)

        # -- Drive Scope config widgets (hidden by default) --
        self.drv_sample_label = QLabel("Capture Duration:")
        self.drv_sample_label.setVisible(False)
        config_layout.addWidget(self.drv_sample_label, 1, 0)
        self.drv_sample_edit = QLineEdit("1.0")
        self.drv_sample_edit.setFixedWidth(80)
        self.drv_sample_edit.setToolTip(
            "Total capture duration in seconds.\n"
            "Sample period = duration / 1000 samples\n"
            "(rounded to nearest 125 μs, min 125 μs)")
        self.drv_sample_edit.textChanged.connect(lambda: self._update_drive_info_label())
        self.drv_sample_edit.setVisible(False)
        config_layout.addWidget(self.drv_sample_edit, 1, 1)
        self.drv_sample_unit = QLabel("s  (res: 1.00 ms)")
        self.drv_sample_unit.setVisible(False)
        config_layout.addWidget(self.drv_sample_unit, 1, 2)

        self.drv_trigger_label = QLabel("Trigger:")
        self.drv_trigger_label.setVisible(False)
        config_layout.addWidget(self.drv_trigger_label, 2, 0)
        self.drv_trigger_combo = QComboBox()
        for mode_id, mode_name in sorted(TRIGGER_MODES.items()):
            self.drv_trigger_combo.addItem(mode_name, mode_id)
        self.drv_trigger_combo.currentIndexChanged.connect(self._on_drive_trigger_changed)
        self.drv_trigger_combo.setVisible(False)
        config_layout.addWidget(self.drv_trigger_combo, 2, 1, 1, 2)

        # Trigger value inputs (shown only for modes that need them)
        self.drv_trig_val_label = QLabel("Trigger Value:")
        self.drv_trig_val_label.setVisible(False)
        config_layout.addWidget(self.drv_trig_val_label, 3, 0)
        self.drv_trig_val1_edit = QLineEdit("0")
        self.drv_trig_val1_edit.setFixedWidth(80)
        self.drv_trig_val1_edit.setToolTip("Trigger threshold value")
        self.drv_trig_val1_edit.setVisible(False)
        config_layout.addWidget(self.drv_trig_val1_edit, 3, 1)
        self.drv_trig_val2_edit = QLineEdit("0")
        self.drv_trig_val2_edit.setFixedWidth(80)
        self.drv_trig_val2_edit.setToolTip("Second threshold (for window trigger)")
        self.drv_trig_val2_edit.setVisible(False)
        config_layout.addWidget(self.drv_trig_val2_edit, 3, 2)

        self.drv_axis_label = QLabel("Drive Axis:")
        self.drv_axis_label.setVisible(False)
        config_layout.addWidget(self.drv_axis_label, 4, 0)
        self.drv_axis_spin = QSpinBox()
        self.drv_axis_spin.setRange(0, 15)
        self.drv_axis_spin.setFixedWidth(60)
        self.drv_axis_spin.setVisible(False)
        config_layout.addWidget(self.drv_axis_spin, 4, 1)

        self.drv_info_label = QLabel("")
        self.drv_info_label.setStyleSheet("color: #03DAC6; font-size: 8pt;")
        self.drv_info_label.setVisible(False)
        # Row 7: rows 5-6 hold the shared Plot Mode selector and path info
        config_layout.addWidget(self.drv_info_label, 7, 0, 1, 3)

        # Plot mode selector (shared)
        config_layout.addWidget(QLabel("Plot Mode:"), 5, 0)
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
        config_layout.addWidget(self.plot_mode_combo, 5, 1, 1, 2)

        self.path_info_label = QLabel("")
        self.path_info_label.setStyleSheet("color: #FFA500; font-size: 8pt;")
        config_layout.addWidget(self.path_info_label, 6, 0, 1, 3)

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

        btn_save_profile = QPushButton("\U0001f4be Save Profile")
        btn_save_profile.setToolTip("Save current traces as a named profile")
        btn_save_profile.clicked.connect(self._show_save_profile_dialog)
        th_layout.addWidget(btn_save_profile)

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

        # Row 1: Clear
        btn_clear = QPushButton("\u239a Clear")
        btn_clear.clicked.connect(self.clear_data)
        ctrl_grid.addWidget(btn_clear, 1, 0, 1, 2)

        left_layout.addLayout(ctrl_grid)

        main_layout.addWidget(left_panel)

        # === RIGHT PANEL ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # 2D Plot area — single shared GraphicsLayoutWidget (1 scene, 1 view).
        # Previously each plot was its own widget in a QSplitter; that cost a
        # full paintEvent per widget per pan tick (N-linked axes = N repaints).
        self.plot_layout_widget = pg.GraphicsLayoutWidget()
        self.plot_layout_widget.setBackground('#0A0A0A')
        self.plot_layout_widget.scene().sigMouseMoved.connect(self._on_main_plot_mouse_moved)
        self.plot_layout_widget.ci.setSpacing(4)
        self.plot_layout_widget.ci.setContentsMargins(0, 0, 0, 0)
        # Kept for backward compat with show/hide and layout code paths
        self.plot_splitter = self.plot_layout_widget
        right_layout.addWidget(self.plot_layout_widget, 1)

        # 3D Plot area (hidden by default)
        self.gl_widget = Path3DView()
        self.gl_widget.viewScaleChanged.connect(self._sync_path_view_scale)
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

        self.btn_compare = QPushButton("\u29c9 Compare")
        self.btn_compare.setFixedWidth(110)
        self.btn_compare.setToolTip(
            "Open a new compare window for 2 or more enabled scopes")
        self.btn_compare.clicked.connect(self._open_compare)
        status_layout.addWidget(self.btn_compare)

        self.btn_measurements = QPushButton("\u25a3 Measurements")
        self.btn_measurements.setFixedWidth(130)
        self.btn_measurements.setToolTip("Open the live measurement window")
        self.btn_measurements.clicked.connect(self._toggle_measurement_panel)
        status_layout.addWidget(self.btn_measurements)

        self.btn_report = QPushButton("HTML Report")
        self.btn_report.setFixedWidth(115)
        self.btn_report.setToolTip(
            "Create a self-contained commissioning report from the current capture")
        self.btn_report.clicked.connect(self.export_html_report)
        status_layout.addWidget(self.btn_report)

        self.path_scale_control = QWidget()
        path_scale_layout = QHBoxLayout(self.path_scale_control)
        path_scale_layout.setContentsMargins(5, 0, 0, 0)
        path_scale_layout.setSpacing(5)
        path_scale_label = QLabel("3D Scale")
        path_scale_label.setToolTip("Scale the complete 3D view: grid and path")
        path_scale_layout.addWidget(path_scale_label)
        self.path_view_scale_slider = QSlider(Qt.Horizontal)
        self.path_view_scale_slider.setRange(25, 400)
        self.path_view_scale_slider.setSingleStep(5)
        self.path_view_scale_slider.setPageStep(25)
        self.path_view_scale_slider.setValue(100)
        self.path_view_scale_slider.setFixedWidth(115)
        self.path_view_scale_slider.setToolTip(
            "Scale the complete 3D view from 0.25× to 4.00×"
        )
        self.path_view_scale_value = QLabel("1.00×")
        self.path_view_scale_value.setFixedWidth(44)
        self.path_view_scale_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.path_view_scale_value.setStyleSheet(
            "color: #03DAC6; font-family: Consolas; font-size: 8pt;"
        )
        self.path_view_scale_slider.valueChanged.connect(
            self._on_path_view_scale_changed
        )
        path_scale_layout.addWidget(self.path_view_scale_slider)
        path_scale_layout.addWidget(self.path_view_scale_value)
        self.path_scale_control.setVisible(False)
        status_layout.addWidget(self.path_scale_control)

        status_layout.addStretch()

        self.progress_label = QLabel("")
        status_layout.addWidget(self.progress_label)

        self.sample_counter_label = QLabel("Samples: 0")
        status_layout.addWidget(self.sample_counter_label)

        right_layout.addWidget(status_frame)

        main_layout.addWidget(right_panel, 1)  # stretch factor 1 → plot expands

        # === LOG BAR (single line, full width, bottom) ===
        self._log_window = _LogWindow(self)

        self._log_bar = QLabel("Log")
        self._log_bar.setFixedHeight(20)
        self._log_bar.setContentsMargins(6, 0, 6, 0)
        self._log_bar.setCursor(Qt.CursorShape.PointingHandCursor)
        self._log_bar.setStyleSheet(
            "background-color: #1a1a1a; color: #aaaaaa; font-size: 8pt;"
            " border-top: 1px solid #444;"
        )

        self._log_bar.mousePressEvent = lambda _: self._log_window.show()
        # Wire Python logging into the bar and window
        self._log_handler = _LogBarHandler(self._log_bar, self._log_window)
        self._log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._log_handler)
        outer_layout.addWidget(self._log_bar)

    def closeEvent(self, event):
        self.actions_controller.closeEvent(event)


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
