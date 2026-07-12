# Design system tokens and styles

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

AXIS_PARAMETERS_STYLESHEET = """
QWidget#axisParametersTab { background-color: #24272b; }
QLabel#axisSetupTitle { color: #f0f2f4; font-size: 18pt; font-weight: 600; }
QLabel#axisSetupSubtitle { color: #9aa3ad; font-size: 9pt; }
QFrame#axisAccentRule { background-color: #f3a712; border: none; }
QPushButton#axisAddButton { color: #f3b63f; font-weight: 600; }
QTableWidget#axisParametersTable {
    background-color: #24272b;
    alternate-background-color: #292d32;
    color: #d7dce1;
    border: 1px solid #383e45;
    selection-background-color: #3b3b30;
    selection-color: #ffffff;
}
QTableWidget#axisParametersTable QHeaderView::section {
    background-color: #30353b;
    color: #f3b63f;
    border: none;
    border-right: 1px solid #3c4249;
    border-bottom: 1px solid #f3a712;
    padding: 8px 6px;
    font-weight: 600;
}
QTableWidget#axisParametersTable QDoubleSpinBox,
QTableWidget#axisParametersTable QSpinBox,
QTableWidget#axisParametersTable QComboBox {
    background-color: #202328;
    color: #e1e5e9;
    border: 1px solid #414851;
    border-radius: 3px;
    padding: 4px 6px;
    font-family: 'Consolas';
}
QLabel#axisSetupStatus { color: #9aa3ad; }
QLabel#axisConnectionStatus { color: #d06b64; font-weight: 600; }
"""

MOTION_WINDOW_STYLESHEET = """
QWidget#motionRoot, QWidget#motionRows, QScrollArea#motionScroll {
    background-color: #24272b;
    color: #d7dce1;
}
QLabel#motionTitle {
    color: #f0f2f4;
    font-size: 18pt;
    font-weight: 600;
}
QLabel#motionSubtitle { color: #9aa3ad; font-size: 9pt; }
QFrame#motionAccentRule { background-color: #f3a712; border: none; }
QLabel#motionColumnLabel {
    color: #8e98a3;
    font-family: 'Consolas';
    font-size: 8pt;
    font-weight: 600;
}
QFrame#motionAxisRow {
    background-color: #2b2f34;
    border: 1px solid #3a4148;
    border-left: 3px solid #f3a712;
    border-radius: 4px;
}
QLabel#motionMoveIndex {
    color: #f3b63f;
    font-family: 'Consolas';
    font-size: 11pt;
    font-weight: 700;
}
QComboBox#motionAxisCombo, QFrame#motionAxisRow QDoubleSpinBox {
    background-color: #202328;
    color: #e1e5e9;
    border: 1px solid #414851;
    border-radius: 3px;
    padding: 6px 8px;
    font-family: 'Consolas';
    font-size: 10pt;
}
QComboBox#motionAxisCombo:focus, QFrame#motionAxisRow QDoubleSpinBox:focus {
    border-color: #f3a712;
}
QPushButton#motionAddButton { color: #f3b63f; font-weight: 600; }
QPushButton#motionRemoveButton { color: #c98783; }
QPushButton#motionEnableButton {
    background-color: #4a4130;
    color: #f3c86b;
    border: 1px solid #75633d;
    font-weight: 700;
    padding: 7px 14px;
}
QPushButton#motionEnableButton:hover { background-color: #5a4e37; }
QPushButton#motionEnableButton:checked {
    background-color: #1f6d64;
    color: #e7fffb;
    border: 1px solid #37a698;
}
QPushButton#motionRowStartButton {
    background-color: #287a43;
    color: #ffffff;
    border: 1px solid #3a9d5b;
    font-weight: 700;
    font-size: 13pt;
    padding: 3px;
}
QPushButton#motionRowStartButton:hover { background-color: #329451; }
QPushButton#motionStopButton {
    background-color: #713638;
    color: #f3dada;
    border: 1px solid #9b4b4e;
    font-weight: 700;
    padding: 7px 14px;
}
QPushButton#motionStopButton:hover { background-color: #894144; }
QPushButton#motionEnableButton:disabled,
QPushButton#motionRowStartButton:disabled,
QPushButton#motionStopButton:disabled {
    background-color: #34383d;
    color: #6f777f;
    border-color: #454a50;
}
QLabel#motionStatus { color: #9aa3ad; }
QLabel#motionStatus[error="true"] { color: #d06b64; }
QScrollArea#motionScroll { border: none; }
"""

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
QTabWidget#mainTabs::pane {
    border: none;
    border-top: 1px solid #454a50;
}
QTabWidget#mainTabs QTabBar::tab {
    background-color: #303439;
    color: #aeb5bd;
    border: none;
    border-right: 1px solid #454a50;
    padding: 9px 22px;
    min-width: 90px;
}
QTabWidget#mainTabs QTabBar::tab:hover {
    background-color: #3a3f45;
    color: #ffffff;
}
QTabWidget#mainTabs QTabBar::tab:selected {
    background-color: #24272b;
    color: #f3b63f;
    border-top: 2px solid #f3a712;
    font-weight: bold;
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
