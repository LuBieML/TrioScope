"""
AI Analysis panel — a dockable Qt widget providing chat-style interaction
with NanoGPT for interpreting scope capture data.

Drive profile section at the top lets the user assign a Trio DX3 or DX4
servo drive to each axis so that the AI receives drive-level tuning context
alongside the scope metrics.
"""

import logging
import numpy as np

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QComboBox, QLabel, QFrame, QSizePolicy,
    QSpinBox, QFormLayout, QScrollArea, QGroupBox,
)
from PySide6.QtCore import Qt, Signal, QObject

from .nanogpt_client import NanoGPTClient
from .signal_metrics import SignalMetrics
from .drive_profile import (
    DriveProfile, DRIVE_TYPES, PARAM_DEFS, COMBO_ATTRS,
    TUNING_MODE_LABELS, TUNING_MODE_VALUES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — updated to include DX3/DX4 drive context awareness
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert motion control and signal analysis engineer embedded in \
TrioScope, an oscilloscope application for **Trio Motion Controllers**.

The system uses Trio MC-series controllers communicating with servo drives \
(typically Trio DX3 or DX4) over EtherCAT. All captured data comes from the \
controller's built-in SCOPE command, which samples servo parameters \
deterministically at the servo rate.

**Control loop architecture (cascade):**
  Controller position loop (Trio)  →  Drive speed loop (DX3/DX4)  →  Drive torque loop
  - Trio controller parameters: P_GAIN, D_GAIN, I_GAIN, VFF_GAIN, OV_GAIN
  - DX4/DX3 drive parameters: Pn102 (speed Kp), Pn103 (speed Ti), \
Pn104 (position Kp), Pn105 (torque filter), Pn106 (load inertia), \
Pn112 (speed feedforward)

**Common Trio scope parameters:**
- MPOS / DPOS — measured vs demand position
- FE — following error (MPOS − DPOS); key indicator of servo quality
- DAC / DAC_OUT — drive torque/velocity demand output
- SPEED / MSPEED — demand vs measured speed
- ENCODER — raw encoder counts
- P_GAIN, I_GAIN, D_GAIN, VFF_GAIN, OV_GAIN — Trio controller servo gains
- FE_LIMIT — maximum following error before axis fault
- DRIVE_CURRENT / DRIVE_TORQUE — actual drive current/torque feedback
- OUTLIMIT — output limit (caps DAC output)

You will receive pre-computed signal metrics AND raw sampled data. \
If a drive profile is provided, also use the Pn parameter values to inform \
your analysis — they explain the drive-level behaviour you are seeing in the \
scope signals.

When suggesting adjustments, distinguish between:
1. **Trio controller gains** (P_GAIN, I_GAIN, D_GAIN, VFF_GAIN) — adjusted \
in Motion Perfect / BASIC program
2. **Drive-level gains** (Pn102, Pn103, Pn104 etc.) — adjusted in the drive \
parameter editor

Be concise and technical. Use specific numbers from the data. \
Identify the symptom, the likely root cause, and which specific parameter \
to change and in which direction.\
"""

# ---------------------------------------------------------------------------
# Quick-action prompt templates
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = (
    "Analyze all captured signals and raw data. Identify any issues with "
    "motion quality, servo tuning, or signal anomalies. Summarise findings "
    "and refer to specific parameter values from both the scope data and the "
    "drive profile (if provided)."
)

TUNE_PROMPT = (
    "Based on the captured data and drive profile (if provided), suggest "
    "servo tuning improvements. Consider both Trio controller gains "
    "(P_GAIN, I_GAIN, D_GAIN, VFF_GAIN) and drive-level parameters "
    "(Pn102 speed loop gain, Pn103 integral time, Pn104 position gain, "
    "Pn112 speed feedforward). For each suggestion state: which parameter, "
    "which direction to adjust, and what improvement to expect."
)


# ---------------------------------------------------------------------------
# Thread-safe Qt signal relay
# ---------------------------------------------------------------------------
class _Signals(QObject):
    chunk_received = Signal(str)
    stream_done = Signal()
    error_occurred = Signal(str)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
class AIAnalysisPanel(QDockWidget):
    """Dockable AI analysis panel with chat interface and per-axis drive profiles."""

    def __init__(self, parent=None):
        super().__init__("AI Analysis", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.setMinimumWidth(380)

        self._client = NanoGPTClient()
        self._signals = _Signals()
        self._signals.chunk_received.connect(self._on_chunk)
        self._signals.stream_done.connect(self._on_stream_done)
        self._signals.error_occurred.connect(self._on_error)

        self._streaming = False
        self._current_response = ""
        self._data_provider = None  # callable → (time_arr, params_dict)

        # Per-axis drive profiles: {axis_int: DriveProfile}
        self._profiles: dict[int, DriveProfile] = {}

        # Widgets populated in _build_ui — referenced later
        self._param_widgets: dict[str, QWidget] = {}   # attr → spinbox/combo
        self._param_frame: QFrame | None = None
        self._axis_combo: QComboBox | None = None
        self._drive_combo: QComboBox | None = None
        self._read_btn: QPushButton | None = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Model selector ──────────────────────────────────────────────────
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(NanoGPTClient.AVAILABLE_MODELS)
        self.model_combo.setCurrentText(self._client.model)
        self.model_combo.currentTextChanged.connect(self._client.set_model)
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        # ── Drive profile section ───────────────────────────────────────────
        layout.addWidget(self._build_drive_profile_section())

        # ── Quick action buttons (Analyze / Tune only) ──────────────────────
        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)

        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setFixedHeight(28)
        self.btn_analyze.setToolTip(ANALYZE_PROMPT)
        self.btn_analyze.clicked.connect(lambda: self._send_query(ANALYZE_PROMPT))
        actions_row.addWidget(self.btn_analyze)

        self.btn_tune = QPushButton("Tune")
        self.btn_tune.setFixedHeight(28)
        self.btn_tune.setToolTip(TUNE_PROMPT)
        self.btn_tune.clicked.connect(lambda: self._send_query(TUNE_PROMPT))
        actions_row.addWidget(self.btn_tune)

        layout.addLayout(actions_row)

        # ── Chat display ────────────────────────────────────────────────────
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit { background-color: #1a1a2e; color: #d4d4d4;"
            " font-family: Consolas, monospace; font-size: 9pt;"
            " border: 1px solid #4b4a4a; border-radius: 3px; }"
        )
        self.chat_display.setPlaceholderText(
            "Set a drive profile above (optional), capture scope data, "
            "then click Analyze or Tune.\n\n"
            "You can also type a custom question below."
        )
        layout.addWidget(self.chat_display, 1)

        # ── Input row ───────────────────────────────────────────────────────
        input_row = QHBoxLayout()
        input_row.setSpacing(3)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Ask about the captured data...")
        self.input_edit.setStyleSheet(
            "QLineEdit { background-color: #2a2a3e; color: #d4d4d4;"
            " border: 1px solid #4b4a4a; border-radius: 3px; padding: 4px; }"
        )
        self.input_edit.returnPressed.connect(self._on_send_clicked)
        input_row.addWidget(self.input_edit, 1)

        self.btn_send = QPushButton("Send")
        self.btn_send.setFixedWidth(60)
        self.btn_send.setFixedHeight(28)
        self.btn_send.clicked.connect(self._on_send_clicked)
        input_row.addWidget(self.btn_send)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedWidth(50)
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.clicked.connect(self._clear_chat)
        input_row.addWidget(self.btn_clear)

        layout.addLayout(input_row)

        # ── Status label ─────────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(self.status_label)

        self.setWidget(container)

    def _build_drive_profile_section(self) -> QWidget:
        """Build the collapsible drive profile configurator."""
        group = QGroupBox("Drive Profile")
        group.setStyleSheet(
            "QGroupBox { color: #aaa; font-size: 8pt; border: 1px solid #444;"
            " border-radius: 3px; margin-top: 6px; padding-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
        )
        outer = QVBoxLayout(group)
        outer.setContentsMargins(4, 2, 4, 4)
        outer.setSpacing(3)

        # ── Axis + Drive type row ──────────────────────────────────────────
        selector_row = QHBoxLayout()
        selector_row.setSpacing(4)

        selector_row.addWidget(QLabel("Axis:"))
        self._axis_combo = QComboBox()
        self._axis_combo.setFixedWidth(50)
        for i in range(16):
            self._axis_combo.addItem(str(i))
        self._axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        selector_row.addWidget(self._axis_combo)

        selector_row.addWidget(QLabel("Drive:"))
        self._drive_combo = QComboBox()
        self._drive_combo.addItems(DRIVE_TYPES)
        self._drive_combo.setFixedWidth(80)
        self._drive_combo.currentTextChanged.connect(self._on_drive_type_changed)
        selector_row.addWidget(self._drive_combo)

        selector_row.addStretch()

        # Read from Drive button — placeholder for future EtherCAT auto-read
        self._read_btn = QPushButton("Read from Drive")
        self._read_btn.setFixedHeight(22)
        self._read_btn.setEnabled(False)
        self._read_btn.setToolTip(
            "Future feature: read Pn parameters directly from the drive "
            "via EtherCAT CoE SDO (object IDs 0x31C8–0x31D4)."
        )
        selector_row.addWidget(self._read_btn)

        outer.addLayout(selector_row)

        # ── Parameter fields (shown only for DX3 / DX4) ────────────────────
        self._param_frame = QFrame()
        self._param_frame.setVisible(False)
        param_layout = QFormLayout(self._param_frame)
        param_layout.setContentsMargins(0, 2, 0, 0)
        param_layout.setSpacing(2)
        param_layout.setLabelAlignment(Qt.AlignLeft)

        label_style = "color: #ccc; font-size: 8pt;"
        spin_style = (
            "QSpinBox { background: #2a2a3e; color: #d4d4d4;"
            " border: 1px solid #555; border-radius: 2px; padding: 1px 3px;"
            " font-size: 8pt;"
            " QSpinBox::up-button { width: 0; } QSpinBox::down-button { width: 0; } }"
        )
        combo_style = (
            "QComboBox { background: #2a2a3e; color: #d4d4d4;"
            " border: 1px solid #555; border-radius: 2px; padding: 1px 3px;"
            " font-size: 8pt; }"
        )
        # Same arrow button style as TraceControl axis arrows
        arrow_style = (
            "QPushButton { background-color: #4b4a4a; color: #ccc;"
            " border: 1px solid #606060; border-radius: 2px;"
            " font-size: 7pt; padding: 0px; }"
            "QPushButton:pressed { background-color: #666; }"
        )

        for entry in PARAM_DEFS:
            attr, pn_code, label, unit, min_v, max_v, default, tooltip = entry

            row_label = QLabel(f"{pn_code} {label}:")
            row_label.setStyleSheet(label_style)
            row_label.setToolTip(tooltip)

            if attr in COMBO_ATTRS:
                # Pn100 — tuning mode dropdown, no arrows needed
                w = QComboBox()
                w.setStyleSheet(combo_style)
                w.addItems(TUNING_MODE_LABELS)
                w.setToolTip(tooltip)
                w.currentIndexChanged.connect(self._on_param_changed)
                self._param_widgets[attr] = w
                param_layout.addRow(row_label, w)
            else:
                # Numeric spinbox with custom ▲/▼ buttons — native arrows hidden
                spin = QSpinBox()
                spin.setRange(min_v, max_v)
                spin.setValue(default)
                spin.setToolTip(tooltip)
                spin.setFixedWidth(110)
                # Hide native up/down buttons; custom arrows replace them
                spin.setStyleSheet(
                    "QSpinBox { background: #2a2a3e; color: #d4d4d4;"
                    " border: 1px solid #555; border-radius: 2px;"
                    " padding: 1px 3px; font-size: 8pt; }"
                    "QSpinBox::up-button { width: 0; border: none; }"
                    "QSpinBox::down-button { width: 0; border: none; }"
                )
                spin.valueChanged.connect(self._on_param_changed)

                btn_up = QPushButton("\u25b2")
                btn_up.setFixedSize(18, 12)
                btn_up.setStyleSheet(arrow_style)
                btn_up.setToolTip(f"Increase {label}")
                btn_up.clicked.connect(
                    lambda _, s=spin, mx=max_v: s.setValue(min(mx, s.value() + 1))
                )

                btn_down = QPushButton("\u25bc")
                btn_down.setFixedSize(18, 12)
                btn_down.setStyleSheet(arrow_style)
                btn_down.setToolTip(f"Decrease {label}")
                btn_down.clicked.connect(
                    lambda _, s=spin, mn=min_v: s.setValue(max(mn, s.value() - 1))
                )

                arrows = QVBoxLayout()
                arrows.setSpacing(1)
                arrows.setContentsMargins(0, 0, 0, 0)
                arrows.addWidget(btn_up)
                arrows.addWidget(btn_down)

                unit_lbl = QLabel(unit)
                unit_lbl.setStyleSheet("color: #888; font-size: 8pt;")

                field_row = QHBoxLayout()
                field_row.setSpacing(2)
                field_row.setContentsMargins(0, 0, 0, 0)
                field_row.addWidget(spin)
                field_row.addLayout(arrows)
                field_row.addWidget(unit_lbl)
                field_row.addStretch()

                field_container = QWidget()
                field_container.setLayout(field_row)

                self._param_widgets[attr] = spin
                param_layout.addRow(row_label, field_container)

        outer.addWidget(self._param_frame)
        return group

    # -----------------------------------------------------------------------
    # Drive profile UI callbacks
    # -----------------------------------------------------------------------
    def _current_axis(self) -> int:
        return int(self._axis_combo.currentText())

    def _on_axis_changed(self):
        """Load the profile for the newly selected axis."""
        axis = self._current_axis()
        profile = self._profiles.get(axis, DriveProfile())
        self._load_profile_to_ui(profile)

    def _on_drive_type_changed(self, drive_type: str):
        """Show/hide parameter fields; populate defaults if switching to a Trio drive."""
        is_trio_drive = drive_type in ("DX3", "DX4")
        self._param_frame.setVisible(is_trio_drive)

        axis = self._current_axis()
        existing = self._profiles.get(axis)

        if is_trio_drive and (existing is None or not existing.has_drive_params()):
            # Fill defaults when user first selects a Trio drive
            self._set_ui_to_defaults()

        self._save_ui_to_profile()

    def _on_param_changed(self):
        """Save parameter edits back into the profile dict."""
        self._save_ui_to_profile()

    def _load_profile_to_ui(self, profile: DriveProfile):
        """Populate UI controls from a DriveProfile without triggering saves."""
        # Block signals during bulk load
        self._drive_combo.blockSignals(True)
        drive_idx = DRIVE_TYPES.index(profile.drive_type) if profile.drive_type in DRIVE_TYPES else 0
        self._drive_combo.setCurrentIndex(drive_idx)
        self._drive_combo.blockSignals(False)

        is_trio = profile.has_drive_params()
        self._param_frame.setVisible(is_trio)

        if is_trio:
            for entry in PARAM_DEFS:
                attr = entry[0]
                default = entry[6]
                val = getattr(profile, attr, None)
                w = self._param_widgets.get(attr)
                if w is None:
                    continue
                w.blockSignals(True)
                if attr in COMBO_ATTRS:
                    idx = TUNING_MODE_VALUES.index(val) if val in TUNING_MODE_VALUES else 0
                    w.setCurrentIndex(idx)
                else:
                    w.setValue(val if val is not None else default)
                w.blockSignals(False)

    def _set_ui_to_defaults(self):
        """Reset all parameter widgets to their default values."""
        for entry in PARAM_DEFS:
            attr, _, _, _, _, _, default, _ = entry
            w = self._param_widgets.get(attr)
            if w is None:
                continue
            w.blockSignals(True)
            if attr in COMBO_ATTRS:
                w.setCurrentIndex(0)
            else:
                w.setValue(default)
            w.blockSignals(False)

    def _save_ui_to_profile(self):
        """Read current UI state and store it as a DriveProfile for the selected axis."""
        axis = self._current_axis()
        drive_type = self._drive_combo.currentText()
        profile = DriveProfile(drive_type=drive_type)

        if profile.has_drive_params():
            for entry in PARAM_DEFS:
                attr = entry[0]
                w = self._param_widgets.get(attr)
                if w is None:
                    continue
                if attr in COMBO_ATTRS:
                    profile.__dict__[attr] = TUNING_MODE_VALUES[w.currentIndex()]
                else:
                    profile.__dict__[attr] = w.value()

        self._profiles[axis] = profile

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def set_api_key(self, key: str):
        self._client.set_api_key(key)

    def set_model(self, model: str):
        self._client.set_model(model)
        self.model_combo.setCurrentText(model)

    def set_data_provider(self, provider):
        """
        Set a callable that returns (time_arr: np.ndarray, params: dict[str, np.ndarray])
        or (None, None) if no data is available.
        """
        self._data_provider = provider

    def get_all_profiles(self) -> dict[int, dict]:
        """Return all per-axis profiles as plain dicts (for QSettings persistence)."""
        return {axis: p.to_dict() for axis, p in self._profiles.items()}

    def set_all_profiles(self, profiles: dict[int, dict]):
        """Restore per-axis profiles from plain dicts (loaded from QSettings)."""
        self._profiles = {
            int(axis): DriveProfile.from_dict(d)
            for axis, d in profiles.items()
        }
        # Refresh UI for currently selected axis
        self._on_axis_changed()

    # -----------------------------------------------------------------------
    # Scope data + drive context
    # -----------------------------------------------------------------------
    def _get_scope_context(self) -> tuple[str, str] | None:
        """Get formatted metrics and raw data from current scope data."""
        if not self._data_provider:
            return None

        time_arr, params = self._data_provider()
        if time_arr is None or params is None or len(time_arr) == 0:
            return None

        metrics = SignalMetrics.compute_all(time_arr, params)
        metrics_text = SignalMetrics.format_for_llm(metrics)

        # Downsample raw data to keep token count reasonable
        n = len(time_arr)
        max_raw_samples = 500
        if n > max_raw_samples:
            step = n // max_raw_samples
            indices = np.arange(0, n, step)
        else:
            indices = np.arange(n)

        header = "Time," + ",".join(params.keys())
        rows = []
        for i in indices:
            values = [f"{time_arr[i]:.6f}"]
            for name in params:
                values.append(f"{params[name][i]:.6f}")
            rows.append(",".join(values))

        raw_text = header + "\n" + "\n".join(rows)
        if n > max_raw_samples:
            raw_text += f"\n\n(Downsampled from {n} to {len(indices)} samples)"

        return metrics_text, raw_text

    def _get_drive_context(self) -> str:
        """Build drive profile context string for the selected axis."""
        axis = self._current_axis()
        profile = self._profiles.get(axis)
        if profile is None:
            return ""
        return profile.format_for_ai(axis)

    # -----------------------------------------------------------------------
    # Query / streaming
    # -----------------------------------------------------------------------
    def _send_query(self, user_text: str):
        """Send a query to NanoGPT with scope data + drive profile context."""
        if self._streaming:
            return

        if not self._client.is_configured():
            self._append_system(
                "API key not configured. Go to Settings → AI Analysis to set your NanoGPT API key."
            )
            return

        context = self._get_scope_context()
        if not context:
            self._append_system("No scope data available. Capture data first, then try again.")
            return

        metrics_text, raw_text = context
        drive_context = self._get_drive_context()

        self._append_user(user_text)

        # Build user content — include drive context block only if available
        drive_block = (
            f"Drive profile for the selected axis:\n\n"
            f"```\n{drive_context}\n```\n\n"
            if drive_context else
            "(No drive profile configured for the selected axis.)\n\n"
        )

        user_content = (
            f"{drive_block}"
            f"Pre-computed signal metrics:\n\n"
            f"```\n{metrics_text}\n```\n\n"
            f"Raw sampled data (CSV):\n\n"
            f"```csv\n{raw_text}\n```\n\n"
            f"User question: {user_text}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        self._streaming = True
        self._current_response = ""
        self.btn_send.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_tune.setEnabled(False)
        self.status_label.setText("Analyzing...")
        self._append_assistant_start()

        self._client.chat_stream(
            messages,
            on_chunk=lambda text: self._signals.chunk_received.emit(text),
            on_done=lambda: self._signals.stream_done.emit(),
            on_error=lambda err: self._signals.error_occurred.emit(err),
        )

    def _on_send_clicked(self):
        text = self.input_edit.text().strip()
        if text:
            self.input_edit.clear()
            self._send_query(text)

    # -----------------------------------------------------------------------
    # Streaming callbacks
    # -----------------------------------------------------------------------
    def _on_chunk(self, text: str):
        self._current_response += text
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_stream_done(self):
        self._streaming = False
        self.btn_send.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_tune.setEnabled(True)
        self.status_label.setText("Done")
        self.chat_display.append("")

    def _on_error(self, error: str):
        self._streaming = False
        self.btn_send.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_tune.setEnabled(True)
        self.status_label.setText("")
        self._append_system(f"Error: {error}")

    # -----------------------------------------------------------------------
    # Chat display helpers
    # -----------------------------------------------------------------------
    def _append_user(self, text: str):
        self.chat_display.append(
            f'<span style="color: #03DAC6; font-weight: bold;">You:</span> {text}'
        )
        self.chat_display.append("")

    def _append_assistant_start(self):
        self.chat_display.append(
            '<span style="color: #FFB74D; font-weight: bold;">AI:</span> '
        )

    def _append_system(self, text: str):
        self.chat_display.append(
            f'<span style="color: #F06292;">{text}</span>'
        )
        self.chat_display.append("")

    def _clear_chat(self):
        self.chat_display.clear()
        self.status_label.setText("")
