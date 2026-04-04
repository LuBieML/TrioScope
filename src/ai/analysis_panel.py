"""
AI Analysis panel — a dockable Qt widget providing chat-style interaction
with NanoGPT for interpreting scope capture data.
"""

import logging
import numpy as np

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QComboBox, QLabel, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QObject

from .nanogpt_client import NanoGPTClient
from .signal_metrics import SignalMetrics

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert motion control and signal analysis engineer embedded in \
TrioScope, an oscilloscope application for **Trio Motion Controllers**.

The system uses Trio MC-series controllers communicating with servo/stepper \
drives over EtherCAT, SERCOS, or analog ±10V interfaces. All captured data \
comes from the controller's built-in SCOPE command, which samples servo \
parameters deterministically at the servo rate.

Common servo parameters you will see:
- MPOS / DPOS — measured vs demand position (following error = MPOS − DPOS)
- FE — following error (the controller computes this directly)
- DAC / DAC_OUT — drive command output (torque/velocity demand to the drive)
- SPEED / MSPEED / AXIS_SPEED — demand vs measured speed
- ENCODER — raw encoder counts
- P_GAIN, I_GAIN, D_GAIN, VFF_GAIN, OV_GAIN — servo loop gains
- FE_LIMIT — maximum allowed following error before axis error
- DRIVE_CURRENT / DRIVE_TORQUE — actual drive current/torque feedback
- OUTLIMIT — output limit (caps DAC output)
- D_ZONE_MIN / D_ZONE_MAX — deadzone for derivative term
- BACKLASH_DIST — backlash compensation distance

You will receive both pre-computed signal metrics AND the raw sampled data. \
Use the raw data to spot patterns, correlations, and details that metrics \
alone may miss.

When suggesting adjustments, always refer to **Trio servo parameters** by name \
(e.g. P_GAIN, I_GAIN, VFF_GAIN, FE_LIMIT, OUTLIMIT). Suggest which servo \
parameter to increase, decrease, or check — and explain what effect the \
change will have on the motion behaviour.

Provide actionable insights about:
- Motion quality (vibration, resonance, overshoot, settling)
- Servo tuning (which gains to adjust and in which direction)
- Drive-level issues (DAC saturation, current limits)
- Mechanical issues (backlash, friction, compliance)
- Following error patterns and their likely causes
- Signal anomalies (spikes, drift, saturation, encoder noise)

Be concise and technical. Use specific numbers from the data. \
If you identify a problem, explain both the symptom and likely root cause.\
"""

# Quick-action prompts the user can click instead of typing
QUICK_ACTIONS = [
    ("Analyze", "Analyze all captured signals and raw data. Identify any issues with "
     "motion quality, servo tuning, or anomalies. Summarise findings and refer to "
     "relevant servo parameters."),
    ("Tune", "Based on the captured data, suggest servo tuning improvements. "
     "Which servo parameters (P_GAIN, I_GAIN, D_GAIN, VFF_GAIN, etc.) should be "
     "adjusted and in which direction? Explain expected effect on motion."),
    ("Diagnose", "Check for mechanical or electrical issues: backlash, resonance, "
     "friction, encoder noise, drive saturation. Which servo parameters to check?"),
    ("Compare", "Compare all captured parameters against each other. "
     "Identify correlations, phase relationships, and any discrepancies "
     "that indicate tuning or mechanical problems."),
]


class _Signals(QObject):
    """Thread-safe Qt signals for async API callbacks."""
    chunk_received = Signal(str)
    stream_done = Signal()
    error_occurred = Signal(str)


class AIAnalysisPanel(QDockWidget):
    """Dockable AI analysis panel with chat interface."""

    def __init__(self, parent=None):
        super().__init__("AI Analysis", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.setMinimumWidth(350)

        self._client = NanoGPTClient()
        self._signals = _Signals()
        self._signals.chunk_received.connect(self._on_chunk)
        self._signals.stream_done.connect(self._on_stream_done)
        self._signals.error_occurred.connect(self._on_error)

        self._streaming = False
        self._current_response = ""
        self._data_provider = None  # callable that returns (time_arr, params_dict)

        self._build_ui()

    def _build_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Model selector row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(NanoGPTClient.AVAILABLE_MODELS)
        self.model_combo.setCurrentText(self._client.model)
        self.model_combo.currentTextChanged.connect(self._client.set_model)
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        # Quick action buttons
        actions_row = QHBoxLayout()
        actions_row.setSpacing(3)
        for label, prompt in QUICK_ACTIONS:
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setToolTip(prompt)
            btn.clicked.connect(lambda checked, p=prompt: self._send_query(p))
            actions_row.addWidget(btn)
        layout.addLayout(actions_row)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit { background-color: #1a1a2e; color: #d4d4d4;"
            " font-family: Consolas, monospace; font-size: 9pt;"
            " border: 1px solid #4b4a4a; border-radius: 3px; }"
        )
        self.chat_display.setPlaceholderText(
            "Capture scope data, then click 'Analyze' or type a question below.\n\n"
            "The AI will receive pre-computed signal metrics (statistics, FFT, "
            "anomalies) and provide expert interpretation."
        )
        layout.addWidget(self.chat_display, 1)

        # Input row
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

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(self.status_label)

        self.setWidget(container)

    def set_api_key(self, key: str):
        self._client.set_api_key(key)

    def set_model(self, model: str):
        self._client.set_model(model)
        self.model_combo.setCurrentText(model)

    def set_data_provider(self, provider):
        """
        Set a callable that returns the current scope data.
        provider() should return (time_arr: np.ndarray, params: dict[str, np.ndarray])
        or (None, None) if no data is available.
        """
        self._data_provider = provider

    def _get_scope_context(self) -> tuple[str, str] | None:
        """Get formatted metrics and raw data from current scope data."""
        if not self._data_provider:
            return None

        time_arr, params = self._data_provider()
        if time_arr is None or params is None or len(time_arr) == 0:
            return None

        metrics = SignalMetrics.compute_all(time_arr, params)
        metrics_text = SignalMetrics.format_for_llm(metrics)

        # Build raw data as CSV-style text
        # Downsample if too many points to keep token count reasonable
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
            raw_text += f"\n\n(Downsampled from {n} to {len(indices)} samples for context)"

        return metrics_text, raw_text

    def _send_query(self, user_text: str):
        """Send a query to NanoGPT with scope data context."""
        if self._streaming:
            return

        if not self._client.is_configured():
            self._append_system("API key not configured. Go to Settings to set your NanoGPT API key.")
            return

        # Build data context
        context = self._get_scope_context()
        if not context:
            self._append_system("No scope data available. Capture data first, then try again.")
            return

        metrics_text, raw_text = context

        # Show user message
        self._append_user(user_text)

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Here are the pre-computed signal metrics:\n\n"
                f"```\n{metrics_text}\n```\n\n"
                f"Here is the raw sampled data (CSV format):\n\n"
                f"```csv\n{raw_text}\n```\n\n"
                f"User question: {user_text}"
            )},
        ]

        # Start streaming response
        self._streaming = True
        self._current_response = ""
        self.btn_send.setEnabled(False)
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

    def _on_chunk(self, text: str):
        self._current_response += text
        # Update the last assistant block in-place
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_stream_done(self):
        self._streaming = False
        self.btn_send.setEnabled(True)
        self.status_label.setText("Done")
        self.chat_display.append("")  # blank line after response

    def _on_error(self, error: str):
        self._streaming = False
        self.btn_send.setEnabled(True)
        self.status_label.setText("")
        self._append_system(f"Error: {error}")

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
