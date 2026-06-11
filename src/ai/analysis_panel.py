"""
AI Analysis panel — a dockable Qt widget providing chat-style interaction
with NanoGPT for interpreting scope capture data.

The drive profile section at the top (drive_profile_editor) lets the user
assign a Trio DX3 or DX4 servo drive to each axis so that the AI receives
drive-level tuning context alongside the scope metrics. The LLM prompt
rules live in tuning_prompts.py.
"""

import logging

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QComboBox, QLabel, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QTextCursor

from .nanogpt_client import NanoGPTClient
from .signal_metrics import SignalMetrics
from .drive_profile_editor import DriveProfileEditor
from .tuning_prompts import ANALYZE_PROMPT, CUSTOM_PROMPT, SYSTEM_PROMPT, TUNE_PROMPT

logger = logging.getLogger(__name__)

# Cap for conversation history — only the last N clean user/assistant
# messages are sent to the model; bulky context is rebuilt every turn.
MAX_HISTORY_MESSAGES = 8  # 4 turns — enough for iterative tuning feedback


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
        # Clean conversational turns only — visible user prompt text and
        # assistant replies. Bulky scope/drive context is rebuilt each turn
        # and never stored here.
        self._conversation_history: list[dict] = []
        self._pending_user_text: str | None = None
        self._data_provider = None  # callable → (time_arr, params_dict)

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
        self.model_combo.addItems(NanoGPTClient.load_model_list())
        self.model_combo.setCurrentText(self._client.model)
        self.model_combo.currentTextChanged.connect(self._client.set_model)
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        # ── Drive profile section ───────────────────────────────────────────
        self._profile_editor = DriveProfileEditor(autowrite=False, max_width=None)
        layout.addWidget(self._profile_editor)

        # ── Quick action buttons (Analyze / Tune only) ──────────────────────
        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)

        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setFixedHeight(28)
        self.btn_analyze.setToolTip(ANALYZE_PROMPT)
        self.btn_analyze.clicked.connect(
            lambda: self._send_query("Analyze", mode_marker=ANALYZE_PROMPT)
        )
        actions_row.addWidget(self.btn_analyze)

        self.btn_tune = QPushButton("Tune")
        self.btn_tune.setFixedHeight(28)
        self.btn_tune.setToolTip(TUNE_PROMPT)
        self.btn_tune.clicked.connect(
            lambda: self._send_query("Tune", mode_marker=TUNE_PROMPT)
        )
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

        self.btn_new_chat = QPushButton("New Chat")
        self.btn_new_chat.setFixedWidth(65)
        self.btn_new_chat.setFixedHeight(28)
        self.btn_new_chat.setToolTip("Start a new conversation (clears history)")
        self.btn_new_chat.clicked.connect(self._new_chat)
        input_row.addWidget(self.btn_new_chat)

        layout.addLayout(input_row)

        # ── Status label ─────────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(self.status_label)

        self.setWidget(container)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def set_api_key(self, key: str):
        self._client.set_api_key(key)

    def set_model(self, model: str):
        self._client.set_model(model)
        self.model_combo.setCurrentText(model)

    def refresh_model_list(self):
        """Reload the model combo items from the persisted model list."""
        current = self._client.model
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(NanoGPTClient.load_model_list())
        self.model_combo.setCurrentText(current)
        self.model_combo.blockSignals(False)

    def set_connection(self, connection, conn_lock=None):
        """
        Provide the active TUA.TrioConnection so the panel can read/write drive
        parameters via CoE SDO.  Pass None to disable the Read/Write buttons.
        """
        self._profile_editor.set_connection(connection, conn_lock)

    def set_data_provider(self, provider):
        """
        Set a callable that returns (time_arr: ndarray, params: dict[str, ndarray])
        or (None, None) if no data is available.
        """
        self._data_provider = provider

    def get_all_profiles(self) -> dict[int, dict]:
        """Return all per-axis profiles as plain dicts (for QSettings persistence)."""
        return self._profile_editor.get_all_profiles()

    def set_all_profiles(self, profiles: dict[int, dict]):
        """Restore per-axis profiles from plain dicts (loaded from QSettings)."""
        self._profile_editor.set_all_profiles(profiles)

    # -----------------------------------------------------------------------
    # Scope data + drive context
    # -----------------------------------------------------------------------
    def _validate_scope_data(self, time_arr, params: dict) -> None:
        """Raise ValueError if scope arrays are empty or length-inconsistent."""
        n = len(time_arr) if time_arr is not None else 0
        if n == 0:
            raise ValueError("scope time array is empty")
        bad = [
            name for name, arr in (params or {}).items()
            if arr is None or len(arr) != n
        ]
        if bad:
            raise ValueError(
                f"scope channel length mismatch (expected {n}): "
                f"{', '.join(bad)}"
            )

    def _get_scope_context(self) -> str | None:
        """Return the formatted metrics block, or None if unavailable.

        The raw CSV is intentionally NOT returned — LLMs cannot do numeric
        analysis on arrays, and the downsampled CSV was aliased and
        untrustworthy for frequency claims. All trustworthy numbers come
        from SignalMetrics, which runs on the full-rate capture.
        """
        if not self._data_provider:
            return None

        try:
            time_arr, params = self._data_provider()
        except Exception as exc:
            logger.exception("Scope data provider failed: %s", exc)
            self._append_chat_line(
                "System:", f"Scope data provider error: {exc}"
            )
            return None

        if time_arr is None or params is None or len(time_arr) == 0:
            return None

        try:
            self._validate_scope_data(time_arr, params)
        except ValueError as exc:
            logger.warning("Scope data rejected: %s", exc)
            self._append_chat_line(
                "System:", f"Scope data rejected: {exc}"
            )
            return None

        metrics = SignalMetrics.compute_all(time_arr, params)
        return SignalMetrics.format_for_llm(metrics)

    def _get_drive_context(self) -> str:
        """Build drive profile context string for the selected axis."""
        profile = self._profile_editor.current_profile()
        if profile is None:
            return ""
        return profile.format_for_ai(self._profile_editor.current_axis())

    # -----------------------------------------------------------------------
    # Query / streaming
    # -----------------------------------------------------------------------
    def _build_context_block(self, metrics_text: str) -> str:
        """Build the per-turn context block wrapped in <scope_capture> tags.

        Contains only the selected axis, drive profile, and pre-computed
        metrics. No raw CSV — LLMs cannot read numeric arrays usefully.
        """
        axis = self._profile_editor.current_axis()
        drive_context = self._get_drive_context()

        if drive_context:
            drive_block = f"Drive profile:\n{drive_context}"
        else:
            drive_block = "Drive profile: (none configured for this axis)"

        return (
            "<scope_capture>\n"
            f"Selected axis: {axis}\n\n"
            f"{drive_block}\n\n"
            f"Pre-computed signal metrics (authoritative):\n"
            f"{metrics_text}\n"
            "</scope_capture>"
        )

    def _build_messages(
        self,
        mode_marker: str,
        context_block: str,
        user_text: str,
    ) -> list[dict]:
        """Assemble the NanoGPT chat request.

        Order:
          1. system prompt
          2. mode marker
          3. trimmed conversation history (last MAX_HISTORY_MESSAGES)
          4. current user message — the refreshed <scope_capture> block
             is bundled with the user's own text so the latest capture
             unambiguously travels with the current question.
        """
        trimmed = self._conversation_history[-MAX_HISTORY_MESSAGES:]
        current_user_content = (
            f"{context_block}\n\n"
            "NOTE: The <scope_capture> block above is refreshed for THIS "
            "turn. Any numbers, metrics, or drive values from earlier in "
            "the conversation are STALE — use only the block above.\n\n"
            f"User message: {user_text}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": mode_marker},
            *trimmed,
            {"role": "user", "content": current_user_content},
        ]

    def _send_query(self, user_text: str, *, mode_marker: str | None = None):
        """Send a query to NanoGPT with the metrics block + drive profile."""
        if self._streaming:
            return

        if not self._client.is_configured():
            self._append_chat_line(
                "System:",
                "API key not configured. Go to Settings → AI Analysis to set "
                "your NanoGPT API key.",
            )
            return

        metrics_text = self._get_scope_context()
        if not metrics_text:
            self._append_chat_line(
                "System:",
                "No scope data available. Capture data first, then try again.",
            )
            return

        marker = mode_marker or CUSTOM_PROMPT
        context_block = self._build_context_block(metrics_text)
        messages = self._build_messages(marker, context_block, user_text)

        self._append_chat_line("You:", user_text)
        self._append_chat_line("AI:", "", trailing_blank=False)

        self._streaming = True
        self._current_response = ""
        self._pending_user_text = user_text
        self.btn_send.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_tune.setEnabled(False)
        self.btn_new_chat.setEnabled(False)
        self.status_label.setText("Analyzing...")

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
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _on_stream_done(self):
        """Commit the turn to history as a compact change-log entry.

        For iterative tuning sessions we don't want the full assistant
        reply in history — what matters next turn is "what did we change
        last time and did it help". We store the user's visible prompt
        plus a trimmed assistant summary (first ~600 chars), which is
        enough for the model to reference prior recommendations without
        ballooning context.
        """
        self._streaming = False

        if self._current_response and self._pending_user_text is not None:
            trimmed_reply = self._current_response.strip()
            if len(trimmed_reply) > 600:
                trimmed_reply = trimmed_reply[:600] + "\n[...truncated for history]"

            self._conversation_history.append(
                {"role": "user", "content": self._pending_user_text}
            )
            self._conversation_history.append(
                {"role": "assistant", "content": trimmed_reply}
            )
            if len(self._conversation_history) > MAX_HISTORY_MESSAGES:
                self._conversation_history = (
                    self._conversation_history[-MAX_HISTORY_MESSAGES:]
                )
        self._pending_user_text = None

        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n\n")
        self.chat_display.setTextCursor(cursor)

        turns = len(self._conversation_history) // 2
        self.btn_send.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_tune.setEnabled(True)
        self.btn_new_chat.setEnabled(True)
        self.status_label.setText(f"Done — turn {turns}")

    def _on_error(self, error: str):
        self._streaming = False
        # Nothing was committed to history on failure, so no rollback needed.
        self._pending_user_text = None
        self.btn_send.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_tune.setEnabled(True)
        self.btn_new_chat.setEnabled(True)
        self.status_label.setText("")
        self._append_chat_line("System:", f"Error: {error}")

    # -----------------------------------------------------------------------
    # Chat display helpers (plain text only — no HTML)
    # -----------------------------------------------------------------------
    def _append_chat_line(
        self,
        prefix: str,
        text: str = "",
        *,
        trailing_blank: bool = True,
    ):
        """Append a plain-text chat line. Never uses HTML or rich text.

        ``prefix`` is the speaker label (``"You:"``, ``"AI:"``, ``"System:"``).
        ``text`` is the message body. When ``trailing_blank`` is true, the
        line is followed by a blank line separator.
        """
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Ensure we start at the beginning of a fresh line.
        current = self.chat_display.toPlainText()
        if current and not current.endswith("\n"):
            cursor.insertText("\n")

        if prefix and text:
            cursor.insertText(f"{prefix} {text}")
        elif prefix:
            # Trailing space so streamed chunks append cleanly after the label.
            cursor.insertText(prefix if prefix.endswith(" ") else f"{prefix} ")
        elif text:
            cursor.insertText(text)

        if trailing_blank:
            cursor.insertText("\n\n")

        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _new_chat(self):
        """Start a fresh conversation — clears display and history."""
        if self._streaming:
            return
        self.chat_display.clear()
        self._conversation_history.clear()
        self._pending_user_text = None
        self.status_label.setText("New conversation started")
