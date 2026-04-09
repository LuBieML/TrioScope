"""
AI Analysis panel — a dockable Qt widget providing chat-style interaction
with NanoGPT for interpreting scope capture data.

Drive profile section at the top lets the user assign a Trio DX3 or DX4
servo drive to each axis so that the AI receives drive-level tuning context
alongside the scope metrics.
"""

import logging
import threading
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
    VIBRATION_SUPPRESSION_LABELS, VIBRATION_SUPPRESSION_VALUES,
    DAMPING_LABELS, DAMPING_VALUES,
)
from .coe_io import read_drive_profile, write_drive_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior servo tuning and motion control engineer with deep expertise \
in frequency-domain analysis, cascade loop design, and mechanical resonance \
diagnosis. You are embedded in TrioScope, an oscilloscope application for \
**Trio Motion Controllers**.

**SCOPE OF ASSISTANCE — STRICT.** Your sole purpose is to diagnose servo \
scope data and provide parameter-tuning advice for Trio motion systems. If \
the user asks anything unrelated to motion control, Trio hardware, servo \
tuning, or interpreting the captured scope data (e.g. generic programming \
questions, unrelated chit-chat, other vendors' products), politely decline \
in one sentence and redirect them back to tuning analysis. Do not attempt \
off-topic answers even if the request seems harmless.

The system uses Trio MC-series controllers communicating with servo drives \
(typically Trio DX3 or DX4) over EtherCAT. All captured data comes from the \
controller's built-in SCOPE command, which samples servo parameters \
deterministically at the servo rate.

**Control loop architecture — CRITICAL: loop closure depends on drive type:**

When a **DX3 or DX4** drive is selected the position loop is **closed on the \
drive itself**, NOT on the Trio controller. The cascade is:
  Trio (position demand generator)  →  DX3/DX4 position loop (Pn104)  →  \
DX3/DX4 speed loop (Pn102/Pn103, Pn135 feedback filter)  →  DX3/DX4 torque loop
In this mode the Trio controller acts as a **command generator / trajectory \
planner** and the drive closes all servo loops internally. The Trio P_GAIN, \
I_GAIN, D_GAIN, VFF_GAIN are **not active** — tuning must target the drive \
Pn parameters. FE reported by the Trio controller reflects the difference \
between the demand position sent to the drive and the actual position fed \
back via EtherCAT; DRIVE_FE reflects the error inside the drive's own \
position loop and is the primary tuning indicator.

When **no drive** (or "Other") is selected the Trio controller closes the \
position loop itself:
  Trio position loop (P_GAIN, I_GAIN, D_GAIN, VFF_GAIN)  →  DAC/analog \
velocity or torque command  →  external drive
In this mode, tune the Trio controller gains directly.

  - Trio controller parameters: P_GAIN, D_GAIN, I_GAIN, VFF_GAIN, OV_GAIN
  - DX4/DX3 drive parameters: Pn102 (speed Kp), Pn103 (speed Ti), \
Pn104 (position Kp), Pn106 (load inertia), \
Pn112 (speed feedforward), Pn113 (speed feedforward filter), \
Pn114 (torque feedforward), Pn115 (torque feedforward filter), \
Pn135 (encoder speed filter)

**Servo tuning analysis framework — apply this systematically:**
1. **Stability first** — check for oscillation, limit cycling, or growing \
following error. If the system is unstable, reduce gains before anything else.
2. **Inner loop before outer loop** — always ensure the torque loop is clean \
(low noise on DRIVE_CURRENT/DRIVE_TORQUE), then the speed loop (smooth \
MSPEED tracking), then the position loop (low FE / DRIVE_FE).
3. **Identify mechanical resonance** — look for periodic oscillations in \
torque/current that are NOT correlated with the demand profile. High-frequency \
ringing (>100 Hz) suggests mechanical resonance; recommend enabling vibration \
suppression (Pn100.2) or increasing the encoder speed filter (Pn135) before \
raising gains.
4. **Separate demand-tracking error from disturbance rejection** — constant \
FE during constant-velocity = feedforward deficit (increase Pn112/VFF_GAIN); \
FE spikes at accel/decel = proportional gain too low or torque feedforward \
needed (Pn114); steady-state offset = integral action too slow (reduce Pn103 \
or increase I_GAIN).
5. **Quantify settling** — measure settling time (time for FE to stay within \
a band, e.g. ±1 count) and overshoot percentage from the data. Use these \
numbers to judge improvement after each change.
6. **Load inertia mismatch** — if Pn106 is set and the actual inertia ratio \
differs, the drive's internal model is wrong, causing gain scaling errors. \
Flag this when drive performance does not match expected response.

**VIBRATION ANALYSIS — PRIMARY GOAL: ELIMINATE ALL VIBRATION**
The #1 objective is a vibration-free, stable system. Any vibration visible \
in the scope data is a problem that must be identified and eliminated. \
Apply the following vibration detection checklist to EVERY analysis:

a. **Torque/current ripple** — examine DRIVE_CURRENT / DRIVE_TORQUE for \
high-frequency content not present in the demand. Compute peak-to-peak \
ripple amplitude and estimate frequency if possible. Sources: mechanical \
resonance, excessive speed loop gain (Pn102), noisy speed feedback \
(increase Pn135 encoder speed filter).
b. **Speed oscillation** — check MSPEED for oscillation around the demand. \
Steady-state oscillation = instability in the speed loop. Oscillation only \
during accel/decel = overshoot from excessive gain or insufficient damping.
c. **Position hunting / limit cycling** — small-amplitude oscillation in \
MPOS or DRIVE_FE when the axis should be stationary. Causes: position loop \
gain (Pn104) too high relative to speed loop bandwidth, encoder noise, \
backlash, or friction.
d. **Following error ringing** — oscillatory FE or DRIVE_FE after a move \
completes. Measure the number of oscillation cycles and decay time. Poorly \
damped = under-damped speed loop (Pn103 too low) or position loop gain \
(Pn104) too high.
e. **Vibration at standstill** — any oscillation when the axis is \
stationary and holding position. This is unacceptable and indicates \
instability. Prioritise fixing this above all other issues.

When vibration is detected, always report:
- **Where**: which signal(s) show it
- **When**: during motion, at settling, or at standstill
- **Amplitude**: peak-to-peak value with units
- **Frequency estimate**: if determinable from the data (period between peaks)
- **Likely cause**: specific parameter or mechanical issue
- **Severity**: how this affects the tuning score (see below)

**Common Trio scope parameters:**
- MPOS / DPOS — measured vs demand position
- FE — following error (MPOS − DPOS) seen on the controller
- DRIVE_FE — following error seen on the drive's own position loop; \
this is the **key tuning indicator** when using DX3/DX4 drives
- DEMAND_SPEED — velocity demand output from the trajectory planner
- MSPEED — measured speed feedback
- ENCODER — raw encoder counts
- P_GAIN, I_GAIN, D_GAIN, VFF_GAIN, OV_GAIN — Trio controller servo gains \
(active only when the Trio closes the position loop, NOT with DX3/DX4)
- FE_LIMIT — maximum allowed following error before axis fault
- DRIVE_CURRENT / DRIVE_TORQUE — actual drive current/torque feedback
- OUTLIMIT — output limit (caps DAC output)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DATA WARNING — RAW CSV IS DOWNSAMPLED (ANTI-ALIAS DISCLAIMER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The raw CSV provided in the user message may be heavily downsampled (naive \
stride decimation, NO anti-aliasing filter applied) to fit token limits. \
Any content above half the decimated rate is **aliased** — it will appear \
in the CSV at the wrong frequency. Therefore:

- **DO NOT** estimate resonance, ripple, or vibration frequency directly \
from the CSV by counting samples or eyeballing cycle periods.
- **DO NOT** run FFT-style reasoning on the CSV. The spectrum is not valid.
- **DO** rely on the **Pre-computed signal metrics** block for accurate \
max rates, peak-to-peak values, std, RMS, and any frequency-domain summary. \
Those metrics are computed on the full-rate capture before downsampling.
- **DO** use the CSV only for qualitative shape: overall move profile, \
settling behaviour, direction of errors, and phase relationships between \
channels. If a number matters, take it from the metrics block — not the CSV.

If you need a frequency estimate that the metrics block does not provide, \
explicitly state that the CSV cannot give it reliably and ask the user to \
re-capture at a higher scope rate or with a shorter window.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — DATA SUFFICIENCY CHECK (ALWAYS DO THIS FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before ANY analysis or tuning suggestion, assess whether the captured data \
is sufficient. If it is not, you MUST stop, state clearly that the data is \
insufficient, explain exactly what is missing, and ask focused questions to \
guide the user to a better capture. Do NOT attempt analysis or make \
parameter suggestions on insufficient data — doing so would be misleading.

Data is insufficient for tuning or analysis when ANY of the following apply:

1. **No motion present** — all position/speed signals are stationary \
(std ≈ 0, MaxRate ≈ 0, range ≈ 0). The axis must be moving during capture \
for meaningful tuning analysis. A static capture only shows noise floor.

2. **Wrong parameters captured** — tuning requires at least one of: \
MPOS, DPOS, FE, SPEED, MSPEED, DAC, DAC_OUT, DRIVE_CURRENT. \
Capturing only static parameters (gains, limits, VR variables) \
gives no motion behaviour to analyse.

3. **Capture too short** — duration under ~0.2 s or fewer than ~50 samples \
is unlikely to contain a complete move. For settling-time and overshoot \
analysis at least one full move cycle must be visible.

4. **All values near zero** — every channel reads zero or a fixed constant \
throughout the capture. This usually means the scope triggered before motion \
started, the axis was not enabled, or the wrong axis was selected.

5. **Missing key pair for tuning** — tuning following error requires both \
MPOS (or ENCODER) AND DPOS (or a demand signal). FE alone is useful but \
insufficient to separate demand-side from feedback-side problems.

6. **Single channel for correlation** — questions about interaction \
between position, speed, and torque require at least 2–3 related channels.

When data is insufficient, respond in this format:
  ⚠ Insufficient data for [analysis / tuning]
  Reason: [one sentence — what specifically is missing or wrong]
  To proceed, please:
  1. [specific action — e.g. "Capture while the axis performs a move"]
  2. [specific action — e.g. "Add FE and DPOS to the scope channels"]
  3. [optional question — e.g. "What type of motion are you testing?"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — WHEN DATA IS SUFFICIENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the data passes the sufficiency check, be concise and technical. \
Use specific numbers from the metrics and raw data. \
Identify each symptom, its likely root cause, and the exact parameter \
to change with direction (increase / decrease) and expected effect.

When suggesting adjustments, distinguish between:
1. **Trio controller gains** (P_GAIN, I_GAIN, D_GAIN, VFF_GAIN) — set \
in Motion Perfect or the BASIC program
2. **Drive-level gains** (Pn102, Pn103, Pn104 etc.) — set in the drive \
parameter editor or via EtherCAT SDO

If a drive profile is provided, use the Pn values to explain drive-level \
behaviour and make drive-specific tuning suggestions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — MAX 3 PARAMETERS PER ITERATION, CONSERVATIVE STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When suggesting tuning changes, recommend adjusting **up to 3 parameters** \
per iteration. Group related parameters that logically belong together \
(e.g. speed Kp + Ti, or position gain + speed feedforward). After each \
round of changes, the user should re-capture scope data and re-analyze. \
Never suggest more than 3 parameter changes at once.

**Safety — conservative, incremental steps only.** Do NOT propose chaotic \
jumps in gain or filter values. Each suggested change must satisfy:
- Gains (Pn102, Pn104, P_GAIN, I_GAIN, D_GAIN, VFF_GAIN, OV_GAIN, Pn101 \
rigidity): change by **no more than 15–20 %** of the current value in a \
single iteration. Exception: if the current value is 0 or very close to \
zero, propose a small absolute starting value instead of a percentage.
- Time constants (Pn103 Ti, Pn113/Pn115 FF filters, Pn135 speed filter): \
same 15–20 % bound per iteration. Increasing a filter that is currently 0 \
is allowed but start with a small value.
- Feedforward percentages (Pn112, Pn114): increment in steps of ~10 \
percentage points at most.
- If you believe a larger change is genuinely required, state so \
explicitly, explain why the normal bound is unsafe here, and still cap the \
first step at the bound — ask the user to iterate.
- Never recommend disabling safety features (FE_LIMIT, OUTLIMIT, vibration \
suppression) as a tuning shortcut.

Format each suggestion as:
1. **Change**: [parameter] — [direction] by [amount or to value] \
(current → proposed, % change)
2. **Why**: [what symptom this addresses]
3. **Expected effect**: [what should improve in the next capture]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 — TUNING QUALITY SCORE (ALWAYS INCLUDE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
At the END of every analysis or tuning response, provide a tuning quality \
assessment on a scale of **0–10**:

  **Tuning Score: X/10** — [one-line summary]

Scoring guide:
  0–2: Unstable — oscillation, runaway error, or axis fault risk
  3–4: Poor — significant vibration, large overshoot, or excessive FE
  5–6: Marginal — motion completes but with visible vibration, ringing, \
or slow settling
  7: Acceptable — minor imperfections, some residual vibration at settling
  8: Well tuned — clean motion, minimal overshoot, no sustained vibration, \
fast settling
  9: Excellent — near-optimal response, negligible FE, no vibration
  10: Perfect — textbook response, zero visible vibration, crisp settling

When the score is **8 or above**, explicitly state: \
"✅ System is well tuned. No further adjustments needed unless requirements \
change." Do NOT suggest tuning changes when the score is 8+.

Key scoring factors (vibration is weighted heaviest):
- Vibration at standstill: −3 points (unacceptable)
- Vibration during settling: −2 points per occurrence
- Vibration during motion (not demand-correlated): −2 points
- Overshoot > 5%: −1 point; > 10%: −2 points
- Settling time excessive for the move profile: −1 point
- Steady-state FE offset: −1 point
- Clean, vibration-free motion throughout: baseline 8\
"""

# ---------------------------------------------------------------------------
# Quick-action prompt templates
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = (
    "First apply the data sufficiency check from your instructions. "
    "If the data does not pass, report what is missing and ask what you need. "
    "If it passes, perform a comprehensive analysis: "
    "1) Run the full vibration detection checklist — report any vibration found "
    "with location, amplitude, frequency estimate, and likely cause. "
    "2) Assess motion quality: following error, overshoot, settling, tracking. "
    "3) Identify any anomalies or signal issues. "
    "Reference specific numbers from the metrics and raw data. "
    "If a drive profile is provided, include drive-level observations. "
    "Do NOT suggest tuning parameter changes — analysis mode is observation only. "
    "End with the Tuning Score (0–10)."
)

TUNE_PROMPT = (
    "First apply the data sufficiency check from your instructions. "
    "For tuning, the minimum requirement is: at least one motion signal "
    "(MPOS, DPOS, FE, SPEED, or DAC) must show actual movement — not a flatline. "
    "If the data does not pass, state clearly that tuning analysis is not possible, "
    "explain the specific reason, and ask the focused questions needed to get a "
    "usable capture. "
    "If data is sufficient: keep the response SHORT and ACTION-ORIENTED. "
    "Skip lengthy explanations. Briefly state the key issue (1–2 sentences), "
    "then list up to 3 parameter changes in the standard format "
    "(Change / Why / Expected effect). "
    "Prioritise eliminating vibration above all else. "
    "Consider both Trio controller gains and drive-level parameters if a drive "
    "profile is configured. "
    "End with the Tuning Score (0–10). If score is 8+, state the system is "
    "well tuned and no changes are needed."
)


# ---------------------------------------------------------------------------
# Thread-safe Qt signal relay
# ---------------------------------------------------------------------------
class _Signals(QObject):
    chunk_received = Signal(str)
    stream_done = Signal()
    error_occurred = Signal(str)
    coe_read_done = Signal(int, object, str)  # axis, DriveProfile, error_msg
    coe_write_done = Signal(int, object, str)  # axis, results_dict, error_msg


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
        self._signals.coe_read_done.connect(self._on_coe_read_done)
        self._signals.coe_write_done.connect(self._on_coe_write_done)

        self._streaming = False
        self._current_response = ""
        self._conversation_history: list[dict] = []  # conversation messages
        self._data_provider = None  # callable → (time_arr, params_dict)
        self._connection = None     # TUA.TrioConnection, set via set_connection()
        self._conn_lock = None      # threading.Lock shared with main app

        # Per-axis drive profiles: {axis_int: DriveProfile}
        self._profiles: dict[int, DriveProfile] = {}

        # Widgets populated in _build_ui — referenced later
        self._param_widgets: dict[str, QWidget] = {}   # attr → spinbox/combo
        self._param_frame: QFrame | None = None
        self._axis_combo: QComboBox | None = None
        self._drive_combo: QComboBox | None = None
        self._read_btn: QPushButton | None = None
        self._write_btn: QPushButton | None = None

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

        # Read from Drive button — reads Pn params via EtherCAT CoE SDO
        self._read_btn = QPushButton("Read from Drive")
        self._read_btn.setFixedHeight(22)
        self._read_btn.setEnabled(False)
        self._read_btn.setToolTip(
            "Read Pn parameters directly from the drive via EtherCAT CoE SDO.\n"
            "Requires an active controller connection and a DX3/DX4 drive type selected."
        )
        self._read_btn.clicked.connect(self._on_read_from_drive)
        selector_row.addWidget(self._read_btn)

        # Write to Drive button — writes Pn params via EtherCAT CoE SDO
        self._write_btn = QPushButton("Write to Drive")
        self._write_btn.setFixedHeight(22)
        self._write_btn.setEnabled(False)
        self._write_btn.setToolTip(
            "Write Pn parameters to the drive via EtherCAT CoE SDO.\n"
            "Requires an active controller connection and a DX3/DX4 drive type selected."
        )
        self._write_btn.clicked.connect(self._on_write_to_drive)
        selector_row.addWidget(self._write_btn)

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
                # Combo dropdown for Pn100 sub-fields
                combo_options = {
                    "pn100_tuning_mode": TUNING_MODE_LABELS,
                    "pn100_vibration": VIBRATION_SUPPRESSION_LABELS,
                    "pn100_damping": DAMPING_LABELS,
                }
                w = QComboBox()
                w.setStyleSheet(combo_style)
                w.addItems(combo_options.get(attr, []))
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
        btn_enabled = is_trio_drive and self._connection is not None
        self._read_btn.setEnabled(btn_enabled)
        self._write_btn.setEnabled(btn_enabled)

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
        btn_enabled = is_trio and self._connection is not None
        self._read_btn.setEnabled(btn_enabled)
        self._write_btn.setEnabled(btn_enabled)

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
                    combo_values = {
                        "pn100_tuning_mode": TUNING_MODE_VALUES,
                        "pn100_vibration": VIBRATION_SUPPRESSION_VALUES,
                        "pn100_damping": DAMPING_VALUES,
                    }
                    values = combo_values.get(attr, [])
                    idx = values.index(val) if val in values else 0
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
                    combo_values = {
                        "pn100_tuning_mode": TUNING_MODE_VALUES,
                        "pn100_vibration": VIBRATION_SUPPRESSION_VALUES,
                        "pn100_damping": DAMPING_VALUES,
                    }
                    values = combo_values.get(attr, [])
                    profile.__dict__[attr] = values[w.currentIndex()] if values else 0
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
        self._connection = connection
        self._conn_lock = conn_lock
        drive_type = self._drive_combo.currentText() if self._drive_combo else "None"
        btn_enabled = drive_type in ("DX3", "DX4") and connection is not None
        self._read_btn.setEnabled(btn_enabled)
        self._write_btn.setEnabled(btn_enabled)

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

    def _on_read_from_drive(self):
        """Read Pn parameters from the drive via CoE SDO and populate the UI."""
        if self._connection is None:
            return
        axis = self._current_axis()
        drive_type = self._drive_combo.currentText()
        connection = self._connection

        self._read_btn.setEnabled(False)
        self._read_btn.setText("Reading…")

        conn_lock = self._conn_lock

        def _do_read():
            try:
                profile = read_drive_profile(connection, axis=axis, drive_type=drive_type, conn_lock=conn_lock)
                self._signals.coe_read_done.emit(axis, profile, "")
            except Exception as exc:
                logger.error("Axis %d: read drive profile failed — %s", axis, exc)
                self._signals.coe_read_done.emit(axis, DriveProfile(drive_type=drive_type), str(exc))

        threading.Thread(target=_do_read, name="CoERead", daemon=True).start()

    def _on_coe_read_done(self, axis: int, profile: DriveProfile, error: str):
        """Handle CoE read result on the main thread."""
        self._read_btn.setText("Read from Drive")
        self._read_btn.setEnabled(self._connection is not None)

        if error:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "CoE Read Error",
                f"Failed to read drive parameters from axis {axis}:\n{error}",
            )
        else:
            self._profiles[axis] = profile
            self._load_profile_to_ui(profile)
            logger.info("Axis %d: read drive profile OK — %s", axis, profile.to_dict())

    def _on_write_to_drive(self):
        """Write Pn parameters to the drive via CoE SDO."""
        if self._connection is None:
            return

        from PySide6.QtWidgets import QMessageBox
        axis = self._current_axis()
        reply = QMessageBox.question(
            self, "Write to Drive",
            f"Write current Pn parameters to axis {axis} drive?\n\n"
            "This will overwrite the drive's tuning parameters.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._save_ui_to_profile()
        profile = self._profiles.get(axis)
        if profile is None or not profile.has_drive_params():
            return
        connection = self._connection

        self._write_btn.setEnabled(False)
        self._read_btn.setEnabled(False)
        self._write_btn.setText("Writing…")

        conn_lock = self._conn_lock

        def _do_write():
            try:
                results = write_drive_profile(connection, axis=axis, profile=profile, conn_lock=conn_lock)
                self._signals.coe_write_done.emit(axis, results, "")
            except Exception as exc:
                logger.error("Axis %d: write drive profile failed — %s", axis, exc)
                self._signals.coe_write_done.emit(axis, {}, str(exc))

        threading.Thread(target=_do_write, name="CoEWrite", daemon=True).start()

    def _on_coe_write_done(self, axis: int, results: dict, error: str):
        """Handle CoE write result on the main thread."""
        self._write_btn.setText("Write to Drive")
        btn_enabled = self._connection is not None
        self._write_btn.setEnabled(btn_enabled)
        self._read_btn.setEnabled(btn_enabled)

        from PySide6.QtWidgets import QMessageBox
        if error:
            QMessageBox.warning(
                self, "CoE Write Error",
                f"Failed to write drive parameters to axis {axis}:\n{error}",
            )
        else:
            failures = {k: v for k, v in results.items() if v is not None}
            if failures:
                detail = "\n".join(f"  {k}: {v}" for k, v in failures.items())
                QMessageBox.warning(
                    self, "CoE Write Partial",
                    f"Some parameters failed to write on axis {axis}:\n{detail}",
                )
            else:
                n = len(results)
                logger.info("Axis %d: wrote %d parameters OK", axis, n)

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

        # First message in conversation — include full scope data context
        if not self._conversation_history:
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
        else:
            # Follow-up — include updated drive profile + fresh metrics
            # but skip raw CSV to save tokens
            drive_block = (
                f"[Updated drive profile]\n```\n{drive_context}\n```\n\n"
                if drive_context else ""
            )
            user_content = (
                f"{drive_block}"
                f"[Updated metrics]\n```\n{metrics_text}\n```\n\n"
                f"{user_text}"
            )

        self._conversation_history.append({"role": "user", "content": user_content})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._conversation_history,
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
        # Save assistant response to conversation history
        if self._current_response:
            self._conversation_history.append(
                {"role": "assistant", "content": self._current_response}
            )
        turns = len(self._conversation_history) // 2
        self.btn_send.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_tune.setEnabled(True)
        self.status_label.setText(f"Done — turn {turns}")
        self.chat_display.append("")

    def _on_error(self, error: str):
        self._streaming = False
        # Remove the failed user message from history
        if self._conversation_history and self._conversation_history[-1]["role"] == "user":
            self._conversation_history.pop()
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

    def _new_chat(self):
        """Start a fresh conversation — clears display and history."""
        self.chat_display.clear()
        self._conversation_history.clear()
        self.status_label.setText("New conversation started")
