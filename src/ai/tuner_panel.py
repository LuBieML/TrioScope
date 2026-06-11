"""
Servo Loop Analyser Panel — dockable Qt widget for scope-based loop diagnostics.

Combines:
  - Drive profile editor (drive_profile_editor.DriveProfileEditor)
  - Ziegler-Nichols PI calculator (zn_calculator.ZieglerNicholsCard)
  - Velocity / position loop metric cards (loop_cards)

This module owns the panel shell and the ANALYZE entry point that feeds
captured scope data into classical_tuner.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal

from .classical_tuner import (
    ClassicalTuner, StepResponseMetrics, VelocityLoopMetrics,
)
from .drive_profile_editor import DriveProfileEditor
from .loop_cards import PositionLoopCard, VelocityLoopCard
from .tuner_theme import _ACCENT, _AMBER, _BG_DARK, _GREEN, _RED, _TEXT, _TEXT_DIM
from .zn_calculator import ZieglerNicholsCard

logger = logging.getLogger(__name__)


class TunerPanel(QDockWidget):
    """Dockable servo loop analyser panel with drive profile editor."""

    analysis_complete = Signal()

    def __init__(self, parent=None):
        super().__init__("Servo Loop Analyser", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(560)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # --- State ---
        self._data_provider: Callable | None = None
        self._pos_metrics: StepResponseMetrics | None = None
        self._vel_metrics: VelocityLoopMetrics | None = None

        self._build_ui()

    # ================================================================
    # Public API
    # ================================================================

    def set_data_provider(self, provider: Callable):
        self._data_provider = provider

    def set_connection(self, connection, conn_lock=None):
        self._profile_editor.set_connection(connection, conn_lock)

    def get_all_profiles(self) -> dict[int, dict]:
        return self._profile_editor.get_all_profiles()

    def set_all_profiles(self, profiles: dict[int, dict]):
        self._profile_editor.set_all_profiles(profiles)

    # ================================================================
    # UI construction
    # ================================================================

    def _build_ui(self):
        container = QWidget()
        container.setStyleSheet(
            f"QWidget {{ background-color: {_BG_DARK}; color: {_TEXT}; }}"
        )
        root = QVBoxLayout(container)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header ──────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(6)

        title = QLabel("SERVO LOOP ANALYSER")
        title.setStyleSheet(
            f"color: {_ACCENT}; font-family: Consolas; font-size: 11pt;"
            f" font-weight: bold; letter-spacing: 3px;"
        )
        header.addWidget(title)
        header.addStretch()

        self._btn_analyze = QPushButton("ANALYZE")
        self._btn_analyze.setFixedHeight(30)
        self._btn_analyze.setFixedWidth(110)
        self._btn_analyze.setCursor(Qt.PointingHandCursor)
        self._btn_analyze.setStyleSheet(f"""
            QPushButton {{
                background-color: {_ACCENT};
                color: #000;
                font-family: Consolas;
                font-size: 9pt;
                font-weight: bold;
                letter-spacing: 2px;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{ background-color: #ffb52e; }}
            QPushButton:pressed {{ background-color: #e09000; }}
            QPushButton:disabled {{ background-color: #4a4a4a; color: #777; }}
        """)
        self._btn_analyze.clicked.connect(self._on_analyze)
        header.addWidget(self._btn_analyze)

        root.addLayout(header)

        # ── Thin accent line ────────────────────────────────────────
        accent_line = QFrame()
        accent_line.setFixedHeight(2)
        accent_line.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f" stop:0 {_ACCENT}, stop:0.5 {_ACCENT}44, stop:1 transparent);"
        )
        root.addWidget(accent_line)

        # ── Status ──────────────────────────────────────────────────
        self._status_label = QLabel("Capture scope data, then click ANALYZE")
        self._status_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 8pt; padding: 2px 0;"
        )
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        # ── Two-column scrollable content area ──────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {_BG_DARK}; border: none; }}"
            f"QScrollBar:vertical {{ background: {_BG_DARK}; width: 8px; }}"
            f"QScrollBar::handle:vertical {{ background: #555; border-radius: 4px;"
            f" min-height: 20px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{"
            f" height: 0; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_content.setStyleSheet(f"background-color: {_BG_DARK};")
        self._scroll_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        columns = QHBoxLayout(self._scroll_content)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(8)

        # ── Left column: Drive Profile + ZN calculator ──────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.setContentsMargins(0, 0, 0, 0)

        self._profile_editor = DriveProfileEditor(autowrite=True)
        left_col.addWidget(self._profile_editor)

        self._zn_card = ZieglerNicholsCard()
        left_col.addWidget(self._zn_card)

        left_col.addStretch()
        columns.addLayout(left_col, 1)

        # ── Right column: Analysis cards ────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.setContentsMargins(0, 0, 0, 0)

        self._vel_card = VelocityLoopCard()
        right_col.addWidget(self._vel_card)

        self._pos_card = PositionLoopCard()
        right_col.addWidget(self._pos_card)

        right_col.addStretch()
        columns.addLayout(right_col, 1)

        scroll.setWidget(self._scroll_content)
        root.addWidget(scroll, 1)

        self.setWidget(container)
        self._vel_card.reset()
        self._pos_card.reset()

    # ================================================================
    # Analysis entry point
    # ================================================================

    def _on_analyze(self):
        if not self._data_provider:
            self._status_label.setText("No data provider connected")
            self._status_label.setStyleSheet(f"color: {_RED}; font-size: 8pt;")
            return

        provider_result = self._data_provider()
        if provider_result is None:
            time_arr, params, servo_period_sec = None, None, None
        elif len(provider_result) == 3:
            time_arr, params, servo_period_sec = provider_result
        else:
            time_arr, params = provider_result
            servo_period_sec = None
        if time_arr is None or params is None:
            self._status_label.setText(
                "No captured data available — run a capture first"
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        if len(time_arr) < 20:
            self._status_label.setText("Capture too short for analysis")
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        from .signal_metrics import _find_channel

        ch_dpos = _find_channel(params, "dpos", "demandposition", "targetposition")
        ch_mvel = _find_channel(
            params, "mspeed", "measuredvel", "actualvel", "vactual",
        )
        ch_dvel = _find_channel(params, "demandspeed", "demandvel", "dspeed")
        ch_fe = _find_channel(params, "drivefe", "fe", "followingerror")

        dpos = params.get(ch_dpos) if ch_dpos else None
        mvel = params.get(ch_mvel) if ch_mvel else None
        dvel_raw = params.get(ch_dvel) if ch_dvel else None
        drive_fe_raw = params.get(ch_fe) if ch_fe else None

        if dpos is None:
            self._status_label.setText(
                "Need DPOS channel for analysis. "
                "Capture demand position."
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        if drive_fe_raw is None:
            self._status_label.setText(
                "Need DRIVE_FE channel for analysis. "
                "Capture drive following error."
            )
            self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
            return

        command = np.asarray(dpos, dtype=np.float64)
        velocity = np.asarray(mvel, dtype=np.float64) if mvel is not None else None

        demand_velocity = None
        if dvel_raw is not None:
            if not servo_period_sec or servo_period_sec <= 0:
                self._status_label.setText(
                    "Servo period unknown — cannot scale DEMAND_SPEED. "
                    "Reconnect to the controller and re-capture."
                )
                self._status_label.setStyleSheet(f"color: {_AMBER}; font-size: 8pt;")
                return
            demand_velocity = np.asarray(dvel_raw, dtype=np.float64) / float(servo_period_sec)
        drive_fe = np.asarray(drive_fe_raw, dtype=np.float64)
        time_np = np.asarray(time_arr, dtype=np.float64)

        self._status_label.setText("Analyzing…")
        self._status_label.setStyleSheet(f"color: {_ACCENT}; font-size: 8pt;")

        try:
            pos_m, vel_m = ClassicalTuner.analyze_step_response(
                time_np, command, drive_fe, velocity, demand_velocity,
            )
        except Exception as exc:
            logger.exception("Step response analysis failed")
            self._status_label.setText(f"Analysis error: {exc}")
            self._status_label.setStyleSheet(f"color: {_RED}; font-size: 8pt;")
            return

        self._pos_metrics = pos_m
        self._vel_metrics = vel_m

        self._vel_card.populate(vel_m)
        self._pos_card.populate(pos_m)

        n_samples = len(time_np)
        dur_s = float(time_np[-1] - time_np[0])
        channels_used = ["DPOS", "DRIVE_FE"]
        if ch_mvel:
            channels_used.append("MSPEED")
        if ch_dvel:
            channels_used.append("DEMAND_SPEED")
        ch_str = " + ".join(channels_used)

        self._status_label.setText(
            f"Analyzed {n_samples} samples ({dur_s:.2f}s) | {ch_str}"
        )
        self._status_label.setStyleSheet(f"color: {_GREEN}; font-size: 8pt;")

        self.analysis_complete.emit()
