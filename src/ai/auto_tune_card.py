"""Supervised controls for the manual drive-position auto-tuner."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from .auto_tune import TrialDecision, TuneCandidate
from .tuner_theme import ACCENT, AMBER, GREEN, GROUP_STYLE, RED, TEXT, TEXT_DIM


class AutoTuneCard(QGroupBox):
    """Small UI surface; all decisions and hardware operations live elsewhere."""

    startRequested = Signal()
    applyRequested = Signal()
    stopRequested = Signal()
    restoreRequested = Signal()

    def __init__(self, parent=None):
        super().__init__("Manual Drive-Position Auto Tune", parent)
        self.setMaximumWidth(300)
        self.setStyleSheet(GROUP_STYLE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(5)

        scope = QLabel(
            "DX3/DX4 · Pn100.0=5 · fixed Pn106\n"
            "Pn102 → Pn103 → Pn104 → Pn112 → Pn114"
        )
        scope.setWordWrap(True)
        scope.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
        outer.addWidget(scope)

        self.status = QLabel("Ready. Three fresh captures are required per value.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8pt;")
        outer.addWidget(self.status)

        self.candidate = QLabel("No candidate")
        self.candidate.setWordWrap(True)
        self.candidate.setStyleSheet(
            f"color: {TEXT}; font-family: Consolas; font-size: 9pt;"
        )
        outer.addWidget(self.candidate)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.startRequested)
        row.addWidget(self.start_btn)
        self.apply_btn = QPushButton("Apply candidate")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.applyRequested)
        row.addWidget(self.apply_btn, 1)
        outer.addLayout(row)

        recovery = QHBoxLayout()
        recovery.setSpacing(4)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stopRequested)
        recovery.addWidget(self.stop_btn)
        self.restore_btn = QPushButton("Restore original")
        self.restore_btn.setEnabled(False)
        self.restore_btn.clicked.connect(self.restoreRequested)
        recovery.addWidget(self.restore_btn, 1)
        outer.addLayout(recovery)

    def set_available(self, available: bool) -> None:
        if not self.stop_btn.isEnabled():
            self.start_btn.setEnabled(available)

    def begin(self, repeats: int) -> None:
        self.start_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.restore_btn.setEnabled(True)
        self.candidate.setText("Baseline")
        self._status(
            f"Run a fresh identical move/capture, then ANALYZE (0/{repeats}).",
            AMBER,
        )

    def collecting(self, *, candidate: TuneCandidate | None,
                   count: int, repeats: int) -> None:
        self.apply_btn.setEnabled(False)
        if candidate is None:
            self.candidate.setText("Baseline")
            label = "baseline"
        else:
            self._show_candidate(candidate)
            label = f"{candidate.label}={candidate.proposed}"
        self._status(
            f"Collecting {label}: {count}/{repeats}. Run a fresh capture "
            "before each ANALYZE.",
            AMBER,
        )

    def candidate_ready(self, candidate: TuneCandidate) -> None:
        self._show_candidate(candidate)
        self.apply_btn.setEnabled(True)
        self._status(
            "Candidate is not on the drive yet. Review it, then Apply candidate.",
            ACCENT,
        )

    def candidate_applied(self, candidate: TuneCandidate, repeats: int) -> None:
        self._show_candidate(candidate)
        self.apply_btn.setEnabled(False)
        self._status(
            f"Verified on drive. Run {repeats} fresh identical captures and "
            "ANALYZE each result.",
            AMBER,
        )

    def decision(self, decision: TrialDecision) -> None:
        verdict = "ACCEPTED" if decision.accepted else "REJECTED — restored"
        color = GREEN if decision.accepted else RED
        self._status(f"{verdict}: {decision.reason}", color)

    def complete(self) -> None:
        self.apply_btn.setEnabled(False)
        self.candidate.setText("Complete")
        self._status("All feedback/feedforward stages are complete.", GREEN)

    def busy(self, message: str) -> None:
        self.start_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.restore_btn.setEnabled(False)
        self._status(message, ACCENT)

    def ended(self, message: str, *, error: bool = False,
              available: bool = True) -> None:
        self.start_btn.setEnabled(available)
        self.apply_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.restore_btn.setEnabled(False)
        self.candidate.setText("No candidate")
        self._status(message, RED if error else TEXT_DIM)

    def write_failed(self, message: str) -> None:
        """Keep recovery available when hardware state may be uncertain."""
        self.start_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.restore_btn.setEnabled(True)
        self._status(message, RED)

    def _show_candidate(self, candidate: TuneCandidate) -> None:
        sign = "+" if candidate.direction > 0 else "−"
        step = (
            f"{candidate.step * 100:g}%"
            if candidate.parameter in ("pn102", "pn103", "pn104")
            else f"{candidate.step:g}"
        )
        self.candidate.setText(
            f"{candidate.label}: {candidate.current} → {candidate.proposed} "
            f"({sign}{step}, trial {candidate.trial_number})"
        )

    def _status(self, message: str, color: str) -> None:
        self.status.setText(message)
        self.status.setStyleSheet(f"color: {color}; font-size: 8pt;")
