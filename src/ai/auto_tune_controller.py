"""Qt orchestration for supervised manual drive-position auto-tuning."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

from .auto_tune import (
    AutoTuneError,
    ManualDrivePositionOptimizer,
    STAGES,
    summarize_trials,
)
from .auto_tune_card import AutoTuneCard
from .coe_io import write_single_pn_verified
from .tuner_theme import AMBER


class AutoTuneCoordinator(QObject):
    """Connect the pure optimizer, tuner UI, captures, and verified CoE I/O."""

    write_done = Signal(str, int, object, str)
    REPEATS = 3

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.card: AutoTuneCard | None = None
        self.optimizer: ManualDrivePositionOptimizer | None = None
        self.axis: int | None = None
        self.runs: list[dict] = []
        self.candidate_applied = False
        self.write_done.connect(self._on_write_done)

    @property
    def active(self) -> bool:
        return self.optimizer is not None

    def attach_card(self, card: AutoTuneCard) -> None:
        self.card = card
        card.startRequested.connect(self.start)
        card.applyRequested.connect(self.apply_candidate)
        card.stopRequested.connect(self.stop)
        card.restoreRequested.connect(self.restore_original)
        card.set_available(self.panel._connection is not None)

    def connection_changed(self) -> None:
        if self.card is not None:
            self.card.set_available(self.panel._connection is not None)

    def reject_axis_change(self, axis: int) -> bool:
        if self.axis is None or axis == self.axis:
            return False
        self.panel._set_status(f"Auto-tune is locked to axis {self.axis}.", AMBER)
        return True

    def start(self) -> None:
        panel = self.panel
        if panel._connection is None or self.card is None:
            return
        panel._save_ui_to_profile()
        axis = panel._current_axis()
        profile = panel._profiles.get(axis)
        if profile is None:
            self.card.ended(
                "Read or configure a DX drive profile first.", error=True,
                available=True,
            )
            return
        try:
            optimizer = ManualDrivePositionOptimizer(profile.to_dict())
        except AutoTuneError as exc:
            self.card.ended(str(exc), error=True, available=True)
            return

        reply = QMessageBox.question(
            panel,
            "Start Manual Drive Auto Tune",
            f"Start a supervised auto-tune on axis {axis}?\n\n"
            "Confirm that the drive closes BOTH the velocity and position "
            "loops, Pn100.0=5 is already active, travel is clear, and the "
            "E-stop is within reach.\n\n"
            "Every candidate still requires an explicit Apply and three "
            "fresh captures.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.optimizer = optimizer
        self.axis = axis
        self.runs = []
        self.candidate_applied = False
        self._set_profile_locked(True)
        self.card.begin(self.REPEATS)

    def _set_profile_locked(self, locked: bool) -> None:
        panel = self.panel
        if panel._axis_combo is not None:
            panel._axis_combo.setEnabled(not locked)
        if panel._drive_combo is not None:
            panel._drive_combo.setEnabled(not locked)
        for widget in panel._param_widgets.values():
            widget.setEnabled(not locked)
        if panel._autowrite_chk is not None:
            if locked:
                panel._autowrite_chk.setChecked(False)
            panel._autowrite_chk.setEnabled(
                not locked and panel._connection is not None
            )
        panel._update_drive_buttons()

    def record_metrics(self, metrics: dict, axis: int) -> None:
        optimizer = self.optimizer
        card = self.card
        if optimizer is None or card is None:
            return
        if axis != self.axis:
            card.write_failed(
                f"Analyzed axis {axis}, but the session is locked to axis "
                f"{self.axis}. Restore the original profile before retrying."
            )
            return
        if metrics.get("data_sufficiency") != "OK":
            return

        single = summarize_trials([metrics])
        if not single.safe:
            reason = "; ".join(single.failures)
            if optimizer.reference is None:
                self.end(f"Baseline rejected immediately: {reason}", error=True)
            elif self.candidate_applied:
                pending = optimizer.pending_candidate
                if pending is None:
                    self.end("Unsafe capture with no rollback candidate.", error=True)
                    return
                decision = optimizer.assess([metrics])
                self.runs.clear()
                self.candidate_applied = False
                self._start_write(
                    "rollback", {pending.parameter: pending.current}, decision
                )
            return

        if optimizer.reference is not None and not self.candidate_applied:
            return

        self.runs.append(metrics)
        pending = optimizer.pending_candidate
        card.collecting(
            candidate=pending if self.candidate_applied else None,
            count=len(self.runs), repeats=self.REPEATS,
        )
        if len(self.runs) < self.REPEATS:
            return

        runs = self.runs[:]
        self.runs.clear()
        if optimizer.reference is None:
            try:
                optimizer.set_baseline(runs)
            except AutoTuneError as exc:
                self.end(f"Baseline rejected: {exc}", error=True)
                return
            self._show_next_candidate()
            return

        if pending is None:
            self.end("Internal auto-tune state error: no applied candidate.",
                     error=True)
            return

        decision = optimizer.assess(runs)
        self.candidate_applied = False
        if decision.accepted:
            card.decision(decision)
            self._show_next_candidate()
            return
        self._start_write(
            "rollback", {pending.parameter: pending.current}, decision
        )

    def _show_next_candidate(self) -> None:
        if self.optimizer is None or self.card is None:
            return
        candidate = self.optimizer.next_candidate()
        if candidate is None:
            self.card.complete()
        else:
            self.card.candidate_ready(candidate)

    def apply_candidate(self) -> None:
        if self.optimizer is None or self.optimizer.pending_candidate is None:
            return
        candidate = self.optimizer.pending_candidate
        self._start_write(
            "candidate", {candidate.parameter: candidate.proposed}, candidate
        )

    def stop(self) -> None:
        if self.optimizer is None:
            return
        pending = self.optimizer.pending_candidate
        if self.candidate_applied and pending is not None:
            self._start_write("stop", {pending.parameter: pending.current}, pending)
            return
        self.end("Auto-tune stopped; accepted values were retained.")

    def restore_original(self) -> None:
        if self.optimizer is None or self.axis is None:
            return
        profile = self.panel._profiles.get(self.axis)
        if profile is None:
            return
        values = {
            spec.parameter: int(self.optimizer.original_profile[spec.parameter])
            for spec in STAGES
            if getattr(profile, spec.parameter, None)
            != self.optimizer.original_profile[spec.parameter]
        }
        if not values:
            self.end("Original profile was already active.")
            return
        self._start_write("restore", values, None)

    def _start_write(self, action: str, values: dict[str, int], context) -> None:
        panel = self.panel
        if panel._connection is None or self.axis is None or self.card is None:
            return
        axis = self.axis
        connection = panel._connection
        conn_lock = panel._conn_lock
        self.card.busy(
            "Restoring verified drive value…"
            if action in ("rollback", "restore", "stop")
            else "Writing and verifying candidate…"
        )

        def _do_write() -> None:
            results = {}
            for attr, value in values.items():
                try:
                    results[attr] = write_single_pn_verified(
                        connection, axis, attr, value, conn_lock=conn_lock,
                    )
                except Exception as exc:
                    results[attr] = exc
            payload = {"values": values, "context": context, "results": results}
            self.write_done.emit(action, axis, payload, "")

        threading.Thread(
            target=_do_write,
            name=f"AutoTune{action.title()}Write",
            daemon=True,
        ).start()

    def _on_write_done(self, action: str, axis: int, payload, error: str) -> None:
        if self.card is None or self.optimizer is None:
            return
        results = payload.get("results") or {}
        failures = {
            attr: result for attr, result in results.items()
            if isinstance(result, Exception)
        }
        if error or failures:
            detail = error or "; ".join(
                f"{attr.upper()}: {exc}" for attr, exc in failures.items()
            )
            self.card.write_failed(
                "Drive write/readback failed. Hardware state may be uncertain; "
                f"use Restore original or read the drive profile. {detail}"
            )
            return

        values = payload.get("values") or {}
        self._apply_profile_values(axis, values)
        context = payload.get("context")
        if action == "candidate":
            self.candidate_applied = True
            self.runs.clear()
            self.card.candidate_applied(context, self.REPEATS)
        elif action == "rollback":
            self.card.decision(context)
            self._show_next_candidate()
        elif action == "stop":
            self.end("Auto-tune stopped; the untested candidate was rolled back.")
        elif action == "restore":
            self.end("Original drive profile restored and verified.")

    def _apply_profile_values(self, axis: int, values: dict[str, int]) -> None:
        panel = self.panel
        profile = panel._profiles.get(axis)
        if profile is None:
            return
        for attr, value in values.items():
            setattr(profile, attr, int(value))
            widget = panel._param_widgets.get(attr)
            if widget is not None and axis == panel._current_axis():
                widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(False)

    def end(self, message: str, *, error: bool = False) -> None:
        card = self.card
        self.optimizer = None
        self.axis = None
        self.runs.clear()
        self.candidate_applied = False
        self._set_profile_locked(False)
        if card is not None:
            card.ended(
                message, error=error,
                available=self.panel._connection is not None,
            )
