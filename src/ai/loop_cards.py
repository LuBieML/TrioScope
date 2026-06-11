"""
Velocity-loop and position-loop metric cards for the Servo Loop Analyser.

Each card renders the metrics computed by classical_tuner with a health
dot and per-metric colour coding; reset() restores the placeholder rows.
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel

from .classical_tuner import StepResponseMetrics, VelocityLoopMetrics
from .tuner_theme import (
    _AMBER, _BG_CARD, _BORDER, _CYAN, _RED, _TEXT_BRIGHT, _TEXT_DIM,
    _HealthDot, _health_color, _metric_label, _separator,
    clear_layout,
)


class _LoopCard(QFrame):
    """Common card chrome: health dot, title, separator, metrics layout."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background-color: {_BG_CARD}; border: 1px solid {_BORDER};"
            f" border-radius: 6px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        self._dot = _HealthDot()
        hdr.addWidget(self._dot)
        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"color: {_TEXT_BRIGHT}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        hdr.addWidget(lbl)
        self._header_layout = hdr
        hdr.addStretch()
        lay.addLayout(hdr)
        lay.addWidget(_separator())

        self._metrics_layout = QVBoxLayout()
        self._metrics_layout.setSpacing(2)
        lay.addLayout(self._metrics_layout)
        self._body_layout = lay

    def _add_metric_row(self, name, value, unit="", color=_CYAN):
        self._metrics_layout.addLayout(_metric_label(name, value, unit, color))


class VelocityLoopCard(_LoopCard):
    """Velocity loop metrics card with status text and issue list."""

    def __init__(self, parent=None):
        super().__init__("VELOCITY LOOP", parent)

        self._status_lbl = QLabel("--")
        self._status_lbl.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 8pt; font-style: italic;"
        )
        # Insert before the trailing stretch added by _LoopCard
        self._header_layout.addWidget(self._status_lbl)

        self._issues_label = QLabel("")
        self._issues_label.setWordWrap(True)
        self._issues_label.setStyleSheet(
            f"color: {_RED}; font-size: 8pt; padding: 2px 0 0 0;"
        )
        self._issues_label.hide()
        self._body_layout.addWidget(self._issues_label)

    def reset(self):
        self._dot.set_healthy(None)
        self._status_lbl.setText("--")
        self._issues_label.hide()
        clear_layout(self._metrics_layout)
        self._add_metric_row("Accel overshoot", "--", "%")
        self._add_metric_row("Cruise tracking", "--")
        self._add_metric_row("Settle time", "--", "ms")
        self._add_metric_row("Oscillations", "--")

    def populate(self, vm: VelocityLoopMetrics | None):
        clear_layout(self._metrics_layout)
        if vm is None:
            self._dot.set_healthy(None)
            self._status_lbl.setText("No MSPEED data")
            self._issues_label.hide()
            self._add_metric_row("Status", "No velocity data", color=_TEXT_DIM)
            return

        self._dot.set_healthy(vm.is_healthy)
        self._status_lbl.setText(
            "No issues" if vm.is_healthy else "Issues detected"
        )
        self._status_lbl.setStyleSheet(
            f"color: {_health_color(vm.is_healthy)}; font-size: 8pt;"
            f" font-style: italic;"
        )

        ov_color = _CYAN if vm.accel_overshoot_pct <= 15 else _RED
        self._add_metric_row(
            "Accel overshoot", f"{vm.accel_overshoot_pct:.1f}", "%", ov_color,
        )

        ratio = vm.cruise_tracking_ratio
        r_color = _CYAN if 0.90 <= ratio <= 1.10 else _AMBER
        self._add_metric_row("Cruise tracking", f"{ratio:.3f}", "", r_color)

        st_color = _CYAN if vm.accel_settle_time_ms <= 100 else _AMBER
        self._add_metric_row(
            "Settle time", f"{vm.accel_settle_time_ms:.0f}", "ms", st_color,
        )

        osc_color = _CYAN if vm.accel_oscillation_count <= 3 else _RED
        self._add_metric_row(
            "Oscillations", str(vm.accel_oscillation_count), "", osc_color,
        )

        self._add_metric_row(
            "Cruise vel. std", f"{vm.cruise_velocity_std:.3f}", "", _CYAN,
        )

        if vm.issues:
            self._issues_label.setText(
                "\n".join(f"⚠ {iss}" for iss in vm.issues)
            )
            self._issues_label.show()
        else:
            self._issues_label.hide()


class PositionLoopCard(_LoopCard):
    """Position loop metrics card."""

    def __init__(self, parent=None):
        super().__init__("POSITION LOOP", parent)

    def reset(self):
        self._dot.set_healthy(None)
        clear_layout(self._metrics_layout)
        self._add_metric_row("Overshoot", "--", "%")
        self._add_metric_row("Settling time", "--", "ms")
        self._add_metric_row("Rise time", "--", "ms")
        self._add_metric_row("Oscillations", "--")
        self._add_metric_row("Steady-state err", "--")
        self._add_metric_row("Drive FE (peak)", "--", "u")
        self._add_metric_row("Drive FE (cruise)", "--", "u")
        self._add_metric_row("Damping ratio", "--")

    def populate(self, pm: StepResponseMetrics):
        clear_layout(self._metrics_layout)

        is_empty = (
            pm.overshoot_pct == 0 and pm.oscillation_count == 0
            and pm.settling_time_ms == 0
        )
        if is_empty:
            self._dot.set_healthy(None)
            self._add_metric_row("Status", "No analysable move", color=_TEXT_DIM)
            return

        # Simple health indicator from raw metrics (no scoring/verdict).
        healthy = (
            pm.overshoot_pct <= 15
            and pm.oscillation_count <= 3
            and pm.settling_time_ms <= 500
            and pm.drive_fe_peak_pct <= 1.5
        )
        self._dot.set_healthy(healthy)

        ov_color = _CYAN if pm.overshoot_pct <= 5 else (
            _AMBER if pm.overshoot_pct <= 15 else _RED
        )
        self._add_metric_row("Overshoot", f"{pm.overshoot_pct:.1f}", "%", ov_color)

        st_color = _CYAN if pm.settling_time_ms <= 200 else (
            _AMBER if pm.settling_time_ms <= 500 else _RED
        )
        self._add_metric_row(
            "Settling time", f"{pm.settling_time_ms:.0f}", "ms", st_color,
        )

        self._add_metric_row("Rise time", f"{pm.rise_time_ms:.0f}", "ms")

        osc_color = _CYAN if pm.oscillation_count <= 1 else (
            _AMBER if pm.oscillation_count <= 3 else _RED
        )
        self._add_metric_row(
            "Oscillations", str(pm.oscillation_count), "", osc_color,
        )

        ss_pct = pm.steady_state_error * 100
        ss_color = _CYAN if ss_pct <= 2 else _AMBER
        self._add_metric_row("Steady-state err", f"{ss_pct:.2f}", "%", ss_color)

        fe_color = _CYAN if pm.drive_fe_peak_pct <= 0.2 else (
            _AMBER if pm.drive_fe_peak_pct <= 0.5 else _RED
        )
        self._add_metric_row(
            "Drive FE (peak)", f"{pm.drive_fe_peak:.4g}", "u", fe_color,
        )

        fe_cruise_color = _CYAN if pm.drive_fe_cruise_mean_pct <= 0.2 else (
            _AMBER if pm.drive_fe_cruise_mean_pct <= 0.5 else _RED
        )
        self._add_metric_row(
            "Drive FE (cruise)", f"{pm.drive_fe_cruise_mean:.4g}", "u", fe_cruise_color,
        )

        self._add_metric_row("Damping ratio", f"{pm.damping_ratio:.3f}", "")

        if pm.natural_freq_est_hz > 0:
            self._add_metric_row(
                "Natural freq", f"{pm.natural_freq_est_hz:.1f}", "Hz",
            )
