"""Metric display cards for the Servo Loop Analyser panel.

Each card renders one slice of the SignalMetrics result dict:
  VelocityLoopCard — velocity tracking + overshoot + oscillation
  PositionLoopCard — tolerance-band settling, ringing, damping
  FePhaseCard      — FE by motion phase + VFF / reversal / asymmetry diagnostics
"""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from .tuner_theme import (
    AMBER, CARD_STYLE, CYAN, RED, TEXT_BRIGHT, TEXT_DIM,
    HealthDot, clear_layout, health_color, metric_label, separator,
)


class _MetricCard(QFrame):
    """Card frame with a title header, optional health dot, and metric rows."""

    def __init__(self, title: str, with_dot: bool = True, parent=None):
        super().__init__(parent)
        self.setStyleSheet(CARD_STYLE)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        self.dot: HealthDot | None = HealthDot() if with_dot else None
        if self.dot:
            hdr.addWidget(self.dot)
        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"color: {TEXT_BRIGHT}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        hdr.addWidget(lbl)
        hdr.addStretch()
        self.status_lbl = QLabel("--")
        self.status_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 8pt; font-style: italic;"
        )
        hdr.addWidget(self.status_lbl)
        lay.addLayout(hdr)
        lay.addWidget(separator())

        self.rows = QVBoxLayout()
        self.rows.setSpacing(2)
        lay.addLayout(self.rows)

        self.issues_lbl = QLabel("")
        self.issues_lbl.setWordWrap(True)
        self.issues_lbl.setStyleSheet(
            f"color: {RED}; font-size: 8pt; padding: 2px 0 0 0;"
        )
        self.issues_lbl.hide()
        lay.addWidget(self.issues_lbl)

    # ---- helpers -----------------------------------------------------
    def add_row(self, name: str, value: str, unit: str = "", color: str = CYAN):
        self.rows.addLayout(metric_label(name, value, unit, color))

    def set_status(self, text: str, healthy: bool | None):
        if self.dot:
            self.dot.set_healthy(healthy)
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(
            f"color: {health_color(healthy)}; font-size: 8pt; font-style: italic;"
        )

    def set_issues(self, issues: list[str], color: str = RED):
        if issues:
            self.issues_lbl.setStyleSheet(
                f"color: {color}; font-size: 8pt; padding: 2px 0 0 0;")
            self.issues_lbl.setText("\n".join(f"⚠ {i}" for i in issues))
            self.issues_lbl.show()
        else:
            self.issues_lbl.hide()

    def clear_rows(self):
        clear_layout(self.rows)
        self.set_issues([])


class VelocityLoopCard(_MetricCard):
    def __init__(self, parent=None):
        super().__init__("VELOCITY LOOP", parent=parent)
        self.reset()

    def reset(self):
        self.clear_rows()
        self.set_status("--", None)
        self.add_row("Cruise tracking", "--")
        self.add_row("Cruise vel. err std", "--")
        self.add_row("Accel overshoot", "--", "%")
        self.add_row("Oscillation", "--")

    def populate(self, metrics: dict):
        self.clear_rows()
        vel = metrics.get("velocity") or {}
        if not vel:
            self.set_status("No velocity data", None)
            self.add_row("Status", "need MSPEED + demand velocity",
                         color=TEXT_DIM)
            return

        health = metrics.get("health", {}).get("velocity")
        issues = metrics.get("health", {}).get("velocity_issues", [])
        self.set_status("No issues" if health else "Issues detected", health)

        ratio = vel.get("cruise_velocity_reach_ratio")
        if ratio is not None:
            r_color = CYAN if 0.90 <= ratio <= 1.10 else AMBER
            self.add_row("Cruise tracking", f"{ratio:.3f}", "", r_color)

        cruise_err = vel.get("cruise_err")
        if cruise_err:
            self.add_row("Cruise vel. err std", f"{cruise_err['std']:.4g}", "u/s")

        overshoot = vel.get("velocity_overshoot_per_move", {})
        ov_pct = overshoot.get("max_pct", 0.0)
        ov_color = CYAN if ov_pct <= 15 else RED
        self.add_row("Accel overshoot", f"{ov_pct:.1f}", "%", ov_color)
        if overshoot.get("n_moves", 0) > 1:
            self.add_row("  (worst of moves)", str(overshoot["n_moves"]),
                         "", TEXT_DIM)

        osc = (metrics.get("oscillation") or {}).get("velocity_error") or {}
        if osc.get("has_significant_oscillation"):
            self.add_row("Oscillation", f"{osc.get('dominant_hz')}", "Hz", RED)
        else:
            self.add_row("Oscillation", "none", "", CYAN)

        self.set_issues(issues)


class PositionLoopCard(_MetricCard):
    def __init__(self, parent=None):
        super().__init__("POSITION LOOP", parent=parent)
        self.reset()

    def reset(self):
        self.clear_rows()
        self.set_status("--", None)
        self.add_row("Settle time (to ±band)", "--", "ms")
        self.add_row("Ringing crossings", "--")
        self.add_row("Post-move FE peak", "--", "u")
        self.add_row("Steady-state FE", "--", "u")
        self.add_row("Damping ratio", "--")
        self.add_row("Tolerance band", "--", "u")

    def populate(self, metrics: dict):
        self.clear_rows()
        settle = metrics.get("settle") or {}
        if not settle:
            self.set_status("--", None)
            self.add_row("Status", "No analysable move", color=TEXT_DIM)
            return

        health = metrics.get("health", {}).get("position")
        issues = metrics.get("health", {}).get("position_issues", [])
        self.set_status("No issues" if health else "Issues detected", health)

        ttb = settle.get("time_to_band_ms")
        if settle.get("settled_within_window") and ttb is not None:
            st_color = CYAN if ttb <= 200 else (AMBER if ttb <= 500 else RED)
            self.add_row("Settle time (to ±band)", f"{ttb:.0f}", "ms", st_color)
        else:
            self.add_row("Settle time (to ±band)", "not settled", "", RED)

        zc = settle.get("zero_crossings", 0)
        zc_color = CYAN if not settle.get("ringing") else RED
        self.add_row("Ringing crossings", str(zc), "", zc_color)

        band = settle.get("band", 0.0)
        peak = settle.get("fe_peak_during_settle", 0.0)
        if band > 0:
            peak_ratio = peak / band
            pk_color = CYAN if peak_ratio <= 2 else (
                AMBER if peak_ratio <= 5 else RED)
        else:
            pk_color = CYAN
        self.add_row("Post-move FE peak", f"{peak:.4g}", "u", pk_color)

        steady = settle.get("fe_steady_state", 0.0)
        ss_color = RED if settle.get("steady_state_offset_nonzero") else CYAN
        self.add_row("Steady-state FE", f"{steady:.4g}", "u", ss_color)

        damping = settle.get("damping_ratio")
        if damping is not None:
            self.add_row("Damping ratio", f"{damping:.3f}")
        else:
            self.add_row("Damping ratio", "— (no ringdown)", "", TEXT_DIM)

        nat = settle.get("natural_freq_hz")
        if nat:
            self.add_row("Natural freq", f"{nat:.1f}", "Hz")

        n_windows = settle.get("n_windows", 1)
        if n_windows > 1:
            self.add_row("  (worst of moves)", str(n_windows), "", TEXT_DIM)

        self.add_row(
            "Tolerance band",
            f"±{band:.4g} ({settle.get('band_source', '?')})",
            "u", TEXT_DIM)

        self.set_issues(issues)


class FePhaseCard(_MetricCard):
    """Following error by motion phase + the headline tuning diagnostics."""

    _PHASES = ("idle", "accel", "cruise", "decel", "settle", "reversal")

    def __init__(self, parent=None):
        super().__init__("FOLLOWING ERROR / DIAGNOSTICS", with_dot=False,
                         parent=parent)
        self.reset()

    def reset(self):
        self.clear_rows()
        self.status_lbl.setText("--")
        for phase in self._PHASES:
            self.add_row(f"FE {phase}", "--")

    def populate(self, metrics: dict):
        self.clear_rows()
        fe = metrics.get("fe") or {}
        if not fe:
            self.status_lbl.setText("no FE channel")
            self.add_row("Status", "No FE data", color=TEXT_DIM)
            return
        self.status_lbl.setText("rms | peak")

        for phase in self._PHASES:
            if phase in fe:
                s = fe[phase]
                self.add_row(f"FE {phase}",
                             f"{s['rms']:.4g} | {s['peak_abs']:.4g}", "u")

        hints: list[str] = []

        fit = fe.get("cruise_fe_vs_velocity")
        if fit:
            proportional = fit.get("proportional_to_velocity")
            color = AMBER if proportional else CYAN
            self.add_row("Cruise FE vs velocity",
                         f"slope {fit['slope']:.3g}", "", color)
            if proportional:
                hints.append(
                    "FE scales with speed → increase VFF_GAIN (CSP) / Pn112")

        rev = fe.get("reversal")
        cruise = fe.get("cruise")
        if rev and cruise and cruise.get("peak_abs", 0) > 0:
            ratio = rev["peak_abs"] / cruise["peak_abs"]
            color = AMBER if ratio > 5 else CYAN
            self.add_row("Reversal / cruise FE", f"{ratio:.1f}×", "", color)
            if ratio > 5:
                hints.append(
                    "FE spikes at reversals → mechanical (stiction/backlash) "
                    "or profile discontinuity, not gains")

        asym = metrics.get("asymmetry") or {}
        if "asymmetry_ratio" in asym:
            significant = asym.get("significant")
            color = AMBER if significant else CYAN
            self.add_row("Direction asymmetry",
                         f"{asym['asymmetry_ratio']:.2f}", "", color)
            if significant:
                hints.append(
                    "Direction-dependent FE → friction, backlash, or "
                    "gravity load — mechanical")

        osc = (metrics.get("oscillation") or {}).get("fe") or {}
        if osc.get("has_significant_oscillation"):
            self.add_row("FE oscillation", f"{osc.get('dominant_hz')}", "Hz", RED)
            phase_info = (metrics.get("oscillation") or {}).get(
                "current_vs_velocity_phase") or {}
            if phase_info.get("interpretation"):
                hints.append(phase_info["interpretation"])
        else:
            self.add_row("FE oscillation", "none", "", CYAN)

        self.set_issues(hints, color=AMBER)
