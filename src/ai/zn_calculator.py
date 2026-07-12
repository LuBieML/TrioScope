"""Ziegler-Nichols PI calculator card for the Servo Loop Analyser panel.

Pure math lives in :func:`zn_pi_table` so it is testable without Qt.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QVBoxLayout,
)
from PySide6.QtCore import Qt

from .tuner_theme import (
    ACCENT, BG_PANEL, BORDER, CYAN, GROUP_STYLE, TEXT, TEXT_DIM, separator,
)

# (name, Ku multiplier, Tu multiplier for Ti). Ti = Tu * ti_mult.
ZN_PI_METHODS: tuple[tuple[str, float, float], ...] = (
    ("Classical ZN",     0.45,      1.0 / 1.2),  # Kp=0.45 Ku, Ti=Tu/1.2
    ("Tyreus-Luyben",    1.0 / 3.2, 2.2),        # Kp=Ku/3.2, Ti=2.2 Tu (conservative)
    ("Ciancone-Marlin",  0.303,     1.74),       # robust / low overshoot
)


def zn_pi_table(ku: float, tu_s: float) -> list[dict]:
    """PI settings for each ZN variant.

    ku: ultimate gain (Pn102 value at sustained-oscillation onset, rad/s).
    tu_s: ultimate period in seconds.
    Returns [{method, kp, ti_s, pn103}] — pn103 in the drive's 0.1 ms units.
    """
    rows: list[dict] = []
    for name, kp_mult, ti_mult in ZN_PI_METHODS:
        if ku <= 0 or tu_s <= 0:
            rows.append({"method": name, "kp": None, "ti_s": None, "pn103": None})
            continue
        kp = ku * kp_mult
        ti_s = tu_s * ti_mult
        rows.append({
            "method": name,
            "kp": kp,
            "ti_s": ti_s,
            "pn103": ti_s / 0.1e-3,  # Pn103 is in units of 0.1 ms
        })
    return rows


class ZNCalculatorCard(QGroupBox):
    """Manual-entry Ziegler-Nichols PI calculator (speed loop)."""

    def __init__(self, parent=None):
        super().__init__("Ziegler-Nichols PI Calculator", parent)
        self.setMaximumWidth(300)
        self.setStyleSheet(GROUP_STYLE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        hint = QLabel(
            "Speed loop: raise gain until sustained oscillation.\n"
            "Enter Ku (gain at onset) and Tu (period)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
        outer.addWidget(hint)

        spin_style = (
            f"QDoubleSpinBox {{ background: {BG_PANEL}; color: {TEXT};"
            f" border: 1px solid {BORDER}; border-radius: 2px;"
            f" padding: 1px 3px; font-size: 8pt; }}"
        )

        inputs = QFormLayout()
        inputs.setContentsMargins(0, 2, 0, 2)
        inputs.setSpacing(3)
        inputs.setLabelAlignment(Qt.AlignLeft)
        lbl_style = f"color: {TEXT}; font-size: 8pt;"

        self._ku = QDoubleSpinBox()
        self._ku.setRange(0.0, 100000.0)
        self._ku.setDecimals(2)
        self._ku.setValue(500.0)
        self._ku.setSuffix("  rad/s")
        self._ku.setFixedWidth(130)
        self._ku.setStyleSheet(spin_style)
        self._ku.setToolTip(
            "Ultimate gain — Pn102 value where the loop just begins "
            "sustained oscillation."
        )
        self._ku.valueChanged.connect(self._recalc)
        ku_lbl = QLabel("Ku (ultimate gain):")
        ku_lbl.setStyleSheet(lbl_style)
        inputs.addRow(ku_lbl, self._ku)

        self._tu = QDoubleSpinBox()
        self._tu.setRange(0.0, 100000.0)
        self._tu.setDecimals(2)
        self._tu.setValue(10.0)
        self._tu.setSuffix("  ms")
        self._tu.setFixedWidth(130)
        self._tu.setStyleSheet(spin_style)
        self._tu.setToolTip(
            "Ultimate period — period of the sustained oscillation, "
            "in milliseconds."
        )
        self._tu.valueChanged.connect(self._recalc)
        tu_lbl = QLabel("Tu (period):")
        tu_lbl.setStyleSheet(lbl_style)
        inputs.addRow(tu_lbl, self._tu)

        outer.addLayout(inputs)
        outer.addWidget(separator())

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        grid.setContentsMargins(0, 2, 0, 0)

        hdr_style = (
            f"color: {TEXT_DIM}; font-family: Consolas; font-size: 7pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        for col, text in enumerate(("METHOD", "Kp (Pn102)", "Ti (Pn103)")):
            h = QLabel(text)
            h.setStyleSheet(hdr_style)
            grid.addWidget(h, 0, col)

        self._result_labels: list[tuple[QLabel, QLabel]] = []
        for row, (name, _, _) in enumerate(ZN_PI_METHODS, start=1):
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"color: {TEXT}; font-size: 8pt;")
            grid.addWidget(name_lbl, row, 0)

            kp_lbl = QLabel("--")
            kp_lbl.setStyleSheet(
                f"color: {CYAN}; font-family: Consolas; font-size: 8pt;"
                f" font-weight: bold;"
            )
            kp_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(kp_lbl, row, 1)

            ti_lbl = QLabel("--")
            ti_lbl.setStyleSheet(
                f"color: {ACCENT}; font-family: Consolas; font-size: 8pt;"
                f" font-weight: bold;"
            )
            ti_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(ti_lbl, row, 2)

            self._result_labels.append((kp_lbl, ti_lbl))

        outer.addLayout(grid)

        note = QLabel(
            "Kp shown in rad/s → write to Pn102.\n"
            "Ti shown in ×0.1 ms units → write to Pn103."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt; padding-top: 4px;")
        outer.addWidget(note)

        self._recalc()

    def _recalc(self):
        rows = zn_pi_table(float(self._ku.value()),
                           float(self._tu.value()) * 1e-3)
        for (kp_lbl, ti_lbl), row in zip(self._result_labels, rows):
            if row["kp"] is None:
                kp_lbl.setText("--")
                ti_lbl.setText("--")
            else:
                kp_lbl.setText(f"{row['kp']:.1f}")
                ti_lbl.setText(f"{row['pn103']:.0f}")
