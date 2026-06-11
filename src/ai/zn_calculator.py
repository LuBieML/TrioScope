"""
Ziegler-Nichols PI calculator card.

Computes speed-loop Kp (→Pn102) and Ti (→Pn103) suggestions from a
measured ultimate gain Ku and oscillation period Tu, using three
classical PI tuning rules.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QDoubleSpinBox, QFormLayout, QGridLayout,
)
from PySide6.QtCore import Qt

from .tuner_theme import (
    _ACCENT, _BG_PANEL, _BORDER, _CYAN, _TEXT, _TEXT_DIM, _separator,
)


class ZieglerNicholsCard(QGroupBox):
    """Self-contained Ziegler-Nichols PI calculator group box."""

    # (Ku multiplier, Tu multiplier for Ti). Ti = Tu * ti_mult.
    _ZN_PI_METHODS: tuple[tuple[str, float, float], ...] = (
        ("Classical ZN",     0.45,  1.0 / 1.2),   # Kp=0.45 Ku, Ti=Tu/1.2
        ("Tyreus-Luyben",    1.0 / 3.2, 2.2),     # Kp=Ku/3.2, Ti=2.2 Tu  (conservative)
        ("Ciancone-Marlin",  0.303, 1.74),        # robust / low overshoot
    )

    def __init__(self, parent=None):
        super().__init__("Ziegler-Nichols PI Calculator", parent)
        self.setMaximumWidth(300)
        self.setStyleSheet(
            f"QGroupBox {{ color: {_TEXT_DIM}; font-size: 8pt;"
            f" border: 1px solid {_BORDER}; border-radius: 4px;"
            f" margin-top: 8px; padding-top: 6px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px;"
            f" padding: 0 4px; color: {_TEXT}; }}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        hint = QLabel(
            "Speed loop: raise gain until sustained oscillation.\n"
            "Enter Ku (gain at onset) and Tu (period)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 7pt;")
        outer.addWidget(hint)

        spin_style = (
            f"QDoubleSpinBox {{ background: {_BG_PANEL}; color: {_TEXT};"
            f" border: 1px solid {_BORDER}; border-radius: 2px;"
            f" padding: 1px 3px; font-size: 8pt; }}"
        )

        inputs = QFormLayout()
        inputs.setContentsMargins(0, 2, 0, 2)
        inputs.setSpacing(3)
        inputs.setLabelAlignment(Qt.AlignLeft)

        lbl_style = f"color: {_TEXT}; font-size: 8pt;"

        self._zn_ku = QDoubleSpinBox()
        self._zn_ku.setRange(0.0, 100000.0)
        self._zn_ku.setDecimals(2)
        self._zn_ku.setValue(500.0)
        self._zn_ku.setSuffix("  rad/s")
        self._zn_ku.setFixedWidth(130)
        self._zn_ku.setStyleSheet(spin_style)
        self._zn_ku.setToolTip("Ultimate gain — Pn102 value where the loop just begins sustained oscillation.")
        self._zn_ku.valueChanged.connect(self._recalc_zn)
        ku_lbl = QLabel("Ku (ultimate gain):")
        ku_lbl.setStyleSheet(lbl_style)
        inputs.addRow(ku_lbl, self._zn_ku)

        self._zn_tu = QDoubleSpinBox()
        self._zn_tu.setRange(0.0, 100000.0)
        self._zn_tu.setDecimals(2)
        self._zn_tu.setValue(10.0)
        self._zn_tu.setSuffix("  ms")
        self._zn_tu.setFixedWidth(130)
        self._zn_tu.setStyleSheet(spin_style)
        self._zn_tu.setToolTip("Ultimate period — period of the sustained oscillation, in milliseconds.")
        self._zn_tu.valueChanged.connect(self._recalc_zn)
        tu_lbl = QLabel("Tu (period):")
        tu_lbl.setStyleSheet(lbl_style)
        inputs.addRow(tu_lbl, self._zn_tu)

        outer.addLayout(inputs)
        outer.addWidget(_separator())

        # Results grid: Method | Kp (→Pn102) | Ti ms (→Pn103)
        self._zn_results_grid = QGridLayout()
        self._zn_results_grid.setHorizontalSpacing(6)
        self._zn_results_grid.setVerticalSpacing(2)
        self._zn_results_grid.setContentsMargins(0, 2, 0, 0)

        hdr_style = (
            f"color: {_TEXT_DIM}; font-family: Consolas; font-size: 7pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        for col, text in enumerate(("METHOD", "Kp (Pn102)", "Ti (Pn103)")):
            h = QLabel(text)
            h.setStyleSheet(hdr_style)
            self._zn_results_grid.addWidget(h, 0, col)

        self._zn_result_labels: list[tuple[QLabel, QLabel]] = []
        for row, (name, _, _) in enumerate(self._ZN_PI_METHODS, start=1):
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 8pt;")
            self._zn_results_grid.addWidget(name_lbl, row, 0)

            kp_lbl = QLabel("--")
            kp_lbl.setStyleSheet(
                f"color: {_CYAN}; font-family: Consolas; font-size: 8pt;"
                f" font-weight: bold;"
            )
            kp_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._zn_results_grid.addWidget(kp_lbl, row, 1)

            ti_lbl = QLabel("--")
            ti_lbl.setStyleSheet(
                f"color: {_ACCENT}; font-family: Consolas; font-size: 8pt;"
                f" font-weight: bold;"
            )
            ti_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._zn_results_grid.addWidget(ti_lbl, row, 2)

            self._zn_result_labels.append((kp_lbl, ti_lbl))

        outer.addLayout(self._zn_results_grid)

        note = QLabel(
            "Kp shown in rad/s → write to Pn102.\n"
            "Ti shown in ×0.1 ms units → write to Pn103."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 7pt; padding-top: 4px;")
        outer.addWidget(note)

        self._recalc_zn()

    def _recalc_zn(self):
        ku = float(self._zn_ku.value())
        tu_ms = float(self._zn_tu.value())
        tu_s = tu_ms * 1e-3

        for (kp_lbl, ti_lbl), (_, kp_mult, ti_mult) in zip(
            self._zn_result_labels, self._ZN_PI_METHODS,
        ):
            if ku <= 0 or tu_s <= 0:
                kp_lbl.setText("--")
                ti_lbl.setText("--")
                continue
            kp = ku * kp_mult
            ti_s = tu_s * ti_mult
            # Pn103 is in units of 0.1 ms
            pn103 = ti_s / 0.1e-3
            kp_lbl.setText(f"{kp:.1f}")
            ti_lbl.setText(f"{pn103:.0f}")
