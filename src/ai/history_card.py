"""Tuning-run history table for the Servo Loop Analyser panel.

One row per ANALYZE, newest first. Each KPI cell is coloured against the
most recent earlier run on the same axis (green = improved, red = worse),
and the last column summarises which Pn parameters changed between the two
runs — the "I changed Pn102 500→600 and settle went 180→95 ms" view.
Clicking a row recalls that run into the metric cards.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from .tuner_theme import (
    AMBER, BG_CARD, BG_DARK, BG_PANEL, BORDER, CARD_STYLE, CYAN, GREEN, RED,
    TEXT, TEXT_BRIGHT, TEXT_DIM, separator,
)
from .tuning_history import (
    KPI_DEFS, TuningHistory, TuningRun, compare_runs, format_kpi, pn_changes,
)

_VERDICT_COLOR = {"better": GREEN, "worse": RED, "same": TEXT, None: CYAN}

_BTN_STYLE = (
    f"QPushButton {{ background-color: {BG_PANEL}; color: {TEXT};"
    f" border: 1px solid {BORDER}; border-radius: 3px;"
    f" font-size: 8pt; padding: 2px 10px; }}"
    f"QPushButton:hover {{ border-color: {TEXT_DIM}; }}"
    f"QPushButton:disabled {{ color: #666; }}"
)


class HistoryCard(QFrame):
    """Full-width run-history table with per-KPI improvement colouring."""

    run_selected = Signal(object)   # TuningRun

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(CARD_STYLE)
        self._history: TuningHistory | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        title = QLabel("TUNING HISTORY")
        title.setStyleSheet(
            f"color: {TEXT_BRIGHT}; font-family: Consolas; font-size: 9pt;"
            f" font-weight: bold; letter-spacing: 1px;"
        )
        hdr.addWidget(title)
        self._count_lbl = QLabel("no runs yet")
        self._count_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 8pt; font-style: italic;")
        hdr.addWidget(self._count_lbl)
        hdr.addStretch()

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setStyleSheet(_BTN_STYLE)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        hdr.addWidget(self._export_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setStyleSheet(_BTN_STYLE)
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._on_clear)
        hdr.addWidget(self._clear_btn)

        lay.addLayout(hdr)
        lay.addWidget(separator())

        headers = (["Time", "Ax"]
                   + [f"{kpi.label}\n{kpi.unit}" if kpi.unit else kpi.label
                      for kpi in KPI_DEFS]
                   + ["Δ Pn (vs previous run)"])
        self._table = QTableWidget(0, len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(90)
        self._table.setMaximumHeight(220)
        self._table.setStyleSheet(
            f"QTableWidget {{ background-color: {BG_DARK};"
            f" alternate-background-color: {BG_CARD}; color: {TEXT};"
            f" border: 1px solid {BORDER}; font-size: 8pt;"
            f" font-family: Consolas; }}"
            f"QTableWidget::item {{ padding: 1px 4px; border: none; }}"
            f"QTableWidget::item:selected {{ background: {BORDER}; }}"
            f"QHeaderView::section {{ background-color: {BG_PANEL};"
            f" color: {TEXT_DIM}; border: none; padding: 2px 4px;"
            f" font-size: 7pt; font-weight: bold; }}"
            f"QScrollBar:horizontal {{ background: {BG_DARK}; height: 8px; }}"
            f"QScrollBar:vertical {{ background: {BG_DARK}; width: 8px; }}"
            f"QScrollBar::handle {{ background: #555; border-radius: 4px; }}"
            f"QScrollBar::add-line, QScrollBar::sub-line {{"
            f" width: 0; height: 0; }}"
        )
        header_view = self._table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
        header_view.setStretchLastSection(True)
        self._table.cellClicked.connect(self._on_row_clicked)
        lay.addWidget(self._table)

        hint = QLabel(
            "Green = improved vs the previous run on the same axis, "
            "red = worse. Click a row to recall that run's full analysis."
        )
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 7pt;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

    # ---------------------------------------------------------------- public
    def refresh(self, history: TuningHistory):
        """Rebuild the table from the history (newest run in the top row)."""
        self._history = history
        runs = history.runs
        self._table.setRowCount(len(runs))
        self._count_lbl.setText(
            f"{len(runs)} run(s)" if runs else "no runs yet")
        self._export_btn.setEnabled(bool(runs))
        self._clear_btn.setEnabled(bool(runs))

        for row_idx, run in enumerate(reversed(runs)):
            prev = history.previous_for(run)
            verdicts = compare_runs(prev, run)
            self._fill_row(row_idx, run, prev, verdicts)

    # --------------------------------------------------------------- internal
    def _fill_row(self, row_idx: int, run: TuningRun,
                  prev: TuningRun | None, verdicts: dict):
        ctx = run.context
        time_item = QTableWidgetItem(run.timestamp)
        time_item.setForeground(QColor(TEXT_BRIGHT))
        time_item.setToolTip(
            f"{ctx.get('n_samples')} samples, {ctx.get('duration_s')}s, "
            f"{ctx.get('n_moves')} move(s)\n"
            f"band ±{ctx.get('band')} ({ctx.get('band_source')})")
        time_item.setData(Qt.UserRole, run)
        self._table.setItem(row_idx, 0, time_item)

        axis_item = QTableWidgetItem(str(run.axis))
        axis_item.setForeground(QColor(TEXT))
        self._table.setItem(row_idx, 1, axis_item)

        for col_offset, kpi in enumerate(KPI_DEFS):
            value = run.kpis.get(kpi.key)
            verdict = verdicts.get(kpi.key)
            text = format_kpi(value, kpi)
            prev_value = prev.kpis.get(kpi.key) if prev else None
            if verdict in ("better", "worse") and value is not None \
                    and prev_value is not None:
                text += " ▼" if value < prev_value else " ▲"
            item = QTableWidgetItem(text)
            item.setForeground(QColor(_VERDICT_COLOR.get(verdict, TEXT)))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if prev is not None:
                item.setToolTip(
                    f"{kpi.label}: {format_kpi(prev_value, kpi)} → "
                    f"{format_kpi(value, kpi)} {kpi.unit}".rstrip())
            self._table.setItem(row_idx, 2 + col_offset, item)

        if prev is None:
            changes_text = "baseline" if run.pn_snapshot else "—"
            color = TEXT_DIM
        else:
            changes = pn_changes(prev.pn_snapshot, run.pn_snapshot)
            if run.pn_snapshot is None:
                changes_text, color = "no Pn profile", TEXT_DIM
            elif changes:
                changes_text, color = ", ".join(changes), AMBER
            else:
                changes_text, color = "no changes", TEXT_DIM
        changes_item = QTableWidgetItem(changes_text)
        changes_item.setForeground(QColor(color))
        changes_item.setToolTip(changes_text)
        self._table.setItem(row_idx, 2 + len(KPI_DEFS), changes_item)

    def _on_row_clicked(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if item is None:
            return
        run = item.data(Qt.UserRole)
        if run is not None:
            self.run_selected.emit(run)

    def _on_clear(self):
        if self._history is None:
            return
        reply = QMessageBox.question(
            self, "Clear history",
            "Discard all recorded tuning runs?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._history.clear()
            self.refresh(self._history)

    def _on_export(self):
        if self._history is None or not len(self._history):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export tuning history", "tuning_history.csv",
            "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                fh.write(self._history.to_csv())
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
