"""Separate measurement window for TrioScope."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor, QPalette

from scope.measurements import (
    CaptureSummary,
    TraceMeasurement,
    compute_capture_summary,
    compute_trace_measurements,
)


_BG = "#1f1f24"
_PANEL = "#2a2a30"
_BORDER = "#44444d"
_TEXT = "#d4d4d4"
_DIM = "#8f8f99"
_CYAN = "#03DAC6"
_ACCENT = "#FFA500"


class MeasurementPanel(QMainWindow):
    """Live measurement table for the current capture."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measurements")
        self.setWindowFlag(Qt.Window, True)
        self.resize(920, 520)
        self.setMinimumSize(700, 360)

        self._time_arr: np.ndarray | None = None
        self._params: dict[str, np.ndarray] = {}
        self._trace_names: list[str] = []
        self._cursor_window: tuple[float, float] | None = None
        self._segment_breaks: list[int] = []
        self._last_signature = None

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(300)
        self._refresh_timer.timeout.connect(self._refresh)

        self._build_ui()
        self.clear()

    def _build_ui(self):
        container = QWidget()
        container.setStyleSheet(
            f"QWidget {{ background-color: {_BG}; color: {_TEXT}; }}"
            "QComboBox { background-color: #3a3a40; color: #d4d4d4;"
            " border: 1px solid #55555d; border-radius: 3px; padding: 3px; }"
            "QPushButton { background-color: #3a3a40; color: #d4d4d4;"
            " border: 1px solid #55555d; border-radius: 3px; padding: 5px 8px; }"
            "QPushButton:hover { background-color: #484850; }"
            "QCheckBox { color: #d4d4d4; }"
            "QTableWidget { background-color: #17171b; color: #d4d4d4;"
            " alternate-background-color: #202026;"
            " selection-background-color: #3a3a5c; selection-color: #ffffff;"
            " gridline-color: #33333a; border: 1px solid #3d3d45; }"
            "QTableWidget::item { padding: 3px; }"
            "QHeaderView::section { background-color: #2f2f36; color: #d4d4d4;"
            " border: 0; border-right: 1px solid #44444d; padding: 5px; }"
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(QLabel("Window:"))

        self.window_combo = QComboBox()
        self.window_combo.addItem("Full capture", "full")
        self.window_combo.addItem("Cursor window", "cursor")
        self.window_combo.currentIndexChanged.connect(lambda _: self.refresh_now())
        controls.addWidget(self.window_combo)

        self.auto_update_chk = QCheckBox("Auto")
        self.auto_update_chk.setChecked(True)
        controls.addWidget(self.auto_update_chk)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_now)
        controls.addWidget(self.btn_refresh)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setToolTip("Copy the measurement table to the clipboard")
        self.btn_copy.clicked.connect(self.copy_table)
        controls.addWidget(self.btn_copy)
        controls.addStretch()
        layout.addLayout(controls)

        self.summary_frame = QFrame()
        self.summary_frame.setStyleSheet(
            f"QFrame {{ background-color: {_PANEL}; border: 1px solid {_BORDER};"
            " border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }"
        )
        summary_layout = QGridLayout(self.summary_frame)
        summary_layout.setContentsMargins(8, 6, 8, 6)
        summary_layout.setHorizontalSpacing(14)
        summary_layout.setVerticalSpacing(3)

        self.summary_labels: dict[str, QLabel] = {}
        for row, (key, label) in enumerate([
            ("samples", "Samples"),
            ("duration", "Duration"),
            ("dt", "dt"),
            ("fs", "Fs"),
            ("nyquist", "Nyquist"),
            ("segments", "Segments"),
        ]):
            name = QLabel(label)
            name.setStyleSheet(f"color: {_DIM}; font-size: 8pt;")
            value = QLabel("--")
            value.setStyleSheet(
                f"color: {_CYAN}; font-family: Consolas; font-weight: bold;"
            )
            summary_layout.addWidget(name, row // 2, (row % 2) * 2)
            summary_layout.addWidget(value, row // 2, (row % 2) * 2 + 1)
            self.summary_labels[key] = value
        layout.addWidget(self.summary_frame)

        self.scope_label = QLabel("")
        self.scope_label.setStyleSheet(f"color: {_ACCENT}; font-size: 8pt;")
        self.scope_label.setWordWrap(True)
        layout.addWidget(self.scope_label)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Trace",
            "N",
            "Latest",
            "Min",
            "Max",
            "Mean",
            "RMS",
            "P-P",
            "Std",
            "Slope/s",
            "Peak Hz",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        table_palette = self.table.palette()
        table_palette.setColor(QPalette.Base, QColor("#17171b"))
        table_palette.setColor(QPalette.AlternateBase, QColor("#202026"))
        table_palette.setColor(QPalette.Text, QColor("#d4d4d4"))
        table_palette.setColor(QPalette.Highlight, QColor("#3a3a5c"))
        table_palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        self.table.setPalette(table_palette)
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, self.table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.empty_label = QLabel("")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet(f"color: {_DIM}; padding: 18px;")
        layout.addWidget(self.empty_label)

        self.setCentralWidget(container)

    def closeEvent(self, event):
        self.hide()
        event.ignore()

    def set_capture_data(
        self,
        time_arr: np.ndarray | None,
        params: Mapping[str, np.ndarray] | None,
        *,
        trace_names: Iterable[str] | None = None,
        cursor_window: tuple[float, float] | None = None,
        segment_breaks: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self._time_arr = time_arr
        self._params = dict(params or {})
        self._trace_names = list(trace_names or [])
        self._cursor_window = cursor_window
        self._segment_breaks = list(segment_breaks or [])
        if self.auto_update_chk.isChecked() and self.isVisible():
            self._schedule_refresh()

    def clear(self) -> None:
        self._time_arr = None
        self._params = {}
        self._last_signature = None
        self.scope_label.setText("No capture loaded")
        self._set_summary(None)
        self.table.setRowCount(0)
        self.empty_label.setText("Run or import a capture to see measurements.")
        self.empty_label.show()

    def refresh_now(self) -> None:
        self._last_signature = None
        self._refresh()

    def _schedule_refresh(self):
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _refresh(self):
        if self._time_arr is None or len(self._time_arr) == 0 or not self._params:
            self.clear()
            return

        time_arr, params, segment_breaks, label = self._selected_window_data()
        signature = self._signature(time_arr, params, label)
        if signature == self._last_signature:
            return
        self._last_signature = signature

        if len(time_arr) == 0:
            self._set_summary(None)
            self.table.setRowCount(0)
            self.scope_label.setText(label)
            self.empty_label.setText("The selected window contains no samples.")
            self.empty_label.show()
            return

        summary = compute_capture_summary(time_arr, segment_breaks)
        measurements = compute_trace_measurements(time_arr, params)
        self._set_summary(summary)
        self.scope_label.setText(label)
        self._populate_table(measurements)
        self.empty_label.setVisible(len(measurements) == 0)

    def _selected_window_data(self):
        assert self._time_arr is not None
        time_arr = self._time_arr
        params = self._ordered_params()
        segment_breaks = self._segment_breaks

        if self.window_combo.currentData() != "cursor":
            return time_arr, params, segment_breaks, "Full capture"

        if self._cursor_window is None:
            return (
                time_arr,
                params,
                segment_breaks,
                "Cursor window selected; enable cursors to measure a bounded window.",
            )

        t1, t2 = sorted(self._cursor_window)
        lo = int(np.searchsorted(time_arr, t1, side="left"))
        hi = int(np.searchsorted(time_arr, t2, side="right"))
        lo = max(0, min(lo, len(time_arr)))
        hi = max(lo, min(hi, len(time_arr)))
        sliced_time = time_arr[lo:hi]
        sliced_params = {k: v[lo:hi] for k, v in params.items()}
        sliced_breaks = [b - lo for b in segment_breaks if lo <= b < hi]
        return (
            sliced_time,
            sliced_params,
            sliced_breaks,
            f"Cursor window: {t1:.6f} s to {t2:.6f} s",
        )

    def _ordered_params(self) -> dict[str, np.ndarray]:
        if not self._trace_names:
            return dict(self._params)

        ordered = {
            name: self._params[name]
            for name in self._trace_names
            if name in self._params
        }
        return ordered

    def _signature(self, time_arr: np.ndarray, params: Mapping[str, np.ndarray], label: str):
        if len(time_arr) == 0:
            time_sig = (0, None, None)
        else:
            time_sig = (len(time_arr), float(time_arr[0]), float(time_arr[-1]))
        return (
            self.window_combo.currentData(),
            label,
            time_sig,
            tuple(params.keys()),
            tuple((k, len(v), float(v[-1]) if len(v) else None) for k, v in params.items()),
        )

    def _set_summary(self, summary: CaptureSummary | None):
        if summary is None:
            values = {
                "samples": "--",
                "duration": "--",
                "dt": "--",
                "fs": "--",
                "nyquist": "--",
                "segments": "--",
            }
        else:
            values = {
                "samples": str(summary.samples),
                "duration": _fmt(summary.duration_s, unit=" s"),
                "dt": _fmt(summary.dt_ms, unit=" ms"),
                "fs": _fmt(summary.sample_rate_hz, unit=" Hz"),
                "nyquist": _fmt(summary.nyquist_hz, unit=" Hz"),
                "segments": str(summary.segment_count),
            }
        for key, value in values.items():
            self.summary_labels[key].setText(value)

    def _populate_table(self, measurements: list[TraceMeasurement]):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(measurements))
        for row, measurement in enumerate(measurements):
            cells = [
                measurement.name,
                str(measurement.samples),
                _fmt(measurement.latest),
                _fmt(measurement.minimum),
                _fmt(measurement.maximum),
                _fmt(measurement.mean),
                _fmt(measurement.rms),
                _fmt(measurement.peak_to_peak),
                _fmt(measurement.std),
                _fmt(measurement.slope_per_s),
                _freq_text(measurement),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setForeground(Qt.white)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setForeground(Qt.cyan)
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)

    def copy_table(self):
        rows = []
        headers = [
            self.table.horizontalHeaderItem(i).text()
            for i in range(self.table.columnCount())
        ]
        rows.append("\t".join(headers))
        for row in range(self.table.rowCount()):
            rows.append("\t".join(
                self.table.item(row, col).text()
                if self.table.item(row, col) is not None else ""
                for col in range(self.table.columnCount())
            ))
        QApplication.clipboard().setText("\n".join(rows))
        self.scope_label.setText("Measurements copied to clipboard")


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "--"
    abs_value = abs(value)
    if abs_value != 0 and (abs_value >= 100000 or abs_value < 0.001):
        text = f"{value:.4e}"
    elif abs_value >= 1000:
        text = f"{value:.2f}"
    else:
        text = f"{value:.4f}"
    return f"{text}{unit}"


def _freq_text(measurement: TraceMeasurement) -> str:
    if measurement.dominant_freq_hz is None:
        return "--"
    return _fmt(measurement.dominant_freq_hz)
