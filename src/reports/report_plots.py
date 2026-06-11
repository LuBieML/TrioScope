"""SVG line plots and FFT spectrum helpers for the report."""

from __future__ import annotations

from html import escape
from typing import Sequence

import numpy as np

from .report_format import _fmt_tick


def _line_plot_svg(
    *,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    title: str,
    x_label: str,
    y_label: str,
    segment_breaks: Sequence[int],
) -> str:
    width = 760
    height = 270
    left = 58
    right = 16
    top = 30
    bottom = 38
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_vals, y_vals = _finite_xy(x, y)
    if len(x_vals) < 2:
        return "<p class=\"empty\">Not enough finite samples to plot.</p>"

    x_vals, y_vals = _downsample(x_vals, y_vals, 900)
    x_min, x_max = _bounds(x_vals)
    y_min, y_max = _bounds(y_vals)

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h

    points = " ".join(f"{sx(float(a)):.2f},{sy(float(b)):.2f}" for a, b in zip(x_vals, y_vals))
    grid_lines = []
    labels = []
    for i in range(5):
        gx = left + (plot_w * i / 4)
        gy = top + (plot_h * i / 4)
        grid_lines.append(
            f"<line x1=\"{gx:.2f}\" y1=\"{top}\" x2=\"{gx:.2f}\" y2=\"{top + plot_h}\" class=\"grid\" />"
        )
        grid_lines.append(
            f"<line x1=\"{left}\" y1=\"{gy:.2f}\" x2=\"{left + plot_w}\" y2=\"{gy:.2f}\" class=\"grid\" />"
        )
        x_tick = x_min + (x_max - x_min) * i / 4
        y_tick = y_max - (y_max - y_min) * i / 4
        labels.append(
            f"<text x=\"{gx:.2f}\" y=\"{height - 18}\" text-anchor=\"middle\" class=\"tick\">{escape(_fmt_tick(x_tick))}</text>"
        )
        labels.append(
            f"<text x=\"{left - 8}\" y=\"{gy + 4:.2f}\" text-anchor=\"end\" class=\"tick\">{escape(_fmt_tick(y_tick))}</text>"
        )

    segment_lines = []
    source_len = min(len(x), len(y))
    for break_idx in segment_breaks:
        if 0 < break_idx < source_len:
            x_break = float(x[break_idx])
            if x_min <= x_break <= x_max:
                x_pos = sx(x_break)
                segment_lines.append(
                    f"<line x1=\"{x_pos:.2f}\" y1=\"{top}\" x2=\"{x_pos:.2f}\" y2=\"{top + plot_h}\" class=\"segment\" />"
                )

    return f"""
<svg class="plot" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
  <style>
    .grid {{ stroke: #2c3038; stroke-width: 1; }}
    .axis {{ stroke: #777d89; stroke-width: 1.2; }}
    .series {{ fill: none; stroke-width: 1.8; stroke-linejoin: round; stroke-linecap: round; }}
    .tick {{ fill: #a3a9b6; font: 11px Consolas, monospace; }}
    .label {{ fill: #d4d7de; font: 12px "Segoe UI", Arial, sans-serif; }}
    .title {{ fill: #e7e9ee; font: 13px "Segoe UI", Arial, sans-serif; font-weight: 650; }}
    .segment {{ stroke: #ffa500; stroke-width: 1; stroke-dasharray: 4 4; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#0a0a0a" />
  <text x="{left}" y="18" class="title">{escape(title)}</text>
  {"".join(grid_lines)}
  {"".join(segment_lines)}
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis" />
  <polyline points="{points}" class="series" stroke="{escape(color)}" />
  {"".join(labels)}
  <text x="{left + plot_w / 2}" y="{height - 4}" text-anchor="middle" class="label">{escape(x_label)}</text>
  <text x="16" y="{top + plot_h / 2}" text-anchor="middle" class="label" transform="rotate(-90 16 {top + plot_h / 2})">{escape(y_label)}</text>
</svg>""".strip()


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    x_arr = np.asarray(x[:n], dtype=float)
    y_arr = np.asarray(y[:n], dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx], y[idx]


def _bounds(values: np.ndarray) -> tuple[float, float]:
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        return 0.0, 1.0
    if v_min == v_max:
        pad = max(abs(v_min) * 0.05, 1.0)
        return v_min - pad, v_max + pad
    pad = (v_max - v_min) * 0.04
    return v_min - pad, v_max + pad


def _fft_spectrum(
    time_arr: np.ndarray,
    values: np.ndarray,
    max_samples: int = 16384,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    n = min(len(time_arr), len(values))
    if n < 4:
        return None, None
    t = np.asarray(time_arr[:n], dtype=float)
    y = np.asarray(values[:n], dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if len(t) < 4:
        return None, None
    if len(t) > max_samples:
        t = t[-max_samples:]
        y = y[-max_samples:]
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None, None
    dt = float(np.median(diffs))
    if dt <= 0:
        return None, None
    centered = y - float(np.mean(y))
    if float(np.max(np.abs(centered))) <= 0:
        return None, None
    window = np.hanning(len(centered))
    window_sum = float(np.sum(window))
    if window_sum <= 0:
        window = np.ones(len(centered))
        window_sum = float(len(centered))
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    magnitude = np.abs(np.fft.rfft(centered * window)) * 2.0 / window_sum
    if magnitude.size:
        magnitude[0] = 0.0
    return freqs, magnitude


def _fft_peaks(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    top_n: int = 5,
) -> list[tuple[float, float]]:
    if len(freqs) <= 1 or len(magnitude) <= 1:
        return []
    peaks: list[tuple[float, float]] = []
    for idx in range(1, len(magnitude) - 1):
        mag = float(magnitude[idx])
        if mag > float(magnitude[idx - 1]) and mag >= float(magnitude[idx + 1]) and mag > 0:
            peaks.append((float(freqs[idx]), mag))
    if not peaks:
        idx = int(np.argmax(magnitude[1:]) + 1)
        if float(magnitude[idx]) > 0:
            peaks.append((float(freqs[idx]), float(magnitude[idx])))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks[:top_n]
