"""Self-contained HTML commissioning reports for TrioScope captures."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from scope.measurements import (
        TraceMeasurement,
        compute_capture_summary,
        compute_trace_measurements,
    )
except ImportError:  # pragma: no cover - used by tests importing through src.*
    from src.scope.measurements import (
        TraceMeasurement,
        compute_capture_summary,
        compute_trace_measurements,
    )


_DEFAULT_COLORS = [
    "#03DAC6",
    "#FFA500",
    "#4FC3F7",
    "#E57373",
    "#BA68C8",
    "#AED581",
    "#FFD54F",
    "#90A4AE",
]

_MEASUREMENT_HEADERS = [
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
    "Dominant Hz",
    "Dominant Mag",
]

_PROFILE_LABELS = {
    "drive_type": "Drive",
    "pn100": "Pn100",
    "pn100_tuning_mode": "Pn100.0 Tuning Mode",
    "pn100_vibration": "Pn100.2 Vibration",
    "pn100_damping": "Pn100.3 Damping",
    "pn101": "Pn101 Servo Rigidity",
    "pn102": "Pn102 Speed Loop Gain",
    "pn103": "Pn103 Speed Loop Ti",
    "pn104": "Pn104 Position Loop Gain",
    "pn105": "Pn105",
    "pn106": "Pn106 Load Inertia",
    "pn112": "Pn112 Speed Feedforward",
    "pn113": "Pn113 Speed FF Filter",
    "pn114": "Pn114 Torque Feedforward",
    "pn115": "Pn115 Torque FF Filter",
    "pn135": "Pn135 Speed Filter",
}


def build_html_report(
    *,
    time_arr: np.ndarray,
    params: Mapping[str, np.ndarray],
    trace_order: Sequence[str] | None = None,
    trace_colors: Mapping[str, str] | None = None,
    trace_fft_flags: Mapping[str, bool] | None = None,
    controller_metadata: Mapping[str, object] | None = None,
    drive_metadata: Mapping[str, object] | None = None,
    drive_profiles: Mapping[int, Mapping[str, object]] | None = None,
    user_notes: str = "",
    generated_at: datetime | None = None,
    segment_breaks: Sequence[int] | None = None,
    title: str = "TrioScope Commissioning Report",
) -> str:
    """Return a complete standalone HTML report for a scope capture."""

    if time_arr is None or len(time_arr) == 0:
        raise ValueError("Cannot build a report without time samples")
    if not params:
        raise ValueError("Cannot build a report without captured parameters")

    generated = generated_at or datetime.now().astimezone()
    ordered_params = _ordered_params(params, trace_order)
    summary = compute_capture_summary(np.asarray(time_arr), list(segment_breaks or []))
    measurements = compute_trace_measurements(np.asarray(time_arr), ordered_params)
    colors = _color_map(ordered_params, trace_colors)
    fft_flags = dict(trace_fft_flags or {})

    sections = [
        _hero_section(title, generated, summary),
        _notes_section(user_notes),
        _metadata_section("Controller Metadata", controller_metadata or {}),
        _metadata_section("Drive Metadata", drive_metadata or {}),
        _drive_profiles_section(drive_profiles or {}),
        _summary_section(summary),
        _measurements_section(measurements),
        _plots_section(
            np.asarray(time_arr),
            ordered_params,
            colors,
            fft_flags,
            list(segment_breaks or []),
        ),
    ]

    return "\n".join([
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{escape(title)}</title>",
        _style_block(),
        "</head>",
        "<body>",
        "<main>",
        *sections,
        "</main>",
        "</body>",
        "</html>",
    ])


def write_html_report(path: str | Path, **kwargs) -> Path:
    """Write a standalone HTML report and return the resolved path."""

    target = Path(path)
    if target.suffix.lower() not in {".html", ".htm"}:
        target = target.with_suffix(".html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_html_report(**kwargs), encoding="utf-8")
    return target


def _ordered_params(
    params: Mapping[str, np.ndarray],
    trace_order: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    ordered: dict[str, np.ndarray] = {}
    if trace_order:
        for name in trace_order:
            if name in params and name not in ordered:
                ordered[name] = np.asarray(params[name])
    for name, values in params.items():
        if name not in ordered:
            ordered[name] = np.asarray(values)
    return ordered


def _style_block() -> str:
    return """
<style>
:root {
  color-scheme: dark;
  --bg: #101114;
  --panel: #191b20;
  --panel-2: #20232a;
  --border: #353943;
  --text: #e7e9ee;
  --muted: #a3a9b6;
  --accent: #03dac6;
  --warning: #ffa500;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 "Segoe UI", Arial, sans-serif;
}
main { max-width: 1240px; margin: 0 auto; padding: 28px; }
section {
  border-top: 1px solid var(--border);
  padding: 22px 0;
}
h1, h2, h3 { margin: 0; font-weight: 650; letter-spacing: 0; }
h1 { font-size: 30px; }
h2 { font-size: 18px; margin-bottom: 12px; }
h3 { font-size: 14px; color: var(--muted); margin-bottom: 8px; }
.hero {
  border-top: 0;
  padding-top: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 18px;
  align-items: end;
}
.subtitle { color: var(--muted); margin: 6px 0 0; }
.stamp {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  min-width: 240px;
}
.stamp .label, .metric .label { color: var(--muted); font-size: 12px; }
.stamp .value, .metric .value {
  color: var(--accent);
  font-family: Consolas, "Cascadia Mono", monospace;
  font-weight: 650;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}
.metric {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
}
.grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}
.note {
  white-space: pre-wrap;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
}
.empty { color: var(--muted); }
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}
th, td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
th {
  color: var(--muted);
  background: var(--panel-2);
  font-size: 12px;
  font-weight: 650;
}
td.num {
  text-align: right;
  font-family: Consolas, "Cascadia Mono", monospace;
  color: var(--accent);
  white-space: nowrap;
}
tr:last-child td { border-bottom: 0; }
.plot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
  gap: 16px;
}
.plot-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
}
.plot-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.swatch {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
svg.plot {
  width: 100%;
  height: auto;
  display: block;
  background: #0a0a0a;
  border: 1px solid #2c3038;
  border-radius: 4px;
}
.plot-pair {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}
.small-table { margin-top: 8px; font-size: 12px; }
@media print {
  body { background: #ffffff; color: #111111; }
  main { max-width: none; padding: 14mm; }
  section, .plot-card, table, .metric, .stamp, .note { break-inside: avoid; }
}
</style>""".strip()


def _hero_section(title: str, generated: datetime, summary) -> str:
    generated_text = generated.strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"""
<section class="hero">
  <div>
    <h1>{escape(title)}</h1>
    <p class="subtitle">Commissioning and support record generated from the current TrioScope capture.</p>
  </div>
  <div class="stamp">
    <div class="label">Generated</div>
    <div class="value">{escape(generated_text)}</div>
    <div class="label" style="margin-top: 6px;">Samples</div>
    <div class="value">{summary.samples}</div>
  </div>
</section>""".strip()


def _notes_section(notes: str) -> str:
    if notes.strip():
        body = f"<div class=\"note\">{escape(notes.strip())}</div>"
    else:
        body = "<p class=\"empty\">No user notes entered.</p>"
    return f"<section><h2>User Notes</h2>{body}</section>"


def _metadata_section(title: str, metadata: Mapping[str, object]) -> str:
    rows = [
        (str(key), _value_to_text(value))
        for key, value in metadata.items()
        if value is not None and _value_to_text(value) != ""
    ]
    if not rows:
        body = "<p class=\"empty\">No metadata available.</p>"
    else:
        body = _key_value_table(rows)
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _drive_profiles_section(
    drive_profiles: Mapping[int, Mapping[str, object]],
) -> str:
    rows: list[list[str]] = []
    keys = [
        "drive_type",
        "pn100",
        "pn100_tuning_mode",
        "pn100_vibration",
        "pn100_damping",
        "pn101",
        "pn102",
        "pn103",
        "pn104",
        "pn105",
        "pn106",
        "pn112",
        "pn113",
        "pn114",
        "pn115",
        "pn135",
    ]
    for axis, profile in sorted(drive_profiles.items(), key=lambda item: int(item[0])):
        if not profile:
            continue
        for key in keys:
            if key not in profile:
                continue
            value = _value_to_text(profile.get(key))
            if value == "" or value.lower() == "none":
                continue
            rows.append([str(axis), _PROFILE_LABELS.get(key, key), value])

    if not rows:
        body = "<p class=\"empty\">No drive profile parameters configured.</p>"
    else:
        body = _table(["Axis", "Parameter", "Value"], rows, numeric_cols={0, 2})
    return f"<section><h2>Drive Parameters</h2>{body}</section>"


def _summary_section(summary) -> str:
    cells = [
        ("Samples", str(summary.samples)),
        ("Duration", _fmt(summary.duration_s, " s")),
        ("dt", _fmt(summary.dt_ms, " ms")),
        ("Sample Rate", _fmt(summary.sample_rate_hz, " Hz")),
        ("Nyquist", _fmt(summary.nyquist_hz, " Hz")),
        ("Segments", str(summary.segment_count)),
    ]
    body = "\n".join(
        f"<div class=\"metric\"><div class=\"label\">{escape(label)}</div>"
        f"<div class=\"value\">{escape(value)}</div></div>"
        for label, value in cells
    )
    return f"<section><h2>Capture Summary</h2><div class=\"metrics\">{body}</div></section>"


def _measurements_section(measurements: Sequence[TraceMeasurement]) -> str:
    rows = [
        [
            m.name,
            str(m.samples),
            _fmt(m.latest),
            _fmt(m.minimum),
            _fmt(m.maximum),
            _fmt(m.mean),
            _fmt(m.rms),
            _fmt(m.peak_to_peak),
            _fmt(m.std),
            _fmt(m.slope_per_s),
            _fmt(m.dominant_freq_hz, " Hz"),
            _fmt(m.dominant_magnitude),
        ]
        for m in measurements
    ]
    body = _table(_MEASUREMENT_HEADERS, rows, numeric_cols=set(range(1, 12)))
    return f"<section><h2>Measurement Table</h2>{body}</section>"


def _plots_section(
    time_arr: np.ndarray,
    params: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    fft_flags: Mapping[str, bool],
    segment_breaks: Sequence[int],
) -> str:
    cards: list[str] = []
    for name, values in params.items():
        color = colors[name]
        fft_freqs, fft_mag = _fft_spectrum(time_arr, values)
        time_svg = _line_plot_svg(
            x=time_arr,
            y=np.asarray(values),
            color=color,
            title=f"{name} vs time",
            x_label="Time (s)",
            y_label=name,
            segment_breaks=segment_breaks,
        )
        if fft_freqs is not None and fft_mag is not None:
            fft_svg = _line_plot_svg(
                x=fft_freqs,
                y=fft_mag,
                color="#ffa500" if not fft_flags.get(name, False) else color,
                title=f"{name} FFT magnitude",
                x_label="Frequency (Hz)",
                y_label="Magnitude",
                segment_breaks=[],
            )
            peaks = _fft_peaks(fft_freqs, fft_mag)
            peak_rows = [
                [str(idx + 1), _fmt(freq, " Hz"), _fmt(mag)]
                for idx, (freq, mag) in enumerate(peaks)
            ]
            peak_table = _table(
                ["Rank", "Frequency", "Magnitude"],
                peak_rows,
                numeric_cols={0, 1, 2},
                css_class="small-table",
            )
        else:
            fft_svg = "<p class=\"empty\">FFT unavailable for this trace.</p>"
            peak_table = "<p class=\"empty\">No FFT peaks available.</p>"

        cards.append(f"""
<article class="plot-card">
  <div class="plot-title">
    <span class="swatch" style="background: {escape(color)}"></span>
    <h3>{escape(name)}</h3>
  </div>
  <div class="plot-pair">
    {time_svg}
    {fft_svg}
  </div>
  <h3 style="margin-top: 10px;">FFT Peaks</h3>
  {peak_table}
</article>""".strip())

    return "<section><h2>Plots and FFT Peaks</h2><div class=\"plot-grid\">" + "\n".join(cards) + "</div></section>"


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


def _color_map(
    params: Mapping[str, np.ndarray],
    trace_colors: Mapping[str, str] | None,
) -> dict[str, str]:
    colors: dict[str, str] = {}
    provided = trace_colors or {}
    for idx, name in enumerate(params.keys()):
        colors[name] = str(provided.get(name) or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])
    return colors


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


def _key_value_table(rows: Sequence[tuple[str, str]]) -> str:
    body = "\n".join(
        "<tr>"
        f"<th>{escape(key)}</th>"
        f"<td>{escape(value)}</td>"
        "</tr>"
        for key, value in rows
    )
    return f"<table><tbody>{body}</tbody></table>"


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    numeric_cols: set[int] | None = None,
    css_class: str = "",
) -> str:
    numeric = numeric_cols or set()
    class_attr = f" class=\"{escape(css_class)}\"" if css_class else ""
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for idx, value in enumerate(row):
            cls = " class=\"num\"" if idx in numeric else ""
            cells.append(f"<td{cls}>{escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    if not body_rows:
        body_rows.append(
            f"<tr><td colspan=\"{len(headers)}\" class=\"empty\">No rows available.</td></tr>"
        )
    return f"<table{class_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _fmt(value: float | int | None, unit: str = "") -> str:
    if value is None:
        return "--"
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f_value):
        return "--"
    abs_value = abs(f_value)
    if abs_value != 0 and (abs_value >= 100000 or abs_value < 0.001):
        text = f"{f_value:.4e}"
    elif abs_value >= 1000:
        text = f"{f_value:.2f}"
    else:
        text = f"{f_value:.4f}"
    return f"{text}{unit}"


def _fmt_tick(value: float) -> str:
    abs_value = abs(value)
    if abs_value != 0 and (abs_value >= 10000 or abs_value < 0.01):
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _value_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _fmt(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_value_to_text(item) for item in value)
    return str(value)
