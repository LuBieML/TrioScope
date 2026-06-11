"""Self-contained HTML commissioning reports for TrioScope captures.

The report builder lives here; supporting pieces are in sibling modules:
    report_style.py   CSS style block
    report_plots.py   SVG line plots and FFT spectrum helpers
    report_format.py  value formatting and HTML table helpers
"""

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

from .report_format import _fmt, _key_value_table, _table, _value_to_text
from .report_plots import _fft_peaks, _fft_spectrum, _line_plot_svg
from .report_style import _style_block

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


def _color_map(
    params: Mapping[str, np.ndarray],
    trace_colors: Mapping[str, str] | None,
) -> dict[str, str]:
    colors: dict[str, str] = {}
    provided = trace_colors or {}
    for idx, name in enumerate(params.keys()):
        colors[name] = str(provided.get(name) or _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])
    return colors
