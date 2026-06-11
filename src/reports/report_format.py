"""Value formatting and HTML table helpers for the report."""

from __future__ import annotations

from html import escape
from typing import Sequence

import numpy as np


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
