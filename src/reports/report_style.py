"""CSS style block for the standalone HTML report."""

from __future__ import annotations


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
