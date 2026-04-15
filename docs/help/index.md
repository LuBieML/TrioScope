# TrioScope — User Manual

Welcome to **TrioScope**, an oscilloscope-style data capture and analysis tool
for Trio Motion Controllers and Trio DX-series servo drives.

TrioScope captures servo parameters at the servo rate using the controller's
built-in `SCOPE` command, or directly from a DX3/DX4 drive at 125 µs resolution
over EtherCAT SDO. It then visualises them in real time with a familiar
oscilloscope UI: multiple traces, cursors, FFT, XY/XYZ/XYZW path views, and an
integrated AI analysis panel for servo tuning advice.

---

## Table of Contents

1. [Getting Started](01_getting_started.md) — Installation, connection, first capture
2. [Capture Modes](02_capture_modes.md) — Controller SCOPE vs Drive Scope (SDO)
3. [Traces & Parameters](03_traces.md) — Adding and configuring traces
4. [Plot Modes](04_plot_modes.md) — Time, XY, XYZ, XYZW path views
5. [Navigation & Cursors](05_navigation.md) — Mouse, zoom, pan, cursors
6. [FFT Analysis](06_fft.md) — Frequency-domain inspection
7. [Export / Import](07_export_import.md) — CSV save & restore
8. [AI Analysis Panel](08_ai_analysis.md) — NanoGPT-powered tuning advisor
9. [EtherCAT Map](09_ethercat_map.md) — Network topology browser
10. [Settings](10_settings.md) — Display, capture, plot style, AI configuration
11. [Keyboard & Mouse Reference](11_shortcuts.md) — Quick reference
12. [Troubleshooting](12_troubleshooting.md) — Common problems and fixes
13. [About](about.md) — Version, credits, links

---

## Quick Start

1. Power up your Trio Motion Controller and confirm it is reachable on the
   network (default IP: `192.168.0.245`).
2. Launch TrioScope.
3. Enter the controller IP and press **Connect**. The status dot turns green
   when connected.
4. Choose a **capture source** (Controller SCOPE or Drive Scope SDO).
5. Press **+ Add New Trace** and pick a parameter (e.g. `MPOS`, `FE`, `MSPEED`).
6. Press **▶ RUN** to start streaming data. Press **■ STOP** to stop.

For more detailed walkthroughs, see [Getting Started](01_getting_started.md).
