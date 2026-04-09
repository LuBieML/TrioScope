# About TrioScope

**TrioScope** is an oscilloscope-style data capture and analysis tool for
**Trio Motion Controllers** and **Trio DX-series servo drives**. It is
designed to make servo tuning, motion debugging, and EtherCAT diagnostics
fast and intuitive — without leaving a single window.

## Features

- Real-time multi-trace plotting at servo rate using PySide6 + pyqtgraph
- GPU-accelerated rendering for smooth scrolling, even with many traces
- Two capture sources:
  - **Controller SCOPE** — any axis parameter, continuous scrolling
  - **Drive Scope (SDO)** — DX3/DX4 internal variables at 125 µs
- Time, XY, XYZ, and XYZW (3D + colour) plot modes
- Per-trace FFT with peak detection
- Two-cursor measurement tool with delta readouts
- CSV export and import
- Pinned reference traces for before/after comparisons
- AI Analysis panel powered by NanoGPT, trained on servo tuning context
- EtherCAT topology browser
- Live drive profile (Pn parameter) read/write over CoE

## Built With

- **Python 3.13** — application runtime
- **PySide6** — Qt UI bindings
- **pyqtgraph** — high-performance plotting
- **PyOpenGL** — 3D path rendering
- **NumPy** — numerical core
- **Trio_UnifiedApi** — Trio Motion controller bindings
- **PyInstaller** — Windows packaging

## Documentation

The [User Manual](index.md) covers every feature in detail. Start with
[Getting Started](01_getting_started.md) if you're new.

## Source Documents

If you need a deeper reference for the underlying hardware:

- **TrioBASIC Reference** — full Trio parameter and command catalogue
- **Trio DX3 / DX4 User Manual** — drive `Pn` parameter definitions
- **IPD-PLN-T22 Combo Function Design Document** — combo controller details

These PDFs are typically distributed with your Trio installation.

## License

TrioScope is internal tooling. Distribution and licensing follow your
organisation's policies.
