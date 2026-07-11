# TrioScope

[![CI](https://github.com/LuBieML/TrioScope/actions/workflows/ci.yml/badge.svg)](https://github.com/LuBieML/TrioScope/actions/workflows/ci.yml)

An oscilloscope-style data capture and analysis tool for **Trio Motion
Controllers** and **Trio DX-series servo drives**.

TrioScope captures servo parameters at the servo rate using the controller's
built-in `SCOPE` command, or directly from a DX3/DX4 drive at 125 µs
resolution over EtherCAT SDO, and visualises them in real time with a
familiar oscilloscope UI.

## Features

- **Dual capture modes** — controller `SCOPE` capture (up to 8 parameters at
  servo rate) or Drive Scope SDO capture from DX3/DX4 drives at 125 µs
  resolution, with one-shot, continuous, and externally triggered modes
- **Real-time plotting** — up to 10 traces, GPU-accelerated rendering
  (pyqtgraph + OpenGL), time / XY / XYZ / XYZW path views including a 3D
  path widget
- **Measurements** — dual cursors, per-trace min/max/mean/RMS/std-dev/P-P
  statistics, FFT with dominant-frequency detection
- **Export & reporting** — CSV export/import and self-contained HTML
  commissioning reports with plots, drive parameters, and notes
- **AI tuning advisor** — chat panel that analyses captured servo metrics
  and suggests tuning changes, with live CoE SDO read/write of drive
  tuning parameters and per-axis drive profile presets
- **EtherCAT network map** — live topology browser with slave discovery,
  drive model detection (DX1/DX3/DX4/DX5), and axis mapping
- **Robust connectivity** — connection watchdog with heartbeat and
  auto-reconnect, retry with escalating timeouts
- Built-in user manual (Help menu) — sources in [`docs/help/`](docs/help/index.md)

## Requirements

- Python 3.11+ (3.13 recommended; release builds target 3.13)
- A Trio Motion Controller reachable over Ethernet (default IP
  `192.168.0.245`) and, for Drive Scope / CoE features, DX3/DX4 drives on
  EtherCAT
- The proprietary **Trio Unified API** Python package
  (`Trio_UnifiedApi`). It is not on PyPI — obtain it from Trio Motion
  Technology and install the wheel into your environment. The app's UI
  starts without it, but no hardware features will work.

## Installation (from source)

```bash
git clone https://github.com/LuBieML/TrioScope.git
cd TrioScope
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt  # requires the Trio Unified API wheel to be available
python scope_app.py
```

For development without Trio hardware/SDK, use the dev dependencies
instead (the test suite stubs out `Trio_UnifiedApi`):

```bash
pip install -r requirements-dev.txt
```

## Quick start

1. Power up the controller and confirm it is reachable on the network.
2. Launch TrioScope, enter the controller IP, and press **Connect** — the
   status dot turns green when connected.
3. Choose a capture source (Controller SCOPE or Drive Scope SDO).
4. Press **+ Add New Trace** and pick a parameter (e.g. `MPOS`, `FE`,
   `MSPEED`).
5. Press **▶ RUN** to start streaming. **■ STOP** to stop.

The full manual is available from the in-app **Help** menu or in
[`docs/help/`](docs/help/index.md).

### AI analysis panel

The AI tuning advisor needs an API key for the
[NanoGPT](https://nano-gpt.com) chat-completions endpoint, entered in
**Settings → AI**. The key is stored in your per-user application
settings; treat that machine account as trusted.

## Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

On a headless machine, run Qt offscreen: `QT_QPA_PLATFORM=offscreen
python -m pytest tests/`. CI runs the suite on Python 3.11 and 3.13 (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Building a Windows executable

```bash
pip install pyinstaller
python build_exe.py
```

This produces `dist/TrioScope_v<version>/` with the Trio native binaries,
the help manual, third-party license texts, and Windows version metadata
bundled. The build discovers the Trio Unified API in the active environment,
the repository `.venv`, or the directory specified by `TRIOSCOPE_SDK_DIR`.

## Project layout

```
scope_app.py          Application entry point
src/
  scope/              Capture engines (controller SCOPE, drive SDO) and measurements
  ui/                 Main window, plot rendering, controllers, dialogs
  ai/                 AI analysis panel, tuners, EtherCAT scan/map, CoE I/O
  plot/               Custom pyqtgraph view box
  models/             Trace/app settings dataclasses
  storage/            Settings persistence, CSV I/O, drive profiles
  reports/            HTML commissioning report generator
docs/help/            In-app user manual (Markdown)
tests/                Unit tests (no hardware required)
build_exe.py          PyInstaller build script
```

## Versioning

TrioScope uses Semantic Versioning and automated release pull requests. Use
Conventional Commit titles such as `fix(scope): correct trigger timing` and
`feat(ui): add path inspection`, then squash-merge pull requests into `main`.
Release Please updates [`src/version.py`](src/version.py), the release manifest,
and [`CHANGELOG.md`](CHANGELOG.md); merging its release pull request creates the
version tag and GitHub Release. See [`docs/RELEASING.md`](docs/RELEASING.md) for
the complete workflow and Windows release-runner setup.

## Licenses

TrioScope depends on PySide6 (LGPL-3.0), pyqtgraph (MIT), NumPy and
PyOpenGL (BSD), and the proprietary Trio Unified API. See
[`THIRD_PARTY_LICENSES.txt`](THIRD_PARTY_LICENSES.txt) and
[`license_analysis.md`](license_analysis.md) for distribution
implications.
