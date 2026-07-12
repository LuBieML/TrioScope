# TrioScope — agent guide

Oscilloscope-style capture/analysis tool for Trio Motion controllers and
Trio DX-series servo drives. PySide6 + pyqtgraph desktop app.

## Commands

```bash
pip install -r requirements-dev.txt          # deps without the Trio SDK
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q   # full test suite
python scope_app.py                          # run the app (needs Trio SDK + display)
pyright                                      # type check (informational, has known errors)
```

CI (`.github/workflows/ci.yml`) runs pytest on Python 3.13 and a
non-blocking pyright job.

## Import & runtime conventions

- `scope_app.py` puts `src/` on `sys.path`, so app code imports
  `ui.x`, `scope.x`, `ai.x`. Tests import the same modules as
  `src.ui.x`, `src.scope.x`, `src.ai.x`. Both paths must keep working.
- The proprietary `Trio_UnifiedApi` (TUA) SDK is not on PyPI;
  `tests/conftest.py` stubs it. Modules guard the import with
  `try/except ImportError`.
- UI controllers (`ConnectionController`, `CaptureController`,
  `PlotRenderer`, `MainWindowActions`) extend `WindowBackedController`
  (`src/ui/window_controller.py`): attribute reads/writes proxy to the
  main window, so shared state lives on the window and controller mixins
  carry methods only. `src/ui/main_window_bindings.py` binds their
  methods onto the window — new public controller methods must be added
  to the method lists there.
- Hardware access is serialized with `_conn_lock`; a watchdog pings the
  connection every 0.5 s — never hold the lock for long operations
  (stop the watchdog first, see drive scope capture thread).

## Map — where things live

| Area | Path | Notes |
|---|---|---|
| Entry point | `scope_app.py` → `src/ui/main_window.py` | window + widget construction |
| Controller SCOPE engine | `src/scope/scope_engine.py` | TABLE-based capture lifecycle |
| Scope param parsing | `src/scope/parameter_parser.py` | "MPOS(0)" → SCOPE format |
| Drive scope (SDO) engine | `src/scope/drive_scope_engine.py` | facade; constants / coe / config / transfer / parsing in `drive_scope_*.py` siblings |
| Capture orchestration | `src/ui/capture_controller/` | threads + data pipeline (see pkg `__init__`) |
| Plotting | `src/ui/plot_renderer/` | layout / rendering / cursors / hover / compare (see pkg `__init__`) |
| Menus, export, settings | `src/ui/main_window_actions/` | export_import / profiles / menu / panels / settings |
| Connection + watchdog | `src/ui/connection_controller.py` | connect/disconnect, lost-connection handling |
| Trace row widget | `src/ui/trace_control.py` | param/axis selectors per trace |
| Servo Loop Analyser panel | `src/ai/tuner_panel.py` | + `loop_cards.py`, `zn_calculator.py`, `tuner_theme.py`, `history_card.py`, `tuning_history.py` |
| Drive profile editor | `src/ai/drive_profile_editor.py` | shared per-axis Pn editor + CoE read/write |
| AI chat panel | `src/ai/analysis_panel.py` | prompts in `tuning_prompts.py`, LLM client in `nanogpt_client.py` |
| Signal metrics for LLM | `src/ai/signal_metrics.py` | facade; constants / phases / analyzers / spectral / report in `signal_*.py` siblings |
| ZN tuning helpers | `src/ai/classical_tuner.py` | oscillation detection + bandwidth estimates |
| CoE SDO drive profile I/O | `src/ai/coe_io.py` | Pn read/write over EtherCAT |
| EtherCAT network map | `src/ai/ethercat_scan.py`, `ethercat_map_window.py` | |
| HTML reports | `src/reports/html_report.py` | + `report_style.py`, `report_plots.py`, `report_format.py` |
| Measurements | `src/scope/measurements.py`, `src/ui/measurement_panel.py` | |
| Persistence | `src/storage/` | QSettings, CSV, trace profiles |
| Data models | `src/models/` | AppSettings, TraceConfig |
| Tests | `tests/` | unit + offscreen Qt smoke tests |
| User help (in-app) | `docs/help/*.md` | rendered by `src/help_window.py` |
| Vendor manuals & C# reference | `reference/` | PDFs, COMBO protocol doc, `Uapi.cs` |

## Domain cheat sheet

- Two capture paths: **Controller SCOPE** (controller TABLE, servo-rate)
  and **Drive Scope** (drive-internal COMBO protocol over CoE SDO,
  125 µs units, 1000 samples/channel, payload parsed from an
  EC_COE_FIFO file download).
- SCOPE stores TABLE data in **sequential blocks per parameter**, not
  interleaved; drive scope payload is **interleaved across active
  channels only** (see `drive_scope_engine.py` docstring).
- Pn parameters (Pn100–Pn135) are DX3/DX4 drive tuning registers;
  definitions in `src/ai/drive_profile.py`.

## Editing rules of thumb

- Run the test suite after changes; Qt tests need
  `QT_QPA_PLATFORM=offscreen` (CI installs libegl1/libgl1/libxkbcommon0).
- Keep modules under ~500 lines; split along the existing package
  seams rather than growing facades back.
- Keep `src/scope/` and `src/reports/` Qt-free (in `src/storage/` only
  `settings_store.py` and `profiles.py` touch QSettings).
