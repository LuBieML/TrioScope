# Getting Started

## Requirements

- A **Trio Motion Controller** (MC4xx, MC5xx, Flex-6 Nano, etc.) reachable over
  Ethernet (TCP).
- Optional: a **Trio DX3 or DX4** servo drive on the EtherCAT bus for the
  Drive Scope (SDO) capture mode and AI tuning context.
- Windows 10/11. The bundled `Trio_UnifiedApi` is required for controller
  communication and is included in the installer.

## Launching the Application

Run `TrioScope.exe` from the install folder, or `python scope_app.py` from a
development checkout. The main window opens with the left configuration panel
and the right plot area.

## Connecting to a Controller

1. In the **Connection** group, enter the controller IP (default
   `192.168.0.245`).
2. Click **Connect**.
3. Watch the status dot:
   - **Red** — disconnected
   - **Yellow** — connecting / re-connecting
   - **Green** — connected, ready

If the connection fails, TrioScope will retry up to three times with escalating
timeouts. Check the IP, firewall, and that no other TrioPC tool is currently
holding the controller.

## Selecting a Capture Source

In the **Configuration** group, the **Source** dropdown selects how data is
captured:

- **Controller SCOPE** — uses the Trio controller's built-in `SCOPE` command.
  Captures any axis parameter at the servo rate. This is the default.
- **Drive Scope (SDO)** — reads internal drive variables directly from a Trio
  DX3/DX4 drive at 125 µs resolution. Requires a DX-series drive on the
  EtherCAT bus.

See [Capture Modes](02_capture_modes.md) for the full comparison.

## Your First Capture

1. Click **+ Add New Trace** in the Traces header.
2. Pick a parameter (e.g. `MPOS`) and an axis number.
3. Make sure the trace **Enable** checkbox is on.
4. Press **▶ RUN**. Data starts streaming and the plot scrolls in real time.
5. Press **■ STOP** when you have enough data.
6. Use the mouse to zoom, pan, and inspect (see
   [Navigation & Cursors](05_navigation.md)).

## Saving Your Work

- **Export CSV** — saves the entire captured buffer as a CSV file with one
  column per trace plus a `time` column.
- **Import CSV** — loads a previously exported file back into the plot. The
  trace names and units are restored automatically.

See [Export / Import](07_export_import.md) for details.
