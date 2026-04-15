# Keyboard & Mouse Reference

A one-page cheat sheet of every interactive control in TrioScope.

## Plot — Time / XY (2D)

| Input | Action |
|---|---|
| Left-drag | Pan X and Y |
| Scroll wheel | Zoom Y |
| Ctrl + scroll wheel | Zoom X (time) |
| Right-drag | Rubber-band zoom |
| Double-click | Reset view, re-enable auto-scroll |

In **XY** mode, scroll zooms both axes uniformly to preserve aspect ratio.

## Plot — XYZ / XYZW (3D)

| Input | Action |
|---|---|
| Left-drag | Orbit camera |
| Scroll wheel | Dolly in / out |
| Right-drag | Pan camera target |

## Cursors

| Input | Action |
|---|---|
| Click cursor button | Toggle C1 / C2 cursors on/off |
| Drag a cursor line | Move that cursor in time |
| Cursors active + change cursor pos | Live readout updates per trace |

## Trace Controls

| Input | Action |
|---|---|
| **Enable** checkbox | Show / hide the trace |
| **Parameter** dropdown | Choose the signal |
| **Axis** spinbox / arrows | Choose Trio axis number |
| **FFT** button | Toggle frequency spectrum panel |
| **PIN** button | Pin current data as a static reference trace |
| **✕** button | Delete the trace |

## Main Controls

| Button | Action |
|---|---|
| **▶ RUN** | Start capturing |
| **■ STOP** | Stop capturing |
| **⏧ Clear** | Clear the buffer and all subplots |
| **⚙ Settings** | Open the Settings dialog |
| **⤓ Export CSV** | Save buffer to CSV |
| **⤒ Import CSV** | Load CSV into buffer |
| **✨ AI Analysis** | Toggle AI dock panel |
| **⚡ EtherCAT Map** | Open EtherCAT topology window |

## Help Menu

| Menu Item | Action |
|---|---|
| Help → User Manual | Opens this manual |
| Help → Keyboard Shortcuts | Jumps to this page |
| Help → About TrioScope | Version and credits |

## Tips

- **Auto-scroll** is automatically disabled when you pan or zoom. Double-click
  any plot to re-enable.
- **Ctrl + scroll** is the most useful shortcut — it lets you zoom into a
  brief event without losing your Y range.
- **Pin a baseline** before tuning, then run a new capture, and use the
  cursors to measure the delta. This is the standard tuning workflow.
