# Troubleshooting

## Connection Issues

### Status dot stays red, "Connect" fails

- **Wrong IP** — verify the controller's IP using Motion Perfect or `ping`.
- **Network unreachable** — check Windows is on the same subnet as the
  controller (default `192.168.0.x / 255.255.255.0`).
- **Firewall** — Windows Firewall may block TCP outbound. Allow `TrioScope.exe`
  through the firewall when prompted, or add an outbound rule manually.
- **Controller in use** — only one client can hold the controller at a time.
  Close any open Motion Perfect or Trio diagnostic tool that may already be
  connected.
- **Wrong API DLL** — if you self-built TrioScope, confirm
  `Trio_UnifiedApi.pyd`, `Trio_UnifiedApi_TCP.dll`, and
  `Trio_UnifiedApi_PCMCAT.dll` are next to the executable.

### Status dot is yellow forever

The connection escalates through three timeouts (5s → 10s → 15s) before giving
up. If it stays yellow, the controller is reachable at the IP level but not
responding to the Trio handshake — check controller power, EtherCAT bus, and
that no error LEDs are lit.

### "Disconnected" appears mid-capture

Either the controller rebooted or the network was interrupted. TrioScope
auto-retries during a cooldown window and resumes the capture if it can.
Check the cable, switch, and any error/diagnostic LEDs on the controller.

## Capture Issues

### Empty plot after pressing RUN

- No traces are **enabled** — check the per-trace **Enable** checkboxes.
- The chosen parameter doesn't exist on the configured axis (e.g. asking for
  `WORLD_DPOS` on a non-robot axis). Pick a valid parameter for that axis.
- The capture finished too quickly — increase **Duration** to at least 1
  second.

### "TABLE address conflict" or partially-filled buffers

The Controller SCOPE buffer overlaps with `TABLE` memory used by your motion
program. Open Settings → Capture and either:

- Tick **Use end of TABLE** (recommended), or
- Set **Table Start** to an explicit safe address higher than your program
  uses.

### Drive Scope (SDO) returns no data

- The selected axis is not a Trio DX3/DX4 drive — only DX-series drives have
  the internal scope feature.
- The trigger never fires — pick **Free Run** to verify the bus is working,
  then re-enable triggers.
- The drive is in fault state — clear the fault first.

## Plot / UI Issues

### XY / XYZ / XYZW mode shows nothing

You don't have enough enabled traces. XY needs 2, XYZ needs 3, XYZW needs 4.
Add traces and re-select the plot mode.

### 3D view is laggy

The OpenGL viewport renders every captured sample. For very long captures,
clear the buffer (`⏧ Clear`) more often or reduce the capture **Duration**.

### Auto-scroll seems stuck

You panned or zoomed manually, which **disables** auto-scroll. Double-click
the plot to re-enable.

### Cursor readout shows the wrong values

- The trace was added or removed since the cursor was placed — toggle cursors
  off and on.
- The cursor is outside the captured time range. Drag it back into the
  visible window.

## AI Analysis Issues

### "AI module not available"

The `src/ai/` package is missing or failed to import. If you're running from
source, make sure all dependencies in `requirements.txt` are installed. If
you're running the installed exe, reinstall — the AI module may have been
excluded from the build.

### "API key invalid" or 401 errors

Your NanoGPT API key is wrong or expired. Open Settings → AI Analysis and
paste a current key.

### AI gives generic / unhelpful answers

- Make sure you have a **drive profile** configured for the relevant axis so
  the AI knows your `Pn` parameters.
- Capture **enough data** before sending — at least one full move cycle
  including settle.
- Be specific in your prompt. See [AI Analysis Panel](08_ai_analysis.md) for
  prompt examples.

## Build / Install Issues

### `python build_exe.py` fails

- Check that PyInstaller is installed: `pip install pyinstaller`.
- Check that the Trio binaries (`Trio_UnifiedApi*.dll/.pyd`) exist in your
  `.venv/Lib/site-packages` folder. The build script copies them into the
  output.

### App opens but immediately closes

Run from a console (`TrioScope.exe` from `cmd`) so you can see the traceback.
The most common cause is a missing native DLL — see the install issues above.
