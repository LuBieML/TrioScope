# Navigation & Cursors

TrioScope uses oscilloscope-style mouse controls for fast plot navigation.

## Mouse Controls (Time / XY plots)

| Action | Result |
|---|---|
| **Left-drag** | Pan (X and Y) |
| **Scroll wheel** | Zoom **Y** axis |
| **Ctrl + scroll wheel** | Zoom **X** axis (time) |
| **Right-drag** | Rubber-band zoom — drag a rectangle, release to zoom into it |
| **Double-click** | Reset view to auto-range and re-enable auto-scroll |

In XY mode, scroll wheel zooms both axes uniformly so the aspect ratio is
preserved.

## Mouse Controls (3D plots — XYZ / XYZW)

| Action | Result |
|---|---|
| **Left-drag** | Orbit the camera around the path |
| **Scroll wheel** | Dolly in / out |
| **Right-drag** | Pan the camera target |

## Auto-Scroll

In **Continuous** capture mode the time view scrolls so the most recent
samples stay visible. The scroll window length is set in
[Settings](10_settings.md) → Display.

If you pan or zoom away, auto-scroll is automatically **disabled** so your
manual view is preserved. **Double-click** any plot to re-enable auto-scroll.

## Cursors (Measurement Tool)

The **Cursors** toolbar button enables two vertical measurement cursors,
**C1** (gold) and **C2** (turquoise).

Each cursor reports:

- The **time** at which it sits.
- The **value** of every visible trace at that cursor's position.
- The **delta** between C1 and C2 (Δt and Δvalue per trace).

### Using Cursors

1. Click the **Cursors** button to toggle them on. A readout panel appears
   below the plot.
2. Drag a cursor by clicking and dragging the coloured vertical line.
3. The trace value displays update to show **the value at C1** while cursors
   are active (instead of the live latest sample).
4. Click the cursor button again to hide them and return to live readout.

### Tips

- **Settling time**: drop C1 at the end of a move and C2 where the following
  error first stays within your tolerance band — Δt is your settling time.
- **Overshoot**: drop C2 on the peak and C1 on the steady-state value — Δvalue
  is the overshoot.
- **Frequency**: drop C1 and C2 on two zero crossings of an oscillation —
  `1 / Δt` is the frequency.

## Lock X-Axis Across Subplots

When multiple subplots are stacked, enable **Lock X-Axis** in
[Settings](10_settings.md) so panning or zooming one plot moves all of them
together. This is essential for comparing multiple signals at the same instant.
