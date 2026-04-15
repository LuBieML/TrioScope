# Plot Modes

The **Plot Mode** dropdown in the Configuration group selects how traces are
visualised. There are four modes.

## Time

The classic oscilloscope view: each trace is plotted as **value vs. time**.
Multiple traces can share a subplot or be stacked into separate subplots,
depending on the per-trace **subplot** assignment.

- The X axis is time in seconds, starting from the beginning of the capture.
- The Y axis is in the parameter's native units.
- In Continuous mode the view auto-scrolls; double-click to re-enable
  auto-scroll if you have manually panned away.

## XY (2D path)

Trace 1 → X, Trace 2 → Y. The plot becomes a **2D path** showing the
relationship between two signals — e.g. plotting `MPOS` of axis 0 against
`MPOS` of axis 1 reveals the actual CNC tool path.

- Aspect ratio is locked.
- Wheel scroll zooms both axes uniformly.
- Set the auto-range mode in [Settings](10_settings.md) → Display.

## XYZ (3D path)

Trace 1 → X, Trace 2 → Y, Trace 3 → Z. Renders a **3D path** in an OpenGL
viewport. Useful for visualising true 3D toolpaths or 3-axis robot motion.

- Drag with the left mouse button to **orbit**.
- Wheel scroll to **dolly** in/out.
- Right-drag to **pan** the camera target.

## XYZW (4D path)

Trace 1 → X, Trace 2 → Y, Trace 3 → Z, **Trace 4 → Colour**. The path is
drawn in 3D and each segment is coloured by the value of the fourth signal.
A colour-bar legend is shown on the right side.

This is ideal for visualising **velocity-coloured paths** (X, Y, Z position
plus speed) or **torque-coloured paths** (X, Y, Z plus drive torque).

## Switching Modes

You can switch plot modes at any time, even while capture is running. The
underlying data buffer is preserved — only the visualisation changes.

> **Note:** XY/XYZ/XYZW modes require at least 2/3/4 enabled traces. The
> Configuration group will show an orange info label if you don't have enough
> traces assigned.
