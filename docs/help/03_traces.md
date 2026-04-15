# Traces & Parameters

A **trace** is one signal on the plot — like one channel on an oscilloscope.
TrioScope supports up to 10 traces simultaneously.

## Adding a Trace

Click **+ Add New Trace** in the Traces header on the left panel. A new
control box appears with a coloured border matching the trace colour.

## Trace Controls

Each trace has the following controls, laid out in three rows:

### Row 0 — Identification

| Control | Purpose |
|---|---|
| **Enable** checkbox | Show / hide this trace on the plot. |
| **Parameter** dropdown | Trio axis parameter (Controller mode) — e.g. `MPOS`, `FE`, `MSPEED`, `DAC_OUT`. |
| **Drive variable** dropdown | DX3/DX4 internal variable (Drive Scope mode). |
| **✕** delete button | Remove this trace. |

### Row 1 — Axis & Value

| Control | Purpose |
|---|---|
| **Axis** spinbox | Trio axis number (0–15) for this parameter. Use the up/down arrows or type a value. |
| **Value display** | Live numeric readout — most recent sample (or value at cursor C1 if cursors are on). |
| **FFT** button | Toggle a frequency-domain spectrum plot for this trace. See [FFT Analysis](06_fft.md). |

### Row 2 — Pin / Reference

| Control | Purpose |
|---|---|
| **PIN** | Pin the current trace as a static **reference trace**. The reference stays on the plot even when capture restarts, so you can compare before/after a tuning change. |
| **Color** | Open a colour picker to recolour the trace and its reference. |

## Parameter Catalogue (Controller SCOPE)

Common Trio axis parameters available in Controller SCOPE mode include:

| Parameter | Meaning |
|---|---|
| `MPOS` | Measured (encoder) position |
| `DPOS` | Demand position |
| `FE` | Following error (DPOS − MPOS) |
| `MSPEED` | Measured speed |
| `MSPEEDF` | Filtered measured speed |
| `DAC_OUT` | DAC / torque demand sent to the drive |
| `T_REF`, `S_REF` | Torque / speed reference |
| `DRIVE_FE`, `DRIVE_TORQUE` | Mirrored drive parameters (when EtherCAT cyclic data is mapped) |
| `VP_*` | Virtual axis profile signals |
| `WORLD_*`, `TCP_*` | Robot kinematic frame variables |

The full list (200+ parameters) is in the parameter dropdown. See the Trio
**Motion Perfect Help** or the **TrioBASIC Reference Manual** for the full
semantics of each parameter.

## Drive Variable Catalogue (Drive Scope SDO)

In Drive Scope mode the dropdown shows DX3/DX4 internal scope variables, e.g.:

- Position feedback (encoder count)
- Velocity feedback / command
- Current command (Iq) / current feedback
- Torque command
- Position error (drive-internal FE)
- Bus voltage / temperature

The exact list depends on the drive firmware. Refer to your DX3/DX4 user
manual for definitions.

## Tips

- **Use distinct colours** if you compare multiple signals visually. Each new
  trace gets a unique default colour, but you can recolour any trace.
- **Pin a baseline** before you make a tuning change so you can compare against
  the original response.
- **Per-axis stacking** — assign related signals (e.g. `DPOS` and `MPOS`) to
  the same axis spinbox value to overlay them on the same subplot.
