# Capture Modes

TrioScope offers two capture sources, selected from the **Source** dropdown in
the Configuration group.

## Controller SCOPE

This mode uses the Trio controller's built-in `SCOPE` command. It can capture
**any axis parameter** that exists on the controller, at any servo cycle rate.

### Configuration

| Field | Description |
|---|---|
| **Sample Period** | Number of servo cycles between samples. `1` = every servo cycle (highest rate). `2` = every other cycle, etc. |
| **Duration** | How many seconds of data to capture in one shot. |
| **Capture Mode** | **Single** stops after one buffer is filled. **Continuous** automatically restarts captures and accumulates data into one long timeline. |

### How it works

The controller fills its internal `TABLE` memory with parameter samples and
TrioScope reads back the data once the buffer is full. In **Continuous** mode,
TrioScope chains captures back-to-back so the plot scrolls indefinitely. Small
gaps between buffers are visible as **segment breaks** (vertical guides).

### When to use

- You want a wide range of parameters (FE, MPOS, MSPEED, DAC_OUT, …).
- You need a long, continuously scrolling timeline.
- You are not concerned with sub-servo-cycle resolution.

## Drive Scope (SDO)

This mode bypasses the controller's `SCOPE` command and instead samples
**internal drive variables** directly from a **Trio DX3 or DX4** servo drive
over EtherCAT SDO at **125 µs** resolution.

### Configuration

| Field | Description |
|---|---|
| **Capture Duration** | Total length in seconds. The drive captures 1000 samples; period = duration ÷ 1000, rounded to the nearest 125 µs (minimum 125 µs). |
| **Trigger** | Drive trigger mode (Free Run, Rising Edge, Falling Edge, Window, etc.). |
| **Trigger Value** | Threshold values for triggered modes. |
| **Drive Axis** | EtherCAT axis number of the drive to read from. |

### How it works

TrioScope arms the drive's internal scope, sets the trigger, waits for the
trigger condition, then reads the 1000-sample buffer back as one block. There
is no real-time scrolling — each capture is a complete shot.

### When to use

- You need **125 µs** resolution to see fast disturbances or current ripple.
- You want to inspect drive-internal variables that are **not exposed** to the
  controller (e.g. internal current loop signals).
- You need precise edge-triggered captures synchronised to a specific event
  inside the drive.

### Limitations

- Fixed 1000 samples per capture.
- Single shot only — no continuous scrolling.
- Only works with Trio DX3/DX4 drives.

## Choosing Between Modes

| Need | Use |
|---|---|
| Long continuous scrolling | Controller SCOPE |
| Mix of axis parameters | Controller SCOPE |
| Sub-millisecond resolution | Drive Scope (SDO) |
| Triggered capture on a drive event | Drive Scope (SDO) |
| Drive-internal current/torque signals | Drive Scope (SDO) |
