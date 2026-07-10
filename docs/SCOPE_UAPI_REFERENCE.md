# Trio Controller SCOPE with Unified API

This document describes how to prepare, start, monitor, read, and stop the
Trio controller's native `SCOPE` capture using Trio Unified API (UAPI).

> **Scope of this document:** this is the controller-side `SCOPE` function,
> which records controller parameters into `TABLE` or a FIFO. It is not the
> DX3/DX4 drive-internal ASCOPE/COMBO capture transported over CoE SDO. The two
> mechanisms have different configuration and data layouts.

## Capture sequence

The normal TABLE-based sequence is:

```text
Open connection
  -> read SERVO_PERIOD and TSIZE
  -> calculate the TABLE allocation
  -> configure/arm with SCOPE(ON, ...)
  -> start with TRIGGER or TRIGGER(1)
  -> monitor SCOPE_POS
  -> read TABLE into double[] / float64 storage
  -> split TABLE into one sequential block per parameter
  -> disable with SCOPE(OFF)
```

## Trio BASIC command syntax

Configure and arm a TABLE capture:

```basic
SCOPE(ON, period_cycles, table_start, table_end, parameter_1, parameter_2, ...)
```

Start a single capture:

```basic
TRIGGER
```

Start with automatic retriggering after every completed scan:

```basic
TRIGGER(1)
```

Disable the scope:

```basic
SCOPE(OFF)
```

Example:

```basic
SCOPE(ON, 1, 1000, 3999, MPOS AXIS(0), DPOS AXIS(0), FE AXIS(0))
TRIGGER
```

This captures `MPOS`, `DPOS`, and `FE` for axis 0 once per servo cycle and
stores the results in the inclusive TABLE range 1000 through 3999.

`SCOPE(ON, ...)` configures and arms the capture. Sampling begins only after a
`TRIGGER` command.

## UAPI methods

The documented C++ UAPI signatures are:

```cpp
void ScopeOff();

void ScopeOn();

void ScopeOn(uint32_t period_cycles);

void ScopeOn(
    uint32_t period_cycles,
    uint32_t table_start,
    uint32_t table_end,
    const std::string_view parameters[],
    uint32_t parameter_count
);

void ScopeOn(
    uint32_t period_cycles,
    const std::string_view& fifo_name,
    uint32_t fifo_size,
    uint32_t sample_count,
    const std::string_view parameters[],
    uint32_t parameter_count
);

void Trigger(bool rearm = false);

int32_t GetSystemParameter_SCOPE_POS();
int32_t GetSystemParameter_SCOPE_TRIGGER_POS();
int32_t GetSystemParameter_SCOPE_CYCLE_COUNT();
int64_t GetSystemParameter_SCOPE_DELAY();
int32_t GetSystemParameter_SERVO_PERIOD();
int32_t GetSystemParameter_TSIZE();

void GetMultiTableValues(
    uint32_t start_index,
    uint32_t count,
    double values[]
);
```

The Python binding used by TrioScope exposes the TABLE overload as:

```python
parameters = [
    "MPOS AXIS(0)",
    "DPOS AXIS(0)",
    "FE AXIS(0)",
]

connection.ScopeOn(
    1,          # period_cycles
    1000,       # first TABLE address
    3999,       # last TABLE address (inclusive)
    parameters,
)

connection.Trigger(False)
```

The Python wrapper derives the parameter count from the list, so a separate
`parameter_count` is normally not passed.

Other overloads have the following meanings:

- `ScopeOn()` enables SCOPE using its previous settings.
- `ScopeOn(period_cycles)` enables it using its previous settings and a new
  sampling period.
- The TABLE overload records to an inclusive TABLE range.
- The FIFO overload records `sample_count` samples to a controller FIFO file;
  `fifo_size` controls the created FIFO size.

## `Execute()` compatibility fallback

Some Python UAPI builds fail to marshal the `std::string_view` parameter array
and raise an error mentioning `std::basic_string_view`. In that case, send the
equivalent Trio BASIC commands through `Execute()`:

```python
connection.Execute(
    "SCOPE(ON, 1, 1000, 3999, "
    "MPOS AXIS(0), DPOS AXIS(0), FE AXIS(0))"
)
connection.Execute("TRIGGER")
```

For automatic retriggering:

```python
connection.Execute("TRIGGER(1)")
```

To stop:

```python
connection.Execute("SCOPE(OFF)")
```

TrioScope implements and caches this compatibility fallback in
`src/scope/scope_engine.py`.

## Parameter syntax

### Axis parameters

Axis parameters use:

```text
PARAMETER AXIS(axis_number)
```

Examples:

```text
MPOS AXIS(0)
DPOS AXIS(0)
FE AXIS(0)
MSPEED AXIS(1)
DRIVE_TORQUE AXIS(2)
DRIVE_CURRENT AXIS(2)
```

Convenient user notation such as `MPOS(0)` must be converted to
`MPOS AXIS(0)` before being passed to `ScopeOn()`.

### VR variables

```text
VR(index)
```

Example:

```text
VR(100)
```

### TABLE values

```text
TABLE(index)
```

Example:

```text
TABLE(500)
```

Do not monitor a `TABLE()` address that overlaps the TABLE destination region
used by the same capture.

### Channel parameters

Channel-oriented parameters retain their indexed syntax:

```text
IN(4)
AIN(0)
DV_IN(1)
```

For digital output capture, TrioScope translates `OUT(n)` to the
command-compatible `READ_OP(n)` form when it uses `Execute()`:

```basic
SCOPE(ON, 1, 1000, 1999, READ_OP(0))
```

The known axis/channel parameter lists and the user-input conversion rules are
in `src/scope/parameters.py` and `src/scope/parameter_parser.py`. A parameter
not included in those lists may still work if it is valid for the connected
controller and firmware.

## Timing calculations

`GetSystemParameter_SERVO_PERIOD()` returns the servo period in microseconds.

```python
servo_period_us = connection.GetSystemParameter_SERVO_PERIOD()
servo_period_s = servo_period_us / 1_000_000.0
```

The effective sample period and frequency are:

```text
sample_period_s = period_cycles * servo_period_s
sample_rate_hz  = 1 / sample_period_s
```

Example:

```text
SERVO_PERIOD = 1000 us
period_cycles = 2

sample period = 2 * 1000 us = 2000 us = 2 ms
sample rate   = 1 / 0.002 s = 500 Hz
```

For a requested duration:

```text
samples_per_parameter = floor(duration_s / sample_period_s)
required_entries      = samples_per_parameter * parameter_count
table_end             = table_start + required_entries - 1
```

Example:

```text
Servo period:          1 ms
period_cycles:         2
Capture duration:      1 second
Parameters:            3
TABLE start:           1000

Sample period:         2 ms
Samples per parameter: 500
Required entries:      500 * 3 = 1500
TABLE end:             1000 + 1500 - 1 = 2499
```

The resulting command is:

```basic
SCOPE(ON, 2, 1000, 2499, MPOS AXIS(0), DPOS AXIS(0), FE AXIS(0))
```

Check the allocation against `TSIZE` before arming:

```python
tsize = int(connection.GetSystemParameter_TSIZE())

if table_start < 0 or table_end >= tsize:
    raise ValueError("SCOPE TABLE range exceeds TSIZE")
```

`table_end` is inclusive, and the normally valid TABLE indexes are
`0` through `TSIZE - 1`. The selected region must not overlap TABLE memory used
by the controller's motion program or another subsystem.

## TABLE data layout

Controller SCOPE data is stored in sequential parameter blocks, not
sample-interleaved. For three parameters and 500 samples per parameter:

```text
TABLE[start +    0 ... start +  499] = parameter 1 samples
TABLE[start +  500 ... start +  999] = parameter 2 samples
TABLE[start + 1000 ... start + 1499] = parameter 3 samples
```

It is **not** stored as:

```text
param1[0], param2[0], param3[0],
param1[1], param2[1], param3[1], ...
```

This differs from TrioScope's DX drive-internal scope payload, which is
interleaved across active drive channels.

## Data types

| Item | C++ UAPI type | Python representation |
|---|---:|---:|
| Period in servo cycles | `uint32_t` | `int` |
| TABLE start/end | `uint32_t` | `int` |
| FIFO size/sample count | `uint32_t` | `int` |
| Parameter count | `uint32_t` | inferred from `list[str]` in the Python binding |
| Parameter names | `std::string_view[]` | `list[str]` |
| Trigger rearm | `bool` | `bool` |
| `SCOPE_POS` | `int32_t` | `int` |
| `SCOPE_TRIGGER_POS` | `int32_t` | `int` |
| `SCOPE_CYCLE_COUNT` | `int32_t` | `int` |
| `SCOPE_DELAY` | `int64_t` | `int` |
| `SERVO_PERIOD` | `int32_t` | `int` in microseconds |
| `TSIZE` | `int32_t` | `int` |
| One TABLE value | `double` | Python `float` / NumPy `float64` |
| TABLE output buffer | `double[]` | contiguous NumPy `float64` array |

Prepare a Python destination buffer as follows:

```python
import numpy as np

raw = np.empty(entry_count, dtype=np.float64)
connection.GetMultiTableValues(table_start, entry_count, raw)
```

Do not substitute `float32`, an integer dtype, or a structured/interleaved
dtype unless the particular UAPI binding explicitly documents support for it.

## Reading and splitting captured data

After the capture is complete, read the entire allocated TABLE region and
slice it into equal parameter blocks:

```python
import numpy as np

parameters = [
    "MPOS AXIS(0)",
    "DPOS AXIS(0)",
    "FE AXIS(0)",
]

period_cycles = 1
table_start = 1000
samples_per_parameter = 1000

entry_count = samples_per_parameter * len(parameters)
table_end = table_start + entry_count - 1

servo_period_us = connection.GetSystemParameter_SERVO_PERIOD()
sample_period_s = period_cycles * servo_period_us / 1_000_000.0

connection.ScopeOn(period_cycles, table_start, table_end, parameters)
connection.Trigger(False)

# Wait for completion here; see the monitoring notes below.

raw = np.empty(entry_count, dtype=np.float64)
connection.GetMultiTableValues(table_start, entry_count, raw)

channels = {}
for channel_index, parameter in enumerate(parameters):
    first = channel_index * samples_per_parameter
    last = first + samples_per_parameter
    channels[parameter] = raw[first:last].copy()

time_s = np.arange(samples_per_parameter, dtype=np.float64) * sample_period_s

connection.ScopeOff()
```

## Monitoring and trigger behavior

### Single shot

```python
connection.ScopeOn(period, table_start, table_end, parameters)
connection.Trigger(False)
```

Equivalent commands:

```basic
SCOPE(ON, period, table_start, table_end, parameters...)
TRIGGER
```

### Continuous automatic retrigger

```python
connection.ScopeOn(period, table_start, table_end, parameters)
connection.Trigger(True)
```

Equivalent commands:

```basic
SCOPE(ON, period, table_start, table_end, parameters...)
TRIGGER(1)
```

With rearm enabled, `SCOPE_POS` can wrap when one scan finishes and the next
begins. A streaming application must detect that wrap and preserve the
completed data before it is overwritten.

### Trigger from a controller program

The PC can arm the SCOPE without triggering it:

```python
connection.ScopeOn(period, table_start, table_end, parameters)
```

A Trio BASIC program can then start the already configured capture:

```basic
TRIGGER
```

This is useful when acquisition must be synchronized with controller motion or
I/O logic.

### Interpreting `SCOPE_POS`

Read the live position with:

```python
position = int(connection.GetSystemParameter_SCOPE_POS())
```

In TrioScope's controller capture path, `SCOPE_POS` is treated as a zero-based
sample position relative to the start of a parameter block, rather than as an
absolute TABLE address. Continuous mode detects capture boundaries when the
position wraps to a lower value.

Because exact terminal/wrap behavior can vary by controller generation and
firmware, applications should:

- confirm the observed final value on the target controller;
- use a timeout rather than waiting indefinitely;
- detect a position that stops advancing;
- detect wrap-around in automatic-retrigger mode; and
- avoid polling so rapidly that the controller is flooded with requests.

## Reusable Python capture function

The following example performs validation, arms and triggers a single-shot
capture, waits with a timeout, reads the TABLE region, and separates the
parameter blocks.

```python
import time
import numpy as np


def capture_scope(
    connection,
    parameters,
    duration_s,
    period_cycles=1,
    table_start=0,
    timeout_s=None,
):
    if not parameters:
        raise ValueError("At least one SCOPE parameter is required")

    if period_cycles < 1:
        raise ValueError("period_cycles must be at least 1")

    servo_period_us = int(
        connection.GetSystemParameter_SERVO_PERIOD()
    )
    tsize = int(connection.GetSystemParameter_TSIZE())

    sample_period_s = (
        period_cycles * servo_period_us / 1_000_000.0
    )
    samples_per_parameter = int(duration_s / sample_period_s)

    if samples_per_parameter < 1:
        raise ValueError(
            f"Duration must be at least {sample_period_s:g} seconds"
        )

    entry_count = samples_per_parameter * len(parameters)
    table_end = table_start + entry_count - 1

    if table_start < 0 or table_end >= tsize:
        raise ValueError(
            f"TABLE {table_start}..{table_end} exceeds "
            f"valid range 0..{tsize - 1}"
        )

    if timeout_s is None:
        timeout_s = max(2.0, duration_s * 2.0 + 1.0)

    connection.ScopeOn(
        period_cycles,
        table_start,
        table_end,
        list(parameters),
    )
    connection.Trigger(False)

    deadline = time.monotonic() + timeout_s

    try:
        while True:
            position = int(
                connection.GetSystemParameter_SCOPE_POS()
            )

            if position >= samples_per_parameter:
                break

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "SCOPE timed out at position "
                    f"{position}/{samples_per_parameter}"
                )

            time.sleep(0.01)

        raw = np.empty(entry_count, dtype=np.float64)
        connection.GetMultiTableValues(
            table_start,
            entry_count,
            raw,
        )

    finally:
        connection.ScopeOff()

    channels = {}

    for index, parameter in enumerate(parameters):
        first = index * samples_per_parameter
        last = first + samples_per_parameter
        channels[parameter] = raw[first:last].copy()

    time_axis_s = (
        np.arange(samples_per_parameter, dtype=np.float64)
        * sample_period_s
    )

    return {
        "time_s": time_axis_s,
        "channels": channels,
        "sample_period_s": sample_period_s,
        "sample_rate_hz": 1.0 / sample_period_s,
        "samples_per_parameter": samples_per_parameter,
        "table_start": table_start,
        "table_end": table_end,
    }
```

Example use:

```python
result = capture_scope(
    connection=connection,
    parameters=[
        "MPOS AXIS(0)",
        "DPOS AXIS(0)",
        "FE AXIS(0)",
    ],
    duration_s=2.0,
    period_cycles=1,
    table_start=1000,
)

time_s = result["time_s"]
mpos = result["channels"]["MPOS AXIS(0)"]
dpos = result["channels"]["DPOS AXIS(0)"]
fe = result["channels"]["FE AXIS(0)"]
```

If the target firmware reports a different single-shot terminal value for
`SCOPE_POS`, adapt the completion condition after verifying it on that
controller. TrioScope's production implementation contains additional logic
for single-shot, streaming, external-trigger, and auto-retrigger operation.

## Operational recommendations

- Prefer an unused region near the end of TABLE unless the application has a
  deliberately reserved capture region.
- Never overlap the capture buffer with TABLE memory used by a motion program.
- Validate the inclusive end address against `TSIZE` before arming.
- Use `period_cycles >= 1`.
- Preallocate a contiguous `float64` output array for bulk TABLE reads.
- Use bulk reads instead of reading individual TABLE entries over the network.
- Add a timeout to every capture wait.
- Poll `SCOPE_POS` at a moderate interval, such as 5-20 ms, rather than in a
  tight loop.
- Call `ScopeOff()` in cleanup/error handling.
- Serialize hardware access when the same connection is shared by a watchdog
  or another controller operation.
- Stop or coordinate watchdog activity before any operation that holds the
  connection lock for an extended period.

## Project references

- UAPI signature reference: `reference/Trio_UnifiedApi_CPP.pdf`
- Controller implementation: `src/scope/scope_engine.py`
- Parameter conversion: `src/scope/parameter_parser.py`
- Known parameter lists: `src/scope/parameters.py`
- Capture orchestration: `src/ui/capture_controller/`
- Unit tests: `tests/test_scope_engine.py`

