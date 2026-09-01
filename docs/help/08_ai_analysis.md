# AI Analysis Panel

TrioScope includes an integrated **AI Analysis** panel — a chat-style interface
to a large language model (via NanoGPT) that interprets your scope data and
gives servo tuning advice.

## Opening the Panel

Click **✨ AI Analysis** in the left button column. The panel docks to the
right side of the main window. Click again to hide it.

## What Gets Sent to the AI

When you press **Send**, TrioScope automatically attaches:

1. **Trace metadata** — names, units, sample rate, and per-trace **signal
   metrics** (min, max, mean, RMS, peak-to-peak, settling time, dominant
   frequency, etc.).
2. **Drive profile** — for each axis you have configured, the drive type
   (DX3 / DX4) and the relevant `Pn` parameters (gains, filters, vibration
   suppression, etc.) read live from the drive over CoE.
3. **Your typed message** — the question or request you want answered.

The AI receives a **system prompt** specialised in servo tuning, cascade-loop
design, and mechanical resonance analysis. It is told that DX3/DX4 close all
servo loops on the drive (so Trio P/I/D gains are inactive) and is instructed
to apply a systematic vibration-elimination workflow.

## Configuring the AI

### API Key

You need a NanoGPT API key. Open **Settings → AI Analysis (NanoGPT)**, paste
your key, and choose a model (default `openai/gpt-4.1-mini`). The key is saved
locally in your user profile.

### Model

The model dropdown is editable. You can add or remove model identifiers in the
**Available models** list inside the Settings dialog. Typical choices:

- `openai/gpt-4.1-mini` — fast, good general advice
- `openai/gpt-4.1` — more thorough, slower
- `anthropic/claude-sonnet-4` — strong technical analysis
- `anthropic/claude-opus-4-6` — deepest analysis, slowest

### Drive Profile

The top of the AI panel has a **Drive Profile** section with one entry per
axis. For each axis you can:

- Pick the drive type: **None / Other**, **DX3**, or **DX4**.
- Read the live `Pn` parameters from the drive with **Read from Drive**.
- Manually edit any parameter that should override the live values.
- Save the profile so it is included in every AI conversation.

The currently selected axis's drive type and parameters are sent with each
prompt, giving the AI direct visibility into your current tuning state.

## Preparing an Analyzer-Ready Test Move

Open **View → Servo Tuning Workspace** (`Ctrl+T`). The window is organized
into **Tune & Analyze**, **Motion & Inertia**, and **History** tabs. The
**Tuning Axis Motion** shortcut (`Ctrl+Shift+M`) opens the same workspace and
selects **Motion & Inertia** directly.

Use the numbered workflow in **Test motion**:

1. Choose the axis and enter **Distance**, **Speed**, and **Acceleration**.
   The motion axis and Drive Profile axis stay synchronized.
2. Click **Enable axis**. This enables controller `WDOG`, `SERVO`, and
   `AXIS_ENABLE` for the selected axis.
3. Click **Move −** or **Move +**. Immediately before every relative move,
   TrioScope writes the displayed `SPEED` and writes the displayed
   acceleration to both `ACCEL` and `DECEL`.

Use **STOP** to cancel active and buffered motion. Disabling the axis, leaving
the **Motion & Inertia** tab, hiding the workspace, disconnecting, or closing
TrioScope follows the safe shutdown path and disables the motion axis. Capture
stationary data before and after the move so the analyser can measure the noise
floor and settling window.

## Estimating Load Inertia Without Drive Identification

The **Inertia estimate** card is intended for gantries, limited-travel axes,
and other mechanisms where the drive's repeated-rotation inertia routine is
not practical. The result is inertia reflected to the motor shafts.

1. Select the captured signal and calculation method. `DRIVE_TORQUE` in
   0.1% rated torque is preferred. `DRIVE_CURRENT` in 0.1% rated current is
   also supported directly in its normalized scale. Rated motor current is
   optional for this mode and is only used to display equivalent amperes.
2. Place C1 and C2 around a constant-acceleration interval and click
   **Use AVG** beside Acceleration. Repeat for a steady-speed interval.
3. For better rejection of friction and gravity, select the recommended
   acceleration/deceleration method and capture all three phase averages.
4. Under **Test motion data**, verify the separate **Axis scaling (UNITS)**
   value loaded from Axis setup and enter the motor encoder resolution in
   counts/revolution. **Calculated acceleration** then updates automatically
   in rev/s² from the Test motion `ACCEL` value.
5. When connected to a DX3 or DX4 drive, click **Read from drive** to load
   rated torque (Pn810), rated current (Pn812), motor rotor inertia (Pn831),
   and encoder resolution bits (Pn880) over CoE. Successful values replace
   their fields; unavailable values remain editable for manual entry. Rotor
   inertia uses `1e-8 kg·m²` (for example, `230` means `2.30e-6 kg·m²`).
   Also enter the number of identical equally-loaded gantry motors.
6. Review acceleration-only current/torque, total inertia, load inertia, and
   the calculated Pn106 percentage. **Apply estimate to Pn106** copies the
   rounded value into the selected DX drive profile; use **Write** to send it.

The simple method subtracts steady torque/current from acceleration. The
recommended method compares acceleration with deceleration and reports a
phase-symmetry warning when the two sides do not agree. Repeat the measurement
at comparable speed if the mismatch exceeds 20%.

The motor acceleration conversion is:

`motor rev/s² = ACCEL × axis UNITS ÷ encoder resolution`

Axis `UNITS` is counts per user unit; encoder resolution is counts per motor
revolution. They are separate values and are not assumed to be equal.

## Effective Prompts

Good things to ask:

- *"My axis 0 has FE oscillation around the in-position window. Diagnose and
  recommend Pn changes."*
- *"There is a 600 Hz peak in the FFT of DRIVE_TORQUE. What's the root cause
  and how do I suppress it?"*
- *"Compare the current capture against the reference trace. Did the gain
  change improve settling time?"*

Less useful:

- Vague requests with no data context (the AI is blind to anything not in the
  capture buffer or the drive profile).

## Privacy

- Captures and drive profiles are sent to the NanoGPT endpoint you configure.
- No data is sent unless you press **Send**.
- The API key is stored only in your local Windows user profile registry
  (`HKCU\Software\TrioScope\ParameterScope`).
