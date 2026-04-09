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
