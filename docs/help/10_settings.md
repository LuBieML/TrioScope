# Settings

Open the Settings dialog with the **⚙ Settings** button in the left panel.
Settings are persisted across sessions in the Windows registry under
`HKCU\Software\TrioScope\ParameterScope`.

## Display

| Setting | Meaning |
|---|---|
| **Scroll window (s)** | Width of the visible time window in Continuous mode. Older samples scroll off the left edge but remain in the buffer. |
| **Lock X-Axis across subplots** | When multiple subplots are stacked, panning or zooming one moves all of them together. Essential for cross-signal comparison. |

## Capture

| Setting | Meaning |
|---|---|
| **Use end of TABLE** | Use the highest available addresses in the controller's `TABLE` for the SCOPE buffer. Recommended unless you have a specific reason to place the buffer elsewhere. |
| **Table Start** | If **Use end of TABLE** is off, manually choose the starting address. Make sure this region isn't used by your motion program. |

## Plot Style

| Setting | Meaning |
|---|---|
| **Line width** | Trace line thickness in pixels. Default `1.8`. |
| **Grid opacity** | Grid line transparency, `0.0` (invisible) to `1.0` (opaque). |
| **Plot background** | Hex colour string for the plot background, e.g. `#0A0A0A` for near-black. |

## AI Analysis (NanoGPT)

| Setting | Meaning |
|---|---|
| **API Key** | Your NanoGPT API key. Stored locally, never displayed in plaintext. |
| **Model** | Currently active model identifier. Edit directly or pick from the dropdown. |
| **Available models** | List of model identifiers that appear in the dropdown. Use **Add** / **Remove** to manage the list. |

See [AI Analysis Panel](08_ai_analysis.md) for usage details.

## Where Settings Are Stored

- **Registry**: `HKCU\Software\TrioScope\ParameterScope`
- Includes: window geometry, IP address, source mode, plot mode, line width,
  grid opacity, AI key, model, and per-axis drive profiles.

To reset everything, delete the registry key (or use `regedit`) and restart
the application.
