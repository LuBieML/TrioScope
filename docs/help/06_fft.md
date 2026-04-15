# FFT Analysis

Each trace has its own **FFT** button (orange when active). When toggled on,
TrioScope computes a Fast Fourier Transform of the trace data and displays the
amplitude spectrum in a separate subplot.

## Enabling FFT

1. Run a capture (or import a CSV).
2. Click the **FFT** button on any trace.
3. A new subplot appears showing **amplitude vs. frequency** for that trace.

You can enable FFT on multiple traces at once — each gets its own spectrum
panel.

## What It Shows

- **X axis**: frequency in Hz, from 0 to half the sample rate (Nyquist).
- **Y axis**: magnitude (linear amplitude of each frequency bin).
- A **peak marker** highlights the strongest non-DC frequency. The peak
  frequency and magnitude are shown in the panel header.

## Sample Rate

The sample rate is derived from the **Sample Period × servo cycle time**
(Controller SCOPE) or the **Drive Scope period** (DX3/DX4). The FFT
automatically uses the correct rate so frequencies are reported in real Hz.

## Window Function

A Hann window is applied before the FFT to reduce spectral leakage from
non-periodic capture lengths. This is the right default for most servo
analysis.

## Cursor Interaction

When [cursors](05_navigation.md) are enabled, the FFT is computed only on the
**slice between C1 and C2** rather than the full capture. This lets you
isolate a specific event (e.g. a single move) and look at its spectrum
without contamination from idle periods.

## Common Use Cases

- **Mechanical resonance** — look for sharp peaks in `DRIVE_TORQUE` or `MSPEED`
  that aren't related to the demand profile. Peaks above 100 Hz typically
  indicate mechanical resonance and call for a torque filter (DX `Pn105`) or
  vibration suppression.
- **Notch filter design** — once you find a resonance frequency, configure the
  drive notch (DX `Pn202–Pn207`) to that frequency.
- **Servo bandwidth** — sweep the demand and look at the magnitude rolloff in
  `MSPEED` to estimate the closed-loop bandwidth.
- **Encoder ripple** — periodic ripple in `MPOS` at multiples of one
  revolution suggests encoder geometry errors.

## Performance

FFT computation is cached so repeatedly toggling cursors does not recompute
unnecessarily. For very large captures (>16k samples) the FFT is downsampled
to keep the UI responsive when cursors are not active.
