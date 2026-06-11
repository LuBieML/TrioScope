"""Formatting of the computed metrics dict into the LLM-facing text block."""

from __future__ import annotations


def format_for_llm(metrics: dict) -> str:
    """Format the metrics dict as a compact named-scalar block for the LLM."""
    lines: list[str] = []

    lines.append(f"DATA SUFFICIENCY: {metrics.get('data_sufficiency', 'UNKNOWN')}")

    cap = metrics.get("capture", {})
    if cap:
        lines.append("\n## Capture")
        for k, v in cap.items():
            lines.append(f"  {k}: {v}")

    ch = metrics.get("channels_detected", {})
    if ch:
        lines.append("\n## Channels detected")
        for k, v in ch.items():
            lines.append(f"  {k}: {v or '(MISSING)'}")

    phases = metrics.get("phases", {})
    if phases:
        lines.append("\n## Phase segmentation")
        for k, v in phases.items():
            lines.append(f"  {k}: {v}")

    fe = metrics.get("fe", {})
    if fe:
        lines.append("\n## Following error per phase")
        for phase in ("idle", "accel", "cruise", "decel", "settle"):
            if phase in fe:
                s = fe[phase]
                lines.append(
                    f"  {phase}: mean={s['mean']} std={s['std']} "
                    f"rms={s['rms']} peak_abs={s['peak_abs']}"
                )
        if "cruise_fe_vs_velocity" in fe:
            c = fe["cruise_fe_vs_velocity"]
            lines.append(
                f"  cruise_fe_vs_velocity: slope={c['slope']} "
                f"intercept={c['intercept']} "
                f"proportional_to_velocity={c['proportional_to_velocity']}"
            )
            lines.append(f"    note: {c['note']}")

    vel = metrics.get("velocity", {})
    if vel:
        lines.append("\n## Velocity tracking (measured - demand)")
        for k, v in vel.items():
            if k == "velocity_overshoot_per_move":
                lines.append(
                    f"  velocity_overshoot_per_move: n_moves={v['n_moves']} "
                    f"max={v['max']}")
            else:
                lines.append(f"  {k}: {v}")

    cur = metrics.get("current", {})
    if cur:
        lines.append("\n## Drive current / torque")
        for k, v in cur.items():
            if k == "saturation_note":
                lines.append(f"  note: {v}")
            elif k == "cruise_bimodal_warning":
                lines.append(f"  WARNING: {v}")
            else:
                lines.append(f"  {k}: {v}")

    osc = metrics.get("oscillation", {})
    if osc:
        lines.append("\n## Oscillation (FFT, cruise-only, hann-windowed)")
        for sig in ("fe", "velocity_error", "current"):
            if sig in osc:
                lines.append(f"  {sig}: {osc[sig]}")
        if "current_vs_velocity_phase" in osc:
            cvp = osc["current_vs_velocity_phase"]
            if cvp:
                lines.append(f"  current_vs_velocity_phase: {cvp}")

    asym = metrics.get("asymmetry", {})
    if asym:
        lines.append("\n## Directional asymmetry")
        for k, v in asym.items():
            lines.append(f"  {k}: {v}")

    settle = metrics.get("settle", {})
    if settle:
        lines.append("\n## Settling (first 200 ms after move end)")
        for k, v in settle.items():
            lines.append(f"  {k}: {v}")

    # Reversal transients
    fe_data = metrics.get("fe", {})
    cur_data = metrics.get("current", {})
    vel_data = metrics.get("velocity", {})
    has_reversal = ("reversal" in fe_data or "reversal" in cur_data
                    or "reversal_err" in vel_data)
    if has_reversal:
        n_rev = metrics.get("phases", {}).get("n_reversals", 0)
        lines.append(f"\n## Reversal transients ({n_rev} reversals detected)")
        if "reversal" in fe_data:
            s = fe_data["reversal"]
            lines.append(
                f"  fe: mean={s['mean']} std={s['std']} "
                f"rms={s['rms']} peak_abs={s['peak_abs']}")
        if "reversal_err" in vel_data:
            s = vel_data["reversal_err"]
            lines.append(
                f"  velocity_err: mean={s['mean']} std={s['std']} "
                f"rms={s['rms']} peak_abs={s['peak_abs']}")
        if "reversal" in cur_data:
            s = cur_data["reversal"]
            lines.append(
                f"  current: mean={s['mean']} std={s['std']} "
                f"rms={s['rms']} peak_abs={s['peak_abs']}")

    warnings = metrics.get("warnings", [])
    if warnings:
        lines.append("\n## Warnings")
        for w in warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)
