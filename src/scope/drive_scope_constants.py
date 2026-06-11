"""
Drive scope (COMBO protocol) constant tables.

SDO object indices, trigger modes, data types, and the drive-variable
address catalogue shared by the drive scope engine and the UI.
"""

import numpy as np

# ── SDO object indices ──────────────────────────────────────────────────
SETUP_INDEX = 0x368C     # Capture setup (sub 1–15)
CONTROL_INDEX = 0x368B   # Start/stop (sub 0)
STATUS_INDEX = 0x3680    # Capture status (bits 14-15)
DATA_INDEX = 0x3687      # Capture data buffer (domain, 16000 bytes)

# ── Constants ───────────────────────────────────────────────────────────
NUM_CHANNELS = 8
SAMPLES_PER_CHANNEL = 1000
TOTAL_WORDS = NUM_CHANNELS * SAMPLES_PER_CHANNEL  # 8000
EXPECTED_CAPTURE_BYTES = TOTAL_WORDS * 2
SAMPLE_TIME_UNIT_US = 125  # each sample time unit = 125 μs

# ── Trigger modes ───────────────────────────────────────────────────────
TRIGGER_MODES = {
    0: "Free Run (no trigger)",
    1: "Rising Edge",
    2: "Falling Edge",
    3: "Greater Than",
    4: "Less Than",
    5: "Window Inside",
    6: "Window Outside",
}

# ── Data type codes (for Channel1 variable data type) ───────────────────
DATA_TYPES = {
    1: ("Int16", np.int16),
    2: ("Uint16", np.uint16),
    3: ("Int32", np.int32),
    4: ("Uint32", np.uint32),
    5: ("Int64", np.int64),
    6: ("Uint64", np.uint64),
}

# ── Common drive variable addresses ─────────────────────────────────────
DRIVE_VARIABLES = {
    0x0F10: ("SPD_FB_RPM", "Speed feedback", "rpm", 1, "Int16"),
    0x0F11: ("SPD_CMD_RPM", "Speed command", "rpm", 1, "Int16"),
    0x0F13: ("TN", "Torque command %", "%Tn", 1, "Int16"),
    0x0F16: ("CURRENT_POS_L1", "Current pos low 16b", "pulse", 5, "Int64"),
    0x0F17: ("CURRENT_POS_H1", "Current pos mid-low 16b", "pulse", 5, "Int64"),
    0x0F18: ("CURRENT_POS_L2", "Current pos mid-high 16b", "pulse", 5, "Int64"),
    0x0F19: ("CURRENT_POS_H2", "Current pos high 16b", "pulse", 5, "Int64"),
    0x0F1C: ("IU", "Phase U current", "0.1%rated", 1, "Int16"),
    0x0F1D: ("IV", "Phase V current", "0.1%rated", 1, "Int16"),
    0x0F1E: ("ID_REF", "Id reference", "0.1%rated", 1, "Int16"),
    0x0F1F: ("ID", "Id actual", "0.1%rated", 1, "Int16"),
    0x0F20: ("IQ_REF", "Iq reference", "0.1%rated", 1, "Int16"),
    0x0F21: ("IQ", "Iq actual", "0.1%rated", 1, "Int16"),
    0x0F22: ("UD", "Ud voltage", "V", 2, "Uint16"),
    0x0F23: ("UQ", "Uq voltage", "V", 2, "Uint16"),
    0x0F2A: ("EST_SPD_L", "Observer speed low 16b", "0.1rpm", 3, "Int32"),
    0x0F2B: ("EST_SPD_H", "Observer speed high 16b", "0.1rpm", 3, "Int32"),
    0x0F2C: ("EST_TORQ_PER", "Observer torque", "0.1%rated", 1, "Int16"),
    0x0F2D: ("FF_SPEED", "Speed feedforward", "rpm", 1, "Int16"),
    0x0F2E: ("FF_TORQUE", "Torque feedforward", "0.1%rated", 2, "Uint16"),
    0x0F2F: ("PGERR_SPEED", "Pos cmd speed", "rpm", 1, "Int16"),
    0x0F32: ("EK_L1", "Pos error low 16b", "enc pulse", 5, "Int64"),
    0x0F33: ("EK_H1", "Pos error mid-low 16b", "enc pulse", 5, "Int64"),
    0x0F34: ("EK_L2", "Pos error mid-high 16b", "enc pulse", 5, "Int64"),
    0x0F35: ("EK_H2", "Pos error high 16b", "enc pulse", 5, "Int64"),
    0x0F36: ("PG_L1", "Pos cmd low 16b", "pulse", 5, "Int64"),
    0x0F37: ("PG_H1", "Pos cmd mid-low 16b", "pulse", 5, "Int64"),
    0x0F38: ("PG_L2", "Pos cmd mid-high 16b", "pulse", 5, "Int64"),
    0x0F39: ("PG_H2", "Pos cmd high 16b", "pulse", 5, "Int64"),
}

# Subset of commonly used variables for the UI dropdown
COMMON_DRIVE_VARIABLES = [
    (0x0F10, "Speed Feedback (rpm)"),
    (0x0F11, "Speed Command (rpm)"),
    (0x0F13, "Torque Command (%Tn)"),
    (0x0F1E, "Id Reference (0.1%rated)"),
    (0x0F1F, "Id Actual (0.1%rated)"),
    (0x0F20, "Iq Reference (0.1%rated)"),
    (0x0F21, "Iq Actual (0.1%rated)"),
    (0x0F22, "Ud Voltage (V)"),
    (0x0F23, "Uq Voltage (V)"),
    (0x0F2A, "Observer Speed Low (0.1rpm)"),
    (0x0F2C, "Observer Torque (0.1%rated)"),
    (0x0F2D, "Speed Feedforward (rpm)"),
    (0x0F2E, "Torque Feedforward (0.1%rated)"),
    (0x0F2F, "Position Cmd Speed (rpm)"),
    (0x0000, "(Disabled)"),
]

SUPPORTED_DRIVE_TYPES = {
    41: "DX3",
    42: "DX4",
    43: "DX1",
    45: "DX5",
}
