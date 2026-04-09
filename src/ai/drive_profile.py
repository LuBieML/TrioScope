"""
Drive profile definitions for Trio DX3 / DX4 servo drives.

Both DX3 and DX4 share the same Pn parameter codes and EtherCAT object IDs.
EtherCAT object IDs are stored here for future auto-read implementation via
CoE (CANopen over EtherCAT) SDO reads.

Control loop structure (cascade, inner to outer):
  Torque loop  ← Pn105 (torque filter)
  Speed loop   ← Pn102 (Kp, rad/s), Pn103 (Ti, ×0.1ms)
  Position loop ← Pn104 (Kp, 1/s)
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# EtherCAT CoE object IDs — reserved for future auto-read via SDO
# ---------------------------------------------------------------------------
ETHERCAT_OBJECT_IDS: dict[str, int] = {
    "pn100": 0x31C8,  # Tuning mode selection
    "pn101": 0x31C9,  # Servo rigidity (auto-tuning response)
    "pn102": 0x31CA,  # Speed loop gain
    "pn103": 0x31CB,  # Speed loop integral time
    "pn104": 0x31CC,  # Position loop gain
    "pn105": 0x31CD,  # Torque command filter time
    "pn106": 0x31CE,  # Load inertia percentage
    "pn112": 0x31D4,  # Speed feedforward
    "pn113": 0x31D5,  # Speed feedforward filter time
    "pn114": 0x31D6,  # Torque feedforward
    "pn115": 0x31D7,  # Torque feedforward filter time
}

# ---------------------------------------------------------------------------
# Pn100 sub-field definitions (nibble-packed composite parameter)
# ---------------------------------------------------------------------------
# Raw CoE value is a 16-bit integer where each hex nibble is a sub-field:
#   Nibble 0 (bits 0–3):  Pn100.0  Tuning Mode
#   Nibble 1 (bits 4–7):  Pn100.1  Reserved (always 0)
#   Nibble 2 (bits 8–11): Pn100.2  Automatic Vibration Suppression
#   Nibble 3 (bits 12–15): Pn100.3  Damping Selection

TUNING_MODES: list[tuple[int, str]] = [
    (1, "Tuning-less (automatic)"),
    (3, "One-param auto-tuning"),
    (5, "Manual tuning"),
]

TUNING_MODE_VALUES = [m[0] for m in TUNING_MODES]
TUNING_MODE_LABELS = [m[1] for m in TUNING_MODES]

VIBRATION_SUPPRESSION_OPTIONS: list[tuple[int, str]] = [
    (0, "Disabled"),
    (1, "Enabled"),
]
VIBRATION_SUPPRESSION_VALUES = [v[0] for v in VIBRATION_SUPPRESSION_OPTIONS]
VIBRATION_SUPPRESSION_LABELS = [v[1] for v in VIBRATION_SUPPRESSION_OPTIONS]

DAMPING_OPTIONS: list[tuple[int, str]] = [
    (0, "Disabled"),
    (1, "Enabled"),
]
DAMPING_VALUES = [v[0] for v in DAMPING_OPTIONS]
DAMPING_LABELS = [v[1] for v in DAMPING_OPTIONS]


def decode_pn100(raw: int) -> dict:
    """Decode nibble-packed Pn100 into sub-fields."""
    return {
        "tuning_mode": (raw >> 0) & 0xF,
        "vibration_suppression": (raw >> 8) & 0xF,
        "damping": (raw >> 12) & 0xF,
    }


def encode_pn100(tuning_mode: int, vibration_suppression: int = 0, damping: int = 0) -> int:
    """Encode Pn100 sub-fields back to nibble-packed raw value."""
    return (tuning_mode & 0xF) | ((vibration_suppression & 0xF) << 8) | ((damping & 0xF) << 12)

# ---------------------------------------------------------------------------
# Parameter definitions — each entry:
#   (attr, pn_code, label, unit, min_val, max_val, default, tooltip)
# attr=None means it uses a QComboBox (special handling)
# ---------------------------------------------------------------------------
PARAM_DEFS: list[tuple] = [
    (
        "pn100_tuning_mode", "Pn100.0", "Tuning Mode", "",
        None, None, 1,
        "1=Tuning-less (auto), 3=One-param auto, 5=Manual. "
        "Affects which parameters the drive uses actively.",
    ),
    (
        "pn100_vibration", "Pn100.2", "Vibration Suppression", "",
        None, None, 0,
        "Automatic vibration suppression. 0=Disabled, 1=Enabled.",
    ),
    (
        "pn100_damping", "Pn100.3", "Damping Selection", "",
        None, None, 0,
        "Damping selection. 0=Disabled, 1=Enabled.",
    ),
    (
        "pn101", "Pn101", "Servo Rigidity", "Hz",
        0, 500, 40,
        "Response frequency for auto-tuning modes. Higher = stiffer/faster "
        "response but more prone to oscillation. Only used in modes 1 and 3.",
    ),
    (
        "pn102", "Pn102", "Speed Loop Gain", "rad/s",
        1, 10000, 500,
        "Bandwidth of the speed PI loop. Increase to improve speed response "
        "and reduce following error. Decrease if high-frequency vibration appears.",
    ),
    (
        "pn103", "Pn103", "Speed Loop Ti", "×0.1 ms",
        1, 5000, 125,
        "Integral time of the speed loop. Reduce to eliminate steady-state "
        "speed error faster. Too low causes instability.",
    ),
    (
        "pn104", "Pn104", "Position Loop Gain", "1/s",
        0, 1000, 40,
        "Bandwidth of the outer position loop (Kp). Increase for tighter "
        "position tracking. Must stay below ~1/3 of speed loop bandwidth.",
    ),
    (
        "pn105", "Pn105", "Torque Filter", "×0.01 ms",
        0, 2500, 50,
        "Low-pass filter on the torque (current) command. Increase to reduce "
        "high-frequency torque noise / motor noise. Reduces effective torque "
        "loop bandwidth.",
    ),
    (
        "pn106", "Pn106", "Load Inertia", "%",
        0, 9999, 0,
        "Load-to-motor inertia ratio in percent. "
        "(load inertia / motor rotor inertia) × 100. "
        "Required for modes 3 and 5; used by auto-tuning to scale gains.",
    ),
    (
        "pn112", "Pn112", "Speed Feedforward", "%",
        0, 100, 0,
        "Percentage of speed feedforward injected into the speed loop. "
        "Increase to reduce position following error during constant-velocity "
        "moves. Can introduce overshoot if too high.",
    ),
    (
        "pn114", "Pn114", "Torque Feedforward", "%",
        0, 100, 0,
        "Percentage of torque feedforward injected into the torque loop. "
        "Reduces speed following error during acceleration/deceleration. "
        "Used when internal torque feedforward is selected (Pn005.2=0).",
    ),
    (
        "pn115", "Pn115", "Torque FF Filter", "×0.1 ms",
        0, 640, 0,
        "Low-pass filter on the torque feedforward signal. "
        "Increase to filter noise from the torque feedforward differential. "
        "Too high may increase overshoot.",
    ),
]

# Attrs that use QComboBox (not QSpinBox)
COMBO_ATTRS = {"pn100_tuning_mode", "pn100_vibration", "pn100_damping"}

# Drive type choices shown in the UI
DRIVE_TYPES = ["None", "DX3", "DX4", "Other"]


# ---------------------------------------------------------------------------
# DriveProfile dataclass
# ---------------------------------------------------------------------------
@dataclass
class DriveProfile:
    """Holds the user-configured tuning parameters for one servo axis."""

    drive_type: str = "None"   # "None" | "DX3" | "DX4" | "Other"

    # Pn100 sub-fields (nibble-packed in the drive)
    pn100_tuning_mode: Optional[int] = None      # Pn100.0 Tuning mode (1/3/5)
    pn100_vibration: Optional[int] = None         # Pn100.2 Vibration suppression (0/1)
    pn100_damping: Optional[int] = None           # Pn100.3 Damping selection (0/1)

    # Pn parameters — None means "not set / unknown"
    pn101: Optional[int] = None   # Servo rigidity (Hz)
    pn102: Optional[int] = None   # Speed loop gain (rad/s)
    pn103: Optional[int] = None   # Speed loop Ti (×0.1ms)
    pn104: Optional[int] = None   # Position loop gain (1/s)
    pn105: Optional[int] = None   # Torque filter (×0.01ms)
    pn106: Optional[int] = None   # Load inertia (%)
    pn112: Optional[int] = None   # Speed feedforward (%)
    pn114: Optional[int] = None   # Torque feedforward (%)
    pn115: Optional[int] = None   # Torque feedforward filter (×0.1ms)

    # -----------------------------------------------------------------------
    def has_drive_params(self) -> bool:
        """True if the profile type is a known Trio drive with parameters."""
        return self.drive_type in ("DX3", "DX4")

    def has_any_values(self) -> bool:
        """True if at least one Pn parameter was successfully read (non-None)."""
        return any(v is not None for k, v in self.to_dict().items() if k != "drive_type")

    def to_dict(self) -> dict:
        return {
            "drive_type": self.drive_type,
            "pn100_tuning_mode": self.pn100_tuning_mode,
            "pn100_vibration": self.pn100_vibration,
            "pn100_damping": self.pn100_damping,
            "pn101": self.pn101,
            "pn102": self.pn102,
            "pn103": self.pn103,
            "pn104": self.pn104,
            "pn105": self.pn105,
            "pn106": self.pn106,
            "pn112": self.pn112,
            "pn114": self.pn114,
            "pn115": self.pn115,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriveProfile":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    # -----------------------------------------------------------------------
    def format_for_ai(self, axis: int) -> str:
        """
        Return a text block describing this drive profile for injection into
        the AI prompt.  Returns empty string for None/Other profiles.
        """
        if not self.has_drive_params():
            return ""

        lines = [
            f"=== Drive Profile: Axis {axis} — Trio {self.drive_type} ===",
            "The following parameters are the current drive-level tuning values",
            "(manually entered by the user from the drive commissioning tool).",
            "Use these when suggesting drive-level tuning changes.",
            "",
        ]

        # Tuning mode (Pn100 sub-fields)
        if self.pn100_tuning_mode is not None:
            mode_label = {
                1: "Tuning-less (automatic)",
                3: "One-parameter auto-tuning",
                5: "Manual tuning",
            }.get(self.pn100_tuning_mode, str(self.pn100_tuning_mode))
            lines.append(f"  Pn100.0 Tuning Mode     : {mode_label}")
        if self.pn100_vibration is not None:
            lines.append(f"  Pn100.2 Vibration Suppr.: {'Enabled' if self.pn100_vibration else 'Disabled'}")
        if self.pn100_damping is not None:
            lines.append(f"  Pn100.3 Damping         : {'Enabled' if self.pn100_damping else 'Disabled'}")

        param_map = [
            ("pn101", "Pn101 Servo Rigidity     ", "Hz",
             "auto-tuning response stiffness"),
            ("pn102", "Pn102 Speed Loop Gain    ", "rad/s",
             "inner speed loop Kp — higher = faster speed response"),
            ("pn103", "Pn103 Speed Loop Ti      ", "×0.1 ms",
             "speed loop integral time — lower = faster integral action"),
            ("pn104", "Pn104 Position Loop Gain ", "1/s",
             "outer position loop Kp — higher = tighter position tracking"),
            ("pn105", "Pn105 Torque Filter      ", "×0.01 ms",
             "torque command low-pass filter — higher = smoother but slower"),
            ("pn106", "Pn106 Load Inertia       ", "%",
             "load/motor inertia ratio — critical for gain scaling"),
            ("pn112", "Pn112 Speed Feedforward  ", "%",
             "reduces position following error during constant-velocity moves"),
            ("pn114", "Pn114 Torque Feedforward ", "%",
             "reduces speed following error during accel/decel"),
            ("pn115", "Pn115 Torque FF Filter  ", "×0.1 ms",
             "low-pass filter on torque feedforward — higher = smoother but more overshoot"),
        ]

        for attr, label, unit, note in param_map:
            val = getattr(self, attr)
            if val is not None:
                default = next(
                    (d[6] for d in PARAM_DEFS if d[0] == attr), None
                )
                default_str = f", default={default}" if default is not None else ""
                lines.append(f"  {label}: {val} {unit}  ({note}{default_str})")

        lines.append("")
        lines.append(
            "EtherCAT object IDs (for reference, auto-read not yet implemented): "
            + ", ".join(
                f"Pn{attr[2:].zfill(3)}=0x{oid:04X}"
                for attr, oid in ETHERCAT_OBJECT_IDS.items()
            )
        )
        return "\n".join(lines)
