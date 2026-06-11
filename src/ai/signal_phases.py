"""
Channel detection, per-phase statistics, and motion-phase segmentation.

Splits a capture into idle / accel / cruise / decel / settle / reversal
masks from the demand velocity, which every downstream analyzer keys off.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .signal_constants import EPS_ACC_FRAC, EPS_VEL_FRAC, SETTLE_MS


def _find_channel(params: dict, *keywords: str) -> str | None:
    """Case-insensitive fuzzy channel name match."""
    for key in params:
        k = key.lower().replace("_", "").replace(" ", "").replace("-", "")
        for kw in keywords:
            if kw in k:
                return key
    return None


@dataclass
class PhaseStats:
    n: int = 0
    mean: float = 0.0
    std: float = 0.0
    rms: float = 0.0
    vmin: float = 0.0
    vmax: float = 0.0

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PhaseStats":
        if arr.size == 0:
            return cls()
        return cls(
            n=int(arr.size),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            rms=float(np.sqrt(np.mean(arr ** 2))),
            vmin=float(np.min(arr)),
            vmax=float(np.max(arr)),
        )

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "rms": round(self.rms, 4),
            "peak_abs": round(max(abs(self.vmin), abs(self.vmax)), 4),
        }


def segment_phases(t: np.ndarray, dvel: np.ndarray, dt: float) -> dict:
    n = len(dvel)
    dacc = np.gradient(dvel, t)

    v_max = float(np.max(np.abs(dvel)))
    a_max = float(np.max(np.abs(dacc)))
    v_thresh = EPS_VEL_FRAC * v_max if v_max > 0 else 1e-9
    a_thresh = EPS_ACC_FRAC * a_max if a_max > 0 else 1e-9

    # --- Reversal detection: zero-crossings of demand velocity ---
    signs = np.sign(dvel)
    sign_diff = signs[:-1] != signs[1:]
    above_thresh = (np.abs(dvel[:-1]) > v_thresh) | (np.abs(dvel[1:]) > v_thresh)
    reversal_indices = np.where(sign_diff & above_thresh)[0]

    reversal_half_width = max(1, int(0.080 / dt))
    reversal = np.zeros(n, dtype=bool)
    for idx in reversal_indices:
        lo = max(0, idx - reversal_half_width + 1)
        hi = min(n, idx + 1 + reversal_half_width)
        reversal[lo:hi] = True

    # --- Phase masks (reversal excluded from everything) ---
    kinematic_idle = np.abs(dvel) <= v_thresh
    moving = ~kinematic_idle & ~reversal

    accel = moving & (dacc > a_thresh)
    decel = moving & (dacc < -a_thresh)
    cruise = moving & ~accel & ~decel
    idle = kinematic_idle & ~reversal

    # --- Settle (moving → ~moving edges, reversal wins on overlap) ---
    settle = np.zeros(n, dtype=bool)
    settle_samples = max(1, int(SETTLE_MS * 1e-3 / dt))
    transitions = np.where(moving[:-1] & ~moving[1:])[0]
    for idx in transitions:
        end = min(n, idx + 1 + settle_samples)
        settle[idx + 1:end] = True
    settle = settle & ~reversal
    idle = idle & ~settle

    # --- Identify individual moves as contiguous runs of `moving` ---
    changes = np.diff(moving.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0]
    if moving[0]:
        starts = np.concatenate(([0], starts))
    if moving[-1]:
        ends = np.concatenate((ends, [n - 1]))
    moves = list(zip(starts.tolist(), ends.tolist()))

    return {
        "idle": idle, "accel": accel, "cruise": cruise,
        "decel": decel, "settle": settle, "moving": moving,
        "reversal": reversal, "moves": moves,
        "n_moves": len(moves), "n_reversals": len(reversal_indices),
        "transitions": transitions,
    }
