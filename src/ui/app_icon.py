"""Resolve TrioScope's application icon in source and frozen builds."""

from __future__ import annotations

import sys
from pathlib import Path


def app_icon_path() -> Path:
    """Return the PNG used by Qt for window and taskbar icons."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parents[2]
    return base / "assets" / "trioscope-icon.png"
