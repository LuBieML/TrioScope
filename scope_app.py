#!/usr/bin/env python3
"""TrioScope application entry point."""

import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ui.main_window import ParameterScopeOscilloscope, main

__all__ = ["ParameterScopeOscilloscope", "main"]


if __name__ == "__main__":
    main()
