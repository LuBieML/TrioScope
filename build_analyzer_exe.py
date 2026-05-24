"""
Build script for TrioScope Binary Analyzer executable.
Usage: python build_analyzer_exe.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Add src to path so we can import the version string
sys.path.insert(0, str(ROOT / "src"))
try:
    from version import __version__
except ImportError:
    __version__ = "1.0.0"

# Find virtualenv site packages path to bundle dependencies if needed
# (Here we don't have native C++ DLLs like Trio API since the analyzer is local/static parsing only!)
# This is a huge benefit of the analyzer being a pure python parsing tool.

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--name", f"TrioScopeAnalyzer_v{__version__}",
    "--onedir",
    "--windowed",
    # Hidden imports that PyInstaller may miss
    "--hidden-import", "pyqtgraph",
    # Exclude unnecessary modules to minimize exe size
    "--exclude-module", "PySide6.QtWebEngine",
    "--exclude-module", "PySide6.QtWebEngineWidgets",
    "--exclude-module", "PySide6.QtMultimedia",
    "--exclude-module", "PySide6.QtBluetooth",
    "--exclude-module", "PySide6.QtNfc",
    "--exclude-module", "PySide6.QtPositioning",
    "--exclude-module", "PySide6.QtRemoteObjects",
    "--exclude-module", "PySide6.QtSensors",
    "--exclude-module", "PySide6.QtSerialPort",
    "--exclude-module", "PySide6.QtTextToSpeech",
    "--exclude-module", "PySide6.Qt3DCore",
    "--exclude-module", "PySide6.Qt3DRender",
    "--exclude-module", "PySide6.QtQuick",
    "--exclude-module", "PySide6.QtQml",
    # Entry point
    str(ROOT / "bin_analyzer_app.py"),
]

print("Running PyInstaller to compile TrioScope Binary Analyzer...")
print(" ".join(cmd))
try:
    subprocess.run(cmd, check=True)
    print(f"\nBuild complete! Standalone app created in: dist/TrioScopeAnalyzer_v{__version__}/")
except subprocess.CalledProcessError as e:
    print(f"Error during compilation: {e}", file=sys.stderr)
    sys.exit(1)
