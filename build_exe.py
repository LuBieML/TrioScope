"""
Build script for TrioScope executable.
Usage: python build_exe.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"

# Trio .pyd and its companion DLLs
trio_binaries = [
    (str(VENV_SITE / "Trio_UnifiedApi.cp313-win_amd64.pyd"), "."),
    (str(VENV_SITE / "Trio_UnifiedApi_PCMCAT.dll"), "."),
    (str(VENV_SITE / "Trio_UnifiedApi_TCP.dll"), "."),
]

add_binary_args = []
for src, dst in trio_binaries:
    add_binary_args += ["--add-binary", f"{src};{dst}"]

# Bundle the docs/help markdown manual so the in-app Help menu works in
# the frozen build. PyInstaller --add-data syntax: "src;dest" on Windows.
help_dir = ROOT / "docs" / "help"
add_data_args = []
if help_dir.is_dir():
    add_data_args += ["--add-data", f"{help_dir};docs/help"]

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--name", "TrioScope",
    "--onedir",
    "--windowed",
    # Include local src/ package
    "--paths", str(ROOT / "src"),
    # Include Trio native binaries
    *add_binary_args,
    # Include user manual markdown files
    *add_data_args,
    # Hidden imports that PyInstaller may miss
    "--hidden-import", "Trio_UnifiedApi",
    "--hidden-import", "scope.scope_engine",
    "--hidden-import", "help_window",
    "--hidden-import", "pyqtgraph.opengl",
    "--hidden-import", "OpenGL",
    "--hidden-import", "OpenGL.platform.win32",
    "--hidden-import", "OpenGL.GL",
    # Exclude unnecessary Qt modules to reduce size
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
    str(ROOT / "scope_app.py"),
]

print("Running PyInstaller...")
print(" ".join(cmd))
subprocess.run(cmd, check=True)

print("\nBuild complete! Output in dist/TrioScope/")
