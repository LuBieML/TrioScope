"""
Build script for TrioScope executable.
Usage: python build_exe.py
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

VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
ICON_FILE = ROOT / "assets" / "trioscope.ico"
APP_ICON_FILE = ROOT / "assets" / "trioscope-icon.png"

if not ICON_FILE.is_file() or not APP_ICON_FILE.is_file():
    raise FileNotFoundError(
        "TrioScope icon assets are missing; expected "
        f"{ICON_FILE} and {APP_ICON_FILE}"
    )

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

licenses_file = ROOT / "THIRD_PARTY_LICENSES.txt"
if licenses_file.is_file():
    add_data_args += ["--add-data", f"{licenses_file};."]

# The ICO is embedded in the executable by --icon. The PNG is also bundled so
# Qt can use the same artwork for the running app's windows and taskbar entry.
add_data_args += ["--add-data", f"{APP_ICON_FILE};assets"]

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--name", f"TrioScope_v{__version__}",
    "--onedir",
    "--windowed",
    "--noconfirm",
    "--icon", str(ICON_FILE),
    # Include local src/ package
    "--paths", str(ROOT / "src"),
    # Include Trio native binaries
    *add_binary_args,
    # Include user manual markdown files
    *add_data_args,
    # Bundle every module of the local packages, so a new module is picked up
    # without having to be added to a hidden-import list by hand.
    "--collect-submodules", "scope",
    "--collect-submodules", "ai",
    "--collect-submodules", "ui",
    "--collect-submodules", "models",
    "--collect-submodules", "storage",
    "--collect-submodules", "reports",
    "--collect-submodules", "plot",
    # Hidden imports that PyInstaller may miss
    "--hidden-import", "Trio_UnifiedApi",
    "--hidden-import", "version",
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

print(f"\nBuild complete! Output in dist/TrioScope_v{__version__}/")
