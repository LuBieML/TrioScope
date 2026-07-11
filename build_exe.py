"""Build the versioned TrioScope Windows application with PyInstaller.

Usage: python build_exe.py

The Trio SDK is discovered from ``TRIOSCOPE_SDK_DIR``, the active Python
installation, or the repository's ``.venv``. The environment variable is the
recommended option for the self-hosted release runner.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from version import __version__


TRIO_DLL_NAMES = ("Trio_UnifiedApi_PCMCAT.dll", "Trio_UnifiedApi_TCP.dll")


def _candidate_sdk_directories() -> list[Path]:
    """Return likely Trio SDK directories in priority order."""
    candidates: list[Path] = []

    configured = os.environ.get("TRIOSCOPE_SDK_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())

    spec = importlib.util.find_spec("Trio_UnifiedApi")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().parent)

    candidates.extend(
        [
            Path(sysconfig.get_paths()["platlib"]),
            ROOT / ".venv" / "Lib" / "site-packages",
        ]
    )

    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def find_trio_binaries() -> tuple[Path, list[Path]]:
    """Locate the Trio extension module and both companion DLLs."""
    searched: list[Path] = []
    for directory in _candidate_sdk_directories():
        searched.append(directory)
        if not directory.is_dir():
            continue

        extension_modules = sorted(directory.glob("Trio_UnifiedApi*.pyd"))
        companion_dlls = [directory / name for name in TRIO_DLL_NAMES]
        if extension_modules and all(path.is_file() for path in companion_dlls):
            return directory, [*extension_modules, *companion_dlls]

    locations = "\n  - ".join(str(path) for path in searched)
    raise FileNotFoundError(
        "Could not find the Trio Unified API extension and companion DLLs. "
        "Install the Trio SDK or set TRIOSCOPE_SDK_DIR to their directory. "
        f"Searched:\n  - {locations}"
    )


def write_windows_version_file() -> Path:
    """Generate PyInstaller metadata from the SemVer application version."""
    core_version = __version__.split("-", 1)[0].split("+", 1)[0]
    major, minor, patch = (int(part) for part in core_version.split("."))
    numeric_version = f"({major}, {minor}, {patch}, 0)"

    version_file = ROOT / "build" / "windows_version_info.txt"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(
        f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={numeric_version},
    prodvers={numeric_version},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [StringStruct('FileDescription', 'TrioScope oscilloscope and analysis tool'),
         StringStruct('FileVersion', '{__version__}'),
         StringStruct('InternalName', 'TrioScope'),
         StringStruct('OriginalFilename', 'TrioScope_v{__version__}.exe'),
         StringStruct('ProductName', 'TrioScope'),
         StringStruct('ProductVersion', '{__version__}')]
      )
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
        encoding="utf-8",
    )
    return version_file


def build_command() -> list[str]:
    sdk_dir, trio_binaries = find_trio_binaries()
    version_file = write_windows_version_file()

    add_binary_args: list[str] = []
    for binary in trio_binaries:
        add_binary_args.extend(["--add-binary", f"{binary};."])

    add_data_args: list[str] = []
    help_dir = ROOT / "docs" / "help"
    if help_dir.is_dir():
        add_data_args.extend(["--add-data", f"{help_dir};docs/help"])

    licenses_file = ROOT / "THIRD_PARTY_LICENSES.txt"
    if licenses_file.is_file():
        add_data_args.extend(["--add-data", f"{licenses_file};."])

    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        f"TrioScope_v{__version__}",
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--version-file",
        str(version_file),
        "--paths",
        str(ROOT / "src"),
        "--paths",
        str(sdk_dir),
        *add_binary_args,
        *add_data_args,
        "--hidden-import",
        "Trio_UnifiedApi",
        "--hidden-import",
        "scope.scope_engine",
        "--hidden-import",
        "help_window",
        "--hidden-import",
        "pyqtgraph.opengl",
        "--hidden-import",
        "OpenGL",
        "--hidden-import",
        "OpenGL.platform.win32",
        "--hidden-import",
        "OpenGL.GL",
        "--exclude-module",
        "PySide6.QtWebEngine",
        "--exclude-module",
        "PySide6.QtWebEngineWidgets",
        "--exclude-module",
        "PySide6.QtMultimedia",
        "--exclude-module",
        "PySide6.QtBluetooth",
        "--exclude-module",
        "PySide6.QtNfc",
        "--exclude-module",
        "PySide6.QtPositioning",
        "--exclude-module",
        "PySide6.QtRemoteObjects",
        "--exclude-module",
        "PySide6.QtSensors",
        "--exclude-module",
        "PySide6.QtSerialPort",
        "--exclude-module",
        "PySide6.QtTextToSpeech",
        "--exclude-module",
        "PySide6.Qt3DCore",
        "--exclude-module",
        "PySide6.Qt3DRender",
        "--exclude-module",
        "PySide6.QtQuick",
        "--exclude-module",
        "PySide6.QtQml",
        str(ROOT / "scope_app.py"),
    ]


def main() -> int:
    command = build_command()
    print("Running PyInstaller...")
    print(subprocess.list2cmdline(command))
    subprocess.run(command, check=True)
    print(f"\nBuild complete! Output in dist/TrioScope_v{__version__}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
