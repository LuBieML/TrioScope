"""
Main window actions package.

MainWindowActions carries the menu/file/panel/settings actions bound onto
the main window. Implementation modules:

    controller.py     MainWindowActions class composition
    export_import.py  screenshot, CSV export/import, HTML report
    profiles_ui.py    trace profile save/load/manage dialogs
    menu_bar.py       menu bar construction, help and about
    panels.py         tuner/measurements/EtherCAT panel toggles
    settings_ui.py    settings dialog, QSettings persistence, closeEvent
"""

from .controller import MainWindowActions

__all__ = ["MainWindowActions"]
