"""MainWindowActions class composition."""

from ui.window_controller import WindowBackedController

from .export_import import ExportImportMixin
from .menu_bar import MenuBarMixin
from .panels import PanelsMixin
from .profiles_ui import TraceProfilesMixin
from .settings_ui import SettingsMixin


class MainWindowActions(
    ExportImportMixin,
    TraceProfilesMixin,
    MenuBarMixin,
    PanelsMixin,
    SettingsMixin,
    WindowBackedController,
):
    """Menu, file, panel, and settings actions for the main window.

    All mutable state lives on the main window (WindowBackedController
    proxies attribute access), so the mixins are pure method carriers.
    """
