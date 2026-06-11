"""CaptureController class composition and Qt signals."""

from PySide6.QtCore import Signal

from ui.window_controller import WindowBackedController

from .controller_scope import ControllerScopeCaptureMixin
from .drive_scope import DriveScopeCaptureMixin
from .external_trigger import ExternalTriggerCaptureMixin
from .pipeline import CapturePipelineMixin
from .source_ui import CaptureSourceUiMixin


class CaptureController(
    CaptureSourceUiMixin,
    ControllerScopeCaptureMixin,
    ExternalTriggerCaptureMixin,
    DriveScopeCaptureMixin,
    CapturePipelineMixin,
    WindowBackedController,
):
    """Runs controller SCOPE and drive scope captures for the main window.

    All mutable state lives on the main window (WindowBackedController
    proxies attribute access), so the mixins are pure method carriers.
    """

    sig_capture_progress = Signal(str)
    sig_capture_status = Signal(str)
    sig_capture_stopped = Signal()
