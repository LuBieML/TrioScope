"""
Capture controller package.

CaptureController drives both capture paths (controller SCOPE and drive
scope) and the shared data pipeline. Implementation modules:

    controller.py        CaptureController class composition + Qt signals
    source_ui.py         capture-source switching and drive UI helpers
    controller_scope.py  controller SCOPE start + single/continuous threads
    external_trigger.py  externally triggered (Trio BASIC TRIGGER) threads
    drive_scope.py       drive scope (SDO) start + capture thread
    pipeline.py          data buffering, update timer, stop handling
"""

from .controller import CaptureController

__all__ = ["CaptureController"]
