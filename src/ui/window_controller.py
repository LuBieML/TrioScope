from PySide6.QtCore import QObject


class WindowBackedController(QObject):
    """Controller helper that delegates shared UI state back to the main window."""

    _local_attrs = frozenset({"_window"})

    def __init__(self, window):
        super().__init__(window)
        # QObject.__setattr__ (not object.__setattr__): PySide6 >= 6.10
        # rejects object.__setattr__ on QObject-derived instances.
        QObject.__setattr__(self, "_window", window)

    @property
    def window(self):
        return self._window

    def __getattr__(self, name):
        return getattr(self._window, name)

    def __setattr__(self, name, value):
        if name in self._local_attrs:
            QObject.__setattr__(self, name, value)
        else:
            setattr(self._window, name, value)
