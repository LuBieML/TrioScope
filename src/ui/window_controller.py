from PySide6.QtCore import QObject


class WindowBackedController(QObject):
    """Controller helper that delegates shared UI state back to the main window."""

    _local_attrs = frozenset({"_window"})

    def __init__(self, window):
        super().__init__(window)
        # Write via __dict__: shiboken ≥ 6.11 rejects object.__setattr__ on
        # QObject-derived instances ("can't apply this __setattr__").
        self.__dict__["_window"] = window

    @property
    def window(self):
        return self._window

    def __getattr__(self, name):
        return getattr(self._window, name)

    def __setattr__(self, name, value):
        if name in self._local_attrs:
            self.__dict__[name] = value
        else:
            setattr(self._window, name, value)
