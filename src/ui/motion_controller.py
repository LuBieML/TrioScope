"""Asynchronous UAPI orchestration for the Axis Motion popup."""

import logging
import threading

from PySide6.QtCore import Qt, Signal, Slot

from scope.axis_motion import (
    axes_are_idle,
    cancel_axis_moves,
    execute_relative_moves,
    set_axes_enabled,
)
from ui.motion_window import AxisMotionWindow
from ui.window_controller import WindowBackedController


logger = logging.getLogger(__name__)


class MotionController(WindowBackedController):
    """Own the modeless motion window and serialize its hardware operations."""

    motionEnableFinished = Signal(bool, object)
    motionMoveFinished = Signal(object, object)
    motionStopFinished = Signal(object)

    def __init__(self, window):
        super().__init__(window)
        self._motion_operation_lock = threading.Lock()
        self._motion_monitor_lock = threading.Lock()
        self._motion_monitor_events = {}
        self._motion_enabled_commands = []
        self._motion_state_generation = 0
        self.motionEnableFinished.connect(self._on_motion_enable_finished_ui)
        self.motionMoveFinished.connect(self._on_motion_move_finished_ui)
        self.motionStopFinished.connect(self._on_motion_stop_finished_ui)

    def _open_motion_window(self):
        """Open the multi-axis move scheme editor as a modeless window."""
        if self._motion_window is None:
            self._motion_window = AxisMotionWindow(parent=self.window)
            self._motion_window.setAttribute(Qt.WA_DeleteOnClose)
            self._motion_window.destroyed.connect(
                lambda: setattr(self, "_motion_window", None)
            )
            self._motion_window.startRequested.connect(
                self.window._on_motion_start_requested
            )
            self._motion_window.stopRequested.connect(
                self.window._on_motion_stop_requested
            )
            self._motion_window.enableRequested.connect(
                self.window._on_motion_enable_requested
            )
            self._motion_window.set_connection_available(
                bool(self.trio_connected and self.trio_connection)
            )
        self._motion_window.show()
        self._motion_window.raise_()
        self._motion_window.activateWindow()

    @Slot(bool, object)
    def _on_motion_enable_requested(self, enabled, commands):
        """Apply WDOG/SERVO/AXIS_ENABLE without blocking the UI thread."""
        if not self.trio_connected or not self.trio_connection:
            self.motionEnableFinished.emit(
                bool(enabled), RuntimeError("Controller is not connected.")
            )
            return

        connection = self.trio_connection
        command_list = list(commands)
        # Every toggle supersedes an earlier in-flight request. This matters
        # when the popup is closed while an enable worker is still starting.
        self._motion_state_generation += 1
        generation = self._motion_state_generation
        if not enabled:
            self._stop_motion_monitors()

        def _apply_enable_state():
            error = None
            try:
                with self._motion_operation_lock:
                    if generation != self._motion_state_generation:
                        raise RuntimeError("Motion state changed before the command ran.")
                    set_axes_enabled(
                        connection,
                        command_list,
                        bool(enabled),
                        self._conn_lock,
                    )
                    if enabled and (
                        generation != self._motion_state_generation
                        or not self.trio_connected
                        or self.trio_connection is not connection
                    ):
                        set_axes_enabled(
                            connection,
                            command_list,
                            False,
                            self._conn_lock,
                        )
                        raise RuntimeError("Controller disconnected while enabling axes.")
                    if enabled:
                        self._motion_enabled_commands = command_list
                    else:
                        self._motion_enabled_commands = []
            except Exception as exc:
                error = exc
                logger.error("Axis motion enable state failed: %s", exc)
                if enabled:
                    self._motion_enabled_commands = []
            self.motionEnableFinished.emit(bool(enabled), error)

        threading.Thread(
            target=_apply_enable_state,
            name="AxisMotionEnable",
            daemon=True,
        ).start()

    @Slot(object)
    def _on_motion_start_requested(self, commands):
        """Set SPEED and execute UAPI MOVE for every armed axis."""
        command_list = list(commands)
        requested_axes = {command.axis for command in command_list}
        enabled_axes = {command.axis for command in self._motion_enabled_commands}
        if not self.trio_connected or not self.trio_connection:
            self.motionMoveFinished.emit(
                requested_axes, RuntimeError("Controller is not connected.")
            )
            return
        if not requested_axes or not requested_axes.issubset(enabled_axes):
            self.motionMoveFinished.emit(
                requested_axes,
                RuntimeError("The current move scheme is not enabled.")
            )
            return

        connection = self.trio_connection
        monitor_event = threading.Event()
        with self._motion_monitor_lock:
            if any(axis in self._motion_monitor_events for axis in requested_axes):
                self.motionMoveFinished.emit(
                    requested_axes, RuntimeError("That axis is already moving.")
                )
                return
            for axis in requested_axes:
                self._motion_monitor_events[axis] = monitor_event

        def _move_and_monitor():
            try:
                with self._motion_operation_lock:
                    execute_relative_moves(
                        connection,
                        command_list,
                        self._conn_lock,
                    )
                logger.info("Relative MOVE dispatched: %s", command_list)

                # MOVE is asynchronous. Poll IDLE with short, individually
                # locked reads so Stop and the watchdog remain responsive.
                if monitor_event.wait(0.05):
                    return
                while not monitor_event.is_set():
                    if axes_are_idle(
                        connection,
                        requested_axes,
                        self._conn_lock,
                    ):
                        self._remove_motion_monitors(requested_axes, monitor_event)
                        self.motionMoveFinished.emit(requested_axes, None)
                        return
                    if monitor_event.wait(0.1):
                        return
            except Exception as exc:
                logger.error("Axis MOVE failed: %s", exc)
                self._remove_motion_monitors(requested_axes, monitor_event)
                self.motionMoveFinished.emit(requested_axes, exc)

        threading.Thread(
            target=_move_and_monitor,
            name="AxisMotionMove",
            daemon=True,
        ).start()

    @Slot()
    def _on_motion_stop_requested(self):
        """Cancel active and buffered moves only on the involved axes."""
        with self._motion_monitor_lock:
            axes = list(self._motion_monitor_events)
            for event in self._motion_monitor_events.values():
                event.set()
        if not self.trio_connected or not self.trio_connection:
            self.motionStopFinished.emit(RuntimeError("Controller is not connected."))
            return
        connection = self.trio_connection

        def _stop_axes():
            error = None
            try:
                with self._motion_operation_lock:
                    cancel_axis_moves(connection, axes, self._conn_lock)
            except Exception as exc:
                error = exc
                logger.error("Axis motion stop failed: %s", exc)
            with self._motion_monitor_lock:
                for axis in axes:
                    self._motion_monitor_events.pop(axis, None)
            self.motionStopFinished.emit(error)

        threading.Thread(
            target=_stop_axes,
            name="AxisMotionStop",
            daemon=True,
        ).start()

    @Slot(bool, object)
    def _on_motion_enable_finished_ui(self, requested_enabled, error):
        if self._motion_window is not None:
            self._motion_window.complete_enable(requested_enabled, error)

    @Slot(object, object)
    def _on_motion_move_finished_ui(self, axes, error):
        if self._motion_window is not None:
            self._motion_window.complete_move(axes, error)

    @Slot(object)
    def _on_motion_stop_finished_ui(self, error):
        if self._motion_window is not None:
            self._motion_window.complete_stop(error)

    def _sync_motion_connection_state(self):
        if self._motion_window is not None:
            self._motion_window.set_connection_available(
                bool(self.trio_connected and self.trio_connection)
            )

    def _reset_motion_on_disconnect(self):
        self._motion_state_generation += 1
        self._stop_motion_monitors()
        self._motion_enabled_commands = []
        if self._motion_window is not None:
            self._motion_window.set_connection_available(False)

    def _disable_motion_axes_before_disconnect(self):
        """Best-effort synchronous safety shutdown while UAPI is still usable."""
        self._motion_state_generation += 1
        self._stop_motion_monitors()
        connection = self.trio_connection
        if not connection:
            self._reset_motion_on_disconnect()
            return
        try:
            with self._motion_operation_lock:
                commands = list(self._motion_enabled_commands)
                if commands:
                    set_axes_enabled(
                        connection,
                        commands,
                        False,
                        self._conn_lock,
                    )
        except Exception as exc:
            logger.error("Could not disable motion axes before disconnect: %s", exc)
        finally:
            self._reset_motion_on_disconnect()

    def _stop_motion_monitors(self):
        with self._motion_monitor_lock:
            for event in self._motion_monitor_events.values():
                event.set()
            self._motion_monitor_events.clear()

    def _remove_motion_monitors(self, axes, event):
        with self._motion_monitor_lock:
            for axis in axes:
                if self._motion_monitor_events.get(axis) is event:
                    self._motion_monitor_events.pop(axis, None)
