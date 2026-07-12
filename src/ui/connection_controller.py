import logging
import threading
import time

from PySide6.QtCore import QTimer, Signal

try:
    import Trio_UnifiedApi as TUA
except ImportError:
    TUA = None

from scope.scope_engine import ScopeEngine
from scope.drive_scope_engine import DriveScopeEngine
from ui.window_controller import WindowBackedController

logger = logging.getLogger(__name__)


class ConnectionController(WindowBackedController):
    sig_connect_progress = Signal(str)
    sig_connect_result = Signal(object, object, object, str, object)

    def _on_connect_clicked(self):
        if self.trio_connected:
            self.do_disconnect()
        else:
            self.do_connect()

    def _event_handler(self, et, ival, sval):
        """Handle Trio API events — ignore during shutdown."""
        if not self.trio_connection or self._shutting_down:
            return
        if et == TUA.EventType.Error or et == TUA.EventType.Warning:
            ival_repr = hex(ival) if isinstance(ival, int) else ival
            logger.error(f"Trio event: ({ival_repr}) {sval}")

    def _watchdog_loop(self):
        """Heartbeat loop — polls VR(66) every 0.5s, 5s timeout on dead socket."""
        stop_event = self._watchdog_stop
        while not stop_event.wait(0.5):
            if not (self.trio_connection and self.trio_connected):
                continue
            try:
                heartbeat_done = threading.Event()
                heartbeat_error = []

                def _heartbeat():
                    try:
                        with self._conn_lock:
                            # Read VR(0) as a non-destructive connection probe
                            self.trio_connection.GetVrValue(0)
                    except Exception as e:
                        heartbeat_error.append(e)
                    finally:
                        heartbeat_done.set()

                t = threading.Thread(target=_heartbeat, name="ScopeWatchdog", daemon=True)
                t.start()
                if not heartbeat_done.wait(timeout=5.0):
                    if stop_event.is_set():
                        break
                    logger.warning("Watchdog heartbeat timed out after 5s")
                    self._mark_connection_lost()
                    break
                if stop_event.is_set():
                    break
                if heartbeat_error:
                    raise heartbeat_error[0]
            except Exception as exc:
                if stop_event.is_set():
                    break
                if 'Disconnected' in str(exc) or 'No connection' in str(exc):
                    logger.warning(f"Watchdog detected connection loss: {exc}")
                    self._mark_connection_lost()
                    break
                logger.debug(f"Watchdog heartbeat error: {exc}")

    def _start_watchdog(self):
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="ScopeWatchdog", daemon=True)
        self._watchdog_thread.start()

    def _stop_watchdog(self):
        if not self._watchdog_thread:
            return
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=1.0)
        self._watchdog_thread = None

    def _mark_connection_lost(self):
        """Called by watchdog when connection is lost — thread-safe."""
        with self._state_lock:
            if not self.trio_connected:
                return
            self.trio_connected = False
        self._watchdog_stop.set()
        self._disconnect_cooldown_end = time.monotonic() + self._disconnect_cooldown_seconds
        # Schedule UI update on main thread
        QTimer.singleShot(0, self.window._on_connection_lost_ui)

    def _on_connection_lost_ui(self):
        """Update UI after connection loss — runs on main thread."""
        self.trio_connected = False
        self.trio_connection = None
        self.scope_engine = None
        self.drive_scope_engine = None
        if self._tuner_panel is not None:
            self._tuner_panel.set_connection(None)
        self.axis_parameters_tab.set_connection(None)
        self.window._reset_motion_on_disconnect()
        if self._ethercat_map is not None:
            self._ethercat_map.close()
            self._ethercat_map = None
            
        self.status_dot.setStyleSheet("color: #f14c4c; font-size: 16pt;")
        self.status_label.setStyleSheet("color: #f14c4c;")
        self.status_label.setText("Connection lost. Reconnecting in 2s...")
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)
        if self.is_running:
            self.stop_capture()
            self.capture_controller.sig_capture_stopped.emit()
            self.capture_controller.sig_capture_progress.emit("")
            
        # Queue auto-reconnect if not intentionally shutting down
        if not self._shutting_down:
            self._reconnect_timer.start(2000)

    def _attempt_connection_with_timeout(self, conn, timeout_seconds):
        """Open connection with timeout. Returns True/False."""
        connection_opened = threading.Event()
        connection_error = []

        def _open():
            try:
                conn.OpenConnection()
                connection_opened.set()
            except Exception as e:
                connection_error.append(e)
                connection_opened.set()

        thread = threading.Thread(target=_open, name="ScopeConnOpen", daemon=True)
        thread.start()

        elapsed = 0.0
        poll_interval = 0.5
        while elapsed < timeout_seconds:
            remaining = min(poll_interval, timeout_seconds - elapsed)
            if connection_opened.wait(timeout=remaining):
                if connection_error:
                    raise connection_error[0]
                return True
            elapsed += remaining

        logger.warning(f"Connection attempt timed out after {timeout_seconds}s")
        return False

    def _cleanup_connection_async(self, conn):
        """Close a connection in a fire-and-forget thread (avoids blocking on dead socket)."""
        if conn is None:
            return
        def _close():
            try:
                conn.CloseConnection()
            except Exception:
                pass
        threading.Thread(target=_close, name="ScopeCloseCleanup", daemon=True).start()

    def do_connect(self):
        ip = self.ip_edit.text()

        # Check disconnect cooldown
        cooldown_remaining = max(0.0, self._disconnect_cooldown_end - time.monotonic())
        if cooldown_remaining > 0:
            self.status_label.setText(f"Please wait {cooldown_remaining:.1f}s before reconnecting")
            return

        self.btn_connect.setEnabled(False)
        self.status_label.setText(f"Connecting to {ip}...")

        def _connect_worker():
            """Retry loop with escalating timeouts — runs in background thread."""
            for attempt in range(self._max_connection_attempts):
                timeout = self._connection_timeout_seconds[attempt]
                attempt_label = f"Attempt {attempt + 1}/{self._max_connection_attempts}"
                self.sig_connect_progress.emit(f"{attempt_label} (timeout: {timeout}s)")

                try:
                    conn = TUA.TrioConnectionTCP(self._event_handler, ip)
                    succeeded = self._attempt_connection_with_timeout(conn, timeout)

                    if not succeeded:
                        # Timeout — clean up and retry
                        self._cleanup_connection_async(conn)
                        if attempt < self._max_connection_attempts - 1:
                            time.sleep(1.0)
                            continue
                        else:
                            self.sig_connect_result.emit(
                                None, None, None, ip,
                                TimeoutError(f"Connection timed out after {self._max_connection_attempts} attempts"))
                            return

                    # Verify connection with VR probe
                    try:
                        # Read VR(0) as a non-destructive connection probe
                        conn.GetVrValue(0)
                    except Exception as probe_err:
                        logger.warning(f"Connection probe failed: {probe_err}")
                        self._cleanup_connection_async(conn)
                        if attempt < self._max_connection_attempts - 1:
                            time.sleep(1.0)
                            continue
                        else:
                            self.sig_connect_result.emit(
                                None, None, None, ip,
                                ConnectionError(f"Connection verification failed: {probe_err}"))
                            return

                    # Connection verified — initialize scope engine
                    engine = ScopeEngine(conn)
                    servo_period = engine.read_servo_period()
                    engine.read_table_size()
                    self.sig_connect_result.emit(conn, engine, servo_period, ip, None)
                    return

                except TUA.TrioConnectionError as e:
                    logger.error(f"TrioConnectionError attempt {attempt + 1}: {e}")
                    if attempt < self._max_connection_attempts - 1:
                        time.sleep(1.0)
                        continue
                    self.sig_connect_result.emit(None, None, None, ip, e)
                    return

                except Exception as e:
                    logger.error(f"Unexpected error attempt {attempt + 1}: {e}")
                    if attempt < self._max_connection_attempts - 1:
                        time.sleep(1.0)
                        continue
                    self.sig_connect_result.emit(None, None, None, ip, e)
                    return

        threading.Thread(target=_connect_worker, daemon=True).start()

    def _on_connect_progress(self, msg: str):
        self.status_label.setText(f"Connecting to {self.ip_edit.text()}... {msg}")

    def _on_connect_result(self, conn, engine, servo_period, ip_addr, err):
        if err is not None:
            self.btn_connect.setEnabled(True)
            logger.exception("Connection failed")
            
            # Error classification instead of raw QMessageBox Stack Trace
            err_str = str(err)
            if isinstance(err, TimeoutError):
                msg = "Connection timed out"
            elif isinstance(err, ConnectionRefusedError) or "refused" in err_str.lower():
                msg = "Connection refused (controller not ready)"
            elif "unreachable" in err_str.lower() or "host" in err_str.lower() or "10065" in err_str:
                msg = "Host unreachable. Check IP and network."
            else:
                msg = f"Connection failed: {err}"
                
            self.status_label.setText(msg)
            self.status_label.setStyleSheet("color: #f14c4c;")
        else:
            self.trio_connection = conn
            self.trio_connected = True
            self.scope_engine = engine
            self.drive_scope_engine = DriveScopeEngine(conn, axis=0)
            if self._tuner_panel is not None:
                self._tuner_panel.set_connection(conn, self._conn_lock)
            self.axis_parameters_tab.set_connection(conn, self._conn_lock)
            self.window._sync_motion_connection_state()
            self._start_watchdog()
            self.status_dot.setStyleSheet("color: #00cc00; font-size: 16pt;")
            self.status_label.setStyleSheet("color: #d4d4d4;")
            sp_ms = servo_period * 1000 if servo_period else 0
            self.status_label.setText(f"Connected to {ip_addr} (Servo: {sp_ms:.1f}ms)")
            self.table_usage_label.setText(f"TABLE size: {engine.tsize}")
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setEnabled(True)

    def do_disconnect(self):
        """Disconnect with proper cleanup — matching gcode parser pattern."""
        self.btn_connect.setEnabled(False)
        self.status_label.setText("Disconnecting...")

        if self.is_running:
            self.stop_capture()

        self.window._disable_motion_axes_before_disconnect()
        self._stop_watchdog()
        self._shutting_down = True

        if self.trio_connection:
            # Close with 5s timeout to avoid hanging on dead socket
            close_done = threading.Event()

            def _close_thread():
                try:
                    self.trio_connection.CloseConnection()
                except Exception:
                    pass
                finally:
                    close_done.set()

            t = threading.Thread(target=_close_thread, name="ScopeCloseConn", daemon=True)
            t.start()
            if not close_done.wait(timeout=5.0):
                logger.warning("CloseConnection() timed out after 5s — abandoning")

        self.trio_connection = None
        self.trio_connected = False
        self.scope_engine = None
        self.drive_scope_engine = None
        self._shutting_down = False
        if self._tuner_panel is not None:
            self._tuner_panel.set_connection(None)
        self.axis_parameters_tab.set_connection(None)
        self.window._reset_motion_on_disconnect()
        self._disconnect_cooldown_end = time.monotonic() + self._disconnect_cooldown_seconds

        self.status_dot.setStyleSheet("color: #f14c4c; font-size: 16pt;")
        self.status_label.setText("Disconnected")
        self.table_usage_label.setText("")
        self.btn_connect.setText("Connect")
        self.btn_connect.setEnabled(True)

