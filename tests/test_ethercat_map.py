import unittest
from unittest.mock import MagicMock, patch
import sys

from PySide6.QtWidgets import QApplication, QWidget

# Create QApplication if it doesn't exist (needed for Qt widgets)
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from src.ai.ethercat_map_window import EthercatMapWindow
from src.ai.ethercat_scan import EthercatNetwork

class DummyParent(QWidget):
    def __init__(self):
        super().__init__()
        self._stop_watchdog = MagicMock()
        self._start_watchdog = MagicMock()

class TestEthercatMapWatchdog(unittest.TestCase):
    @patch('src.ai.ethercat_map_window.scan_network')
    def test_watchdog_toggled_during_scan(self, mock_scan_network):
        # Setup mock network scan return value
        mock_net = MagicMock(spec=EthercatNetwork)
        mock_net.all_slaves = []
        mock_net.active_slots = []
        mock_net.slots = []
        mock_scan_network.return_value = mock_net

        # Create parent and connection mocks
        parent = DummyParent()
        connection = MagicMock()
        conn_lock = MagicMock()

        # Instantiate window
        window = EthercatMapWindow(connection, parent=parent, conn_lock=conn_lock)

        # Trigger scan
        window._start_scan()

        # Watchdog should have been stopped
        parent._stop_watchdog.assert_called_once()

        # Simulate scan finished
        window._on_scan_finished(mock_net)

        # Watchdog should have been started again
        parent._start_watchdog.assert_called_once()

    @patch('src.ai.ethercat_map_window.scan_network')
    def test_watchdog_restarted_on_error(self, mock_scan_network):
        parent = DummyParent()
        connection = MagicMock()
        conn_lock = MagicMock()

        window = EthercatMapWindow(connection, parent=parent, conn_lock=conn_lock)

        # Trigger scan
        window._start_scan()
        parent._stop_watchdog.assert_called_once()

        # Simulate error
        window._on_scan_error("Test Error")

        # Watchdog should have been restarted
        parent._start_watchdog.assert_called_once()

    def test_watchdog_restarted_on_close(self):
        parent = DummyParent()
        connection = MagicMock()
        conn_lock = MagicMock()

        window = EthercatMapWindow(connection, parent=parent, conn_lock=conn_lock)

        # Close window
        window.close()

        # Watchdog should have been restarted (idempotent call)
        parent._start_watchdog.assert_called_once()

if __name__ == "__main__":
    unittest.main()
