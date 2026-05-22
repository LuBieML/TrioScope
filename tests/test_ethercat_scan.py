import unittest
from unittest.mock import MagicMock, patch
from src.ai.ethercat_scan import scan_network, EthercatNetwork, EthercatSlave

class TestEthercatScanFallback(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        # Mock VR-based state retrieval
        self.connection.Ethercat_GetState_VR.return_value = True
        self.connection.GetVrValue.return_value = 8.0  # Operational state

    def test_fallback_scan_runs_when_all_slaves_unmapped(self):
        # 2 slaves on Slot 0
        self.connection.Ethercat_CheckNumberOfSlaves.return_value = 2
        
        # Both slaves online
        self.connection.Ethercat_CheckSlaveOnline.side_effect = [True, True]
        # Slave addresses 1000 and 1001
        self.connection.Ethercat_GetSlaveAddress.side_effect = [1000, 1001]
        # Both slaves have no axis mapped (-1)
        self.connection.Ethercat_GetSlaveAxis.side_effect = [-1, -1]

        # For the fallback loop:
        # We query axis drive types. Let's return DRIVE_TYPE = 0 for all except axis 0 and 1.
        # Axis 0 will match slave 1000, Axis 1 will match slave 1001.
        def get_drive_type(ax):
            if ax in (0, 1):
                return 15  # some drive type
            return 0

        def get_slot_number(ax, default=0):
            if ax == 0:
                return 1000
            elif ax == 1:
                return 1001
            return 0

        self.connection.GetAxisParameter_DRIVE_TYPE.side_effect = get_drive_type
        self.connection.GetAxisParameter_SLOT_NUMBER.side_effect = get_slot_number
        self.connection.GetAxisParameter_DRIVE_STATUS.return_value = 1

        # Run scan
        net = scan_network(self.connection)

        # Fallback should have run, mapping the slaves
        slot = net.slots[0]
        self.assertEqual(slot.num_slaves, 2)
        self.assertEqual(slot.slaves[0].axis, 0)
        self.assertEqual(slot.slaves[0].drive_type, 15)
        self.assertEqual(slot.slaves[1].axis, 1)
        self.assertEqual(slot.slaves[1].drive_type, 15)

        # Verify get_drive_type was called for axes (at least up to 1)
        self.connection.GetAxisParameter_DRIVE_TYPE.assert_any_call(0)
        self.connection.GetAxisParameter_DRIVE_TYPE.assert_any_call(1)

    def test_fallback_scan_bypassed_when_at_least_one_slave_mapped(self):
        # 2 slaves on Slot 0
        self.connection.Ethercat_CheckNumberOfSlaves.return_value = 2
        
        # Both slaves online
        self.connection.Ethercat_CheckSlaveOnline.side_effect = [True, True]
        # Slave addresses 1000 and 1001
        self.connection.Ethercat_GetSlaveAddress.side_effect = [1000, 1001]
        # One slave has axis 0, other is unmapped (-1)
        self.connection.Ethercat_GetSlaveAxis.side_effect = [0, -1]

        # Reset call history of connection
        self.connection.reset_mock()
        # Setup GetVrValue mock again since reset_mock clears side effects/return values
        self.connection.GetVrValue.return_value = 8.0

        # Run scan
        net = scan_network(self.connection)

        # Verify the slaves have their original axes
        slot = net.slots[0]
        self.assertEqual(slot.num_slaves, 2)
        self.assertEqual(slot.slaves[0].axis, 0)
        self.assertEqual(slot.slaves[1].axis, -1)

        # Fallback scan should NOT have queried axes (GetAxisParameter_DRIVE_TYPE should NOT be called during fallback)
        # Note: GetAxisParameter_DRIVE_TYPE *is* called during pos loop for slave 0 (axis 0 >= 0),
        # but NOT for axis fallback probing. So it should only be called for axis 0.
        # Let's check calls.
        drive_type_calls = [
            call for call in self.connection.mock_calls 
            if call[0] == 'GetAxisParameter_DRIVE_TYPE'
        ]
        # Should only contain a call for axis 0 (the mapped slave)
        self.assertEqual(len(drive_type_calls), 1)
        self.assertEqual(drive_type_calls[0][1], (0,))

if __name__ == "__main__":
    unittest.main()
