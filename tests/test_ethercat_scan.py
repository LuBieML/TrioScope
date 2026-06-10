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

class TestEthercatScanIdentity(unittest.TestCase):
    """Identity Object (0x1018) and Device Type (0x1000) reads during scan."""

    def setUp(self):
        self.connection = MagicMock()
        self.vr = {}

        # VR store emulation
        self.connection.SetVrValue.side_effect = lambda i, v: self.vr.__setitem__(i, v)
        self.connection.GetVrValue.side_effect = lambda i: self.vr.get(i, 0)
        # Master state lands in the scratch VR as Operational (8)
        self.connection.Ethercat_GetState_VR.side_effect = (
            lambda slot, vr_i: self.vr.__setitem__(vr_i, 8.0)
        )

        # One online slave, address 1000, axis 0
        self.connection.Ethercat_CheckNumberOfSlaves.return_value = 1
        self.connection.Ethercat_CheckSlaveOnline.return_value = True
        self.connection.Ethercat_GetSlaveAddress.return_value = 1000
        self.connection.Ethercat_GetSlaveAxis.return_value = 0
        self.connection.GetAxisParameter_DRIVE_TYPE.return_value = 42  # DX4
        self.connection.GetAxisParameter_DRIVE_STATUS.return_value = 0x37
        self.connection.GetAxisParameter_SLOT_NUMBER.return_value = 1000

        # CoE objects served by the fake slave
        coe_objects = {
            (0x1018, 1): 0x000002DE,   # vendor: Trio
            (0x1018, 2): 0x00001234,   # product code
            (0x1018, 3): 0x00010002,   # revision 1.2
            (0x1018, 4): 987654,       # serial
            (0x1000, 0): 0x00020192,   # CiA 402 drive
        }
        self.connection.Ethercat_CoRead.side_effect = (
            lambda slot, pos, idx, sub, typ, vr_i:
                self.vr.__setitem__(vr_i, float(coe_objects[(idx, sub)]))
        )

    def test_identity_populated(self):
        net = scan_network(self.connection)
        slave = net.slots[0].slaves[0]

        self.assertEqual(slave.vendor_id, 0x2DE)
        self.assertEqual(slave.vendor_name, "Trio Motion Technology")
        self.assertEqual(slave.product_code, 0x1234)
        self.assertEqual(slave.revision_str, "1.2")
        self.assertEqual(slave.serial_number, 987654)
        self.assertEqual(slave.device_type, 0x00020192)
        self.assertEqual(slave.profile_name, "Servo drive (CiA 402)")
        # Trio DRIVE_TYPE wins for the short product name
        self.assertEqual(slave.product_name, "DX4")

    def test_identity_skipped_when_disabled(self):
        net = scan_network(self.connection, read_identity=False)
        slave = net.slots[0].slaves[0]

        self.connection.Ethercat_CoRead.assert_not_called()
        self.assertEqual(slave.vendor_id, 0)

    def test_identity_failure_is_non_fatal(self):
        self.connection.Ethercat_CoRead.side_effect = RuntimeError("mailbox down")
        net = scan_network(self.connection)
        slave = net.slots[0].slaves[0]

        self.assertTrue(slave.online)
        self.assertEqual(slave.vendor_id, 0)
        self.assertEqual(slave.drive_type, 42)


if __name__ == "__main__":
    unittest.main()
