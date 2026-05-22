import unittest
import numpy as np
from src.scope.drive_scope_engine import DriveScopeEngine

class DriveScopeParsingTests(unittest.TestCase):
    def test_parse_binary_data(self):
        # Path to actual binary file in workspace
        bin_path = r"e:\SynologySynchro\Projects\TrioScope\drive_scope.bin"
        with open(bin_path, "rb") as f:
            raw_bytes = f.read()

        engine = DriveScopeEngine(None)
        engine.active_channels = 1
        engine.channel_addresses = [0x0F10] + [0]*7
        engine.sample_time = 8 # 1.0 ms sample period

        result = engine._parse_raw_bytes(raw_bytes)

        # Check result keys
        self.assertIn('time', result)
        self.assertIn('sample_period', result)
        self.assertIn('num_samples', result)
        self.assertIn('params', result)

        # Check shapes and properties
        self.assertEqual(result['sample_period'], 0.001)
        self.assertEqual(result['num_samples'], 1000)
        self.assertEqual(len(result['time']), 1000)

        # 0x0F10 corresponds to SPD_FB_RPM
        expected_name = "SPD_FB_RPM (0x0F10)"
        self.assertIn(expected_name, result['params'])
        
        ch_data = result['params'][expected_name]
        self.assertEqual(len(ch_data), 1000)

        # Verify parsed signal details
        self.assertEqual(ch_data.min(), -31831)
        self.assertEqual(ch_data.max(), 30272)
        self.assertAlmostEqual(ch_data.mean(), 20.37, places=2)
        self.assertEqual(np.count_nonzero(ch_data), 98)

if __name__ == "__main__":
    unittest.main()
