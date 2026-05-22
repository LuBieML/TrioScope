import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
from src.scope.parameters import SCOPE_PARAMETERS, CHANNEL_PARAMETERS_SET
from src.models.trace_config import TraceConfig
from src.models.app_settings import AppSettings, ConnectionSettings, CaptureSettings, PlotSettings, DisplaySettings

class TestConfigSerialization(unittest.TestCase):
    def test_parameters_contain_expected_items(self):
        self.assertIn("MPOS", SCOPE_PARAMETERS)
        self.assertIn("DPOS", SCOPE_PARAMETERS)
        self.assertIn("AIN", CHANNEL_PARAMETERS_SET)
        self.assertNotIn("MPOS", CHANNEL_PARAMETERS_SET)

    def test_trace_config_defaults(self):
        config = TraceConfig()
        self.assertEqual(config.param, "MPOS")
        self.assertEqual(config.axis, 0)
        self.assertTrue(config.enabled)
        self.assertFalse(config.fft)
        self.assertFalse(config.drive_mode)
        self.assertIsNone(config.drive_var_address)

    def test_trace_config_serialization(self):
        config = TraceConfig(
            param="DPOS",
            axis=1,
            enabled=False,
            fft=True,
            drive_mode=True,
            drive_var_address=0x0F10
        )
        d = config.to_dict()
        self.assertEqual(d["param"], "DPOS")
        self.assertEqual(d["axis"], 1)
        self.assertFalse(d["enabled"])
        self.assertTrue(d["fft"])
        self.assertTrue(d["drive_mode"])
        self.assertEqual(d["drive_var_address"], 0x0F10)

        # Deserialize
        reconstructed = TraceConfig.from_dict(d)
        self.assertEqual(reconstructed, config)

    def test_app_settings_serialization(self):
        settings = AppSettings(
            connection=ConnectionSettings(ip="192.168.1.100"),
            capture=CaptureSettings(duration="10.0"),
            traces=[TraceConfig(param="MPOS", axis=0), TraceConfig(param="DPOS", axis=1)],
            drive_profiles={0: {"pn100": 1000, "drive_type": "EtherCAT"}}
        )
        d = settings.to_dict()
        self.assertEqual(d["connection"]["ip"], "192.168.1.100")
        self.assertEqual(d["capture"]["duration"], "10.0")
        self.assertEqual(len(d["traces"]), 2)
        self.assertEqual(d["traces"][0]["param"], "MPOS")
        self.assertEqual(d["drive_profiles"][0]["pn100"], 1000)

    def test_settings_store_load_save(self):
        import tempfile
        import os
        from src.storage.settings_store import SettingsStore
        
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as tmp:
            tmp_name = tmp.name
        
        try:
            store = SettingsStore(filename=tmp_name)
            
            # Save configuration
            settings = AppSettings(
                connection=ConnectionSettings(ip="10.0.0.5"),
                capture=CaptureSettings(duration="3.5", use_end_of_table=False),
                traces=[TraceConfig(param="FE", axis=2, enabled=False)]
            )
            store.save(settings)
            
            # Load it back
            loaded = store.load()
            self.assertEqual(loaded.connection.ip, "10.0.0.5")
            self.assertEqual(loaded.capture.duration, "3.5")
            self.assertFalse(loaded.capture.use_end_of_table)
            self.assertEqual(len(loaded.traces), 1)
            self.assertEqual(loaded.traces[0].param, "FE")
            self.assertEqual(loaded.traces[0].axis, 2)
            self.assertFalse(loaded.traces[0].enabled)
        finally:
            os.remove(tmp_name)

    def test_profile_store_load_save(self):
        import tempfile
        import os
        from src.storage.profiles import ProfileStore
        
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as tmp:
            tmp_name = tmp.name
            
        try:
            store = ProfileStore(filename=tmp_name)
            self.assertEqual(store.get_profile_names(), [])
            
            traces = [
                TraceConfig(param="MPOS", axis=0, enabled=True),
                TraceConfig(param="DPOS", axis=1, enabled=False, fft=True)
            ]
            store.save_profile("TestProfile", traces)
            self.assertEqual(store.get_profile_names(), ["TestProfile"])
            
            loaded = store.load_profile("TestProfile")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].param, "MPOS")
            self.assertEqual(loaded[0].axis, 0)
            self.assertTrue(loaded[0].enabled)
            self.assertFalse(loaded[0].fft)
            
            self.assertEqual(loaded[1].param, "DPOS")
            self.assertEqual(loaded[1].axis, 1)
            self.assertFalse(loaded[1].enabled)
            self.assertTrue(loaded[1].fft)
            
            # Rename profile
            store.rename_profile("TestProfile", "NewProfile")
            self.assertEqual(store.get_profile_names(), ["NewProfile"])
            
            # Delete profile
            store.delete_profile("NewProfile")
            self.assertEqual(store.get_profile_names(), [])
        finally:
            os.remove(tmp_name)

    def test_csv_storage_load_save(self):
        import tempfile
        import os
        import numpy as np
        from src.storage.csv_io import CSVStorage
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_name = tmp.name
            
        try:
            time_data = np.array([0.0, 0.1, 0.2])
            params_data = {
                "MPOS(0)": np.array([10.0, 11.0, 12.0]),
                "DPOS(1)": np.array([10.5, 11.5, 12.5])
            }
            CSVStorage.export_data(tmp_name, time_data, params_data)
            
            # Read back
            time_arr, params_dict, traces = CSVStorage.import_data(tmp_name)
            np.testing.assert_array_equal(time_arr, time_data)
            np.testing.assert_array_equal(params_dict["MPOS(0)"], params_data["MPOS(0)"])
            np.testing.assert_array_equal(params_dict["DPOS(1)"], params_data["DPOS(1)"])
            self.assertEqual(traces, [("MPOS", 0), ("DPOS", 1)])
        finally:
            os.remove(tmp_name)
