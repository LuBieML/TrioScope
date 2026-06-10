import unittest

import numpy as np

from src.scope.scope_engine import ScopeEngine


class FakeScopeConnection:
    def __init__(self, scope_on_error=None):
        self.calls = []
        self.scope_on_error = scope_on_error
        self.scope_pos = 0

    def ScopeOn(self, *args):
        self.calls.append(("ScopeOn", args))
        if self.scope_on_error is not None:
            raise self.scope_on_error

    def Trigger(self, rearm=False):
        self.calls.append(("Trigger", rearm))

    def ScopeOff(self):
        self.calls.append(("ScopeOff",))

    def Execute(self, command):
        self.calls.append(("Execute", command))

    def GetSystemParameter_SCOPE_POS(self):
        return self.scope_pos

    def GetMultiTableValues(self, start, count, out):
        self.calls.append(("GetMultiTableValues", start, count))
        # TABLE[i] = i so tests can verify which entries were sliced out
        out[:] = np.arange(start, start + count, dtype=np.float64)


class ScopeEngineStartCaptureTests(unittest.TestCase):
    def test_arm_capture_uses_scope_api_without_trigger(self):
        connection = FakeScopeConnection()
        engine = ScopeEngine(connection)
        engine.servo_period_sec = 0.001
        engine.tsize = 1000
        engine.configure(
            ["MPOS AXIS(0)"],
            ["MPOS(0)"],
            period_cycles=1,
            duration_seconds=0.01,
            table_start=30,
        )

        engine.arm_capture()

        self.assertEqual(
            connection.calls,
            [("ScopeOn", (1, 30, 39, ["MPOS AXIS(0)"]))],
        )
        self.assertTrue(engine.is_armed)
        self.assertFalse(engine.is_capturing)

    def test_trigger_capture_starts_already_armed_scope(self):
        connection = FakeScopeConnection()
        engine = ScopeEngine(connection)
        engine.servo_period_sec = 0.001
        engine.tsize = 1000
        engine.configure(
            ["MPOS AXIS(0)"],
            ["MPOS(0)"],
            period_cycles=1,
            duration_seconds=0.01,
            table_start=30,
        )
        engine.arm_capture()
        connection.calls.clear()

        engine.trigger_capture(auto_retrigger=True)

        self.assertEqual(connection.calls, [("Trigger", True)])
        self.assertTrue(engine.is_armed)
        self.assertTrue(engine.is_capturing)

    def test_trigger_capture_requires_armed_scope(self):
        connection = FakeScopeConnection()
        engine = ScopeEngine(connection)

        with self.assertRaisesRegex(RuntimeError, "SCOPE is not armed"):
            engine.trigger_capture()

        self.assertEqual(connection.calls, [])

    def test_start_capture_uses_scope_api_not_execute(self):
        connection = FakeScopeConnection()
        engine = ScopeEngine(connection)
        engine.servo_period_sec = 0.001
        engine.tsize = 1000
        engine.configure(
            ["MPOS AXIS(0)", "DPOS AXIS(0)"],
            ["MPOS(0)", "DPOS(0)"],
            period_cycles=2,
            duration_seconds=0.01,
            table_start=10,
        )

        engine.start_capture(auto_retrigger=True)

        self.assertEqual(
            connection.calls,
            [
                (
                    "ScopeOn",
                    (2, 10, 19, ["MPOS AXIS(0)", "DPOS AXIS(0)"]),
                ),
                ("Trigger", True),
            ],
        )
        self.assertTrue(engine.is_capturing)

    def test_stop_capture_uses_scope_off_api(self):
        connection = FakeScopeConnection()
        engine = ScopeEngine(connection)
        engine.is_capturing = True

        engine.stop_capture()

        self.assertEqual(connection.calls, [("ScopeOff",)])
        self.assertFalse(engine.is_capturing)

    def test_start_capture_falls_back_for_scope_on_string_view_error(self):
        connection = FakeScopeConnection(
            RuntimeError(
                "NumPy type info missing for class "
                "std::basic_string_view<char,struct std::char_traits<char> >"
            )
        )
        engine = ScopeEngine(connection)
        engine.servo_period_sec = 0.001
        engine.tsize = 1000
        engine.configure(
            ["OUT(0)"],
            ["OUT Ch(0)"],
            period_cycles=1,
            duration_seconds=0.01,
            table_start=20,
        )

        engine.start_capture(auto_retrigger=True)

        self.assertEqual(
            connection.calls,
            [
                ("ScopeOn", (1, 20, 29, ["OUT(0)"])),
                ("Execute", "SCOPE(ON, 1, 20, 29, READ_OP(0))"),
                ("Execute", "TRIGGER(1)"),
            ],
        )
        self.assertTrue(engine.is_capturing)

    def test_scope_on_string_view_fallback_is_silent_and_cached(self):
        connection = FakeScopeConnection(
            RuntimeError(
                "NumPy type info missing for class "
                "std::basic_string_view<char,struct std::char_traits<char> >"
            )
        )
        engine = ScopeEngine(connection)
        engine.servo_period_sec = 0.001
        engine.tsize = 1000
        engine.configure(
            ["OUT(0)"],
            ["OUT Ch(0)"],
            period_cycles=1,
            duration_seconds=0.01,
            table_start=20,
        )

        with self.assertNoLogs("src.scope.scope_engine", level="WARNING"):
            engine.arm_capture()

        connection.calls.clear()

        with self.assertNoLogs("src.scope.scope_engine", level="WARNING"):
            engine.arm_capture()

        self.assertEqual(
            connection.calls,
            [("Execute", "SCOPE(ON, 1, 20, 29, READ_OP(0))")],
        )


class ScopeEngineReadNewDataTests(unittest.TestCase):
    def _make_engine(self):
        connection = FakeScopeConnection()
        engine = ScopeEngine(connection)
        engine.servo_period_sec = 0.001
        engine.tsize = 1000
        # 2 params, table 0..19 → 10 samples per parameter block
        engine.configure(
            ["MPOS AXIS(0)", "DPOS AXIS(0)"],
            ["MPOS(0)", "DPOS(0)"],
            period_cycles=1,
            duration_seconds=0.01,
            table_start=0,
        )
        return engine, connection

    def test_large_batch_uses_single_spanning_read(self):
        engine, connection = self._make_engine()
        connection.scope_pos = 10  # 6 new samples since last_read_pos=4

        data, new_pos = engine.read_new_data(4)

        # span = 1*10 + 6 = 16 ≤ 2 * useful (24) → one round-trip
        self.assertEqual(
            connection.calls,
            [("GetMultiTableValues", 4, 16)],
        )
        self.assertEqual(new_pos, 10)
        self.assertEqual(data['num_samples'], 6)
        # Param 0 block at 0..9, new region 4..9; param 1 block at 10..19,
        # new region 14..19 (TABLE[i] = i in the fake connection)
        np.testing.assert_array_equal(
            data['params']['MPOS(0)'], np.arange(4, 10, dtype=np.float64))
        np.testing.assert_array_equal(
            data['params']['DPOS(0)'], np.arange(14, 20, dtype=np.float64))

    def test_small_batch_reads_each_parameter_block(self):
        engine, connection = self._make_engine()
        connection.scope_pos = 1  # 1 new sample since last_read_pos=0

        data, new_pos = engine.read_new_data(0)

        # span = 1*10 + 1 = 11 > 2 * useful (4) → per-parameter reads
        self.assertEqual(
            connection.calls,
            [
                ("GetMultiTableValues", 0, 1),
                ("GetMultiTableValues", 10, 1),
            ],
        )
        self.assertEqual(new_pos, 1)
        np.testing.assert_array_equal(
            data['params']['MPOS(0)'], np.array([0.0]))
        np.testing.assert_array_equal(
            data['params']['DPOS(0)'], np.array([10.0]))

    def test_both_paths_return_identical_data(self):
        for last_read, scope_pos in [(0, 10), (0, 1), (3, 7), (5, 10)]:
            engine, connection = self._make_engine()
            connection.scope_pos = scope_pos
            data, new_pos = engine.read_new_data(last_read)
            self.assertEqual(new_pos, scope_pos)
            n_new = scope_pos - last_read
            for i, name in enumerate(["MPOS(0)", "DPOS(0)"]):
                block_start = i * 10 + last_read
                np.testing.assert_array_equal(
                    data['params'][name],
                    np.arange(block_start, block_start + n_new,
                              dtype=np.float64),
                    err_msg=f"{name} last_read={last_read} pos={scope_pos}",
                )


if __name__ == "__main__":
    unittest.main()
