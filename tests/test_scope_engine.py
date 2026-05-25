import unittest

from src.scope.scope_engine import ScopeEngine


class FakeScopeConnection:
    def __init__(self, scope_on_error=None):
        self.calls = []
        self.scope_on_error = scope_on_error

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


if __name__ == "__main__":
    unittest.main()
