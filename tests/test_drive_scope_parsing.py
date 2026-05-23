import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from src.scope.drive_scope_engine import (
    CONTROL_INDEX,
    DriveScopeEngine,
    FIFO_CHUNK_BYTES,
    FIFO_CONTAINER_PREFIX_BYTES,
    NUM_CHANNELS,
    SAMPLES_PER_CHANNEL,
)


class FakeDownloadConnection:
    def __init__(self, payload: bytes, slot_number: int = 1, created_devices=None):
        self.payload = payload
        self.slot_number = slot_number
        self.created_devices = set(created_devices or [])
        self.commands = []
        self.vr_values = {}
        self.remote_file_exists = True

    def Execute(self, command):
        self.commands.append(command)
        if command.startswith("ethercat($161"):
            device = int(command.split(",", 3)[2].strip())
            self.remote_file_exists = device in self.created_devices
        if command.startswith("VR(") and "=ETHERCAT(" in command:
            vr = int(command.split(")", 1)[0][3:])
            if "$142" in command:
                self.vr_values[vr] = 100
            else:
                self.vr_values[vr] = 0

    def SetVrValue(self, vr, value):
        self.vr_values[vr] = value

    def GetVrValue(self, vr):
        return self.vr_values.get(vr, -9999.0)

    def GetAxisParameter_SLOT_NUMBER(self, axis):
        return self.slot_number

    def Delete(self, remote_name):
        self.remote_file_exists = False

    def FileExists(self, remote_name):
        return -1 if self.remote_file_exists else 0

    def GetRemoteFileCRC(self, remote_name):
        return 0x1234 if self.remote_file_exists else 0

    def DownloadFile(self, local_filename, remote_name, progress_callback):
        if not self.remote_file_exists:
            raise RuntimeError("remote file missing")
        Path(local_filename).write_bytes(self.payload)

        class Info:
            current_pos = 0

        Info.current_pos = len(self.payload)

        progress_callback(Info())


class FakeCaptureConnection:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.writes = []
        self.vr_values = {}

    def Ethercat_CoWriteAxis_Value(self, axis, index, subindex, obj_type, value):
        self.writes.append((axis, index, subindex, value))

    def Ethercat_CoReadAxis(self, axis, index, subindex, obj_type, vr):
        if self.statuses:
            status = self.statuses.pop(0)
        else:
            status = 2
        self.vr_values[vr] = int(status) << 14

    def SetVrValue(self, vr, value):
        self.vr_values[vr] = value

    def GetVrValue(self, vr):
        return self.vr_values.get(vr, -9999.0)


def make_capture_bytes(ch1_values):
    raw = np.zeros((SAMPLES_PER_CHANNEL, NUM_CHANNELS), dtype=np.uint16)
    raw[:, 0] = np.asarray(ch1_values, dtype=np.int16).view(np.uint16)
    return raw.tobytes()


def make_fifo_chunk(ch1_values):
    payload = make_capture_bytes(ch1_values)
    return (
        bytes(FIFO_CONTAINER_PREFIX_BYTES)
        + payload
        + bytes(FIFO_CHUNK_BYTES - FIFO_CONTAINER_PREFIX_BYTES - len(payload))
    )


class DriveScopeParsingTests(unittest.TestCase):
    def test_parse_binary_data(self):
        values = np.zeros(SAMPLES_PER_CHANNEL, dtype=np.int16)
        values[10:14] = [100, -200, 300, -400]
        raw_bytes = make_capture_bytes(values)

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
        self.assertEqual(ch_data.min(), -400)
        self.assertEqual(ch_data.max(), 300)
        self.assertAlmostEqual(ch_data.mean(), -0.2, places=3)
        self.assertEqual(np.count_nonzero(ch_data), 4)

    def test_read_data_replaces_previous_local_file(self):
        values = np.arange(SAMPLES_PER_CHANNEL, dtype=np.int16)
        payload = make_capture_bytes(values)
        conn = FakeDownloadConnection(payload, created_devices={1})

        engine = DriveScopeEngine(conn)
        engine.active_channels = 1
        engine.channel_addresses = [0x0F10] + [0] * 7
        engine.sample_time = 8

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "drive_scope.bin"
            target.write_bytes(b"\xff" * (len(payload) * 2))

            with patch("src.scope.drive_scope_engine.time.sleep", lambda _seconds: None):
                result = engine.read_data(local_filename=str(target))

            parsed = result["params"]["SPD_FB_RPM (0x0F10)"]
            np.testing.assert_array_equal(parsed, values.astype(np.float64))
            self.assertTrue(target.exists())
            self.assertEqual(target.stat().st_size, len(payload))

    def test_start_capture_rearms_and_records_sampling_transition(self):
        conn = FakeCaptureConnection(statuses=[0, 0, 1])

        engine = DriveScopeEngine(conn)
        engine.is_configured = True
        engine.sample_time = 8

        with patch("src.scope.drive_scope_engine.time.sleep", lambda _seconds: None):
            engine.start_capture()

        self.assertEqual(
            conn.writes,
            [
                (0, CONTROL_INDEX, 0, 0),
                (0, CONTROL_INDEX, 0, 1),
            ],
        )
        self.assertTrue(engine.last_start_saw_sampling)
        self.assertEqual(engine.last_start_status_sequence, [0, 0, 1])

    def test_read_data_uses_axis_slot_number_for_fifo_transfer(self):
        values = np.zeros(SAMPLES_PER_CHANNEL, dtype=np.int16)
        conn = FakeDownloadConnection(make_capture_bytes(values), slot_number=5, created_devices={1})

        engine = DriveScopeEngine(conn, axis=0)
        engine.active_channels = 1
        engine.channel_addresses = [0x0F10] + [0] * 7

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "drive_scope.bin"
            with patch("src.scope.drive_scope_engine.time.sleep", lambda _seconds: None):
                engine.read_data(local_filename=str(target))

        self.assertIn("ethercat($161, 0, 1, $3687, 0, 16000)", conn.commands)

    def test_select_capture_bytes_strips_fifo_container_prefix(self):
        values = np.arange(SAMPLES_PER_CHANNEL, dtype=np.int16)
        raw = make_fifo_chunk(values)

        engine = DriveScopeEngine(None)
        capture = engine._select_capture_bytes(raw)

        self.assertEqual(capture, make_capture_bytes(values))

    def test_read_data_uses_newest_fifo_chunk_when_remote_file_accumulates(self):
        old_values = np.full(SAMPLES_PER_CHANNEL, 111, dtype=np.int16)
        new_values = np.arange(SAMPLES_PER_CHANNEL, dtype=np.int16)
        conn = FakeDownloadConnection(
            make_fifo_chunk(old_values) + make_fifo_chunk(new_values),
            created_devices={1},
        )

        engine = DriveScopeEngine(conn)
        engine.active_channels = 1
        engine.channel_addresses = [0x0F10] + [0] * 7

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "drive_scope.bin"
            with patch("src.scope.drive_scope_engine.time.sleep", lambda _seconds: None):
                result = engine.read_data(local_filename=str(target))

            parsed = result["params"]["SPD_FB_RPM (0x0F10)"]
            np.testing.assert_array_equal(parsed, new_values.astype(np.float64))
            self.assertEqual(target.stat().st_size, SAMPLES_PER_CHANNEL * NUM_CHANNELS * 2)
            raw_target = target.with_name(f"{target.stem}_fifo_raw{target.suffix}")
            self.assertEqual(raw_target.stat().st_size, FIFO_CHUNK_BYTES * 2)

    def test_read_data_retries_when_fifo_candidate_creates_no_file(self):
        values = np.arange(SAMPLES_PER_CHANNEL, dtype=np.int16)
        conn = FakeDownloadConnection(make_capture_bytes(values), slot_number=5, created_devices={5})

        engine = DriveScopeEngine(conn, axis=0)
        engine.active_channels = 1
        engine.channel_addresses = [0x0F10] + [0] * 7

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "drive_scope.bin"
            with patch("src.scope.drive_scope_engine.time.sleep", lambda _seconds: None):
                result = engine.read_data(local_filename=str(target))

        parsed = result["params"]["SPD_FB_RPM (0x0F10)"]
        np.testing.assert_array_equal(parsed, values.astype(np.float64))
        self.assertIn("ethercat($161, 0, 1, $3687, 0, 16000)", conn.commands)
        self.assertIn("ethercat($161, 0, 5, $3687, 0, 16000)", conn.commands)

    def test_read_data_keeps_previous_file_when_fifo_not_created(self):
        values = np.arange(SAMPLES_PER_CHANNEL, dtype=np.int16)
        conn = FakeDownloadConnection(make_capture_bytes(values), created_devices=set())

        engine = DriveScopeEngine(conn, axis=0)
        engine.active_channels = 1
        engine.channel_addresses = [0x0F10] + [0] * 7

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "drive_scope.bin"
            previous = b"previous capture"
            target.write_bytes(previous)

            with patch("src.scope.drive_scope_engine.time.sleep", lambda _seconds: None):
                with self.assertRaises(RuntimeError):
                    engine.read_data(local_filename=str(target))

            self.assertEqual(target.read_bytes(), previous)

if __name__ == "__main__":
    unittest.main()
