"""
Drive scope payload parsing (mixin for DriveScopeEngine).

Decodes the downloaded capture into per-channel numpy arrays: the
DX3/DX4 binary interleaved-word layout (including 32/64-bit multi-word
reconstruction) and the DX5/DX1 converter-tool CSV format.
"""

import logging
import pathlib
from typing import Any, Dict

import numpy as np

from .drive_scope_constants import DRIVE_VARIABLES, SAMPLES_PER_CHANNEL

logger = logging.getLogger(__name__)


class DriveScopeParsingMixin:
    """Capture payload decoding for DriveScopeEngine."""

    def _convert_and_parse_dx5_data(self, local_bin_path: str) -> Dict[str, Any]:
        """Convert DX5/DX1 binary scope data to CSV using AScope2DataDx5.exe, and parse it."""
        import subprocess
        import sys

        bin_dir = pathlib.Path(local_bin_path).parent.resolve()
        csv_path = bin_dir / "data.csv"

        if csv_path.exists():
            try:
                csv_path.unlink()
            except OSError:
                pass

        exe_name = "AScope2DataDx5.exe"
        exe_candidates = [
            pathlib.Path.cwd() / exe_name,
            pathlib.Path(__file__).parent / exe_name,
            pathlib.Path(__file__).parent.parent / exe_name,
            pathlib.Path(sys.argv[0]).parent / exe_name,
        ]

        exe_path = None
        for cand in exe_candidates:
            if cand.exists():
                exe_path = cand
                break

        if exe_path is None:
            exe_path = pathlib.Path(exe_name)

        logger.info("Running converter: %s with args: %s data.csv", exe_path, local_bin_path)

        try:
            res = subprocess.run(
                [str(exe_path), str(pathlib.Path(local_bin_path).resolve()), "data.csv"],
                cwd=str(bin_dir),
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Converter stdout: %s", res.stdout)
        except Exception as e:
            logger.error("Failed to run AScope2DataDx5.exe converter: %s", e)
            raise FileNotFoundError(
                f"Could not convert binary data to CSV. Please ensure {exe_name} is in "
                f"the application directory. Error: {e}"
            ) from e

        if not csv_path.exists():
            raise FileNotFoundError(f"Converter failed to output data.csv at {csv_path}")

        return self._parse_csv_file(str(csv_path))

    def _parse_csv_file(self, csv_path: str) -> Dict[str, Any]:
        """Parse the CSV file outputted by the converter tool."""
        logger.info("Parsing CSV file: %s", csv_path)

        times = []
        col1 = []
        col2 = []
        col3 = []
        col4 = []

        factor = 125e-6 if self.drive_model == "DX5" else 62.5e-6
        sample_period = self.sample_time * factor

        with open(csv_path, "r", encoding="utf-8") as f:
            line_num = 0
            for line in f:
                line_num += 1
                if line_num <= 8:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                fields = stripped.split("\t")
                if len(fields) < 5:
                    continue
                try:
                    v1 = float(fields[1])
                    v2 = float(fields[2])
                    v3 = float(fields[3])
                    v4 = float(fields[4])

                    j = len(times)
                    times.append(j * sample_period)
                    col1.append(v1)
                    col2.append(v2)
                    col3.append(v3)
                    col4.append(v4)
                except ValueError:
                    continue

        num_samples = len(times)
        logger.info("Parsed %d samples from CSV", num_samples)

        result = {
            'time': np.array(times, dtype=np.float64),
            'sample_period': sample_period,
            'num_samples': num_samples,
            'params': {},
        }

        temp_cols = [col1, col2, col3, col4]
        for idx in range(min(len(temp_cols), self.active_channels)):
            addr = self.channel_addresses[idx]
            if addr == 0:
                continue

            if addr in DRIVE_VARIABLES:
                name, desc, unit, dtype_code, dtype_str = DRIVE_VARIABLES[addr]
                display_name = f"{name} (0x{addr:04X})"
            else:
                display_name = f"Ch{idx+1} (0x{addr:04X})"

            result['params'][display_name] = np.array(temp_cols[idx], dtype=np.float64)

        return result

    def _parse_raw_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Parse binary data downloaded via EC_COE_FIFO.

        The data layout is interleaved across the ACTIVE channels only: the
        drive packs the channels with a nonzero address contiguously, 1000
        samples each, so the word stride per sample equals the number of
        active channels (matching the C# reference, which parses a 6-channel
        capture with a 6-word stride).  Everything past the first
        stride × 1000 words of the upload is stale drive memory, not samples,
        and is discarded.
        """
        n_bytes = len(raw_bytes)
        n_words = n_bytes // 2

        # Active channels: configure() packs them at the front of
        # channel_addresses; skip zeros defensively, keeping the original
        # index so display_names stays aligned.
        active = [
            (idx, addr)
            for idx, addr in enumerate(self.channel_addresses[:self.active_channels])
            if addr
        ]
        stride = len(active)

        logger.info(
            "Parsing %d bytes (%d words), %d active channels, "
            "stride=%d words/sample",
            n_bytes, n_words, stride, stride,
        )

        # Build time array
        time_array = np.arange(SAMPLES_PER_CHANNEL) * self.sample_period_sec

        result = {
            'time': time_array,
            'sample_period': self.sample_period_sec,
            'num_samples': SAMPLES_PER_CHANNEL,
            'params': {},
        }

        if stride == 0:
            logger.warning("No active drive scope channels; nothing to parse")
            result['num_samples'] = 0
            result['raw_words'] = np.zeros(0, dtype=np.uint16)
            return result

        # Convert bytes to uint16 array (little-endian)
        raw_words = np.frombuffer(raw_bytes[:n_words * 2], dtype=np.dtype('<u2'))

        # Useful capture data: stride × 1000 words; the rest of the
        # 16000-byte upload is stale drive memory and must not be parsed.
        expected_words = stride * SAMPLES_PER_CHANNEL
        if len(raw_words) < expected_words:
            logger.warning(
                "Got %d words, expected %d (%d ch × %d samples) — padding",
                len(raw_words), expected_words, stride, SAMPLES_PER_CHANNEL,
            )
            padded = np.zeros(expected_words, dtype=np.uint16)
            padded[:len(raw_words)] = raw_words
            raw_words = padded
        elif len(raw_words) > expected_words:
            logger.debug(
                "Got %d words from drive scope FIFO; using first %d, "
                "discarding %d trailing words of stale buffer memory",
                len(raw_words), expected_words, len(raw_words) - expected_words,
            )
            raw_words = raw_words[:expected_words]

        result['raw_words'] = raw_words

        # Reshape to (1000, stride) — each row is one sample across the
        # active channels
        data_2d = raw_words.reshape(SAMPLES_PER_CHANNEL, stride)

        # Extract each active channel with signed interpretation
        skip_channels = 0
        for col, (ch_idx, addr) in enumerate(active):
            if skip_channels > 0:
                skip_channels -= 1
                continue

            # No copy needed — the astype() calls below always allocate new
            # arrays, and data_2d is never mutated.
            raw_ch = data_2d[:, col]

            # Determine display name and data type
            if addr in DRIVE_VARIABLES:
                name, desc, unit, dtype_code, dtype_str = DRIVE_VARIABLES[addr]
                if self.display_names and ch_idx < len(self.display_names):
                    display_name = self.display_names[ch_idx]
                else:
                    display_name = f"{name} (0x{addr:04X})"
            else:
                if self.display_names and ch_idx < len(self.display_names):
                    display_name = self.display_names[ch_idx]
                else:
                    display_name = f"Ch{ch_idx+1} (0x{addr:04X})"
                dtype_str = "Int16"

            # Reconstruction Logic
            if dtype_str == "Int32" and col + 1 < stride:
                raw_high = data_2d[:, col+1]
                combined = (raw_high.astype(np.uint32) << 16) | raw_ch.astype(np.uint32)
                values = combined.astype(np.int32).astype(np.float64)
                skip_channels = 1
                display_name = display_name.replace("_L", "").replace("_L1", "")
            elif dtype_str == "Int64" and col + 3 < stride:
                raw_h1 = data_2d[:, col+1]
                raw_l2 = data_2d[:, col+2]
                raw_h2 = data_2d[:, col+3]
                combined = (
                    (raw_h2.astype(np.uint64) << 48) |
                    (raw_l2.astype(np.uint64) << 32) |
                    (raw_h1.astype(np.uint64) << 16) |
                    raw_ch.astype(np.uint64)
                )
                values = combined.astype(np.int64).astype(np.float64)
                skip_channels = 3
                display_name = display_name.replace("_L1", "")
            else:
                # Convert to signed if needed (C# does: (short)(hi<<8 | lo))
                if dtype_str in ("Int16", "Int32", "Int64"):
                    values = raw_ch.astype(np.int16).astype(np.float64)
                else:
                    values = raw_ch.astype(np.float64)

            result['params'][display_name] = values
            logger.debug(
                "Ch%d %s: min=%.1f max=%.1f mean=%.1f",
                ch_idx, display_name,
                values.min(), values.max(), values.mean(),
            )

        return result
