"""
Drive scope data transfer (mixin for DriveScopeEngine).

Moves the captured 0x3687 buffer from the drive to the PC via the
controller-side EC_COE_FIFO file: resolve the EtherCAT device number,
issue the ETHERCAT FIFO transfer, download the file, and hand the
payload to the parsing layer.
"""

import logging
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .drive_scope_coe import _SDO_POLL_MS, _VR_SENTINEL
from .drive_scope_constants import EXPECTED_CAPTURE_BYTES

logger = logging.getLogger(__name__)


class DriveScopeTransferMixin:
    """EC_COE_FIFO transfer and download for DriveScopeEngine."""

    def _candidate_fifo_devices(self) -> List[Tuple[int, str]]:
        """Return plausible EtherCAT device numbers for the FIFO BASIC command.

        co_write_axis/co_read_axis accept Trio axis numbers, but
        ETHERCAT($161, ...) addresses the EtherCAT slave/device.  Different
        controller/API layers expose physical positions and configured station
        addresses differently, so keep all plausible IDs and try them in order.
        """
        candidates: List[Tuple[int, str]] = []

        def add(value: int, source: str) -> None:
            if value <= 0:
                return
            if any(existing == value for existing, _ in candidates):
                return
            logger.info("Adding device candidate %d (%s) for axis %d", value, source, self.axis)
            candidates.append((value, source))

        # 1. API physical position and slave address (highly specific)
        try:
            check_slaves = getattr(self.connection, "Ethercat_CheckNumberOfSlaves", None)
            get_slave_axis = getattr(self.connection, "Ethercat_GetSlaveAxis", None)
            get_slave_addr = getattr(self.connection, "Ethercat_GetSlaveAddress", None)
            if check_slaves is not None and get_slave_axis is not None:
                num_slaves = int(check_slaves(0))
                logger.debug("API: checking %d slaves on slot 0", num_slaves)
                for pos in range(max(0, num_slaves)):
                    mapped_axis = int(get_slave_axis(0, pos))
                    logger.debug("API: slave at position %d has axis %d (expected %d)", pos, mapped_axis, self.axis)
                    if mapped_axis == self.axis:
                        logger.info("API matched axis %d to slave position %d", self.axis, pos)
                        add(pos + 1, f"slave position {pos}")
                        if get_slave_addr is not None:
                            addr = int(get_slave_addr(0, pos))
                            if addr > 0:
                                add(addr, f"slave address {addr}")
        except Exception as exc:
            logger.warning("Could not resolve EtherCAT device via API for axis %d: %s", self.axis, exc)

        # 2. BASIC ETHERCAT functions physical position and address fallback (extremely robust)
        try:
            num_slaves = self._execute_ethercat_vr_function(3, "0", timeout=1.0)
            logger.debug("BASIC ETHERCAT: checking %d slaves on slot 0", num_slaves)
            for pos in range(max(0, num_slaves)):
                slave_axis = self._execute_ethercat_vr_function(5, f"0, {pos}", timeout=1.0)
                logger.debug("BASIC ETHERCAT: slave at position %d has axis %d (expected %d)", pos, slave_axis, self.axis)
                if slave_axis == self.axis:
                    logger.info("BASIC ETHERCAT matched axis %d to slave position %d", self.axis, pos)
                    add(pos + 1, f"slave position {pos} (via BASIC ETHERCAT)")
                    slave_address = self._execute_ethercat_vr_function(4, f"0, {pos}", timeout=1.0)
                    if slave_address > 0:
                        add(slave_address, f"slave address {slave_address} (via BASIC ETHERCAT)")
        except Exception as exc:
            logger.debug("Could not resolve EtherCAT device via ETHERCAT BASIC fallback for axis %d: %s", self.axis, exc)

        # 3. Axis SLOT_NUMBER parameter (configured station address)
        try:
            get_slot_number = getattr(self.connection, "GetAxisParameter_SLOT_NUMBER", None)
            if get_slot_number is not None:
                slot_number = int(get_slot_number(self.axis))
                if slot_number > 0:
                    logger.info("Axis SLOT_NUMBER parameter read: %d", slot_number)
                    add(slot_number, "axis SLOT_NUMBER")
        except Exception as exc:
            logger.warning("Could not read SLOT_NUMBER for axis %d: %s", self.axis, exc)

        # 4. Standard fallback: axis + 1
        fallback = self.axis + 1
        logger.info("Adding standard fallback candidate: %d", fallback)
        add(fallback, "axis+1 reference")

        logger.info("Drive scope FIFO device candidates for axis %d: %s", self.axis, candidates)
        return candidates

    def _execute_ethercat_vr_function(self, func_num: int, extra_args: str = "", timeout: float = 1.0) -> int:
        """Execute an ETHERCAT function that writes its output to a VR parameter.

        Assigning/writing the function value to a scratch VR gives us a numeric
        completion/error/progress code while still using the same BASIC command
        surface as the reference C# implementation.
        """
        vr = self.vr_scratch
        self.connection.SetVrValue(vr, _VR_SENTINEL)
        if extra_args:
            cmd = f"ETHERCAT({func_num}, {extra_args}, {vr})"
        else:
            cmd = f"ETHERCAT({func_num}, {vr})"
        logger.debug("Executing BASIC command: %s", cmd)
        self.connection.Execute(cmd)

        deadline = time.monotonic() + timeout
        poll_s = _SDO_POLL_MS / 1000.0
        while time.monotonic() < deadline:
            val = self.connection.GetVrValue(vr)
            if val != _VR_SENTINEL:
                logger.debug("BASIC command %s returned VR(%d)=%d", cmd, vr, int(val))
                return int(val)
            time.sleep(poll_s)

        raise TimeoutError(f"{cmd} did not return a value")

    def _wait_for_fifo_transfer(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Wait until the controller-side EC_COE_FIFO transfer is complete."""
        # ETHERCAT($161, ...) SDO transfers do not report progress via $142.
        # We sleep for a fixed duration of 2.0 seconds to allow the controller to
        # copy the SDO FIFO to the local file system (matching reference implementation).
        logger.info("Waiting 2.0 seconds for SDO FIFO transfer to complete on the controller...")
        if progress_callback:
            progress_callback(0.15, "FIFO transfer in progress...")
        time.sleep(2.0)
        if progress_callback:
            progress_callback(0.3, "FIFO transfer complete")

    def _start_fifo_transfer(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[int, str]:
        """Start EC_COE_FIFO transfer using the first working device candidate."""
        errors: List[str] = []

        for device, source in self._candidate_fifo_devices():
            self._delete_remote_fifo_file()
            if self.drive_model in ("DX5", "DX1"):
                cmd = f'ETHERCAT($141, 0, {device}, "C", "EC_COE_FIFO", "ASCOPE_data0", -1)'
            else:
                ethercat_args = f"$161, 0, {device}, $3687, 0, {EXPECTED_CAPTURE_BYTES}"
                cmd = f"ethercat({ethercat_args})"
            logger.debug("FIFO transfer candidate from %s: %s", source, cmd)

            try:
                # Match the C# reference: start as a BASIC command.
                self.connection.Execute(cmd)
            except Exception as exc:
                errors.append(f"{device} ({source}): {exc}")
                logger.debug("FIFO transfer candidate failed: %s", errors[-1])
                continue

            logger.info(
                "EC_COE_FIFO transfer command issued using device %d (%s)",
                device, source,
            )

            if progress_callback:
                progress_callback(0.1, "Waiting for FIFO transfer...")

            try:
                if self.drive_model in ("DX5", "DX1"):
                    logger.info("Waiting 5.0 seconds for SDO FIFO file transfer on controller...")
                    time.sleep(5.0)
                else:
                    self._wait_for_fifo_transfer(progress_callback)
            except Exception as exc:
                errors.append(f"{device} ({source}): wait failed: {exc}")
                logger.debug("FIFO transfer candidate failed: %s", errors[-1])
                continue

            fifo_state = self._remote_file_state("EC_COE_FIFO")
            fifo_crc = self._remote_file_crc("EC_COE_FIFO")
            logger.info(
                "Controller EC_COE_FIFO after transfer candidate %d (%s): "
                "FileExists=%s, CRC=%s",
                device, source,
                fifo_state if fifo_state is not None else "n/a",
                f"0x{fifo_crc:04X}" if fifo_crc is not None else "n/a",
            )
            if fifo_state is None or fifo_state != 0:
                return device, source

            errors.append(f"{device} ({source}): EC_COE_FIFO was not created")

        raise RuntimeError(
            "Could not start/verify EC_COE_FIFO transfer. Tried: "
            + "; ".join(errors)
        )

    def _delete_remote_fifo_file(self) -> None:
        """Best-effort cleanup of the controller-side FIFO transfer file."""
        before_state = self._remote_file_state("EC_COE_FIFO")
        if before_state is not None:
            logger.info("Controller EC_COE_FIFO before cleanup: FileExists=%d", before_state)

        delete = getattr(self.connection, "Delete", None)
        if delete is not None:
            try:
                delete("EC_COE_FIFO")
                logger.info("Deleted previous controller EC_COE_FIFO file")
                return
            except Exception as exc:
                logger.debug("Could not delete previous controller EC_COE_FIFO file: %s", exc)

        try:
            self.connection.Execute('FILE "DEL" "EC_COE_FIFO"')
            time.sleep(0.05)
            after_state = self._remote_file_state("EC_COE_FIFO")
            if after_state is not None:
                logger.info("Controller EC_COE_FIFO after cleanup: FileExists=%d", after_state)
            else:
                logger.info('Issued controller cleanup with FILE "DEL" "EC_COE_FIFO"')
        except Exception as exc:
            logger.debug("Could not delete controller EC_COE_FIFO with FILE DEL: %s", exc)

    def _remote_file_state(self, name: str) -> Optional[int]:
        """Return Trio FileExists flag for a controller file, if available."""
        file_exists = getattr(self.connection, "FileExists", None)
        if file_exists is None:
            return None
        try:
            return int(file_exists(name))
        except Exception as exc:
            logger.debug("Could not check controller file %s: %s", name, exc)
            return None

    def _remote_file_crc(self, name: str) -> Optional[int]:
        """Return controller file CRC, if available."""
        get_crc = getattr(self.connection, "GetRemoteFileCRC", None)
        if get_crc is None:
            return None
        try:
            return int(get_crc(name))
        except Exception as exc:
            logger.debug("Could not read controller file CRC for %s: %s", name, exc)
            return None

    def _select_capture_bytes(self, raw_bytes: bytes) -> bytes:
        """Return the 16000-byte capture payload from a downloaded FIFO file.

        The 0x3687 object payload starts at byte zero of the controller's
        EC_COE_FIFO file; the controller merely rounds the file up (typically
        to 0x8100 bytes) with padding after the payload.  This matches the
        working C# reference, which parses the downloaded file from byte 0,
        and was verified against real DX4 captures: with N active channels
        the stale-memory junk past the capture begins exactly at byte
        N × 1000 × 2 of the file, which is only consistent with the payload
        starting at offset 0.  (The remote file is deleted before every
        transfer, so the download never accumulates older captures.)
        """
        n_bytes = len(raw_bytes)
        if n_bytes < EXPECTED_CAPTURE_BYTES:
            logger.warning(
                "FIFO file has %d bytes; expected at least %d, padding capture",
                n_bytes, EXPECTED_CAPTURE_BYTES,
            )
            return raw_bytes + bytes(EXPECTED_CAPTURE_BYTES - n_bytes)

        if n_bytes > EXPECTED_CAPTURE_BYTES:
            logger.info(
                "FIFO file has %d bytes; using first %d (rest is container padding)",
                n_bytes, EXPECTED_CAPTURE_BYTES,
            )
        return raw_bytes[:EXPECTED_CAPTURE_BYTES]

    def _nonzero_byte_ranges(self, data: bytes, merge_gap: int = 16) -> List[Tuple[int, int]]:
        """Return merged nonzero byte ranges as [start, end) pairs."""
        ranges: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for idx, byte in enumerate(data):
            if byte and start is None:
                start = idx
            elif not byte and start is not None:
                ranges.append((start, idx))
                start = None
        if start is not None:
            ranges.append((start, len(data)))

        merged: List[Tuple[int, int]] = []
        for start, end in ranges:
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def read_data(
        self,
        table_start: int = 0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        local_filename: str = "drive_scope.bin",
    ) -> Dict[str, Any]:
        """
        Read captured data from drive data buffer (0x3687) using EC_COE_FIFO
        file transfer — matching the C# reference implementation.

        Steps:
          1. ethercat($161, 0, slave, $3687, 0, 16000) — initiate FIFO transfer
          2. DownloadFile("drive_scope.bin", "EC_COE_FIFO") — download to PC
          3. Parse the binary file (16-bit interleaved words)

        Args:
            table_start: (unused, kept for API compat)
            progress_callback: Called with (progress_0_to_1, status_message).
            local_filename: Local path for the downloaded binary file.

        Returns:
            Dict with 'time', 'sample_period', 'num_samples', 'params'.
        """
        if progress_callback:
            progress_callback(0.0, "Initiating FIFO transfer from drive...")

        logger.info("Reading drive scope data via EC_COE_FIFO transfer...")
        read_start = time.monotonic()

        # Step 1: Initiate CoE FIFO file transfer on the controller
        # $161 = EC_COE_FIFO transfer function
        # 16000 bytes = 8000 words × 2 bytes/word
        self._start_fifo_transfer(progress_callback)

        if progress_callback:
            progress_callback(0.3, "Downloading file from controller...")

        fifo_state = self._remote_file_state("EC_COE_FIFO")
        fifo_crc = self._remote_file_crc("EC_COE_FIFO")
        if fifo_state is not None or fifo_crc is not None:
            logger.info(
                "Controller EC_COE_FIFO before download: FileExists=%s, CRC=%s",
                fifo_state if fifo_state is not None else "n/a",
                f"0x{fifo_crc:04X}" if fifo_crc is not None else "n/a",
            )

        # Step 2: Download the controller-side FIFO to a raw diagnostic file,
        # then compose drive_scope.bin as exactly the 0x3687 payload bytes.
        file_path = pathlib.Path(local_filename)
        raw_file_path = file_path.with_name(f"{file_path.stem}_fifo_raw{file_path.suffix}")
        try:
            # Remove only the raw diagnostic file before transfer.  Keep the
            # previous composed capture until a new FIFO download succeeds.
            if raw_file_path.exists():
                raw_file_path.unlink()
        except OSError as e:
            raise RuntimeError(
                f"Failed to replace previous drive scope files for {local_filename}: {e}"
            ) from e

        # Python API requires a progress callback: (ProgressInfo) -> None
        def _download_progress(info):
            logger.debug("DownloadFile progress: pos=%s", info.current_pos)

        try:
            self.connection.DownloadFile(str(raw_file_path), "EC_COE_FIFO", _download_progress)
        except Exception as e:
            logger.error("DownloadFile failed: %s", e)
            raise RuntimeError(f"Failed to download drive scope data: {e}") from e

        if progress_callback:
            progress_callback(0.8, "Composing binary data...")

        # Step 3: Compose a clean local binary payload.
        if not raw_file_path.exists():
            raise FileNotFoundError(f"Downloaded FIFO file not found: {raw_file_path}")

        raw_bytes = raw_file_path.read_bytes()
        if self.drive_model in ("DX5", "DX1"):
            # No select/strip for DX5/DX1 binary file, run the converter tool on raw bytes
            file_path.write_bytes(raw_bytes)
            if progress_callback:
                progress_callback(0.9, "Running CSV converter...")
            result = self._convert_and_parse_dx5_data(str(file_path))
            if progress_callback:
                progress_callback(1.0, "Data download and parsing complete")
            return result
        else:
            capture_bytes = self._select_capture_bytes(raw_bytes)
            file_path.write_bytes(capture_bytes)
            elapsed = time.monotonic() - read_start

            raw_ranges = self._nonzero_byte_ranges(raw_bytes)
            capture_ranges = self._nonzero_byte_ranges(capture_bytes)
            logger.info(
                "FIFO raw download complete: %d bytes in %.1f s, nonzero ranges=%s",
                len(raw_bytes), elapsed, raw_ranges[:8],
            )
            logger.info(
                "Composed %s: %d bytes, nonzero ranges=%s",
                file_path.name, len(capture_bytes), capture_ranges[:8],
            )

            if progress_callback:
                progress_callback(1.0, "Data download complete")

            return self._parse_raw_bytes(capture_bytes)
