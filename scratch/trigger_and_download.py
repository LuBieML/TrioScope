import time
import hashlib
import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print("Connected.")

try:
    vr = 901
    
    # 1. Stop/re-arm
    print("Stopping capture...")
    conn.Execute("co_write_axis(0, $368b, 0, 6, -1, 0)")
    time.sleep(0.1)
    
    # Check status
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"co_read_axis(0, $3680, 0, 6, {vr})")
    time.sleep(0.1)
    status = (int(conn.GetVrValue(vr)) >> 14) & 0x3
    print(f"Status after stop: {status}")
    
    # 2. Start capture
    print("Starting capture...")
    conn.Execute("co_write_axis(0, $368b, 0, 6, -1, 1)")
    
    # Wait for completion
    print("Waiting for capture to complete (status == 2)...")
    start = time.monotonic()
    while time.monotonic() - start < 10.0:
        conn.SetVrValue(vr, -9999)
        conn.Execute(f"co_read_axis(0, $3680, 0, 6, {vr})")
        time.sleep(0.1)
        status = (int(conn.GetVrValue(vr)) >> 14) & 0x3
        print(f"  status={status}")
        if status == 2:
            print("Capture complete!")
            break
        time.sleep(0.5)
    else:
        print("Capture did not complete!")
        
    # 3. Delete remote file
    print("Deleting EC_COE_FIFO...")
    try:
        conn.Delete("EC_COE_FIFO")
        print("  Deleted.")
    except Exception as e:
        print(f"  Delete failed: {e}")
        
    # 4. Start transfer
    print("Starting FIFO transfer...")
    conn.Execute("ethercat($161, 0, 1, $3687, 0, 16000)")
    print("Waiting 2 seconds...")
    time.sleep(2.0)
    
    # Check CRC
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"ETHERCAT(3, 0, {vr})") # Check slaves online
    # Get CRC
    crc = int(conn.GetRemoteFileCRC("EC_COE_FIFO"))
    print(f"Controller EC_COE_FIFO: CRC=0x{crc:04X}")
    
    # Download
    local_file = "e:/SynologySynchro/Projects/TrioScope/scratch/test_diag_capture.bin"
    def _prog(info):
        pass
    print("Downloading file...")
    conn.DownloadFile(local_file, "EC_COE_FIFO", _prog)
    
    # Calculate MD5
    import pathlib
    p = pathlib.Path(local_file)
    data = p.read_bytes()
    md5 = hashlib.md5(data).hexdigest()
    print(f"Downloaded size: {len(data)} bytes, MD5={md5}")

finally:
    conn.CloseConnection()
