import time
import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print("Connected.")

try:
    # 1. Clean up file first
    print("Deleting previous file...")
    try:
        conn.Delete("EC_COE_FIFO")
        print("  Deleted.")
    except Exception as e:
        print(f"  Delete failed: {e}")
        
    # 2. Run transfer
    vr = 901
    conn.SetVrValue(vr, -9999)
    cmd = f"VR({vr})=ethercat($161, 0, 1, $3687, 0, 16000)"
    print(f"Executing: {cmd}")
    conn.Execute(cmd)
    
    # Wait for completion
    start = time.monotonic()
    while time.monotonic() - start < 3.0:
        val = conn.GetVrValue(vr)
        if val != -9999:
            print(f"Command returned: {int(val)}")
            break
        time.sleep(0.05)
    else:
        print("Command timed out!")
        
    # Check if file exists
    exists = int(conn.FileExists("EC_COE_FIFO"))
    crc = int(conn.GetRemoteFileCRC("EC_COE_FIFO"))
    print(f"EC_COE_FIFO after: FileExists={exists}, CRC=0x{crc:04X}")

finally:
    conn.CloseConnection()
