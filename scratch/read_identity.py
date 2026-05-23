import time
import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print("Connected.")

try:
    vr = 901
    
    # Read Vendor ID (0x1018:1)
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"co_read_axis(0, $1018, 1, 7, {vr})") # type 7 = Unsigned32
    time.sleep(0.2)
    vendor = int(conn.GetVrValue(vr))
    
    # Read Product Code (0x1018:2)
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"co_read_axis(0, $1018, 2, 7, {vr})")
    time.sleep(0.2)
    product = int(conn.GetVrValue(vr))
    
    # Read Software Version (0x100A:0)
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"co_read_axis(0, $100A, 0, 3, {vr})") # type 3 = VisibleString or Unsigned8? Let's try 3
    time.sleep(0.2)
    sw_ver = int(conn.GetVrValue(vr))
    
    # Read Device Name (0x1008:0)
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"co_read_axis(0, $1008, 0, 3, {vr})")
    time.sleep(0.2)
    dev_name = int(conn.GetVrValue(vr))

    print(f"Drive Identity:")
    print(f"  Vendor ID: 0x{vendor:08X}")
    print(f"  Product Code: 0x{product:08X}")
    print(f"  Software Version (raw VR): {sw_ver}")
    print(f"  Device Name (raw VR): {dev_name}")

finally:
    conn.CloseConnection()
