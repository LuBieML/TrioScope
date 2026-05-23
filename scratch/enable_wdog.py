import time
import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print("Connected.")

try:
    vr = 901
    
    print("Turning WDOG ON...")
    conn.Execute("WDOG=ON")
    time.sleep(1.0)
    
    # Read WDOG
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"VR({vr})=WDOG")
    time.sleep(0.2)
    wdog = int(conn.GetVrValue(vr))
    print(f"WDOG state: {wdog}")
    
    for ax in [0, 1, 2]:
        dt = conn.GetAxisParameter_DRIVE_TYPE(ax)
        sn = conn.GetAxisParameter_SLOT_NUMBER(ax)
        print(f"Axis {ax}: DRIVE_TYPE={int(dt) if dt is not None else 'N/A'}, SLOT_NUMBER={int(sn) if sn is not None else 'N/A'}")

finally:
    conn.CloseConnection()
