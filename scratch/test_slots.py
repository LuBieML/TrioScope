import Trio_UnifiedApi as TUA
import time

def event_handler(et, ival, sval):
    print(f"EVENT: {et} {ival} {sval}")

ip = "192.168.0.245"
print(f"Connecting to {ip}...")
try:
    conn = TUA.TrioConnectionTCP(event_handler, ip)
    conn.OpenConnection()
except Exception as e:
    print(f"Failed to instantiate/open connection: {e}")
    sys.exit(1)

try:
    print("Connection open. Verifying connection...")
    val = conn.GetVrValue(0)
    print(f"VR(0) = {val}")
    
    for slot in range(4):
        print(f"\nQuerying slot {slot}...")
        try:
            state = conn.Ethercat_GetState(slot)
            print(f"Slot {slot} Ethercat_GetState returned: {state}")
        except Exception as e:
            print(f"Slot {slot} Ethercat_GetState raised error: {type(e).__name__}: {e}")
            
        try:
            n = conn.Ethercat_CheckNumberOfSlaves(slot)
            print(f"Slot {slot} Ethercat_CheckNumberOfSlaves returned: {n}")
        except Exception as e:
            print(f"Slot {slot} Ethercat_CheckNumberOfSlaves raised error: {type(e).__name__}: {e}")
            
except Exception as e:
    print(f"General error: {e}")
finally:
    print("Closing connection...")
    conn.CloseConnection()
