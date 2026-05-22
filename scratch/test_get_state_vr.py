import Trio_UnifiedApi as TUA
import sys
import time

def event_handler(et, ival, sval):
    pass

ip = "192.168.0.245"
conn = TUA.TrioConnectionTCP(event_handler, ip)
conn.OpenConnection()

try:
    print("Reading VR(0)...")
    print("VR(0) =", conn.GetVrValue(0))
    
    # Try Ethercat_GetState_VR
    # Typically, the signature is Ethercat_GetState_VR(slot, vr_index)
    vr_idx = 901
    conn.SetVrValue(vr_idx, -999.0)
    print(f"Calling Ethercat_GetState_VR(0, {vr_idx})...")
    conn.Ethercat_GetState_VR(0, vr_idx)
    time_start = time.time()
    val = conn.GetVrValue(vr_idx)
    print(f"State value in VR({vr_idx}) = {val}")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.CloseConnection()
