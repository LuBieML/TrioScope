import Trio_UnifiedApi as TUA
import time
import sys

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

def _call(name, fn, *args):
    print(f"Calling {name}{args}...")
    try:
        res = fn(*args)
        print(f"  -> Returned: {res}")
        return res
    except Exception as e:
        print(f"  -> Raised error: {type(e).__name__}: {e}")
        return None

try:
    print("Connection open. Starting step-by-step scan simulator...")
    
    # 1. State check
    vr_scratch = 901
    _call("SetVrValue", conn.SetVrValue, vr_scratch, -999.0)
    _call("Ethercat_GetState_VR", conn.Ethercat_GetState_VR, 0, vr_scratch)
    state_val = _call("GetVrValue", conn.GetVrValue, vr_scratch)
    print(f"State value: {state_val}")

    # 2. Number of slaves
    num_slaves = _call("Ethercat_CheckNumberOfSlaves", conn.Ethercat_CheckNumberOfSlaves, 0)
    if num_slaves is None:
        num_slaves = 0
    else:
        num_slaves = int(num_slaves)
    print(f"Number of slaves: {num_slaves}")

    # 3. Loop slaves
    for pos in range(num_slaves):
        print(f"\n--- Slave {pos} ---")
        online = _call("Ethercat_CheckSlaveOnline", conn.Ethercat_CheckSlaveOnline, 0, pos)
        addr = _call("Ethercat_GetSlaveAddress", conn.Ethercat_GetSlaveAddress, 0, pos)
        axis = _call("Ethercat_GetSlaveAxis", conn.Ethercat_GetSlaveAxis, 0, pos)
        
        if axis is not None:
            axis = int(axis)
            if axis >= 0:
                print(f"  Axis {axis} mapped to Slave {pos}. Querying axis parameters...")
                _call("GetAxisParameter_DRIVE_TYPE", conn.GetAxisParameter_DRIVE_TYPE, axis)
                _call("GetAxisParameter_DRIVE_STATUS", conn.GetAxisParameter_DRIVE_STATUS, axis)
                _call("GetAxisParameter_SLOT_NUMBER", conn.GetAxisParameter_SLOT_NUMBER, axis)

    # 4. Check if connection is still alive
    print("\nVerifying if connection is still alive at the end...")
    _call("GetVrValue", conn.GetVrValue, 0)

except Exception as e:
    print(f"General error: {e}")
finally:
    print("Closing connection...")
    try:
        conn.CloseConnection()
    except Exception as e:
        print(f"Error closing: {e}")
