import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print("Connected.")

try:
    print("\nCalling API methods directly:")
    try:
        n = conn.Ethercat_CheckNumberOfSlaves(0)
        print(f"  Ethercat_CheckNumberOfSlaves(0): {n} (type: {type(n)})")
    except Exception as e:
        print(f"  Ethercat_CheckNumberOfSlaves(0) failed: {e}")
        
    try:
        state = conn.Ethercat_GetState(0)
        print(f"  Ethercat_GetState(0): {state} (type: {type(state)})")
    except Exception as e:
        print(f"  Ethercat_GetState(0) failed: {e}")
        
    try:
        online = conn.Ethercat_CheckSlaveOnline(0, 6)
        print(f"  Ethercat_CheckSlaveOnline(0, 6): {online} (type: {type(online)})")
    except Exception as e:
        print(f"  Ethercat_CheckSlaveOnline(0, 6) failed: {e}")
        
    try:
        addr = conn.Ethercat_GetSlaveAddress(0, 6)
        print(f"  Ethercat_GetSlaveAddress(0, 6): {addr} (type: {type(addr)})")
    except Exception as e:
        print(f"  Ethercat_GetSlaveAddress(0, 6) failed: {e}")
        
    try:
        axis = conn.Ethercat_GetSlaveAxis(0, 6)
        print(f"  Ethercat_GetSlaveAxis(0, 6): {axis} (type: {type(axis)})")
    except Exception as e:
        print(f"  Ethercat_GetSlaveAxis(0, 6) failed: {e}")

finally:
    conn.CloseConnection()
