import time
import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print("Connected.")

try:
    vr = 901
    
    # Check current slaves
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"ETHERCAT(3, 0, {vr})")
    time.sleep(0.5)
    print(f"Initial number of slaves: {int(conn.GetVrValue(vr))}")
    
    # Restart network
    print("Stopping EtherCAT network...")
    conn.Execute("ETHERCAT(1, 0)")
    time.sleep(2.0)
    
    print("Starting EtherCAT network (force re-scan)...")
    conn.Execute("ETHERCAT(0, 0, 2)")
    
    print("Waiting 10 seconds for network to initialize...")
    for i in range(10):
        time.sleep(1.0)
        conn.SetVrValue(vr, -9999)
        conn.Execute(f"ETHERCAT($22, 0, {vr})")
        time.sleep(0.1)
        state = int(conn.GetVrValue(vr))
        
        conn.SetVrValue(vr, -9999)
        conn.Execute(f"ETHERCAT(3, 0, {vr})")
        time.sleep(0.1)
        slaves = int(conn.GetVrValue(vr))
        print(f"  [{i+1}s]: EtherCAT State={state}, Slaves={slaves}")

finally:
    conn.CloseConnection()
