import time
import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print("Connected.")

try:
    vr = 901
    
    # Read OP(9)
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"VR({vr})=OP(9)")
    time.sleep(0.2)
    op9 = int(conn.GetVrValue(vr))
    print(f"OP(9) state: {op9}")
    
    # Read WDOG
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"VR({vr})=WDOG")
    time.sleep(0.2)
    wdog = int(conn.GetVrValue(vr))
    print(f"WDOG state: {wdog}")
    
    # If OP(9) is 0, let's turn it ON!
    if op9 == 0:
        print("Drive power (OP(9)) is OFF! Turning it ON...")
        conn.Execute("OP(9, ON)")
        print("Waiting 15 seconds for drive to power up...")
        time.sleep(15.0)
        
        # Now start EtherCAT
        print("Starting EtherCAT network...")
        try:
            conn.Execute("ETHERCAT(0, 0, 0)") # option_flags=0 (no print)
            print("EtherCAT start command sent.")
        except Exception as e:
            print(f"Start command raised exception (could be due to bindings event bug, ignoring): {e}")
            
        print("Waiting 5 seconds for network initialization...")
        time.sleep(5.0)
        
    # Check number of slaves
    conn.SetVrValue(vr, -9999)
    conn.Execute(f"ETHERCAT(3, 0, {vr})")
    time.sleep(0.2)
    num_slaves = int(conn.GetVrValue(vr))
    print(f"Number of slaves: {num_slaves}")
    
    if num_slaves > 0:
        # Check slave axis mappings
        for pos in range(num_slaves):
            conn.SetVrValue(vr, -9999)
            conn.Execute(f"ETHERCAT(5, 0, {pos}, {vr})")
            time.sleep(0.2)
            axis = int(conn.GetVrValue(vr))
            
            conn.SetVrValue(vr, -9999)
            conn.Execute(f"ETHERCAT(4, 0, {pos}, {vr})")
            time.sleep(0.2)
            addr = int(conn.GetVrValue(vr))
            
            print(f"  Slave {pos}: configured address={addr}, axis={axis}")

finally:
    conn.CloseConnection()
