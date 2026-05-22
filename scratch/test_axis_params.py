import Trio_UnifiedApi as TUA
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

try:
    print("Connection open. Querying GetAxisParameter_DRIVE_TYPE for axis 0..31...")
    consecutive_fails = 0
    for ax in range(32):
        print(f"Calling GetAxisParameter_DRIVE_TYPE for axis {ax}...")
        try:
            dt = conn.GetAxisParameter_DRIVE_TYPE(ax)
            print(f"  -> Axis {ax} DRIVE_TYPE = {dt}")
            consecutive_fails = 0
        except Exception as e:
            print(f"  -> Axis {ax} raised error: {type(e).__name__}: {e}")
            consecutive_fails += 1
            if consecutive_fails >= 3:
                print("Too many consecutive failures, stopping.")
                break
                
    print("\nVerifying if connection is still alive at the end...")
    try:
        val = conn.GetVrValue(0)
        print(f"  -> VR(0) = {val} (connection is alive!)")
    except Exception as e:
        print(f"  -> Error checking connection: {type(e).__name__}: {e}")

finally:
    print("Closing connection...")
    try:
        conn.CloseConnection()
    except Exception as e:
        print(f"Error closing: {e}")
