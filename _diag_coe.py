"""
Read DX4 Pn parameters using U32 type (confirmed working).
"""
import time
import sys
import Trio_UnifiedApi as TUA

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.0.250"
VR = 900
SENTINEL = -9999.0
U32 = TUA.Co_ObjectType.Unsigned32
U16 = TUA.Co_ObjectType.Unsigned16

def _event_handler(et, ival, sval):
    pass

conn = TUA.TrioConnectionTCP(_event_handler, IP)
conn.SetTcpCommandTimeout(10000)
print(f"Connecting to {IP}...")
conn.OpenConnection()
print(f"Connected: {conn.IsConnected()}")

def coe_read(axis, index, subindex, obj_type, timeout=0.5):
    conn.SetVrValue(VR, SENTINEL)
    conn.Ethercat_CoReadAxis(axis, index, subindex, obj_type, VR)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = conn.GetVrValue(VR)
        if val != SENTINEL:
            return int(val)
        time.sleep(0.03)
    return None

# --- Read ALL Pn objects at 0x31C8–0x31D5 with U32 ---
print("\n=== DX4 Pn parameters (0x31C8–0x31D5) as U32 — axis 0 ===")
pn_map = {
    0x31C8: "Pn100 Tuning Mode",
    0x31C9: "Pn101 Servo Rigidity",
    0x31CA: "Pn102 Speed Loop Gain",
    0x31CB: "Pn103 Speed Loop Ti",
    0x31CC: "Pn104 Position Loop Gain",
    0x31CD: "Pn105 Torque Filter",
    0x31CE: "Pn106 Load Inertia",
    0x31CF: "Pn107",
    0x31D0: "Pn108",
    0x31D1: "Pn109",
    0x31D2: "Pn110",
    0x31D3: "Pn111",
    0x31D4: "Pn112 Speed FF",
    0x31D5: "Pn113 Speed FF Filter",
}
for idx, name in pn_map.items():
    val_u32 = coe_read(0, idx, 0x00, U32)
    val_u16 = coe_read(0, idx, 0x00, U16)
    u32_str = f"{val_u32} (0x{val_u32:08X})" if val_u32 is not None else "N/A"
    u16_str = f"{val_u16}" if val_u16 is not None else "N/A"
    print(f"  0x{idx:04X} {name:25s}  U32={u32_str}  U16={u16_str}")

# --- Read on all 3 axes with U32 ---
print("\n=== Key Pn params across axes 0–2 (U32) ===")
key_pns = [
    (0x31C8, "Pn100"),
    (0x31CA, "Pn102"),
    (0x31CB, "Pn103"),
    (0x31CC, "Pn104"),
    (0x31CD, "Pn105"),
    (0x31CE, "Pn106"),
    (0x31D4, "Pn112"),
]
for axis in range(3):
    print(f"\n  Axis {axis}:")
    for idx, name in key_pns:
        val = coe_read(axis, idx, 0x00, U32)
        if val is not None:
            print(f"    {name} (0x{idx:04X}): {val}")
        else:
            print(f"    {name} (0x{idx:04X}): N/A")

try:
    conn.CloseConnection()
except:
    pass
print("\nDone.")
