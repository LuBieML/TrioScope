import Trio_UnifiedApi as TUA

IP = "192.168.0.250"
conn = TUA.TrioConnectionTCP(lambda *a: None, IP)

print("Methods on conn:")
for attr in sorted(dir(conn)):
    if any(term in attr.lower() for term in ["ethercat", "read", "write", "sdo", "coe"]):
        print(f"  {attr}")
