import hashlib
import pathlib
import datetime

files = ["drive_scope.bin", "drive_scope_fifo_raw.bin"]

print("File stats in workspace:")
for f_name in files:
    p = pathlib.Path(f"e:/SynologySynchro/Projects/TrioScope/{f_name}")
    if p.exists():
        data = p.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        print(f"  {f_name}: size={len(data)} bytes, MD5={md5}, Modified={mtime}")
    else:
        print(f"  {f_name} does not exist!")
