import Trio_UnifiedApi as TUA

conn_cls = TUA.TrioConnectionTCP
print("\nTrioConnectionTCP GetSystemParameter methods:")
for x in sorted(dir(conn_cls)):
    if not x.startswith('_'):
        if 'getsystemparameter' in x.lower():
            print(f"  {x}")
