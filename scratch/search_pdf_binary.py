import re

pdf_path = r"e:\SynologySynchro\Projects\TrioScope\Trio_UnifiedApi_CPP.pdf"

with open(pdf_path, 'rb') as f:
    content = f.read()

# Search for Ethercat_GetState with some surrounding characters
matches = re.findall(b'.{0,100}Ethercat_GetState.{0,100}', content)
print(f"Found {len(matches)} binary matches for 'Ethercat_GetState':")
for i, m in enumerate(matches[:10]):
    try:
        print(f"  {i}: {m.decode('latin1')}")
    except Exception as e:
        print(f"  {i}: {m}")
