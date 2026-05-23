import zipfile
import xml.etree.ElementTree as ET
import sys

sys.stdout.reconfigure(encoding='utf-8')

docx_path = "e:/SynologySynchro/Projects/TrioScope/IPD-PLN-T22 COMBO-function design documentV1.0_20200120 1.docx"

def docx_to_text(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # The namespace for Word XML
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Find all paragraph elements
            paragraphs = []
            for p in root.findall('.//w:p', ns):
                texts = [t.text for t in p.findall('.//w:t', ns) if t.text]
                if texts:
                    paragraphs.append("".join(texts))
            return paragraphs
    except Exception as e:
        print(f"Error reading docx: {e}")
        return []

paragraphs = docx_to_text(docx_path)

indices_to_check = ["0x3681", "0x3682", "0x3684", "0x3686"]
printed = set()
for idx, p in enumerate(paragraphs):
    for val in indices_to_check:
        if val.upper() in p.upper():
            print(f"\n================ Match {val} at Paragraph {idx} ================")
            start = max(0, idx - 4)
            end = min(len(paragraphs), idx + 8)
            for j in range(start, end):
                prefix = ">>>" if j == idx else "   "
                print(f"{prefix} [{j}] {paragraphs[j]}")
            printed.add(idx)
