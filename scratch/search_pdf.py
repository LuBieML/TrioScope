import pypdf
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

pdfs = [
    r"e:\SynologySynchro\Projects\TrioScope\TRIO-DX4-Servo-Drive.pdf",
    r"e:\SynologySynchro\Projects\TrioScope\Trio_manual_DX3_servo_drive.pdf",
    r"e:\SynologySynchro\Projects\TrioScope\Trio_UnifiedApi_CPP.pdf"
]

search_term = "368"

for pdf_path in pdfs:
    print(f"\n================ Searching {pdf_path} ================")
    try:
        reader = pypdf.PdfReader(pdf_path)
        print(f"Loaded {len(reader.pages)} pages.")
        
        matches_found = 0
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            
            # Find occurrences of 368
            for match in re.finditer(r'(?:0x)?368[0-9a-fA-F]', text, re.IGNORECASE):
                start = max(0, match.start() - 150)
                end = min(len(text), match.end() + 150)
                snippet = text[start:end].replace('\n', ' ')
                print(f"  Page {i+1}: ... {snippet} ...")
                matches_found += 1
                if matches_found >= 15:
                    print("  Too many matches, truncating search for this PDF.")
                    break
            if matches_found >= 15:
                break
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
