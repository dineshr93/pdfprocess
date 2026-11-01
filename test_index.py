"""Self-checks for process_pdfs.py. Plain asserts, no framework."""
from pathlib import Path

import process_pdfs

pdf = next(Path("pdfs").glob("*.pdf"))
pages = process_pdfs.extract_pages(pdf)
assert pages, "extract_pages returned nothing for a text-layer PDF"
assert all(p.metadata["page"] >= 1 and p.metadata["file_name"] == pdf.name for p in pages)
assert pages[0].metadata["page"] == 1, "page numbering starts at 1"

print("self-check passed")
