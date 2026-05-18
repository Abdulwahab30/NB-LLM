import fitz
from typing import List, Dict, Any


def extract_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    pages = []

    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        text = page.get_text("text")
        tables = []

        try:
            found_tables = page.find_tables()

            for table in found_tables:
                extracted = table.extract()
                tables.append(extracted)

        except Exception:
            tables = []

        pages.append({
            "page": page_index + 1,
            "text": text.strip(),
            "tables": tables,
            "is_scanned": len(text.strip()) < 30
        })

    doc.close()
    return pages