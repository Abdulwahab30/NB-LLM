import re
from collections import Counter
from typing import List, Dict, Any


def normalize_spaces(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_repeated_page_numbers(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Removes standalone page numbers like: 1, 12, Page 3, - 4 -
        if re.match(r"^(page\s*)?\-?\s*\d+\s*\-?$", stripped, re.IGNORECASE):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def detect_repeated_headers_footers(pages: List[Dict[str, Any]]) -> set[str]:
    candidates = []

    for page in pages:
        lines = [
            line.strip()
            for line in page["text"].splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        # first 2 and last 2 lines are likely header/footer candidates
        candidates.extend(lines[:2])
        candidates.extend(lines[-2:])

    counts = Counter(candidates)
    total_pages = len(pages)

    repeated = {
        line
        for line, count in counts.items()
        if count >= max(2, int(total_pages * 0.4))
    }

    return repeated


def remove_headers_footers(text: str, repeated_lines: set[str]) -> str:
    lines = text.splitlines()

    cleaned = [
        line
        for line in lines
        if line.strip() not in repeated_lines
    ]

    return "\n".join(cleaned)


def fix_broken_lines(text: str) -> str:
    lines = text.splitlines()
    fixed = []

    for i, line in enumerate(lines):
        current = line.strip()

        if not current:
            fixed.append("")
            continue

        if not fixed:
            fixed.append(current)
            continue

        previous = fixed[-1]

        # Keep headings, bullets, and numbered lists separate
        if is_heading(current) or is_bullet(current):
            fixed.append(current)
            continue

        if previous == "":
            fixed.append(current)
            continue

        # Join broken sentence lines
        if not previous.endswith((".", "?", "!", ":", ";")):
            fixed[-1] = previous + " " + current
        else:
            fixed.append(current)

    return "\n".join(fixed)


def remove_references_noise(text: str) -> str:
    # Remove common PDF noise
    text = re.sub(r"(?i)\bconfidential\b", "", text)
    text = re.sub(r"(?i)\ball rights reserved\b", "", text)
    text = re.sub(r"(?i)\bcopyright\s*©?\s*\d{4}.*", "", text)

    # Remove very long URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove isolated reference markers like [1], [23]
    text = re.sub(r"\[\d+\]", "", text)

    return text


def is_bullet(line: str) -> bool:
    return bool(re.match(r"^(\-|\*|•|\d+\.|\([a-zA-Z0-9]\))\s+", line.strip()))


def is_heading(line: str) -> bool:
    stripped = line.strip()

    if len(stripped) < 3 or len(stripped) > 120:
        return False

    # Numbered headings: 1. Introduction, 2.3 System Design
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", stripped):
        return True

    # All caps headings
    if stripped.isupper() and len(stripped.split()) <= 12:
        return True

    # Title-style short heading
    words = stripped.split()
    if len(words) <= 10:
        capitalized = sum(1 for w in words if w[:1].isupper())
        if capitalized >= max(1, int(len(words) * 0.7)):
            return True

    return False


def mark_headings(text: str) -> str:
    lines = text.splitlines()
    marked = []

    for line in lines:
        stripped = line.strip()

        if is_heading(stripped):
            marked.append(f"\n## {stripped}\n")
        else:
            marked.append(line)

    return "\n".join(marked)


def format_tables(page: Dict[str, Any]) -> str:
    """
    Converts extracted tables into simple markdown-like text.
    Works only if pdf_reader.py provides page['tables'].
    """
    tables = page.get("tables", [])

    if not tables:
        return ""

    output = []

    for table_index, table in enumerate(tables):
        output.append(f"\n\n## Extracted Table {table_index + 1}\n")

        for row in table:
            clean_row = [
                str(cell).strip() if cell is not None else ""
                for cell in row
            ]
            output.append(" | ".join(clean_row))

    return "\n".join(output)


def preprocess_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    repeated_lines = detect_repeated_headers_footers(pages)

    processed_pages = []

    for page in pages:
        text = page["text"]

        text = remove_headers_footers(text, repeated_lines)
        text = remove_repeated_page_numbers(text)
        text = fix_broken_lines(text)
        text = remove_references_noise(text)
        text = normalize_spaces(text)
        text = mark_headings(text)

        table_text = format_tables(page)

        if table_text:
            text = text + "\n\n" + table_text

        processed_pages.append({
            **page,
            "text": text
        })

    return processed_pages