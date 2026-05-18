import re
from typing import List, Dict, Any


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_into_sections(text: str) -> List[str]:
    heading_pattern = r"\n(?=(?:[A-Z][A-Z\s\d\-:]{5,}|(?:\d+\.?\s+)?[A-Z][^\n]{3,80})\n)"
    sections = re.split(heading_pattern, text)
    return [s.strip() for s in sections if s.strip()]


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_bullet_lists(text: str) -> List[str]:
    lines = text.splitlines()
    blocks = []
    current = []

    for line in lines:
        stripped = line.strip()

        if re.match(r"^(\-|\*|•|\d+\.)\s+", stripped):
            current.append(stripped)
        else:
            if current:
                blocks.append("\n".join(current))
                current = []

            if stripped:
                blocks.append(stripped)

    if current:
        blocks.append("\n".join(current))

    return [b.strip() for b in blocks if b.strip()]


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_token_limit(text: str, max_tokens: int) -> List[str]:
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)

        if estimate_tokens(" ".join(current)) >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def recursive_split(text: str, max_tokens: int) -> List[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()]

    splitters = [
        split_into_sections,
        split_into_paragraphs,
        split_bullet_lists,
        split_into_sentences,
    ]

    for splitter in splitters:
        parts = splitter(text)

        if len(parts) > 1:
            result = []

            for part in parts:
                if estimate_tokens(part) <= max_tokens:
                    result.append(part.strip())
                else:
                    result.extend(recursive_split(part, max_tokens))

            return [r for r in result if r.strip()]

    return split_by_token_limit(text, max_tokens)


def get_overlap_text(text: str, overlap_tokens: int) -> str:
    words = text.split()

    if len(words) <= overlap_tokens:
        return text.strip()

    return " ".join(words[-overlap_tokens:]).strip()


def merge_with_overlap(
    pieces: List[str],
    target_tokens: int,
    overlap_tokens: int
) -> List[str]:
    chunks = []
    current = []

    for piece in pieces:
        candidate = "\n\n".join(current + [piece])

        if estimate_tokens(candidate) <= target_tokens:
            current.append(piece)
        else:
            if current:
                chunks.append("\n\n".join(current).strip())

            overlap_text = get_overlap_text("\n\n".join(current), overlap_tokens)
            current = []

            if overlap_text:
                current.append(overlap_text)

            current.append(piece)

    if current:
        chunks.append("\n\n".join(current).strip())

    return chunks


def build_parent_child_chunks(
    pages: List[Dict[str, Any]],
    document_id: str,
    child_chunk_tokens: int = 650,
    child_overlap_tokens: int = 100,
    parent_chunk_tokens: int = 2000,
) -> Dict[str, List[Dict[str, Any]]]:

    parent_chunks = []
    child_chunks = []

    parent_counter = 0
    child_counter = 0

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        if not text.strip():
            continue

        parent_pieces = recursive_split(text, parent_chunk_tokens)

        parents = merge_with_overlap(
            parent_pieces,
            target_tokens=parent_chunk_tokens,
            overlap_tokens=150
        )

        for parent_text in parents:
            parent_id = f"{document_id}_parent_{parent_counter}"

            parent_chunks.append({
                "parent_id": parent_id,
                "document_id": document_id,
                "page": page_number,
                "text": parent_text
            })

            child_pieces = recursive_split(parent_text, child_chunk_tokens)

            children = merge_with_overlap(
                child_pieces,
                target_tokens=child_chunk_tokens,
                overlap_tokens=child_overlap_tokens
            )

            for child_text in children:
                child_id = f"{document_id}_child_{child_counter}"

                child_chunks.append({
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "page": page_number,
                    "text": child_text
                })

                child_counter += 1

            parent_counter += 1

    return {
        "parents": parent_chunks,
        "children": child_chunks
    }