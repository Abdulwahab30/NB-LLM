import json
import os
from typing import List, Dict, Any


PARENT_STORE_DIR = "data/parent_chunks"

os.makedirs(PARENT_STORE_DIR, exist_ok=True)


def _get_parent_file(document_id: str) -> str:
    return os.path.join(PARENT_STORE_DIR, f"{document_id}_parents.json")


def save_parent_chunks(
    document_id: str,
    parent_chunks: List[Dict[str, Any]]
) -> None:
    file_path = _get_parent_file(document_id)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(parent_chunks, f, ensure_ascii=False, indent=2)


def load_parent_chunks(document_id: str) -> List[Dict[str, Any]]:
    file_path = _get_parent_file(document_id)

    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_parent_chunks_by_ids(
    document_id: str,
    parent_ids: List[str]
) -> List[Dict[str, Any]]:

    all_parents = load_parent_chunks(document_id)
    parent_id_set = set(parent_ids)

    return [
        parent
        for parent in all_parents
        if parent["parent_id"] in parent_id_set
    ]