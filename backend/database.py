import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = "data/app.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        status TEXT NOT NULL,
        pages INTEGER DEFAULT 0,
        parent_chunks INTEGER DEFAULT 0,
        child_chunks INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS parent_chunks (
        parent_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        page INTEGER NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(document_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        sources TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def create_or_update_document(
    document_id: str,
    filename: str,
    file_path: str,
    status: str,
    pages: int = 0,
    parent_chunks: int = 0,
    child_chunks: int = 0,
    error: Optional[str] = None
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO documents (
        document_id, filename, file_path, status, pages,
        parent_chunks, child_chunks, created_at, updated_at, error
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(document_id) DO UPDATE SET
        filename=excluded.filename,
        file_path=excluded.file_path,
        status=excluded.status,
        pages=excluded.pages,
        parent_chunks=excluded.parent_chunks,
        child_chunks=excluded.child_chunks,
        updated_at=excluded.updated_at,
        error=excluded.error
    """, (
        document_id,
        filename,
        file_path,
        status,
        pages,
        parent_chunks,
        child_chunks,
        now_iso(),
        now_iso(),
        error
    ))

    conn.commit()
    conn.close()


def list_documents() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT * FROM documents
    ORDER BY created_at DESC
    """)

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT * FROM documents WHERE document_id = ?
    """, (document_id,))

    row = cur.fetchone()
    conn.close()

    return dict(row) if row else None


def delete_document_records(document_id: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM chat_history WHERE document_id = ?", (document_id,))
    cur.execute("DELETE FROM parent_chunks WHERE document_id = ?", (document_id,))
    cur.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    conn.commit()
    conn.close()


def save_parent_chunks(parent_chunks: List[Dict[str, Any]]):
    conn = get_connection()
    cur = conn.cursor()

    for chunk in parent_chunks:
        cur.execute("""
        INSERT OR REPLACE INTO parent_chunks (
            parent_id, document_id, page, text
        )
        VALUES (?, ?, ?, ?)
        """, (
            chunk["parent_id"],
            chunk["document_id"],
            chunk["page"],
            chunk["text"]
        ))

    conn.commit()
    conn.close()


def get_parent_chunks_by_ids(
    document_id: str,
    parent_ids: List[str]
) -> List[Dict[str, Any]]:

    if not parent_ids:
        return []

    placeholders = ",".join(["?"] * len(parent_ids))

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(f"""
    SELECT * FROM parent_chunks
    WHERE document_id = ?
    AND parent_id IN ({placeholders})
    """, [document_id] + parent_ids)

    rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    order = {pid: i for i, pid in enumerate(parent_ids)}
    rows.sort(key=lambda x: order.get(x["parent_id"], 999))

    return rows


def save_chat_history(
    document_id: str,
    question: str,
    answer: str,
    sources: List[Dict[str, Any]]
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO chat_history (
        document_id, question, answer, sources, created_at
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        document_id,
        question,
        answer,
        json.dumps(sources),
        now_iso()
    ))

    conn.commit()
    conn.close()


def get_chat_history(document_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT * FROM chat_history
    WHERE document_id = ?
    ORDER BY created_at ASC
    """, (document_id,))

    rows = []

    for row in cur.fetchall():
        item = dict(row)
        item["sources"] = json.loads(item["sources"])
        rows.append(item)

    conn.close()
    return rows