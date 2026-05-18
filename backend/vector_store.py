import chromadb
from typing import List, Dict, Any

from backend.embeddings import embed_texts, embed_query


client = chromadb.PersistentClient(path="data/chroma_db")

child_collection = client.get_or_create_collection(
    name="pdf_child_chunks"
)


def add_child_chunks(child_chunks: List[Dict[str, Any]]) -> None:
    if not child_chunks:
        return

    texts = [chunk["text"] for chunk in child_chunks]
    embeddings = embed_texts(texts)

    ids = [chunk["child_id"] for chunk in child_chunks]

    metadatas = [
        {
            "child_id": chunk["child_id"],
            "parent_id": chunk["parent_id"],
            "document_id": chunk["document_id"],
            "page": chunk["page"]
        }
        for chunk in child_chunks
    ]

    child_collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search_child_chunks(
    document_id: str,
    question: str,
    top_k: int = 8
) -> List[Dict[str, Any]]:

    query_embedding = embed_query(question)

    results = child_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_id": document_id}
    )

    output = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        output.append({
            "text": doc,
            "metadata": meta,
            "distance": distance
        })

    return output


def delete_document_vectors(document_id: str) -> None:
    child_collection.delete(
        where={"document_id": document_id}
    )