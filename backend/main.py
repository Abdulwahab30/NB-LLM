import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.pdf_reader import extract_pdf_text
from backend.preprocessing import preprocess_pages
from backend.chunker import build_parent_child_chunks
from backend.vector_store import add_child_chunks, search_child_chunks, delete_document_vectors
from backend.reranker import rerank_child_chunks
from backend.rag_service import answer_question
from backend.database import (
    init_db,
    create_or_update_document,
    list_documents,
    get_document,
    delete_document_records,
    save_parent_chunks,
    get_parent_chunks_by_ids,
    get_chat_history
)


app = FastAPI()

PDF_DIR = "backend/storage/pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

init_db()


class AskRequest(BaseModel):
    document_id: str
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    document_id = (
        file.filename
        .replace(".pdf", "")
        .replace(".PDF", "")
        .replace(" ", "_")
    )

    file_path = os.path.join(PDF_DIR, file.filename)

    create_or_update_document(
        document_id=document_id,
        filename=file.filename,
        file_path=file_path,
        status="processing"
    )

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pages = extract_pdf_text(file_path)
        pages = preprocess_pages(pages)

        chunk_data = build_parent_child_chunks(
            pages=pages,
            document_id=document_id,
            child_chunk_tokens=650,
            child_overlap_tokens=100,
            parent_chunk_tokens=2000
        )

        parent_chunks = chunk_data["parents"]
        child_chunks = chunk_data["children"]

        save_parent_chunks(parent_chunks)
        add_child_chunks(child_chunks)

        create_or_update_document(
            document_id=document_id,
            filename=file.filename,
            file_path=file_path,
            status="ready",
            pages=len(pages),
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks)
        )

        return {
            "message": "PDF processed successfully",
            "document_id": document_id,
            "pages": len(pages),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks)
        }

    except Exception as e:
        create_or_update_document(
            document_id=document_id,
            filename=file.filename,
            file_path=file_path,
            status="failed",
            error=str(e)
        )

        return {
            "error": str(e),
            "document_id": document_id
        }


@app.post("/ask")
def ask(request: AskRequest):
    return answer_question(
        document_id=request.document_id,
        question=request.question
    )


@app.post("/debug-retrieval")
def debug_retrieval(request: AskRequest):
    candidate_child_results = search_child_chunks(
        document_id=request.document_id,
        question=request.question,
        top_k=10
    )

    reranked_child_results = rerank_child_chunks(
        question=request.question,
        child_results=candidate_child_results,
        top_n=3
    )

    parent_ids = []

    for result in reranked_child_results:
        parent_id = result["metadata"]["parent_id"]

        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

    parent_chunks = get_parent_chunks_by_ids(
        document_id=request.document_id,
        parent_ids=parent_ids
    )

    return {
        "document_id": request.document_id,
        "question": request.question,
        "candidate_results_before_reranking": len(candidate_child_results),
        "reranked_results_after_reranking": len(reranked_child_results),
        "parent_ids": parent_ids,
        "reranked_child_results": reranked_child_results,
        "parent_chunks": parent_chunks
    }


@app.get("/list-documents")
def list_all_documents():
    return {
        "documents": list_documents()
    }


@app.get("/get-document-status/{document_id}")
def get_document_status(document_id: str):
    document = get_document(document_id)

    if not document:
        return {
            "error": "Document not found",
            "document_id": document_id
        }

    return document


@app.delete("/delete-document/{document_id}")
def delete_document(document_id: str):
    document = get_document(document_id)

    if not document:
        return {
            "error": "Document not found",
            "document_id": document_id
        }

    try:
        delete_document_vectors(document_id)
        delete_document_records(document_id)

        file_path = document.get("file_path")

        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }

    except Exception as e:
        return {
            "error": str(e),
            "document_id": document_id
        }


@app.post("/reindex-document/{document_id}")
def reindex_document(document_id: str):
    document = get_document(document_id)

    if not document:
        return {
            "error": "Document not found",
            "document_id": document_id
        }

    file_path = document["file_path"]

    if not os.path.exists(file_path):
        return {
            "error": "Original PDF file not found",
            "document_id": document_id
        }

    try:
        create_or_update_document(
            document_id=document_id,
            filename=document["filename"],
            file_path=file_path,
            status="processing"
        )

        delete_document_vectors(document_id)

        pages = extract_pdf_text(file_path)
        pages = preprocess_pages(pages)

        chunk_data = build_parent_child_chunks(
            pages=pages,
            document_id=document_id,
            child_chunk_tokens=650,
            child_overlap_tokens=100,
            parent_chunk_tokens=2000
        )

        parent_chunks = chunk_data["parents"]
        child_chunks = chunk_data["children"]

        save_parent_chunks(parent_chunks)
        add_child_chunks(child_chunks)

        create_or_update_document(
            document_id=document_id,
            filename=document["filename"],
            file_path=file_path,
            status="ready",
            pages=len(pages),
            parent_chunks=len(parent_chunks),
            child_chunks=len(child_chunks)
        )

        return {
            "message": "Document reindexed successfully",
            "document_id": document_id,
            "pages": len(pages),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks)
        }

    except Exception as e:
        create_or_update_document(
            document_id=document_id,
            filename=document["filename"],
            file_path=file_path,
            status="failed",
            error=str(e)
        )

        return {
            "error": str(e),
            "document_id": document_id
        }


@app.get("/chat-history/{document_id}")
def chat_history(document_id: str):
    return {
        "document_id": document_id,
        "history": get_chat_history(document_id)
    }


app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")