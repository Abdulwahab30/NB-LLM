from huggingface_hub.inference._generated.types import document_question_answering
import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI, APIError

from backend.vector_store import search_child_chunks
from backend.database import get_parent_chunks_by_ids, save_chat_history
from backend.reranker import rerank_child_chunks


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "NB_LLM"
    }
)


def _unique_parent_ids(search_results: List[Dict[str, Any]]) -> List[str]:
    parent_ids = []

    for result in search_results:
        parent_id = result["metadata"]["parent_id"]

        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

    return parent_ids


def answer_question(document_id: str, question: str) -> Dict[str, Any]:
    candidate_child_results = search_child_chunks(
        document_id=document_id,
        question=question,
        top_k=10
    )

    child_results = rerank_child_chunks(
        question=question,
        child_results=candidate_child_results,
        top_n=3
    
)

    

    parent_ids = _unique_parent_ids(child_results)

    parent_chunks = get_parent_chunks_by_ids(
        document_id=document_id,
        parent_ids=parent_ids
    )

    if not parent_chunks:
        return {
            "answer": "I could not find this information in the PDF.",
            "confidence": "low",
            "retrieved_parent_count": 0,
            "sources": []
        }

    context = ""

    for parent in parent_chunks:
        context += f"\n[Page {parent['page']}]\n"
        context += parent["text"]
        context += "\n"

    prompt = f"""
You are a PDF question-answering assistant.

Answer only using the PDF context below.
If the answer is not available in the context, say:
"I could not find this information in the PDF."

Rules:
- Do not use outside knowledge.
- Do not guess.
- Mention page numbers when possible.
- Keep the answer clear and direct.

PDF Context:
{context}

Question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content

        sources = [
            {
                "page": parent["page"],
        "parent_id": parent["parent_id"]
    }
    for parent in parent_chunks
]

        save_chat_history(
            document_id=document_id,
            question=question,
            answer=answer,
            sources=sources
        )

        return {
            "answer": answer,
            "confidence": "high",
            "retrieved_parent_count": len(parent_chunks),
            "sources": sources
        }

    except APIError as e:
        error_msg = "We are currently experiencing high traffic and the AI model is rate-limited. Please wait a moment and try again." if e.status_code == 429 else "An error occurred while communicating with the AI model. Please try again later."
        return {
            "answer": error_msg,
            "sources": []
        }
    except Exception as e:
        return {
            "answer": "An unexpected error occurred while processing your request. Please try again later.",
            "sources": []
        }