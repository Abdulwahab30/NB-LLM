from huggingface_hub.inference._generated.types import document_question_answering
import os
import json
from typing import Dict, Any, List, Generator
import time

from dotenv import load_dotenv
from openai import OpenAI, APIError

from backend.vector_store import search_child_chunks
from backend.database import (
    get_parent_chunks_by_ids,
    save_chat_history,
    save_semantic_cache,
    query_semantic_cache,
    get_chat_history
)
from backend.embeddings import embed_query
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

MAX_RETRIES = 3
MAX_CONTEXT_TOKENS = 2000


def _unique_parent_ids(search_results: List[Dict[str, Any]]) -> List[str]:
    parent_ids = []
    for result in search_results:
        parent_id = result["metadata"]["parent_id"]
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)
    return parent_ids


def _rewrite_query(question: str, history: List[Dict[str, Any]]) -> str:
    if not history:
        return question

    history_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in history[-2:]])
    prompt = f"""Given the following conversation history and the user's current question, rewrite the current question to be a standalone, fully contextualized query that can be used for searching a document. Do not answer the question, only rewrite it. If it doesn't need rewriting, output it as is.

History:
{history_text}

Current Question: {question}
Standalone Question:"""

    try:
        response = client.chat.completions.create(
            model="z-ai/glm-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return question



def _build_rag_context(document_id: str, question: str):
    """
    Shared retrieval logic used by both streaming and non-streaming paths.
    Returns (prompt, sources, parent_chunks) or a dict error response.
    """
    # 0. Fetch recent chat history
    history = get_chat_history(document_id)
    recent_history = history[-3:] # get last 3 turns

    # 1. Rewrite query for follow-ups
    search_query = _rewrite_query(question, recent_history)

    # 2. Retrieve candidate child chunks
    candidate_child_results = search_child_chunks(
        document_id=document_id,
        question=search_query,
        top_k=10
    )

    # 2. Rerank
    child_results = rerank_child_chunks(
        question=question,
        child_results=candidate_child_results,
        top_n=3
    )

    # 3. If no child chunks found, return early
    if not child_results:
        return None

    # 4. Get parent chunks
    parent_ids = _unique_parent_ids(child_results)
    parent_chunks = get_parent_chunks_by_ids(
        document_id=document_id,
        parent_ids=parent_ids
    )

    if not parent_chunks:
        return None

    # 5. Build LLM prompt (truncate to avoid huge token count)
    current_tokens = 0
    truncated_parents = []
    for parent in parent_chunks:
        text_tokens = len(parent['text']) // 4  # rough token estimate
        if current_tokens + text_tokens > MAX_CONTEXT_TOKENS:
            break
        truncated_parents.append(parent)
        current_tokens += text_tokens

    context = ""
    for parent in truncated_parents:
        context += f"\n[Page {parent['page']}]\n{parent['text']}\n"

    history_context = ""
    if recent_history:
        history_context = "\nRecent Conversation History:\n"
        for item in recent_history:
            history_context += f"User: {item['question']}\nAssistant: {item['answer']}\n\n"

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
- Use the recent conversation history to understand context if the user asks follow-up questions.
{history_context}
PDF Context:
{context}

Question:
{question}
"""

    sources = [
        {"page": parent["page"], "parent_id": parent["parent_id"]}
        for parent in parent_chunks
    ]

    return {
        "prompt": prompt,
        "sources": sources,
        "parent_chunks": parent_chunks,
    }


def _no_info_response():
    return {
        "answer": "I could not find this information in the PDF.",
        "confidence": "low",
        "retrieved_parent_count": 0,
        "sources": []
    }


def answer_question(document_id: str, question: str) -> Dict[str, Any]:
    """
    Main function to answer questions over PDF documents.
    Implements semantic caching, reranking, LLM call with error handling,
    and stores chat history.
    """

    # 1. Check semantic cache first
    cached = query_semantic_cache(document_id, question, threshold=0.85)
    if cached:
        return {
            "answer": cached["answer"],
            "confidence": cached["confidence"],
            "retrieved_parent_count": len(cached["sources"]),
            "sources": cached["sources"]
        }

    # 2. Build RAG context
    rag_context = _build_rag_context(document_id, question)
    if rag_context is None:
        return _no_info_response()

    prompt = rag_context["prompt"]
    sources = rag_context["sources"]
    parent_chunks = rag_context["parent_chunks"]

    # 3. Call LLM with retry and exponential backoff
    answer = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="z-ai/glm-5.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=800
            )
            answer = response.choices[0].message.content
            break
        except APIError as e:
            last_error = e
            if e.status_code == 429:
                # Rate-limited, exponential backoff
                time.sleep(2 ** attempt)
                continue
            elif e.status_code == 402:
                return {
                    "answer": "API quota exceeded. Please upgrade your OpenRouter account or reduce request size.",
                    "confidence": "low",
                    "retrieved_parent_count": len(parent_chunks),
                    "sources": sources
                }
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {
                    "answer": "An error occurred while communicating with the AI model. Please try again later.",
                    "confidence": "low",
                    "retrieved_parent_count": len(parent_chunks),
                    "sources": sources
                }
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "answer": "An unexpected error occurred while processing your request. Please try again later.",
                "confidence": "low",
                "retrieved_parent_count": len(parent_chunks),
                "sources": sources
            }

    if answer is None:
        return {
            "answer": f"Failed after {MAX_RETRIES} retries. Please try again later.",
            "confidence": "low",
            "retrieved_parent_count": len(parent_chunks),
            "sources": sources
        }

    # 4. Save chat and semantic cache
    save_chat_history(
        document_id=document_id,
        question=question,
        answer=answer,
        sources=sources
    )

    save_semantic_cache(
        document_id=document_id,
        question=question,
        answer=answer,
        sources=sources,
        embedding=embed_query(question)
    )

    return {
        "answer": answer,
        "confidence": "high",
        "retrieved_parent_count": len(parent_chunks),
        "sources": sources
    }


def answer_question_stream(document_id: str, question: str) -> Generator[str, None, None]:
    """
    Generator that yields SSE-formatted events as the LLM streams tokens.
    Events:
      data: {"type": "token", "content": "..."}
      data: {"type": "sources", "sources": [...]}
      data: {"type": "done"}
      data: {"type": "error", "content": "..."}
    """

    # 1. Check semantic cache first
    cached = query_semantic_cache(document_id, question, threshold=0.85)
    if cached:
        # Stream the cached answer token-by-token (word-by-word for a nice effect)
        words = cached["answer"].split(" ")
        for word in words:
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': cached['sources'], 'confidence': cached['confidence']})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # 2. Build RAG context
    rag_context = _build_rag_context(document_id, question)
    if rag_context is None:
        no_info = "I could not find this information in the PDF."
        yield f"data: {json.dumps({'type': 'token', 'content': no_info})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'confidence': 'low'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    prompt = rag_context["prompt"]
    sources = rag_context["sources"]
    parent_chunks = rag_context["parent_chunks"]

    # 3. Stream LLM response with retry and exponential backoff
    full_answer = ""
    last_error = None
    success = False

    for attempt in range(MAX_RETRIES):
        try:
            stream = client.chat.completions.create(
                model="z-ai/glm-5.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=800,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            success = True
            break

        except APIError as e:
            last_error = e
            if e.status_code == 429:
                time.sleep(2 ** attempt)
                full_answer = ""
                continue
            elif e.status_code == 402:
                yield f"data: {json.dumps({'type': 'error', 'content': 'API quota exceeded. Please upgrade your OpenRouter account.'})}\n\n"
                return
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    full_answer = ""
                    continue
                yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred while communicating with the AI model. Please try again later.'})}\n\n"
                return

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                full_answer = ""
                continue
            yield f"data: {json.dumps({'type': 'error', 'content': 'An unexpected error occurred. Please try again later.'})}\n\n"
            return

    if not success:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Failed after {MAX_RETRIES} retries. Please try again later.'})}\n\n"
        return

    # 4. Send sources
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'confidence': 'high'})}\n\n"

    # 5. Send done signal
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # 6. Save chat history and semantic cache (after stream completes)
    if full_answer:
        save_chat_history(
            document_id=document_id,
            question=question,
            answer=full_answer,
            sources=sources
        )

        save_semantic_cache(
            document_id=document_id,
            question=question,
            answer=full_answer,
            sources=sources,
            embedding=embed_query(question)
        )