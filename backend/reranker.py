from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_child_chunks(
    question: str,
    child_results: List[Dict[str, Any]],
    top_n: int = 3
) -> List[Dict[str, Any]]:

    if not child_results:
        return []

    pairs = [
        [question, result["text"]]
        for result in child_results
    ]

    scores = _reranker.predict(pairs)

    reranked = []

    for result, score in zip(child_results, scores):
        item = dict(result)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    filtered = [
    item for item in reranked
    if item["rerank_score"] >= 1.5
]

    if not filtered:
        return reranked[:1]

    return filtered[:top_n]