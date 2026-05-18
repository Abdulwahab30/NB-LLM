from sentence_transformers import SentenceTransformer


_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    return _model.encode([query], normalize_embeddings=True).tolist()[0]