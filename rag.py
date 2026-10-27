
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import get_embeddings


def search(question, chunks, embeddings, top_k=5):
    if not chunks or embeddings is None or len(embeddings) == 0:
        return []

    query_vector = get_embeddings([question])

    scores = cosine_similarity(
        query_vector, embeddings
    )[0]

    indices = scores.argsort()[::-1][:top_k]

    return [
        chunks[i]
        for i in indices
        if scores[i] > 0
    ]

