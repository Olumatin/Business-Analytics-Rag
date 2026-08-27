from app import config
from app.rag import ingest


def search(query, top_k=None):
    top_k = top_k or config.TOP_K

    if ingest._index is None or ingest._index.ntotal == 0:
        return []

    query_vec = ingest._embed([query])
    scores, indices = ingest._index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = ingest._metadata[idx]
        results.append({
            "text": meta["text"],
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "score": float(score),
        })
    return results
