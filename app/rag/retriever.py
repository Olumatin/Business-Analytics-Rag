from app import config
from app.rag import ingest


def search(query, top_k=None):
    top_k = top_k or config.TOP_K

    if ingest._index is None or ingest._index.ntotal == 0:
        return []

    query_vec = ingest._embed([query])
    scores, indices = ingest._index.search(query_vec, top_k)

    results = []
    # Flat 1D loop mapping protects the server from matrix indexing failures
    for score_row, idx_row in zip(scores, indices):
        for score, idx in zip(score_row, idx_row):
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
