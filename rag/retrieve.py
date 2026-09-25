from datetime import datetime, timezone
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag import keyword_index
from rag.index import DEVICE, MAX_SEQ_TOKENS, embed_queries, get_collection

RRF_K = 60
HYBRID_POOL = 50
RERANK_CANDIDATES = 30
RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge": "BAAI/bge-reranker-base",
}


def _since_ts(since):
    if since is None:
        return None
    return int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp())


@lru_cache(maxsize=None)
def _reranker(name):
    return CrossEncoder(RERANKERS[name], device=DEVICE, max_length=MAX_SEQ_TOKENS)


def _hit(chunk_id, meta, text, score, source):
    return {
        "chunk_id": chunk_id,
        "arxiv_id": meta["arxiv_id"],
        "title": meta["title"],
        "section": meta["section"],
        "kind": meta["kind"],
        "published": meta["published"],
        "text": text,
        "score": score,
        "source": source,
    }


def vector_search(query, strategy, model_key="bge-small", k=10, since=None):
    where = {"published_ts": {"$gte": _since_ts(since)}} if since else None
    result = get_collection(strategy, model_key).query(
        query_embeddings=embed_queries(model_key, [query]), n_results=k, where=where,
        include=["metadatas", "documents", "distances"],
    )
    return [
        _hit(i, m, d, 1 - dist, "vector")
        for i, m, d, dist in zip(result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0])
    ]


def bm25_search(query, strategy, k=10, since=None):
    return [_hit(r["chunk_id"], r, r["text"], r["score"], "bm25")
            for r in keyword_index.search(query, strategy, k, _since_ts(since))]


def hybrid_search(query, strategy, model_key="bge-small", k=10, since=None, pool=HYBRID_POOL):
    fused = {}
    for results in (vector_search(query, strategy, model_key, pool, since), bm25_search(query, strategy, pool, since)):
        for rank, hit in enumerate(results, 1):
            entry = fused.setdefault(hit["chunk_id"], {**hit, "score": 0.0, "source": "hybrid"})
            entry["score"] += 1 / (RRF_K + rank)
    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:k]


def rerank(query, hits, reranker="minilm", k=10):
    if not hits:
        return []
    passages = [f"{h['title']}\nSection: {h['section']}\n{h['text']}" for h in hits]
    scores = _reranker(reranker).predict([(query, p) for p in passages])
    ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)[:k]
    return [{**h, "score": float(s), "source": f"{h['source']}+rerank"} for h, s in ranked]


def search(query, method="hybrid_rerank", strategy="section_header", model_key="bge-small", k=10, since=None,
           reranker="minilm", translate=False, rerank_candidates=RERANK_CANDIDATES):
    if translate:
        from rag.translate import needs_translation, to_english

        if needs_translation(query):
            query = to_english(query)
    if method == "vector":
        return vector_search(query, strategy, model_key, k, since)
    if method == "bm25":
        return bm25_search(query, strategy, k, since)
    if method == "hybrid":
        return hybrid_search(query, strategy, model_key, k, since)
    if method == "hybrid_rerank":
        candidates = hybrid_search(query, strategy, model_key, rerank_candidates, since,
                                   pool=max(HYBRID_POOL, rerank_candidates))
        return rerank(query, candidates, reranker, k)
    raise ValueError(f"unknown method {method}")
