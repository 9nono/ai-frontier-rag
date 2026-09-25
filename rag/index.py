import argparse
import hashlib
import json
import time
from datetime import datetime

import chromadb
from fastembed import TextEmbedding

from ingest.build_chunks import CHUNK_DIR
from ingest.chunk import STRATEGIES
from ingest.fetch_arxiv import DATA

CHROMA_DIR = DATA / "chroma"
BATCH = 128

EMBED_MODELS = {
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "jina-zh": {
        "name": "jinaai/jina-embeddings-v2-base-zh",
        "query_prefix": "",
    },
}

_models = {}


def embedder(model_key):
    if model_key not in _models:
        _models[model_key] = TextEmbedding(EMBED_MODELS[model_key]["name"])
    return _models[model_key]


def embed_queries(model_key, queries):
    prefix = EMBED_MODELS[model_key]["query_prefix"]
    return [v.tolist() for v in embedder(model_key).embed([prefix + q for q in queries])]


def client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def collection_name(strategy, model_key):
    return f"{strategy}__{model_key}"


def get_collection(strategy, model_key):
    return client().get_or_create_collection(
        collection_name(strategy, model_key), metadata={"hnsw:space": "cosine"}
    )


def load_chunks(strategy):
    with open(CHUNK_DIR / f"{strategy}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _metadata(chunk):
    return {
        "arxiv_id": chunk["arxiv_id"],
        "version": chunk["version"],
        "title": chunk["title"],
        "section": chunk["section"],
        "kind": chunk["kind"],
        "published": chunk["published"],
        "published_ts": int(datetime.fromisoformat(chunk["published"].replace("Z", "+00:00")).timestamp()),
        "embed_hash": chunk["embed_hash"],
    }


def _existing_hashes(collection):
    hashes, offset = {}, 0
    while True:
        page = collection.get(include=["metadatas"], limit=5000, offset=offset)
        if not page["ids"]:
            return hashes
        hashes.update({i: m.get("embed_hash") for i, m in zip(page["ids"], page["metadatas"])})
        offset += len(page["ids"])


def build(strategy, model_key):
    collection = get_collection(strategy, model_key)
    chunks = load_chunks(strategy)
    for c in chunks:
        c["embed_hash"] = hashlib.sha1(c["embed_text"].encode("utf-8")).hexdigest()
    existing = _existing_hashes(collection)
    wanted = {c["chunk_id"] for c in chunks}
    todo = [c for c in chunks if existing.get(c["chunk_id"]) != c["embed_hash"]]
    stale = [i for i in existing if i not in wanted]
    for start in range(0, len(stale), 5000):
        collection.delete(ids=stale[start:start + 5000])
    print(f"{collection_name(strategy, model_key)}: {len(chunks)} chunks | "
          f"new/changed {len(todo)} | unchanged {len(chunks) - len(todo)} | deleted {len(stale)}")
    model = embedder(model_key)
    started = time.monotonic()
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        vectors = [v.tolist() for v in model.embed([c["embed_text"] for c in batch], batch_size=BATCH)]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[_metadata(c) for c in batch],
        )
        done = start + len(batch)
        if done % (BATCH * 20) == 0 or done == len(todo):
            rate = done / (time.monotonic() - started)
            print(f"  {done}/{len(todo)} ({rate:.0f} chunks/s)")
    return collection.count()


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into Chroma")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--model", default="bge-small", choices=list(EMBED_MODELS))
    args = parser.parse_args()
    for strategy in args.strategies:
        print(f"-> {collection_name(strategy, args.model)} now has {build(strategy, args.model)} vectors")


if __name__ == "__main__":
    main()
