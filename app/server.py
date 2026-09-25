import json
import os
import sqlite3
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from ingest.fetch_arxiv import DB_PATH
from rag.retrieve import search

APP_DIR = Path(__file__).resolve().parent
FEATURED_PATH = APP_DIR / "featured_answers.json"
LIVE_ANSWERS = os.environ.get("ENABLE_LIVE_ANSWERS") == "1"
DEFAULT_METHOD = os.environ.get("RAG_METHOD", "hybrid_rerank")
DEFAULT_STRATEGY = os.environ.get("RAG_STRATEGY", "section_header")
DEFAULT_MODEL = os.environ.get("RAG_EMBED_MODEL", "bge-small")
METHODS = {"vector", "bm25", "hybrid", "hybrid_rerank"}
MAX_QUERY_CHARS = 500

app = Flask(__name__, static_folder=str(APP_DIR / "static"))


def _hit_view(h):
    return {k: h[k] for k in ("arxiv_id", "title", "section", "kind", "published", "text", "score")}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/stats")
def stats():
    with sqlite3.connect(DB_PATH) as conn:
        n, first, last = conn.execute("SELECT count(*), min(published), max(published) FROM papers").fetchone()
    return jsonify({"papers": n, "from": first[:10], "to": last[:10], "live_answers": LIVE_ANSWERS,
                    "method": DEFAULT_METHOD, "strategy": DEFAULT_STRATEGY, "embed_model": DEFAULT_MODEL})


@app.get("/api/search")
def api_search():
    query = (request.args.get("q") or "").strip()[:MAX_QUERY_CHARS]
    if not query:
        return jsonify({"hits": []})
    method = request.args.get("method", DEFAULT_METHOD)
    if method not in METHODS:
        abort(400, "unknown method")
    since = request.args.get("since") or None
    k = min(max(int(request.args.get("k", 8)), 1), 20)
    hits = search(query, method=method, strategy=DEFAULT_STRATEGY, model_key=DEFAULT_MODEL, k=k, since=since,
                  translate=True)
    return jsonify({"hits": [_hit_view(h) for h in hits]})


@app.get("/api/featured")
def featured():
    if not FEATURED_PATH.exists():
        return jsonify([])
    return jsonify(json.loads(FEATURED_PATH.read_text(encoding="utf-8")))


@app.post("/api/answer")
def api_answer():
    if not LIVE_ANSWERS:
        abort(403, "live answers are disabled on this deployment")
    from rag.generate import answer

    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()[:MAX_QUERY_CHARS]
    if not question:
        abort(400, "question is required")
    result = answer(question, method=DEFAULT_METHOD, strategy=DEFAULT_STRATEGY, model_key=DEFAULT_MODEL,
                    translate=True)
    result["hits"] = [_hit_view(h) for h in result["hits"]]
    return jsonify(result)


def warm_up():
    search("warm up", method=DEFAULT_METHOD, strategy=DEFAULT_STRATEGY, model_key=DEFAULT_MODEL, k=1, translate=True)
    search("预热", method=DEFAULT_METHOD, strategy=DEFAULT_STRATEGY, model_key=DEFAULT_MODEL, k=1, translate=True)


if __name__ == "__main__":
    warm_up()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5077)), debug=False)
