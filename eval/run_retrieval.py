import argparse
import json
import time
from datetime import datetime

from rag.retrieve import search

from eval.common import EVAL_DIR, is_relevant, load_questions

RESULTS_DIR = EVAL_DIR / "results"
K = 10

CONFIGS = {
    "bm25/fixed": dict(method="bm25", strategy="fixed"),
    "bm25/section": dict(method="bm25", strategy="section"),
    "bm25/section_header": dict(method="bm25", strategy="section_header"),
    "vector/fixed": dict(method="vector", strategy="fixed"),
    "vector/section": dict(method="vector", strategy="section"),
    "vector/section_header": dict(method="vector", strategy="section_header"),
    "hybrid/section_header": dict(method="hybrid", strategy="section_header"),
    "hybrid+rerank(minilm)/section_header": dict(method="hybrid_rerank", strategy="section_header", reranker="minilm"),
    "hybrid+rerank(bge)/section_header": dict(method="hybrid_rerank", strategy="section_header", reranker="bge"),
}


def evaluate(config, questions):
    rows, latencies = [], []
    for q in questions:
        started = time.perf_counter()
        hits = search(q["question"], k=K, **config)
        latencies.append(time.perf_counter() - started)
        ranks = [i for i, h in enumerate(hits, 1) if is_relevant(h, q)]
        paper_ranks = [i for i, h in enumerate(hits, 1) if h["arxiv_id"] == q["arxiv_id"]]
        rows.append({
            "id": q["id"], "lang": q["lang"], "type": q["type"],
            "first_rank": ranks[0] if ranks else None,
            "first_paper_rank": paper_ranks[0] if paper_ranks else None,
            "top_hits": [h["chunk_id"] for h in hits[:3]],
        })
    return rows, latencies


def summarize(rows):
    n = len(rows)
    if not n:
        return {}
    hit = lambda k: sum(1 for r in rows if r["first_rank"] and r["first_rank"] <= k) / n
    return {
        "n": n,
        "hit@1": hit(1),
        "hit@5": hit(5),
        "hit@10": hit(10),
        "paper@5": sum(1 for r in rows if r["first_paper_rank"] and r["first_paper_rank"] <= 5) / n,
        "mrr@10": sum(1 / r["first_rank"] for r in rows if r["first_rank"]) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = parser.parse_args()
    questions = load_questions()
    RESULTS_DIR.mkdir(exist_ok=True)
    report = {}
    header = f"{'config':40s} {'lang':4s} {'n':>3s} {'hit@1':>6s} {'hit@5':>6s} {'hit@10':>6s} {'paper@5':>7s} {'mrr':>6s} {'ms/q':>6s}"
    print(header)
    for name in args.configs:
        rows, latencies = evaluate(CONFIGS[name], questions)
        ms = 1000 * sum(latencies) / len(latencies)
        report[name] = {"rows": rows, "ms_per_query": ms}
        for lang in ("en", "zh"):
            s = summarize([r for r in rows if r["lang"] == lang])
            report[name][lang] = s
            print(f"{name:40s} {lang:4s} {s['n']:3d} {s['hit@1']:6.2f} {s['hit@5']:6.2f} {s['hit@10']:6.2f} "
                  f"{s['paper@5']:7.2f} {s['mrr@10']:6.3f} {ms:6.0f}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"retrieval-{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
