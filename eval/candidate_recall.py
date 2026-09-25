import json
from datetime import datetime

from rag.retrieve import HYBRID_POOL, hybrid_search
from rag.translate import needs_translation, to_english

from eval.common import is_relevant, load_questions
from eval.run_retrieval import RESULTS_DIR

SIZES = (10, 20, 30, 50, 100)
STRATEGY = "section_header"


def main():
    """How often a relevant passage is among the hybrid candidates the reranker would receive."""
    questions = load_questions()
    rows = []
    for q in questions:
        query = to_english(q["question"]) if needs_translation(q["question"]) else q["question"]
        found = {}
        for n in SIZES:
            hits = hybrid_search(query, STRATEGY, k=n, pool=max(HYBRID_POOL, n))
            ranks = [i for i, h in enumerate(hits, 1) if is_relevant(h, q)]
            found[n] = ranks[0] if ranks else None
        rows.append({"id": q["id"], "lang": q["lang"], "first_rank_by_size": found})

    print(f"{'lang':4s} {'n':>3s} " + " ".join(f"{'@' + str(n):>6s}" for n in SIZES))
    for lang in ("en", "zh"):
        subset = [r for r in rows if r["lang"] == lang]
        recall = [sum(1 for r in subset if r["first_rank_by_size"][n]) / len(subset) for n in SIZES]
        print(f"{lang:4s} {len(subset):3d} " + " ".join(f"{x:6.2f}" for x in recall))
    for small, large in zip(SIZES, SIZES[1:]):
        gained = [r["id"] for r in rows if r["first_rank_by_size"][large] and not r["first_rank_by_size"][small]]
        print(f"in top {large} but not top {small}: {', '.join(gained) or '-'}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"candidate-recall-{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"strategy": STRATEGY, "sizes": SIZES, "rows": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
