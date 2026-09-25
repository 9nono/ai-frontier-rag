from ingest.chunk import STRATEGIES
from rag.index import load_chunks

from eval.common import is_relevant, load_questions


def main():
    questions = load_questions()
    ok = True
    for strategy in STRATEGIES:
        chunks = load_chunks(strategy)
        by_paper = {}
        for c in chunks:
            by_paper.setdefault(c["arxiv_id"], []).append(c)
        missing = [q["id"] for q in questions if not any(is_relevant(c, q) for c in by_paper.get(q["arxiv_id"], []))]
        counts = [sum(is_relevant(c, q) for c in by_paper.get(q["arxiv_id"], [])) for q in questions]
        print(f"{strategy:15s} questions={len(questions)} evidence found={len(questions) - len(missing)} "
              f"avg relevant chunks/question={sum(counts) / len(counts):.1f} missing={missing}")
        ok &= not missing
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
