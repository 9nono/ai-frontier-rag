import json
import sqlite3
from collections import Counter

from ingest.chunk import STRATEGIES, chunk_paper
from ingest.fetch_arxiv import DATA, DB_PATH, ROOT
from ingest.parse import parse

CHUNK_DIR = DATA / "chunks"


def load_papers(conn):
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """SELECT p.arxiv_id, p.version, p.title, p.summary, p.published, f.format, f.path
           FROM papers p JOIN fulltext f ON f.arxiv_id = p.arxiv_id AND f.version = p.version
           WHERE f.status = 'ok' ORDER BY p.published DESC"""
    ).fetchall()


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    papers = load_papers(conn)
    outputs = {s: open(CHUNK_DIR / f"{s}.jsonl", "w", encoding="utf-8") for s in STRATEGIES}
    stats = {s: Counter() for s in STRATEGIES}
    failed = []
    for paper in papers:
        try:
            blocks = parse(ROOT / paper["path"], paper["format"]).blocks
        except Exception as exc:
            failed.append((paper["arxiv_id"], repr(exc)))
            continue
        for strategy in STRATEGIES:
            for record in chunk_paper(dict(paper), blocks, strategy):
                outputs[strategy].write(json.dumps(record, ensure_ascii=False) + "\n")
                stats[strategy]["chunks"] += 1
                stats[strategy]["words"] += record["n_words"]
                stats[strategy][f"kind:{record['kind']}"] += 1
    for f in outputs.values():
        f.close()
    print(f"papers parsed: {len(papers) - len(failed)}/{len(papers)}")
    for arxiv_id, error in failed:
        print(f"  parse failed: {arxiv_id} {error}")
    for strategy, c in stats.items():
        avg = c["words"] / c["chunks"] if c["chunks"] else 0
        kinds = {k[5:]: v for k, v in c.items() if k.startswith("kind:")}
        print(f"{strategy:15s} chunks={c['chunks']:6d} avg_words={avg:6.1f} {kinds}")


if __name__ == "__main__":
    main()
