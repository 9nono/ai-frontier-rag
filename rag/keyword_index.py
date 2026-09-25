import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from functools import lru_cache

from ingest.build_chunks import CHUNK_DIR
from ingest.fetch_arxiv import DATA

KEYWORD_DIR = DATA / "keyword"
# Terms are stored already tokenized and space-separated; keeping '-' and '.' inside a token
# preserves names like gpt-4o and numbers like 1.5 the same way tokenize() does.
FTS_TOKENIZER = "unicode61 tokenchars '-.'"
INSERT_BATCH = 5000

TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
STOPWORDS = set(
    "a an the of to in on for and or is are was were be been by with as at from that this these those it its "
    "we our they their which what how why when who does do did can could would should will than then there "
    "into over under about between via using use used also such not no".split()
)


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def index_path(strategy):
    return KEYWORD_DIR / f"{strategy}.sqlite"


def _source_stamp(strategy):
    stat = (CHUNK_DIR / f"{strategy}.jsonl").stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _is_current(strategy):
    path = index_path(strategy)
    if not path.exists():
        return False
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = 'source'").fetchone()
    return row is not None and row[0] == _source_stamp(strategy)


def build(strategy):
    """Rebuild the index for one strategy from its chunk file; swapped in only once complete."""
    KEYWORD_DIR.mkdir(parents=True, exist_ok=True)
    path, tmp = index_path(strategy), index_path(strategy).with_suffix(".building")
    tmp.unlink(missing_ok=True)
    stamp = _source_stamp(strategy)
    conn = sqlite3.connect(tmp)
    conn.executescript(f"""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE chunks (
            rowid INTEGER PRIMARY KEY, chunk_id TEXT, arxiv_id TEXT, title TEXT, section TEXT, kind TEXT,
            published TEXT, published_ts INTEGER, text TEXT
        );
        CREATE VIRTUAL TABLE chunk_terms USING fts5(terms, content='', tokenize="{FTS_TOKENIZER}");
    """)
    rows, n = [], 0
    with open(CHUNK_DIR / f"{strategy}.jsonl", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            c = json.loads(line)
            ts = int(datetime.fromisoformat(c["published"].replace("Z", "+00:00")).timestamp())
            rows.append((n, c["chunk_id"], c["arxiv_id"], c["title"], c["section"], c["kind"], c["published"], ts,
                         c["text"], " ".join(tokenize(c["embed_text"]))))
            if len(rows) == INSERT_BATCH:
                _insert(conn, rows)
                rows = []
    _insert(conn, rows)
    conn.execute("INSERT INTO chunk_terms(chunk_terms) VALUES ('optimize')")
    conn.execute("INSERT INTO meta VALUES ('source', ?)", (stamp,))
    conn.commit()
    conn.close()
    tmp.replace(path)
    return n


def _insert(conn, rows):
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [r[:9] for r in rows])
    conn.executemany("INSERT INTO chunk_terms(rowid, terms) VALUES (?, ?)", [(r[0], r[9]) for r in rows])


@lru_cache(maxsize=None)
def _connection(strategy):
    if not _is_current(strategy):
        build(strategy)
    return sqlite3.connect(f"file:{index_path(strategy)}?mode=ro", uri=True, check_same_thread=False)


def search(query, strategy, k=10, since_ts=None):
    """BM25 over the chunk terms; returns chunk rows with score = -bm25 (higher is better)."""
    tokens = tokenize(query)
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in tokens)
    cursor = _connection(strategy).execute(
        """SELECT c.chunk_id, c.arxiv_id, c.title, c.section, c.kind, c.published, c.text,
                  bm25(chunk_terms) AS score
           FROM chunk_terms JOIN chunks c ON c.rowid = chunk_terms.rowid
           WHERE chunk_terms MATCH ? AND c.published_ts >= ?
           ORDER BY score LIMIT ?""",
        (match, since_ts or 0, k),
    )
    columns = [d[0] for d in cursor.description]
    return [{**dict(zip(columns, row)), "score": -row[-1]} for row in cursor]


def main():
    parser = argparse.ArgumentParser(description="Build the SQLite FTS5 keyword index")
    parser.add_argument("--strategies", nargs="+", default=["section_header"])
    args = parser.parse_args()
    for strategy in args.strategies:
        started = time.perf_counter()
        n = build(strategy)
        size = index_path(strategy).stat().st_size / 2**20
        print(f"{strategy}: {n} chunks in {time.perf_counter() - started:.1f}s, {size:.0f} MB")


if __name__ == "__main__":
    main()
