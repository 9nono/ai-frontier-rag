import argparse
import hashlib
import json
import sqlite3
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DB_PATH = DATA / "papers.db"

API_URL = "https://export.arxiv.org/api/query"
HTML_URL = "https://arxiv.org/html/{id}v{version}"
PDF_URL = "https://arxiv.org/pdf/{id}v{version}"
REQUEST_GAP_SECONDS = 3.1
USER_AGENT = "ai-frontier-rag/0.1 (personal research project)"

NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    authors TEXT NOT NULL,
    categories TEXT NOT NULL,
    primary_category TEXT,
    published TEXT NOT NULL,
    updated TEXT NOT NULL,
    comment TEXT,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fulltext (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
    version INTEGER NOT NULL,
    format TEXT,
    path TEXT,
    sha256 TEXT,
    status TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    DATA.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


class Throttle:
    def __init__(self, gap):
        self.gap = gap
        self.last = 0.0

    def wait(self):
        delay = self.gap - (time.monotonic() - self.last)
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def split_id(entry_id):
    tail = entry_id.rsplit("/abs/", 1)[1]
    base, _, version = tail.rpartition("v")
    return base, int(version)


def parse_entry(entry):
    arxiv_id, version = split_id(entry.findtext("atom:id", namespaces=NS))
    text = lambda path: " ".join((entry.findtext(path, default="", namespaces=NS) or "").split())
    primary = entry.find("arxiv:primary_category", NS)
    return {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": text("atom:title"),
        "summary": text("atom:summary"),
        "authors": json.dumps([a.findtext("atom:name", namespaces=NS) for a in entry.findall("atom:author", NS)]),
        "categories": json.dumps([c.get("term") for c in entry.findall("atom:category", NS)]),
        "primary_category": primary.get("term") if primary is not None else None,
        "published": entry.findtext("atom:published", namespaces=NS),
        "updated": entry.findtext("atom:updated", namespaces=NS),
        "comment": text("arxiv:comment") or None,
    }


def upsert_paper(conn, paper):
    row = conn.execute("SELECT version FROM papers WHERE arxiv_id = ?", (paper["arxiv_id"],)).fetchone()
    if row and row[0] >= paper["version"]:
        return "unchanged"
    conn.execute(
        """INSERT INTO papers (arxiv_id, version, title, summary, authors, categories, primary_category,
                               published, updated, comment, fetched_at)
           VALUES (:arxiv_id, :version, :title, :summary, :authors, :categories, :primary_category,
                   :published, :updated, :comment, :fetched_at)
           ON CONFLICT(arxiv_id) DO UPDATE SET
               version = excluded.version, title = excluded.title, summary = excluded.summary,
               authors = excluded.authors, categories = excluded.categories,
               primary_category = excluded.primary_category, updated = excluded.updated,
               comment = excluded.comment, fetched_at = excluded.fetched_at""",
        {**paper, "fetched_at": now()},
    )
    return "updated" if row else "inserted"


def _fetch_page(session, throttle, query, start, size, attempts=4):
    for attempt in range(attempts):
        throttle.wait()
        resp = session.get(API_URL, params={
            "search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
            "start": start, "max_results": size,
        }, timeout=60)
        if resp.status_code == 200:
            entries = ET.fromstring(resp.content).findall("atom:entry", NS)
            if entries:
                return entries
        time.sleep(5 * (attempt + 1))
    return []


def fetch_metadata(categories, limit, page_size, days=None):
    conn = connect()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    throttle = Throttle(REQUEST_GAP_SECONDS)
    query = " OR ".join(f"cat:{c}" for c in categories)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days else None
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    start = 0
    while start < limit:
        entries = _fetch_page(session, throttle, query, start, min(page_size, limit - start))
        if not entries:
            break
        papers = [parse_entry(e) for e in entries]
        in_window = [p for p in papers if not cutoff or p["published"] >= cutoff]
        for paper in in_window:
            counts[upsert_paper(conn, paper)] += 1
        conn.commit()
        start += len(entries)
        print(f"metadata: {start} fetched, oldest {papers[-1]['published'][:10]} {counts}", flush=True)
        if len(in_window) < len(papers):
            break
    conn.close()
    return counts


def save_raw(arxiv_id, version, fmt, content):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{arxiv_id}v{version}.{fmt}"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def fetch_fulltext(limit=None):
    conn = connect()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    throttle = Throttle(REQUEST_GAP_SECONDS)
    todo = conn.execute(
        """SELECT p.arxiv_id, p.version FROM papers p
           LEFT JOIN fulltext f ON f.arxiv_id = p.arxiv_id
           WHERE f.arxiv_id IS NULL OR f.version < p.version OR f.status = 'failed'
           ORDER BY p.published DESC"""
    ).fetchall()
    if limit:
        todo = todo[:limit]
    stats = {"html": 0, "pdf": 0, "failed": 0}
    for i, (arxiv_id, version) in enumerate(todo, 1):
        record = {"arxiv_id": arxiv_id, "version": version, "format": None, "path": None, "sha256": None,
                  "status": "failed", "fetched_at": now()}
        for fmt, url in (("html", HTML_URL), ("pdf", PDF_URL)):
            throttle.wait()
            try:
                resp = session.get(url.format(id=arxiv_id, version=version), timeout=90)
            except requests.RequestException:
                continue
            if resp.status_code == 200 and resp.content:
                path, digest = save_raw(arxiv_id, version, fmt, resp.content)
                record.update(format=fmt, path=str(path.relative_to(ROOT)), sha256=digest, status="ok")
                break
        stats[record["format"] or "failed"] += 1
        conn.execute(
            """INSERT INTO fulltext (arxiv_id, version, format, path, sha256, status, fetched_at)
               VALUES (:arxiv_id, :version, :format, :path, :sha256, :status, :fetched_at)
               ON CONFLICT(arxiv_id) DO UPDATE SET version = excluded.version, format = excluded.format,
                   path = excluded.path, sha256 = excluded.sha256, status = excluded.status,
                   fetched_at = excluded.fetched_at""",
            record,
        )
        conn.commit()
        print(f"fulltext: {i}/{len(todo)} {arxiv_id}v{version} -> {record['format'] or 'failed'} {stats}")
    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Fetch recent arXiv papers into data/papers.db")
    sub = parser.add_subparsers(dest="command", required=True)
    meta = sub.add_parser("metadata")
    meta.add_argument("--categories", nargs="+", default=["cs.CL", "cs.AI"])
    meta.add_argument("--limit", type=int, default=300)
    meta.add_argument("--page-size", type=int, default=100)
    meta.add_argument("--days", type=int, help="stop once papers are older than this many days")
    full = sub.add_parser("fulltext")
    full.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "metadata":
        fetch_metadata(args.categories, args.limit, args.page_size, args.days)
    else:
        fetch_fulltext(args.limit)


if __name__ == "__main__":
    main()
