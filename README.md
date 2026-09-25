# AI Frontier Search

Retrieval-augmented question answering over the latest arXiv AI papers (cs.CL, cs.AI), with an evaluation harness that drives every design decision.

- **Search** the full text of recent papers in English or Chinese, filter by publication date, and see the exact passages that match.
- **Ask** questions and get short answers with sentence-level citations back to the source passages; when the papers don't contain the answer, the system says so instead of guessing.
- **Measure** retrieval quality on a hand-built question set, so each change (chunking, hybrid search, reranking, query translation) is kept or dropped based on numbers.

> 中文简介：一个基于最新 arXiv AI 论文的检索增强问答系统。支持中英文提问、按日期过滤、句子级引用和"证据不足就拒答"。分块策略、混合检索、重排序和中文查询翻译的每个选择，都用自建的评估集量化对比后才决定。

## Architecture

```
arXiv API ──► fetch_arxiv ──► papers.db (SQLite, idempotent upsert by arXiv id + version)
                   │
                   ▼
        HTML (LaTeXML) first, PDF fallback ──► parse ──► blocks with section paths
                                                          │
                                                          ▼
                                   chunk (fixed | section | section + header)
                                                          │
                          ┌───────────────────────────────┴──────────────┐
                          ▼                                              ▼
            bge-small-en-v1.5 embeddings                          BM25 index
            Chroma (HNSW, cosine), content-hash sync              (in memory)
                          └──────────────┬───────────────────────────────┘
                                         ▼
     query ─► zh→en translation (if Chinese) ─► hybrid retrieval (RRF) ─► cross-encoder rerank
                                         │
                     ┌───────────────────┴────────────────────┐
                     ▼                                        ▼
           web UI (retrieval only, free)          Claude Haiku 4.5 + native citations
                                                  (local, or precomputed featured answers)
```

## How it works

**Ingestion.** Papers come from the arXiv API sorted by submission date. Each paper is keyed by its arXiv id; re-running the fetcher inserts new papers, updates a paper only when a newer version appears, and leaves everything else untouched. Requests are spaced 3 seconds apart per arXiv's API guidance.

**Parsing.** arXiv's HTML rendering (LaTeXML) preserves the section tree, so it is preferred over PDF. Math is replaced by its LaTeX source (`alttext`), tables are flattened row by row because results questions are usually answered by tables, and references, footnotes and acknowledgements are dropped. Papers without an HTML version fall back to PDF parsing with a two-column reading-order heuristic and numbered-heading detection.

**Chunking.** Three strategies are implemented and compared:

| Strategy | Rule |
|---|---|
| `fixed` | 200-word windows with 40-word overlap, ignoring structure |
| `section` | paragraphs packed up to 200 words without crossing section boundaries; tables and figure captions are their own chunks |
| `section_header` | same chunks as `section`, but the text that is embedded and indexed starts with `paper title + section path` |

Abstracts are always kept as their own chunk.

**Indexing.** Chunks are embedded with `BAAI/bge-small-en-v1.5` (with its query instruction prefix) and stored in Chroma. Indexing is a sync, not an append: every chunk carries a hash of its indexed text, so a run embeds new or changed chunks, skips unchanged ones, and deletes chunks that no longer exist (for example after a paper is revised).

**Retrieval.** BM25 and vector search each return 50 candidates; they are merged with reciprocal rank fusion, and the top 30 are rescored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`) that sees the same `title + section + text` the index saw. Chinese queries are translated locally (`Helsinki-NLP/opus-mt-zh-en`) after a small glossary of general AI terms is applied, and English terms in the query are kept verbatim.

**Answering.** The top passages are sent to Claude Haiku 4.5 as `document` blocks with citations enabled, so every cited span points back to an exact passage. The system prompt requires answering only from the passages, saying so when they don't contain the answer, and attributing disagreements between papers.

**Cost control.** Everything except answer generation runs locally. The public web app serves retrieval plus a set of answers generated once in advance; live answering is enabled only when `ENABLE_LIVE_ANSWERS=1` is set on a local machine.

## Evaluation

### Method

60 questions (50 English, 10 Chinese) were written from 60 passages sampled with a fixed seed. Each question is labeled with its source paper and a short **evidence string** copied from the passage (a number, a method name, a phrase). A retrieved chunk counts as relevant if it comes from that paper and contains the evidence string. Because relevance is defined by evidence rather than by chunk id, the same labels work for every chunking strategy. `eval/validate.py` checks that every evidence string actually occurs in each strategy's chunks.

Question types: 30 factual lookups (numbers, datasets, results), 12 method or mechanism questions, 5 findings, 3 "find the paper" questions. Questions paraphrase rather than copy the passage, so keyword search doesn't get free matches.

Metrics: hit@k (a relevant chunk in the top k), paper@5 (the right paper in the top 5, regardless of passage), and MRR@10.

### Results (300-paper corpus, 2026-09-22 to 2026-09-23)

English questions (n = 50):

| Configuration | hit@1 | hit@5 | hit@10 | paper@5 | MRR |
|---|---|---|---|---|---|
| BM25 / fixed | 0.46 | 0.72 | 0.84 | 0.94 | 0.575 |
| BM25 / section | 0.42 | 0.66 | 0.74 | 0.94 | 0.511 |
| BM25 / section + header | 0.54 | 0.78 | 0.90 | 0.96 | 0.640 |
| Vector / fixed | 0.24 | 0.52 | 0.74 | 0.98 | 0.380 |
| Vector / section | 0.24 | 0.50 | 0.70 | 0.96 | 0.368 |
| Vector / section + header | 0.38 | 0.76 | 0.90 | 1.00 | 0.529 |
| Hybrid (RRF) / section + header | 0.50 | 0.86 | 0.92 | 1.00 | 0.649 |
| **Hybrid + rerank (MiniLM) / section + header** | **0.64** | **0.92** | **0.96** | **1.00** | **0.749** |
| Hybrid + rerank (bge-reranker-base) / section + header | 0.62 | 0.88 | 0.96 | 1.00 | 0.748 |

Chinese questions (n = 10), best configuration:

| Query handling | hit@1 | hit@5 | paper@5 | MRR |
|---|---|---|---|---|
| No translation | 0.00 | 0.20 | 0.50 | 0.100 |
| Translation only | 0.60 | 0.70 | 1.00 | 0.664 |
| Translation + glossary | 0.60 | 0.80 | 1.00 | 0.670 |

### What the numbers changed

1. **Contextual headers matter most for dense retrieval.** Section-sized chunks alone did worse than fixed windows, because a short paragraph often never names the paper or the method it describes. Prefixing `title + section` raised vector hit@5 from 0.50 to 0.76.
2. **Hybrid search beats either retriever alone.** BM25 is strong on model names, dataset names and numbers; dense retrieval is strong on paraphrased questions. Fusing them reached hit@5 0.86.
3. **Reranking only helped once its input matched the index.** The first reranking run passed the cross-encoder the bare passage text and *lowered* MRR (0.649 → 0.605). Giving it the same `title + section` context the index uses raised MRR to 0.749 and hit@1 from 0.50 to 0.64. Paired by question, reranking improved hit@1 on 10 questions and hurt 3; the MRR gain is +0.100 with a 95% bootstrap interval of [+0.005, +0.196], so the gain is real but its size is uncertain.
4. **Query translation fixed Chinese questions cheaply.** The English embedding model found almost nothing for Chinese questions. A multilingual embedding model (`BAAI/bge-m3`) was the alternative, but it indexed at about 3 chunks/s on this machine versus about 110 chunks/s for `bge-small`, which is impractical at corpus scale. Translating the query instead took hit@5 from 0.20 to 0.80. Most of that gain comes from translation itself; the glossary added one question.

### Scaling to two weeks (2,440 papers, 190k chunks)

The same 60 questions against the full two-week corpus (the 300 original papers plus 2,140 more), `section + header` index only. Latency is the mean per query on an Apple-silicon laptop, embedding and reranking on the GPU.

| Configuration (English, n = 50) | hit@1 | hit@5 | hit@10 | MRR | ms/query | hit@5 at 300 papers |
|---|---|---|---|---|---|---|
| BM25 | 0.46 | 0.76 | 0.80 | 0.572 | 349 | 0.78 |
| Vector | 0.36 | 0.70 | 0.80 | 0.498 | 374 | 0.76 |
| Hybrid (RRF) | 0.46 | 0.72 | 0.86 | 0.585 | 261 | 0.86 |
| **Hybrid + rerank** | **0.58** | **0.88** | **0.92** | **0.693** | 872 | 0.92 |
| Hybrid + rerank, Chinese questions with translation (n = 10) | 0.50 | 0.70 | 0.70 | 0.570 | 802 | 0.80 |

Eight times more text means eight times more near-misses. Every retriever loses some precision; hybrid search loses the most (hit@5 0.86 → 0.72), and reranking recovers most of it (0.88). Two problems surfaced that did not exist at 300 papers:

- **Reranking can only reorder what it is given.** Hybrid hit@10 is 0.86, so for some questions the right passage is not among the 30 candidates sent to the reranker.
- **BM25 latency grows with the corpus.** `rank_bm25` scores every chunk in Python; going from 23k to 190k chunks took a query from about 40 ms to about 350 ms.

### Limitations

- One person wrote the questions and the evidence labels, and 50 questions is a small sample: a difference of 0.06 in hit@5 is three questions.
- The 10 Chinese questions were used while building the translation step (including choosing glossary terms), so they act as a development set, not a held-out test. The glossary only contains general AI terms, not terms specific to any question.
- Evidence matching is lenient: a chunk that contains the evidence string for an unrelated reason still counts.

## Next iterations

Each item is a question the evaluation can answer:

1. **Rerank pool size.** Does sending 50 or 100 candidates to the reranker instead of 30 recover the passages hybrid search ranks 31st to 100th, and what does it cost in latency?
2. **Keyword index that scales.** Replace in-memory BM25 with SQLite FTS5; compare latency and hit@k at 190k chunks.
3. **Time-aware ranking.** The date filter is a hard cutoff. Add a recency prior and detect "latest / recent" intent, with new questions whose correct answer depends on publication date.
4. **Answer-level evaluation.** Measure abstention on questions the corpus cannot answer, and check that every cited span actually contains the supporting evidence.
5. **A larger, independent question set.** 100+ questions, a separate held-out Chinese set, and a second annotator.

## Engineering notes

- **Embedding throughput.** On this Apple-silicon laptop, ONNX on CPU embedded 11 to 17 chunks/s regardless of batch size, CoreML acceleration gave 17.5 chunks/s, and PyTorch on the Apple GPU (MPS) gave 113 chunks/s with identical vectors (cosine similarity 1.0000), so indexing moved to MPS without re-embedding.
- **Resumable indexing.** Because indexing compares content hashes, an interrupted run picks up where it stopped; this was exercised when the backend was switched mid-run. Growing the corpus to two weeks embedded only the 166,910 new chunks and skipped the 23,354 already indexed.
- **Answer language.** Every passage is in English, and with only the system prompt asking for the question's language, one of the three Chinese featured questions was answered in English. Restating the language next to the question fixed it; `rag.precompute` now flags a Chinese question whose answer contains no Chinese, and `--only N` regenerates single questions instead of paying for all ten.
- **Ingestion at scale.** 2,440 papers downloaded at one request every 3 seconds with no failures: 2,279 from HTML, 161 from PDF.

## Running it

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m ingest.fetch_arxiv metadata --days 14 --limit 6000 --page-size 500
.venv/bin/python -m ingest.fetch_arxiv fulltext
.venv/bin/python -m ingest.build_chunks
.venv/bin/python -m rag.index --strategies section_header

.venv/bin/python -m eval.validate
.venv/bin/python -m eval.run_retrieval

.venv/bin/python -m app.server            # http://127.0.0.1:5077, retrieval only
```

Answer generation needs an Anthropic API key in `ANTHROPIC_API_KEY`:

```bash
.venv/bin/python -m rag.generate "How are recent papers handling memory for long-horizon LLM agents?"
.venv/bin/python -m rag.precompute                 # regenerate the featured answers
ENABLE_LIVE_ANSWERS=1 .venv/bin/python -m app.server
```

## Project layout

```
ingest/   fetch_arxiv.py  parse.py  chunk.py  build_chunks.py
rag/      index.py  retrieve.py  translate.py  generate.py  precompute.py
eval/     questions.jsonl  validate.py  run_retrieval.py  results/
app/      server.py  static/index.html  featured_answers.json
```
