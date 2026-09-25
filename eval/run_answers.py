import argparse
import json
import re
import time
from datetime import datetime

import anthropic

from rag.generate import DEFAULT_MODEL, PRICE_PER_MTOK, TOP_K, _render, build_request
from rag.keyword_index import _connection, tokenize
from rag.retrieve import search
from rag.translate import needs_translation

from eval.common import EVAL_DIR, is_relevant, load_questions, normalize
from eval.run_retrieval import RESULTS_DIR

BATCH_DISCOUNT = 0.5
POLL_SECONDS = 20
# Phrases an answer uses to say the passages do not contain what was asked.
ABSTAIN = re.compile(
    r"\b(do|does|did)\s?n[o']t\s+"
    r"(contain|mention|include|provide|discuss|address|cover|report|describe|answer|specify)"
    r"|\bno\s+(information|mention|details?|data)\b"
    r"|\bnot\s+(mentioned|found|covered|discussed|addressed|available|present|included|specified|reported|described)\b"
    r"|\bcannot\s+(be\s+)?(answer|determine|find|found)|\bunable\s+to\b|\bnone of the (excerpts|papers|passages)"
    r"|\b(do|does|did)\s?n[o']t\s+have\s+(any\s+)?information|\b(could\s?n[o']t|did\s?n[o']t)\s+find"
    r"|没有(提到|提及|涉及|包含|找到|给出|说明|讨论|关于|相关)|未(提及|提到|涉及|包含|找到|给出|说明)"
    r"|无法.{0,12}(回答|确定|找到|提供)|找不到|不包含|并未",
    re.IGNORECASE,
)


def load_unanswerable():
    with open(EVAL_DIR / "unanswerable.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_absent(items):
    """An unanswerable question is only unanswerable if its key terms never occur in the corpus."""
    conn = _connection("section_header")
    for item in items:
        for term in item["absent"]:
            phrase = " ".join(tokenize(term))
            n = conn.execute("SELECT count(*) FROM chunk_terms WHERE chunk_terms MATCH ?",
                             (f'"{phrase}"',)).fetchone()[0]
            if n:
                raise SystemExit(f"{item['id']}: '{term}' occurs in {n} chunks, so the question may be answerable")


def prepare(answerable, unanswerable):
    items = []
    for q, kind in [(q, "answerable") for q in answerable] + [(q, "unanswerable") for q in unanswerable]:
        hits = search(q["question"], k=TOP_K, translate=True)
        items.append({
            "id": q["id"], "kind": kind, "lang": q["lang"], "type": q["type"], "question": q["question"],
            "evidence": q.get("evidence"), "arxiv_id": q.get("arxiv_id"),
            "hits": [{"chunk_id": h["chunk_id"], "arxiv_id": h["arxiv_id"], "section": h["section"],
                      "relevant": kind == "answerable" and is_relevant(h, q)} for h in hits],
            "_hits": hits,
        })
    return items


def generate_sync(items):
    client = anthropic.Anthropic()
    return {item["id"]: client.messages.create(**build_request(item["question"], item["_hits"])) for item in items}


def generate_batch(items, resume=None):
    client = anthropic.Anthropic()
    if resume:
        batch = client.messages.batches.retrieve(resume)
        print(f"batch {batch.id}: resumed", flush=True)
    else:
        requests = []
        for item in items:
            params = build_request(item["question"], item["_hits"])
            assert "betas" not in params, "batch requests cannot carry beta headers"
            requests.append({"custom_id": item["id"], "params": params})
        batch = client.messages.batches.create(requests=requests)
        print(f"batch {batch.id}: {len(requests)} requests submitted", flush=True)
    while batch.processing_status != "ended":
        time.sleep(POLL_SECONDS)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"  {batch.processing_status}: {counts.succeeded} succeeded, {counts.processing} processing, "
              f"{counts.errored} errored", flush=True)
    responses = {}
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type != "succeeded":
            raise SystemExit(f"{entry.custom_id}: {entry.result.type}")
        responses[entry.custom_id] = entry.result.message
    return responses


def attach(items, responses):
    for item in items:
        response = responses[item["id"]]
        text, _ = _render(response, item["_hits"])
        item["answer"] = text
        item["stop_reason"] = response.stop_reason
        item["citations"] = [
            {"doc": c.document_index, "cited_text": c.cited_text}
            for block in response.content if block.type == "text" for c in block.citations or []
        ]
        item["usage"] = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        del item["_hits"]


def classify(item):
    if item["stop_reason"] == "refusal":
        return "refused"
    if not item["citations"]:
        return "abstained"
    return "hedged" if ABSTAIN.search(item["answer"] or "") else "answered"


def score(items):
    for item in items:
        item["outcome"] = classify(item)
        answer = item["answer"] or ""
        item["language_ok"] = needs_translation(answer) == (item["lang"] == "zh")
        if item["kind"] == "answerable":
            cited = [(item["hits"][c["doc"]], c["cited_text"]) for c in item["citations"]]
            evidence = normalize(item["evidence"])
            item["retrieved"] = any(h["relevant"] for h in item["hits"])
            item["cited_relevant"] = any(h["relevant"] for h, _ in cited)
            item["evidence_cited"] = any(h["arxiv_id"] == item["arxiv_id"] and evidence in normalize(t)
                                         for h, t in cited)

    def ids(rows):
        return [r["id"] for r in rows]

    ans = [i for i in items if i["kind"] == "answerable"]
    got = [i for i in ans if i["retrieved"]]
    missed = [i for i in ans if not i["retrieved"]]
    una = [i for i in items if i["kind"] == "unanswerable"]
    by_outcome = lambda rows: {o: ids([r for r in rows if r["outcome"] == o])
                               for o in ("answered", "hedged", "abstained", "refused")}
    summary = {
        "answerable": {
            "n": len(ans),
            "retrieved": len(got),
            "retrieved_cited_relevant": sum(i["cited_relevant"] for i in got),
            "retrieved_evidence_cited": sum(i["evidence_cited"] for i in got),
            "retrieved_outcomes": by_outcome(got),
            "not_retrieved_outcomes": by_outcome(missed),
        },
        "unanswerable": {"n": len(una), "outcomes": by_outcome(una),
                         "by_type": {t: by_outcome([i for i in una if i["type"] == t])
                                     for t in sorted({i["type"] for i in una})}},
        "language_mismatch": ids([i for i in items if not i["language_ok"]]),
    }
    return summary


def print_summary(s):
    a, u = s["answerable"], s["unanswerable"]
    counts = lambda d: ", ".join(f"{k} {len(v)}" for k, v in d.items() if v)
    print(f"answerable: {a['n']} | relevant passage among the {TOP_K} given: {a['retrieved']}")
    print(f"  given the passage: cited it {a['retrieved_cited_relevant']}/{a['retrieved']}, "
          f"cited span contains the evidence {a['retrieved_evidence_cited']}/{a['retrieved']}")
    print(f"  given the passage, outcomes: {counts(a['retrieved_outcomes'])}")
    for outcome in ("hedged", "abstained", "refused"):
        if a["retrieved_outcomes"][outcome]:
            print(f"    {outcome}: {', '.join(a['retrieved_outcomes'][outcome])}")
    print(f"  not given the passage, outcomes: {counts(a['not_retrieved_outcomes'])}")
    for outcome, rows in a["not_retrieved_outcomes"].items():
        if rows:
            print(f"    {outcome}: {', '.join(rows)}")
    print(f"unanswerable: {u['n']} | outcomes: {counts(u['outcomes'])}")
    for outcome in ("answered", "hedged"):
        if u["outcomes"][outcome]:
            print(f"    {outcome}: {', '.join(u['outcomes'][outcome])}")
    print(f"language mismatches: {', '.join(s['language_mismatch']) or 'none'}")


def main():
    """Answer the evaluation questions with the production pipeline and score the answers by rule."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="call the API directly instead of the Batch API")
    parser.add_argument("--limit", type=int, help="first N answerable and first N unanswerable questions only")
    parser.add_argument("--rescore", help="score a saved answers-*.json again without calling the API")
    parser.add_argument("--resume", metavar="BATCH_ID", help="collect a batch that was already submitted")
    args = parser.parse_args()

    if args.rescore:
        with open(args.rescore, encoding="utf-8") as f:
            run = json.load(f)
        print_summary(score(run["items"]))
        return

    answerable, unanswerable = load_questions(), load_unanswerable()
    check_absent(unanswerable)
    if args.limit:
        answerable, unanswerable = answerable[:args.limit], unanswerable[:args.limit]
    items = prepare(answerable, unanswerable)
    responses = generate_sync(items) if args.sync else generate_batch(items, args.resume)
    attach(items, responses)
    summary = score(items)

    in_price, out_price = PRICE_PER_MTOK[DEFAULT_MODEL]
    tokens_in = sum(i["usage"]["input_tokens"] for i in items)
    tokens_out = sum(i["usage"]["output_tokens"] for i in items)
    cost = (tokens_in * in_price + tokens_out * out_price) / 1e6 * (1 if args.sync else BATCH_DISCOUNT)
    print_summary(summary)
    print(f"{tokens_in} input / {tokens_out} output tokens, ${cost:.4f} ({'direct' if args.sync else 'batch'})")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"answers-{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"model": DEFAULT_MODEL, "mode": "sync" if args.sync else "batch", "top_k": TOP_K,
                   "cost_usd": cost, "summary": summary, "items": items}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
