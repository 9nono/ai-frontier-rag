import json
import statistics
import time
from datetime import datetime

import torch
from sentence_transformers import CrossEncoder

from rag.index import DEVICE, MAX_SEQ_TOKENS
from rag.retrieve import RERANK_CANDIDATES, RERANKERS, hybrid_search
from rag.translate import needs_translation, to_english

from eval.common import load_questions
from eval.run_retrieval import RESULTS_DIR

TOP = 6  # passages the generator receives
SETTINGS = [
    # (label, device, precision, max_length, batch_size); the first is the reference
    ("fp32, batch 32", DEVICE, "fp32", MAX_SEQ_TOKENS, 32),
    ("fp32, batch 16", DEVICE, "fp32", MAX_SEQ_TOKENS, 16),
    ("fp32, batch 8", DEVICE, "fp32", MAX_SEQ_TOKENS, 8),
    ("fp32, batch 4", DEVICE, "fp32", MAX_SEQ_TOKENS, 4),
    ("fp32, cut to 384 tokens", DEVICE, "fp32", 384, 32),
    ("fp32, cut to 256 tokens", DEVICE, "fp32", 256, 32),
    ("fp16, batch 32", DEVICE, "fp16", MAX_SEQ_TOKENS, 32),
    ("fp16, batch 8", DEVICE, "fp16", MAX_SEQ_TOKENS, 8),
    ("cpu fp32, batch 32", "cpu", "fp32", MAX_SEQ_TOKENS, 32),
]


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    """Rerank the same hybrid candidates under each setting; time it and compare top-6 order to the reference."""
    questions = load_questions()
    queries = [to_english(q["question"]) if needs_translation(q["question"]) else q["question"] for q in questions]
    pairs = [
        [(query, f"{h['title']}\nSection: {h['section']}\n{h['text']}")
         for h in hybrid_search(query, "section_header", k=RERANK_CANDIDATES)]
        for query in queries
    ]
    tokenizer = CrossEncoder(RERANKERS["minilm"], device="cpu").tokenizer
    lengths = [[len(tokenizer(*p)["input_ids"]) for p in ps] for ps in pairs]
    flat = [n for ls in lengths for n in ls]
    print(f"pair tokens: median {statistics.median(flat):.0f}, over 384: {sum(n > 384 for n in flat) / len(flat):.0%}, "
          f"longest per query (batch padding): median {statistics.median(max(ls) for ls in lengths):.0f}")

    report, reference = {}, None
    print(f"{'setting':26s} {'p50 ms':>7s}  same top {TOP}")
    for label, device, precision, max_length, batch in SETTINGS:
        model = CrossEncoder(RERANKERS["minilm"], device=device, max_length=max_length)
        if precision == "fp16":
            model.half()
        model.predict(pairs[0], batch_size=batch)
        _sync(device)
        ms, tops = [], []
        for ps in pairs:
            started = time.perf_counter()
            scores = model.predict(ps, batch_size=batch)
            _sync(device)
            ms.append(1000 * (time.perf_counter() - started))
            tops.append(sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP])
        reference = reference or tops
        same = sum(a == b for a, b in zip(tops, reference))
        report[label] = {"device": device, "precision": precision, "max_length": max_length, "batch_size": batch,
                         "p50_ms": statistics.median(ms), "same_top": same, "n": len(pairs)}
        print(f"{label:26s} {statistics.median(ms):7.0f}  {same}/{len(pairs)}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"rerank-latency-{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
