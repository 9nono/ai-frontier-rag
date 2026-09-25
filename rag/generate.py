import argparse
import json

import anthropic

from rag.retrieve import search

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_MODELS = {"claude-opus-5"}
NO_EFFORT_MODELS = {"claude-haiku-4-5"}
TOP_K = 6

SYSTEM = """You answer questions about recent AI research papers using only the paper excerpts provided.
- Support every factual claim with the excerpts and cite them.
- If the excerpts do not contain the answer, say plainly that the indexed papers do not answer it. Do not guess or fill gaps from general knowledge.
- If excerpts from different papers disagree, say so and attribute each claim to its paper.
- Answer in the language of the question. Keep it short: one paragraph or a few bullets."""


def build_documents(hits):
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h["text"]},
            "title": f"{h['title']} (arXiv:{h['arxiv_id']})",
            "context": f"Section: {h['section']}. Published {h['published'][:10]}.",
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def _request(model, effort, hits, question):
    kwargs = dict(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": [*build_documents(hits), {"type": "text", "text": question}]}],
    )
    if model not in NO_EFFORT_MODELS:
        kwargs["output_config"] = {"effort": effort}
    if model in FALLBACK_MODELS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return anthropic.Anthropic().beta.messages.create(**kwargs)


def _render(response, hits):
    parts, sources = [], {}
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        markers = []
        for citation in block.citations or []:
            hit = hits[citation.document_index]
            n = sources.setdefault(hit["chunk_id"], {
                "n": len(sources) + 1, "arxiv_id": hit["arxiv_id"], "title": hit["title"],
                "section": hit["section"], "cited": [],
            })["n"]
            if citation.cited_text not in sources[hit["chunk_id"]]["cited"]:
                sources[hit["chunk_id"]]["cited"].append(citation.cited_text)
            if n not in markers:
                markers.append(n)
        if markers:
            parts.append("".join(f"[{n}]" for n in markers))
    return "".join(parts).strip(), sorted(sources.values(), key=lambda s: s["n"])


def answer(question, model=DEFAULT_MODEL, effort="low", k=TOP_K, **search_kwargs):
    hits = search(question, k=k, **search_kwargs)
    response = _request(model, effort, hits, question)
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        return {"question": question, "answer": None, "refused": True, "sources": [], "hits": hits,
                "model": response.model, "usage": usage}
    text, sources = _render(response, hits)
    return {"question": question, "answer": text, "refused": False, "sources": sources, "hits": hits,
            "model": response.model, "usage": usage}


def main():
    parser = argparse.ArgumentParser(description="Answer a question from the indexed papers with citations")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--since", help="only papers published on/after this date, e.g. 2026-09-20")
    args = parser.parse_args()
    result = answer(args.question, model=args.model, effort=args.effort, since=args.since)
    if result["refused"]:
        print("The model declined to answer this question.")
    else:
        print(result["answer"], "\n")
        for s in result["sources"]:
            print(f"[{s['n']}] {s['title']} (arXiv:{s['arxiv_id']}) — {s['section']}")
    print(f"\nmodel={result['model']} usage={json.dumps(result['usage'])}")


if __name__ == "__main__":
    main()
