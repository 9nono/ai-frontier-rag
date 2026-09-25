import argparse
import json
from datetime import datetime, timezone

from app.server import FEATURED_PATH
from rag.generate import DEFAULT_MODEL, answer
from rag.translate import needs_translation

PRICE_PER_MTOK = {"claude-haiku-4-5": (1.00, 5.00)}

FEATURED_QUESTIONS = [
    "How are recent papers handling memory for long-horizon LLM agents?",
    "What new methods reduce KV cache memory for long-context inference?",
    "How is reinforcement learning being used to improve LLM reasoning in these papers?",
    "What benchmarks evaluate whether AI agents act safely over multi-step tasks?",
    "How are diffusion language models being made faster or better at reasoning?",
    "What did the Hunyuan-A13B technical report find about its mixture-of-experts model?",
    "最近有哪些检测或缓解大模型幻觉的新方法？",
    "检索增强生成（RAG）最近有什么改进？",
    "有哪些工作研究多智能体系统的安全风险？",
    "When will GPT-7 be released and what is its context window?",
]


def main():
    parser = argparse.ArgumentParser(description="Generate the featured answers shown in the web UI")
    parser.add_argument("--only", type=int, nargs="+", metavar="N",
                        help="regenerate only these questions (1-based) and keep the rest of the file")
    args = parser.parse_args()
    existing = json.loads(FEATURED_PATH.read_text(encoding="utf-8")) if FEATURED_PATH.exists() else []
    kept = {item["question"]: item for item in existing}
    todo = [FEATURED_QUESTIONS[n - 1] for n in args.only] if args.only else FEATURED_QUESTIONS

    cost = 0.0
    in_price, out_price = PRICE_PER_MTOK[DEFAULT_MODEL]
    for i, question in enumerate(todo, 1):
        result = answer(question, translate=True)
        usage = result["usage"]
        cost += usage["input_tokens"] / 1e6 * in_price + usage["output_tokens"] / 1e6 * out_price
        item = {k: result[k] for k in ("question", "answer", "refused", "sources", "model", "usage")}
        item["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        kept[question] = item
        wrong_language = needs_translation(question) and not needs_translation(result["answer"] or "")
        print(f"[{i}/{len(todo)}] {question[:60]} -> {len(result['sources'])} sources, "
              f"{usage['input_tokens']} in / {usage['output_tokens']} out"
              f"{'  WARNING: answered in the wrong language' if wrong_language else ''}", flush=True)
    results = [kept[q] for q in FEATURED_QUESTIONS if q in kept]
    FEATURED_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {FEATURED_PATH} | estimated cost ${cost:.4f}")


if __name__ == "__main__":
    main()
