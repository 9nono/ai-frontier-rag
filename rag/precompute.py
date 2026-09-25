import json
from datetime import datetime, timezone

from app.server import FEATURED_PATH
from rag.generate import DEFAULT_MODEL, answer

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
    results, cost = [], 0.0
    in_price, out_price = PRICE_PER_MTOK[DEFAULT_MODEL]
    for question in FEATURED_QUESTIONS:
        result = answer(question, translate=True)
        usage = result["usage"]
        cost += usage["input_tokens"] / 1e6 * in_price + usage["output_tokens"] / 1e6 * out_price
        results.append({k: result[k] for k in ("question", "answer", "refused", "sources", "model", "usage")})
        print(f"[{len(results)}/{len(FEATURED_QUESTIONS)}] {question[:60]} -> {len(result['sources'])} sources, "
              f"{usage['input_tokens']} in / {usage['output_tokens']} out", flush=True)
    for item in results:
        item["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    FEATURED_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {FEATURED_PATH} | estimated cost ${cost:.4f}")


if __name__ == "__main__":
    main()
