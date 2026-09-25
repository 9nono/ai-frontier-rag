import argparse
import json
import random


def reciprocal_rank(row):
    return 1 / row["first_rank"] if row["first_rank"] else 0.0


def fmt_rank(rank):
    return str(rank) if rank else "-"


def bootstrap_ci(diffs, resamples, seed):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def main():
    """Compare two configurations from one results file, question by question."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with open(args.results, encoding="utf-8") as f:
        report = json.load(f)
    base = {r["id"]: r for r in report[args.baseline]["rows"]}
    cand = {r["id"]: r for r in report[args.candidate]["rows"]}

    for lang in ("en", "zh"):
        ids = [i for i in base if base[i]["lang"] == lang]
        diffs = [reciprocal_rank(cand[i]) - reciprocal_rank(base[i]) for i in ids]
        low, high = bootstrap_ci(diffs, args.resamples, args.seed)
        moved = [(i, base[i]["first_rank"], cand[i]["first_rank"]) for i, d in zip(ids, diffs) if d]
        print(f"{lang} (n={len(ids)}): MRR {sum(diffs) / len(ids):+.3f}, 95% CI [{low:+.3f}, {high:+.3f}]")
        for label, keep in (("better", lambda b, c: (c or 99) < (b or 99)), ("worse", lambda b, c: (c or 99) > (b or 99))):
            items = [f"{i} {fmt_rank(b)}->{fmt_rank(c)}" for i, b, c in moved if keep(b, c)]
            print(f"  {label}: {', '.join(items) or '-'}")


if __name__ == "__main__":
    main()
