import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def load_questions():
    with open(EVAL_DIR / "questions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(hit, question):
    return hit["arxiv_id"] == question["arxiv_id"] and normalize(question["evidence"]) in normalize(hit["text"])
