import re
from functools import lru_cache

from transformers import MarianMTModel, MarianTokenizer

from rag.index import DEVICE

MT_MODEL = "Helsinki-NLP/opus-mt-zh-en"
CJK = re.compile(r"[一-鿿]")
LATIN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9.\-]*[A-Za-z0-9]|[A-Za-z]")


GLOSSARY = {
    "多智能体": " multi-agent ", "智能体": " agent ", "大语言模型": " LLM ", "大模型": " LLM ", "语言模型": " language model ",
    "检索增强生成": " retrieval-augmented generation ", "检索增强": " retrieval-augmented ", "向量检索": " vector retrieval ",
    "嵌入": " embedding ", "重排序": " reranking ", "提示词": " prompt ", "上下文": " context ", "长上下文": " long-context ",
    "微调": " fine-tuning ", "预训练": " pre-training ", "蒸馏": " distillation ", "剪枝": " pruning ", "量化": " quantization ",
    "去重": " deduplication ", "幻觉": " hallucination ", "对齐": " alignment ", "强化学习": " reinforcement learning ",
    "奖励模型": " reward model ", "思维链": " chain-of-thought ", "推理模型": " reasoning model ", "多模态": " multimodal ",
    "视觉语言模型": " vision-language model ", "扩散模型": " diffusion model ", "混合专家": " mixture-of-experts ",
    "注意力": " attention ", "基准": " benchmark ", "评测": " evaluation ", "越狱": " jailbreak ", "投毒": " poisoning ",
    "提示注入": " prompt injection ", "缓存": " cache ", "消融": " ablation ", "数据集": " dataset ", "准确率": " accuracy ",
    "混元": " Hunyuan ", "通义千问": " Qwen ", "千问": " Qwen ",
}
GLOSSARY_ORDER = sorted(GLOSSARY, key=len, reverse=True)


def needs_translation(text):
    return bool(CJK.search(text))


def apply_glossary(text):
    for term in GLOSSARY_ORDER:
        text = text.replace(term, GLOSSARY[term])
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _model():
    tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
    model = MarianMTModel.from_pretrained(MT_MODEL).to(DEVICE).eval()
    return tokenizer, model


def to_english(text):
    tokenizer, model = _model()
    source = apply_glossary(text)
    batch = tokenizer([source], return_tensors="pt", truncation=True).to(DEVICE)
    output = model.generate(**batch, max_new_tokens=128, max_length=None, num_beams=4)
    translated = tokenizer.decode(output[0], skip_special_tokens=True)
    kept = [t for t in LATIN_TERM.findall(source) if t.lower() not in translated.lower()]
    return " ".join([translated, *kept]).strip()
