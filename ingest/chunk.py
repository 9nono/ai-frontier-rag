import re

MAX_WORDS = 200
ABSTRACT_MAX_WORDS = 300
CAPTION_HEAD_WORDS = 25
OVERLAP_WORDS = 40
STRATEGIES = ("fixed", "section", "section_header")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\[(])")


def _words(text):
    return text.split()


def _split_long(text, max_words):
    pieces, current = [], []
    for sentence in SENTENCE_END.split(text):
        words = _words(sentence)
        while len(words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if current and len(current) + len(words) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_table(text, max_words):
    caption, *rows = text.split("\n")
    caption_words = _words(caption)
    if len(caption_words) > CAPTION_HEAD_WORDS:
        head = " ".join(caption_words[:CAPTION_HEAD_WORDS]) + " …"
        pieces = _split_long(caption, max_words)
    else:
        head, pieces = caption, []
    budget = max_words - len(_words(head))
    current, size = [], 0
    for row in rows:
        for part in _split_long(row, budget):
            n = len(_words(part))
            if current and size + n > budget:
                pieces.append("\n".join([head, *current]))
                current, size = [], 0
            current.append(part)
            size += n
    if current or not pieces:
        pieces.append("\n".join([head, *current]))
    return pieces


def fixed_chunks(blocks, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
    stream = []
    for block in blocks:
        section = " > ".join(block.section)
        stream.extend((word, section) for word in _words(block.text))
    step = max_words - overlap
    chunks = []
    for start in range(0, max(len(stream) - overlap, 1), step):
        window = stream[start:start + max_words]
        if window:
            chunks.append({"section": window[0][1], "kind": "text", "text": " ".join(w for w, _ in window)})
    return chunks


def section_chunks(blocks, max_words=MAX_WORDS):
    chunks = []
    pending, pending_section = [], None

    def flush():
        if pending:
            chunks.append({"section": pending_section, "kind": "para", "text": " ".join(pending)})
            pending.clear()

    for block in blocks:
        section = " > ".join(block.section)
        if block.kind in ("table", "figure"):
            flush()
            splitter = _split_table if block.kind == "table" else _split_long
            for piece in splitter(block.text, max_words):
                chunks.append({"section": section, "kind": block.kind, "text": piece})
            continue
        if section != pending_section:
            flush()
            pending_section = section
        for piece in _split_long(block.text, max_words):
            if pending and len(_words(" ".join(pending))) + len(_words(piece)) > max_words:
                flush()
            pending.append(piece)
    flush()
    return chunks


def chunk_paper(paper, blocks, strategy):
    body = fixed_chunks(blocks) if strategy == "fixed" else section_chunks(blocks)
    abstract = [{"section": "Abstract", "kind": "abstract", "text": piece}
                for piece in _split_long(paper["summary"], ABSTRACT_MAX_WORDS)]
    records = []
    for i, chunk in enumerate([*abstract, *body]):
        embed_text = chunk["text"]
        if strategy == "section_header":
            embed_text = f"{paper['title']}\nSection: {chunk['section']}\n{chunk['text']}"
        records.append({
            "chunk_id": f"{paper['arxiv_id']}v{paper['version']}:{strategy}:{i:04d}",
            "arxiv_id": paper["arxiv_id"],
            "version": paper["version"],
            "title": paper["title"],
            "published": paper["published"],
            "strategy": strategy,
            "section": chunk["section"],
            "kind": chunk["kind"],
            "text": chunk["text"],
            "embed_text": embed_text,
            "n_words": len(_words(chunk["text"])),
        })
    return records
