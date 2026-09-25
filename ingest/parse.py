import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

SECTION_CLASSES = ("ltx_section", "ltx_subsection", "ltx_subsubsection", "ltx_appendix", "ltx_paragraph")
SKIP_SECTION = re.compile(r"acknowledg|disclosure of interests|funding|author contributions|competing interests", re.I)
NOISE_SELECTORS = (
    "script", "style", "nav", ".ltx_bibliography", ".ltx_TOC", ".ltx_page_footer", ".ltx_page_header",
    ".ltx_authors", ".ltx_note", ".ltx_dates", ".ltx_role_footnote", ".ltx_abstract", ".ltx_keywords",
)
PDF_HEADING = re.compile(r"^(?:\d+(?:\.\d+){0,2}|[A-Z](?:\.\d+)?)\.?\s+[A-Z][^\n]{1,80}$")
PDF_STOP = re.compile(r"^(references|bibliography|acknowledg(e)?ments?)\s*$", re.I)


@dataclass
class Block:
    section: list
    kind: str
    text: str


@dataclass
class Document:
    blocks: list = field(default_factory=list)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _section_path(node):
    path = []
    for parent in node.parents:
        classes = parent.get("class") or []
        if parent.name == "section" and any(c in SECTION_CLASSES for c in classes):
            heading = parent.find(re.compile("^h[1-6]$"), class_="ltx_title", recursive=False)
            if heading:
                path.append(clean(heading.get_text(" ")))
    return list(reversed(path)) or ["Body"]


def _table_text(figure):
    rows = []
    for tr in figure.select("tr"):
        cells = [clean(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_html(path):
    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8", errors="replace"), "lxml")
    for selector in NOISE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    for math in soup.find_all("math"):
        alt = math.get("alttext")
        math.replace_with(f" ${alt}$ " if alt else " ")
    doc = Document()
    for node in soup.select("div.ltx_para, figure.ltx_figure, figure.ltx_table"):
        if node.find_parent("figure") is not None:
            continue
        if node.name == "figure":
            caption = node.find("figcaption")
            caption_text = clean(caption.get_text(" ")) if caption else ""
            if "ltx_table" in node.get("class", []):
                body = _table_text(node)
                text = f"{caption_text}\n{body}".strip()
                kind = "table"
            else:
                text, kind = caption_text, "figure"
        else:
            if node.find_parent("div", class_="ltx_para") is not None:
                continue
            text, kind = clean(node.get_text(" ")), "para"
        section = _section_path(node)
        if text and not SKIP_SECTION.search(section[-1]):
            doc.blocks.append(Block(section, kind, text))
    return doc


def _ordered_blocks(page):
    width = page.rect.width
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    right_col = lambda b: b[0] >= width * 0.45 and (b[2] - b[0]) < width * 0.6
    return sorted(blocks, key=lambda b: (right_col(b), round(b[1])))


def parse_pdf(path):
    doc = Document()
    section = ["Body"]
    with pymupdf.open(path) as pdf:
        for page_no, page in enumerate(pdf, 1):
            for block in _ordered_blocks(page):
                text = clean(block[4])
                first_line = block[4].strip().splitlines()[0].strip()
                if PDF_STOP.match(first_line):
                    return doc
                if len(text.split()) <= 12 and PDF_HEADING.match(text):
                    section = [text]
                    continue
                front_matter = page_no == 1 and section == ["Body"]
                if front_matter or len(text.split()) < 4 or SKIP_SECTION.search(section[0]):
                    continue
                doc.blocks.append(Block(section + [f"p.{page_no}"], "para", text))
    return doc


def parse(path, fmt):
    return parse_html(path) if fmt == "html" else parse_pdf(path)
