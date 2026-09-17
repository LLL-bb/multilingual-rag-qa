"""分块。

策略是"先按段落、再按句子、最后带重叠地打包":

1. 段落是作者给的语义边界,优先尊重;
2. 单个段落超过字数预算时按句子切,避免一个块里塞进多个主题;
3. 块与块之间留 overlap_chars 的重叠,防止答案正好被切在边界上。

字数预算按语种区分(见 ``ChunkConfig``):中文字符信息密度约为英文的两倍,
共用一个上限会让中文块过大、英文块过碎,两边的检索粒度就不可比了。

分块是检索质量的第一决定因素 —— 块太大,检索命中了也定位不到答案;
块太小,单块信息不足以回答问题。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .config import ChunkConfig
from .language import detect_language
from .loaders import Document

# 中文句末标点后直接切;英文句末标点要求后面跟空格 + 大写/数字,
# 以免把 "e.g. foo"、"3.5 秒" 这类内部标点误当成句末。
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[。！？；!?;])\s*|(?<=[.])\s+(?=[\"'(]?[A-Z0-9])|\n+"
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    lang: str
    index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _budget_for(text: str, cfg: ChunkConfig) -> int:
    """按这一段的语言取字数预算(中文取英文的一半左右)。"""
    return cfg.budget(detect_language(text))


def _split_long_block(text: str, cfg: ChunkConfig) -> list[str]:
    """把超长段落按句子切成若干段,再贪心打包回不超过预算的块。"""
    max_chars = _budget_for(text, cfg)
    sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s and s.strip()]
    pieces: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            # 单句就超长(多见于没有标点的长串),硬切。
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(
                sentence[i : i + max_chars]
                for i in range(0, len(sentence), max_chars)
            )
        elif buf and len(buf) + len(sentence) > max_chars:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf}{sentence}"
    if buf:
        pieces.append(buf)
    return pieces


def _pack(units: list[str], cfg: ChunkConfig) -> list[str]:
    chunks: list[str] = []
    buf = ""
    budget = 0  # 当前块的字数预算,由块内第一段的语言决定
    for unit in units:
        unit_budget = _budget_for(unit, cfg)
        if buf and len(buf) + len(unit) + 1 > budget:
            chunks.append(buf)
            tail = buf[-cfg.overlap_chars :] if cfg.overlap_chars > 0 else ""
            budget = unit_budget
            # 重叠段按剩余空间裁剪,而不是"放不下就整个丢掉"。
            # 后面那种写法会自我否定:单位块一旦接近预算上限(超长段落按句子
            # 切出来正是这种情况),重叠就永远放不下,overlap_chars 配了也等于没配。
            room = budget - len(unit) - 1
            if tail and room > 0:
                buf = f"{tail[-room:]}\n{unit}"
            else:
                buf = unit
        else:
            if not buf:
                budget = unit_budget
            buf = f"{buf}\n{unit}" if buf else unit
    if buf:
        if len(buf) >= cfg.min_chars or not chunks:
            chunks.append(buf)
        else:
            # 尾块太短,并回上一块,避免丢掉文档结尾的信息。
            chunks[-1] = f"{chunks[-1]}\n{buf}"
    return chunks


def chunk_blocks(blocks: list[str], cfg: ChunkConfig | None = None) -> list[str]:
    cfg = cfg or ChunkConfig()
    units: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= _budget_for(block, cfg):
            units.append(block)
        else:
            units.extend(_split_long_block(block, cfg))
    return _pack(units, cfg)


def chunk_document(document: Document, cfg: ChunkConfig | None = None) -> list[Chunk]:
    texts = chunk_blocks(document.blocks, cfg)
    return [
        Chunk(
            chunk_id=f"{document.doc_id}#{i:03d}",
            doc_id=document.doc_id,
            source=document.source,
            lang=detect_language(text),
            index=i,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


def chunk_corpus(documents: list[Document], cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, cfg))
    return chunks
