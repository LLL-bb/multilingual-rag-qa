from rag.chunking import chunk_blocks, chunk_corpus, chunk_document
from rag.config import LANG_EN, LANG_ZH, ChunkConfig
from rag.loaders import Document


def test_chinese_budget_is_smaller_than_english():
    cfg = ChunkConfig()
    assert cfg.budget(LANG_ZH) < cfg.budget(LANG_EN)


def test_chinese_is_split_finer_than_english_at_same_length():
    """同样字符量下中文应切出更多块 —— 这正是按语种分预算的目的。"""
    cfg = ChunkConfig()
    chinese = "这是一个用于测试分块的中文句子。" * 40  # 640 字
    english = "This is a sentence used to test chunking. " * 15  # 630 字
    assert len(chunk_blocks([chinese], cfg)) > len(chunk_blocks([english], cfg))


def test_chunks_do_not_grossly_exceed_budget():
    cfg = ChunkConfig()
    chunks = chunk_blocks(["这是一个测试句子。" * 200], cfg)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= cfg.max_chars_zh + cfg.min_chars


def test_consecutive_chunks_share_overlap():
    cfg = ChunkConfig()
    blocks = ["句子编号甲乙丙丁戊己庚辛壬癸。" for _ in range(60)]
    chunks = chunk_blocks(blocks, cfg)
    assert len(chunks) >= 2
    # 下一块的起始行必须来自上一块的结尾 —— 这就是重叠
    assert chunks[1].split("\n")[0] in chunks[0]


def test_short_tail_is_merged_not_dropped():
    cfg = ChunkConfig()
    blocks = ["比较长的一个段落,用于把缓冲区填得差不多满。" * 8, "短尾。"]
    chunks = chunk_blocks(blocks, cfg)
    assert any("短尾。" in c for c in chunks)


def test_chunk_document_assigns_ids_and_language():
    cfg = ChunkConfig()
    doc = Document(
        doc_id="rules",
        source="rules.md",
        fmt="md",
        blocks=["图书馆借阅规则。本科生一次最多可借 10 册。" * 20],
    )
    chunks = chunk_document(doc, cfg)
    assert [c.chunk_id for c in chunks] == [f"rules#{i:03d}" for i in range(len(chunks))]
    assert all(c.lang == LANG_ZH for c in chunks)
    assert all(c.source == "rules.md" for c in chunks)


def test_chunk_corpus_covers_all_documents():
    docs = [
        Document(doc_id="a", source="a.txt", fmt="txt", blocks=["中文内容。" * 30]),
        Document(doc_id="b", source="b.txt", fmt="txt", blocks=["English content. " * 30]),
    ]
    chunks = chunk_corpus(docs)
    assert {c.doc_id for c in chunks} == {"a", "b"}
