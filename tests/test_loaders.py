from pathlib import Path

import pytest

from rag.loaders import load_corpus, load_document

DEMO_CORPUS = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_loads_text_split_by_blank_lines(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("第一段。\n\n第二段。", encoding="utf-8")
    doc = load_document(target, tmp_path)
    assert doc.blocks == ["第一段。", "第二段。"]
    assert doc.source == "a.txt"
    assert doc.fmt == "txt"
    assert doc.char_count == len("第一段。") + len("第二段。")


def test_utf8_bom_is_tolerated(tmp_path):
    target = tmp_path / "bom.txt"
    target.write_text("内容。", encoding="utf-8-sig")
    assert load_document(target, tmp_path).blocks == ["内容。"]


def test_gbk_encoded_text_is_decoded(tmp_path):
    target = tmp_path / "gbk.txt"
    target.write_bytes("中文内容。\n\n第二段。".encode("gb18030"))
    assert load_document(target, tmp_path).blocks[0] == "中文内容。"


def test_loads_docx_paragraphs(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("奖学金评定办法")
    document.add_paragraph("国家奖学金每人每年 8000 元。")
    target = tmp_path / "s.docx"
    document.save(str(target))
    blocks = load_document(target, tmp_path).blocks
    assert "奖学金评定办法" in blocks
    assert any("8000" in b for b in blocks)


def test_unsupported_suffix_raises(tmp_path):
    target = tmp_path / "a.csv"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(target, tmp_path)


def test_load_document_accepts_string_path(tmp_path):
    target = tmp_path / "a.md"
    target.write_text("正文。", encoding="utf-8")
    assert load_document(str(target), tmp_path).blocks == ["正文。"]


@pytest.mark.skipif(not DEMO_CORPUS.exists(), reason="示例语料未生成")
def test_demo_corpus_covers_all_four_formats():
    docs = load_corpus(DEMO_CORPUS)
    assert {d.fmt for d in docs} == {"md", "txt", "docx", "pdf"}
    assert len(docs) == 8
    assert all(d.blocks for d in docs), "每个文档都应解析出内容"


@pytest.mark.skipif(not DEMO_CORPUS.exists(), reason="示例语料未生成")
def test_demo_corpus_loading_is_ordered_and_repeatable():
    first = [d.source for d in load_corpus(DEMO_CORPUS)]
    second = [d.source for d in load_corpus(DEMO_CORPUS)]
    assert first == second == sorted(first)
