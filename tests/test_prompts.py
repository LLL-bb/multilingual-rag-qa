from rag.index import Hit
from rag.prompts import (
    COMPARE,
    FACTUAL,
    SUMMARY,
    build_prompt,
    classify_query,
    format_sources,
)


def _hit(chunk_id: str, lang: str, source: str, text: str) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=source.split(".")[0],
        source=source,
        lang=lang,
        text=text,
        score=0.5,
        space=lang,
    )


def test_classify_summary():
    assert classify_query("总结一下宿舍管理规定") == SUMMARY
    assert classify_query("Summarize the course handbook") == SUMMARY


def test_classify_compare():
    assert classify_query("国家奖学金和校级奖学金有什么区别?") == COMPARE
    assert classify_query("Compare the two library policies") == COMPARE


def test_classify_defaults_to_factual():
    assert classify_query("本科生一次最多能借几本书?") == FACTUAL
    assert classify_query("What time does the library close?") == FACTUAL


def test_prompt_numbers_sources_from_one():
    hits = [_hit("a#000", "zh", "a.md", "内容一"), _hit("b#000", "en", "b.md", "content two")]
    prompt = build_prompt("问题", hits, "zh")
    assert "[1] 来源: a.md" in prompt
    assert "[2] 来源: b.md" in prompt
    assert "内容一" in prompt


def test_low_confidence_allows_abstaining():
    hits = [_hit("a#000", "zh", "a.md", "内容")]
    normal = build_prompt("问题", hits, "zh", low_confidence=False)
    low = build_prompt("问题", hits, "zh", low_confidence=True)
    assert "根据现有资料无法确定" in low
    assert "根据现有资料无法确定" not in normal


def test_cross_lingual_hits_add_a_language_rule():
    mono = [_hit("a#000", "zh", "a.md", "内容")]
    mixed = mono + [_hit("b#000", "en", "b.md", "content")]
    assert "英文材料" not in build_prompt("问题", mono, "zh")
    assert "英文材料" in build_prompt("问题", mixed, "zh")


def test_prompt_language_follows_question_language():
    hits = [_hit("a#000", "en", "a.md", "content")]
    assert "## Answer" in build_prompt("What is X?", hits, "en")
    assert "【回答】" in build_prompt("X 是什么?", hits, "zh")


def test_format_sources_deduplicates():
    hits = [
        _hit("a#000", "zh", "a.md", "1"),
        _hit("a#001", "zh", "a.md", "2"),
        _hit("b#000", "en", "b.md", "3"),
    ]
    out = format_sources(hits)
    assert out.count("a.md") == 1
    assert "b.md" in out
