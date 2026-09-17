from rag.language import cjk_ratio, describe, detect_language, has_letters

from rag.config import LANG_EN, LANG_ZH


def test_pure_chinese():
    assert detect_language("图书馆借阅规则") == LANG_ZH


def test_pure_english():
    assert detect_language("Library borrowing rules") == LANG_EN


def test_chinese_with_english_terms_still_chinese():
    # 中文技术文档里夹英文术语是常态,不应因此判成英文
    assert detect_language("用 SQL 查询 MySQL 中的数据表") == LANG_ZH


def test_english_with_one_chinese_word_stays_english():
    assert detect_language("The course handbook mentions 学分 requirements in detail") == LANG_EN


def test_cjk_ratio_bounds():
    assert cjk_ratio("中文") == 1.0
    assert cjk_ratio("english") == 0.0


def test_text_without_letters_falls_back_to_english():
    # 已知局限:纯数字/符号无法判定,按默认分支归为 en
    assert not has_letters("12345 !@#$%")
    assert detect_language("12345 !@#$%") == LANG_EN


def test_describe_reports_evidence():
    info = describe("图书馆 borrowing")
    assert info["lang"] == LANG_ZH
    assert 0.0 < info["cjk_ratio"] < 1.0
