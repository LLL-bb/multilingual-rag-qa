"""语言检测。

这里刻意**不**用 langdetect / fastText 这类统计模型,而是用 CJK 字符占比:

1. 无额外依赖、无模型下载,"克隆即可复现"才成立;
2. 本项目的语料是中英混合的短文本(一段话只有一两句),
   统计模型在这么短的文本上反而不稳;
3. 判定规则可解释、可单测。

代价是:纯数字/纯符号的文本会被判成 en。本项目的语料不会出现这种情况,
真遇到时调用方可以先用 ``has_letters`` 过滤。
"""

from __future__ import annotations

from .config import LANG_EN, LANG_ZH

# 基本汉字、扩展 A、兼容汉字
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))

# 中文语料里常混有英文术语(如 "GPA"、"SQL"),所以阈值取得较低:
# 只要有两成字符是汉字,就按中文处理。
DEFAULT_THRESHOLD = 0.20


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def cjk_ratio(text: str) -> float:
    """汉字数占"汉字 + 拉丁字母"总数的比例。"""
    cjk = sum(1 for ch in text if is_cjk(ch))
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + latin
    return 0.0 if total == 0 else cjk / total


def has_letters(text: str) -> bool:
    """文本里是否含有可判定的字母(汉字或拉丁字母)。"""
    return any(is_cjk(ch) or (ch.isascii() and ch.isalpha()) for ch in text)


def detect_language(text: str, threshold: float = DEFAULT_THRESHOLD) -> str:
    """返回 ``"zh"`` 或 ``"en"``。"""
    return LANG_ZH if cjk_ratio(text) >= threshold else LANG_EN


def describe(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float | str]:
    """给出判定结果及其依据,便于调试。"""
    ratio = cjk_ratio(text)
    return {
        "lang": LANG_ZH if ratio >= threshold else LANG_EN,
        "cjk_ratio": round(ratio, 4),
        "threshold": threshold,
    }
