"""动态 Prompt Engineering。

"动态"在这里不是形容词,而是三个可判定的信号 —— 每个都会实质改变送进模型的文本:

1. **查询意图**:事实查找 / 摘要归纳 / 对比。三种意图需要完全不同的作答结构。
2. **检索置信度**:最高分低于阈值时,检索结果大概率不相关。这时必须在指令里
   显式允许模型回答"资料不足"。不写这句,模型几乎一定会硬编一个答案 ——
   这是 RAG 里幻觉最主要的来源。
3. **命中结果的语种分布**:命中跨语种时,要额外要求"用提问的语言作答、
   并标注引用来源",否则模型会跟着资料的语言跑(中文提问答出英文)。

用启发式规则而不是再训一个分类器来判意图:规则可解释、零延迟、零依赖,
而且在本项目的查询分布上足够准。
"""

from __future__ import annotations

from typing import Sequence

from .index import Hit

LANG_NAME = {"zh": "中文", "en": "英文"}

FACTUAL, SUMMARY, COMPARE = "factual", "summary", "compare"

_SUMMARY_CUES = (
    "总结", "概括", "摘要", "归纳", "综述", "大意", "讲了什么", "说了什么",
    "summar", "overview", "tl;dr", "gist", "what is this about",
)
_COMPARE_CUES = (
    "区别", "对比", "比较", "差异", "相比", "哪个更", "不同",
    "difference", "compare", "versus", " vs ", "distinguish",
)


def classify_query(query: str) -> str:
    lowered = query.lower()
    if any(cue in lowered for cue in _SUMMARY_CUES):
        return SUMMARY
    if any(cue in lowered for cue in _COMPARE_CUES):
        return COMPARE
    return FACTUAL


# 按意图给不同的作答结构
_INTENT_RULES = {
    "zh": {
        FACTUAL: "- 直接给出答案,并在句末用 [编号] 标注依据的资料。",
        SUMMARY: "- 分点归纳资料要点,每点用 [编号] 标注来源,不要逐字照抄。",
        COMPARE: "- 按维度逐条对比,明确指出各项差异分别来自哪份资料 [编号]。",
    },
    "en": {
        FACTUAL: "- Answer directly, citing the supporting source with [n] at the end of the sentence.",
        SUMMARY: "- Summarize the key points as a list; cite [n] for each point; do not copy verbatim.",
        COMPARE: "- Compare dimension by dimension, stating which source [n] each difference comes from.",
    },
}

_FALLBACK_RULES = {
    "zh": {
        "strict": "- 资料不足时,直接回答「根据现有资料无法确定」,不要推测或编造。",
        "loose": "- 若资料未覆盖问题的某个方面,说明哪部分无法确定,不要推测。",
    },
    "en": {
        "strict": '- If the sources are insufficient, reply "cannot be determined from the provided '
        'sources" instead of guessing.',
        "loose": "- If the sources only partly cover the question, state which part is undetermined; "
        "do not guess.",
    },
}

# 命中结果跨语种时才加:否则模型会跟着资料的语言跑
_CROSS_LINGUAL_RULES = {
    "zh": "- 命中资料包含英文材料:请用中文作答,并在引用英文资料时注明其出处文件名。",
    "en": "- Some sources are in Chinese: answer in English and note the source filename when citing them.",
}

_GROUNDING_RULES = {
    "zh": "- 只依据资料作答,不要引入资料之外的知识。",
    "en": "- Use only the provided sources; do not rely on outside knowledge.",
}


def _answer_rules(
    intent: str, query_lang: str, cross_lingual: bool, low_confidence: bool
) -> list[str]:
    lang = "zh" if query_lang == "zh" else "en"
    rules = [
        _INTENT_RULES[lang][intent],
        _GROUNDING_RULES[lang],
        _FALLBACK_RULES[lang]["strict" if low_confidence else "loose"],
    ]
    if cross_lingual:
        rules.append(_CROSS_LINGUAL_RULES[lang])
    return rules


def build_prompt(
    query: str,
    hits: Sequence[Hit],
    query_lang: str,
    intent: str | None = None,
    low_confidence: bool = False,
) -> str:
    """把检索结果和动态指令组装成最终提示词。"""
    intent = intent or classify_query(query)
    cross_lingual = len({hit.lang for hit in hits}) > 1

    if query_lang == "zh":
        header = "你是一个严谨的资料问答助手。"
        rules_title = "【回答要求】"
        sources_title = "【资料】"
        question_title = "【问题】"
        answer_title = "【回答】"
    else:
        header = "You are a rigorous assistant that answers strictly from provided sources."
        rules_title = "## Requirements"
        sources_title = "## Sources"
        question_title = "## Question"
        answer_title = "## Answer"

    lines = [header, "", rules_title]
    lines.extend(_answer_rules(intent, query_lang, cross_lingual, low_confidence))

    lines += ["", sources_title]
    for i, hit in enumerate(hits, start=1):
        lang_label = LANG_NAME.get(hit.lang, hit.lang)
        if query_lang == "zh":
            lines.append(f"[{i}] 来源: {hit.source}（{lang_label}）")
        else:
            lines.append(f"[{i}] source: {hit.source} ({lang_label})")
        lines.append(hit.text.strip())
        lines.append("")

    lines += [question_title, query, "", answer_title]
    return "\n".join(lines)


def format_sources(hits: Sequence[Hit]) -> str:
    """给命令行/界面用的出处列表。"""
    seen: set[str] = set()
    rows: list[str] = []
    for hit in hits:
        if hit.source in seen:
            continue
        seen.add(hit.source)
        rows.append(f"  - {hit.source}  ({LANG_NAME.get(hit.lang, hit.lang)})")
    return "\n".join(rows)
