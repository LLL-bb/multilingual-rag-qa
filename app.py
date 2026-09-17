"""Streamlit 演示界面。

运行:
    pip install streamlit
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import bootstrap  # noqa: E402

PROJECT_ROOT = bootstrap()

import streamlit as st  # noqa: E402

from rag.config import LLMConfig  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

MODE_HELP = {
    "baseline": "单一多语言模型 + 单索引(对照组)",
    "routed": "按查询语种路由到对应单语子索引。同语种最准,跨语言会漏检。",
    "cross_lingual": "同时查两个子索引,等权 RRF 融合。跨语言召回高,同语种精度下降。",
    "hybrid": "同 cross_lingual,但给查询语种那一路加权。",
}


@st.cache_resource(show_spinner=False)
def load_pipeline(mode: str, llm_backend: str, llm_model: str) -> RAGPipeline:
    return RAGPipeline.build(
        PROJECT_ROOT / "data" / "raw",
        mode=mode,
        llm_cfg=LLMConfig(backend=llm_backend, model=llm_model),
    )


def render_hit(rank: int, hit, score_label: str) -> None:
    with st.expander(f"[{rank}] {hit.source}　·　{hit.lang}　·　{score_label} {hit.score:.4f}"):
        st.write(hit.text)


def main() -> None:
    st.set_page_config(page_title="中英混合文档 RAG", page_icon="🔎", layout="wide")
    st.title("中英混合文档 RAG 问答")
    st.caption(
        "语言感知嵌入:中文走 bge-small-zh,英文走 bge-small-en,"
        "分别建两个 FAISS 子索引。"
    )

    corpus_dir = PROJECT_ROOT / "data" / "raw"
    if not corpus_dir.exists():
        st.error("找不到示例语料,请先运行 `python scripts/build_demo_corpus.py`")
        return

    with st.sidebar:
        st.header("检索设置")
        mode = st.selectbox(
            "检索模式",
            ["baseline", "routed", "cross_lingual", "hybrid"],
            index=2,
            help="切换后会重建索引,首次加载需下载对应模型。",
        )
        st.caption(MODE_HELP[mode])
        top_k = st.slider("返回条数 top-k", 1, 10, 5)

        st.header("生成设置")
        llm_backend = st.selectbox("大模型后端", ["none", "ollama"], index=0)
        llm_model = st.text_input("模型名", "qwen2.5:7b", disabled=llm_backend == "none")
        if llm_backend == "none":
            st.caption("不接大模型,只显示检索结果。需要生成答案请选 ollama。")

    pipeline = load_pipeline(mode, llm_backend, llm_model)
    stats = pipeline.stats()

    with st.sidebar:
        st.header("索引状态")
        st.write(f"模式:`{stats['mode']}`")
        for space, name in stats["models"].items():
            st.write(f"`{space}` → {name}")
        st.write(f"分块数:{stats['chunks_per_space']}")

    question = st.text_input(
        "提问",
        value="",
        placeholder="例如:本科生一次最多能借几本书?　/　What time does the library close on weekends?",
    )

    if not question.strip():
        st.info("输入一个问题开始。试试中文问英文文档,或英文问中文文档 —— 那是这个项目最有意思的地方。")
        return

    payload = pipeline.answer(question, top_k=top_k)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("答案")
        if payload["answer"]:
            st.write(payload["answer"])
            st.caption(f"由 {payload['llm']} 生成")
        else:
            st.info("未启用大模型后端。下面是检索到的原始片段。")

    with col_right:
        st.subheader("检索诊断")
        st.write(f"查询语种:`{payload['query_lang']}`")
        st.write(f"查询意图:`{payload['intent']}`")
        st.write(f"检索空间:`{payload['spaces']}`")
        st.metric("置信度", f"{payload['confidence']:.3f}")
        if payload["low_confidence"]:
            st.warning("置信度偏低 —— 检索结果可能不相关。")

    st.subheader(f"检索结果 top-{len(payload['hits'])}")
    score_label = "RRF" if len(payload["spaces"]) > 1 else "余弦"
    for rank, hit in enumerate(payload["hits"], start=1):
        render_hit(rank, hit, score_label)

    if payload["prompt"]:
        with st.expander("送给大模型的提示词"):
            st.code(payload["prompt"], language="text")


if __name__ == "__main__":
    main()
