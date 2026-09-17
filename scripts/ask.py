"""提问。

默认只做检索、不生成答案(``--llm none``),这样不需要装任何模型服务就能看到效果。
要生成答案就接本地 Ollama:

    ollama pull qwen2.5:7b
    python scripts/ask.py "图书馆周末几点开门?" --llm ollama

用法:
    python scripts/ask.py "本科生一次最多能借几本书?"
    python scripts/ask.py "What time does the library close on weekends?" --mode routed
    python scripts/ask.py "食堂能用现金付款吗?" --llm ollama --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap  # noqa: E402

PROJECT_ROOT = bootstrap()

from rag.config import LLMConfig  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

CONFIDENCE_NOTE = "低 —— 检索结果可能不相关,已指示模型允许回答「无法确定」"


def _load(args) -> RAGPipeline:
    index_dir = Path(args.index)
    if args.rebuild or not (index_dir / "pipeline.json").exists():
        pipeline = RAGPipeline.build(Path(args.corpus), mode=args.mode)
        pipeline.save(index_dir)
        return pipeline
    return RAGPipeline.load(index_dir, llm_cfg=LLMConfig(backend=args.llm, model=args.model))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="向 RAG 系统提问")
    parser.add_argument("question")
    parser.add_argument("--corpus", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--index", default=str(PROJECT_ROOT / "data" / "index"))
    parser.add_argument(
        "--mode",
        default="cross_lingual",
        choices=["baseline", "routed", "cross_lingual", "hybrid"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", default="none", choices=["none", "ollama"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--rebuild", action="store_true", help="忽略已有索引,重新构建")
    parser.add_argument("--show-prompt", action="store_true", help="打印送给大模型的提示词")
    args = parser.parse_args(argv)

    pipeline = _load(args)
    pipeline.retrieve_cfg.top_k = args.top_k

    payload = pipeline.answer(args.question, top_k=args.top_k)
    fused = len(payload["spaces"]) > 1
    score_label = "rrf" if fused else "cosine"

    print(f"问题     {payload['query']}")
    print(f"查询语种  {payload['query_lang']}")
    print(f"查询意图  {payload['intent']}")
    print(f"检索空间  {payload['spaces']}")
    confidence = payload["confidence"]
    flag = f"  ({CONFIDENCE_NOTE})" if payload["low_confidence"] else ""
    print(f"置信度    {confidence:.3f}{flag}")
    print()

    if not payload["hits"]:
        print("没有检索到任何结果。")
        return 0

    print(f"检索结果 top-{len(payload['hits'])}  (分数列 = {score_label})")
    for i, hit in enumerate(payload["hits"], start=1):
        snippet = " ".join(hit.text.split())[:88]
        print(f"  [{i}] {hit.score:>8.4f}  {hit.source}  ({hit.lang})")
        print(f"        {snippet}...")
    print()

    print("出处")
    print(payload["sources"])
    print()

    if payload["answer"]:
        print(f"答案  (由 {payload['llm']} 生成)")
        print(payload["answer"])
    else:
        print("未启用大模型后端,以上仅为检索结果。")
        print("要生成答案:python scripts/ask.py \"...\" --llm ollama")

    if args.show_prompt and payload["prompt"]:
        print()
        print("=" * 60)
        print(payload["prompt"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
