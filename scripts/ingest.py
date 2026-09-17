"""解析语料、建立索引并落盘。

用法:
    python scripts/ingest.py
    python scripts/ingest.py --mode routed --out data/index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap  # noqa: E402

PROJECT_ROOT = bootstrap()

from rag.pipeline import RAGPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="建立 RAG 索引")
    parser.add_argument("--corpus", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "index"))
    parser.add_argument(
        "--mode",
        default="cross_lingual",
        choices=["baseline", "routed", "cross_lingual", "hybrid"],
    )
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"语料目录不存在: {corpus_dir}")
        print("先运行: python scripts/build_demo_corpus.py")
        return 1

    pipeline = RAGPipeline.build(corpus_dir, mode=args.mode)
    out = pipeline.save(Path(args.out))

    stats = pipeline.stats()
    print(f"语料目录: {corpus_dir}")
    print(f"检索模式: {stats['mode']}")
    print(f"向量空间: {stats['spaces']}")
    for space, name in stats["models"].items():
        print(f"  {space:<6} {name}")
    print(f"分块数量: {stats['chunks_per_space']}")
    print(f"索引已写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
