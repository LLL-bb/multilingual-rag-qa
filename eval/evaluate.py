"""检索质量评测。

对比三种方案在同一个语料、同一批问题上的表现:

    baseline       单一多语言模型 + 单索引(对照组)
    routed         语言感知嵌入,按查询语种路由到对应单语子索引
    cross_lingual  语言感知嵌入,同时查两个子索引,等权 RRF 融合
    hybrid         同上,但给查询语种那一路加权(--primary-weight)

结果按问题类型拆成两档:

    same   查询语种与答案所在文档的语种一致
    cross  不一致(中文提问、答案只在英文文档里,或反过来)

实测结论不是"某个方案全面胜出",而是两条曲线之间的取舍:

    routed          same 上接近满分,跨语言直接归零
    cross_lingual  跨语言拉满,但要付出一部分同语种 R@1
    hybrid          用 --primary-weight 在两者之间移动

所以 hybrid 的权重没有"最优值",取决于你的业务里跨语言查询占多大比例。
README 里的权衡曲线就是扫这个参数扫出来的。

评测**只测检索**,不调用大模型 —— 否则生成的随机性会污染指标。

用法:
    python eval/evaluate.py
    python eval/evaluate.py --modes routed,cross_lingual   # 跳过 470MB 的多语言基线模型
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap  # noqa: E402

PROJECT_ROOT = bootstrap()

from rag.config import RetrieveConfig  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

MODES = ("baseline", "routed", "cross_lingual", "hybrid")
MODE_LABELS = {
    "baseline": "多语言单模型",
    "routed": "语言路由",
    "cross_lingual": "等权 RRF",
    "hybrid": "加权 RRF",
}
KINDS = ("same", "cross")
KIND_LABELS = {"same": "同语种", "cross": "跨语言", "all": "合计"}


def load_qa_set(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def document_languages(pipeline: RAGPipeline) -> dict[str, str]:
    """每篇文档的主语种,用来判断某个问题是同语种还是跨语言。"""
    counts: dict[str, Counter] = defaultdict(Counter)
    for chunk in pipeline.index.chunks:
        counts[chunk.doc_id][chunk.lang] += 1
    return {doc: counter.most_common(1)[0][0] for doc, counter in counts.items()}


def run_mode(
    corpus_dir: Path,
    qa_set: list[dict],
    mode: str,
    top_k: int,
    primary_weight: float = 1.5,
) -> list[dict]:
    pipeline = RAGPipeline.build(
        corpus_dir, mode=mode, retrieve_cfg=RetrieveConfig(mode=mode, primary_weight=primary_weight)
    )
    languages = document_languages(pipeline)
    rows = []
    for item in qa_set:
        gold = set(item["gold"])
        result = pipeline.retrieve(item["query"], top_k=top_k)
        rank = next(
            (i for i, hit in enumerate(result.hits, start=1) if hit.doc_id in gold),
            None,
        )
        gold_lang = languages.get(next(iter(gold)), "?")
        rows.append(
            {
                "query": item["query"],
                "lang": item["lang"],
                "kind": "same" if gold_lang == item["lang"] else "cross",
                "gold": sorted(gold),
                "rank": rank,
                "top1": result.hits[0].doc_id if result.hits else None,
                "retrieved": [h.doc_id for h in result.hits],
            }
        )
    return rows


def summarize(rows: list[dict], top_k: int) -> dict:
    out: dict[str, dict] = {}
    for kind in (*KINDS, "all"):
        subset = rows if kind == "all" else [r for r in rows if r["kind"] == kind]
        if not subset:
            continue
        n = len(subset)
        out[kind] = {
            "n": n,
            "recall@1": sum(1 for r in subset if r["rank"] == 1) / n,
            f"recall@{top_k}": sum(
                1 for r in subset if r["rank"] is not None and r["rank"] <= top_k
            )
            / n,
            "mrr": sum(1.0 / r["rank"] for r in subset if r["rank"]) / n,
        }
    return out


def render(results: dict[str, dict]) -> str:
    lines = []
    header = f"{'方案':<22}{'子集':<10}{'n':>4}{'R@1':>9}{'R@k':>9}{'MRR':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for mode in MODES:
        if mode not in results:
            continue
        summary = results[mode]
        for kind in (*KINDS, "all"):
            if kind not in summary:
                continue
            row = summary[kind]
            recall_k = next(v for k, v in row.items() if k.startswith("recall@") and k != "recall@1")
            label = f"{MODE_LABELS[mode]}" if kind == "same" else ""
            lines.append(
                f"{label:<22}{KIND_LABELS[kind]:<10}{row['n']:>4}"
                f"{row['recall@1']:>9.3f}{recall_k:>9.3f}{row['mrr']:>9.3f}"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--corpus", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--qa", default=str(PROJECT_ROOT / "eval" / "qa_set.jsonl"))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--primary-weight",
        type=float,
        default=1.5,
        help="hybrid 模式下给查询语种那一路的权重",
    )
    parser.add_argument("--json", default=None, help="把逐条结果写到指定文件")
    parser.add_argument("--show-misses", action="store_true", help="打印失败案例")
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        parser.error(f"未知模式 {unknown},可用: {list(MODES)}")

    corpus_dir = Path(args.corpus)
    qa_set = load_qa_set(Path(args.qa))
    print(f"语料: {corpus_dir}")
    print(f"问题: {len(qa_set)} 条  (同语种 "
          f"{sum(1 for q in qa_set if q['lang'] == 'zh')} 中文 / "
          f"{sum(1 for q in qa_set if q['lang'] == 'en')} 英文)")
    print(f"评测指标: recall@1 / recall@{args.top_k} / MRR @ top-{args.top_k}\n")

    results: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}
    for mode in modes:
        print(f"  构建索引: {mode} ...", flush=True)
        rows = run_mode(corpus_dir, qa_set, mode, args.top_k, args.primary_weight)
        details[mode] = rows
        results[mode] = summarize(rows, args.top_k)

    print()
    print(render(results))

    if args.show_misses:
        for mode, rows in details.items():
            misses = [r for r in rows if r["kind"] == "cross" and r["rank"] is None]
            if misses:
                print(f"{MODE_LABELS[mode]} 跨语言漏检 {len(misses)} 条:")
                for r in misses:
                    print(f"  [{r['lang']}] {r['query']}")
                    print(f"      期望 {r['gold']} / 实际 {r['retrieved']}")
                print()

    if args.json:
        Path(args.json).write_text(
            json.dumps({"results": results, "details": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"逐条结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
