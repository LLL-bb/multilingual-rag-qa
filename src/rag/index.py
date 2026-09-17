"""FAISS 向量索引,以及跨语言检索的 RRF 融合。

因为语言感知嵌入把中英文放进了**两个互不可比的向量空间**,
查询时就会碰到一个绕不开的问题:

    中文查询该只查中文索引,还是也查英文索引?

- 只查中文索引(``routed``):同语种检索最准,但"中文提问、答案在英文文档里"必然漏检。
- 两个都查(``cross_lingual``):能捞回跨语言文档,但两个模型给出的余弦分数
  量纲不同,直接比大小是错的。

这里用 RRF(Reciprocal Rank Fusion)解决第二个问题:它**只看排名、不看分数**,
天然免疫分数量纲不一致。代价是丢掉了分数的绝对信息,但在这种场景下,
"两路都排得靠前的结果更可信"这个假设足够稳。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    """一条检索结果。score 在 routed 模式下是余弦相似度,在融合模式下是 RRF 分。"""

    chunk_id: str
    doc_id: str
    source: str
    lang: str
    text: str
    score: float
    space: str


def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[str]],
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """把多路排名融成一个。``rankings`` 是 {空间名: [chunk_id 按相关性降序]}。

    RRF 分 = 各路上 ``weight / (rrf_k + rank)`` 之和。常数 rrf_k 用来压低头部
    结果的绝对优势,原论文取 60。

    ``weights`` 给某一路加权,用来表达"我更信这一路的排名"。加权后的分数仍然
    只由排名决定,不引入任何跨模型的分数比较 —— 这是 RRF 相对"直接比余弦分数"
    的核心优势,加权不能把这个性质弄丢。

    排序时用 chunk_id 做次级键,保证同分结果顺序稳定、评测可复现。
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    for space, ids in rankings.items():
        weight = weights.get(space, 1.0)
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class VectorIndex:
    """按向量空间分开建 FAISS 索引。

    - ``("zh", "en")``:语言感知嵌入,每篇文档只进它所属语种的索引;
      查询时跨语言模式会同时查两个空间。
    - ``("multi",)``:多语言基线,全部文档进同一个索引。

    小语料下用 IndexFlatIP 做精确检索 —— 这样评测结果不会混入
    近似检索的误差,对比才干净。
    """

    def __init__(self, spaces: Iterable[str]):
        self.spaces = tuple(spaces)
        self._faiss: dict[str, object] = {}
        self._space_ids: dict[str, list[str]] = {s: [] for s in self.spaces}
        self._vectors: dict[str, list[np.ndarray]] = {s: [] for s in self.spaces}
        self._chunks: dict[str, Chunk] = {}
        self._built = False

    # ---------- 构建 ----------

    def add(self, space: str, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        if space not in self.spaces:
            raise ValueError(f"未知的向量空间 {space!r},可用: {self.spaces}")
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) != len(chunks):
            raise ValueError(
                f"向量数({len(vectors)})与块数({len(chunks)})不一致"
            )
        self._vectors[space].append(vectors)
        for chunk in chunks:
            self._space_ids[space].append(chunk.chunk_id)
            self._chunks[chunk.chunk_id] = chunk
        self._built = False

    def build(self) -> "VectorIndex":
        import faiss

        for space in self.spaces:
            pending = self._vectors[space]
            if not pending:
                continue
            matrix = np.vstack(pending)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._faiss[space] = index
        self._built = True
        return self

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def stats(self) -> dict[str, int]:
        return {space: len(ids) for space, ids in self._space_ids.items()}

    # ---------- 检索 ----------

    def search(self, space: str, vector: np.ndarray, k: int) -> list[Hit]:
        index = self._faiss.get(space)
        if index is None or index.ntotal == 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        k = min(k, int(index.ntotal))
        scores, positions = index.search(query, k)
        hits: list[Hit] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            chunk = self._chunks[self._space_ids[space][position]]
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    lang=chunk.lang,
                    text=chunk.text,
                    score=float(score),
                    space=space,
                )
            )
        return hits

    # 多路检索的融合编排放在 pipeline 层 —— 那里还需要拿到各路的原始余弦分数
    # 来判断置信度,不适合藏在索引内部。这里只提供 RRF 这个纯函数原语。

    # ---------- 持久化 ----------

    def save(self, directory: Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for space in self.spaces:
            if space in self._faiss:
                faiss.write_index(self._faiss[space], str(directory / f"{space}.faiss"))
        meta = {
            "spaces": list(self.spaces),
            "space_ids": self._space_ids,
            "chunks": [asdict(c) for c in self._chunks.values()],
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        import faiss

        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        index = cls(meta["spaces"])
        index._space_ids = {s: list(meta["space_ids"].get(s, [])) for s in index.spaces}
        index._chunks = {c["chunk_id"]: Chunk(**c) for c in meta["chunks"]}
        for space in index.spaces:
            path = directory / f"{space}.faiss"
            if path.exists():
                index._faiss[space] = faiss.read_index(str(path))
        index._built = True
        return index
