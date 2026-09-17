"""端到端流程:解析 → 分块 → 嵌入 → 建索引 → 检索 → 生成。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .chunking import Chunk, chunk_corpus
from .config import (
    DEFAULT_INDEX_DIR,
    ChunkConfig,
    EmbedConfig,
    LLMConfig,
    RetrieveConfig,
)
from .embedders import Embedder, build_embedder
from .index import Hit, VectorIndex, reciprocal_rank_fusion
from .language import detect_language
from .llm import build_llm
from .loaders import load_corpus
from .prompts import build_prompt, classify_query, format_sources


@dataclass
class RetrievalResult:
    query: str
    query_lang: str
    intent: str
    spaces: tuple[str, ...]
    hits: list[Hit] = field(default_factory=list)
    confidence: float = 0.0
    low_confidence: bool = False


class RAGPipeline:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        retrieve_cfg: RetrieveConfig | None = None,
        llm_cfg: LLMConfig | None = None,
    ):
        self.index = index
        self.embedder = embedder
        self.retrieve_cfg = retrieve_cfg or RetrieveConfig()
        self.llm = build_llm(llm_cfg or LLMConfig())

    # ---------- 构建 ----------

    @classmethod
    def build(
        cls,
        corpus_dir: Path,
        mode: str = "cross_lingual",
        chunk_cfg: ChunkConfig | None = None,
        embed_cfg: EmbedConfig | None = None,
        retrieve_cfg: RetrieveConfig | None = None,
        llm_cfg: LLMConfig | None = None,
    ) -> "RAGPipeline":
        retrieve_cfg = retrieve_cfg or RetrieveConfig()
        retrieve_cfg.mode = mode
        pipeline = cls(VectorIndex(()), build_embedder(mode, embed_cfg), retrieve_cfg, llm_cfg)
        documents = load_corpus(corpus_dir)
        pipeline.ingest(chunk_corpus(documents, chunk_cfg))
        return pipeline

    def ingest(self, chunks: Sequence[Chunk]) -> None:
        """按向量空间把块分组、嵌入、入索引。

        单空间(基线)时所有块进同一个索引;语言感知时按块自身语种分流。
        """
        index = VectorIndex(self.embedder.spaces)
        single_space = len(self.embedder.spaces) == 1
        for space in self.embedder.spaces:
            group = list(chunks) if single_space else [c for c in chunks if c.lang == space]
            if not group:
                continue
            vectors = self.embedder.encode_documents([c.text for c in group], space)
            index.add(space, vectors, group)
        index.build()
        self.index = index

    # ---------- 检索 ----------

    def _target_spaces(self, query_lang: str) -> tuple[str, ...]:
        if len(self.embedder.spaces) == 1:
            return tuple(self.embedder.spaces)
        if self.retrieve_cfg.mode == "routed":
            # 只查同语种子索引:同语种最准,但跨语言文档必然漏检。
            return (query_lang,)
        # cross_lingual 与 hybrid 都查两路,区别只在融合时是否给主语言加权。
        return tuple(self.embedder.spaces)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        cfg = self.retrieve_cfg
        query_lang = detect_language(query)
        spaces = self._target_spaces(query_lang)
        k = top_k or cfg.top_k

        per_space: dict[str, list[Hit]] = {}
        for space in spaces:
            vector = self.embedder.encode_query(query, space)
            per_space[space] = self.index.search(space, vector, cfg.candidate_k)

        # 置信度取各路的原始余弦最高分 —— 不能用融合后的 RRF 分,
        # 那个分数没有绝对含义,阈值定不了。
        confidence = max(
            (hits[0].score for hits in per_space.values() if hits), default=0.0
        )

        if len(spaces) == 1:
            hits = list(per_space[spaces[0]][:k])
        else:
            rankings = {s: [h.chunk_id for h in hs] for s, hs in per_space.items()}
            pool = {h.chunk_id: h for hs in per_space.values() for h in hs}
            # hybrid 给查询语种那一路加权,表达"更信任同语种的排名"。
            weights = (
                {s: cfg.primary_weight if s == query_lang else 1.0 for s in spaces}
                if cfg.mode == "hybrid"
                else None
            )
            fused = reciprocal_rank_fusion(rankings, cfg.rrf_k, weights)[:k]
            hits = [_rescore(pool[cid], score) for cid, score in fused]

        return RetrievalResult(
            query=query,
            query_lang=query_lang,
            intent=classify_query(query),
            spaces=spaces,
            hits=hits,
            confidence=confidence,
            low_confidence=confidence < cfg.low_confidence_threshold,
        )

    # ---------- 生成 ----------

    def answer(self, query: str, top_k: int | None = None) -> dict:
        result = self.retrieve(query, top_k)
        payload = {
            "query": query,
            "query_lang": result.query_lang,
            "intent": result.intent,
            "spaces": list(result.spaces),
            "confidence": round(result.confidence, 4),
            "low_confidence": result.low_confidence,
            "hits": result.hits,
            "sources": format_sources(result.hits),
            "llm": self.llm.name,
            "prompt": None,
            "answer": None,
        }
        if result.hits and self.llm.available:
            prompt = build_prompt(
                query,
                result.hits,
                result.query_lang,
                result.intent,
                result.low_confidence,
            )
            payload["prompt"] = prompt
            payload["answer"] = self.llm.generate(prompt)
        return payload

    # ---------- 持久化 ----------

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or DEFAULT_INDEX_DIR)
        self.index.save(directory)
        (directory / "pipeline.json").write_text(
            json.dumps(
                {
                    "mode": self.retrieve_cfg.mode,
                    "spaces": list(self.embedder.spaces),
                    "model_names": self.embedder.model_names,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(
        cls,
        directory: Path | None = None,
        llm_cfg: LLMConfig | None = None,
        embed_cfg: EmbedConfig | None = None,
    ) -> "RAGPipeline":
        directory = Path(directory or DEFAULT_INDEX_DIR)
        meta = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
        mode = meta["mode"]
        return cls(
            index=VectorIndex.load(directory),
            embedder=build_embedder(mode, embed_cfg),
            retrieve_cfg=RetrieveConfig(mode=mode),
            llm_cfg=llm_cfg,
        )

    def stats(self) -> dict:
        return {
            "mode": self.retrieve_cfg.mode,
            "spaces": list(self.embedder.spaces),
            "models": self.embedder.model_names,
            "chunks_per_space": self.index.stats(),
        }


def _rescore(hit: Hit, score: float) -> Hit:
    """RRF 融合后分数不再是余弦相似度,换掉分数、保留其余字段。"""
    return Hit(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        source=hit.source,
        lang=hit.lang,
        text=hit.text,
        score=score,
        space=hit.space,
    )
