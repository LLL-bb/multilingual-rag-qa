"""嵌入模型,以及"语言感知"的路由逻辑。

核心设计:中文和英文各自使用在该语言上训练的单语模型,而不是一个多语言模型。

理由:单语模型只在一种语言上训练,词表和语义空间更专注;多语言模型为了
覆盖上百种语言,单语言精度是换来的。本项目的语料恰好只有中英两种语言,
所以精确路由的收益大于通用性的代价。

代价同样明确:**路由会切断跨语言检索**。中文查询走中文模型,而中文索引里
没有英文文档,于是"中文提问、答案在英文文档里"这种情况必然漏检。
这个代价由 ``index.py`` 的 RRF 融合来补 —— 见那里的说明。
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .config import (
    LANG_EN,
    LANG_ZH,
    SUPPORTED_LANGS,
    EmbedConfig,
)


class Embedder(Protocol):
    """嵌入模型的统一接口。

    ``spaces`` 是这套嵌入能产出的向量空间名字。语言感知嵌入有 ``("zh", "en")``
    两个空间;多语言基线只有一个 ``("multi",)``。检索层据此决定往哪些空间发查询。
    """

    spaces: tuple[str, ...]
    model_names: dict[str, str]

    def encode_documents(self, texts: Sequence[str], space: str) -> np.ndarray: ...

    def encode_query(self, text: str, space: str) -> np.ndarray: ...


class _ModelPool:
    """按需加载并缓存模型。

    延迟加载是为了让克隆者不必一次性下载全部权重 ——
    只跑中文语料时,英文模型永远不会被下载。
    """

    def __init__(self, device: str | None = None, batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._models: dict[str, object] = {}

    def get(self, name: str):
        if name not in self._models:
            from sentence_transformers import SentenceTransformer

            self._models[name] = SentenceTransformer(name, device=self.device)
        return self._models[name]

    def encode(self, name: str, texts: Sequence[str], prefix: str = "") -> np.ndarray:
        model = self.get(name)
        payload = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        vectors = model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # 归一化后内积等价于余弦相似度
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def dimension(self, name: str) -> int:
        return int(self.get(name).get_sentence_embedding_dimension())


class LanguageAwareEmbedder:
    """按语种路由到单语模型。"""

    spaces = SUPPORTED_LANGS  # ("zh", "en")

    def __init__(self, cfg: EmbedConfig | None = None):
        self.cfg = cfg or EmbedConfig()
        self._pool = _ModelPool(self.cfg.device, self.cfg.batch_size)
        self.model_names = {LANG_ZH: self.cfg.zh_model, LANG_EN: self.cfg.en_model}

    def _instruction(self, space: str) -> str:
        # BGE 系列只在查询侧加指令前缀,文档侧不加。漏掉或加错都会掉召回。
        return {
            LANG_ZH: self.cfg.query_instruction_zh,
            LANG_EN: self.cfg.query_instruction_en,
        }[space]

    def encode_documents(self, texts: Sequence[str], space: str) -> np.ndarray:
        self._check_space(space)
        return self._pool.encode(self.model_names[space], texts)

    def encode_query(self, text: str, space: str) -> np.ndarray:
        self._check_space(space)
        return self._pool.encode(
            self.model_names[space], [text], prefix=self._instruction(space)
        )[0]

    def dimension(self, space: str) -> int:
        return self._pool.dimension(self.model_names[space])

    def _check_space(self, space: str) -> None:
        if space not in self.spaces:
            raise ValueError(f"未知的向量空间 {space!r},可用: {self.spaces}")


class MultilingualEmbedder:
    """对照组:单一多语言模型,不做任何语言路由。"""

    spaces = ("multi",)

    def __init__(self, cfg: EmbedConfig | None = None):
        self.cfg = cfg or EmbedConfig()
        self._pool = _ModelPool(self.cfg.device, self.cfg.batch_size)
        self.model_names = {"multi": self.cfg.multilingual_model}

    def encode_documents(self, texts: Sequence[str], space: str = "multi") -> np.ndarray:
        self._check_space(space)
        return self._pool.encode(self.model_names["multi"], texts)

    def encode_query(self, text: str, space: str = "multi") -> np.ndarray:
        self._check_space(space)
        return self._pool.encode(self.model_names["multi"], [text])[0]

    def dimension(self, space: str = "multi") -> int:
        return self._pool.dimension(self.model_names["multi"])

    def _check_space(self, space: str) -> None:
        if space != "multi":
            raise ValueError(f"未知的向量空间 {space!r},可用: {self.spaces}")


def build_embedder(mode: str, cfg: EmbedConfig | None = None) -> Embedder:
    """按检索模式返回对应的嵌入器。"""
    if mode == "baseline":
        return MultilingualEmbedder(cfg)
    if mode in ("routed", "cross_lingual", "hybrid"):
        return LanguageAwareEmbedder(cfg)
    raise ValueError(f"未知的检索模式 {mode!r}")
