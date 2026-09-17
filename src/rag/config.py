"""集中配置。所有影响复现结果的超参数都放在这里。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"

LANG_ZH = "zh"
LANG_EN = "en"
SUPPORTED_LANGS = (LANG_ZH, LANG_EN)

# 语言感知嵌入:中英各自使用在该语言上表现更好的单语模型。
ZH_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EN_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# 对照组:单一多语言模型,作为"不做语言路由"的基线。
MULTILINGUAL_EMBED_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


@dataclass
class ChunkConfig:
    """分块参数。

    字数预算按语种分开,而不是共用一个上限。理由:中文字符的信息密度
    约为英文的两倍,同样的 400 字,中文段落承载的内容远多于英文段落。
    用同一个上限会导致中文块过大、英文块过碎,两边的检索粒度不可比。

    中文预算取英文的一半左右,换算成 token 量级后两者才大致相当。
    """

    max_chars_zh: int = 200
    max_chars_en: int = 420
    overlap_chars: int = 60
    min_chars: int = 40

    def budget(self, lang: str) -> int:
        return self.max_chars_zh if lang == LANG_ZH else self.max_chars_en


@dataclass
class EmbedConfig:
    zh_model: str = ZH_EMBED_MODEL
    en_model: str = EN_EMBED_MODEL
    multilingual_model: str = MULTILINGUAL_EMBED_MODEL
    batch_size: int = 32
    device: str | None = None  # None 表示交给 sentence-transformers 自动选择
    # BGE 系列在检索场景下要求给查询加指令前缀(文档侧不加),否则召回会明显变差。
    query_instruction_zh: str = "为这个句子生成表示以用于检索相关文章："
    query_instruction_en: str = "Represent this sentence for searching relevant passages: "


@dataclass
class RetrieveConfig:
    """检索参数。

    mode:
        baseline      —— 单一多语言模型 + 单索引(对照组)
        routed        —— 按查询语种路由到对应单语子索引
        cross_lingual —— 同时查询两个子索引,等权 RRF 融合
        hybrid        —— 同时查询两个子索引,给查询语种那一路加权后再融合

    rrf_k 是 RRF 的平滑常数(原论文取 60)。RRF 只依赖排名、不比较分数,
    正好绕开"中文模型与英文模型的余弦分数不可比"这个问题。

    primary_weight 只在 hybrid 模式下生效。它调节的是同语种精度与跨语言召回
    之间的取舍:权重越大越信任查询语种那一路,同语种排名越稳,但跨语言文档
    越难被捞回来。没有哪个取值能同时最优,见 README 的权衡曲线。
    """

    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    mode: str = "cross_lingual"
    primary_weight: float = 1.5
    low_confidence_threshold: float = 0.35


@dataclass
class LLMConfig:
    """生成参数。backend="none" 时只做检索,不调用任何大模型。"""

    backend: str = "none"  # none | ollama
    model: str = "qwen2.5:7b"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout: int = 120
