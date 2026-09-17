"""大模型后端。

默认 ``backend="none"``:不调用任何大模型,只做检索。

这不是偷懒,有两个实际理由:

1. 仓库在**零外部服务**的情况下就能跑通 —— 克隆下来立刻能看到结果;
2. 评测**检索**质量时本来就不该接生成模型,否则噪声会混进指标。

需要真正生成答案时用 Ollama(本地、免费、不需要 API key)::

    ollama pull qwen2.5:7b
    python scripts/ask.py "图书馆周末几点关门?" --llm ollama

用标准库 urllib 而不是 requests,是为了不再多一个依赖。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import LLMConfig


class NullLLM:
    """占位后端:只做检索,不生成。"""

    available = False
    name = "none"

    def generate(self, prompt: str) -> str:  # pragma: no cover - 不应被调用
        raise RuntimeError("当前未启用大模型后端,无法生成答案")


class OllamaLLM:
    available = True

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.name = f"ollama:{cfg.model}"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self.cfg.host.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"连接 Ollama 失败({self.cfg.host})。请确认 `ollama serve` 已启动、"
                f"且已执行 `ollama pull {self.cfg.model}`。原始错误: {exc}"
            ) from exc
        return (body.get("response") or "").strip()


def build_llm(cfg: LLMConfig | None = None):
    cfg = cfg or LLMConfig()
    if cfg.backend == "none":
        return NullLLM()
    if cfg.backend == "ollama":
        return OllamaLLM(cfg)
    raise ValueError(f"未知的 LLM 后端 {cfg.backend!r},可用: none | ollama")
