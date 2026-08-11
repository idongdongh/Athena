"""上下文引擎契约。

结构对齐 Hermes ``agent/context_engine.py``，但当前只使用内置压缩器，
不引入第三方 context-engine 插件。
"""

from abc import ABC, abstractmethod
from typing import Any


class ContextEngine(ABC):
    """管理上下文压力与压缩生命周期的最小接口。"""

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """上下文引擎名称。"""

    @abstractmethod
    def update_from_response(self, usage: dict[str, Any]) -> None:
        """用一次模型响应的归一化 usage 更新 token 状态。"""

    @abstractmethod
    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        """当前上下文是否应该压缩。"""

    @abstractmethod
    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """返回压缩后的消息；失败时必须返回原消息对象。"""

    def should_compress_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        rough_tokens: int | None = None,
    ) -> bool:
        """API 调用前的粗估检查。"""
        return False

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        return True

    def on_session_reset(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
