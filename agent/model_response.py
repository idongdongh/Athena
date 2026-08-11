"""把 Anthropic 原始响应转换为 Agent 使用的稳定语义。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelStopReason(str, Enum):
    """模型停止生成的归一化原因。"""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    PAUSE_TURN = "pause_turn"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TokenUsage:
    """一次模型请求的 token 用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    estimated: bool = False

    @property
    def effective_input_tokens(self) -> int:
        """本次请求实际占用的完整输入，包括缓存命中与缓存创建部分。"""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


@dataclass(frozen=True)
class ModelResponseState:
    """Agent 主循环需要的模型响应状态。"""

    stop_reason: ModelStopReason
    raw_stop_reason: str | None
    usage: TokenUsage | None
    has_text: bool
    has_tool_calls: bool


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_count(value: Any) -> int:
    """只接受非负整数；Provider 的缺失或异常字段按 0 处理。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _extract_usage(response: Any) -> TokenUsage | None:
    usage = _field(response, "usage")
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=_token_count(_field(usage, "input_tokens", 0)),
        output_tokens=_token_count(_field(usage, "output_tokens", 0)),
        cache_creation_input_tokens=_token_count(
            _field(usage, "cache_creation_input_tokens", 0)
        ),
        cache_read_input_tokens=_token_count(
            _field(usage, "cache_read_input_tokens", 0)
        ),
    )


def inspect_model_response(response: Any) -> ModelResponseState:
    """读取停止原因、内容类型和 usage，不让 Provider 字段散落到主循环。"""
    raw_reason = _field(response, "stop_reason")
    normalized_reason = (
        ModelStopReason(raw_reason)
        if isinstance(raw_reason, str)
        and raw_reason in ModelStopReason._value2member_map_
        else ModelStopReason.UNKNOWN
    )
    content = _field(response, "content", []) or []
    return ModelResponseState(
        stop_reason=normalized_reason,
        raw_stop_reason=raw_reason if isinstance(raw_reason, str) else None,
        usage=_extract_usage(response),
        has_text=any(
            _field(block, "type") == "text"
            and isinstance(_field(block, "text", ""), str)
            and bool(_field(block, "text", "").strip())
            for block in content
        ),
        has_tool_calls=any(_field(block, "type") == "tool_use" for block in content),
    )
