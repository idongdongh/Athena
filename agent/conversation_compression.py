"""对话压缩编排。

对应 Hermes ``agent/conversation_compression.py`` 的 Agent 内核职责：调用压缩器、
保证失败时历史不变，并在成功后原子替换唯一的 messages 列表。
"""

from dataclasses import dataclass
from typing import Any

from agent.context_compressor import ContextCompressor
from agent.context_state import ContextState, estimate_request_tokens_rough


@dataclass(frozen=True)
class CompressionResult:
    changed: bool
    before_messages: int
    after_messages: int
    estimated_tokens: int
    error: str | None = None


def compress_context(
    compressor: ContextCompressor,
    context_state: ContextState,
    messages: list[dict[str, Any]],
    *,
    system: str,
    tools: list[dict[str, Any]],
    current_tokens: int | None = None,
) -> CompressionResult:
    """执行一次压缩；只有成功产生新上下文后才原子替换 ``messages``。"""
    before_count = len(messages)
    compressed = compressor.compress(messages, current_tokens=current_tokens)
    if compressed is messages or compressor._last_compress_aborted:
        return CompressionResult(
            changed=False,
            before_messages=before_count,
            after_messages=before_count,
            estimated_tokens=current_tokens or 0,
            error=compressor._last_summary_error,
        )

    estimated_tokens = estimate_request_tokens_rough(
        system=system,
        messages=compressed,
        tools=tools,
    )
    # compressor 内部只能估算 messages；编排层拥有 system + tools，使用完整请求
    # 粗估校准 Hermes 的 preflight 防重复压缩基线。
    compressor.last_compression_rough_tokens = estimated_tokens
    # history 是唯一消息真相源：保留 list 对象身份，只替换其内容。
    messages[:] = compressed
    context_state.mark_compressed(estimated_tokens)
    return CompressionResult(
        changed=True,
        before_messages=before_count,
        after_messages=len(messages),
        estimated_tokens=estimated_tokens,
    )
