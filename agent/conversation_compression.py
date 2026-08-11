"""对话压缩编排。

对应 Hermes ``agent/conversation_compression.py`` 的 Agent 内核职责：调用压缩器、
保证失败时历史不变，并在成功后原子替换唯一的 messages 列表。
"""

from dataclasses import dataclass
from typing import Any

from agent.context_compressor import ContextCompressor
from agent.context_state import ContextState, estimate_request_tokens_rough
from agent.session_runtime import SessionRuntime


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
    session_runtime: SessionRuntime | None = None,
    in_place: bool = False,
) -> CompressionResult:
    """执行一次压缩；只有成功产生新上下文后才原子替换 ``messages``。"""
    before_count = len(messages)
    compressor_state_before = {
        "compression_count": compressor.compression_count,
        "_previous_summary": compressor._previous_summary,
        "_last_compression_savings_pct": compressor._last_compression_savings_pct,
        "_ineffective_compression_count": compressor._ineffective_compression_count,
        "last_compression_rough_tokens": compressor.last_compression_rough_tokens,
        "last_prompt_tokens": compressor.last_prompt_tokens,
        "awaiting_real_usage_after_compression": (
            compressor.awaiting_real_usage_after_compression
        ),
    }
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
    if session_runtime is not None:
        try:
            session_runtime.persist_compression(
                messages,
                compressed,
                in_place=in_place,
            )
        except Exception as exc:
            for name, value in compressor_state_before.items():
                setattr(compressor, name, value)
            return CompressionResult(
                changed=False,
                before_messages=before_count,
                after_messages=before_count,
                estimated_tokens=current_tokens or 0,
                error=f"会话压缩持久化失败：{exc}",
            )
    # history 是唯一消息真相源：保留 list 对象身份，只替换其内容。
    messages[:] = compressed
    context_state.mark_compressed(estimated_tokens)
    return CompressionResult(
        changed=True,
        before_messages=before_count,
        after_messages=len(messages),
        estimated_tokens=estimated_tokens,
    )
