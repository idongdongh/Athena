"""会话级 token 统计与上下文压力状态。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent.model_response import ModelResponseState, TokenUsage


@dataclass(frozen=True)
class ContextSettings:
    """模型输出和上下文管理配置；非法字段回退到代码默认值。"""

    max_output_tokens: int = 4096
    context_window: int = 128_000
    compression_threshold: float = 0.75
    # 模型输出达到 max_output_tokens 后，最多额外请求模型续写 2 次
    max_length_continuations: int = 2
    compression_enabled: bool = True
    compression_target_ratio: float = 0.20
    protect_first_n: int = 3
    protect_last_n: int = 6
    abort_on_summary_failure: bool = True
    max_compressions_per_turn: int = 2
    compression_in_place: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "ContextSettings":
        defaults = cls()
        if not isinstance(config, Mapping):
            return defaults
        model = config.get("model")
        context = config.get("context")
        compression = config.get("compression")
        model = model if isinstance(model, Mapping) else {}
        context = context if isinstance(context, Mapping) else {}
        compression = compression if isinstance(compression, Mapping) else {}
        return cls(
            max_output_tokens=_positive_int(
                model.get("max_output_tokens"), defaults.max_output_tokens
            ),
            context_window=_positive_int(
                model.get("context_window"), defaults.context_window
            ),
            compression_threshold=_ratio(
                compression.get(
                    "threshold",
                    context.get("compression_threshold"),
                ),
                defaults.compression_threshold,
            ),
            max_length_continuations=_nonnegative_int(
                context.get("max_length_continuations"),
                defaults.max_length_continuations,
            ),
            compression_enabled=_boolean(
                compression.get("enabled"), defaults.compression_enabled
            ),
            compression_target_ratio=_ratio(
                compression.get("target_ratio"),
                defaults.compression_target_ratio,
            ),
            protect_first_n=_nonnegative_int(
                compression.get("protect_first_n"), defaults.protect_first_n
            ),
            protect_last_n=_nonnegative_int(
                compression.get("protect_last_n"), defaults.protect_last_n
            ),
            abort_on_summary_failure=_boolean(
                compression.get("abort_on_summary_failure"),
                defaults.abort_on_summary_failure,
            ),
            max_compressions_per_turn=_positive_int(
                compression.get("max_per_turn"),
                defaults.max_compressions_per_turn,
            ),
            compression_in_place=_boolean(
                compression.get("in_place"), defaults.compression_in_place
            ),
        )


# 以下解析函数只接受明确类型；非法 YAML 值统一回退到代码默认值。
def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value

def _nonnegative_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value

def _ratio(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    value = float(value)
    return value if 0.0 < value < 1.0 else default


def _boolean(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def estimate_tokens_rough(value: Any) -> int:
    """Hermes 同款粗估：按字符数向上取整，每 4 个字符约 1 token。"""
    text = value if isinstance(value, str) else str(value)
    if not text:
        return 0
    return (len(text) + 3) // 4


def _block_mapping(block: Any) -> Mapping[str, Any] | None:
    if isinstance(block, Mapping):
        return block
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _count_image_tokens(message: Mapping[str, Any], cost_per_image: int) -> int:
    """图片按固定成本计数，避免把 Base64 字符串误算成上下文文本。"""
    count = 0
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            block = _block_mapping(part)
            if block is not None and block.get("type") in {
                "image",
                "image_url",
                "input_image",
            }:
                count += 1
    stashed = message.get("_anthropic_content_blocks")
    if isinstance(stashed, list):
        for part in stashed:
            block = _block_mapping(part)
            if block is not None and block.get("type") == "image":
                count += 1
    if isinstance(content, Mapping) and content.get("_multimodal"):
        inner = content.get("content")
        if isinstance(inner, list):
            for part in inner:
                block = _block_mapping(part)
                if block is not None and block.get("type") in {
                    "image",
                    "image_url",
                }:
                    count += 1
    return count * cost_per_image


def _estimate_message_chars(message: Any) -> int:
    """计算消息字符数，但移除由固定 token 成本单独计算的图片数据。"""
    if not isinstance(message, Mapping):
        return len(str(message))
    shadow: dict[str, Any] = {}
    for key, value in message.items():
        if key == "_anthropic_content_blocks":
            continue
        if key != "content":
            shadow[key] = value
            continue
        if isinstance(value, list):
            cleaned: list[Any] = []
            for part in value:
                block = _block_mapping(part)
                if block is not None and block.get("type") in {
                    "image",
                    "image_url",
                    "input_image",
                }:
                    cleaned.append({"type": block.get("type"), "image": "[stripped]"})
                elif block is not None:
                    cleaned.append(dict(block))
                else:
                    cleaned.append(part)
            shadow[key] = cleaned
        elif isinstance(value, Mapping) and value.get("_multimodal"):
            shadow[key] = value.get("text_summary", "")
        else:
            shadow[key] = value
    return len(str(shadow))


def estimate_messages_tokens_rough(messages: list[dict[str, Any]] | Any) -> int:
    """Hermes preflight 消息估算，图片固定按约 1500 token 计算。"""
    if not isinstance(messages, list):
        return estimate_tokens_rough(messages)
    total_chars = 0
    image_tokens = 0
    for message in messages:
        total_chars += _estimate_message_chars(message)
        if isinstance(message, Mapping):
            image_tokens += _count_image_tokens(message, 1500)
    return ((total_chars + 3) // 4) + image_tokens


def estimate_request_tokens_rough(*, system: Any, messages: Any, tools: Any) -> int:
    """粗估完整请求，分别计入 system、messages 和工具 Schema。"""
    total = 0
    if system:
        total += estimate_tokens_rough(system)
    if messages:
        total += estimate_messages_tokens_rough(messages)
    if tools:
        total += estimate_tokens_rough(tools)
    return total


@dataclass
class ContextState:
    """只保存派生统计；消息内容仍以 conversation history 为唯一真相源。"""

    context_window: int
    compression_threshold: float
    last_usage: TokenUsage | None = None
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    _current_input_tokens: int = field(default=0, init=False, repr=False)
    _context_compressor: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: ContextSettings) -> "ContextState":
        return cls(
            context_window=settings.context_window,
            compression_threshold=settings.compression_threshold,
        )

    @property
    def usage_ratio(self) -> float:
        return self._current_input_tokens / self.context_window

    @property
    def should_compress(self) -> bool:
        return self.usage_ratio >= self.compression_threshold

    def estimate_request_tokens(self, *, system: Any, messages: Any, tools: Any) -> int:
        """粗估一次请求的完整输入，包括 system、历史消息和工具 schema。"""
        return estimate_request_tokens_rough(
            system=system,
            messages=messages,
            tools=tools,
        )

    def mark_compressed(self, estimated_input_tokens: int) -> None:
        """压缩后先使用粗估值，等待下一次 Provider usage 校准。"""
        self._current_input_tokens = max(0, estimated_input_tokens)

    def restore_session_totals(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        current_input_tokens: int,
    ) -> None:
        """恢复持久会话统计；当前压力使用恢复请求的粗估值。"""
        self.session_input_tokens = max(0, int(input_tokens))
        self.session_output_tokens = max(0, int(output_tokens))
        self._current_input_tokens = max(0, int(current_input_tokens))

    def update_from_response(
        self,
        response_state: ModelResponseState,
        *,
        system: Any,
        messages: Any,
        tools: Any,
        response_content: Any,
    ) -> TokenUsage:
        """用真实 usage 更新状态；Provider 未返回 usage 时使用本地估算。"""
        usage = response_state.usage
        if usage is None:
            usage = TokenUsage(
                input_tokens=self.estimate_request_tokens(
                    system=system,
                    messages=messages,
                    tools=tools,
                ),
                output_tokens=estimate_tokens_rough(response_content),
                estimated=True,
            )
        self.last_usage = usage
        self._current_input_tokens = usage.effective_input_tokens
        self.session_input_tokens += usage.effective_input_tokens
        self.session_output_tokens += usage.output_tokens
        return usage
