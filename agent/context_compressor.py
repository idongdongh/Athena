"""Hermes 风格的对话上下文压缩器。

保留内置压缩器的核心算法：旧工具结果预裁剪、首尾保护、中段结构化摘要、
迭代摘要、工具调用配对修复和无效压缩防抖。Provider 插件、多媒体和持久化
轮换属于外围能力，不在本模块实现。
"""

import copy
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from agent.context_engine import ContextEngine
from agent.context_state import estimate_tokens_rough

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SUMMARY_PREFIX = "[CONTEXT COMPACTION — REFERENCE ONLY]"
SUMMARY_END_MARKER = (
    "[END CONTEXT COMPACTION — continue from the protected recent messages below]"
)
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 60.0
_MAX_TAIL_MESSAGE_FLOOR = 20
_MIN_SUMMARY_TOKENS = 256
_MAX_SUMMARY_TOKENS = 8192

SummaryCallback = Callable[[str, int], str]

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _blocks(content: Any) -> list[Any]:
    return content if isinstance(content, list) else []


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_content_text(item) for item in content)))
    block_type = _field(content, "type")
    if block_type == "text":
        return str(_field(content, "text", ""))
    if block_type == "tool_use":
        name = _field(content, "name", "unknown")
        tool_input = _field(content, "input", {})
        return f"[TOOL CALL {name}] {json.dumps(tool_input, ensure_ascii=False, default=str)}"
    if block_type == "tool_result":
        return _content_text(_field(content, "content", ""))
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return ""


def _redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)(api"):
            redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and any(
        _field(block, "type") == "tool_result"
        for block in _blocks(message.get("content"))
    )


def _tool_use_blocks(message: dict[str, Any]) -> list[Any]:
    if message.get("role") != "assistant":
        return []
    return [
        block
        for block in _blocks(message.get("content"))
        if _field(block, "type") == "tool_use"
    ]


def _tool_result_blocks(message: dict[str, Any]) -> list[Any]:
    if message.get("role") != "user":
        return []
    return [
        block
        for block in _blocks(message.get("content"))
        if _field(block, "type") == "tool_result"
    ]


def _tool_use_id(block: Any) -> str:
    return str(_field(block, "id", "") or "")


def _tool_result_id(block: Any) -> str:
    return str(
        _field(block, "tool_use_id", "")
        or _field(block, "tool_call_id", "")
        or ""
    )


def _summary_body(content: Any) -> str:
    text = _content_text(content)
    start = text.find(SUMMARY_PREFIX)
    if start < 0:
        return ""
    body = text[start + len(SUMMARY_PREFIX):]
    end = body.find(SUMMARY_END_MARKER)
    if end >= 0:
        body = body[:end]
    return body.strip()


def _is_summary_message(message: dict[str, Any]) -> bool:
    return SUMMARY_PREFIX in _content_text(message.get("content"))


def _message_role(message: dict[str, Any]) -> str:
    role = message.get("role")
    return role if role in {"user", "assistant"} else "user"


def _make_summary_message(role: str, summary: str) -> dict[str, Any]:
    if role == "assistant":
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": summary}],
        }
    return {"role": "user", "content": summary}


def _prepend_content(content: Any, prefix: str) -> Any:
    if isinstance(content, str):
        return f"{prefix}\n\n{content}" if content else prefix
    if isinstance(content, list):
        return [{"type": "text", "text": prefix}, *content]
    return prefix


def _replace_block_content(block: Any, content: str) -> dict[str, Any]:
    if isinstance(block, dict):
        return {**block, "content": content}
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return {**dumped, "content": content}
    return {
        "type": "tool_result",
        "tool_use_id": _tool_result_id(block),
        "content": content,
    }


def _block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block.copy()
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {
        "type": _field(block, "type", "tool_use"),
        "id": _field(block, "id", ""),
        "name": _field(block, "name", ""),
        "input": _field(block, "input", {}),
    }


def _shrink_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 500:
        return value[:300] + f"...[{len(value) - 400} chars omitted]..." + value[-100:]
    if isinstance(value, dict):
        return {key: _shrink_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shrink_value(item) for item in value]
    return value


class ContextCompressor(ContextEngine):
    """默认上下文引擎，通过有损摘要压缩对话中段。"""

    @property
    def name(self) -> str:
        return "compressor"

    def __init__(
        self,
        *,
        # 上下文窗口对应配置里面的 context_window
        context_length: int,
        # 单次调用模型允许输出的最大 token 数：max_output_tokens
        max_tokens: int,
        # 压缩比例（输入 token 数占上下文窗口的比例）：threshold
        threshold_percent: float = 0.75,
        # 首次压缩时开头保留的消息条数：protect_first_n
        protect_first_n: int = 3,
        # 压缩时至少保留最近的消息条数：protect_last_n
        protect_last_n: int = 6,
        # 压缩时最近消息允许使用的 token 预算比例：target_ratio
        summary_target_ratio: float = 0.20,
        # 模型压缩失败时，是放弃压缩还是使用本地兜底摘要：abort_on_summary_failure
        abort_on_summary_failure: bool = True,

        # 调用模型生成摘要的函数：_generate_context_summary
        summary_callback: SummaryCallback | None = None,
    ) -> None:
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = summary_target_ratio
        self.abort_on_summary_failure = abort_on_summary_failure
        self.summary_callback = summary_callback
        # 要预留 max_tokens 给模型输出
        effective_input_window = max(1, context_length - max_tokens)
        # 实际压缩触发 token 阈值
        self.threshold_tokens = max(
            1,
            int(effective_input_window * threshold_percent),
        )
        # 压缩时，最近对话允许占用的大致 token 数
        self.tail_token_budget = max(
            1,
            int(self.threshold_tokens * summary_target_ratio),
        )
        # 摘要 token 预算，一般为上下文窗大小的 5%，最低 256，最高 8192
        self.max_summary_tokens = min(
            max(_MIN_SUMMARY_TOKENS, int(context_length * 0.05)),
            _MAX_SUMMARY_TOKENS,
        )
        # 最近一次模型请求实际消耗的输入 token
        self.last_prompt_tokens = 0
        # 最近一次模型回复产生的输出 token
        self.last_completion_tokens = 0
        # 最近一次请求的总 token
        self.last_total_tokens = 0
        # 最近一次由 Provider 返回的真实输入 token，来自 usage 的值
        self.last_real_prompt_tokens = 0
        # 最近一次压缩完成后，本地估算的新上下文 token 数
        self.last_compression_rough_tokens = 0 
        # 
        self.last_rough_tokens_when_real_prompt_fit = 0
        # 刚完成压缩，目前只有本地估算，True 表示正在等待下一次模型响应返回真实 usage
        self.awaiting_real_usage_after_compression = False
        # 当前会话成功压缩次数
        self.compression_count = 0
        # 上一轮生成的摘要
        self._previous_summary: str | None = None
        # 最近一次摘要生成失败的原因
        self._last_summary_error: str | None = None
        # 表示最近一次压缩是否因为摘要失败而中止
        self._last_compress_aborted = False
        # 最近一次压缩了多少上下文，计算公式：(压缩前 token - 压缩后 token) / 压缩前 token
        self._last_compression_savings_pct = 100.0
        # 连续无效压缩的次数
        self._ineffective_compression_count = 0
        # 摘要失败后的冷却结束时间，使用单调时钟记录
        self._summary_failure_cooldown_until = 0.0

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False
        self._previous_summary = None
        self._last_summary_error = None
        self._last_compress_aborted = False
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._summary_failure_cooldown_until = 0.0

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(
            usage.get(
                "total_tokens",
                self.last_prompt_tokens + self.last_completion_tokens,
            )
            or 0
        )
        if self.last_prompt_tokens > 0 and not usage.get("estimated", False):
            self.last_real_prompt_tokens = self.last_prompt_tokens
            if self.last_prompt_tokens < self.threshold_tokens:
                if (
                    self.awaiting_real_usage_after_compression
                    and self.last_compression_rough_tokens > 0
                ):
                    self.last_rough_tokens_when_real_prompt_fit = (
                        self.last_compression_rough_tokens
                    )
            else:
                self.last_rough_tokens_when_real_prompt_fit = 0
        self.awaiting_real_usage_after_compression = False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        if rough_tokens < self.threshold_tokens:
            return False
        if self.awaiting_real_usage_after_compression:
            return True
        if not 0 < self.last_real_prompt_tokens < self.threshold_tokens:
            return False
        baseline = (
            self.last_rough_tokens_when_real_prompt_fit
            or self.last_compression_rough_tokens
        )
        if baseline <= 0:
            return False
        tolerated_growth = max(4096, int(self.threshold_tokens * 0.05))
        if rough_tokens - baseline > tolerated_growth:
            return False
        self.last_rough_tokens_when_real_prompt_fit = max(baseline, rough_tokens)
        return True

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = self.last_prompt_tokens if prompt_tokens is None else prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        if self._ineffective_compression_count >= 2:
            return False
        if time.monotonic() < self._summary_failure_cooldown_until:
            return False
        return True

    def should_compress_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        rough_tokens: int | None = None,
    ) -> bool:
        if rough_tokens is None or not self.has_content_to_compress(messages):
            return False
        if self.should_defer_preflight_to_real_usage(rough_tokens):
            return False
        return self.should_compress(rough_tokens)

    def _effective_protect_first_n(self, messages: list[dict[str, Any]]) -> int:
        if self.compression_count >= 1 or self._previous_summary:
            return 0
        if any(_is_summary_message(message) for message in messages):
            return 0
        return self.protect_first_n

    def _align_boundary_forward(
        self,
        messages: list[dict[str, Any]],
        index: int,
    ) -> int:
        while index < len(messages) and _is_tool_result_message(messages[index]):
            index += 1
        return index

    def _align_boundary_backward(
        self,
        messages: list[dict[str, Any]],
        index: int,
    ) -> int:
        if index >= len(messages) or not _is_tool_result_message(messages[index]):
            return index
        cursor = index - 1
        while cursor >= 0:
            if _tool_use_blocks(messages[cursor]):
                return cursor
            if not _is_tool_result_message(messages[cursor]):
                break
            cursor -= 1
        return index

    def _find_tail_cut_by_tokens(
        self,
        messages: list[dict[str, Any]],
        head_end: int,
    ) -> int:
        total = len(messages)
        available = max(0, total - head_end)
        if available <= 2:
            return total
        min_tail = min(
            max(1, min(self.protect_last_n, _MAX_TAIL_MESSAGE_FLOOR)),
            available - 2,
        )
        soft_ceiling = max(1, int(self.tail_token_budget * 1.5))
        accumulated = 0
        cut = total
        for index in range(total - 1, head_end - 1, -1):
            message_tokens = estimate_tokens_rough(messages[index])
            if (
                accumulated + message_tokens > soft_ceiling
                and total - index >= min_tail
            ):
                break
            accumulated += message_tokens
            cut = index
        cut = min(cut, total - min_tail)
        if cut <= head_end:
            cut = max(head_end + 1, total - min_tail)
        cut = self._align_boundary_backward(messages, cut)

        # 与 Hermes 一样，最新用户任务和最近可见 assistant 回复必须留在尾部。
        for role in ("user", "assistant"):
            for index in range(total - 1, head_end - 1, -1):
                if messages[index].get("role") == role:
                    if index < cut:
                        cut = index
                    break
        return max(head_end + 1, cut)

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        head_end = self._align_boundary_forward(
            messages,
            self._effective_protect_first_n(messages),
        )
        return head_end < self._find_tail_cut_by_tokens(messages, head_end)

    def _prune_boundary(self, messages: list[dict[str, Any]]) -> int:
        accumulated = 0
        protected = 0
        for index in range(len(messages) - 1, -1, -1):
            message_tokens = estimate_tokens_rough(messages[index])
            if (
                accumulated + message_tokens > self.tail_token_budget
                and protected >= self.protect_last_n
            ):
                return index + 1
            accumulated += message_tokens
            protected += 1
        return 0

    def _prune_old_tool_results(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        result = copy.deepcopy(messages)
        boundary = self._prune_boundary(result)
        call_names: dict[str, str] = {}
        for message in result:
            for block in _tool_use_blocks(message):
                call_names[_tool_use_id(block)] = str(_field(block, "name", "tool"))

        seen_hashes: set[str] = set()
        pruned = 0
        for index in range(len(result) - 1, -1, -1):
            message = result[index]
            content = message.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            changed = False
            for block in content:
                if _field(block, "type") != "tool_result":
                    new_content.append(block)
                    continue
                text = _content_text(_field(block, "content", ""))
                digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                if len(text) >= 200 and digest in seen_hashes:
                    new_text = "[Duplicate tool output — same content as a more recent call]"
                elif index < boundary and len(text) > 500:
                    tool_name = call_names.get(_tool_result_id(block), "tool")
                    clean = _redact_sensitive_text(text)
                    new_text = (
                        f"[{tool_name}] earlier result ({len(text)} chars): "
                        f"{clean[:240]} ... {clean[-120:]}"
                    )
                else:
                    new_content.append(block)
                    if len(text) >= 200:
                        seen_hashes.add(digest)
                    continue
                new_content.append(_replace_block_content(block, new_text))
                seen_hashes.add(digest)
                changed = True
                pruned += 1
            if changed:
                message["content"] = new_content

            if index < boundary:
                shrunk_blocks = []
                modified = False
                for block in _blocks(message.get("content")):
                    if _field(block, "type") != "tool_use":
                        shrunk_blocks.append(block)
                        continue
                    tool_input = _field(block, "input", {})
                    shrunk_input = _shrink_value(tool_input)
                    if shrunk_input == tool_input:
                        shrunk_blocks.append(block)
                        continue
                    block_dict = _block_to_dict(block)
                    block_dict["input"] = shrunk_input
                    shrunk_blocks.append(block_dict)
                    modified = True
                if modified:
                    message["content"] = shrunk_blocks
        return result, pruned

    def _serialize_for_summary(self, messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "unknown").upper()
            text = _redact_sensitive_text(_content_text(message.get("content")))
            if len(text) > 6000:
                text = text[:4000] + "\n...[truncated]...\n" + text[-1500:]
            parts.append(f"[{role}]\n{text}")
        return "\n\n".join(parts)

    def _summary_budget(self, messages: list[dict[str, Any]]) -> int:
        source_tokens = estimate_tokens_rough(messages)
        return max(
            _MIN_SUMMARY_TOKENS,
            min(int(source_tokens * 0.15), self.max_summary_tokens),
        )

    def _build_summary_prompt(self, messages: list[dict[str, Any]]) -> str:
        source = self._serialize_for_summary(messages)
        if self._previous_summary:
            task = (
                "Update the previous checkpoint with the new conversation. Preserve all "
                "still-relevant details and remove only clearly obsolete state.\n\n"
                f"PREVIOUS CHECKPOINT:\n{self._previous_summary}\n\n"
                f"NEW CONVERSATION:\n{source}"
            )
        else:
            task = f"Create a checkpoint from this conversation:\n\n{source}"
        return f"""You are producing a context-compaction checkpoint for another coding agent.
Do not answer the user. Preserve concrete details and use the user's language.
Never include API keys, passwords, tokens, cookies, or credentials.

Use exactly these sections:
## Active Task
## Completed Actions
## Key Decisions
## Files and Commands
## Failures and Recovery
## Pending Work

{task}

Write only the checkpoint body."""

    def _build_static_fallback_summary(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        snippets = []
        for message in messages[-12:]:
            text = _redact_sensitive_text(_content_text(message.get("content"))).strip()
            if text:
                snippets.append(f"- {message.get('role', 'unknown')}: {text[:300]}")
        return (
            "## Active Task\nUnknown — verify the protected recent messages.\n\n"
            "## Completed Actions\n"
            + ("\n".join(snippets) or "- No recoverable details")
            + "\n\n## Key Decisions\n- Summary model unavailable.\n\n"
            "## Files and Commands\n- Verify current repository state.\n\n"
            "## Failures and Recovery\n- LLM summary generation failed.\n\n"
            "## Pending Work\n- Continue from protected recent messages."
        )

    def _generate_summary(self, messages: list[dict[str, Any]]) -> str | None:
        if self.summary_callback is None:
            self._last_summary_error = "summary callback is not configured"
            return None
        try:
            summary = self.summary_callback(
                self._build_summary_prompt(messages),
                self._summary_budget(messages),
            ).strip()
        except Exception as exc:
            self._last_summary_error = f"{type(exc).__name__}: {exc}"
            self._summary_failure_cooldown_until = (
                time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            )
            return None
        if not summary:
            self._last_summary_error = "summary model returned empty text"
            self._summary_failure_cooldown_until = (
                time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            )
            return None
        return _redact_sensitive_text(summary)

    def _sanitize_tool_pairs(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        surviving_calls = {
            _tool_use_id(block)
            for message in messages
            for block in _tool_use_blocks(message)
            if _tool_use_id(block)
        }
        surviving_results = {
            _tool_result_id(block)
            for message in messages
            for block in _tool_result_blocks(message)
            if _tool_result_id(block)
        }

        cleaned: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                cleaned.append(message)
                continue
            filtered = [
                block
                for block in content
                if not (
                    _field(block, "type") == "tool_result"
                    and _tool_result_id(block) not in surviving_calls
                )
            ]
            if filtered:
                cleaned.append({**message, "content": filtered})

        missing = surviving_calls - surviving_results
        if not missing:
            return cleaned
        patched: list[dict[str, Any]] = []
        index = 0
        while index < len(cleaned):
            message = cleaned[index]
            patched.append(message)
            call_ids = [
                _tool_use_id(block)
                for block in _tool_use_blocks(message)
                if _tool_use_id(block) in missing
            ]
            if not call_ids:
                index += 1
                continue
            stubs = [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": "[Result from earlier conversation — see context summary]",
                }
                for call_id in call_ids
            ]
            if (
                index + 1 < len(cleaned)
                and cleaned[index + 1].get("role") == "user"
            ):
                next_message = copy.deepcopy(cleaned[index + 1])
                next_content = next_message.get("content")
                if isinstance(next_content, list):
                    next_message["content"] = [*stubs, *next_content]
                else:
                    next_message["content"] = [
                        *stubs,
                        {"type": "text", "text": str(next_content or "")},
                    ]
                patched.append(next_message)
                index += 2
                continue
            patched.append({"role": "user", "content": stubs})
            index += 1
        return patched

    def _assemble(
        self,
        messages: list[dict[str, Any]],
        compress_start: int,
        compress_end: int,
        summary: str,
    ) -> list[dict[str, Any]]:
        head = [copy.deepcopy(message) for message in messages[:compress_start]]
        tail = [copy.deepcopy(message) for message in messages[compress_end:]]
        handoff = f"{SUMMARY_PREFIX}\n{summary}\n\n{SUMMARY_END_MARKER}"
        # Anthropic 历史应从 user 开始。后续压缩 protect_first_n 会衰减到 0，
        # 此时把 handoff 合并进受保护尾部的首条 user，而不是生成首条 assistant。
        if not head and tail:
            tail[0]["content"] = _prepend_content(tail[0].get("content"), handoff)
            return self._sanitize_tool_pairs(tail)
        last_head_role = _message_role(head[-1]) if head else None
        first_tail_role = _message_role(tail[0]) if tail else None
        summary_role = "user" if last_head_role == "assistant" else "assistant"
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            elif tail:
                tail[0]["content"] = _prepend_content(tail[0].get("content"), handoff)
                return self._sanitize_tool_pairs([*head, *tail])
        return self._sanitize_tool_pairs([
            *head,
            _make_summary_message(summary_role, handoff),
            *tail,
        ])

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        self._last_summary_error = None
        self._last_compress_aborted = False
        minimum = self._effective_protect_first_n(messages) + 4
        if len(messages) <= minimum:
            self._ineffective_compression_count += 1
            self._last_compression_savings_pct = 0.0
            return messages

        working, _ = self._prune_old_tool_results(messages)
        compress_start = self._align_boundary_forward(
            working,
            self._effective_protect_first_n(working),
        )
        compress_end = self._find_tail_cut_by_tokens(working, compress_start)
        if compress_start >= compress_end:
            self._ineffective_compression_count += 1
            self._last_compression_savings_pct = 0.0
            return messages

        previous_summary = None
        summary_index = None
        for index in range(compress_end - 1, -1, -1):
            body = _summary_body(working[index].get("content"))
            if body:
                previous_summary = body
                summary_index = index
                break
        if previous_summary:
            self._previous_summary = previous_summary
        turns_start = max(
            compress_start,
            (summary_index + 1) if summary_index is not None else compress_start,
        )
        turns_to_summarize = working[turns_start:compress_end]
        if not turns_to_summarize:
            self._ineffective_compression_count += 1
            return messages

        summary = self._generate_summary(turns_to_summarize)
        if not summary:
            if self.abort_on_summary_failure:
                self._last_compress_aborted = True
                return messages
            summary = self._build_static_fallback_summary(turns_to_summarize)

        compressed = self._assemble(
            working,
            compress_start,
            compress_end,
            summary,
        )
        old_estimate = current_tokens or estimate_tokens_rough(messages)
        new_estimate = estimate_tokens_rough(compressed)
        savings_pct = (
            (old_estimate - new_estimate) / old_estimate * 100
            if old_estimate > 0
            else 0.0
        )
        self._last_compression_savings_pct = savings_pct
        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0
        self._previous_summary = summary
        self.compression_count += 1
        self.last_compression_rough_tokens = new_estimate
        self.last_prompt_tokens = -1
        self.awaiting_real_usage_after_compression = True
        return compressed
