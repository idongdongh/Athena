"""Athena Agent 运行对象。

CLI 负责交互和消息历史；``AIAgent`` 持有一次会话的 Agent 状态，并把单个
用户 turn 委托给 ``agent.conversation_loop``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anthropic import Anthropic
from httpx import Timeout

from agent.context_state import ContextSettings, ContextState, estimate_request_tokens_rough
from agent.model_response import TokenUsage
from agent.file_mutation_tracker import FileMutationTracker
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
from session_db import SessionDB, new_session_id


class AIAgent:
    """Agent 会话状态与持久化边界。"""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        context_settings: ContextSettings,
        session_db: SessionDB | None = None,
        session_id: str | None = None,
        source: str = "cli",
        model_config: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        tool_guardrail_config: ToolCallGuardrailConfig | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.context_settings = context_settings
        self.session_db = session_db
        self.session_id = session_id or new_session_id()
        self.source = source
        self.client = client or Anthropic(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=Timeout(timeout=300.0, connect=10.0),
        )
        self.tool_guardrails = ToolCallGuardrailController(tool_guardrail_config)
        self.file_mutation_tracker = FileMutationTracker()
        self._last_flushed_db_idx = 0

        if self.session_db is not None and session_id is None:
            self.session_db.create_session(
                self.session_id,
                source,
                model=model,
                model_config=model_config,
                system_prompt=system_prompt,
            )

        self.context_state = ContextState.from_settings(context_settings)
        self.context_compressor = None
        self.reset_session_state([])

    @property
    def db(self) -> SessionDB:
        """兼容压缩编排所需的窄持久化接口。"""
        if self.session_db is None:
            raise RuntimeError("会话数据库不可用")
        return self.session_db

    @property
    def flushed_count(self) -> int:
        return self._last_flushed_db_idx

    @classmethod
    def resume(
        cls,
        db: SessionDB,
        session_id: str,
        *,
        context_settings: ContextSettings,
        client: Any = None,
        tool_guardrail_config: ToolCallGuardrailConfig | None = None,
    ) -> tuple["AIAgent", list[dict[str, Any]]]:
        resolved = db.resolve_resume_session_id(session_id)
        session = db.get_session(resolved)
        if session is None:
            raise KeyError(f"会话不存在：{session_id}")
        messages = db.get_messages_as_conversation(resolved)
        agent = cls(
            model=str(session.get("model") or ""),
            system_prompt=str(session.get("system_prompt") or ""),
            context_settings=context_settings,
            session_db=db,
            session_id=resolved,
            source=str(session.get("source") or "cli"),
            client=client,
            tool_guardrail_config=tool_guardrail_config,
        )
        agent._last_flushed_db_idx = len(messages)
        agent.reset_session_state(messages)
        return agent, messages

    def reset_session_state(self, messages: list[dict[str, Any]]) -> None:
        """切换会话后重建所有由消息历史派生的上下文状态。"""
        from agent.conversation_loop import create_context_compressor
        from tools.registry import registry

        state = ContextState.from_settings(self.context_settings)
        session = (
            self.session_db.get_session(self.session_id)
            if self.session_db is not None
            else None
        ) or {}
        current_tokens = estimate_request_tokens_rough(
            system=self.system_prompt,
            messages=messages,
            tools=registry.definitions(),
        )
        state.restore_session_totals(
            input_tokens=int(session.get("input_tokens") or 0),
            output_tokens=int(session.get("output_tokens") or 0),
            current_input_tokens=current_tokens,
        )
        compressor = create_context_compressor(self)
        state._context_compressor = compressor
        self.context_state = state
        self.context_compressor = compressor

    def flush_new_messages(self, messages: list[dict[str, Any]]) -> int:
        if self.session_db is None:
            return 0
        if len(messages) < self._last_flushed_db_idx:
            raise RuntimeError("消息历史已缩短，请先为压缩结果重置持久化基线")
        written = 0
        while self._last_flushed_db_idx < len(messages):
            message = messages[self._last_flushed_db_idx]
            self.session_db.append_message(
                self.session_id,
                str(message.get("role", "unknown")),
                message.get("content"),
                tool_name=message.get("tool_name"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
                token_count=message.get("token_count"),
                finish_reason=message.get("finish_reason"),
                timestamp=message.get("timestamp"),
            )
            self._last_flushed_db_idx += 1
            written += 1
        return written

    def record_usage(self, usage: TokenUsage) -> None:
        if self.session_db is not None:
            self.session_db.update_session_usage(
                self.session_id,
                input_tokens=usage.effective_input_tokens,
                output_tokens=usage.output_tokens,
                api_calls=1,
            )

    def persist_compression(
        self,
        original_messages: list[dict[str, Any]],
        compressed_messages: list[dict[str, Any]],
        *,
        in_place: bool,
    ) -> str:
        if self.session_db is None:
            return self.session_id
        self.flush_new_messages(original_messages)
        if in_place:
            self.session_db.archive_and_compact(self.session_id, compressed_messages)
        else:
            old_session_id = self.session_id
            next_session_id = new_session_id()
            self.session_db.rotate_after_compression(
                old_session_id,
                next_session_id,
                compressed_messages,
            )
            self.session_id = next_session_id
        self._last_flushed_db_idx = len(compressed_messages)
        return self.session_id

    def end(self, reason: str) -> None:
        if self.session_db is not None:
            self.session_db.end_session(self.session_id, reason)

    def run_conversation(
        self,
        messages: list[dict[str, Any]],
        *,
        stream_output: bool = False,
    ) -> None:
        from agent.conversation_loop import run_conversation

        run_conversation(self, messages, stream_output=stream_output)


def project_root() -> Path:
    return Path(__file__).resolve().parent
