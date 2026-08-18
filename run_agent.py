"""Athena Agent 运行对象。

CLI 负责交互和消息历史；``AIAgent`` 持有一次会话的 Agent 状态，并把单个
用户 turn 委托给 ``agent.conversation_loop``。
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from anthropic import Anthropic
from httpx import Timeout

from agent.context_state import ContextSettings, ContextState, estimate_request_tokens_rough
from agent.interrupt_controller import InterruptController, interrupt_controller
from agent.model_response import TokenUsage
from agent.file_mutation_tracker import FileMutationTracker
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController
from athena_cli.config import MemorySettings
from session_db import SessionDB, new_session_id
from tools.memory_tool import MemoryStore
from tools.registry import ensure_tools_discovered


class AIAgent:
    """代表一个正在运行的会话，并协调模型、工具、上下文、记忆和持久化。"""

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
        memory_settings: MemorySettings | None = None,
        memory_root: Path | None = None,
        working_directory: Path | str | None = None,
        tool_allowlist: frozenset[str] | None = None,
        is_background_review: bool = False,
        trace_sink: Any = None,
    ) -> None:
        self.model = model
        self.caller_system_prompt = system_prompt
        self.working_directory = Path(working_directory or Path.cwd())
        self.system_prompt = ""
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
        self._tool_allowlist = tool_allowlist
        self._is_background_review = is_background_review
        self.trace_sink = trace_sink
        self._trace_turn_counter = 0
        self.interrupt_controller = (
            InterruptController() if is_background_review else interrupt_controller
        )
        self._background_review_lock = threading.Lock()
        self._background_review_thread: threading.Thread | None = None
        self._last_flushed_db_idx = 0
        self.memory_settings = memory_settings
        self._memory_nudge_interval = memory_settings.nudge_interval if memory_settings else 0
        self._turns_since_memory = 0
        self._memory_store: MemoryStore | None = None
        if memory_settings is not None and (
            memory_settings.memory_enabled or memory_settings.user_profile_enabled
        ):
            try:
                memory_dir = memory_root or memory_settings.resolve_directory(project_root())
                self._memory_store = MemoryStore(
                    memory_dir,
                    memory_char_limit=memory_settings.memory_char_limit,
                    user_char_limit=memory_settings.user_char_limit,
                )
                self._memory_store.load_from_disk()
            except (OSError, UnicodeError, ValueError) as exc:
                print(f"\033[33m⚠️  记忆存储初始化失败，已禁用本次会话记忆：{exc}\033[0m")
                self._memory_store = None
        # system prompt 的工具指导必须基于本次请求真正会暴露的完整 registry，
        # 不能依赖后续导入 conversation_loop 时产生的模块级副作用。
        ensure_tools_discovered()
        self._rebuild_system_prompt()

        if self.session_db is not None and session_id is None:
            self.session_db.create_session(
                self.session_id,
                source,
                model=model,
                model_config=model_config,
                system_prompt=self.system_prompt,
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
        system_prompt: str | None = None,
        memory_settings: MemorySettings | None = None,
        memory_root: Path | None = None,
    ) -> tuple["AIAgent", list[dict[str, Any]]]:
        resolved = db.resolve_resume_session_id(session_id)
        session = db.get_session(resolved)
        if session is None:
            raise KeyError(f"会话不存在：{session_id}")
        messages = db.get_messages_as_conversation(resolved)
        agent = cls(
            model=str(session.get("model") or ""),
            # 数据库字段是上次已经拼装完成的 prompt，只用于审计，不能再次当作
            # caller prompt 嵌套。恢复时总是使用本次调用方传入的稳定补充文本。
            system_prompt=system_prompt or "",
            context_settings=context_settings,
            session_db=db,
            session_id=resolved,
            source=str(session.get("source") or "cli"),
            client=client,
            tool_guardrail_config=tool_guardrail_config,
            memory_settings=memory_settings,
            memory_root=memory_root,
        )
        agent._last_flushed_db_idx = len(messages)
        agent.reset_session_state(messages)
        return agent, messages

    def _rebuild_system_prompt(self) -> None:
        """按 stable/context/volatile 三层重建完整 system prompt。"""
        from agent.system_prompt import build_system_prompt

        self.system_prompt = build_system_prompt(self)

    def refresh_memory_snapshot(self) -> None:
        """压缩边界重建 prompt 时，从磁盘捕获新的冻结快照。"""
        if self._memory_store is None:
            return
        self._memory_store.load_from_disk()
        self._rebuild_system_prompt()

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
            tools=self.tool_definitions(),
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
        if self._memory_nudge_interval > 0 and not self._is_background_review:
            user_turns = sum(1 for message in messages if _is_real_user_message(message))
            self._turns_since_memory = user_turns % self._memory_nudge_interval

    def tool_definitions(self) -> list[dict[str, Any]]:
        """返回当前 Agent 实际可用的工具；禁用记忆时不暴露 memory schema。"""
        from tools.registry import registry

        definitions = registry.definitions()
        if self._memory_store is None:
            definitions = [definition for definition in definitions if definition.get("name") != "memory"]
        if self.session_db is None:
            definitions = [
                definition for definition in definitions
                if definition.get("name") != "session_search"
            ]
        if self._tool_allowlist is not None:
            definitions = [
                definition for definition in definitions
                if definition.get("name") in self._tool_allowlist
            ]
        return definitions

    def begin_memory_review_cycle(self) -> bool:
        """记录一个真实用户 turn，并返回本轮正常完成后是否应触发复盘。"""
        if (
            self._is_background_review
            or self._memory_store is None
            or self._memory_nudge_interval <= 0
        ):
            return False
        self._turns_since_memory += 1
        if self._turns_since_memory < self._memory_nudge_interval:
            return False
        self._turns_since_memory = 0
        return True

    def note_memory_tool_call(self) -> None:
        """前台已主动尝试维护记忆，重新开始定期复盘周期。"""
        if not self._is_background_review:
            self._turns_since_memory = 0

    def spawn_background_memory_review(self, messages: list[dict[str, Any]]) -> bool:
        """若没有复盘正在运行，异步检查本轮对话是否遗漏长期记忆。"""
        if self._is_background_review or self._memory_store is None:
            return False
        from agent.background_review import spawn_background_review_thread

        with self._background_review_lock:
            if self._background_review_thread is not None and self._background_review_thread.is_alive():
                return False
            thread = spawn_background_review_thread(self, list(messages))
            self._background_review_thread = thread
            thread.start()
        return True

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


def _is_real_user_message(message: dict[str, Any]) -> bool:
    """排除 Anthropic tool_result 和 Agent 自己插入的协议修复 user 消息。"""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, list) and any(
        (isinstance(block, dict) and block.get("type") == "tool_result")
        or getattr(block, "type", None) == "tool_result"
        for block in content
    ):
        return False
    if not isinstance(content, str):
        return True
    return content not in {
        "Continue exactly where the previous response stopped. Do not repeat earlier text. Return plain text only and do not call tools.",
        "The previous response was empty. Process the tool results and provide a complete answer to the user.",
    }
