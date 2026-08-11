"""把内存 conversation history 增量同步到 Hermes 风格 SessionDB。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.model_response import TokenUsage
from agent.session_db import SessionDB, new_session_id


@dataclass(frozen=True)
class SessionSettings:
    enabled: bool = True
    database: str = ".hello-agent/state.db"
    # Hermes CLI 默认新建会话；只有显式 --continue/--resume 或 /resume 才恢复。
    auto_resume: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "SessionSettings":
        defaults = cls()
        if not isinstance(config, Mapping):
            return defaults
        section = config.get("session")
        if not isinstance(section, Mapping):
            return defaults
        enabled = section.get("enabled")
        auto_resume = section.get("auto_resume")
        database = section.get("database")
        return cls(
            enabled=enabled if isinstance(enabled, bool) else defaults.enabled,
            database=(
                database.strip()
                if isinstance(database, str) and database.strip()
                else defaults.database
            ),
            auto_resume=(
                auto_resume
                if isinstance(auto_resume, bool)
                else defaults.auto_resume
            ),
        )

    def resolve_database_path(self, project_root: Path) -> Path:
        path = Path(self.database).expanduser()
        return path if path.is_absolute() else project_root / path


@dataclass
class SessionRuntime:
    """当前会话的持久化游标；不持有第二份消息内容。"""

    db: SessionDB
    session_id: str
    source: str = "cli"
    _flushed_count: int = 0

    @classmethod
    def start(
        cls,
        db: SessionDB,
        *,
        source: str = "cli",
        model: str | None = None,
        model_config: Any = None,
        system_prompt: str | None = None,
        parent_session_id: str | None = None,
    ) -> "SessionRuntime":
        session_id = new_session_id()
        db.create_session(
            session_id,
            source,
            model=model,
            model_config=model_config,
            system_prompt=system_prompt,
            parent_session_id=parent_session_id,
        )
        return cls(db=db, session_id=session_id, source=source)

    @classmethod
    def resume(
        cls,
        db: SessionDB,
        session_id: str,
    ) -> tuple["SessionRuntime", list[dict[str, Any]]]:
        resolved = db.resolve_resume_session_id(session_id)
        session = db.get_session(resolved)
        if session is None:
            raise KeyError(f"会话不存在：{session_id}")
        messages = db.get_messages_as_conversation(resolved)
        runtime = cls(
            db=db,
            session_id=resolved,
            source=str(session.get("source") or "cli"),
            _flushed_count=len(messages),
        )
        return runtime, messages

    @property
    def flushed_count(self) -> int:
        return self._flushed_count

    def flush_new_messages(self, messages: list[dict[str, Any]]) -> int:
        """从游标开始逐条写入；成功一条才推进一格，重试不会重复。"""
        if len(messages) < self._flushed_count:
            raise RuntimeError("消息历史已缩短，请先为压缩结果重置持久化基线")
        written = 0
        while self._flushed_count < len(messages):
            message = messages[self._flushed_count]
            self.db.append_message(
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
            self._flushed_count += 1
            written += 1
        return written

    def record_usage(self, usage: TokenUsage) -> None:
        self.db.update_session_usage(
            self.session_id,
            input_tokens=usage.effective_input_tokens,
            output_tokens=usage.output_tokens,
            api_calls=1,
        )

    def reset_flush_baseline(self, messages: list[dict[str, Any]]) -> None:
        """数据库已原子写入整份新活动历史后，同步内存游标。"""
        self._flushed_count = len(messages)

    def persist_compression(
        self,
        original_messages: list[dict[str, Any]],
        compressed_messages: list[dict[str, Any]],
        *,
        in_place: bool,
    ) -> str:
        """持久化压缩边界；成功后才更新 session ID 和 flush 游标。"""
        self.flush_new_messages(original_messages)
        if in_place:
            self.db.archive_and_compact(self.session_id, compressed_messages)
        else:
            old_session_id = self.session_id
            next_session_id = new_session_id()
            self.db.rotate_after_compression(
                old_session_id,
                next_session_id,
                compressed_messages,
            )
            self.session_id = next_session_id
        self.reset_flush_baseline(compressed_messages)
        return self.session_id

    def end(self, reason: str) -> None:
        self.db.end_session(self.session_id, reason)
