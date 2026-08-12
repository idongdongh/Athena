"""经典交互式 CLI 的斜杠命令处理方法。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from athena_cli.session_listing import (
    format_session_listing,
    parse_session_listing_args,
    query_session_listing,
)
from run_agent import AIAgent
from session_db import SessionDB


class CLICommandsMixin:
    """向 ``AthenaCLI`` 混入各个斜杠命令的处理方法。"""

    agent: AIAgent
    conversation_history: list[dict]
    _session_db: SessionDB | None
    _pending_resume_sessions: list[dict] | None

    if TYPE_CHECKING:
        def _create_agent(self) -> AIAgent: ...

        def _resume_agent(self, session_id: str) -> tuple[AIAgent, list[dict]]: ...

    def _handle_new_command(self, _argument: str) -> None:
        self.agent.flush_new_messages(self.conversation_history)
        self.agent.end("session_reset")
        self.conversation_history = []
        self.agent = self._create_agent()
        print(f"已创建新会话：{self.agent.session_id}")

    def _handle_sessions_command(self, argument: str) -> None:
        if self._session_db is None:
            print("会话数据库不可用")
            return
        should_list, target = parse_session_listing_args(argument)
        if not should_list:
            self._handle_resume_command(target)
            return
        sessions = query_session_listing(self._session_db, limit=50)
        if not sessions:
            print("暂无历史会话")
            return
        print(format_session_listing(
            sessions,
            current_session_id=self.agent.session_id,
        ))

    def _handle_resume_command(self, argument: str) -> None:
        if self._session_db is None:
            print("会话数据库不可用")
            return
        if not argument:
            sessions = query_session_listing(self._session_db, limit=10)
            self._pending_resume_sessions = sessions
            if not sessions:
                print("暂无可恢复会话")
                return
            print("近期会话：")
            print(format_session_listing(sessions, numbered=True))
            print("输入编号，或使用 /resume <session_id> 恢复")
            return
        self._pending_resume_sessions = None
        if argument.isdigit():
            sessions = query_session_listing(self._session_db, limit=10)
            index = int(argument)
            if index < 1 or index > len(sessions):
                print("恢复编号超出范围，请使用 /resume 查看列表")
                return
            argument = sessions[index - 1]["id"]
        try:
            resolved = self._session_db.resolve_resume_session_id(argument)
        except (KeyError, ValueError) as exc:
            print(str(exc))
            return
        if resolved == self.agent.session_id:
            print("已经位于该会话")
            return
        old_agent = self.agent
        try:
            next_agent, messages = self._resume_agent(resolved)
        except (KeyError, ValueError) as exc:
            print(str(exc))
            return
        old_agent.flush_new_messages(self.conversation_history)
        old_agent.end("resumed_other")
        self._session_db.reopen_session(next_agent.session_id)
        self.agent = next_agent
        self.conversation_history = messages
        print(f"已恢复会话：{next_agent.session_id}")

    def _handle_search_command(self, argument: str) -> None:
        if self._session_db is None:
            print("会话数据库不可用")
            return
        if not argument:
            print("用法：/search <关键词>")
            return
        matches = self._session_db.search_messages(argument)
        if not matches:
            print("没有找到匹配消息")
            return
        print("\n".join(
            f"{item['session_id']}  {item['role']}: {item['snippet']}"
            for item in matches
        ))

    def _handle_archive_command(self, argument: str) -> None:
        if self._session_db is None:
            print("会话数据库不可用")
            return
        target = argument or self.agent.session_id
        archived = self._session_db.archive_session(target)
        print(f"已归档会话：{target}" if archived else f"会话不存在：{target}")
