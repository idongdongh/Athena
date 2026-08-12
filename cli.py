#!/usr/bin/env python3
"""Athena 经典交互式 CLI。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from agent.context_state import ContextSettings
from agent.tool_guardrails import ToolCallGuardrailConfig
from athena_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from athena_cli.cli_commands_mixin import CLICommandsMixin
from athena_cli.commands import command_names, resolve_command
from athena_cli.config import MemorySettings, SessionSettings, load_config
from session_db import SessionDB


def _handle_idle_ctrl_c(event) -> None:
    """输入框有内容时清空；空输入框时退出 CLI。"""
    buffer = event.app.current_buffer
    if buffer.text:
        buffer.reset()
        event.app.invalidate()
        return
    event.app.exit(exception=KeyboardInterrupt)


def _create_key_bindings() -> KeyBindings:
    bindings = KeyBindings()
    bindings.add("c-c")(_handle_idle_ctrl_c)
    return bindings


class AthenaCLI(CLIAgentSetupMixin, CLICommandsMixin):
    """持有终端交互状态、消息历史和当前 Agent。"""

    def __init__(self) -> None:
        load_dotenv(override=True)
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY（或 API_KEY）未设置")
        self.api_key = api_key
        self.model = os.getenv("MODEL_ID") or ""
        if not self.model:
            raise RuntimeError("MODEL_ID 未设置")
        self.base_url = os.getenv("BASE_URL")
        self.system_prompt = ""

        config = load_config()
        self.tool_guardrail_config = ToolCallGuardrailConfig.from_mapping(
            config.get("tool_loop_guardrails")
        )
        self.context_settings = ContextSettings.from_mapping(config)
        self.session_settings = SessionSettings.from_mapping(config)
        self.memory_settings = MemorySettings.from_mapping(config)
        # 支持的所有命令名称（元组）
        self.command_names = command_names()
        self.conversation_history: list[dict] = []
        # 待回复的会话：用来支持会话 /resume 命令，存当前所有会话记录，便于用户使用编号恢复
        self._pending_resume_sessions: list[dict] | None = None
        self._session_db: SessionDB | None = None

        # 初始化会话数据库
        if self.session_settings.enabled:
            try:
                db_path = self.session_settings.resolve_database_path(
                    Path(__file__).resolve().parent
                )
                self._session_db = SessionDB(db_path)
            except Exception as exc:
                print(f"\033[33m⚠️  会话数据库初始化失败，改用内存模式：{exc}\033[0m")
        # 创建了一个 AIAgent 对象
        self.agent = self._create_agent()
        self.prompt_session = (
            PromptSession(history=InMemoryHistory(), key_bindings=_create_key_bindings())
            if sys.stdin.isatty()
            else None
        )

    def _read_input(self) -> str:
        if self.prompt_session is not None:
            return self.prompt_session.prompt(ANSI("\033[33m >> \033[0m"))
        return input("\033[33m >> \033[0m")

    def process_command(self, query: str) -> bool:
        """处理斜杠命令；普通用户文本返回 ``False``。"""
        stripped = query.strip()
        if not stripped.startswith("/"):
            return False
        # 以空格为分界解析命令和参数，返回一个三元组(/resume, " ", session_id)
        command_word, _, argument = stripped.partition(" ")
        definition = resolve_command(command_word)
        if definition is None:
            names = "、".join(f"/{name}" for name in self.command_names)
            print(f"未知命令：{command_word}。可用命令：{names}")
            return True
        handler = getattr(self, f"_handle_{definition.name}_command")
        handler(argument.strip())
        return True

    def run(self) -> None:
        try:
            while True:
                try:
                    # 获取用户输入
                    query = self._read_input()
                    # 处理 /resume 命令的序号输入
                    if self._pending_resume_sessions is not None:
                        pending = self._pending_resume_sessions
                        self._pending_resume_sessions = None
                        if query.strip().isdigit():
                            index = int(query.strip())
                            if 1 <= index <= len(pending):
                                self._handle_resume_command(pending[index - 1]["id"])
                                continue
                    # 处理其他命令，未知命令会提示
                    if self.process_command(query):
                        continue
                    self.conversation_history.append({"role": "user", "content": query})
                    # agent_loop
                    self.agent.run_conversation(
                        self.conversation_history,
                        stream_output=True,
                    )
                except EOFError:
                    print("\n输入结束，退出交互")
                    break
                except KeyboardInterrupt:
                    print("\n收到 Ctrl+C，退出交互")
                    break
        finally:
            try:
                # 保存哪些刚追加，但还没进入 loop 的消息等
                self.agent.flush_new_messages(self.conversation_history)
                # 标记会话正常结束，在 session 表中记录会话结束时间和结束原因
                self.agent.end("user_exit")
            # 关闭数据库连接
            finally:
                if self._session_db is not None:
                    self._session_db.close()


def main() -> None:
    AthenaCLI().run()


if __name__ == "__main__":
    main()
