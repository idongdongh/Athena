"""Hermes 风格的最小 REPL 会话命令。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.session_db import SessionDB
from agent.session_runtime import SessionRuntime


@dataclass(frozen=True)
class SessionCommandResult:
    handled: bool
    output: str = ""
    runtime: SessionRuntime | None = None
    messages: list[dict[str, Any]] | None = None
    reset_context: bool = False


def handle_session_command(
    query: str,
    *,
    db: SessionDB,
    runtime: SessionRuntime,
    model: str,
    system_prompt: str,
) -> SessionCommandResult:
    """解析一个会话命令；普通用户输入返回 ``handled=False``。"""
    stripped = query.strip()
    if not stripped.startswith("/"):
        return SessionCommandResult(handled=False)
    command, _, argument = stripped.partition(" ")
    argument = argument.strip()

    if command == "/new":
        runtime.end("session_reset")
        new_runtime = SessionRuntime.start(
            db,
            model=model,
            system_prompt=system_prompt,
        )
        return SessionCommandResult(
            handled=True,
            output=f"已创建新会话：{new_runtime.session_id}",
            runtime=new_runtime,
            messages=[],
            reset_context=True,
        )

    if command == "/sessions":
        if argument and argument.lower() not in {"list", "ls", "browse"}:
            return handle_session_command(
                f"/resume {argument}",
                db=db,
                runtime=runtime,
                model=model,
                system_prompt=system_prompt,
            )
        sessions = db.list_sessions_rich()
        if not sessions:
            return SessionCommandResult(handled=True, output="暂无历史会话")
        lines = []
        for session in sessions:
            marker = "*" if session["id"] == runtime.session_id else " "
            title = session.get("title") or "未命名"
            lines.append(
                f"{marker} {session['id']}  {title}  "
                f"{session.get('message_count', 0)} 条消息"
            )
        return SessionCommandResult(handled=True, output="\n".join(lines))

    if command == "/resume":
        if not argument:
            sessions = db.list_sessions_rich(limit=10)
            if not sessions:
                return SessionCommandResult(handled=True, output="暂无可恢复会话")
            lines = ["近期会话："]
            for index, session in enumerate(sessions, start=1):
                title = session.get("title") or "未命名"
                lines.append(
                    f"{index}. {session['id']}  {title}  "
                    f"{session.get('message_count', 0)} 条消息"
                )
            lines.append("使用 /resume <session_id> 恢复")
            return SessionCommandResult(handled=True, output="\n".join(lines))
        resumed, messages = SessionRuntime.resume(db, argument)
        if resumed.session_id == runtime.session_id:
            return SessionCommandResult(
                handled=True,
                output="已经位于该会话",
            )
        if resumed.session_id != runtime.session_id:
            runtime.end("resumed_other")
        db.reopen_session(resumed.session_id)
        return SessionCommandResult(
            handled=True,
            output=f"已恢复会话：{resumed.session_id}",
            runtime=resumed,
            messages=messages,
            reset_context=True,
        )

    if command == "/search":
        if not argument:
            return SessionCommandResult(handled=True, output="用法：/search <关键词>")
        try:
            matches = db.search_messages(argument)
        except RuntimeError as exc:
            return SessionCommandResult(handled=True, output=str(exc))
        if not matches:
            return SessionCommandResult(handled=True, output="没有找到匹配消息")
        lines = [
            f"{item['session_id']}  {item['role']}: {item['snippet']}"
            for item in matches
        ]
        return SessionCommandResult(handled=True, output="\n".join(lines))

    if command == "/archive":
        target = argument or runtime.session_id
        archived = db.archive_session(target)
        return SessionCommandResult(
            handled=True,
            output=(f"已归档会话：{target}" if archived else f"会话不存在：{target}"),
        )

    return SessionCommandResult(handled=False)
