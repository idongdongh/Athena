"""CLI 会话列表的查询和文本格式化。"""

from __future__ import annotations

from session_db import SessionDB


def parse_session_listing_args(raw_args: str) -> tuple[bool, str]:
    """返回 ``(should_list, resume_target)``。"""
    argument = raw_args.strip()
    if not argument or argument.lower() in {"list", "ls", "browse"}:
        return True, ""
    return False, argument


def query_session_listing(db: SessionDB, *, limit: int = 10) -> list[dict]:
    return db.list_sessions_rich(limit=limit)


def format_session_listing(
    sessions: list[dict],
    *,
    current_session_id: str = "",
    numbered: bool = False,
) -> str:
    """格式化经典 CLI 使用的内联会话列表。"""
    lines: list[str] = []
    for index, session in enumerate(sessions, start=1):
        title = session.get("title") or "未命名"
        prefix = f"{index}." if numbered else (
            "*" if session["id"] == current_session_id else " "
        )
        lines.append(
            f"{prefix} {session['id']}  {title}  "
            f"{session.get('message_count', 0)} 条消息"
        )
    return "\n".join(lines)
