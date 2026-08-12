"""SQLite 历史会话检索：发现、浏览、读取与围绕锚点滚动。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from tools.registry import registry


def _error(message: str, **extra: Any) -> str:
    return json.dumps({"success": False, "error": message, **extra}, ensure_ascii=False)


def _text(content: Any, limit: int = 1200) -> str:
    if isinstance(content, str):
        value = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if block_type == "text":
                parts.append(str(block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")))
            elif block_type == "tool_use":
                name = block.get("name", "?") if isinstance(block, dict) else getattr(block, "name", "?")
                parts.append(f"[tool_use: {name}]")
            elif block_type == "tool_result":
                parts.append("[tool_result]")
        value = "\n".join(parts)
    else:
        value = str(content or "")
    return value if len(value) <= limit else value[:limit] + "…"


def _shape_message(message: dict[str, Any], anchor_id: int | None = None) -> dict[str, Any]:
    result = {
        "id": message.get("id"),
        "role": message.get("role"),
        "content": _text(message.get("content")),
    }
    if anchor_id is not None and message.get("id") == anchor_id:
        result["anchor"] = True
    if message.get("timestamp"):
        result["timestamp"] = datetime.fromtimestamp(float(message["timestamp"])).isoformat(timespec="seconds")
    return result


def _read(db: Any, session_id: str) -> str:
    session = db.get_session(session_id)
    if session is None or session.get("archived"):
        return _error(f"Session '{session_id}' was not found or is archived.")
    messages = db.get_messages(session_id, include_inactive=True)
    shaped = [_shape_message(message) for message in messages]
    truncated = len(shaped) > 30
    returned = shaped[:20] + shaped[-10:] if truncated else shaped
    return json.dumps({
        "success": True,
        "mode": "read",
        "session_id": session_id,
        "title": session.get("title") or "未命名",
        "messages": returned,
        "total_messages": len(shaped),
        "returned_messages": len(returned),
        "truncated": truncated,
    }, ensure_ascii=False)


def _browse(db: Any, limit: int, current_session_id: str | None) -> str:
    current_root = db.get_lineage_root(current_session_id) if current_session_id else None
    results = []
    seen: set[str] = set()
    for session in db.list_sessions_rich(limit=max(limit * 4, 20)):
        session_id = str(session["id"])
        root = db.get_lineage_root(session_id)
        if root == current_root or root in seen:
            continue
        seen.add(root)
        messages = db.get_messages(session_id)
        preview = next((_text(m.get("content"), 240) for m in messages if m.get("role") == "user"), "")
        results.append({
            "session_id": session_id,
            "parent_session_id": root if root != session_id else None,
            "title": session.get("title") or "未命名",
            "source": session.get("source"),
            "last_activity": session.get("last_activity"),
            "message_count": session.get("message_count", len(messages)),
            "preview": preview,
        })
        if len(results) >= limit:
            break
    return json.dumps({"success": True, "mode": "browse", "results": results, "count": len(results)}, ensure_ascii=False)


def _scroll(db: Any, session_id: str, anchor: int, window: int, current_session_id: str | None) -> str:
    if db.get_session(session_id) is None:
        return _error(f"Session '{session_id}' was not found.")
    if current_session_id and db.get_lineage_root(session_id) == db.get_lineage_root(current_session_id):
        return _error("Scroll rejected: this session is already in the active conversation context.")
    view = db.get_messages_around(session_id, anchor, window=window)
    if not view["window"]:
        return _error(f"Message {anchor} was not found in session '{session_id}'.")
    return json.dumps({
        "success": True,
        "mode": "scroll",
        "session_id": session_id,
        "around_message_id": anchor,
        "window": max(1, min(window, 20)),
        "messages": [_shape_message(m, anchor) for m in view["window"]],
        "messages_before": view["messages_before"],
        "messages_after": view["messages_after"],
    }, ensure_ascii=False)


def _discover(db: Any, query: str, limit: int, sort: str | None, role_filter: str | None, current_session_id: str | None) -> str:
    roles = [role.strip() for role in role_filter.split(",")] if isinstance(role_filter, str) else None
    current_root = db.get_lineage_root(current_session_id) if current_session_id else None
    title_session_id = db.resolve_session_by_title(query)
    title_entry = None
    if title_session_id and db.get_lineage_root(title_session_id) != current_root:
        title_messages = db.get_messages(title_session_id)
        if title_messages:
            title_anchor = int(title_messages[0]["id"])
            title_view = db.get_anchored_view(title_session_id, title_anchor, window=5, bookend=3)
            title_session = db.get_session(title_session_id) or {}
            title_entry = {
                "session_id": title_session_id,
                "title": title_session.get("title") or "未命名",
                "source": title_session.get("source"),
                "snippet": f"Session title matched: {title_session.get('title') or query}",
                "matched_role": "session_title",
                "match_message_id": title_anchor,
                "bookend_start": [_shape_message(m) for m in title_view["bookend_start"]],
                "messages": [_shape_message(m, title_anchor) for m in title_view["window"]],
                "bookend_end": [_shape_message(m) for m in title_view["bookend_end"]],
                "messages_before": title_view["messages_before"],
                "messages_after": title_view["messages_after"],
            }
    try:
        hits = db.search_messages(query, limit=max(50, limit * 10), role_filter=roles, sort=sort)
    except (RuntimeError, ValueError) as exc:
        return _error(str(exc))
    seen: set[str] = set()
    results = [title_entry] if title_entry is not None else []
    if title_session_id and title_entry is not None:
        seen.add(db.get_lineage_root(title_session_id))
    for hit in hits:
        if len(results) >= limit:
            break
        hit_session_id = str(hit["session_id"])
        root = db.get_lineage_root(hit_session_id)
        if root == current_root or root in seen:
            continue
        seen.add(root)
        anchor = int(hit["message_id"])
        view = db.get_anchored_view(hit_session_id, anchor, window=5, bookend=3)
        session = db.get_session(root) or db.get_session(hit_session_id) or {}
        entry = {
            "session_id": hit_session_id,
            "title": session.get("title") or "未命名",
            "source": session.get("source"),
            "snippet": hit.get("snippet", ""),
            "match_message_id": anchor,
            "bookend_start": [_shape_message(m) for m in view["bookend_start"]],
            "messages": [_shape_message(m, anchor) for m in view["window"]],
            "bookend_end": [_shape_message(m) for m in view["bookend_end"]],
            "messages_before": view["messages_before"],
            "messages_after": view["messages_after"],
        }
        if root != hit_session_id:
            entry["parent_session_id"] = root
        results.append(entry)
        if len(results) >= limit:
            break
    return json.dumps({
        "success": True,
        "mode": "discover",
        "query": query,
        "results": results,
        "count": len(results),
        "sessions_searched": len(seen),
    }, ensure_ascii=False)


def session_search(
    *,
    query: str = "",
    role_filter: str | None = None,
    limit: int = 3,
    sort: str | None = None,
    session_id: str | None = None,
    around_message_id: int | None = None,
    window: int = 5,
    db: Any = None,
    current_session_id: str | None = None,
) -> str:
    """根据参数形态分派 Discovery、Browse、Read 或 Scroll。"""
    if db is None:
        return _error("Session database is unavailable in this Agent.")
    try:
        safe_limit = max(1, min(int(limit), 10))
        safe_window = max(1, min(int(window), 20))
        if isinstance(session_id, str) and session_id.strip() and around_message_id is not None:
            return _scroll(db, session_id.strip(), int(around_message_id), safe_window, current_session_id)
        if isinstance(session_id, str) and session_id.strip():
            if current_session_id and db.get_lineage_root(session_id.strip()) == db.get_lineage_root(current_session_id):
                return _error("Read rejected: this session is already in the active conversation context.")
            return _read(db, session_id.strip())
        if not isinstance(query, str) or not query.strip():
            return _browse(db, safe_limit, current_session_id)
        return _discover(db, query.strip(), safe_limit, sort, role_filter, current_session_id)
    except (OSError, TypeError, ValueError) as exc:
        return _error(f"Invalid session_search request: {exc}")
    except Exception as exc:
        return _error(f"Session search failed: {type(exc).__name__}: {exc}")


SESSION_SEARCH_TOOL = {
    "name": "session_search",
    "description": (
        "Search past Athena sessions stored in the local SQLite database. Results are actual "
        "historical messages, not LLM-generated summaries. SOURCE-FIRST: history only proves "
        "what was said before; inspect a current file, URL, account, or live system first when "
        "available. Do not infer 'never happened' from no result. FOUR SHAPES: query=Discovery "
        "(FTS5 hits with session bookends and ±5 context); no args=Browse recent sessions; "
        "session_id=Read; session_id+around_message_id=Scroll around an anchor. Use Scroll with "
        "the first or last returned message id to move backward or forward."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "FTS5 query for Discovery. Omit to Browse."},
            "role_filter": {"type": "string", "description": "Optional comma-separated roles: user,assistant,tool."},
            "limit": {"type": "integer", "description": "Discovery/Browse result count, clamped to 1–10.", "default": 3},
            "sort": {"type": "string", "enum": ["newest", "oldest"], "description": "Optional temporal bias for Discovery."},
            "session_id": {"type": "string", "description": "Session to Read or Scroll."},
            "around_message_id": {"type": "integer", "description": "Message anchor for Scroll."},
            "window": {"type": "integer", "description": "Messages on each side for Scroll, clamped to 1–20.", "default": 5},
        },
    },
}


registry.register(
    name="session_search",
    schema=SESSION_SEARCH_TOOL,
    handler=session_search,
    toolset="session_search",
)
