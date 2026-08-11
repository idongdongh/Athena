"""Hermes 风格的 SQLite 会话存储。

保留 ``hermes_state.SessionDB`` 的单 Agent 核心：WAL、版本化 schema、
增量消息、非破坏压缩、压缩链恢复和 FTS5 搜索。Gateway、计费、分支、
多平台字段及数据库修复工具不属于 hello-agent 当前调用面。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar


SCHEMA_VERSION = 1
_CONTENT_JSON_PREFIX = "__hello_agent_json__:"
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{3,}")
_WriteResult = TypeVar("_WriteResult")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


def new_session_id() -> str:
    """生成与 Hermes 同形态的可排序会话 ID。"""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


def _json_default(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return attributes
    return str(value)


class SessionDB:
    """SQLite-backed session storage，API 与 Hermes 核心子集对齐。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=5.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._trigram_enabled = False
        self._configure_connection()
        self._initialize_schema()

    @property
    def fts_enabled(self) -> bool:
        return self._fts_enabled

    @property
    def trigram_enabled(self) -> bool:
        return self._trigram_enabled

    def _configure_connection(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL")

    def _initialize_schema(self) -> None:
        with self._lock:
            try:
                self._conn.executescript(SCHEMA_SQL)
                row = self._conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                ).fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO schema_version(version) VALUES (?)",
                        (SCHEMA_VERSION,),
                    )
                elif int(row["version"]) > SCHEMA_VERSION:
                    raise RuntimeError(
                        "state.db schema is newer than this hello-agent build: "
                        f"{row['version']} > {SCHEMA_VERSION}"
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            self._fts_enabled = self._try_initialize_fts(FTS_SQL, "messages_fts")
            if self._fts_enabled:
                self._trigram_enabled = self._try_initialize_fts(
                    FTS_TRIGRAM_SQL,
                    "messages_fts_trigram",
                )

    def _try_initialize_fts(self, sql: str, table: str) -> bool:
        try:
            self._conn.executescript(sql)
            self._conn.execute(
                f"""INSERT INTO {table}(rowid, content)
                    SELECT m.id,
                           COALESCE(m.content, '') || ' ' ||
                           COALESCE(m.tool_name, '') || ' ' ||
                           COALESCE(m.tool_calls, '')
                    FROM messages m
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {table} AS indexed WHERE indexed.rowid = m.id
                    )"""
            )
            self._conn.commit()
            return True
        except sqlite3.Error:
            self._conn.rollback()
            return False

    def _execute_write(
        self,
        callback: Callable[[sqlite3.Connection], _WriteResult],
    ) -> _WriteResult:
        with self._lock:
            self._ensure_open()
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                result = callback(self._conn)
                self._conn.commit()
                return result
            except Exception:
                self._conn.rollback()
                raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SessionDB is closed")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "SessionDB":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def create_session(
        self,
        session_id: str,
        source: str,
        **kwargs: Any,
    ) -> str:
        model_config = kwargs.get("model_config")
        if model_config is not None and not isinstance(model_config, str):
            model_config = json.dumps(
                model_config,
                ensure_ascii=False,
                default=_json_default,
            )

        def _do(conn: sqlite3.Connection) -> str:
            conn.execute(
                """INSERT INTO sessions (
                       id, source, model, model_config, system_prompt,
                       parent_session_id, started_at, title
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    source,
                    kwargs.get("model"),
                    model_config,
                    kwargs.get("system_prompt"),
                    kwargs.get("parent_session_id"),
                    float(kwargs.get("started_at") or time.time()),
                    kwargs.get("title"),
                ),
            )
            return session_id

        return self._execute_write(_do)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def end_session(self, session_id: str, end_reason: str) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                (time.time(), end_reason, session_id),
            )

        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> bool:
        """恢复会话时清除结束标记，与 Hermes ``reopen_session`` 同语义。"""
        def _do(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """UPDATE sessions SET ended_at = NULL, end_reason = NULL
                   WHERE id = ?""",
                (session_id,),
            )
            return cursor.rowcount > 0

        return self._execute_write(_do)

    def get_session_title(self, session_id: str) -> str | None:
        session = self.get_session(session_id)
        return session.get("title") if session else None

    def set_session_title(self, session_id: str, title: str) -> bool:
        normalized = title.strip()
        if not normalized:
            return False

        def _do(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (normalized, session_id),
            )
            return cursor.rowcount > 0

        return self._execute_write(_do)

    def update_session_usage(
        self,
        session_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        api_calls: int = 0,
    ) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                """UPDATE sessions
                   SET input_tokens = input_tokens + ?,
                       output_tokens = output_tokens + ?,
                       api_call_count = api_call_count + ?
                   WHERE id = ?""",
                (
                    max(0, int(input_tokens)),
                    max(0, int(output_tokens)),
                    max(0, int(api_calls)),
                    session_id,
                ),
            )

        self._execute_write(_do)

    def archive_session(self, session_id: str) -> bool:
        def _do(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "UPDATE sessions SET archived = 1 WHERE id = ?",
                (session_id,),
            )
            return cursor.rowcount > 0

        return self._execute_write(_do)

    def list_sessions_rich(self, *, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                """SELECT s.*,
                          COALESCE(MAX(m.timestamp), s.started_at) AS last_activity
                   FROM sessions s
                   LEFT JOIN messages m ON m.session_id = s.id AND m.active = 1
                   WHERE s.archived = 0
                   GROUP BY s.id
                   ORDER BY last_activity DESC, s.started_at DESC
                   LIMIT ?""",
                (safe_limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    @classmethod
    def _encode_content(cls, content: Any) -> str | None:
        if content is None or isinstance(content, str):
            return content
        return _CONTENT_JSON_PREFIX + json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        if isinstance(content, str) and content.startswith(_CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(_CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                return content
        return content

    @staticmethod
    def _count_tool_calls(content: Any, tool_calls: Any) -> int:
        if tool_calls is not None:
            return len(tool_calls) if isinstance(tool_calls, list) else 1
        if not isinstance(content, list):
            return 0
        count = 0
        for block in content:
            block_type = (
                block.get("type")
                if isinstance(block, Mapping)
                else getattr(block, "type", None)
            )
            if block_type == "tool_use":
                count += 1
        return count

    def append_message(
        self,
        session_id: str,
        role: str,
        content: Any = None,
        tool_name: str | None = None,
        tool_calls: Any = None,
        tool_call_id: str | None = None,
        token_count: int | None = None,
        finish_reason: str | None = None,
        timestamp: Any = None,
    ) -> int:
        stored_content = self._encode_content(content)
        tool_calls_json = (
            json.dumps(tool_calls, ensure_ascii=False, default=_json_default)
            if tool_calls is not None
            else None
        )
        message_timestamp = time.time()
        if timestamp is not None:
            try:
                message_timestamp = (
                    float(timestamp.timestamp())
                    if hasattr(timestamp, "timestamp")
                    else float(timestamp)
                )
            except (TypeError, ValueError):
                pass
        num_tool_calls = self._count_tool_calls(content, tool_calls)

        def _do(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """INSERT INTO messages (
                       session_id, role, content, tool_call_id, tool_calls,
                       tool_name, timestamp, token_count, finish_reason
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    stored_content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    message_timestamp,
                    token_count,
                    finish_reason,
                ),
            )
            conn.execute(
                """UPDATE sessions
                   SET message_count = message_count + 1,
                       tool_call_count = tool_call_count + ?
                   WHERE id = ?""",
                (num_tool_calls, session_id),
            )
            return int(cursor.lastrowid or 0)

        return self._execute_write(_do)

    def _insert_message_rows(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> tuple[int, int]:
        now = time.time()
        inserted = 0
        tool_call_count = 0
        for message in messages:
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            count = self._count_tool_calls(content, tool_calls)
            stored_tool_calls = (
                json.dumps(tool_calls, ensure_ascii=False, default=_json_default)
                if tool_calls is not None
                else None
            )
            conn.execute(
                """INSERT INTO messages (
                       session_id, role, content, tool_call_id, tool_calls,
                       tool_name, timestamp, token_count, finish_reason
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    message.get("role", "unknown"),
                    self._encode_content(content),
                    message.get("tool_call_id"),
                    stored_tool_calls,
                    message.get("tool_name"),
                    float(message.get("timestamp") or now),
                    message.get("token_count"),
                    message.get("finish_reason"),
                ),
            )
            inserted += 1
            tool_call_count += count
            now += 1e-6
        return inserted, tool_call_count

    def replace_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        def _do(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            inserted, tool_calls = self._insert_message_rows(
                conn,
                session_id,
                messages,
            )
            conn.execute(
                """UPDATE sessions
                   SET message_count = ?, tool_call_count = ? WHERE id = ?""",
                (inserted, tool_calls, session_id),
            )

        self._execute_write(_do)

    def archive_and_compact(
        self,
        session_id: str,
        compacted_messages: list[dict[str, Any]],
    ) -> int:
        def _do(conn: sqlite3.Connection) -> int:
            conn.execute(
                """UPDATE messages SET active = 0, compacted = 1
                   WHERE session_id = ? AND active = 1""",
                (session_id,),
            )
            inserted, tool_calls = self._insert_message_rows(
                conn,
                session_id,
                compacted_messages,
            )
            conn.execute(
                """UPDATE sessions
                   SET message_count = ?, tool_call_count = ? WHERE id = ?""",
                (inserted, tool_calls, session_id),
            )
            return inserted

        return self._execute_write(_do)

    def rotate_after_compression(
        self,
        old_session_id: str,
        new_session_id: str,
        compacted_messages: list[dict[str, Any]],
    ) -> str:
        """结束父会话并建立 continuation；整个切换在单一事务中完成。"""
        def _do(conn: sqlite3.Connection) -> str:
            parent = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (old_session_id,),
            ).fetchone()
            if parent is None:
                raise KeyError(f"会话不存在：{old_session_id}")
            conn.execute(
                """UPDATE sessions SET ended_at = ?, end_reason = 'compression'
                   WHERE id = ?""",
                (time.time(), old_session_id),
            )
            conn.execute(
                """INSERT INTO sessions (
                       id, source, model, model_config, system_prompt,
                       parent_session_id, started_at, title
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_session_id,
                    parent["source"],
                    parent["model"],
                    parent["model_config"],
                    parent["system_prompt"],
                    old_session_id,
                    time.time(),
                    parent["title"],
                ),
            )
            inserted, tool_calls = self._insert_message_rows(
                conn,
                new_session_id,
                compacted_messages,
            )
            conn.execute(
                """UPDATE sessions
                   SET message_count = ?, tool_call_count = ? WHERE id = ?""",
                (inserted, tool_calls, new_session_id),
            )
            return new_session_id

        return self._execute_write(_do)

    def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        active_clause = "" if include_inactive else " AND active = 1"
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ?"
                + active_clause
                + " ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        result = []
        for row in rows:
            message = dict(row)
            message["content"] = self._decode_content(message.get("content"))
            if message.get("tool_calls"):
                try:
                    message["tool_calls"] = json.loads(message["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    message["tool_calls"] = []
            result.append(message)
        return result

    def get_messages_as_conversation(
        self,
        session_id: str,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        messages = self.get_messages(session_id, include_inactive=include_inactive)
        conversation = []
        for message in messages:
            item = {
                "role": message["role"],
                "content": message.get("content"),
            }
            for key in ("tool_call_id", "tool_calls", "tool_name"):
                if message.get(key) is not None:
                    item[key] = message[key]
            conversation.append(item)
        return conversation

    def get_compression_tip(self, session_id: str) -> str:
        current = session_id
        seen = {current}
        with self._lock:
            self._ensure_open()
            for _ in range(32):
                row = self._conn.execute(
                    """SELECT child.id
                       FROM sessions child
                       JOIN sessions parent ON parent.id = child.parent_session_id
                       WHERE child.parent_session_id = ?
                         AND parent.end_reason = 'compression'
                       ORDER BY child.started_at DESC, child.id DESC
                       LIMIT 1""",
                    (current,),
                ).fetchone()
                if row is None:
                    break
                child_id = str(row["id"])
                if child_id in seen:
                    break
                seen.add(child_id)
                current = child_id
        return current

    def resolve_resume_session_id(self, session_id: str) -> str:
        tip = self.get_compression_tip(session_id)
        if self.get_session(tip) is None:
            return session_id
        return tip

    @staticmethod
    def _fts_phrase(query: str) -> str:
        normalized = " ".join(query.strip().split())
        if not normalized:
            return ""
        return '"' + normalized.replace('"', '""') + '"'

    def search_messages(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not self._fts_enabled:
            raise RuntimeError("当前 SQLite 不支持 FTS5，会话保存与恢复仍可使用")
        phrase = self._fts_phrase(query)
        if not phrase:
            return []
        safe_limit = max(1, min(int(limit), 100))
        use_trigram = self._trigram_enabled and _CJK_RUN_RE.search(query) is not None
        table = "messages_fts_trigram" if use_trigram else "messages_fts"
        with self._lock:
            self._ensure_open()
            try:
                rows = self._conn.execute(
                    f"""SELECT m.id AS message_id, m.session_id, m.role,
                               m.timestamp, m.active, m.compacted,
                               snippet({table}, 0, '>>>', '<<<', '…', 18) AS snippet,
                               bm25({table}) AS rank
                        FROM {table}
                        JOIN messages m ON m.id = {table}.rowid
                        JOIN sessions s ON s.id = m.session_id
                        WHERE {table} MATCH ?
                          AND (m.active = 1 OR m.compacted = 1)
                          AND s.archived = 0
                        ORDER BY rank, m.timestamp DESC
                        LIMIT ?""",
                    (phrase, safe_limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            return [dict(row) for row in rows]
