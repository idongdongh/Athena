"""持久化文件记忆：MEMORY.md、USER.md 与单一 ``memory`` 工具。"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tools.registry import registry
from tools.threat_patterns import first_threat_message, scan_for_threats

try:
    import fcntl
except ImportError:  # pragma: no cover - Athena 当前只支持 macOS/Linux
    fcntl = None


ENTRY_DELIMITER = "\n§\n"
def _scan_memory_content(content: str) -> str | None:
    """使用共享 strict 规则阻断持久化 prompt injection 与外传载荷。"""
    if any(ord(char) < 32 and char not in "\n\t" for char in content):
        return "Memory content contains unsupported control characters."
    return first_threat_message(content, scope="strict")


class MemoryStore:
    """每个 Agent 一份 live state，并持有会话级冻结 prompt 快照。"""

    def __init__(
        self,
        memory_dir: Path,
        *,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.memory_entries: list[str] = []
        self.user_entries: list[str] = []
        self._system_prompt_snapshot = {"memory": "", "user": ""}

    def load_from_disk(self) -> None:
        """加载 live state，并捕获本会话不随写操作变化的安全快照。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_entries = list(dict.fromkeys(self._read_file(self._path_for("memory"))))
        self.user_entries = list(dict.fromkeys(self._read_file(self._path_for("user"))))
        safe_memory = self._sanitize_for_snapshot(self.memory_entries, "MEMORY.md")
        safe_user = self._sanitize_for_snapshot(self.user_entries, "USER.md")
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", safe_memory),
            "user": self._render_block("user", safe_user),
        }

    def add(self, target: str, content: str | None) -> dict[str, Any]:
        try:
            content = self._validate_content(content)
        except (TypeError, ValueError) as exc:
            return self._error_with_entries(target, str(exc))
        path = self._path_for(target)
        with self._file_lock(path):
            raw = self._read_raw_file(path)
            entries = list(dict.fromkeys(self._parse_entries(raw)))
            self._set_entries(target, entries)
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")
            candidate = [*entries, content]
            over = self._budget_error(target, candidate)
            if over:
                return over
            # add 不重排或重新序列化已有内容。这样即使用户在外部手工编辑了
            # MEMORY.md，成功追加也会逐字保留原文，只在末尾增加分隔符和新条目。
            separator = "" if not raw or raw.endswith(ENTRY_DELIMITER) else ENTRY_DELIMITER
            self._write_raw_file(path, f"{raw}{separator}{content}")
            self._set_entries(target, list(dict.fromkeys(self._read_file(path))))
        return self._success_response(target, "Entry added.")

    def replace(
        self,
        target: str,
        old_text: str,
        new_content: str | None,
    ) -> dict[str, Any]:
        old_text = old_text.strip()
        if not old_text:
            return self._error_with_entries(target, "old_text cannot be empty.")
        try:
            new_content = self._validate_content(new_content)
        except (TypeError, ValueError) as exc:
            return self._error_with_entries(target, str(exc))
        return self._rewrite_one(target, old_text, new_content)

    def remove(self, target: str, old_text: str) -> dict[str, Any]:
        old_text = old_text.strip()
        if not old_text:
            return self._error_with_entries(target, "old_text cannot be empty.")
        return self._rewrite_one(target, old_text, None)

    def apply_batch(self, target: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        if not operations:
            return self._error_with_entries(target, "operations cannot be empty.")
        path = self._path_for(target)
        with self._file_lock(path):
            drift_backup = self._detect_external_drift(target)
            entries = list(dict.fromkeys(self._read_file(path)))
            self._set_entries(target, entries)
            if drift_backup:
                return self._drift_error(path, drift_backup)
            working = entries.copy()
            try:
                for index, operation in enumerate(operations, 1):
                    if not isinstance(operation, dict):
                        raise ValueError(f"operation {index} must be an object")
                    action = operation.get("action")
                    content = operation.get("content")
                    old_text = operation.get("old_text")
                    if action == "add":
                        validated = self._validate_content(content)
                        if validated not in working:
                            working.append(validated)
                    elif action in {"replace", "remove"}:
                        needle = old_text.strip() if isinstance(old_text, str) else ""
                        match = self._unique_match(working, needle)
                        if action == "replace":
                            working[match] = self._validate_content(content)
                        else:
                            working.pop(match)
                    else:
                        raise ValueError(f"operation {index} has unknown action")
            except (TypeError, ValueError) as exc:
                return self._error_with_entries(
                    target, f"{exc}. No operations were applied (batch is all-or-nothing)."
                )
            over = self._budget_error(target, working)
            if over:
                over["error"] += " No operations were applied (batch is all-or-nothing)."
                return over
            self._write_file(path, working)
            self._set_entries(target, working)
        return self._success_response(target, f"Applied {len(operations)} operation(s).")

    def format_for_system_prompt(self, target: str) -> str | None:
        block = self._system_prompt_snapshot.get(target, "")
        return block or None

    def _rewrite_one(self, target: str, old_text: str, replacement: str | None) -> dict[str, Any]:
        path = self._path_for(target)
        with self._file_lock(path):
            drift_backup = self._detect_external_drift(target)
            entries = list(dict.fromkeys(self._read_file(path)))
            self._set_entries(target, entries)
            if drift_backup:
                return self._drift_error(path, drift_backup)
            try:
                index = self._unique_match(entries, old_text)
            except ValueError as exc:
                return self._error_with_entries(target, str(exc))
            candidate = entries.copy()
            if replacement is None:
                candidate.pop(index)
            else:
                candidate[index] = replacement
            over = self._budget_error(target, candidate)
            if over:
                return over
            self._write_file(path, candidate)
            self._set_entries(target, candidate)
        return self._success_response(target, "Entry removed." if replacement is None else "Entry replaced.")

    @staticmethod
    def _unique_match(entries: list[str], old_text: str) -> int:
        if not old_text:
            raise ValueError("old_text is required")
        matches = [index for index, entry in enumerate(entries) if old_text in entry]
        if not matches:
            raise ValueError(f"No entry matched '{old_text}'")
        if len(matches) > 1:
            raise ValueError(f"'{old_text}' matched multiple entries; use a more specific substring")
        return matches[0]

    @staticmethod
    def _validate_content(content: Any) -> str:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content cannot be empty")
        value = content.strip()
        unsafe = _scan_memory_content(value)
        if unsafe:
            raise ValueError(unsafe)
        return value

    def _path_for(self, target: str) -> Path:
        if target == "user":
            return self.memory_dir / "USER.md"
        if target == "memory":
            return self.memory_dir / "MEMORY.md"
        raise ValueError("target must be 'memory' or 'user'")

    def _entries_for(self, target: str) -> list[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: list[str]) -> None:
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _budget_error(self, target: str, entries: list[str]) -> dict[str, Any] | None:
        total = len(ENTRY_DELIMITER.join(entries)) if entries else 0
        limit = self._char_limit(target)
        if total <= limit:
            return None
        return {
            "success": False,
            "error": f"Memory would exceed its character limit ({total:,}/{limit:,}).",
            "current_entries": self._entries_for(target),
            "usage": f"{self._char_count(target):,}/{limit:,}",
        }

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        return len(ENTRY_DELIMITER.join(entries)) if entries else 0

    def _success_response(self, target: str, message: str) -> dict[str, Any]:
        current = self._char_count(target)
        limit = self._char_limit(target)
        return {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{min(100, int(current / limit * 100))}% — {current:,}/{limit:,} chars",
            "entry_count": len(self._entries_for(target)),
            "message": message,
            "note": "Write saved. This update is complete — do not repeat it.",
        }

    def _error_with_entries(self, target: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "error": message,
            "current_entries": self._entries_for(target),
            "usage": f"{self._char_count(target):,}/{self._char_limit(target):,}",
        }

    def _sanitize_for_snapshot(self, entries: list[str], filename: str) -> list[str]:
        sanitized = []
        for entry in entries:
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; use "
                    "memory(action=remove) to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    def _render_block(self, target: str, entries: list[str]) -> str:
        if not entries:
            return ""
        content = ENTRY_DELIMITER.join(entries)
        limit = self._char_limit(target)
        label = "USER PROFILE (who the user is)" if target == "user" else "MEMORY (your personal notes)"
        separator = "═" * 46
        return f"{separator}\n{label} [{min(100, int(len(content) / limit * 100))}% — {len(content):,}/{limit:,} chars]\n{separator}\n{content}"

    @contextmanager
    def _file_lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(path.with_suffix(path.suffix + ".lock"), "a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    @staticmethod
    def _read_file(path: Path) -> list[str]:
        raw = MemoryStore._read_raw_file(path)
        return MemoryStore._parse_entries(raw)

    @staticmethod
    def _read_raw_file(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _parse_entries(raw: str) -> list[str]:
        return [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]

    @staticmethod
    def _write_file(path: Path, entries: list[str]) -> None:
        MemoryStore._write_raw_file(path, ENTRY_DELIMITER.join(entries))

    @staticmethod
    def _write_raw_file(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".mem_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _detect_external_drift(self, target: str) -> str | None:
        path = self._path_for(target)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return None
        parsed = [entry.strip() for entry in raw.split(ENTRY_DELIMITER) if entry.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)
        if raw.strip() == roundtrip and max(map(len, parsed), default=0) <= self._char_limit(target):
            return None
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        backup.write_text(raw, encoding="utf-8")
        return str(backup)

    @staticmethod
    def _drift_error(path: Path, backup: str) -> dict[str, Any]:
        return {
            "success": False,
            "error": f"Refusing to rewrite {path.name}: external file drift was detected.",
            "drift_backup": backup,
        }


def memory_tool(
    *,
    action: str | None = None,
    target: str = "memory",
    content: str | None = None,
    old_text: str | None = None,
    operations: list[dict[str, Any]] | None = None,
    store: MemoryStore | None = None,
) -> str:
    """校验模型参数并把单步或批量操作分发给当前 Agent 的 Store。"""
    if store is None:
        return json.dumps({
            "success": False,
            "error": "Memory is not available. It may be disabled in config or this environment.",
        })
    try:
        store._path_for(target)
        if operations is not None:
            if not isinstance(operations, list):
                result = {
                    "success": False,
                    "error": "operations must be a list of {action, content?, old_text?} objects.",
                }
            else:
                result = store.apply_batch(target, operations)
        elif action == "add":
            if not content:
                result = store._error_with_entries(target, "Content is required for 'add' action.")
            else:
                result = store.add(target, content)
        elif action == "replace":
            if not old_text:
                result = _missing_old_text_error(store, target, "replace")
            elif not content:
                result = store._error_with_entries(
                    target,
                    "content is required for 'replace' action. Use action='remove' to delete the entry.",
                )
            else:
                result = store.replace(target, old_text, content)
        elif action == "remove":
            if not old_text:
                result = _missing_old_text_error(store, target, "remove")
            else:
                result = store.remove(target, old_text)
        else:
            result = {
                "success": False,
                "error": f"Unknown action '{action}'. Use: add, replace, remove",
            }
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        result = {"success": False, "error": str(exc)}
    return json.dumps(result, ensure_ascii=False)


def _missing_old_text_error(store: MemoryStore, target: str, action: str) -> dict[str, Any]:
    """返回当前条目与明确重试方式，让模型能修正缺失的定位参数。"""
    return store._error_with_entries(
        target,
        (
            f"'{action}' needs old_text -- a short unique substring of the entry to {action}. "
            f"None was provided. Reissue the {action} with old_text set to part of one of "
            "the current_entries below."
        ),
    )


MEMORY_TOOL = {
    "name": "memory",
    "description": (
        "Save durable facts to persistent memory that survive across sessions. Memory is "
        "injected into every future turn, so keep entries compact and high-signal.\n\n"
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?}). The batch applies atomically and the char limit is "
        "checked only on the FINAL result — so a single call can remove/replace stale entries "
        "to free room AND add new ones, even when an add alone would overflow. The response "
        "reports current/limit chars and confirms completion; one batch call finishes the "
        "update, so don't repeat it. Use the bare action/content/old_text fields only for a "
        "single lone change.\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF FULL: an add is rejected with the current entries shown. Reissue as ONE batch that "
        "removes or shortens enough stale entries and adds the new one together.\n\n"
        "TARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your "
        "notes (environment, conventions, tool quirks, lessons).\n\n"
        "SKIP: trivial or obvious information, easily rediscovered facts, raw data dumps, task "
        "progress, completed-work logs, temporary TODO state, commit or issue identifiers, or "
        "anything likely stale within a week. Keep those in session history or project docs. "
        "Write declarative facts, not instructions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform (single-op shape). Omit when using 'operations'.",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile.",
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape).",
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'.",
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call against "
                    "the final char budget. Preferred when making multiple changes or consolidating "
                    "to make room. Each item is {action, content?, old_text?}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
    },
}

registry.register(name="memory", schema=MEMORY_TOOL, handler=memory_tool, toolset="memory")
