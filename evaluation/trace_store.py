"""Append-only JSONL 轨迹与大文本 artifact 存储。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.schema import TRACE_SCHEMA_VERSION, TraceEvent


_SENSITIVE_KEY_RE = re.compile(
    r"^(?:api[_-]?key|authorization|password|passwd|secret|client_secret|"
    r"access_token|refresh_token|auth_token|bearer|cookie|set_cookie)$",
    re.IGNORECASE,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    return str(value)


def redact_sensitive(value: Any) -> Any:
    """递归脱敏常见凭证字段，保留其他评估证据。"""
    safe = _json_safe(value)
    if isinstance(safe, dict):
        return {
            key: "[REDACTED]" if _SENSITIVE_KEY_RE.search(key) else redact_sensitive(item)
            for key, item in safe.items()
        }
    if isinstance(safe, list):
        return [redact_sensitive(item) for item in safe]
    return safe


class JsonlTraceRecorder:
    """一次 Agent turn/run 的线程安全轨迹记录器。"""

    def __init__(
        self,
        root: str | Path,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        inline_text_limit: int = 4_000,
    ) -> None:
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex}"
        self.task_id = task_id
        self.session_id = session_id
        self.inline_text_limit = max(256, inline_text_limit)
        self.root = Path(root).expanduser().resolve() / self.trace_id
        self.artifact_dir = self.root / "artifacts"
        self.events_path = self.root / "events.jsonl"
        self.root.mkdir(parents=True, exist_ok=False)
        self.artifact_dir.mkdir()
        self._lock = threading.Lock()
        self._next_event_id = 1
        self._started = time.monotonic()

    def _store_text(self, event_id: int, field_name: str, value: str) -> dict[str, Any]:
        encoded = value.encode("utf-8", errors="replace")
        digest = hashlib.sha256(encoded).hexdigest()
        suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", field_name)[:40] or "content"
        path = self.artifact_dir / f"event_{event_id:06d}_{suffix}.txt"
        path.write_bytes(encoded)
        return {
            "artifact_ref": str(path.relative_to(self.root)),
            "chars": len(value),
            "sha256": digest,
            "preview": value[: self.inline_text_limit],
            "truncated": True,
        }

    def _externalize_large_text(self, event_id: int, value: Any, field_name: str = "payload") -> Any:
        if isinstance(value, str) and len(value) > self.inline_text_limit:
            return self._store_text(event_id, field_name, value)
        if isinstance(value, dict):
            return {
                key: self._externalize_large_text(event_id, item, key)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._externalize_large_text(event_id, item, f"{field_name}_{index}")
                for index, item in enumerate(value)
            ]
        return value

    def emit(self, event_type: str, payload: Mapping[str, Any] | None = None) -> None:
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        raw = dict(payload or {})
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            sanitized = redact_sensitive(raw)
            materialized = self._externalize_large_text(event_id, sanitized)
            turn_id = materialized.pop("turn_id", None)
            step_id = materialized.pop("step_id", None)
            session_id = materialized.pop("session_id", self.session_id)
            event = TraceEvent(
                schema_version=TRACE_SCHEMA_VERSION,
                trace_id=self.trace_id,
                event_id=event_id,
                event_type=event_type.strip(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_ms=round((time.monotonic() - self._started) * 1000),
                task_id=self.task_id,
                session_id=str(session_id) if session_id is not None else None,
                turn_id=str(turn_id) if turn_id is not None else None,
                step_id=step_id if isinstance(step_id, int) else None,
                payload=materialized,
            )
            line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + os.linesep)


def load_trace_events(path: str | Path) -> list[dict[str, Any]]:
    """加载并执行基础完整性校验。"""
    source = Path(path)
    if source.is_dir():
        source = source / "events.jsonl"
    events: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"line {line_number}: event must be an object")
            events.append(event)
    expected_ids = list(range(1, len(events) + 1))
    actual_ids = [event.get("event_id") for event in events]
    if actual_ids != expected_ids:
        raise ValueError("trace event_id sequence is incomplete or out of order")
    trace_ids = {event.get("trace_id") for event in events}
    if len(trace_ids) > 1:
        raise ValueError("events.jsonl contains multiple trace_id values")
    return events


def iter_event_types(events: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(event.get("event_type", "")) for event in events]
