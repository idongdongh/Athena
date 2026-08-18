"""评测任务清单与约束校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    source: Path
    prompt: str
    test_command: tuple[str, ...]
    timeout_seconds: int = 900
    base_commit: str | None = None
    tags: tuple[str, ...] = ()


def load_tasks(path: str | Path) -> list[EvaluationTask]:
    manifest = Path(path).expanduser().resolve()
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw_tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("task manifest must contain a non-empty tasks list")
    tasks: list[EvaluationTask] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            raise ValueError(f"task {index} must be a mapping")
        task_id = raw.get("id")
        prompt = raw.get("prompt")
        source_value = raw.get("source")
        command = raw.get("test_command")
        if not isinstance(task_id, str) or not task_id.strip() or task_id in seen:
            raise ValueError(f"task {index} has invalid or duplicate id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {task_id} has invalid prompt")
        if not isinstance(source_value, str) or not source_value.strip():
            raise ValueError(f"task {task_id} has invalid source")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError(f"task {task_id} test_command must be a non-empty string list")
        source = Path(source_value).expanduser()
        if not source.is_absolute():
            source = (manifest.parent / source).resolve()
        if not source.is_dir():
            raise ValueError(f"task {task_id} source directory does not exist: {source}")
        timeout = raw.get("timeout_seconds", 900)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError(f"task {task_id} has invalid timeout_seconds")
        tags = raw.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"task {task_id} tags must be strings")
        base_commit = raw.get("base_commit")
        if base_commit is not None and (not isinstance(base_commit, str) or not base_commit.strip()):
            raise ValueError(f"task {task_id} has invalid base_commit")
        seen.add(task_id)
        tasks.append(
            EvaluationTask(
                task_id=task_id,
                source=source,
                prompt=prompt.strip(),
                test_command=tuple(command),
                timeout_seconds=timeout,
                base_commit=base_commit,
                tags=tuple(tags),
            )
        )
    return tasks
