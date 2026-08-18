"""隔离工作区、批量运行与断点续跑。"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.tasks import EvaluationTask, load_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_command(command: list[str], *, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def prepare_workspace(task: EvaluationTask, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    git_dir = task.source / ".git"
    if git_dir.exists():
        result = _run_command(
            ["git", "clone", "--quiet", "--no-hardlinks", str(task.source), str(destination)],
            cwd=task.source.parent,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
        if task.base_commit:
            checkout = _run_command(["git", "checkout", "--quiet", task.base_commit], cwd=destination)
            if checkout.returncode != 0:
                raise RuntimeError(f"git checkout failed: {checkout.stderr.strip()}")
    else:
        shutil.copytree(task.source, destination)
        init = _run_command(["git", "init", "--quiet"], cwd=destination)
        if init.returncode != 0:
            raise RuntimeError(f"git init failed: {init.stderr.strip()}")
        _run_command(["git", "add", "-A"], cwd=destination)
        commit = _run_command(
            [
                "git", "-c", "user.name=Athena Eval", "-c", "user.email=eval@localhost",
                "commit", "--quiet", "-m", "evaluation baseline",
            ],
            cwd=destination,
        )
        if commit.returncode != 0:
            raise RuntimeError(f"baseline commit failed: {commit.stderr.strip()}")


def _task_payload(task: EvaluationTask, workspace: Path, trace_root: Path) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "prompt": task.prompt,
        "test_command": list(task.test_command),
        "workspace": str(workspace),
        "trace_root": str(trace_root),
        "tags": list(task.tags),
    }


def _run_worker(payload_path: Path, workspace: Path, timeout: int) -> tuple[int, str, str, bool]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    process = subprocess.Popen(
        [sys.executable, "-m", "evaluation.worker", "--task", str(payload_path)],
        cwd=workspace,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return 124, stdout, stderr, True


def run_manifest(
    manifest: str | Path,
    output_root: str | Path,
    *,
    repetitions: int = 1,
    resume: bool = True,
) -> Path:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    tasks = load_tasks(manifest)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for repetition in range(1, repetitions + 1):
            run_dir = root / f"{task.task_id}__r{repetition}"
            result_path = run_dir / "runner_result.json"
            if resume and result_path.exists():
                try:
                    previous = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    previous = {}
                if previous.get("status") == "finished":
                    continue
            run_dir.mkdir(parents=True, exist_ok=True)
            workspace = run_dir / "workspace"
            trace_root = run_dir / "traces"
            started = time.monotonic()
            try:
                prepare_workspace(task, workspace)
                payload_path = run_dir / "task.json"
                payload_path.write_text(
                    json.dumps(_task_payload(task, workspace, trace_root), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                returncode, stdout, stderr, timed_out = _run_worker(
                    payload_path, workspace, task.timeout_seconds
                )
                result = {
                    "status": "finished",
                    "task_id": task.task_id,
                    "repetition": repetition,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    "stdout": stdout,
                    "stderr": stderr,
                }
            except Exception as exc:
                result = {
                    "status": "runner_error",
                    "task_id": task.task_id,
                    "repetition": repetition,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return root
