"""本地 Docker/Podman 命令沙箱。"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from agent.interrupt_controller import ToolExecutionCancelled, interrupt_controller
from athena_cli.config import DockerSettings
from tools.environments.base import (
    CommandResult,
    TERMINATE_GRACE_SECONDS,
    read_capped_output,
)


_CONTAINER_GONE_MARKERS = (
    "no such container",
    "is not running",
    "container state improper",
)


def find_container_runtime(requested: str) -> str:
    """解析 Docker 兼容 CLI，不返回未经验证的命令名。"""

    candidates = ("docker", "podman") if requested == "auto" else (requested,)
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable:
            return executable
    expected = "Docker 或 Podman" if requested == "auto" else requested
    raise RuntimeError(f"找不到容器运行时：{expected}")


def _stop_host_process(process: subprocess.Popen) -> None:
    """停止宿主侧 docker/podman CLI 及其进程组。"""

    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


class DockerEnvironment:
    """单 Athena 进程持有的本地容器执行环境。"""

    def __init__(self, settings: DockerSettings, workspace: Path) -> None:
        self.settings = settings
        self.workspace = workspace.resolve()
        if not self.workspace.is_dir():
            raise RuntimeError(f"工作区不存在或不是目录：{self.workspace}")
        if ":" in str(self.workspace):
            raise RuntimeError("Docker 第一阶段不支持路径中包含冒号的工作区")

        self.runtime = find_container_runtime(settings.runtime)
        self._container_id: str | None = None
        self._container_name = f"athena-{uuid.uuid4().hex[:12]}"
        self._closed = False
        self._state_lock = threading.RLock()
        self._probe_runtime()

    def _probe_runtime(self) -> None:
        try:
            result = subprocess.run(
                [self.runtime, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"容器运行时不可用：{exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit {result.returncode}"
            raise RuntimeError(f"容器运行时不可用：{detail}")

    def _run_control(self, args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.runtime, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )

    def _ensure_container(self) -> str:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("命令环境已经关闭")
            if self._container_id is not None:
                return self._container_id

            mode = "ro" if self.settings.workspace_read_only else "rw"
            args = [
                "run",
                "-d",
                "--rm",
                "--name",
                self._container_name,
                "--label",
                "athena-sandbox=1",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(self.settings.pids_limit),
                "--cpus",
                str(self.settings.cpu),
                "--memory",
                f"{self.settings.memory_mb}m",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
                "--volume",
                f"{self.workspace}:{self.workspace}:{mode}",
                "--workdir",
                str(self.workspace),
            ]
            if hasattr(os, "getuid") and hasattr(os, "getgid"):
                args.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
            if not self.settings.network:
                args.extend(["--network", "none"])
            args.extend([self.settings.image, "sleep", "infinity"])

            try:
                created = self._run_control(args, timeout=self.settings.startup_timeout)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._remove_by_name()
                raise RuntimeError(f"创建沙箱容器失败：{exc}") from exc
            if created.returncode != 0:
                self._remove_by_name()
                detail = created.stderr.strip() or created.stdout.strip()
                raise RuntimeError(f"创建沙箱容器失败：{detail or created.returncode}")

            container_id = created.stdout.strip()
            if not container_id:
                self._remove_by_name()
                raise RuntimeError("创建沙箱容器失败：运行时没有返回容器 ID")
            self._container_id = container_id

            requirements = self._run_control(
                [
                    "exec",
                    container_id,
                    "bash",
                    "-c",
                    "command -v bash >/dev/null && command -v setsid >/dev/null && command -v sleep >/dev/null",
                ],
                timeout=15,
            )
            if requirements.returncode != 0:
                detail = requirements.stderr.strip() or "镜像缺少 bash、setsid 或 sleep"
                self._reset_container()
                raise RuntimeError(f"沙箱镜像不满足运行要求：{detail}")
            return container_id

    def _remove_by_name(self) -> None:
        try:
            self._run_control(["rm", "-f", self._container_name], timeout=15)
        except Exception:
            pass

    def _reset_container(self) -> None:
        with self._state_lock:
            container_id, self._container_id = self._container_id, None
        target = container_id or self._container_name
        try:
            self._run_control(["rm", "-f", target], timeout=15)
        except Exception:
            pass

    def _kill_container_process_group(self, container_id: str, pid_file: str) -> bool:
        """终止本次 exec 的容器内进程组；无法确认时返回 False。"""

        quoted = pid_file.replace("'", "'\\''")
        terminate_script = (
            f"test -s '{quoted}' || exit 3; "
            f"pid=$(cat '{quoted}'); "
            "kill -TERM -- -\"$pid\" 2>/dev/null || true; "
            "for _ in 1 2 3 4 5; do "
            "kill -0 -- -\"$pid\" 2>/dev/null || break; sleep 0.1; "
            "done; "
            "kill -KILL -- -\"$pid\" 2>/dev/null || true; sleep 0.1; "
            "if kill -0 -- -\"$pid\" 2>/dev/null; then exit 4; fi; "
            f"rm -f '{quoted}'"
        )
        try:
            result = self._run_control(
                ["exec", container_id, "bash", "-c", terminate_script],
                timeout=5,
            )
        except Exception:
            return False
        return result.returncode == 0

    def execute(self, command: str, *, timeout: int) -> CommandResult:
        container_id = self._ensure_container()
        execution_id = uuid.uuid4().hex
        pid_file = f"/tmp/athena-exec-{execution_id}.pid"
        wrapper = (
            "set +e\n"
            "pid_file=$1\n"
            "user_command=$2\n"
            "setsid bash -c \"$user_command\" &\n"
            "child=$!\n"
            "printf '%s' \"$child\" > \"$pid_file\"\n"
            "wait \"$child\"\n"
            "status=$?\n"
            # 第一阶段不支持后台进程：清理用户 shell 退出后仍留在同一进程组的子进程。
            "kill -TERM -- -\"$child\" 2>/dev/null || true\n"
            "sleep 0.05\n"
            "kill -KILL -- -\"$child\" 2>/dev/null || true\n"
            "rm -f \"$pid_file\"\n"
            "exit \"$status\"\n"
        )
        cli_args = [
            self.runtime,
            "exec",
            container_id,
            "bash",
            "-c",
            wrapper,
            "athena-exec",
            pid_file,
            command,
        ]

        process: subprocess.Popen | None = None
        interrupted = False
        timed_out = False
        try:
            with tempfile.TemporaryFile() as stdout_sink, tempfile.TemporaryFile() as stderr_sink:
                process = subprocess.Popen(
                    cli_args,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_sink,
                    stderr=stderr_sink,
                    start_new_session=(os.name == "posix"),
                )
                started_at = time.monotonic()
                while True:
                    if interrupt_controller.is_requested():
                        interrupted = True
                        break
                    remaining = timeout - (time.monotonic() - started_at)
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        process.wait(timeout=min(0.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue

                if interrupted or timed_out:
                    killed = self._kill_container_process_group(container_id, pid_file)
                    _stop_host_process(process)
                    if not killed:
                        self._reset_container()
                    if interrupted:
                        raise ToolExecutionCancelled("bash interrupted by user")
                    raise TimeoutError(f"Timeout ({timeout}s)")

                stdout, stdout_truncated = read_capped_output(stdout_sink)
                stderr, stderr_truncated = read_capped_output(stderr_sink)
        except (ToolExecutionCancelled, TimeoutError):
            raise
        except KeyboardInterrupt:
            if process is not None:
                killed = self._kill_container_process_group(container_id, pid_file)
                _stop_host_process(process)
                if not killed:
                    self._reset_container()
            raise
        except Exception:
            if process is not None:
                _stop_host_process(process)
            raise

        lowered_error = stderr.lower()
        if process.returncode != 0 and any(marker in lowered_error for marker in _CONTAINER_GONE_MARKERS):
            with self._state_lock:
                self._container_id = None
            raise RuntimeError(f"沙箱容器不可用：{stderr.strip()}")

        return CommandResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            truncated=stdout_truncated or stderr_truncated,
        )

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._reset_container()
