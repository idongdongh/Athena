"""宿主机命令后端，保持 Athena 原有 bash 行为。"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time

from agent.interrupt_controller import ToolExecutionCancelled, interrupt_controller
from tools.environments.base import (
    CommandResult,
    TERMINATE_GRACE_SECONDS,
    read_capped_output,
)


def stop_process(process: subprocess.Popen) -> None:
    """终止命令创建的整个宿主进程组。"""

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


class LocalEnvironment:
    """直接在 Athena 进程 cwd 下执行一次性 shell。"""

    def execute(self, command: str, *, timeout: int) -> CommandResult:
        process: subprocess.Popen | None = None
        try:
            with tempfile.TemporaryFile() as stdout_sink, tempfile.TemporaryFile() as stderr_sink:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=stdout_sink,
                    stderr=stderr_sink,
                    start_new_session=(os.name == "posix"),
                )
                started_at = time.monotonic()
                while True:
                    if interrupt_controller.is_requested():
                        stop_process(process)
                        raise ToolExecutionCancelled("bash interrupted by user")

                    remaining = timeout - (time.monotonic() - started_at)
                    if remaining <= 0:
                        stop_process(process)
                        raise TimeoutError(f"Timeout ({timeout}s)")

                    try:
                        process.wait(timeout=min(0.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue

                stdout, stdout_truncated = read_capped_output(stdout_sink)
                stderr, stderr_truncated = read_capped_output(stderr_sink)
        except KeyboardInterrupt:
            if process is not None:
                stop_process(process)
            raise
        except ToolExecutionCancelled:
            raise
        except TimeoutError:
            raise
        except Exception:
            if process is not None:
                stop_process(process)
            raise

        return CommandResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            truncated=stdout_truncated or stderr_truncated,
        )

    def close(self) -> None:
        """Local 后端不持有跨调用资源。"""
