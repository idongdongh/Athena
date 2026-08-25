"""进程级命令环境配置与生命周期。"""

from __future__ import annotations

import atexit
import threading
from pathlib import Path

from athena_cli.config import TerminalSettings
from tools.environments.base import CommandEnvironment, CommandResult
from tools.environments.local import LocalEnvironment


class CommandEnvironmentManager:
    """按配置惰性创建一个命令后端，并提供幂等清理。"""

    def __init__(self, settings: TerminalSettings, workspace: Path) -> None:
        self.settings = settings
        self.workspace = workspace.expanduser().resolve()
        self._environment: CommandEnvironment | None = None
        self._lock = threading.RLock()

    def _create_environment(self) -> CommandEnvironment:
        if self.settings.backend == "local":
            return LocalEnvironment()
        if self.settings.backend == "docker":
            from tools.environments.docker import DockerEnvironment

            return DockerEnvironment(self.settings.docker, self.workspace)
        raise RuntimeError(f"不支持的命令执行后端：{self.settings.backend}")

    def execute(self, command: str, *, timeout: int | None = None) -> CommandResult:
        effective_timeout = self.settings.timeout if timeout is None else timeout
        if isinstance(effective_timeout, bool) or not isinstance(effective_timeout, int) or effective_timeout <= 0:
            raise ValueError("timeout 必须是正整数")

        # 第一阶段 bash 串行执行，确保超时后的容器重置不会误伤并行命令。
        with self._lock:
            if self._environment is None:
                self._environment = self._create_environment()
            return self._environment.execute(command, timeout=effective_timeout)

    def close(self) -> None:
        with self._lock:
            environment, self._environment = self._environment, None
        if environment is not None:
            environment.close()


_manager_lock = threading.RLock()
_manager = CommandEnvironmentManager(TerminalSettings(), Path.cwd())


def configure_command_environment(settings: TerminalSettings, workspace: Path | str) -> None:
    """替换进程级 manager；若旧后端已创建，先清理旧资源。"""

    global _manager
    replacement = CommandEnvironmentManager(settings, Path(workspace))
    with _manager_lock:
        previous, _manager = _manager, replacement
    previous.close()


def execute_command(command: str, *, timeout: int | None = None) -> CommandResult:
    with _manager_lock:
        manager = _manager
    return manager.execute(command, timeout=timeout)


def close_command_environment() -> None:
    with _manager_lock:
        manager = _manager
    manager.close()


atexit.register(close_command_environment)
