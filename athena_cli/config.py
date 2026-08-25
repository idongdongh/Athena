"""项目 YAML 配置加载。"""

from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass, field

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """读取 YAML 配置；支持读取指定路径的 yaml 文件；不存在时返回空配置，顶层必须是类字典格式否则报错，阻止启动。

    Args:
        path (Path | str | None, optional): 文件路径

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        dict: _description_
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    try:
        # safe_load：将 yaml 文件内容转化为 python 对象
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取配置文件 {config_path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise RuntimeError(f"配置文件顶层必须是 mapping: {config_path}")
    return dict(data)


@dataclass(frozen=True)
class SessionSettings:
    enabled: bool = True
    database: str = ".athena/state.db"

    @classmethod
    def from_mapping(cls, config: Mapping | None) -> "SessionSettings":
        defaults = cls()
        section = config.get("session") if isinstance(config, Mapping) else None
        if not isinstance(section, Mapping):
            return defaults
        enabled = section.get("enabled")
        database = section.get("database")
        return cls(
            enabled=enabled if isinstance(enabled, bool) else defaults.enabled,
            database=(
                database.strip()
                if isinstance(database, str) and database.strip()
                else defaults.database
            ),
        )

    def resolve_database_path(self, project_root: Path) -> Path:
        path = Path(self.database).expanduser()
        resolved = path if path.is_absolute() else project_root / path
        if path == Path(".athena/state.db") and not resolved.exists():
            legacy = project_root / ".hello-agent" / "state.db"
            if legacy.exists():
                return legacy
        return resolved


@dataclass(frozen=True)
class MemorySettings:
    """内置文件记忆配置。"""

    memory_enabled: bool = False
    user_profile_enabled: bool = False
    memory_char_limit: int = 2200
    user_char_limit: int = 1375
    nudge_interval: int = 10
    directory: str = "memories"

    @classmethod
    def from_mapping(cls, config: Mapping | None) -> "MemorySettings":
        defaults = cls()
        section = config.get("memory") if isinstance(config, Mapping) else None
        if not isinstance(section, Mapping):
            return defaults

        def positive_int(name: str, default: int) -> int:
            value = section.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default

        nudge_interval = section.get("nudge_interval")

        directory = section.get("directory")
        return cls(
            memory_enabled=(
                section["memory_enabled"]
                if isinstance(section.get("memory_enabled"), bool)
                else defaults.memory_enabled
            ),
            user_profile_enabled=(
                section["user_profile_enabled"]
                if isinstance(section.get("user_profile_enabled"), bool)
                else defaults.user_profile_enabled
            ),
            memory_char_limit=positive_int("memory_char_limit", defaults.memory_char_limit),
            user_char_limit=positive_int("user_char_limit", defaults.user_char_limit),
            nudge_interval=(
                nudge_interval
                if isinstance(nudge_interval, int)
                and not isinstance(nudge_interval, bool)
                and nudge_interval >= 0
                else defaults.nudge_interval
            ),
            directory=(
                directory.strip()
                if isinstance(directory, str) and directory.strip()
                else defaults.directory
            ),
        )

    def resolve_directory(self, project_root: Path) -> Path:
        path = Path(self.directory).expanduser()
        return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class DockerSettings:
    """本地 Docker/Podman 沙箱配置。"""

    runtime: str = "auto"
    image: str = "python:3.11-slim"
    network: bool = False
    cpu: float = 1.0
    memory_mb: int = 1024
    pids_limit: int = 128
    workspace_read_only: bool = False
    startup_timeout: int = 120

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> "DockerSettings":
        defaults = cls()
        if value is None:
            return defaults
        if not isinstance(value, Mapping):
            raise RuntimeError("terminal.docker 必须是 mapping")

        runtime = value.get("runtime", defaults.runtime)
        if not isinstance(runtime, str) or runtime.strip().lower() not in {"auto", "docker", "podman"}:
            raise RuntimeError("terminal.docker.runtime 必须是 auto、docker 或 podman")

        image = value.get("image", defaults.image)
        if not isinstance(image, str) or not image.strip():
            raise RuntimeError("terminal.docker.image 必须是非空字符串")

        network = value.get("network", defaults.network)
        read_only = value.get("workspace_read_only", defaults.workspace_read_only)
        for name, item in (("network", network), ("workspace_read_only", read_only)):
            if not isinstance(item, bool):
                raise RuntimeError(f"terminal.docker.{name} 必须是布尔值")

        cpu = value.get("cpu", defaults.cpu)
        if isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or cpu <= 0:
            raise RuntimeError("terminal.docker.cpu 必须是正数")

        def positive_int(name: str, default: int) -> int:
            item = value.get(name, default)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise RuntimeError(f"terminal.docker.{name} 必须是正整数")
            return item

        return cls(
            runtime=runtime.strip().lower(),
            image=image.strip(),
            network=network,
            cpu=float(cpu),
            memory_mb=positive_int("memory_mb", defaults.memory_mb),
            pids_limit=positive_int("pids_limit", defaults.pids_limit),
            workspace_read_only=read_only,
            startup_timeout=positive_int("startup_timeout", defaults.startup_timeout),
        )


@dataclass(frozen=True)
class TerminalSettings:
    """命令执行后端配置。"""

    backend: str = "local"
    timeout: int = 120
    docker: DockerSettings = field(default_factory=DockerSettings)

    @classmethod
    def from_mapping(cls, config: Mapping | None) -> "TerminalSettings":
        defaults = cls()
        section = config.get("terminal") if isinstance(config, Mapping) else None
        if section is None:
            return defaults
        if not isinstance(section, Mapping):
            raise RuntimeError("terminal 必须是 mapping")

        backend = section.get("backend", defaults.backend)
        if not isinstance(backend, str) or backend.strip().lower() not in {"local", "docker"}:
            raise RuntimeError("terminal.backend 必须是 local 或 docker")

        timeout = section.get("timeout", defaults.timeout)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise RuntimeError("terminal.timeout 必须是正整数")

        return cls(
            backend=backend.strip().lower(),
            timeout=timeout,
            docker=DockerSettings.from_mapping(section.get("docker")),
        )
