"""项目 YAML 配置加载。"""

from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass

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
