"""项目 YAML 配置加载。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """读取 YAML 配置；文件不存在时返回空配置，格式错误时阻止启动。"""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取配置文件 {config_path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise RuntimeError(f"配置文件顶层必须是 mapping: {config_path}")
    return dict(data)
