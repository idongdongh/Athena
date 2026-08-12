"""交互式 CLI 支持的斜杠命令定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandDef:
    name: str
    aliases: tuple[str, ...] = ()


COMMANDS = (
    CommandDef("new"),
    CommandDef("sessions", ("session",)),
    CommandDef("resume"),
    CommandDef("search"),
    CommandDef("archive"),
)

_COMMAND_BY_NAME = {
    alias: command
    for command in COMMANDS
    for alias in (command.name, *command.aliases)
}


def resolve_command(name: str) -> CommandDef | None:
    """将命令名或别名解析为规范命令。"""
    return _COMMAND_BY_NAME.get(name.strip().lower().lstrip("/"))


def command_names() -> tuple[str, ...]:
    return tuple(command.name for command in COMMANDS)
