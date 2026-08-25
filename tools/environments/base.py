"""命令执行环境的最小公共契约。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import BinaryIO, Protocol


DEFAULT_TIMEOUT = 120
TERMINATE_GRACE_SECONDS = 1.0
MAX_OUTPUT_CHARS = 30_000
MAX_CAPTURE_BYTES = MAX_OUTPUT_CHARS * 4 + 1024


@dataclass(frozen=True)
class CommandResult:
    """一次命令执行的稳定返回结构。"""

    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
        }


class CommandEnvironment(Protocol):
    """当前 bash 工具需要的窄执行接口。"""

    def execute(self, command: str, *, timeout: int) -> CommandResult: ...

    def close(self) -> None: ...


def read_capped_output(stream: BinaryIO) -> tuple[str, bool]:
    """从 seekable 二进制流中有界读取输出。"""

    stream.flush()
    stream.seek(0, os.SEEK_END)
    total_bytes = stream.tell()
    stream.seek(0)
    raw = stream.read(MAX_CAPTURE_BYTES)
    text = raw.decode("utf-8", errors="replace")
    truncated = total_bytes > len(raw) or len(text) > MAX_OUTPUT_CHARS
    if not truncated:
        return text, False

    marker = f"\n... [truncated, {total_bytes:,} bytes total]"
    if len(marker) >= MAX_OUTPUT_CHARS:
        return marker[:MAX_OUTPUT_CHARS], True
    return text[:MAX_OUTPUT_CHARS - len(marker)] + marker, True
