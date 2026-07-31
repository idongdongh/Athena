"""read_file 工具：读文本文件，带行号与分页。核心对齐 hermes read_file_tool。

核心行为（与 hermes 一致）：
- 1-based offset 分页、limit 默认 500（上限 2000）
- 紧凑行号格式 `<n>|<content>`（无填充，省 token，对齐 hermes _add_line_numbers）
- binary 拦截（扩展名 + 内容嗅探前 8KB 含 \\x00）
- ~100K 字符上限，超限拒绝并提示用 offset/limit
- truncated 时给 hint（含下一个 offset）
- offset==1 时去 UTF-8 BOM

砍掉的 hermes 壳（留待权限层/后续）：docx/xlsx 抽取、image 重定向、敏感路径拦截、
去重/staleness 跟踪、redact。后端用纯 Python pathlib（hermes 走 shell/sed）。
"""

import json
from pathlib import Path

try:
    from tools.registry import registry
except ImportError:
    registry = None
from tools._path import resolve
from tools._binary import BINARY_EXTS

MAX_READ_CHARS = 100_000  # 读结果字符上限（对齐 hermes ~100K），超限拒绝
MAX_LIMIT = 2000
DEFAULT_LIMIT = 500


def _add_line_numbers(content: str, start_line: int = 1) -> str:
    """紧凑行号 `<n>|<content>`，对齐 hermes 格式（无填充，省 token）。"""
    out = []
    for i, line in enumerate(content.split("\n"), start=start_line):
        out.append(f"{i}|{line}")
    return "\n".join(out)


def read_file(path: str, offset: int = 1, limit: int = DEFAULT_LIMIT) -> str:
    """读文本文件，返回 JSON 字符串：{content, total_lines, file_size, truncated, hint?} 或 {error}。"""
    # 分页归一化：offset 1-based，limit 夹到 [1, 2000]
    try:
        offset = max(1, int(offset))
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        offset, limit = 1, DEFAULT_LIMIT

    fp = resolve(path)

    if not fp.exists():
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    if fp.is_dir():
        return json.dumps({"error": f"Path is a directory, not a file: {path}"}, ensure_ascii=False)

    file_size = fp.stat().st_size

    # 二进制拦截：扩展名快速路径 + 内容嗅探（前 8KB 含 \x00 即判二进制）
    if fp.suffix.lower() in BINARY_EXTS:
        return json.dumps(
            {"error": f"Cannot read binary file '{path}' ({fp.suffix})."}, ensure_ascii=False
        )
    with fp.open("rb") as bf:
        if b"\x00" in bf.read(8192):
            return json.dumps({"error": f"Cannot read binary file '{path}'."}, ensure_ascii=False)

    # 读全文；offset==1 时去 UTF-8 BOM（对齐 hermes）
    text = fp.read_text(encoding="utf-8", errors="replace")
    if offset == 1 and text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    lines = text.split("\n")
    total_lines = len(lines)
    end_line = offset + limit - 1
    page = "\n".join(lines[offset - 1:end_line])
    content = _add_line_numbers(page, offset)

    # 字符上限：超限拒绝，提示用 offset/limit 缩小范围
    if len(content) > MAX_READ_CHARS:
        return json.dumps({
            "error": (
                f"Read produced {len(content):,} characters which exceeds the safety limit "
                f"({MAX_READ_CHARS:,} chars). Use offset and limit to read a smaller range. "
                f"The file has {total_lines} lines total."
            ),
            "path": path,
            "total_lines": total_lines,
            "file_size": file_size,
        }, ensure_ascii=False)

    truncated = total_lines > end_line
    result = {
        "content": content,
        "total_lines": total_lines,
        "file_size": file_size,
        "truncated": truncated,
    }
    if truncated:
        result["hint"] = (
            f"Use offset={end_line + 1} to continue reading "
            f"(showing {offset}-{min(end_line, total_lines)} of {total_lines} lines)"
        )
    return json.dumps(result, ensure_ascii=False)


READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "读取文本文件，带行号与分页。用这个代替 cat/head/tail。"
        "输出格式：'行号|内容'。大文件用 offset 和 limit 分页读取。"
        "读取结果超过约 100K 字符会被拒绝。不能读取二进制文件。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要读取的文件路径（绝对、相对或 ~/path）"},
            "offset": {"type": "integer", "description": "起始行号（1-based，默认 1）", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "最多读取的行数（默认 500，上限 2000）", "default": 500, "maximum": 2000},
        },
        "required": ["path"],
    },
}

if registry is not None:
    registry.register(name="read_file", schema=READ_FILE_TOOL, handler=read_file)
