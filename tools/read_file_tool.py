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

from agent.interrupt_controller import ToolExecutionCancelled, interrupt_controller

try:
    from tools.registry import registry
except ImportError:
    registry = None
from tools._path import resolve
from tools._binary import BINARY_EXTS

MAX_READ_CHARS = 100_000  # 读结果字符上限（对齐 hermes ~100K），超限拒绝
MAX_LIMIT = 2000
DEFAULT_LIMIT = 500
READ_CHUNK_BYTES = 64 * 1024
# UTF-8 单字符最多 4 字节；只保留足以判断 100K 字符上限的数据。
MAX_CAPTURE_BYTES = MAX_READ_CHARS * 4 + 1024


def _add_line_numbers(content: str, start_line: int = 1) -> str:
    """紧凑行号 `<n>|<content>`，对齐 hermes 格式（无填充，省 token）。"""
    out = []
    for i, line in enumerate(content.split("\n"), start=start_line):
        out.append(f"{i}|{line}")
    return "\n".join(out)


def _iter_bounded_binary_lines(stream):
    """逐行扫描二进制流，单行内存占用不超过 ``MAX_CAPTURE_BYTES``。"""
    while True:
        captured = bytearray()
        line_too_large = False
        saw_data = False

        while True:
            if interrupt_controller.is_requested():
                raise ToolExecutionCancelled("read_file interrupted by user")
            chunk = stream.readline(READ_CHUNK_BYTES)
            if not chunk:
                if saw_data:
                    yield bytes(captured), line_too_large
                return

            saw_data = True
            remaining = MAX_CAPTURE_BYTES - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if len(chunk) > remaining:
                line_too_large = True

            if chunk.endswith(b"\n"):
                yield bytes(captured), line_too_large
                break


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

    end_line = offset + limit - 1
    total_lines = 0
    page_lines: list[str] = []
    captured_bytes = 0
    output_too_large = False

    # 扫描全文以计算 total_lines，但只保存请求页且设置固定捕获上限；因此内存
    # 不再随文件大小增长。超长单行也会分块排空，不会一次读入内存。
    with fp.open("rb") as bf:
        for raw_line, line_too_large in _iter_bounded_binary_lines(bf):
            total_lines += 1
            if total_lines < offset or total_lines > end_line:
                continue
            if line_too_large:
                output_too_large = True
                continue

            raw_line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
            captured_bytes += len(raw_line)
            if captured_bytes > MAX_CAPTURE_BYTES:
                output_too_large = True
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if total_lines == 1 and offset == 1:
                line = line.lstrip("\ufeff")
            page_lines.append(line)

    page = "\n".join(page_lines)
    content = _add_line_numbers(page, offset) if page_lines else ""

    # 字符上限：超限拒绝，提示用 offset/limit 缩小范围
    if output_too_large or len(content) > MAX_READ_CHARS:
        return json.dumps({
            "error": (
                f"Read exceeds the safety limit "
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
