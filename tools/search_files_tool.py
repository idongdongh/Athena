"""search_files 工具：搜文件内容(regex)或按文件名(glob)。核心对齐 hermes search_files。

核心行为（与 hermes 一致）：
- target='content'：正则搜文件内容，返回 {path, line, match, context?}（= grep/rg）
- target='files'：按 glob 找文件名，按 mtime 排序（= find/ls）
- output_mode（content 模式）：content（匹配行+行号）/ files_only（只列文件名）/ count（每文件匹配数）
- 可选 file_glob 过滤文件、limit 限结果数、context 带上下文行

砍掉的 hermes 壳（按「别复杂」）：ripgrep 后端（用纯 Python re + pathlib）、newline-regex 警告、offset 分页。
跳过 .git/.venv/__pycache__/node_modules 等垃圾目录，跳过二进制扩展名。
"""

import json
import re
from pathlib import Path

from agent.interrupt_controller import ToolExecutionCancelled, interrupt_controller

try:
    from tools.registry import registry
except ImportError:
    registry = None
from tools._path import resolve
from tools._binary import BINARY_EXTS

_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".idea", ".vscode", "dist", "build", ".hg", ".svn"}


def _iter_files(root: Path, file_glob: str = None):
    """遍历 root 下文本文件，跳过垃圾目录与二进制扩展名，可选 file_glob 过滤。"""
    for p in root.rglob("*"):
        interrupt_controller.raise_if_requested("search_files interrupted by user")
        if p.is_dir():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in BINARY_EXTS:
            continue
        if file_glob and not p.match(file_glob):
            continue
        yield p


def search_files(pattern: str, target: str = "content", path: str = ".",
                 file_glob: str = None, limit: int = 50,
                 output_mode: str = "content", context: int = 0) -> str:
    """搜文件内容或文件名，返回 JSON。"""
    root = resolve(path)
    if not root.exists():
        return json.dumps({"error": f"Path not found: {path}"}, ensure_ascii=False)

    # ── target='files'：按文件名 glob 找，按 mtime 排序 ──
    if target == "files":
        files = [p for p in _iter_files(root) if p.match(pattern)]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        files = files[:limit]
        return json.dumps(
            {"files": [str(f) for f in files], "count": len(files)},
            ensure_ascii=False,
        )

    # ── target='content'：正则搜内容 ──
    if output_mode not in ("content", "files_only", "count"):
        output_mode = "content"
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex: {e}"}, ensure_ascii=False)

    files = [root] if root.is_file() else list(_iter_files(root, file_glob))
    results = []
    # 两种 output_mode 下 limit 含义不同：content 模式按"匹配行数"截断，
    # files_only/count 模式按"文件数"截断。两个变量分开命名避免语义混用。
    match_count = 0
    file_count = 0
    truncated = False

    for f in files:
        interrupt_controller.raise_if_requested("search_files interrupted by user")
        if output_mode == "content" and match_count >= limit:
            truncated = True
            break
        if output_mode != "content" and file_count >= limit:
            truncated = True
            break
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except ToolExecutionCancelled:
            raise
        except Exception:
            continue
        lines = text.splitlines()
        file_matches = []
        for i, line in enumerate(lines, 1):
            if i % 256 == 0:
                interrupt_controller.raise_if_requested(
                    "search_files interrupted by user"
                )
            if regex.search(line):
                file_matches.append((i, line))
        if not file_matches:
            continue

        if output_mode == "files_only":
            results.append(str(f))
            file_count += 1
        elif output_mode == "count":
            results.append({"path": str(f), "count": len(file_matches)})
            file_count += 1
        else:  # content
            for i, line in file_matches:
                m = {"path": str(f), "line": i, "match": line.strip()[:500]}
                if context > 0:
                    start = max(0, i - 1 - context)
                    end = min(len(lines), i + context)
                    m["context"] = lines[start:end]
                results.append(m)
                match_count += 1
                if match_count >= limit:
                    truncated = True
                    break

    return json.dumps(
        {"results": results, "count": len(results), "truncated": truncated},
        ensure_ascii=False,
    )


SEARCH_FILES_TOOL = {
    "name": "search_files",
    "description": (
        "搜索文件内容或按文件名查找。用这个代替 grep/rg/find/ls。\n\n"
        "内容搜索（target='content'）：在文件内容中做正则搜索。输出模式："
        "content（匹配行+行号）、files_only（仅文件路径）、count（每文件匹配数）。\n\n"
        "文件搜索（target='files'）：按 glob 模式找文件（如 '*.py'、'*config*'），按修改时间排序。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "内容搜索时为正则，文件搜索时为 glob 模式（如 '*.py'）"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' 搜文件内容，'files' 按文件名查找", "default": "content"},
            "path": {"type": "string", "description": "要搜索的目录或文件（默认当前工作目录）", "default": "."},
            "file_glob": {"type": "string", "description": "内容模式下按模式过滤文件（如 '*.py' 只搜 Python 文件）"},
            "limit": {"type": "integer", "description": "最多返回的结果数（默认 50）", "default": 50},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "内容模式的输出格式：'content' 显示匹配，'files_only' 列路径，'count' 显示每文件计数", "default": "content"},
            "context": {"type": "integer", "description": "每个匹配前后的上下文行数（仅内容模式）", "default": 0},
        },
        "required": ["pattern"],
    },
}

if registry is not None:
    registry.register(name="search_files", schema=SEARCH_FILES_TOOL, handler=search_files)
