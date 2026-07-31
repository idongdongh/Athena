"""patch 工具：定点编辑文件（找替换），返回 unified diff。核心对齐 hermes patch(replace 模式)。

核心行为（与 hermes 一致）：
- 精确 find-and-replace：在文件里找 old_string，换成 new_string
- 默认要求 old_string 唯一（否则报错提示带上下文或用 replace_all）；replace_all=true 替换全部
- 返回 unified diff，让模型看到改了啥

砍掉的 hermes 壳（按「别复杂」）：9 策略模糊匹配（精确匹配失败给明确错误，让模型重试带上下文）、
V4A 多文件 patch 模式（高级批量，留待需要）、CRLF/BOM 保留、lint/LSP 诊断。
"""

import difflib
import json
import os
from pathlib import Path

try:
    from tools.registry import registry
except ImportError:
    registry = None
from tools._path import resolve


def patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """定点替换文件中的 old_string 为 new_string，返回 JSON：{success, diff, replaced_count} 或 {error}。"""
    fp = resolve(path)
    if not fp.exists():
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)
    if fp.is_dir():
        return json.dumps({"error": f"Path is a directory, not a file: {path}"}, ensure_ascii=False)
    if not old_string:
        return json.dumps({"error": "old_string must not be empty"}, ensure_ascii=False)

    try:
        original = fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"error": f"Failed to read file: {e}"}, ensure_ascii=False)

    if old_string not in original:
        return json.dumps({"error": f"old_string not found in {path}"}, ensure_ascii=False)

    if not replace_all:
        count = original.count(old_string)
        if count > 1:
            return json.dumps({
                "error": (
                    f"old_string is not unique ({count} occurrences in {path}). "
                    "Include more surrounding context to make it unique, or set replace_all=true."
                ),
            }, ensure_ascii=False)
        new_text = original.replace(old_string, new_string, 1)
        replaced = 1
    else:
        replaced = original.count(old_string)
        new_text = original.replace(old_string, new_string)

    # unified diff（git 风格 a/ 旧、b/ 新；绝对路径转相对，避免 a//abs 双斜杠）
    label = path if not os.path.isabs(path) else os.path.relpath(path)
    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
    ))

    try:
        fp.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": f"Failed to write file: {e}"}, ensure_ascii=False)

    return json.dumps(
        {"success": True, "diff": diff, "replaced_count": replaced},
        ensure_ascii=False,
    )


PATCH_TOOL = {
    "name": "patch",
    "description": (
        "在文件中定点查找替换。用这个代替用 write_file 重写整个文件。"
        "找到 old_string 替换为 new_string。除非 replace_all=true，否则 old_string 必须在文件中唯一。"
        "返回 unified diff 展示改动。在 old_string 中包含上下文行以确保唯一性。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要编辑的文件路径"},
            "old_string": {"type": "string", "description": "要查找的精确文本。除非 replace_all=true 否则必须唯一；为唯一性可包含上下文行"},
            "new_string": {"type": "string", "description": "替换后的文本。传空串 '' 可删除匹配文本"},
            "replace_all": {"type": "boolean", "description": "替换全部出现，而非要求唯一匹配（默认 false）", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    },
}

if registry is not None:
    registry.register(name="patch", schema=PATCH_TOOL, handler=patch)
