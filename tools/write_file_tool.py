"""write_file 工具：写文件，自动建父目录。核心对齐 hermes write_file_tool。

核心行为（与 hermes 一致）：
- 整体覆盖写（不是追加）
- 自动创建父目录
- 返回 bytes_written / dirs_created

砍掉的 hermes 壳（留待权限层/后续）：lint/LSP 诊断、CRLF/BOM 保留、原子写（temp+rename）、
敏感路径/cross_profile 拦截。后端用纯 Python pathlib（hermes 走 shell）。
"""

import json

try:
    from tools.registry import registry
except ImportError:
    registry = None
from tools._path import resolve


def write_file(path: str, content: str) -> str:
    """写文件（整体覆盖），返回 JSON 字符串：{bytes_written, dirs_created} 或 {error}。"""
    fp = resolve(path)
    try:
        parent = fp.parent
        dirs_created = False
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            dirs_created = True
        fp.write_text(content, encoding="utf-8")
        bytes_written = len(content.encode("utf-8"))
        return json.dumps(
            {"bytes_written": bytes_written, "dirs_created": dirs_created},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": f"Failed to write file: {e}"}, ensure_ascii=False)


WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": (
        "写入文件，完全覆盖已有内容。用这个代替 echo/cat heredoc。"
        "自动创建父目录。会整体覆盖整个文件——做局部修改请用 patch。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要写入的文件路径（不存在则创建，存在则覆盖）"},
            "content": {"type": "string", "description": "要写入文件的完整内容"},
        },
        "required": ["path", "content"],
    },
}

if registry is not None:
    registry.register(name="write_file", schema=WRITE_FILE_TOOL, handler=write_file)
