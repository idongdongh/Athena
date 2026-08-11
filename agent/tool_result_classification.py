"""统一的工具执行结果分类。

工具 handler 只返回原始内容；本模块在主线程把内容归类为成功、执行失败、
策略阻断等状态。guardrail 和最终核验都消费同一个 ``ToolOutcome``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# 工具已执行并成功完成。
SUCCESS = "succeeded"
# 工具已执行，但返回了失败结果。
FAILED = "failed"
# 工具尚未执行，被权限策略或安全护栏拦截。
BLOCKED = "blocked"
# 工具没找到，未执行。
UNKNOWN = "unknown"
# 调度器或工具处理代码本身发生了内部异常。
INTERNAL_ERROR = "internal_error"
# 工具执行尚未完成，就被用户或系统取消。
CANCELLED = "cancelled"

FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


@dataclass(frozen=True)
class ToolOutcome:
    """一次工具调用的结果。"""

    # 实际调用的工具名称，例如 bash、read_file。
    tool_name: str
    # 工具返回给模型的原始内容。
    content: str
    # 工具调用的状态，例如 succeeded、failed、blocked。
    status: str
    # 机器可读的错误码；没有错误时为 None。
    error_code: str | None = None
    # 给模型或用户阅读的错误说明；没有错误时为 None。
    error_message: str | None = None
    # 命令行工具的退出码（0 成功，非 0 失败）；非命令行工具通常为 None
    exit_code: int | None = None

    @property
    def executed(self) -> bool:
        """工具执行了：status in {SUCCESS, FAILED, INTERNAL_ERROR}"""
        return self.status in {SUCCESS, FAILED, INTERNAL_ERROR}

    @property
    def counts_as_failure(self) -> bool:
        """重复调用失败：status in {FAILED, UNKNOWN, INTERNAL_ERROR}"""
        return self.status in {FAILED, UNKNOWN, INTERNAL_ERROR}


def file_mutation_result_landed(tool_name: str, content: str) -> bool:
    """根据工具执行返回结果判断写操作是否成功。"""
    if tool_name not in FILE_MUTATING_TOOL_NAMES:
        return False
    try:
        data = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    return data.get("success") is True


def classify_tool_result(
    tool_name: str,
    content: Any,
    *,
    status: str | None = None,
) -> ToolOutcome:
    """将原始 handler 输出归类为 ToolOutcome 类对象。

    Args:
        tool_name (str): 工具名称
        content (Any): 工具执行结果
        status (str | None): 工具调用状态，默认为 None

    Returns:
        ToolOutcome 类对象
    """
    text = "" if content is None else str(content)
    if status is not None:
        code = {
            BLOCKED: "blocked",
            UNKNOWN: "unknown_tool",
            INTERNAL_ERROR: "handler_exception",
            CANCELLED: "cancelled",
        }.get(status)
        return ToolOutcome(tool_name, text, status, code, text or None)

    # Hermes 会把成功但没有正文的工具结果规范化为明确占位文本。
    # 空结果不等于失败：部分命令成功执行后本来就不会产生输出。
    if not text.strip():
        text = "(no output)"
        if tool_name in FILE_MUTATING_TOOL_NAMES:
            return ToolOutcome(
                tool_name,
                text,
                FAILED,
                "empty_mutation_result",
                "写工具没有返回可验证的落盘结果",
            )

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None

    # 与 Hermes terminal 语义一致：非零退出码是 bash 的权威失败信号。
    if tool_name == "bash" and isinstance(data, dict):
        exit_code = data.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return ToolOutcome(
                tool_name, text, FAILED, "nonzero_exit",
                str(data.get("stderr") or data.get("error") or f"exit {exit_code}"),
                exit_code,
            )
        if exit_code == 0:
            return ToolOutcome(tool_name, text, SUCCESS, exit_code=0)

    if file_mutation_result_landed(tool_name, text):
        return ToolOutcome(tool_name, text, SUCCESS)

    if isinstance(data, dict):
        # 结构化字段是权威信号。不能继续扫描整段 JSON 文本，否则
        # success=true + error=null，或结果列表中的局部 error，都会被误判为失败。
        error = data.get("error")
        if data.get("success") is False:
            message = error or data.get("message") or "success=false"
            return ToolOutcome(
                tool_name,
                text,
                FAILED,
                "unsuccessful_result",
                str(message),
            )
        if error:
            return ToolOutcome(tool_name, text, FAILED, "structured_error", str(error))
        if data.get("failed") is True:
            message = data.get("message") or "failed=true"
            return ToolOutcome(tool_name, text, FAILED, "structured_error", str(message))
        return ToolOutcome(tool_name, text, SUCCESS)

    # 其他合法 JSON（例如数组或标量）同样属于结构化结果。只有无法解析的普通
    # 文本才使用错误关键词兜底，避免嵌套数据中的 error 字段污染调用级状态。
    if data is not None:
        return ToolOutcome(tool_name, text, SUCCESS)

    lower = text[:500].lower()
    if text.startswith("Error") or '"error"' in lower or '"failed"' in lower:
        return ToolOutcome(tool_name, text, FAILED, "error_text", text[:500])
    return ToolOutcome(tool_name, text, SUCCESS)
