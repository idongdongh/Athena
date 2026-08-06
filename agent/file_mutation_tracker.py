"""记录一条用户请求中未恢复的文件修改失败。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agent.tool_result_classification import ToolOutcome, file_mutation_result_landed


@dataclass(frozen=True)
class FailedMutation:
    """一条文件修改失败记录。"""
    tool_name: str  # 哪个工具失败了
    path: str  # 哪个文件失败了
    error_message: str  # 失败原因


class FileMutationTracker:
    def __init__(self) -> None:
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        """重置文件修改失败记录字典（文件路径：文件修改失败记录）"""
        self._failures: dict[str, FailedMutation] = {}

    def record(self, outcome: ToolOutcome, args: dict) -> None:
        """根据写工具返回结果判断是否写成功（成功则，删除写失败记录）；根据 outcome 中的 status 字段判断是否写失败（失败则添加写失败记录）

        Args:
            outcome (ToolOutcome): 写工具调用结果
            args (dict): 参数
        """
        if outcome.tool_name not in {"write_file", "patch"}:
            return
        # 检查模型给的 path 是否合法
        raw_path = args.get("path") if isinstance(args, dict) else None
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        # 处理家目录、相对路径、软链接
        path = os.path.realpath(os.path.abspath(os.path.expanduser(raw_path)))
        # 写成功
        if file_mutation_result_landed(outcome.tool_name, outcome.content):
            self._failures.pop(path, None)
        # 写失败
        elif outcome.executed and outcome.counts_as_failure:
            self._failures.setdefault(
                path,
                FailedMutation(
                    tool_name=outcome.tool_name,
                    path=path,
                    error_message=outcome.error_message or outcome.content[:240] or "unknown error",
                ),
            )

    def unresolved_failures(self) -> list[FailedMutation]:
        """
        Returns:
            list[FailedMutation]：未处理的文件修改失败记录
        """
        return list(self._failures.values())

    def format_notice(self) -> str:
        """格式化未处理的文件修改失败记录
        Returns:
            str: 没有未处理的文件修改失败时返回空字符串；
                否则返回包含文件路径和失败原因的提示文本。
        """
        failures = self.unresolved_failures()
        if not failures:
            return ""
        lines = ["注意：以下文件修改没有成功完成："]
        lines.extend(f"- {item.path}: {item.error_message}" for item in failures)
        return "\n".join(lines)
