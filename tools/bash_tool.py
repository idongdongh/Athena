"""bash 工具：执行 shell 命令，返回 stdout/stderr/exit_code。

一次性无状态：每次调用新开 shell，跑完即死——cd 不跨调用保留。
需要持久会话/交互式/后台进程时再升级到 terminal（见 notebook/tool_inventory.md）。
核心能力 = coding agent 地基（跑命令拿输出），砍掉 hermes terminal 的 PTY/进程注册/多环境厚壳。
"""

import json
from tools.command_environment import execute_command
from tools.environments.base import DEFAULT_TIMEOUT, MAX_CAPTURE_BYTES, MAX_OUTPUT_CHARS

try:
    from tools.registry import registry
except ImportError:
    registry = None

def bash(command: str, timeout: int | None = None) -> str:
    """执行 shell 命令，返回 JSON：{exit_code, stdout, stderr, truncated} 或 {error}。

    在配置的工作区运行；相对路径与 read_file/write_file 一致地解析。
    stdout/stderr 分开返回，便于模型诊断；各自超 30K 截断。
    """
    try:
        result = execute_command(command, timeout=timeout)
    except TimeoutError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception as e:
        from agent.interrupt_controller import ToolExecutionCancelled

        if isinstance(e, ToolExecutionCancelled):
            raise
        return json.dumps({"error": f"Failed to run command: {e}"}, ensure_ascii=False)
    return json.dumps(result.as_dict(), ensure_ascii=False)


BASH_TOOL = {
    "name": "bash",
    "description": (
        "执行 bash 命令，返回 stdout/stderr/exit_code。每次调用在新 shell 中运行（无状态："
        "`cd` 不会跨调用保留——需要时用 `cd x && cmd`）。用于 ls、grep、git、pytest、build 等。"
        "单流输出超过约 30K 字符会截断。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 bash 命令"},
            "timeout": {"type": "integer", "description": "超时时间（秒，默认 120）", "default": 120},
        },
        "required": ["command"],
    },
}

if registry is not None:
    registry.register(name="bash", schema=BASH_TOOL, handler=bash)
