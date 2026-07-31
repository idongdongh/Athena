"""bash 工具：执行 shell 命令，返回 stdout/stderr/exit_code。

一次性无状态：每次调用新开 shell，跑完即死——cd 不跨调用保留。
需要持久会话/交互式/后台进程时再升级到 terminal（见 notebook/tool_inventory.md）。
核心能力 = coding agent 地基（跑命令拿输出），砍掉 hermes terminal 的 PTY/进程注册/多环境厚壳。
"""

import json
import subprocess

try:
    from tools.registry import registry
except ImportError:
    registry = None

DEFAULT_TIMEOUT = 120
MAX_OUTPUT_CHARS = 30_000  # 单流输出字符上限，超限截断并标记 truncated（防撑爆 token）


def bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """执行 shell 命令，返回 JSON：{exit_code, stdout, stderr, truncated} 或 {error}。

    在进程 cwd（os.getcwd()）下运行；相对路径与 read_file/write_file 一致地按 cwd 解析。
    stdout/stderr 分开返回，便于模型诊断；各自超 30K 截断。
    """
    try:
        r = subprocess.run(
            command,   # 要执行的命令
            shell=True,  # 通过系统 shell 执行命令
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 以文本（字符串）形式返回输出
            encoding="utf-8", 
            errors="replace",   # 编码错误时的处理方式
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Timeout ({timeout}s)"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to run command: {e}"}, ensure_ascii=False)

    stdout = r.stdout or ""
    stderr = r.stderr or ""
    truncated = False
    if len(stdout) > MAX_OUTPUT_CHARS:
        # :,表示以千分位的形式显示
        stdout = stdout[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(r.stdout):,} chars total]"
        truncated = True
    if len(stderr) > MAX_OUTPUT_CHARS:
        stderr = stderr[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(r.stderr):,} chars total]"
        truncated = True

    # r.returncode：子进程的退出状态码：0 表示成功，非 0 表示失败
    return json.dumps({
        "exit_code": r.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
    }, ensure_ascii=False)


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
