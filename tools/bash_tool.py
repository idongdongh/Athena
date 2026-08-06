"""bash 工具：执行 shell 命令，返回 stdout/stderr/exit_code。

一次性无状态：每次调用新开 shell，跑完即死——cd 不跨调用保留。
需要持久会话/交互式/后台进程时再升级到 terminal（见 notebook/tool_inventory.md）。
核心能力 = coding agent 地基（跑命令拿输出），砍掉 hermes terminal 的 PTY/进程注册/多环境厚壳。
"""

import json
import os
import signal
import subprocess
import tempfile
import time

from agent.interrupt_controller import ToolExecutionCancelled, interrupt_controller

try:
    from tools.registry import registry
except ImportError:
    registry = None

DEFAULT_TIMEOUT = 120
TERMINATE_GRACE_SECONDS = 1.0
MAX_OUTPUT_CHARS = 30_000  # 单流输出字符上限，超限截断并标记 truncated（防撑爆 token）
MAX_CAPTURE_BYTES = MAX_OUTPUT_CHARS * 4 + 1024


def _stop_process(process: subprocess.Popen) -> None:
    """终止 bash 创建的整个进程组并等待操作系统回收。"""
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _read_capped_output(stream) -> tuple[str, bool]:
    """从临时文件读取固定上限内容，返回文本和是否被截断。"""
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


def bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """执行 shell 命令，返回 JSON：{exit_code, stdout, stderr, truncated} 或 {error}。

    在进程 cwd（os.getcwd()）下运行；相对路径与 read_file/write_file 一致地按 cwd 解析。
    stdout/stderr 分开返回，便于模型诊断；各自超 30K 截断。
    """
    process = None
    try:
        # 子进程直接写临时文件，避免 communicate() 先把无限 stdout/stderr 全部
        # 缓冲进 Python 内存。完成后只读取固定上限；代价是极端输出会占用临时磁盘。
        with tempfile.TemporaryFile() as stdout_sink, tempfile.TemporaryFile() as stderr_sink:
            process = subprocess.Popen(
                command,   # 要执行的命令
                shell=True,  # 通过系统 shell 执行命令
                stdout=stdout_sink,
                stderr=stderr_sink,
                # 创建独立进程组，中断时才能连同 shell 启动的子进程一起终止。
                start_new_session=(os.name == "posix"),
            )
            started_at = time.monotonic()
            while True:
                if interrupt_controller.is_requested():
                    _stop_process(process)
                    raise ToolExecutionCancelled("bash interrupted by user")

                remaining = timeout - (time.monotonic() - started_at)
                if remaining <= 0:
                    _stop_process(process)
                    return json.dumps({"error": f"Timeout ({timeout}s)"}, ensure_ascii=False)

                try:
                    process.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue

            stdout, stdout_truncated = _read_capped_output(stdout_sink)
            stderr, stderr_truncated = _read_capped_output(stderr_sink)
    except KeyboardInterrupt:
        # 第二次 Ctrl+C 会沿调用栈退出 REPL；先杀掉子进程组，避免遗留孤儿进程。
        if process is not None:
            _stop_process(process)
        raise
    except ToolExecutionCancelled:
        raise
    except Exception as e:
        if process is not None:
            _stop_process(process)
        return json.dumps({"error": f"Failed to run command: {e}"}, ensure_ascii=False)

    truncated = stdout_truncated or stderr_truncated

    # returncode：子进程的退出状态码，0 表示成功，非 0 表示失败。
    return json.dumps({
        "exit_code": process.returncode,
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
