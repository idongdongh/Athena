import os
import signal
import sys

from anthropic import Anthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory

# 必须在导入 tool_executor 前加载：其模块级 guardrail 会读取环境开关。
load_dotenv(override=True)

from tools.registry import registry, discover
from agent.tool_executor import execute_tool_calls, file_mutation_tracker, tool_guardrails
from agent.tracer import reset_tracer, get_tracer

# 启动期校验：api_key/model 缺失时抛友好 RuntimeError
# 兼容两套命名（API_KEY / ANTHROPIC_API_KEY），与 web_extract_tool.py 保持一致
_api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
if not _api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY（或 API_KEY）未设置。请在项目根 .env 配置后重试。"
    )
_model_id = os.getenv("MODEL_ID")
if not _model_id:
    raise RuntimeError(
        "MODEL_ID 未设置。请在项目根 .env 配置后重试。"
    )

client = Anthropic(base_url=os.getenv("BASE_URL"), api_key=_api_key)
MODEL_ID = _model_id


def _build_system() -> str:
    """组装 system prompt。

    会话级只建一次（模块 import 时求值，整个会话复用）。当前精简版
    只注入工作目录和基本工具策略，不动态拼接 Hermes 的 skills / memory 片段。
    """
    workdir = os.getcwd()
    return (
        f"You are a coding agent at {workdir}. Use tools to solve tasks. Act, don't explain."
        "如果执行工具的过程用户拒绝了你的工具请求，你不应该绕过命令，而是向用户说明原因"
    )


SYSTEM = _build_system()
# 最大迭代步数：达到上限后去掉 tools 做一次总结调用优雅收尾（对齐 hermes max_iterations 默认 90）
MAX_ITERATIONS = 90
# API 调用超时（秒）：防止网络层面无限期阻塞
API_TIMEOUT = 300.0
# 工具并发执行：仅工具名互不重复的只读调用会并行；bash、写操作和重复调用顺序执行。
CONCURRENT_TOOLS = True
# 并发线程池上限
TOOL_MAX_WORKERS = 4
# 单轮 API 输出 token 上限：tool_use 块本就小，1024 够；summary 轮要产长文本，
# 允许 .env 覆盖（MAX_TOKENS_SUMMARY=8192）。
MAX_TOKENS_TOOL = 1024
MAX_TOKENS_SUMMARY = int(os.getenv("MAX_TOKENS_SUMMARY", "4096"))

# ---- 用户中断机制（对齐 hermes _interrupt_requested） ----
_interrupt_requested = False


# signum：信号编号，不同编号对应不同的中断机制
# frame：栈帧，当前程序执行的快照
def _on_interrupt(signum, frame):
    """第一次 Ctrl+C 设置中断标志；第二次 Ctrl+C 恢复默认行为，直接终止进程。"""
    # 只要函数体内对全局变量执行了赋值操作（=），就必须在函数内声明 global
    global _interrupt_requested
    _interrupt_requested = True
    print(
        "\n\033[33m⚠️  Interrupt requested — finishing after current API call..."
        " (Ctrl+C again to force quit)\033[0m"
    )
    signal.signal(signal.SIGINT, signal.SIG_DFL)


# 触发所有工具文件自注册：默认扫描 tools/ 下含 registry.register(...) 的模块并 import，
discover()


def _last_user_text(messages):
    """提取最近一条 user 消息的纯文本，用于 trace 记录 query。"""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(b.get("text", "") for b in c
                    if isinstance(b, dict) and b.get("type") == "text")
    return None


def _assistant_text(res):
    """提取模型本轮回复里的文本块，用于 trace 记录。"""
    return "".join(getattr(b, "text", "") for b in res.content
                   if getattr(b, "type", None) == "text")


def _append_file_mutation_notice(messages) -> None:
    """在最终回答后追加仍未恢复的文件修改失败，避免模型过度声称完成。"""
    notice = file_mutation_tracker.format_notice()
    if not notice or not messages or messages[-1].get("role") != "assistant":
        return
    # 必须修改同一条 assistant 消息：相邻的 assistant 消息会破坏 API 的角色交替约束。
    messages[-1]["content"].append(TextBlock(type="text", text=notice))


def agent_loop(messages):
    global _interrupt_requested
    _interrupt_requested = False

    # 每条用户请求开始时，重置重复调用计数、halt 锁存、文件修改失败记录。
    tool_guardrails.reset_for_turn()
    file_mutation_tracker.reset_for_turn()

    # 开始记录本次会话的运行轨迹（每个 query 一份独立 trace 文件，落盘 logs/）
    tr = reset_tracer()

    # 设置 SIGINT 信号的处理函数
    signal.signal(signal.SIGINT, _on_interrupt)

    api_call_count = 0
    exit_reason = None

    try:
        while api_call_count < MAX_ITERATIONS and not _interrupt_requested:
            api_call_count += 1
            tr.step_start(api_call_count, len(messages), True, query=_last_user_text(messages))
            res = client.messages.create(
                model=MODEL_ID,
                messages=messages,
                system=SYSTEM,
                max_tokens=MAX_TOKENS_TOOL,
                tools=registry.definitions(),
                timeout=API_TIMEOUT,
            )

            messages.append({"role": "assistant", "content": res.content})
            tr.step_done(
                res.stop_reason,
                # 模型要调用的所有工具名称
                [b.name for b in res.content if isinstance(b, ToolUseBlock)],
                # 模型回复的 text（str）
                assistant_text=_assistant_text(res),
            )

            if res.stop_reason != "tool_use":
                exit_reason = "completed"
                break
            if _interrupt_requested:
                # 当前 assistant 只包含尚未执行的 tool_use，必须移除后才能总结。
                messages.pop()
                exit_reason = "interrupted"
                break

            # 把本轮 tool_use 块交给执行引擎调度，结果回写 messages
            execute_tool_calls(
                res.content,
                messages,
                concurrent=CONCURRENT_TOOLS,
                max_workers=TOOL_MAX_WORKERS,
            )

            # 与 Hermes 一致：guardrail halt 是一个明确的受控结束，不再额外调用模型总结。
            halt_decision = tool_guardrails.halt_decision
            # 只允许第一次写入，后面即使工具调用成功，也无法修改
            if halt_decision is not None:
                halt_text = f"工具调用已停止：{halt_decision.message}"
                print(f"\n\033[33m⚠️  {halt_text}\033[0m")
                messages.append({
                    "role": "assistant",
                    "content": [TextBlock(type="text", text=halt_text)],
                })
                exit_reason = "guardrail_halt"
                break

        # 正常完成和 guardrail 受控结束都已有最终 assistant 消息，直接返回。
        if exit_reason in {"completed", "guardrail_halt"}:
            _append_file_mutation_notice(messages)
            tr.finish(exit_reason, api_call_count)
            return

        # 只有迭代预算耗尽或中断在未完成工具轮时，才去掉 tools 请求一次收尾总结。
        if _interrupt_requested:
            label = f"interrupted at {api_call_count}/{MAX_ITERATIONS}"
        else:
            label = f"iteration budget exhausted ({api_call_count}/{MAX_ITERATIONS})"
        print(f"\n\033[33m⚠️  {label} — requesting summary...\033[0m")
        res = client.messages.create(
            model=MODEL_ID,  # type: ignore
            messages=messages,
            system=SYSTEM,
            max_tokens=MAX_TOKENS_SUMMARY,
            timeout=API_TIMEOUT,
        )
        messages.append({"role": "assistant", "content": res.content})
        _append_file_mutation_notice(messages)
        tr.finish(label, api_call_count)

    except KeyboardInterrupt:
        get_tracer().finish("force_quit", api_call_count)
        # 第二次 Ctrl+C（_on_interrupt 已把 SIGINT 复位为 SIG_DFL）：强制中断，不崩 REPL
        print("\n\033[33m⚠️  Force quit — 已中断当前任务，返回输入提示。\033[0m")
        # 修复消息历史，避免下轮 API 因「角色不交替」或「tool_use 缺 tool_result」而 400
        if messages:
            last = messages[-1]
            # 丢掉未完成的 assistant(tool_use)（没有对应 tool_result）
            if last.get("role") == "assistant" and any(
                getattr(b, "type", None) == "tool_use" for b in last.get("content", [])
            ):
                messages.pop()
            # 若仍以 user 结尾，补一条 assistant 占位，保证下轮 user/assistant 交替
            if messages and messages[-1].get("role") == "user":
                messages.append({
                    "role": "assistant",
                    "content": [TextBlock(type="text", text="[任务被用户中断]")],
                })
    finally:
        # 恢复默认 SIGINT 行为，确保外层 REPL 的 input() 正常响应 Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        _interrupt_requested = False


if __name__ == "__main__":
    print("输入问题，回车发送。输入'q'、'exit'、''退出。")
    # 历史消息列表
    history = []
    # prompt_toolkit 能正确处理中文宽字符、光标移动和历史输入。
    prompt_session = (
        PromptSession(history=InMemoryHistory()) if sys.stdin.isatty() else None
    )
    # 持续接收用户输入
    while True:
        # 接收用户输入，并检查是否退出
        try:
            if prompt_session is not None:
                query = prompt_session.prompt(ANSI("\033[33m >> \033[0m"))
            else:
                # 管道输入不是交互终端，保留 input() 以支持脚本和 smoke test。
                query = input("\033[33m >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print("\n收到终止信号，退出交互")
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 将用户 query 追加到历史消息列表中
        history.append({"role": "user", "content": query})

        agent_loop(history)

        for block in history[-1]["content"]:
            if block.type == "text":
                print(block.text)
