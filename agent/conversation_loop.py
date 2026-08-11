import os
import signal
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from anthropic import Anthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import load_dotenv
from httpx import Timeout

# 用于实现更好的命令行输入框
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from tools.registry import registry, discover
from agent.config_loader import load_config
from agent.context_compressor import ContextCompressor
from agent.context_state import (
    ContextSettings,
    ContextState,
    estimate_request_tokens_rough,
)
from agent.conversation_compression import compress_context
from agent.interrupt_controller import interrupt_controller
from agent.model_response import ModelStopReason, inspect_model_response
from agent.provider_error_recovery import (
    ProviderRequestFailed,
    ProviderRequestInterrupted,
    classify_provider_error,
    request_with_retries,
)
from agent.session_runtime import SessionRuntime
from agent.session_runtime import SessionSettings
from agent.session_db import SessionDB
from agent.session_commands import handle_session_command
from agent.tool_executor import execute_tool_calls, file_mutation_tracker, tool_guardrails

load_dotenv(override=True)

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

# 发送请求、等待/读取响应：最多 300 秒
API_TIMEOUT = 300.0
# 建立连接：最多 10 秒
API_CONNECT_TIMEOUT = 10.0

# 重试统一由 provider_error_recovery 管理，避免 SDK 内置重试与外层重试相乘。
client = Anthropic(
    base_url=os.getenv("BASE_URL"),
    api_key=_api_key,
    max_retries=0,
    timeout=Timeout(timeout=API_TIMEOUT, connect=API_CONNECT_TIMEOUT),
)
MODEL_ID = _model_id
_project_config = load_config()
# 基于配置初始化一个 ContextSettings 对象
CONTEXT_SETTINGS = ContextSettings.from_mapping(_project_config)
SESSION_SETTINGS = SessionSettings.from_mapping(_project_config)


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
# 最大迭代步数：达到上限后去掉 tools 做一次总结调用优雅收尾
MAX_ITERATIONS = 90
# 工具并发执行：仅工具名互不重复的只读调用会并行；bash、写操作和重复调用顺序执行。
CONCURRENT_TOOLS = True
# 并发线程池上限
TOOL_MAX_WORKERS = 4
# 模型输出与上下文参数统一来自 config.yaml；缺失或非法字段使用代码默认值。
MAX_OUTPUT_TOKENS = CONTEXT_SETTINGS.max_output_tokens
MAX_TOKENS_SUMMARY = CONTEXT_SETTINGS.max_output_tokens
MAX_LENGTH_CONTINUATIONS = CONTEXT_SETTINGS.max_length_continuations
CONTEXT_WINDOW = CONTEXT_SETTINGS.context_window
COMPRESSION_ENABLED = CONTEXT_SETTINGS.compression_enabled
MAX_COMPRESSIONS_PER_TURN = CONTEXT_SETTINGS.max_compressions_per_turn
COMPRESSION_IN_PLACE = CONTEXT_SETTINGS.compression_in_place

_LENGTH_CONTINUATION_PROMPT = (
    "Continue exactly where the previous response stopped. Do not repeat earlier text. "
    "Return plain text only and do not call tools."
)


# ---- agent_loop 的单点收尾 ----
# 把"break + 写字符串 + 白名单同步"三件套收敛到一处：
# 每个出口只赋 TurnOutcome，最终的"是否再走一次 summary 调用"由
# outcome.needs_summary 单一判断，避免字符串白名单漏改。
class TurnOutcome(Enum):
    """枚举类型：一次 turn 的结束方式。"""

    COMPLETED = "completed"                       # 模型自然 end_turn
    INTERRUPTED = "interrupted"                   # 用户 Ctrl+C
    GUARDRAIL_HALT = "guardrail_halt"             # guardrail 触发 halt
    POST_TOOL_EMPTY = "post_tool_empty_response"  # 调用工具后空回复连续两次
    INVALID_TOOL_LIMIT = "invalid_tool_limit"     # 模型调用未知工具连续 3 轮
    ITERATION_BUDGET_EXHAUSTED = "iter_budget"    # 达到 MAX_ITERATIONS，需要总结
    OUTPUT_TRUNCATED = "output_truncated"         # 输出达到 max_tokens 且补救耗尽
    MODEL_REFUSED = "model_refused"               # Provider 正常返回拒绝状态
    UNSUPPORTED_STOP_REASON = "unsupported_stop"  # 未识别或当前不支持的停止原因

    @property
    def needs_summary(self) -> bool:
        """是否需要在主循环后再发一次无 tools 的 summary 调用。"""
        return self is TurnOutcome.ITERATION_BUDGET_EXHAUSTED


# ---- 用户中断机制 ----
_last_interrupt_time = 0.0
FORCE_QUIT_WINDOW_SECONDS = 2.0


# signum：信号编号，不同编号对应不同的中断机制
# frame：栈帧，当前程序执行的快照
def _on_interrupt(signum, frame):
    """第一次 Ctrl+C 调用这个函数打开线程之间的信号开关（告诉线程某个事件发生了），两秒内第二次 ctrl+c 抛出 KeyboardInterrupt。"""
    global _last_interrupt_time
    now = time.monotonic()
    if interrupt_controller.is_requested():
        if now - _last_interrupt_time < FORCE_QUIT_WINDOW_SECONDS:
            print("\n\033[33m⚠️  Force quit requested.\033[0m")
            # 由 REPL 的统一异常边界结束进程；agent_loop 的 finally 仍会先恢复
            # SIGINT handler 和清理本轮中断状态。
            raise KeyboardInterrupt
        _last_interrupt_time = now
        print(
            "\n\033[33m⚠️  Pause still in progress — "
            "press Ctrl+C again within 2 seconds to quit.\033[0m"
        )
        return
    # 打开信号开关
    interrupt_controller.request()
    _last_interrupt_time = now
    print(
        "\n\033[33m⚠️  Pause requested — keeping output generated so far... "
        "(Ctrl+C again within 2 seconds to quit)\033[0m"
    )


def _handle_idle_ctrl_c(event) -> None:
    """空闲 REPL 中，优先清空已有输入；空输入框才退出。"""
    # event：事件对象；event.app：当前应用对象
    buffer = event.app.current_buffer
    if buffer.text:
        # 清空输入缓冲区
        buffer.reset()
        # 请求刷新中断界面，因为内容已经清空
        event.app.invalidate()
        return
    event.app.exit(exception=KeyboardInterrupt)


def _create_repl_key_bindings() -> KeyBindings:
    """创建 REPL 按键绑定：将 ctrl+c 与 _handle_idle_ctrl_c 函数绑定
    _handle_idle_ctrl_c（处理空闲状态在 ctrl+c，就是用户输入状态）：优先清空已有输入；空输入框才退出
    """
    bindings = KeyBindings()
    # 创建快捷键并绑定回调函数
    bindings.add("c-c")(_handle_idle_ctrl_c)
    return bindings


# 触发所有工具文件自注册：默认扫描 tools/ 下含 registry.register(...) 的模块并 import，
discover()


def _assistant_text(res):
    """提取模型本轮回复里的文本块中的 text 并用 "" 拼接。"""
    return "".join(getattr(b, "text", "") for b in res.content
                   if getattr(b, "type", None) == "text")


def _is_empty_model_response(res) -> bool:
    """模型响应没有工具调用 block 且 text block 中的 text 内容为空。"""
    has_tool_call = any(isinstance(block, ToolUseBlock) for block in res.content)
    return not has_tool_call and not _assistant_text(res).strip()


def _invalid_tool_results(tool_calls: list[ToolUseBlock]) -> list[dict] | None:
    """发现未知工具时返回整批 synthetic results：
    1. 工具名为空："Tool call rejected: the tool name was empty. Use a valid name from the available tool list."
    2. 工具名不在可调用名单中：Unknown tool: {工具名称}. Available tools: {可用工具清单：工具1， 工具2， ....}
    3. 工具存在但这批次存在未知工具："Skipped: another tool call in this batch used an invalid name. Please retry this tool call."

    本批调用工具全部存在时返回 None。"""
    invalid_names = {call.name for call in tool_calls if registry.get_entry(call.name) is None}
    if not invalid_names:
        return None

    available = ", ".join(registry.names()) or "(none)"
    results = []
    for call in tool_calls:
        if call.name in invalid_names:
            # tool name 为空
            if not call.name.strip():
                content = (
                    "Tool call rejected: the tool name was empty. "
                    "Use a valid name from the available tool list."
                )
            else:
                # 未知工具，模型幻觉
                content = f"Unknown tool: {call.name}. Available tools: {available}"
        else:
            # 工具存在，但是由于这批次工具调用存在未知工具所以不调用。retry：重试
            content = (
                "Skipped: another tool call in this batch used an invalid name. "
                "Please retry this tool call."
            )
        results.append({
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": content,
        })
    return results


def _append_file_mutation_notice(messages) -> str:
    """在最终回答后追加仍未恢复的文件修改失败，避免模型过度声称完成。"""
    notice = file_mutation_tracker.format_notice()
    if not notice or not messages or messages[-1].get("role") != "assistant":
        return ""
    # 必须修改同一条 assistant 消息：相邻的 assistant 消息会破坏 API 的角色交替约束。
    messages[-1]["content"].append(TextBlock(type="text", text=notice))
    return notice


def _on_retry(error, retry_number, wait_seconds):
    """打印 Provider 重试提示。"""
    print(
        f"\033[33m⚠️  API {error.kind}，{wait_seconds:.1f} 秒后重试 "
        f"（{retry_number}/3）\033[0m"
    )


class _StreamWorker:
    """流式 Provider 请求 + 中断协作的单一封装。

    SDK 的流迭代可能长时间阻塞在网络读取中，因此把整条调用链放进 daemon worker。
    主线程持续轮询 Ctrl+C，主动 close 当前 stream、锁存一个 cancel signal；worker
    通过同一个 signal 知道自己已被取消但仍会跑完资源清理。

    锁、cancel signal、worker thread 都收到内部——外部只通过 6 个动作驱动：
      - start()           启动 daemon worker（后台工作线程）
      - poll(0.05)         轮询是否已结束
      - request_cancel()  锁存本次请求已取消
      - close_active()    主动关闭当前 stream，唤醒阻塞在迭代里的 worker
      - drain(0.5)         取消后再等一段收尾窗口
      - finalize()        取出 (response, error, partial_text, emitted_text)
    """

    # 轮询间隔
    POLL_INTERVAL = 0.05
    # 取消后清空
    POST_CANCEL_DRAIN = 0.5

    def __init__(self, stream_method, stream_kwargs, on_text, on_retry):
        self._state_lock = threading.Lock()
        self._finished = threading.Event()
        self._request_cancelled = threading.Event() # 表示本次流式请求是否已被取消
        self._state = {
            "manager": None,
            "stream": None,
            "response": None,
            "error": None,
            "text_parts": [], # 模型回复的文本
            "emitted_text": False, # 模型是否已经输出文本
        }
        self._stream_method = stream_method
        self._stream_kwargs = stream_kwargs
        self._on_text = on_text
        self._on_retry = on_retry

    def _is_cancelled(self) -> bool:
        """检查流式请求信号和中断信号是否已触发"""
        # _request_cancelled 是本次请求的锁存信号：一旦设置为 True，在这次对象的生命周期里就保持 True，不会自动恢复
        return self._request_cancelled.is_set() or interrupt_controller.is_requested()

    def _consume_stream(self) -> None:
        manager = None
        try:
            def open_stream():
                # self._stream_method(**self._stream_kwargs)就是调用流式接口client.messages.stream
                candidate_manager = self._stream_method(**self._stream_kwargs)
                candidate_stream = candidate_manager.__enter__()
                with self._state_lock:
                    self._state["manager"] = candidate_manager
                    self._state["stream"] = candidate_stream
                # MessageStreamManager 对象，MessageStream 对象
                return candidate_manager, candidate_stream

            manager, stream = request_with_retries(
                open_stream,
                max_retries=3,
                is_interrupted=self._is_cancelled,
                on_retry=self._on_retry,
            )
            # 中断可能发生在 __enter__ 阻塞期间。当时 stream 尚未发布，
            # close_active() 无对象可关；建立完成后在迭代前再次检查即可由
            # finally 调用 manager.__exit__，避免进入下一次阻塞读取。
            if self._is_cancelled():
                return
            for event in stream:
                if self._is_cancelled():
                    break
                if (
                    getattr(event, "type", None) == "content_block_delta"
                    and getattr(getattr(event, "delta", None), "type", None) == "text_delta"
                ):
                    delta = event.delta.text
                    if delta:
                        with self._state_lock:
                            self._state["text_parts"].append(delta)
                            self._state["emitted_text"] = True
                        if self._on_text is not None:
                            self._on_text(delta)
            if not self._is_cancelled():
                # 获取 ParsedMessage 对象：类似 create 接口返回的内容，但是对象类型不同
                self._state["response"] = stream.get_final_message()
        except (ProviderRequestFailed, ProviderRequestInterrupted) as exc:
            self._state["error"] = exc
        except Exception as exc:
            if not self._is_cancelled():
                # 流式输出一旦开始就不能安全重试，否则用户会看到重复文本。
                self._state["error"] = ProviderRequestFailed(
                    classify_provider_error(exc),
                    exc,
                )
        finally:
            if manager is not None:
                try:
                    manager.__exit__(None, None, None)
                except Exception as exc:
                    if not self._is_cancelled() and self._state["error"] is None:
                        self._state["error"] = ProviderRequestFailed(
                            classify_provider_error(exc),
                            exc,
                        )
            self._finished.set()

    def start(self) -> None:
        """创建一个后台线程对象并启动这个 daemon worker。"""
        threading.Thread(target=self._consume_stream, name="model-stream", daemon=True).start()

    def poll(self, timeout: float) -> bool:
        """等待 _finished 信号 timeout 秒；返回 True 表示已结束。"""
        return self._finished.wait(timeout)

    def request_cancel(self) -> None:
        """锁存本次请求已取消。worker 仍会跑完，但结果被丢弃。"""
        self._request_cancelled.set()

    def close_active(self) -> None:
        """主动关闭当前 stream，唤醒阻塞在迭代里的 worker。"""
        with self._state_lock:
            active_stream = self._state["stream"]
        if active_stream is not None:
            try:
                active_stream.close()
            except Exception:
                pass

    def drain(self, timeout: float) -> None:
        """取消后等一段收尾窗口；即使 Provider 不响应，daemon worker 也不会拖住主线程。"""
        self._finished.wait(timeout)

    def finalize(self) -> tuple[Any, Exception | None, str, bool]:
        """取出 worker 写完的所有状态。"""
        with self._state_lock:
            return (
                self._state["response"],
                self._state["error"],
                "".join(self._state["text_parts"]),
                self._state["emitted_text"],
            )


def _stream_message_with_recovery(
    *,
    on_text: Callable[[str], None] | None = None,
    **kwargs,
):
    """流式请求模型；中断时返回已生成的文本，不保留未完成的工具调用。

    流建立前的错误沿用 Provider 重试；流已经开始后不自动重放，避免重复输出。
    """
    worker = _StreamWorker(client.messages.stream, kwargs, on_text, _on_retry)
    worker.start()

    # 轮询检查中断状态
    while not worker.poll(_StreamWorker.POLL_INTERVAL):
        if not interrupt_controller.is_requested():
            continue
        worker.request_cancel()
        worker.close_active()
        break

    if interrupt_controller.is_requested():
        worker.request_cancel()
        # 给 close 后的 worker 一个短暂收尾窗口；即使 Provider 不响应，daemon
        # worker 也不会阻止 REPL 立即返回或第二次 Ctrl+C 退出进程。
        worker.drain(_StreamWorker.POST_CANCEL_DRAIN)
        _, _, partial_text, _ = worker.finalize()
        content = (
            [TextBlock(type="text", text=partial_text)]
            if partial_text
            else []
        )
        return SimpleNamespace(stop_reason="interrupted", content=content), True

    response, error, _, emitted_text = worker.finalize()
    if error is not None:
        raise error
    if emitted_text and on_text is not None:
        on_text("\n")
    return response, False


def _paused_text_blocks(content) -> tuple[list[TextBlock], bool]:
    """仅保留模型已经生成的文本，丢弃未执行或未完成的工具调用。

    返回 (blocks, is_placeholder)：is_placeholder=True 表示模型没来得及输出任何文本。
    """
    blocks = [block for block in content if isinstance(block, TextBlock) and block.text]
    if blocks:
        return blocks, False
    return [TextBlock(type="text", text="[任务已暂停，模型尚未返回文本]")], True


def _emit_assistant_message(
    messages,
    text: str,
    *,
    stream_output: bool,
    is_placeholder: bool = False,
    stream_message: str = "",
) -> None:
    """往 messages 追加一条 assistant 文本消息，并在 stream_output 时打印。

    在模型没有打印文本的时候就中断，提示用户模型尚未打印文本；否则打印模型返回文本。
    """
    messages.append({
        "role": "assistant",
        "content": [TextBlock(type="text", text=text)],
    })
    if stream_output:
        if is_placeholder:
            print(f"\033[90m【暂停占位】{text}\033[0m")
        elif stream_message:
            print(stream_message)
        else:
            print(text)


def _append_to_last_assistant(messages, text: str, *, stream_output: bool) -> None:
    """给最后一条 assistant 消息追加说明，避免产生相邻 assistant 消息。"""
    if not messages or messages[-1].get("role") != "assistant":
        _emit_assistant_message(messages, text, stream_output=stream_output)
        return
    content = messages[-1].get("content")
    if not isinstance(content, list):
        content = [TextBlock(type="text", text=str(content or ""))]
        messages[-1]["content"] = content
    content.append(TextBlock(type="text", text=text))
    if stream_output:
        print(text)


def _generate_context_summary(prompt: str, max_tokens: int) -> str:
    """使用当前模型生成压缩摘要；不提供工具，不接受截断或中断的摘要。"""
    res, interrupted = _stream_message_with_recovery(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        system=(
            "You compact coding-agent conversation history into a precise checkpoint. "
            "Do not answer the user and do not call tools."
        ),
        max_tokens=max_tokens,
    )
    if interrupted:
        raise RuntimeError("context compression interrupted")
    response_state = inspect_model_response(res)
    if response_state.stop_reason is ModelStopReason.MAX_TOKENS:
        raise RuntimeError("context summary reached max_tokens")
    if response_state.stop_reason is ModelStopReason.REFUSAL:
        raise RuntimeError("context summary was refused by the model")
    summary = _assistant_text(res).strip()
    if not summary:
        raise RuntimeError("context summary was empty")
    return summary


def _create_context_compressor() -> ContextCompressor:
    """按 config.yaml 创建本会话唯一的内置压缩器。"""
    return ContextCompressor(
        context_length=CONTEXT_SETTINGS.context_window,
        max_tokens=CONTEXT_SETTINGS.max_output_tokens,
        threshold_percent=CONTEXT_SETTINGS.compression_threshold,
        protect_first_n=CONTEXT_SETTINGS.protect_first_n,
        protect_last_n=CONTEXT_SETTINGS.protect_last_n,
        summary_target_ratio=CONTEXT_SETTINGS.compression_target_ratio,
        abort_on_summary_failure=CONTEXT_SETTINGS.abort_on_summary_failure,
        summary_callback=_generate_context_summary,
    )


def _run_context_compression(
    messages,
    *,
    context_state: ContextState,
    context_compressor: ContextCompressor,
    tool_definitions: list[dict],
    current_tokens: int,
    session_runtime: SessionRuntime | None = None,
) -> bool:
    """执行一次压缩并显示结果；返回是否真正替换了历史。"""
    print("\n\033[33m⟳ 正在压缩较早的对话上下文...\033[0m")
    result = compress_context(
        context_compressor,
        context_state,
        messages,
        system=SYSTEM,
        tools=tool_definitions,
        current_tokens=current_tokens,
        session_runtime=session_runtime,
        in_place=COMPRESSION_IN_PLACE,
    )
    if result.changed:
        print(
            "\033[33m✓ 上下文压缩完成："
            f"{result.before_messages} → {result.after_messages} 条消息\033[0m"
        )
        return True
    if result.error:
        print(
            "\033[33m⚠️  上下文压缩失败，原历史未修改："
            f"{result.error}\033[0m"
        )
    return False


def _request_summary(
    messages,
    *,
    stream_output: bool,
    context_state: ContextState,
) -> None:
    """迭代预算耗尽后，去掉 tools 再请求一次纯文本总结。"""
    label = f"iteration budget exhausted ({MAX_ITERATIONS}/{MAX_ITERATIONS})"
    # 真正的 api_call_count 由调用方在调用 _request_summary 之前已 ++
    print(f"\n\033[33m⚠️  {label} — requesting summary...\033[0m")
    summary_system = (
        SYSTEM
        + "\nThe tool iteration budget is exhausted. Reply with a concise plain-text "
        "summary only. Do not call tools or emit tool-call markup."
    )
    res, interrupted = _stream_message_with_recovery(
        model=MODEL_ID,  # type: ignore
        messages=messages,
        system=summary_system,
        max_tokens=MAX_TOKENS_SUMMARY,
        # 打印流式输出
        on_text=(lambda text: print(text, end="", flush=True)) if stream_output else None,
    )
    if not interrupted:
        response_state = inspect_model_response(res)
        context_state.update_from_response(
            response_state,
            system=summary_system,
            messages=messages,
            tools=[],
            response_content=res.content,
        )
    messages.append({"role": "assistant", "content": res.content})


def _restore_context_components(
    runtime: SessionRuntime,
    messages: list[dict[str, Any]],
) -> tuple[ContextState, ContextCompressor]:
    """根据持久会话重建派生 token 状态；消息内容只使用传入的 history。"""
    context_state = ContextState.from_settings(CONTEXT_SETTINGS)
    session = runtime.db.get_session(runtime.session_id) or {}
    current_tokens = estimate_request_tokens_rough(
        system=SYSTEM,
        messages=messages,
        tools=registry.definitions(),
    )
    context_state.restore_session_totals(
        input_tokens=int(session.get("input_tokens") or 0),
        output_tokens=int(session.get("output_tokens") or 0),
        current_input_tokens=current_tokens,
    )
    compressor = _create_context_compressor()
    context_state._context_compressor = compressor
    return context_state, compressor


def agent_loop(
    messages,
    *,
    stream_output: bool = False,
    context_state: ContextState | None = None,
    context_compressor: ContextCompressor | None = None,
    session_runtime: SessionRuntime | None = None,
):
    global _last_interrupt_time
    interrupt_controller.clear()
    _last_interrupt_time = 0.0
    if context_state is None:
        context_state = ContextState.from_settings(CONTEXT_SETTINGS)
    if context_compressor is None:
        context_compressor = context_state._context_compressor
        if context_compressor is None:
            context_compressor = _create_context_compressor()
            context_state._context_compressor = context_compressor

    # 每条用户请求开始时，重置重复调用计数、halt 锁存、文件修改失败记录。
    tool_guardrails.reset_for_turn()
    file_mutation_tracker.reset_for_turn()

    # 仅在 loop 执行期间接管 Ctrl+C；结束后恢复进入 loop 前的 handler。
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _on_interrupt)

    # 当前用户请求已经向模型发起的 API 调用次数，用于限制最大迭代轮数。
    api_call_count = 0
    # 本轮停止的原因；正常完成、中断或异常收尾时写入对应标识。
    outcome: TurnOutcome | None = None
    # 上一轮是否调过工具（用于检测"工具执行后模型空回复"异常）。
    prev_step_had_tool_calls = False
    # 工具执行后模型首次返回空回复时是否已经做过一次补救，避免无限重试。
    post_tool_empty_retried = False
    # 本轮模型调用未知工具的连续重试次数，达到上限后停止本轮。
    invalid_tool_retries = 0
    # 纯文本截断和工具调用截断分别有界重试，互不混用计数。
    length_continuations = 0
    truncated_response_retries = 0
    request_max_tokens = MAX_OUTPUT_TOKENS
    compression_attempts_this_turn = 0

    try:
        while api_call_count < MAX_ITERATIONS and not interrupt_controller.is_requested():
            if session_runtime is not None:
                try:
                    session_runtime.flush_new_messages(messages)
                except Exception as exc:
                    print(f"\n\033[33m⚠️  会话保存失败：{exc}\033[0m")
            tool_definitions = registry.definitions()
            rough_tokens = estimate_request_tokens_rough(
                system=SYSTEM,
                messages=messages,
                tools=tool_definitions,
            )
            if (
                COMPRESSION_ENABLED
                and compression_attempts_this_turn < MAX_COMPRESSIONS_PER_TURN
                and context_compressor.should_compress_preflight(
                    messages,
                    rough_tokens=rough_tokens,
                )
            ):
                _run_context_compression(
                    messages,
                    context_state=context_state,
                    context_compressor=context_compressor,
                    tool_definitions=tool_definitions,
                    current_tokens=rough_tokens,
                    session_runtime=session_runtime,
                )
                compression_attempts_this_turn += 1
                if interrupt_controller.is_requested():
                    break

            api_call_count += 1
            res, stream_interrupted = _stream_message_with_recovery(
                model=MODEL_ID,
                messages=messages,
                system=SYSTEM,
                max_tokens=request_max_tokens,
                tools=tool_definitions,
                on_text=(lambda text: print(text, end="", flush=True)) if stream_output else None,
            )

            # 处理流式输出时的中断
            if stream_interrupted:
                # is_placeholder：模型是否已经生成文本（True 表示没有；False 表示生成了）
                paused_content, is_placeholder = _paused_text_blocks(res.content)
                messages.append({"role": "assistant", "content": paused_content})
                # 流式输出开启但是模型尚未返回文本块
                if stream_output and not res.content:
                    text = paused_content[0].text
                    if is_placeholder:
                        print(f"\033[90m【暂停占位】{text}\033[0m")
                    else:
                        print(text)
                outcome = TurnOutcome.INTERRUPTED
                break

            response_state = inspect_model_response(res)
            usage = context_state.update_from_response(
                response_state,
                system=SYSTEM,
                messages=messages,
                tools=tool_definitions,
                response_content=res.content,
            )
            context_compressor.update_from_response({
                "prompt_tokens": usage.effective_input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.effective_input_tokens + usage.output_tokens,
                "estimated": usage.estimated,
            })
            if session_runtime is not None:
                try:
                    session_runtime.record_usage(usage)
                except Exception as exc:
                    print(f"\n\033[33m⚠️  会话 usage 保存失败：{exc}\033[0m")

            # 长度终止必须先于空回复和普通完成处理。截断的工具参数不可信，
            # 因此不写入历史也不执行，只提高输出预算后重试同一个请求。
            if response_state.stop_reason is ModelStopReason.MAX_TOKENS:
                if response_state.has_tool_calls or not response_state.has_text:
                    truncated_kind = (
                        "工具调用" if response_state.has_tool_calls else "空响应"
                    )
                    if truncated_response_retries < MAX_LENGTH_CONTINUATIONS:
                        truncated_response_retries += 1
                        request_max_tokens = min(
                            MAX_OUTPUT_TOKENS * (truncated_response_retries + 1),
                            CONTEXT_WINDOW,
                        )
                        print(
                            f"\033[33m⚠️  模型{truncated_kind}被截断"
                            + ("，未执行" if response_state.has_tool_calls else "")
                            + "；正在扩大输出预算重试 "
                            f"（{truncated_response_retries}/{MAX_LENGTH_CONTINUATIONS}）\033[0m"
                        )
                        continue
                    stopped_text = (
                        "[模型输出持续被截断，未执行不完整的工具调用，本轮已停止]"
                        if response_state.has_tool_calls
                        else "[模型持续在返回可见文本前达到输出上限，本轮已停止]"
                    )
                    _emit_assistant_message(
                        messages,
                        stopped_text,
                        stream_output=stream_output,
                    )
                    outcome = TurnOutcome.OUTPUT_TRUNCATED
                    break

                messages.append({"role": "assistant", "content": res.content})
                prev_step_had_tool_calls = False
                if length_continuations < MAX_LENGTH_CONTINUATIONS:
                    length_continuations += 1
                    messages.append({
                        "role": "user",
                        "content": _LENGTH_CONTINUATION_PROMPT,
                    })
                    print(
                        "\033[33m⚠️  模型文本达到输出上限，正在从断点继续 "
                        f"（{length_continuations}/{MAX_LENGTH_CONTINUATIONS}）\033[0m"
                    )
                    continue
                _append_to_last_assistant(
                    messages,
                    "\n[输出仍然达到长度上限，以上回答可能不完整]",
                    stream_output=stream_output,
                )
                outcome = TurnOutcome.OUTPUT_TRUNCATED
                break

            # 处理工具调用之后模型返回空消息的响应异常，给模型一次重试机会，重试失败就退出
            if prev_step_had_tool_calls and _is_empty_model_response(res):
                if not post_tool_empty_retried:
                    post_tool_empty_retried = True
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": [TextBlock(type="text", text="(empty response)")],
                        },
                        {
                            "role": "user",
                            "content": (
                                "The previous response was empty. Process the tool results "
                                "and provide a complete answer to the user."
                            ),
                        },
                    ])
                    print("\033[33m⚠️  模型在工具执行后返回空回复，正在补救一次...\033[0m")
                    continue

                stopped_text = "[模型在工具执行后连续返回空回复，本轮已停止]"
                _emit_assistant_message(
                    messages, stopped_text, stream_output=stream_output
                )
                outcome = TurnOutcome.POST_TOOL_EMPTY
                break

            messages.append({"role": "assistant", "content": res.content})
            prev_step_had_tool_calls = False
            request_max_tokens = MAX_OUTPUT_TOKENS
            truncated_response_retries = 0

            if response_state.stop_reason in {
                ModelStopReason.END_TURN,
                ModelStopReason.STOP_SEQUENCE,
            }:
                outcome = TurnOutcome.COMPLETED
                break
            if response_state.stop_reason is ModelStopReason.REFUSAL:
                if not response_state.has_text:
                    _append_to_last_assistant(
                        messages,
                        "[模型拒绝了该请求，且未提供说明]",
                        stream_output=stream_output,
                    )
                outcome = TurnOutcome.MODEL_REFUSED
                break
            if response_state.stop_reason is not ModelStopReason.TOOL_USE:
                reason = response_state.raw_stop_reason or "missing"
                _append_to_last_assistant(
                    messages,
                    f"[模型以当前 Agent 不支持的原因停止：{reason}]",
                    stream_output=stream_output,
                )
                outcome = TurnOutcome.UNSUPPORTED_STOP_REASON
                break
            # 处理整个 loop 过程中的中断
            if interrupt_controller.is_requested():
                # 不执行模型刚生成的工具调用，只保留它在暂停前已经输出的文本。
                paused_content, is_placeholder = _paused_text_blocks(res.content)
                messages[-1]["content"] = paused_content
                if stream_output and not any(isinstance(b, TextBlock) and b.text for b in res.content):
                    text = paused_content[0].text
                    if is_placeholder:
                        print(f"\033[90m【暂停占位】{text}\033[0m")
                    else:
                        print(text)
                outcome = TurnOutcome.INTERRUPTED
                break

            # 检查当前批次模型需要调用的工具是否存在，如果模型在本轮调用了三次或者以上的未知工具，直接 break
            tool_calls = [
                block for block in res.content if isinstance(block, ToolUseBlock)
            ]
            invalid_results = _invalid_tool_results(tool_calls)
            if invalid_results is not None:
                invalid_tool_retries += 1
                messages.append({"role": "user", "content": invalid_results})
                print(
                    "\033[33m⚠️  模型调用了未知工具，已返回可用工具列表 "
                    f"（{invalid_tool_retries}/3）\033[0m"
                )
                if invalid_tool_retries >= 3:
                    stopped_text = "[模型连续三轮调用未知工具，本轮已停止]"
                    _emit_assistant_message(
                        messages, stopped_text, stream_output=stream_output
                    )
                    outcome = TurnOutcome.INVALID_TOOL_LIMIT
                    break
                prev_step_had_tool_calls = True
                continue

            invalid_tool_retries = 0

            # 把本轮 tool_use 块交给执行引擎调度，结果回写 messages
            execute_tool_calls(
                res.content,
                messages,
                concurrent=CONCURRENT_TOOLS,
                max_workers=TOOL_MAX_WORKERS,
                is_cancelled=interrupt_controller.is_requested,
            )
            prev_step_had_tool_calls = True

            halt_decision = tool_guardrails.halt_decision
            # 只允许第一次写入，后面即使工具调用成功，也无法修改
            if halt_decision is not None:
                halt_text = f"工具调用已停止：{halt_decision.message}"
                print(f"\n\033[33m⚠️  {halt_text}\033[0m")
                messages.append({
                    "role": "assistant",
                    "content": [TextBlock(type="text", text=halt_text)],
                })
                outcome = TurnOutcome.GUARDRAIL_HALT
                break

            # Hermes 的 post-response 触发：工具结果写回后，用本次真实 prompt usage
            # 判断是否立即压缩；下一轮 API 前仍会再走一次 rough preflight 兜底。
            if (
                COMPRESSION_ENABLED
                and compression_attempts_this_turn < MAX_COMPRESSIONS_PER_TURN
                and context_compressor.should_compress()
            ):
                _run_context_compression(
                    messages,
                    context_state=context_state,
                    context_compressor=context_compressor,
                    tool_definitions=tool_definitions,
                    current_tokens=max(0, context_compressor.last_prompt_tokens),
                    session_runtime=session_runtime,
                )
                compression_attempts_this_turn += 1
                if interrupt_controller.is_requested():
                    break

        # 主循环结束但还没确定 outcome → 要么 iter 预算耗尽，要么 iter 中收到中断。
        if outcome is None:
            if interrupt_controller.is_requested():
                # 可能在工具执行期间收到暂停信号。工具结果已经写回时，补一条
                # assistant 占位消息，保持下一轮请求的角色与工具协议完整。
                if not messages or messages[-1].get("role") != "assistant":
                    paused_content, is_placeholder = _paused_text_blocks([])
                    messages.append({"role": "assistant", "content": paused_content})
                    if stream_output:
                        text = paused_content[0].text
                        if is_placeholder:
                            print(f"\033[90m【暂停占位】{text}\033[0m")
                        else:
                            print(text)
                outcome = TurnOutcome.INTERRUPTED
            else:
                outcome = TurnOutcome.ITERATION_BUDGET_EXHAUSTED

        # 唯一决策点：是否需要再发一次 summary 调用。
        # summary 不算入 api_call_count——它本质上是"预算耗尽"事件的副作用。
        if outcome.needs_summary:
            _request_summary(
                messages,
                stream_output=stream_output,
                context_state=context_state,
            )

        notice = _append_file_mutation_notice(messages)
        if stream_output and notice:
            print(notice)

    except ProviderRequestInterrupted:
        interrupted_text = "[API 重试已被用户暂停]"
        _emit_assistant_message(messages, interrupted_text, stream_output=stream_output)
    except ProviderRequestFailed as exc:
        error = exc.error
        detail = str(exc.cause).replace("\n", " ")[:500]
        message = f"模型服务请求失败（{error.kind}）"
        if error.status_code is not None:
            message += f"，HTTP {error.status_code}"
        if detail:
            message += f"：{detail}"
        _emit_assistant_message(messages, message, stream_output=stream_output)
    finally:
        if session_runtime is not None:
            try:
                session_runtime.flush_new_messages(messages)
            except Exception as exc:
                print(f"\n\033[33m⚠️  会话保存失败：{exc}\033[0m")
        signal.signal(signal.SIGINT, previous_sigint_handler)
        interrupt_controller.clear()
        _last_interrupt_time = 0.0


if __name__ == "__main__":
    # 历史消息列表
    history = []
    session_context = ContextState.from_settings(CONTEXT_SETTINGS)
    session_compressor = _create_context_compressor()
    session_db: SessionDB | None = None
    session_runtime: SessionRuntime | None = None
    if SESSION_SETTINGS.enabled:
        try:
            project_root = Path(__file__).resolve().parents[1]
            session_db = SessionDB(
                SESSION_SETTINGS.resolve_database_path(project_root)
            )
            recent_sessions = session_db.list_sessions_rich(limit=1)
            if SESSION_SETTINGS.auto_resume and recent_sessions:
                session_runtime, history = SessionRuntime.resume(
                    session_db,
                    recent_sessions[0]["id"],
                )
                print(f"\033[90m已恢复会话：{session_runtime.session_id}\033[0m")
            else:
                session_runtime = SessionRuntime.start(
                    session_db,
                    model=MODEL_ID,
                    model_config={"max_output_tokens": MAX_OUTPUT_TOKENS},
                    system_prompt=SYSTEM,
                )
            session_context, session_compressor = _restore_context_components(
                session_runtime,
                history,
            )
        except Exception as exc:
            if session_db is not None:
                session_db.close()
            session_db = None
            session_runtime = None
            print(f"\033[33m⚠️  会话数据库初始化失败，改用内存模式：{exc}\033[0m")
    # prompt_toolkit 能正确处理中文宽字符、光标移动和历史输入。
    prompt_session = (
        PromptSession(
            # 用于实现命令历史记录功能
            history=InMemoryHistory(),
            # 一次 ctrl+c 清空内容，两次退出程序
            key_bindings=_create_repl_key_bindings(),
        )
        # 判断程序的标准输入流式是否是终端
        if sys.stdin.isatty()
        else None
    )
    # REPL：交互式循环
    while True:
        # 第二次 Ctrl+C 抛出的 KeyboardInterrupt。
        try:
            if prompt_session is not None:
                query = prompt_session.prompt(ANSI("\033[33m >> \033[0m"))
            else:
                # 管道输入不是交互终端，保留 input() 以支持脚本和 smoke test。
                query = input("\033[33m >> \033[0m")

            if session_db is not None and session_runtime is not None:
                try:
                    command_result = handle_session_command(
                        query,
                        db=session_db,
                        runtime=session_runtime,
                        model=MODEL_ID,
                        system_prompt=SYSTEM,
                    )
                except (KeyError, ValueError) as exc:
                    print(f"\033[33m⚠️  {exc}\033[0m")
                    continue
                if command_result.handled:
                    if command_result.output:
                        print(command_result.output)
                    if command_result.runtime is not None:
                        session_runtime = command_result.runtime
                    if command_result.messages is not None:
                        history = command_result.messages
                    if command_result.reset_context:
                        session_context, session_compressor = (
                            _restore_context_components(session_runtime, history)
                        )
                    continue
                if query.strip().startswith("/"):
                    print(
                        f"未知命令：{query.strip().split()[0]}。"
                        "可用会话命令：/new、/sessions、/resume、/search、/archive"
                    )
                    continue

            # 将用户 query 追加到历史消息列表中
            history.append({"role": "user", "content": query})

            agent_loop(
                history,
                stream_output=True,
                context_state=session_context,
                context_compressor=session_compressor,
                session_runtime=session_runtime,
            )
        except EOFError:
            print("\n输入结束，退出交互")
            break
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，退出交互")
            break
    if session_runtime is not None:
        try:
            session_runtime.flush_new_messages(history)
            session_runtime.end("user_exit")
        except Exception as exc:
            print(f"\033[33m⚠️  退出时保存会话失败：{exc}\033[0m")
    if session_db is not None:
        session_db.close()
