import os
import signal
import sys
import threading
import time
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable

from anthropic import Anthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import load_dotenv

# 用于实现更好的命令行输入框
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from tools.registry import registry, discover
from agent.interrupt_controller import interrupt_controller
from agent.provider_error_recovery import (
    ProviderRequestFailed,
    ProviderRequestInterrupted,
    classify_provider_error,
    request_with_retries,
)
from agent.tool_executor import execute_tool_calls, file_mutation_tracker, tool_guardrails
from agent.tracer import reset_tracer, get_tracer

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

# 重试统一由 provider_error_recovery 管理，避免 SDK 内置重试与外层重试相乘。
client = Anthropic(base_url=os.getenv("BASE_URL"), api_key=_api_key, max_retries=0)
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
# 最大迭代步数：达到上限后去掉 tools 做一次总结调用优雅收尾
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


def _is_empty_model_response(res) -> bool:
    """模型响应是否既没有工具调用，也没有非空文本。"""
    has_tool_call = any(isinstance(block, ToolUseBlock) for block in res.content)
    return not has_tool_call and not _assistant_text(res).strip()


def _invalid_tool_results(tool_calls: list[ToolUseBlock]) -> list[dict] | None:
    """发现未知工具时返回整批 synthetic results：
    1. 工具名为空："Tool call rejected: the tool name was empty. Use a valid name from the available tool list."
    2. 工具名不在可调用名单中：Unknown tool: {工具名称}. Available tools: {可用工具清单：工具1， 工具2， ....}
    3. 工具存在："Skipped: another tool call in this batch used an invalid name. Please retry this tool call."

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
    """打印 Provider 重试提示（_create_message_with_recovery / _stream_message_with_recovery 共用）。"""
    print(
        f"\033[33m⚠️  API {error.kind}，{wait_seconds:.1f} 秒后重试 "
        f"（{retry_number}/3）\033[0m"
    )


def _create_message_with_recovery(**kwargs):
    """调用模型 Provider，并对暂时性错误执行有界重试。"""
    return request_with_retries(
        lambda: client.messages.create(**kwargs),
        max_retries=3,
        # 这里为什么不直接传变量本身，如果传变量本身，那么 is_interrupted 的值就固定下来，
        # 如果用户在重试的过程中按了 ctrl+c，is_interrupted 值不会变
        is_interrupted=interrupt_controller.is_requested,
        on_retry=_on_retry,
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
        self._request_cancelled = threading.Event()
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
        # request_cancelled 是本次请求的锁存信号：agent_loop 返回后会清除全局
        # controller，但遗留 worker 仍必须知道自己已经被取消。
        return self._request_cancelled.is_set() or interrupt_controller.is_requested()

    def _consume_stream(self) -> None:
        manager = None
        try:
            def open_stream():
                # Provider 的 stream 创建和 __enter__ 可能阻塞网络，不能持有
                # 状态锁；否则主线程中断时 close_active() 会卡在等同一把锁。
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
        """等待 worker 完成；返回 True 表示已结束。"""
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

    测试 fake client 若没有 ``messages.stream``，自动回退到普通 create 接口。
    流建立前的错误沿用 Provider 重试；流已经开始后不自动重放，避免重复输出。
    """
    # 尝试获取 Messages 对象下面的 stream 方法，最常用的是 create 方法
    # 兼容不支持 .stream() 的客户端实现和兼容测试里的 fake client
    # 调用链路：有可用的 stream 就用，没有就用 create 方法
    stream_method = getattr(client.messages, "stream", None)
    if not callable(stream_method):
        return _create_message_with_recovery(**kwargs), False

    # 复制字典，原来的 kwargs 并没有被替换
    stream_kwargs = dict(kwargs)
    request_timeout = stream_kwargs.pop("timeout", None)
    # 取 client 下面的 with_options 方法 = copy 方法的别名：作用是创建一个新的客户端实例，复用当前客户端所使用的全部配置项，并支持选择性覆盖部分配置（通过关键字传入新的值）
    with_options = getattr(client, "with_options", None)
    if request_timeout is not None and callable(with_options):
        # 新客户端带有原来客户端的参数
        stream_client = with_options(timeout=request_timeout)
        stream_method = stream_client.messages.stream

    worker = _StreamWorker(stream_method, stream_kwargs, on_text, _on_retry)
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

    返回 (blocks, is_placeholder)：is_placeholder=True 表示模型没来得及输出任何文本，
    blocks 是构造的兜底占位文本；UI 打印时需明确标注，避免被误以为是模型说的。
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

    占位文本（pause 后无模型输出）会被显式标注，避免被误以为是模型说的。
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


def _request_summary(messages, *, stream_output: bool) -> None:
    """迭代预算耗尽后，去掉 tools 再请求一次纯文本总结。"""
    label = f"iteration budget exhausted ({MAX_ITERATIONS}/{MAX_ITERATIONS})"
    # 真正的 api_call_count 由调用方在调用 _request_summary 之前已 ++
    print(f"\n\033[33m⚠️  {label} — requesting summary...\033[0m")
    res, _ = _stream_message_with_recovery(
        model=MODEL_ID,  # type: ignore
        messages=messages,
        system=(
            SYSTEM
            + "\nThe tool iteration budget is exhausted. Reply with a concise plain-text "
            "summary only. Do not call tools or emit tool-call markup."
        ),
        max_tokens=MAX_TOKENS_SUMMARY,
        timeout=API_TIMEOUT,
        # 打印流式输出
        on_text=(lambda text: print(text, end="", flush=True)) if stream_output else None,
    )
    messages.append({"role": "assistant", "content": res.content})


def agent_loop(messages, *, stream_output: bool = False):
    global _last_interrupt_time
    interrupt_controller.clear()
    _last_interrupt_time = 0.0

    # 每条用户请求开始时，重置重复调用计数、halt 锁存、文件修改失败记录。
    tool_guardrails.reset_for_turn()
    file_mutation_tracker.reset_for_turn()

    # 开始记录本次会话的运行轨迹（每个 query 一份独立 trace 文件，落盘 logs/）
    tr = reset_tracer()

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

    try:
        while api_call_count < MAX_ITERATIONS and not interrupt_controller.is_requested():
            api_call_count += 1
            tr.step_start(api_call_count, len(messages), True, query=_last_user_text(messages))
            res, stream_interrupted = _stream_message_with_recovery(
                model=MODEL_ID,
                messages=messages,
                system=SYSTEM,
                max_tokens=MAX_TOKENS_TOOL,
                tools=registry.definitions(),
                timeout=API_TIMEOUT,
                on_text=(lambda text: print(text, end="", flush=True)) if stream_output else None,
            )

            tr.step_done(
                res.stop_reason,
                # 模型要调用的所有工具名称
                [b.name for b in res.content if isinstance(b, ToolUseBlock)],
                # 模型回复的 text（str）
                assistant_text=_assistant_text(res),
            )

            if stream_interrupted:
                paused_content, is_placeholder = _paused_text_blocks(res.content)
                messages.append({"role": "assistant", "content": paused_content})
                if stream_output and not res.content:
                    text = paused_content[0].text
                    if is_placeholder:
                        print(f"\033[90m【暂停占位】{text}\033[0m")
                    else:
                        print(text)
                outcome = TurnOutcome.INTERRUPTED
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

            if res.stop_reason != "tool_use":
                outcome = TurnOutcome.COMPLETED
                break
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
                outcome = TurnOutcome.GUARDRAIL_HALT
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
            _request_summary(messages, stream_output=stream_output)

        notice = _append_file_mutation_notice(messages)
        if stream_output and notice:
            print(notice)
        if outcome is TurnOutcome.ITERATION_BUDGET_EXHAUSTED:
            tr.finish(
                f"iteration budget exhausted ({api_call_count}/{MAX_ITERATIONS})",
                api_call_count,
            )
        else:
            tr.finish(outcome.value, api_call_count)

    except ProviderRequestInterrupted:
        get_tracer().finish("provider_retry_interrupted", api_call_count)
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
        get_tracer().finish(f"provider_error:{error.kind}", api_call_count)
        _emit_assistant_message(messages, message, stream_output=stream_output)
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        interrupt_controller.clear()
        _last_interrupt_time = 0.0


if __name__ == "__main__":
    # 历史消息列表
    history = []
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

            # 将用户 query 追加到历史消息列表中
            history.append({"role": "user", "content": query})

            agent_loop(history, stream_output=True)
        except EOFError:
            print("\n输入结束，退出交互")
            break
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，退出交互")
            break
