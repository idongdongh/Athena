"""工具执行引擎（不在 tools/ 内，避免被 discover() 扫描误当作工具加载）。

两层职责（对齐 hermes，但合进一个文件——本工程只有 loop 一个调用面）：
- dispatch_tool_call(name, args)：执行单个工具调用，对应 hermes handle_function_call 核心。
- run_tool_calls(blocks, messages, ...)：执行模型这一轮返回的一批 tool_use 调用，
  对应 hermes execute_tool_calls_sequential / _concurrent 核心。

保留的核心能力（区别于「别复杂」砍掉的厚壳）：
- 失败隔离（handler 异常不崩 loop）
- 安全并发（只并发工具名互不重复的只读调用）
- 单轮预算（TURN_BUDGET_CHARS：限制本轮工具结果总大小，溢出截断，防上下文被撑爆）

砍掉的厚壳：中间件、pre/post hook、显示 spinner、tool_search 桥接、ACP 编辑审批、
类型 coerce、取消后重写、按上下文窗口动态缩放预算。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from agent.tool_guardrails import (
    IDEMPOTENT_TOOL_NAMES,
    ToolCallGuardrailController,
    append_guardrail_guidance,
    is_tool_failure,
    toolguard_synthetic_result,
)
from tools._permission import check_tool_permission
from tools.registry import registry
from agent.tracer import get_tracer

# 单轮工具结果总字符上限（对应 hermes enforce_turn_budget）。
# hermes 按上下文窗口动态缩放；本工程用固定值（约 75K token），可按需调大。
TURN_BUDGET_CHARS = 300_000

# 工具调用重复检测控制器（per-turn 状态机）。
# 模块级单例 —— 本项目单 loop 单进程，对应 hermes ``agent._tool_guardrails``。
# ``conversation_loop.agent_loop`` 在每个 turn 开头调用 ``reset_for_turn``。
tool_guardrails = ToolCallGuardrailController()


def dispatch_tool_call(name: str, args: dict) -> str:
    """派发单个工具调用，对应 hermes handle_function_call 核心。

    拿到 (name, args) 交给注册表里正确的 handler，并做失败隔离：handler 异常
    （如模型传错参数名）包成错误文本返回给模型，而不是让 loop 崩溃。

    Args:
        name: 工具名（如 "read_file"）。
        args: 模型传入的参数 dict。

    Returns:
        工具执行结果字符串（成功结果或 "Error: ..." / "Unknown tool: ..."）。
    """
    get_tracer().tool_call(name, args)
    entry = registry.get_entry(name)
    if entry is None:
        return f"Unknown tool: {name}"
    # 重复检测 before_call：判精确失败重复 / 只读无进展是否达 block 阈值。
    # 放在权限闸门之前——更便宜/无 IO 的检查先做。
    guardrail_decision = tool_guardrails.before_call(name, args)
    if not guardrail_decision.allows_execution:
        # block：返回合成 result（不执行 handler），同时 _halt_decision 已在控制器内 latch
        synthetic = toolguard_synthetic_result(guardrail_decision)
        get_tracer().tool_result(name, synthetic)
        print(f"\033[33m⚠️  guardrail block: {guardrail_decision.message}\033[0m")
        return synthetic
    # 权限闸门：被拦（返回非 None）则不执行 handler，直接把拒绝消息作为结果返回给模型。
    # 命令和路径权限检查统一接在 tools/_permission.py。
    gated = check_tool_permission(name, args)
    if gated is not None:
        # 记录未放行原因
        get_tracer().tool_result(name, gated)
        # 权限拒绝也算是「失败」——计入控制器，让连续被拒的调用能被重复检测到
        post_decision = tool_guardrails.after_call(name, args, gated, failed=True)
        if post_decision.action in {"warn", "halt"}:
            gated = append_guardrail_guidance(gated, post_decision)
        return gated
    try:
        output = entry.handler(**args)
    except Exception as e:
        output = f"Error: {e}"
    # 重复检测 after_call：分类失败 + 更新计数 + 可能追加引导/halt
    output_str = str(output)
    is_error = is_tool_failure(name, output_str)
    post_decision = tool_guardrails.after_call(name, args, output_str, failed=is_error)
    if post_decision.action in {"warn", "halt"}:
        output_str = append_guardrail_guidance(output_str, post_decision)
    get_tracer().tool_result(name, output_str)
    return output_str


def _enforce_turn_budget(results: List[dict]) -> None:
    """单轮预算：累加 tool_result 字符数，超上限则截断溢出部分。

    对应 hermes enforce_turn_budget 的核心——限制本轮返回给模型的工具结果总量，
    防止失控循环产出大量大结果把上下文撑爆。原地修改 results 中超标的 content。
    """
    total = 0
    marker = "\n... [工具结果已按单轮预算截断]"
    for r in results:
        content = str(r.get("content", ""))
        remaining = max(0, TURN_BUDGET_CHARS - total)
        if len(content) > remaining:
            if remaining <= len(marker):
                content = marker[:remaining]
            else:
                content = content[:remaining - len(marker)] + marker
            r["content"] = content
        total += len(content)


def run_tool_calls(
    blocks: List[Any],
    messages: List[Any],
    concurrent: bool = False,
    max_workers: int = 4,
) -> None:
    """执行模型这一轮返回的一批 tool_use 调用，把工具结果统一回写 messages。

    具体做 4 件事：
    1. 从 blocks 中挑出 tool_use 块（跳过 text 块）
    2. 并发或顺序执行 dispatch_tool_call（并发时保证结果按原顺序回填）
    3. 单轮工具结果总和超 30 万字符则截断（防撑爆上下文窗口）
    4. 把所有 tool_result 打包成一条 user 消息 append 到 messages
       （API 要求：一轮 assistant 的多个 tool_use 必须用一条 user 消息回复）

    Args:
        blocks: assistant 消息的 content 列表（含 text / tool_use 块）。
        messages: 对话历史（原地修改：追加一条 role=user 的工具结果消息）。
        concurrent: 是否并发派发多个独立工具调用（默认 False，顺序执行）。
        max_workers: 并发时线程池上限。
    """
    # 所有的 ToolUseBlock 对象
    calls = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
    if not calls:
        return

    # 主线程按原顺序打印 use_tool，避免并发时输出穿插
    for b in calls:
        print(f"use_tool:\033[33m{b.name}\033[0m\n")

    # 当前 CLI 没有 Hermes 的跨线程审批回调和完整状态传播。
    # 只并发执行「工具名互不重复的只读调用」；bash、写工具和重复调用保持顺序，
    # 避免多个线程争抢 input()，也保证重复计数在下一次调用前已经落地。
    names = [b.name for b in calls]
    parallel_safe = (
        concurrent
        and len(calls) > 1
        and len(names) == len(set(names))
        and all(name in IDEMPOTENT_TOOL_NAMES for name in names)
    )
    if parallel_safe:
        # 结果是：[None, None]
        outs: List[Any] = [None] * len(calls)
        ex = ThreadPoolExecutor(max_workers=min(max_workers, len(calls)))
        try:
            # 建立任务对象和 id 的映射表
            fut_to_idx = {
                ex.submit(dispatch_tool_call, b.name, b.input): i
                for i, b in enumerate(calls)
            }
            for fut in as_completed(fut_to_idx):
                outs[fut_to_idx[fut]] = fut.result()
        finally:
            # 正常完成：所有 future 已结束，cancel 为空操作；
            # 第二次 Ctrl+C（SIG_DFL 抛 KeyboardInterrupt）：取消未开始任务后异常自动传播
            # 关闭线程池，不等待当前正在执行的任务完成（wait=False），并且取消所有尚未开始的任务（cancel_futures=True）
            ex.shutdown(wait=False, cancel_futures=True)
    else:
        outs = [dispatch_tool_call(b.name, b.input) for b in calls]

    results = [
        {"type": "tool_result", "tool_use_id": b.id, "content": out}
        for b, out in zip(calls, outs)
    ]
    _enforce_turn_budget(results)
    messages.append({"role": "user", "content": results})
