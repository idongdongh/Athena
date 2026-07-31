"""工具执行引擎（不在 tools/ 内，避免被 discover() 扫描误当作工具加载）。

两层职责（对齐 hermes，但合进一个文件——本工程只有 loop 一个调用面）：
- dispatch_tool_call(name, args)：派发「单个」工具调用，对应 hermes handle_function_call 核心。
- run_tool_calls(blocks, messages, ...)：编排「一个 turn 模型返回的一批」tool_use 块，
  对应 hermes execute_tool_calls_sequential / _concurrent 核心。

保留的核心能力（区别于「别复杂」砍掉的厚壳）：
- 失败隔离（handler 异常不崩 loop）
- 并发执行（concurrent=True：ThreadPoolExecutor 并行派发多个独立工具调用）
- 单轮预算（TURN_BUDGET_CHARS：限制本轮工具结果总大小，溢出截断，防上下文被撑爆）

砍掉的厚壳：中间件、pre/post hook、显示 spinner、tool_search 桥接、ACP 编辑审批、
类型 coerce、取消后重写、按上下文窗口动态缩放预算。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from tools._permission import check_tool_permission
from tools.registry import registry
from agent.tracer import get_tracer

# 单轮工具结果总字符上限（对应 hermes enforce_turn_budget）。
# hermes 按上下文窗口动态缩放；本工程用固定值（约 75K token），可按需调大。
TURN_BUDGET_CHARS = 300_000


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
    # 权限闸门：被拦（返回非 None）则不执行 handler，直接把拒绝消息作为结果返回给模型。
    # 被 OS 5 系统调用级别的 pre-call hook，统一接缝在 tools/_permission.py。
    gated = check_tool_permission(name, args)
    if gated is not None:
        # 记录未放行原因
        get_tracer().tool_result(name, gated)
        return gated
    try:
        output = entry.handler(**args)
    except Exception as e:
        return f"Error: {e}"
    get_tracer().tool_result(name, str(output))
    return str(output)


def _enforce_turn_budget(results: List[dict]) -> None:
    """单轮预算：累加 tool_result 字符数，超上限则截断溢出部分。

    对应 hermes enforce_turn_budget 的核心——限制本轮返回给模型的工具结果总量，
    防止失控循环产出大量大结果把上下文撑爆。原地修改 results 中超标的 content。
    """
    total = 0
    for r in results:
        size = len(r["content"])
        if total + size > TURN_BUDGET_CHARS:
            remaining = max(0, TURN_BUDGET_CHARS - total)
            if remaining == 0:
                r["content"] = (
                    f"[工具结果已被截断：单轮预算 {TURN_BUDGET_CHARS} 字符已用尽]"
                )
            else:
                r["content"] = (
                    r["content"][:remaining]
                    + f"\n... [已按单轮预算截断至 {remaining} 字符]"
                )
            size = len(r["content"])
        total += size


def run_tool_calls(
    blocks: List[Any],
    messages: List[Any],
    concurrent: bool = False,
    max_workers: int = 4,
) -> None:
    """编排一个 turn 模型返回的一批 tool_use 块，结果回写 messages。

    对应 hermes execute_tool_calls_sequential（concurrent=False）/
    execute_tool_calls_concurrent（concurrent=True）的核心：
    遍历模型本轮返回的 content，挑出 tool_use 块派发（顺序或并发），
    收集 tool_result，经单轮预算校验后作为一条 user 消息 append 回对话。

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

    # 检查权限并执行工具
    if concurrent and len(calls) > 1:
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
