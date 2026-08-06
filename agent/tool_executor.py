"""工具执行与结果归并。

handler 可以并发运行，但所有共享状态都由主线程按模型原始调用顺序归并：
结果分类、guardrail、文件修改核验、trace 和对话消息因此保持确定性。
"""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any, Callable, List

from agent.file_mutation_tracker import FileMutationTracker
from agent.config_loader import load_config
from agent.tool_guardrails import (
    IDEMPOTENT_TOOL_NAMES,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    append_toolguard_guidance,
    toolguard_synthetic_result,
)
from agent.tool_result_classification import (
    BLOCKED,
    CANCELLED,
    INTERNAL_ERROR,
    UNKNOWN,
    ToolOutcome,
    classify_tool_result,
)
from agent.tracer import get_tracer
from tools._permission import check_tool_permission
from tools.registry import registry

TURN_BUDGET_CHARS = 300_000

# 每条用户请求开始时由 conversation_loop 重置状态；配置在进程启动时读取一次。
_project_config = load_config()
tool_guardrails = ToolCallGuardrailController(
    ToolCallGuardrailConfig.from_mapping(
        _project_config.get("tool_loop_guardrails")
    ).with_environment_overrides()
)
file_mutation_tracker = FileMutationTracker()

_TOOL_ERROR_ROLE_TAG_RE = re.compile(
    r"</?(?:tool_call|function_call|result|response|output|input|system|assistant|user)>",
    re.IGNORECASE,
)
_TOOL_ERROR_FENCE_RE = re.compile(r"```(?:json|xml|html|markdown)?", re.IGNORECASE)
_TOOL_ERROR_CDATA_RE = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)
_TOOL_ERROR_MAX_LEN = 2000
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class RawToolExecution:
    """worker 返回的无状态原始结果，尚未进行 ToolOutcome 分类，后续有分类器根据 content 分类"""

    content: Any
    status: str | None = None


def _sanitize_tool_error(error_message: str) -> str:
    """清除异常中的结构标记并限制长度，避免错误文本干扰模型消息结构。"""
    sanitized = _TOOL_ERROR_ROLE_TAG_RE.sub("", error_message)
    sanitized = _TOOL_ERROR_FENCE_RE.sub("", sanitized)
    sanitized = _TOOL_ERROR_CDATA_RE.sub("", sanitized)
    if len(sanitized) > _TOOL_ERROR_MAX_LEN:
        sanitized = sanitized[:_TOOL_ERROR_MAX_LEN - 3] + "..."
    return f"[TOOL_ERROR] {sanitized}"


def _schema_allows_null(schema: dict) -> bool:
    """JSON Schema 字段是否显式允许 null。"""
    schema_type = schema.get("type")
    if schema_type == "null" or schema.get("nullable") is True:
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and any(
            isinstance(option, dict) and option.get("type") == "null"
            for option in options
        ):
            return True
    return False


def _coerce_tool_value(value: str, expected_type: Any, schema: dict) -> Any:
    """按单个字段的 schema 安全转换字符串；无法确定时保留原值。"""
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        for candidate in expected_type:
            converted = _coerce_tool_value(value, candidate, schema)
            if converted is not value:
                return converted
        return value

    if expected_type in {"integer", "number"}:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return value
        if not math.isfinite(number):
            return value
        if number.is_integer():
            return int(number)
        return value if expected_type == "integer" else number

    if expected_type == "boolean":
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value

    expected_python_type = {"array": list, "object": dict}.get(expected_type)
    if expected_python_type is not None:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value
        return parsed if isinstance(parsed, expected_python_type) else value

    return value


def coerce_tool_args(tool_name: str, args: dict) -> dict:
    """根据注册工具的 input_schema 安全转换参数类型，不修改原始参数。"""
    converted_args = dict(args)
    entry = registry.get_entry(tool_name)
    if entry is None:
        return converted_args
    input_schema = entry.schema.get("input_schema")
    if not isinstance(input_schema, dict):
        return converted_args
    properties = input_schema.get("properties", {})
    if not isinstance(properties, dict):
        return converted_args

    for key, value in list(converted_args.items()):
        schema = properties.get(key)
        if not isinstance(schema, dict):
            continue
        expected_type = schema.get("type")
        if expected_type == "array" and value is not None and not isinstance(value, (list, tuple)):
            if isinstance(value, str):
                converted = _coerce_tool_value(value, expected_type, schema)
                converted_args[key] = [value] if converted is value else converted
            else:
                converted_args[key] = [value]
            continue
        if isinstance(value, str):
            converted_args[key] = _coerce_tool_value(value, expected_type, schema)
    return converted_args


def _execute_raw_tool_call(name: str, args: dict) -> RawToolExecution:
    """只执行 handler，不分类结果，不访问任何共享状态。"""
    entry = registry.get_entry(name)
    if entry is None:
        return RawToolExecution(f"Unknown tool: {name}", UNKNOWN)
    try:
        output = entry.handler(**args)
    except Exception as exc:
        logger.exception("Tool %s execution failed", name)
        error = _sanitize_tool_error(
            f"Tool execution failed: {type(exc).__name__}: {exc}"
        )
        return RawToolExecution(
            json.dumps({"error": error}, ensure_ascii=False),
            INTERNAL_ERROR,
        )
    return RawToolExecution(output)


def _classify_raw_execution(name: str, raw: RawToolExecution) -> ToolOutcome:
    """主线程中将 worker 原始结果转换为统一 ToolOutcome。"""
    return classify_tool_result(name, raw.content, status=raw.status)


def dispatch_tool_call(name: str, args: dict) -> str:
    """兼容旧调用方的单工具执行入口。

    与批量路径使用同一契约：记录调用、执行 preflight、更新
    guardrail / 文件状态 / trace，防止兼容调用方绕过权限边界。
    """
    args = coerce_tool_args(name, args)
    get_tracer().tool_call(name, args)
    outcome = _preflight_call(name, args)
    if outcome is None:
        outcome = _classify_raw_execution(name, _execute_raw_tool_call(name, args))
    return _finalize_outcome(outcome, args).content


def _blocked_outcome(name: str, content: str, code: str) -> ToolOutcome:
    """阻塞结果合成"""
    return ToolOutcome(name, content, BLOCKED, code, content)


def _cancelled_outcome(name: str, reason: str) -> ToolOutcome:
    """为未执行的工具调用生成协议完整的取消结果。"""
    return classify_tool_result(
        name,
        f"[Tool execution cancelled: {reason}]",
        status=CANCELLED,
    )


def _preflight_call(name: str, args: dict) -> ToolOutcome | None:
    """调用前预检。

    1. 工具护栏 before_call
    2. 是否是未注册工具
    3. tool 权限检查

    Args:
        name (str):工具名称
        args (dict):工具调用参数

    Returns:
        ToolOutcome
        None 允许调用
    """
    decision = tool_guardrails.before_call(name, args)
    if not decision.allows_execution:
        content = toolguard_synthetic_result(decision)
        print(f"\033[33m⚠️  guardrail block: {decision.message}\033[0m")
        return _blocked_outcome(name, content, decision.code)

    # conversation loop 会按模型轮次处理未知工具；这里保留兜底，确保兼容入口
    # 或注册表变化时仍不执行 handler，并返回可恢复的明确错误。
    if registry.get_entry(name) is None:
        available_tools = ", ".join(registry.names()) or "(none)"
        return classify_tool_result(
            name,
            f"Unknown tool: {name}. Available tools: {available_tools}",
            status=UNKNOWN,
        )

    blocked_message = check_tool_permission(name, args)
    if blocked_message is not None:
        # 策略拒绝没有执行 handler，不可污染失败计数。
        return _blocked_outcome(name, blocked_message, "permission_denied")
    return None


def _finalize_outcome(outcome: ToolOutcome, args: dict) -> ToolOutcome:
    """最终确定结果：主线程顺序归并一个结果，更新所有 per-turn 共享状态。"""
    if outcome.status not in {BLOCKED, CANCELLED, UNKNOWN}:
        decision = tool_guardrails.after_call(
            outcome.tool_name,
            args,
            outcome.content,
            failed=outcome.counts_as_failure,
        )
    else:
        decision = None

    if decision is not None and decision.action in {"warn", "halt"}:
        # 基于原对象创建一个新对象，只替换指定字段值
        outcome = replace(
            outcome,
            content=append_toolguard_guidance(outcome.content, decision),
        )

    file_mutation_tracker.record(outcome, args)
    get_tracer().tool_result(outcome.tool_name, outcome.content, outcome=outcome)
    return outcome


def _enforce_turn_budget(results: List[dict]) -> None:
    """严格限制本轮返回给模型的工具结果总字符数。"""
    total = 0
    marker = "\n... [工具结果已按单轮预算截断]"
    for result in results:
        content = str(result.get("content", ""))
        remaining = max(0, TURN_BUDGET_CHARS - total)
        if len(content) > remaining:
            content = marker[:remaining] if remaining <= len(marker) else (
                content[:remaining - len(marker)] + marker
            )
            result["content"] = content
        total += len(content)


def _should_parallelize_tool_batch(calls: list[Any]) -> bool:
    """这一批工具调用是否应该并行。

    只有同时满足以下条件时才返回 True：

    1. 至少包含两个工具调用；
    2. 每个工具名称都不重复；
    3. 所有工具都属于只读工具白名单。

    工具名称重复时，即使工具本身是只读的，也必须顺序执行，
    以便后一次调用能够看到前一次调用更新的护栏状态。

    Args:
        calls: 模型本轮返回的 tool_use 调用列表。

    Returns:
        True 表示这批调用可以并发执行；
        False 表示必须按顺序执行。
    """
    names = [call.name for call in calls]
    return (
        len(calls) > 1
        and len(names) == len(set(names))
        and all(name in IDEMPOTENT_TOOL_NAMES for name in names)
    )


def execute_tool_calls(
    blocks: List[Any],
    messages: List[Any],
    concurrent: bool = False,
    max_workers: int = 4,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """执行一批 tool_use，并按模型原顺序归并结果后写回 messages。"""

    calls = [block for block in blocks if getattr(block, "type", None) == "tool_use"]
    if not calls:
        return

    for call in calls:
        print(f"use_tool:\033[33m{call.name}\033[0m\n")

    prepared_args = [coerce_tool_args(call.name, call.input) for call in calls]

    # 顺序执行路径必须逐个 preflight（预检） → 执行 → 归并，才能让第 N 次重复调用在
    # 同一批次内也看到前 N-1 次的失败计数。
    if not concurrent or not _should_parallelize_tool_batch(calls):
        outcomes: list[ToolOutcome] = []
        for call, args in zip(calls, prepared_args):
            get_tracer().tool_call(call.name, args)
            if is_cancelled():
                outcome = _cancelled_outcome(
                    call.name,
                    "not started due to user interrupt",
                )
            else:
                outcome = _preflight_call(call.name, args)
                if outcome is None:
                    outcome = _classify_raw_execution(
                        call.name,
                        _execute_raw_tool_call(call.name, args),
                    )
            outcomes.append(_finalize_outcome(outcome, args))
    else:
        # 并发执行路径在启动 worker 前完成所有 preflight；worker 仅执行 handler。
        raw_outcomes: list[RawToolExecution | ToolOutcome | None] = [None] * len(calls)
        executable: list[tuple[int, Any, dict]] = []
        for index, (call, args) in enumerate(zip(calls, prepared_args)):
            get_tracer().tool_call(call.name, args)
            if is_cancelled():
                raw_outcomes[index] = _cancelled_outcome(
                    call.name,
                    "not started due to user interrupt",
                )
                continue
            preflight = _preflight_call(call.name, args)
            if preflight is None:
                executable.append((index, call, args))
            else:
                raw_outcomes[index] = preflight

        if executable:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(executable))) as executor:
                futures = {
                    executor.submit(_execute_raw_tool_call, call.name, args): index
                    for index, call, args in executable
                }
                pending = set(futures)
                while pending:
                    if is_cancelled():
                        for future in list(pending):
                            if future.cancel():
                                index = futures[future]
                                raw_outcomes[index] = _cancelled_outcome(
                                    calls[index].name,
                                    "not started due to user interrupt",
                                )
                                pending.remove(future)
                    if not pending:
                        break
                    completed, pending = wait(
                        pending,
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        raw_outcomes[futures[future]] = future.result()

        # 所有 stateful 操作都在主线程、模型原始调用顺序中完成。
        outcomes = []
        for call, args, raw_or_outcome in zip(calls, prepared_args, raw_outcomes):
            # worker 已隔离 handler 异常；这里仍保留兜底，确保每个 tool_use
            # 必有一个对应 tool_result，不会破坏 API 协议。
            if raw_or_outcome is None:
                outcome = classify_tool_result(
                    call.name,
                    "Error: worker did not return a result",
                    status=INTERNAL_ERROR,
                )
            elif isinstance(raw_or_outcome, RawToolExecution):
                outcome = _classify_raw_execution(call.name, raw_or_outcome)
            else:
                outcome = raw_or_outcome
            outcomes.append(_finalize_outcome(outcome, args))
    results = [
        {"type": "tool_result", "tool_use_id": call.id, "content": outcome.content}
        for call, outcome in zip(calls, outcomes)
    ]
    _enforce_turn_budget(results)
    messages.append({"role": "user", "content": results})
