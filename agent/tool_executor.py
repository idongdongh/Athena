"""工具执行与结果归并。

handler 可以并发运行，但所有共享状态都由主线程按模型原始调用顺序归并：
结果分类、guardrail、文件修改核验和对话消息因此保持确定性。
"""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Callable, List

from agent.file_mutation_tracker import FileMutationTracker
from agent.interrupt_controller import ToolExecutionCancelled
from agent.tool_guardrails import (
    IDEMPOTENT_TOOL_NAMES,
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
from agent.trace_events import emit_trace
from tools._permission import check_tool_permission
from tools.registry import registry

# 单次工具调用批次返回给模型的所有工具结果，最多允许占用 300,000 个字符
TURN_BUDGET_CHARS = 300_000

# 每条用户请求开始时由 conversation_loop 重置状态；入口可注入 YAML 配置。
tool_guardrails = ToolCallGuardrailController()
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


def _union_schema_types(schema: dict) -> list[str]:
    """提取 anyOf/oneOf 中声明的简单 JSON Schema 类型。"""
    candidates: list[str] = []
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if not isinstance(options, list):
            continue
        for option in options:
            if not isinstance(option, dict):
                continue
            option_type = option.get("type")
            if isinstance(option_type, str):
                candidates.append(option_type)
            elif isinstance(option_type, list):
                candidates.extend(item for item in option_type if isinstance(item, str))
    return list(dict.fromkeys(candidates))


def _coerce_tool_value(value: str, expected_type: Any, schema: dict) -> Any:
    """按单个字段的 schema 安全转换字符串；无法确定时保留原值。"""
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    union_types: list[str] = []
    if isinstance(expected_type, list):
        union_types = [item for item in expected_type if isinstance(item, str)]
    elif expected_type is None:
        union_types = _union_schema_types(schema)

    if union_types:
        # 字符串原值已经满足联合类型中的 string，不应再被机会式地转换成数字、
        # 布尔值等其他合法类型。显式字符串 "null" 的转换已在上方单独处理。
        if "string" in union_types:
            return value
        for candidate in union_types:
            if candidate == "null":
                continue
            # 传空 schema，避免递归时再次展开同一个 anyOf/oneOf。
            converted = _coerce_tool_value(value, candidate, {})
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

    expected_python_type = (
        {"array": list, "object": dict}.get(expected_type)
        if isinstance(expected_type, str)
        else None
    )
    if expected_python_type is not None:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value
        return parsed if isinstance(parsed, expected_python_type) else value

    return value


def coerce_tool_args(tool_name: str, args: dict) -> dict:
    """根据注册工具的 input_schema 安全转换参数类型，不修改原始参数。"""
    if not isinstance(args, Mapping):
        raise TypeError("tool input must be a JSON object")
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

    required = input_schema.get("required", [])
    if isinstance(required, list):
        missing = [key for key in required if key not in converted_args]
        if missing:
            raise ValueError(f"missing required argument(s): {', '.join(missing)}")

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
            if not _value_matches_schema_type(converted_args[key], schema):
                raise TypeError(
                    f"argument '{key}' has invalid type "
                    f"{type(converted_args[key]).__name__}; expected array"
                )
            continue
        if isinstance(value, str):
            converted_args[key] = _coerce_tool_value(value, expected_type, schema)

        value = converted_args[key]
        if not _value_matches_schema_type(value, schema):
            raise TypeError(
                f"argument '{key}' has invalid type {type(value).__name__}; "
                f"expected {schema.get('type') or 'declared schema'}"
            )
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            raise ValueError(f"argument '{key}' must be one of {enum}")
    return converted_args


def _value_matches_schema_type(value: Any, schema: dict) -> bool:
    """值是否满足工具 schema 声明的基础 JSON 类型。"""
    if value is None:
        return _schema_allows_null(schema)

    expected = schema.get("type")
    candidates = expected if isinstance(expected, list) else [expected]
    if expected is None:
        candidates = _union_schema_types(schema)
    if not candidates:
        return True

    python_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": (list, tuple),
        "object": Mapping,
        "null": type(None),
    }
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        expected_python = python_types.get(candidate)
        if expected_python is None:
            continue
        if candidate in {"integer", "number"} and isinstance(value, bool):
            continue
        if isinstance(value, expected_python):
            return True
    return False


def _execute_raw_tool_call(
    name: str,
    args: dict,
    memory_store=None,
    session_db=None,
    current_session_id: str | None = None,
) -> RawToolExecution:
    """只执行 handler，不分类结果，不访问任何共享状态。"""
    entry = registry.get_entry(name)
    if entry is None:
        return RawToolExecution(f"Unknown tool: {name}", UNKNOWN)
    try:
        if name == "memory":
            output = entry.handler(**args, store=memory_store)
        elif name == "session_search":
            output = entry.handler(
                **args,
                db=session_db,
                current_session_id=current_session_id,
            )
        else:
            output = entry.handler(**args)
    except ToolExecutionCancelled as exc:
        return RawToolExecution(
            f"[Tool execution cancelled: {exc}]",
            CANCELLED,
        )
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

    与批量路径使用同一契约：执行 preflight、更新 guardrail / 文件状态，
    防止兼容调用方绕过权限边界。
    """
    try:
        args = coerce_tool_args(name, args)
    except (TypeError, ValueError, OverflowError) as exc:
        outcome = _invalid_arguments_outcome(name, exc)
        return _finalize_outcome(outcome, {}).content
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


def _invalid_arguments_outcome(name: str, exc: Exception) -> ToolOutcome:
    """把模型生成的非法参数转换为可恢复结果，不让异常逃出工具循环。"""
    message = f"Invalid arguments for {name}: {_sanitize_tool_error(str(exc))}"
    return ToolOutcome(
        name,
        message,
        INTERNAL_ERROR,
        "invalid_arguments",
        message,
    )


def _preflight_error_outcome(name: str, exc: Exception) -> ToolOutcome:
    """权限或护栏预检自身异常时返回稳定的内部错误。"""
    message = f"Tool preflight failed for {name}: {_sanitize_tool_error(str(exc))}"
    return ToolOutcome(
        name,
        message,
        INTERNAL_ERROR,
        "preflight_exception",
        message,
    )


def _preflight_call(
    name: str,
    args: dict,
    guardrails: ToolCallGuardrailController | None = None,
) -> ToolOutcome | None:
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
    try:
        controller = guardrails or tool_guardrails
        decision = controller.before_call(name, args)
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
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        return _invalid_arguments_outcome(name, exc)
    except Exception as exc:
        logger.exception("Tool %s preflight failed", name)
        return _preflight_error_outcome(name, exc)


def _finalize_outcome(
    outcome: ToolOutcome,
    args: dict,
    guardrails: ToolCallGuardrailController | None = None,
    mutation_tracker: FileMutationTracker | None = None,
) -> ToolOutcome:
    """最终确定结果：主线程顺序归并一个结果，更新所有 per-turn 共享状态。"""
    if outcome.status not in {BLOCKED, CANCELLED, UNKNOWN}:
        controller = guardrails or tool_guardrails
        decision = controller.after_call(
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

    tracker = mutation_tracker or file_mutation_tracker
    tracker.record(outcome, args)
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


def _prepare_args(calls) -> tuple[list[dict], list[ToolOutcome | None]]:
    """参数校验 + coerce；返回 (prepared_args, preparation_errors)。

    与原 ``execute_tool_calls`` 顶部的 prepare 循环等价；抽出来便于顺序 / 并发路径共享。
    """
    prepared_args: list[dict] = []
    preparation_errors: list[ToolOutcome | None] = []
    for call in calls:
        try:
            prepared_args.append(coerce_tool_args(call.name, call.input))
            preparation_errors.append(None)
        except (TypeError, ValueError, OverflowError) as exc:
            prepared_args.append({})
            preparation_errors.append(_invalid_arguments_outcome(call.name, exc))
    return prepared_args, preparation_errors


def _run_one_or_cancelled(
    call, args, preparation_error, is_cancelled, guardrails=None, memory_store=None,
    session_db=None, current_session_id=None,
) -> ToolOutcome:
    """单步顺序执行：preflight → execute → 分类（不 finalize，由调用方统一做）。

    只有顺序路径使用这条流水线——并发路径要求 preflight 全部先完成、handler 并发跑。
    """
    if is_cancelled():
        return _cancelled_outcome(call.name, "not started due to user interrupt")
    if preparation_error is not None:
        return preparation_error
    preflight = _preflight_call(call.name, args, guardrails)
    if preflight is None:
        return _classify_raw_execution(
            call.name, _execute_raw_tool_call(
                call.name, args, memory_store, session_db, current_session_id
            )
        )
    return preflight


def _run_sequential(
    calls,
    prepared_args: list[dict],
    preparation_errors: list[ToolOutcome | None],
    is_cancelled: Callable[[], bool],
    guardrails=None,
    mutation_tracker=None,
    memory_store=None,
    session_db=None,
    current_session_id=None,
) -> list[ToolOutcome]:
    """逐个 preflight → execute → finalize。

    顺序执行必须让第 N 次重复调用在同批次内也看到前 N-1 次的失败计数，
    guardrail 状态机依赖这一点。
    """
    outcomes: list[ToolOutcome] = []
    for call, args, prep_err in zip(calls, prepared_args, preparation_errors):
        outcome = _run_one_or_cancelled(
            call, args, prep_err, is_cancelled, guardrails, memory_store,
            session_db, current_session_id,
        )
        outcomes.append(_finalize_outcome(
            outcome, args, guardrails, mutation_tracker
        ))
    return outcomes


def _run_parallel(
    calls,
    prepared_args: list[dict],
    preparation_errors: list[ToolOutcome | None],
    is_cancelled: Callable[[], bool],
    max_workers: int,
    guardrails=None,
    mutation_tracker=None,
    memory_store=None,
    session_db=None,
    current_session_id=None,
) -> list[ToolOutcome]:
    """并发执行：批量 preflight 后用 ThreadPoolExecutor 跑 handler，最后主线程顺序 finalize。

    所有 stateful 操作（guardrail、file_mutation_tracker）都在主线程按模型原始
    调用顺序归并——worker 只跑 handler，不触碰共享状态。
    """
    raw_outcomes: list[RawToolExecution | ToolOutcome | None] = [None] * len(calls)
    executable: list[tuple[int, Any, dict]] = []
    for index, (call, args, prep_err) in enumerate(
        zip(calls, prepared_args, preparation_errors)
    ):
        if is_cancelled():
            raw_outcomes[index] = _cancelled_outcome(
                call.name, "not started due to user interrupt"
            )
            continue
        if prep_err is not None:
            raw_outcomes[index] = prep_err
            continue
        preflight = _preflight_call(call.name, args, guardrails)
        if preflight is None:
            executable.append((index, call, args))
        else:
            raw_outcomes[index] = preflight

    if executable:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(executable))) as executor:
            futures = {
                executor.submit(
                    _execute_raw_tool_call,
                    call.name,
                    args,
                    memory_store,
                    session_db,
                    current_session_id,
                ): index
                for index, call, args in executable
            }
            pending = set(futures)
            while pending:
                if is_cancelled():
                    for future in list(pending):
                        if future.cancel():
                            index = futures[future]
                            raw_outcomes[index] = _cancelled_outcome(
                                calls[index].name, "not started due to user interrupt"
                            )
                            pending.remove(future)
                if not pending:
                    break
                completed, pending = wait(
                    pending, timeout=0.1, return_when=FIRST_COMPLETED
                )
                for future in completed:
                    raw_outcomes[futures[future]] = future.result()

    # 主线程顺序归并：worker 已隔离 handler 异常；这里仍保留兜底，确保每个 tool_use
    # 必有一个对应 tool_result，不会破坏 API 协议。
    outcomes: list[ToolOutcome] = []
    for call, args, raw_or_outcome in zip(calls, prepared_args, raw_outcomes):
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
        outcomes.append(_finalize_outcome(
            outcome, args, guardrails, mutation_tracker
        ))
    return outcomes


def execute_tool_calls(
    blocks: List[Any],
    messages: List[Any],
    concurrent: bool = False,
    max_workers: int = 4,
    is_cancelled: Callable[[], bool] = lambda: False,
    guardrails: ToolCallGuardrailController | None = None,
    mutation_tracker: FileMutationTracker | None = None,
    memory_store=None,
    session_db=None,
    current_session_id: str | None = None,
    show_progress: bool = True,
    trace_sink=None,
    trace_context: Mapping[str, Any] | None = None,
) -> None:
    """执行一批 tool_use，并按模型原顺序归并结果后写回 messages。"""

    calls = [block for block in blocks if getattr(block, "type", None) == "tool_use"]
    if not calls:
        return

    context = dict(trace_context or {})
    emit_trace(
        trace_sink,
        "tool_batch_start",
        **context,
        tool_call_count=len(calls),
        concurrent_requested=concurrent,
    )
    for call in calls:
        emit_trace(
            trace_sink,
            "tool_call",
            **context,
            tool_call_id=getattr(call, "id", None),
            tool_name=getattr(call, "name", ""),
            tool_args=getattr(call, "input", {}),
        )

    if show_progress:
        for call in calls:
            print(f"use_tool:\033[33m{call.name}\033[0m\n")

    prepared_args, preparation_errors = _prepare_args(calls)

    # 顺序执行路径必须逐个 preflight → execute → finalize，让第 N 次重复调用在
    # 同一批次内也看到前 N-1 次的失败计数（guardrail 状态机依赖这一点）。
    # 并发路径只对"工具名互不重复 + 全是只读工具"的批次启用。
    parallel = concurrent and _should_parallelize_tool_batch(calls)
    if parallel:
        outcomes = _run_parallel(
            calls, prepared_args, preparation_errors, is_cancelled, max_workers,
            guardrails, mutation_tracker, memory_store, session_db, current_session_id,
        )
    else:
        outcomes = _run_sequential(
            calls, prepared_args, preparation_errors, is_cancelled,
            guardrails, mutation_tracker, memory_store, session_db, current_session_id,
        )

    results = [
        {"type": "tool_result", "tool_use_id": call.id, "content": outcome.content}
        for call, outcome in zip(calls, outcomes)
    ]
    for call, outcome in zip(calls, outcomes):
        emit_trace(
            trace_sink,
            "tool_result",
            **context,
            tool_call_id=getattr(call, "id", None),
            tool_name=outcome.tool_name,
            tool_args=getattr(call, "input", {}),
            status=outcome.status,
            error_code=outcome.error_code,
            error_message=outcome.error_message,
            exit_code=outcome.exit_code,
            output=outcome.content,
        )
    emit_trace(
        trace_sink,
        "tool_batch_end",
        **context,
        statuses=[outcome.status for outcome in outcomes],
        parallel=parallel,
    )
    _enforce_turn_budget(results)
    messages.append({"role": "user", "content": results})
