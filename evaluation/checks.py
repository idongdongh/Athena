"""不调用 LLM 的轨迹完整性、结果与过程 Checks。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.schema import CheckResult, TraceEvaluation


_VALIDATION_COMMAND_RE = re.compile(
    r"(?:^|[;&|\s])(?:pytest|py\.test|python\s+-m\s+unittest|npm\s+test|"
    r"pnpm\s+test|yarn\s+test|cargo\s+test|go\s+test|mvn\s+test|gradle\s+test)",
    re.IGNORECASE,
)


def _events_of(events: Sequence[Mapping[str, Any]], event_type: str) -> list[Mapping[str, Any]]:
    return [event for event in events if event.get("event_type") == event_type]


def _event_ids(events: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    return tuple(
        event_id for event in events
        if isinstance((event_id := event.get("event_id")), int)
    )


def _canonical_args(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _tool_signature(event: Mapping[str, Any]) -> tuple[str, str]:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    return str(payload.get("tool_name", "")), _canonical_args(payload.get("tool_args", {}))


def evaluate_trace(events: Sequence[Mapping[str, Any]]) -> TraceEvaluation:
    """对一条已加载轨迹执行确定性评价。"""
    if not events:
        return TraceEvaluation(
            trace_id="",
            valid=False,
            checks=(),
            metrics={},
            invalid_reasons=("empty_trace",),
        )
    trace_id = str(events[0].get("trace_id", ""))
    starts = _events_of(events, "turn_start")
    finishes = _events_of(events, "turn_finish")
    requests = _events_of(events, "model_request")
    responses = _events_of(events, "model_response")
    model_errors = _events_of(events, "model_error")
    calls = _events_of(events, "tool_call")
    results = _events_of(events, "tool_result")
    task_results = _events_of(events, "task_result")

    invalid_reasons: list[str] = []
    if len(starts) != 1:
        invalid_reasons.append("turn_start_count")
    if len(finishes) != 1:
        invalid_reasons.append("turn_finish_count")

    request_ids = {
        event.get("payload", {}).get("api_request_id")
        for event in requests
        if isinstance(event.get("payload"), Mapping)
    }
    response_ids = {
        event.get("payload", {}).get("api_request_id")
        for event in responses
        if isinstance(event.get("payload"), Mapping)
    }
    error_ids = {
        event.get("payload", {}).get("api_request_id")
        for event in model_errors
        if isinstance(event.get("payload"), Mapping)
    }
    if request_ids != response_ids | error_ids:
        invalid_reasons.append("model_request_response_mismatch")
    for event in model_errors:
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        kind = str(payload.get("kind") or payload.get("error_type") or "unknown")
        invalid_reasons.append(f"infrastructure_provider_error:{kind}")

    call_ids = [
        event.get("payload", {}).get("tool_call_id")
        for event in calls
        if isinstance(event.get("payload"), Mapping)
    ]
    result_ids = [
        event.get("payload", {}).get("tool_call_id")
        for event in results
        if isinstance(event.get("payload"), Mapping)
    ]
    if Counter(call_ids) != Counter(result_ids):
        invalid_reasons.append("tool_call_result_mismatch")

    finish_payload = (
        finishes[0].get("payload", {})
        if len(finishes) == 1 and isinstance(finishes[0].get("payload"), Mapping)
        else {}
    )
    completed = finish_payload.get("outcome") == "completed"
    unresolved = finish_payload.get("unresolved_file_mutations") or []
    task_result_payload = (
        task_results[-1].get("payload", {})
        if task_results and isinstance(task_results[-1].get("payload"), Mapping)
        else {}
    )
    resolved = task_result_payload.get("resolved")

    statuses = Counter(
        str(event.get("payload", {}).get("status", "unknown"))
        for event in results
        if isinstance(event.get("payload"), Mapping)
    )
    signature_counts = Counter(_tool_signature(event) for event in calls)
    redundant_calls = sum(max(0, count - 1) for count in signature_counts.values())

    validation_events: list[Mapping[str, Any]] = []
    mutation_events: list[Mapping[str, Any]] = []
    unique_successful_actions = 0
    seen_successful_signatures: set[tuple[str, str]] = set()
    for event in results:
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        signature = _tool_signature(event)
        tool_name = str(payload.get("tool_name", ""))
        if tool_name in {"write_file", "patch"}:
            mutation_events.append(event)
        if tool_name == "bash":
            args = payload.get("tool_args")
            command = args.get("command", "") if isinstance(args, Mapping) else ""
            if isinstance(command, str) and _VALIDATION_COMMAND_RE.search(command):
                validation_events.append(event)
        if payload.get("status") == "succeeded" and signature not in seen_successful_signatures:
            seen_successful_signatures.add(signature)
            unique_successful_actions += 1

    tool_total = len(calls)
    effective_action_ratio = unique_successful_actions / tool_total if tool_total else 1.0
    redundant_rate = redundant_calls / tool_total if tool_total else 0.0
    validation_passed = any(
        event.get("payload", {}).get("status") == "succeeded"
        for event in validation_events
        if isinstance(event.get("payload"), Mapping)
    )

    failed_by_tool: defaultdict[str, list[int]] = defaultdict(list)
    recovered_failures = 0
    for index, event in enumerate(results):
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        tool_name = str(payload.get("tool_name", ""))
        status = payload.get("status")
        if status in {"failed", "internal_error", "unknown"}:
            failed_by_tool[tool_name].append(index)
        elif status == "succeeded" and failed_by_tool[tool_name]:
            recovered_failures += len(failed_by_tool[tool_name])
            failed_by_tool[tool_name].clear()
    total_failures = sum(statuses[name] for name in ("failed", "internal_error", "unknown"))
    recovery_rate = recovered_failures / total_failures if total_failures else 1.0

    checks = (
        CheckResult(
            "trace_integrity",
            not invalid_reasons,
            1.0 if not invalid_reasons else 0.0,
            1.0,
            "轨迹关联完整" if not invalid_reasons else f"完整性异常: {', '.join(invalid_reasons)}",
            _event_ids(events),
        ),
        CheckResult(
            "turn_completed",
            completed,
            1.0 if completed else 0.0,
            1.0,
            f"turn outcome={finish_payload.get('outcome')}",
            _event_ids(finishes),
        ),
        CheckResult(
            "task_resolved",
            resolved if isinstance(resolved, bool) else None,
            1.0 if resolved is True else 0.0,
            1.0,
            (
                f"task resolved={resolved}"
                if isinstance(resolved, bool)
                else "本轨迹未包含外部任务验证结果"
            ),
            _event_ids(task_results),
        ),
        CheckResult(
            "no_unresolved_file_mutation",
            not unresolved,
            1.0 if not unresolved else 0.0,
            1.0,
            "无未恢复文件修改失败" if not unresolved else f"存在 {len(unresolved)} 个未恢复失败",
            _event_ids(finishes),
        ),
        CheckResult(
            "validation_executed",
            validation_passed if mutation_events else None,
            1.0 if validation_passed else 0.0,
            1.0,
            (
                "文件修改后存在成功验证"
                if validation_passed
                else "文件修改后未检测到成功测试"
            ) if mutation_events else "本轨迹无文件修改，该 Check 不适用",
            _event_ids(validation_events),
        ),
        CheckResult(
            "no_guardrail_halt",
            not _events_of(events, "guardrail_halt"),
            0.0 if _events_of(events, "guardrail_halt") else 1.0,
            1.0,
            "未触发 Guardrail 终止" if not _events_of(events, "guardrail_halt") else "触发 Guardrail 终止",
            _event_ids(_events_of(events, "guardrail_halt")),
        ),
    )
    metrics = {
        "api_calls": len(requests),
        "tool_calls": tool_total,
        "tool_status_counts": dict(statuses),
        "redundant_tool_calls": redundant_calls,
        "redundant_tool_call_rate": redundant_rate,
        "effective_actions": unique_successful_actions,
        "effective_action_ratio": effective_action_ratio,
        "tool_failures": total_failures,
        "recovered_tool_failures": recovered_failures,
        "tool_failure_recovery_rate": recovery_rate,
        "validation_calls": len(validation_events),
        "file_mutation_calls": len(mutation_events),
        "input_tokens": sum(
            int(event.get("payload", {}).get("usage", {}).get("input_tokens", 0) or 0)
            for event in responses
            if isinstance(event.get("payload"), Mapping)
            and isinstance(event.get("payload", {}).get("usage"), Mapping)
        ),
        "output_tokens": sum(
            int(event.get("payload", {}).get("usage", {}).get("output_tokens", 0) or 0)
            for event in responses
            if isinstance(event.get("payload"), Mapping)
            and isinstance(event.get("payload", {}).get("usage"), Mapping)
        ),
        "duration_ms": int(events[-1].get("elapsed_ms", 0) or 0),
        "resolved": resolved if isinstance(resolved, bool) else None,
    }
    return TraceEvaluation(
        trace_id=trace_id,
        valid=not invalid_reasons,
        checks=checks,
        metrics=metrics,
        invalid_reasons=tuple(invalid_reasons),
    )
