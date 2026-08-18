"""将低层事件折叠为 Judge 和 Failure Diagnosis 使用的 step。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, Mapping) else {}


def build_trajectory_steps(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        step_id = event.get("step_id")
        if isinstance(step_id, int):
            grouped[step_id].append(event)
    steps: list[dict[str, Any]] = []
    for step_id in sorted(grouped):
        step_events = grouped[step_id]
        response = next(
            (event for event in step_events if event.get("event_type") == "model_response"),
            None,
        )
        response_payload = _payload(response) if response else {}
        tools = []
        for event in step_events:
            if event.get("event_type") != "tool_result":
                continue
            payload = _payload(event)
            tools.append(
                {
                    "event_id": event.get("event_id"),
                    "tool_call_id": payload.get("tool_call_id"),
                    "tool_name": payload.get("tool_name"),
                    "tool_args": payload.get("tool_args"),
                    "status": payload.get("status"),
                    "error_code": payload.get("error_code"),
                    "output": payload.get("output"),
                }
            )
        steps.append(
            {
                "step_id": step_id,
                "event_ids": [event.get("event_id") for event in step_events],
                "model_response": response_payload.get("content"),
                "stop_reason": response_payload.get("stop_reason"),
                "tools": tools,
                "guardrail_halt": any(event.get("event_type") == "guardrail_halt" for event in step_events),
                "invalid_tool_batch": any(event.get("event_type") == "invalid_tool_batch" for event in step_events),
            }
        )
    return steps


def first_rule_failure_candidate(steps: Sequence[Mapping[str, Any]]) -> int | None:
    """返回需要 Judge 复核的最早规则候选点，不直接宣称因果。"""
    failed_signatures: dict[tuple[str, str], int] = {}
    for step in steps:
        step_id = step.get("step_id")
        if not isinstance(step_id, int):
            continue
        if step.get("invalid_tool_batch"):
            return step_id
        for tool in step.get("tools", []):
            if not isinstance(tool, Mapping):
                continue
            signature = (str(tool.get("tool_name", "")), repr(tool.get("tool_args")))
            status = tool.get("status")
            if status in {"failed", "internal_error", "unknown"}:
                if signature in failed_signatures:
                    return failed_signatures[signature]
                failed_signatures[signature] = step_id
            elif status == "succeeded":
                failed_signatures.pop(signature, None)
        if step.get("guardrail_halt"):
            return min(failed_signatures.values(), default=step_id)
    return min(failed_signatures.values(), default=None)
