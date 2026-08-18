"""LLM-as-Judge：批量 Rubric 评分与 Failure Onset 复核。"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from evaluation.rubric import Rubric
from evaluation.trajectory import build_trajectory_steps, first_rule_failure_candidate


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    api_format: str = "anthropic"
    # 长轨迹需要返回 7 个 Rubric 维度和所有 step label；4K 在真实实验中会截断 JSON。
    max_tokens: int = 8192
    max_retries: int = 3
    max_concurrency: int = 2
    requests_per_minute: int = 20


class DualRateLimiter:
    """同时限制在途请求数和滑动一分钟请求数。"""

    def __init__(self, max_concurrency: int, requests_per_minute: int) -> None:
        if max_concurrency <= 0 or requests_per_minute <= 0:
            raise ValueError("rate limits must be positive")
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._rpm = requests_per_minute
        self._lock = threading.Lock()
        self._starts: list[float] = []

    @contextmanager
    def acquire(self):
        self._semaphore.acquire()
        try:
            while True:
                wait_seconds = 0.0
                with self._lock:
                    now = time.monotonic()
                    self._starts = [started for started in self._starts if now - started < 60.0]
                    if len(self._starts) < self._rpm:
                        self._starts.append(now)
                        break
                    wait_seconds = max(0.01, 60.0 - (now - self._starts[0]))
                time.sleep(wait_seconds)
            yield
        finally:
            self._semaphore.release()


def _extract_text(response: Any) -> str:
    content = response.get("content", []) if isinstance(response, Mapping) else getattr(response, "content", [])
    parts = []
    for block in content or []:
        block_type = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
        text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", None)
        if block_type == "text" and isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("judge response must be a JSON object")
    return data


class JudgeClient:
    def __init__(
        self,
        client: Any,
        settings: JudgeSettings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.client = client
        self.settings = settings
        self.limiter = DualRateLimiter(settings.max_concurrency, settings.requests_per_minute)
        self._sleep = sleep
        self._random = random_value

    def request_json(self, system: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for retry in range(self.settings.max_retries + 1):
            try:
                with self.limiter.acquire():
                    user_content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                    if self.settings.api_format == "anthropic":
                        response = self.client.messages.create(
                            model=self.settings.model,
                            max_tokens=self.settings.max_tokens,
                            temperature=0,
                            system=system,
                            messages=[{"role": "user", "content": user_content}],
                        )
                        text = _extract_text(response)
                    elif self.settings.api_format == "openai":
                        response = self.client.chat.completions.create(
                            model=self.settings.model,
                            max_tokens=self.settings.max_tokens,
                            temperature=0,
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user_content},
                            ],
                        )
                        choices = getattr(response, "choices", None) or []
                        text = (
                            getattr(getattr(choices[0], "message", None), "content", "")
                            if choices else ""
                        )
                    else:
                        raise ValueError(f"unsupported judge api_format: {self.settings.api_format}")
                return _parse_json_object(str(text or ""))
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or (isinstance(status, int) and status >= 500)
                if not retryable or retry >= self.settings.max_retries:
                    raise
                wait_seconds = min(30.0, 2.0 ** retry) + self._random() * 0.25
                self._sleep(wait_seconds)
        assert last_error is not None
        raise last_error


_JUDGE_SYSTEM = """You are evaluating a coding-agent trajectory. Return JSON only.
Every verdict must cite concrete event_ids. Do not infer hidden reasoning.
Score each requested dimension from 0 to 4. For failure diagnosis, label each step as one of
CORRECT, NEUTRAL, RECOVERABLE_ERROR, RECOVERY, FAILURE_ONSET, CASCADE_FAILURE.
FAILURE_ONSET is the earliest consequential wrong decision that was not later recovered;
a tool failure by itself is not necessarily a failure onset."""


def evaluate_semantic_rubric(
    events: Sequence[Mapping[str, Any]],
    rubric: Rubric,
    judge: JudgeClient,
    *,
    task_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    steps = build_trajectory_steps(events)
    candidate = first_rule_failure_candidate(steps)
    payload = {
        "task": dict(task_context or {}),
        "rubric": [
            {
                "id": dimension.dimension_id,
                "name": dimension.name,
                "weight": dimension.weight,
                "description": dimension.description,
            }
            for dimension in rubric.dimensions
        ],
        "trajectory_steps": steps,
        "rule_failure_candidate_step": candidate,
        "required_output": {
            "checks": [{
                "dimension_id": "string",
                "score": "integer 0..4",
                "verdict": "pass|partial|fail|not_applicable",
                "evidence_event_ids": ["integer"],
                "reason": "string",
                "confidence": "number 0..1",
            }],
            "step_labels": [{
                "step_id": "integer",
                "label": "allowed step label",
                "evidence_event_ids": ["integer"],
                "reason": "string",
            }],
            "failure_onset_step": "integer or null",
            "failure_category": "string or null",
            "summary": "string",
        },
    }
    try:
        result = judge.request_json(_JUDGE_SYSTEM, payload)
        return validate_judge_result(result, rubric, events)
    except (json.JSONDecodeError, ValueError) as exc:
        # 只对输出截断/结构缺失回退；证据引用错误仍应显式失败，不能被补判掩盖。
        if not isinstance(exc, json.JSONDecodeError) and not str(exc).startswith("judge result is missing"):
            raise
    checks_payload = {
        key: value for key, value in payload.items()
        if key not in {"required_output"}
    }
    checks_payload["required_output"] = {"checks": payload["required_output"]["checks"]}
    checks_payload["instruction"] = "Return only checks. Keep each reason under 40 words."
    labels_payload = {
        "task": payload["task"],
        "trajectory_steps": steps,
        "rule_failure_candidate_step": candidate,
        "required_output": {
            key: value for key, value in payload["required_output"].items()
            if key != "checks"
        },
        "instruction": "Return only step labels, onset, category, and summary. Keep each reason under 30 words.",
    }
    check_result = judge.request_json(_JUDGE_SYSTEM, checks_payload)
    label_result = judge.request_json(_JUDGE_SYSTEM, labels_payload)
    merged = {**label_result, "checks": check_result.get("checks")}
    return validate_judge_result(merged, rubric, events)


def validate_judge_result(
    result: Mapping[str, Any],
    rubric: Rubric,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    known_dimensions = {dimension.dimension_id for dimension in rubric.dimensions}
    known_events = {event.get("event_id") for event in events}
    checks = result.get("checks")
    if not isinstance(checks, list):
        raise ValueError("judge result is missing checks")
    seen_dimensions: set[str] = set()
    normalized_checks = []
    for raw in checks:
        if not isinstance(raw, Mapping):
            raise ValueError("judge check must be an object")
        dimension_id = raw.get("dimension_id")
        if dimension_id not in known_dimensions or dimension_id in seen_dimensions:
            raise ValueError(f"invalid or duplicate judge dimension: {dimension_id}")
        score = raw.get("score")
        confidence = raw.get("confidence")
        evidence = raw.get("evidence_event_ids")
        verdict = raw.get("verdict")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
            raise ValueError(f"invalid score for {dimension_id}")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError(f"invalid confidence for {dimension_id}")
        if not isinstance(evidence, list) or any(event_id not in known_events for event_id in evidence):
            raise ValueError(f"invalid evidence for {dimension_id}")
        if verdict not in {"pass", "partial", "fail", "not_applicable"}:
            raise ValueError(f"invalid verdict for {dimension_id}")
        seen_dimensions.add(str(dimension_id))
        normalized_checks.append(dict(raw))
    missing = known_dimensions - seen_dimensions
    if missing:
        raise ValueError(f"judge omitted dimensions: {sorted(missing)}")
    normalized = dict(result)
    normalized["checks"] = normalized_checks
    weight_by_dimension = {dimension.dimension_id: dimension.weight for dimension in rubric.dimensions}
    applicable_checks = [
        check for check in normalized_checks if check.get("verdict") != "not_applicable"
    ]
    applicable_weight = sum(
        weight_by_dimension[str(check["dimension_id"])] for check in applicable_checks
    )
    normalized["weighted_score"] = (
        sum(
            float(check["score"]) / 4.0 * weight_by_dimension[str(check["dimension_id"])]
            for check in applicable_checks
        ) / applicable_weight * 100.0
        if applicable_weight else 0.0
    )
    normalized["applicable_weight"] = applicable_weight
    normalized["mean_confidence"] = sum(float(check["confidence"]) for check in normalized_checks) / len(normalized_checks)
    normalized["needs_review"] = any(float(check["confidence"]) < 0.7 for check in normalized_checks)
    onset = normalized.get("failure_onset_step")
    valid_steps = {event.get("step_id") for event in events if isinstance(event.get("step_id"), int)}
    if onset is not None and onset not in valid_steps:
        raise ValueError("failure_onset_step does not exist in trajectory")
    allowed_labels = {
        "CORRECT", "NEUTRAL", "RECOVERABLE_ERROR", "RECOVERY",
        "FAILURE_ONSET", "CASCADE_FAILURE",
    }
    raw_labels = normalized.get("step_labels")
    if not isinstance(raw_labels, list):
        raise ValueError("judge result is missing step_labels")
    seen_steps: set[int] = set()
    normalized_labels = []
    for raw in raw_labels:
        if not isinstance(raw, Mapping):
            raise ValueError("step label must be an object")
        step_id = raw.get("step_id")
        label = raw.get("label")
        evidence = raw.get("evidence_event_ids")
        if step_id not in valid_steps or step_id in seen_steps:
            raise ValueError(f"invalid or duplicate step label: {step_id}")
        if label not in allowed_labels:
            raise ValueError(f"invalid step label: {label}")
        if not isinstance(evidence, list) or any(event_id not in known_events for event_id in evidence):
            raise ValueError(f"invalid step evidence: {step_id}")
        seen_steps.add(step_id)
        normalized_labels.append(dict(raw))
    if seen_steps != valid_steps:
        raise ValueError(f"judge omitted step labels: {sorted(valid_steps - seen_steps)}")
    onset_labels = [item["step_id"] for item in normalized_labels if item["label"] == "FAILURE_ONSET"]
    if onset is None and onset_labels:
        raise ValueError("failure onset label conflicts with null onset")
    if onset is not None and onset_labels != [onset]:
        raise ValueError("failure onset step conflicts with step labels")
    normalized["step_labels"] = normalized_labels
    return normalized
