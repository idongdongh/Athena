"""评测运行的稳定数据协议。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TRACE_SCHEMA_VERSION = "athena.trace.v1"


@dataclass(frozen=True)
class TraceEvent:
    schema_version: str
    trace_id: str
    event_id: int
    event_type: str
    timestamp: str
    elapsed_ms: int
    task_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    step_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool | None
    score: float
    max_score: float
    reason: str
    evidence_event_ids: tuple[int, ...] = ()
    source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_event_ids"] = list(self.evidence_event_ids)
        return data


@dataclass(frozen=True)
class TraceEvaluation:
    trace_id: str
    valid: bool
    checks: tuple[CheckResult, ...]
    metrics: dict[str, Any]
    invalid_reasons: tuple[str, ...] = ()

    @property
    def check_success_rate(self) -> float:
        applicable = [check for check in self.checks if check.passed is not None]
        if not applicable:
            return 0.0
        return sum(1 for check in applicable if check.passed) / len(applicable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "valid": self.valid,
            "invalid_reasons": list(self.invalid_reasons),
            "check_success_rate": self.check_success_rate,
            "checks": [check.to_dict() for check in self.checks],
            "metrics": self.metrics,
        }
