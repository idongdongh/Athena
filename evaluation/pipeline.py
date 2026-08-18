"""轨迹批量评价、Wash 与报告编排。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.checks import evaluate_trace
from evaluation.trace_store import load_trace_events


@dataclass(frozen=True)
class EvaluatedRun:
    source: Path
    output: Path
    evaluations: tuple[dict[str, Any], ...]


def discover_traces(root: str | Path) -> list[Path]:
    source = Path(root).expanduser().resolve()
    if (source / "events.jsonl").is_file():
        return [source]
    return sorted(path.parent for path in source.rglob("events.jsonl"))


def evaluate_run(root: str | Path, output: str | Path | None = None) -> EvaluatedRun:
    source = Path(root).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else source / "evaluated"
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trace_dir in discover_traces(source):
        try:
            evaluation = evaluate_trace(load_trace_events(trace_dir)).to_dict()
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            evaluation = {
                "trace_id": trace_dir.name,
                "valid": False,
                "invalid_reasons": [f"load_error:{type(exc).__name__}:{exc}"],
                "check_success_rate": 0.0,
                "checks": [],
                "metrics": {},
            }
        evaluation["trace_path"] = str(trace_dir)
        rows.append(evaluation)
    jsonl_path = destination / "evaluations.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return EvaluatedRun(source, destination, tuple(rows))


def wash_run(evaluated: EvaluatedRun) -> dict[str, Any]:
    """仅生成派生清单，不删除任何原始轨迹。"""
    included = [row for row in evaluated.evaluations if row.get("valid")]
    excluded = [row for row in evaluated.evaluations if not row.get("valid")]
    manifest = {
        "source": str(evaluated.source),
        "total": len(evaluated.evaluations),
        "included": len(included),
        "excluded": len(excluded),
        "included_trace_ids": [row.get("trace_id") for row in included],
        "excluded_traces": [
            {
                "trace_id": row.get("trace_id"),
                "reasons": row.get("invalid_reasons", []),
            }
            for row in excluded
        ],
    }
    (evaluated.output / "wash_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_report(evaluated: EvaluatedRun) -> str:
    valid = [row for row in evaluated.evaluations if row.get("valid")]
    total = len(evaluated.evaluations)
    judge_results = _load_judge_results(evaluated.source / "judged")
    judged = [judge_results[row.get("trace_id")] for row in valid if row.get("trace_id") in judge_results]
    resolved_rows = [row for row in valid if isinstance(row.get("metrics", {}).get("resolved"), bool)]
    resolved = sum(1 for row in resolved_rows if row.get("metrics", {}).get("resolved") is True)
    def average(metric: str) -> float:
        values = [row.get("metrics", {}).get(metric) for row in valid]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        return sum(numeric) / len(numeric) if numeric else 0.0

    lines = [
        "# Athena Agent Tracer Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Total traces: {total}",
        f"- Valid traces: {len(valid)}",
        f"- Invalid traces: {total - len(valid)}",
        f"- Instance success rate: {resolved / len(resolved_rows):.2%}" if resolved_rows else "- Instance success rate: N/A",
        f"- Mean check success rate: {sum(float(row.get('check_success_rate', 0)) for row in valid) / len(valid):.2%}" if valid else "- Mean check success rate: N/A",
        f"- Mean effective action ratio: {average('effective_action_ratio'):.2%}",
        f"- Mean redundant tool call rate: {average('redundant_tool_call_rate'):.2%}",
        f"- Mean tool failure recovery rate: {average('tool_failure_recovery_rate'):.2%}",
        f"- Mean input tokens: {average('input_tokens'):.1f}",
        f"- Mean output tokens: {average('output_tokens'):.1f}",
        f"- Judge coverage: {len(judged)}/{len(valid)} valid traces",
        (
            f"- Mean Judge weighted score: "
            f"{sum(float(row['weighted_score']) for row in judged) / len(judged):.2f}/100"
            if judged
            else "- Mean Judge weighted score: N/A"
        ),
        f"- Failure Onset detected: {sum(row.get('failure_onset_step') is not None for row in judged)}",
        f"- Judge review queue: {sum(bool(row.get('needs_review')) for row in judged)}",
        "",
        "## Trace Results",
        "",
        "| Trace | Valid | CSR | Tool Calls | EAR | Recovery | Judge | Onset |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluated.evaluations:
        metrics = row.get("metrics", {})
        judge = judge_results.get(str(row.get("trace_id")))
        judge_score = f"{float(judge['weighted_score']):.2f}" if judge else "N/A"
        onset = str(judge.get("failure_onset_step")) if judge and judge.get("failure_onset_step") is not None else "-"
        lines.append(
            f"| {row.get('trace_id', '')} | {'yes' if row.get('valid') else 'no'} "
            f"| {float(row.get('check_success_rate', 0)):.2%} "
            f"| {metrics.get('tool_calls', 0)} "
            f"| {float(metrics.get('effective_action_ratio', 0)):.2%} "
            f"| {float(metrics.get('tool_failure_recovery_rate', 0)):.2%} "
            f"| {judge_score} | {onset} |"
        )
    report = "\n".join(lines) + "\n"
    (evaluated.output / "report.md").write_text(report, encoding="utf-8")
    return report


def _load_judge_results(directory: Path) -> dict[str, dict[str, Any]]:
    """读取逐轨迹 Judge 结果；跳过 summary 和损坏缓存，不影响规则报告。"""
    results: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return results
    for path in sorted(directory.glob("trace_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            continue
        result = payload.get("result")
        trace_id = payload.get("trace_id")
        if (
            isinstance(trace_id, str)
            and isinstance(result, dict)
            and isinstance(result.get("weighted_score"), (int, float))
        ):
            results[trace_id] = result
    return results
