"""可断点续跑的批量 LLM Judge 管线。"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

from evaluation.judge import JudgeClient, JudgeSettings, evaluate_semantic_rubric
from evaluation.pipeline import discover_traces
from evaluation.rubric import load_rubric
from evaluation.trace_store import load_trace_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def judge_plan(root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    source = Path(root).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else source / "judged"
    traces = discover_traces(source)
    pending = [trace for trace in traces if not (destination / f"{trace.name}.json").exists()]
    return {
        "source": str(source),
        "output": str(destination),
        "total_traces": len(traces),
        "cached_traces": len(traces) - len(pending),
        "pending_traces": len(pending),
        "estimated_primary_requests": len(pending),
        "note": "Low-confidence review requests are not included in the primary estimate.",
    }


def judge_run(
    root: str | Path,
    *,
    model: str | None = None,
    output: str | Path | None = None,
    max_concurrency: int = 2,
    requests_per_minute: int = 20,
    force: bool = False,
) -> dict[str, Any]:
    plan = judge_plan(root, output)
    source = Path(plan["source"])
    destination = Path(plan["output"])
    destination.mkdir(parents=True, exist_ok=True)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    judge_model = model or os.getenv("JUDGE_MODEL_ID") or os.getenv("MODEL_ID") or ""
    api_key = (
        os.getenv("JUDGE_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("API_KEY")
    )
    if not judge_model or not api_key:
        raise RuntimeError("Judge requires --model/JUDGE_MODEL_ID/MODEL_ID and an API key")
    api_format = (os.getenv("JUDGE_API_FORMAT") or "anthropic").strip().lower()
    judge_base_url = os.getenv("JUDGE_BASE_URL")
    if api_format == "openai":
        client = OpenAI(api_key=api_key, base_url=judge_base_url, max_retries=0)
    elif api_format == "anthropic":
        client = Anthropic(
            api_key=api_key,
            base_url=judge_base_url or os.getenv("BASE_URL") or os.getenv("ANTHROPIC_BASE_URL"),
            max_retries=0,
        )
    else:
        raise ValueError(f"unsupported JUDGE_API_FORMAT: {api_format}")
    judge = JudgeClient(
        client,
        JudgeSettings(
            model=judge_model,
            api_format=api_format,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
        ),
    )
    rubric = load_rubric()
    pending = [
        trace for trace in discover_traces(source)
        if force or not (destination / f"{trace.name}.json").exists()
    ]
    if force:
        plan = {
            **plan,
            "cached_traces": 0,
            "pending_traces": len(pending),
            "estimated_primary_requests": len(pending),
        }
    lock = threading.Lock()

    def evaluate_one(trace: Path) -> dict[str, Any]:
        events = load_trace_events(trace)
        result = evaluate_semantic_rubric(events, rubric, judge)
        row = {"trace_id": trace.name, "status": "completed", "result": result}
        path = destination / f"{trace.name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return row

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(evaluate_one, trace): trace for trace in pending}
        for future in as_completed(futures):
            trace = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "trace_id": trace.name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            with lock:
                rows.append(row)
    summary = {
        **plan,
        "model": judge_model,
        "api_format": api_format,
        "completed_now": sum(row.get("status") == "completed" for row in rows),
        "failed_now": sum(row.get("status") == "failed" for row in rows),
        "rows": sorted(rows, key=lambda row: str(row.get("trace_id"))),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
