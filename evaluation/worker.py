"""在隔离任务工作区中运行一次 Athena 并采集 Patch/测试证据。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.context_state import ContextSettings
from agent.tool_guardrails import ToolCallGuardrailConfig
from athena_cli.config import load_config
from evaluation.trace_store import JsonlTraceRecorder
from run_agent import AIAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], workspace: Path, timeout: int = 600) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }


def run_task(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(payload["workspace"]).resolve()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    model = os.getenv("EVAL_EXECUTOR_MODEL") or os.getenv("MODEL_ID") or ""
    api_key = (
        os.getenv("EVAL_EXECUTOR_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("API_KEY")
    )
    if not model or not api_key:
        raise RuntimeError("MODEL_ID and ANTHROPIC_API_KEY/API_KEY are required")
    trace = JsonlTraceRecorder(
        payload["trace_root"],
        task_id=payload["task_id"],
    )
    baseline = _run(list(payload["test_command"]), workspace)
    trace.emit("baseline_test", baseline)

    config = load_config(PROJECT_ROOT / "config.yaml")
    guardrail = ToolCallGuardrailConfig.from_mapping(config.get("tool_loop_guardrails"))
    agent = AIAgent(
        model=model,
        system_prompt=(
            "You are running in an isolated coding evaluation workspace. "
            "Solve the task, edit only files inside the workspace, run relevant tests, "
            "and report the result accurately."
        ),
        context_settings=ContextSettings.from_mapping(config),
        api_key=api_key,
        base_url=(
            os.getenv("EVAL_EXECUTOR_BASE_URL")
            or os.getenv("BASE_URL")
            or os.getenv("ANTHROPIC_BASE_URL")
        ),
        tool_guardrail_config=guardrail,
        working_directory=workspace,
        trace_sink=trace,
    )
    messages = [{"role": "user", "content": payload["prompt"]}]
    agent.run_conversation(messages, stream_output=False)

    patch = _run(["git", "diff", "--binary", "--no-ext-diff"], workspace)
    trace.emit("patch_snapshot", {"patch": patch.get("stdout", ""), "returncode": patch["returncode"]})
    final_test = _run(list(payload["test_command"]), workspace)
    patch_nonempty = bool(str(patch.get("stdout", "")).strip())
    resolved = final_test["returncode"] == 0 and patch_nonempty
    trace.emit(
        "task_result",
        {
            "resolved": resolved,
            "patch_nonempty": patch_nonempty,
            "baseline_test": baseline,
            "final_test": final_test,
            "tags": payload.get("tags", []),
        },
    )
    result = {
        "trace_id": trace.trace_id,
        "trace_dir": str(trace.root),
        "resolved": resolved,
        "baseline_returncode": baseline["returncode"],
        "final_returncode": final_test["returncode"],
        "patch_nonempty": patch_nonempty,
    }
    (trace.root / "task_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.task.read_text(encoding="utf-8"))
    result = run_task(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
