import json
import tempfile
import unittest
from pathlib import Path

from evaluation.checks import evaluate_trace
from evaluation.pipeline import build_report, evaluate_run, wash_run
from evaluation.trace_store import JsonlTraceRecorder, load_trace_events


class EvaluationPipelineTests(unittest.TestCase):
    def _successful_trace(self, root: str) -> JsonlTraceRecorder:
        recorder = JsonlTraceRecorder(root, task_id="task-success", trace_id="trace_success")
        common = {"turn_id": "turn-1", "session_id": "session-1"}
        recorder.emit("turn_start", common)
        recorder.emit("model_request", {**common, "step_id": 1, "api_request_id": "api-1"})
        recorder.emit(
            "model_response",
            {
                **common,
                "step_id": 1,
                "api_request_id": "api-1",
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )
        for call_id, tool_name, args in (
            ("call-1", "patch", {"path": "a.py", "patch": "x"}),
            ("call-2", "bash", {"command": "pytest -q"}),
        ):
            recorder.emit(
                "tool_call",
                {**common, "step_id": 1, "tool_call_id": call_id, "tool_name": tool_name, "tool_args": args},
            )
            recorder.emit(
                "tool_result",
                {
                    **common,
                    "step_id": 1,
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "tool_args": args,
                    "status": "succeeded",
                    "output": "ok",
                },
            )
        recorder.emit(
            "turn_finish",
            {**common, "outcome": "completed", "unresolved_file_mutations": []},
        )
        return recorder

    def test_rules_measure_validation_and_effective_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self._successful_trace(tmp)
            result = evaluate_trace(load_trace_events(recorder.root))
            self.assertTrue(result.valid)
            self.assertEqual(1.0, result.check_success_rate)
            self.assertEqual(2, result.metrics["tool_calls"])
            self.assertEqual(1.0, result.metrics["effective_action_ratio"])
            validation = next(check for check in result.checks if check.check_id == "validation_executed")
            self.assertTrue(validation.passed)

    def test_duplicate_calls_reduce_effective_action_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = JsonlTraceRecorder(tmp, trace_id="trace_repeat")
            common = {"turn_id": "turn-1", "session_id": "session-1"}
            recorder.emit("turn_start", common)
            for index in range(2):
                recorder.emit(
                    "tool_call",
                    {**common, "tool_call_id": f"call-{index}", "tool_name": "read_file", "tool_args": {"path": "a.py"}},
                )
                recorder.emit(
                    "tool_result",
                    {
                        **common,
                        "tool_call_id": f"call-{index}",
                        "tool_name": "read_file",
                        "tool_args": {"path": "a.py"},
                        "status": "succeeded",
                        "output": "same",
                    },
                )
            recorder.emit("turn_finish", {**common, "outcome": "completed", "unresolved_file_mutations": []})
            result = evaluate_trace(load_trace_events(recorder.root))
            self.assertEqual(1, result.metrics["redundant_tool_calls"])
            self.assertEqual(0.5, result.metrics["effective_action_ratio"])

    def test_evaluate_wash_and_report_keep_invalid_trace_as_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._successful_trace(tmp)
            broken = Path(tmp) / "trace_broken"
            broken.mkdir()
            (broken / "events.jsonl").write_text(
                json.dumps({"event_id": 2, "trace_id": "trace_broken"}) + "\n",
                encoding="utf-8",
            )
            evaluated = evaluate_run(tmp, Path(tmp) / "output")
            manifest = wash_run(evaluated)
            report = build_report(evaluated)
            self.assertEqual(2, manifest["total"])
            self.assertEqual(1, manifest["included"])
            self.assertEqual(1, manifest["excluded"])
            self.assertIn("trace_success", report)
            self.assertTrue((Path(tmp) / "output" / "evaluations.jsonl").exists())

    def test_provider_error_is_classified_as_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = JsonlTraceRecorder(tmp, trace_id="trace_auth_error")
            common = {"turn_id": "turn-1", "session_id": "session-1", "step_id": 1}
            recorder.emit("turn_start", common)
            recorder.emit("model_request", {**common, "api_request_id": "api-1"})
            recorder.emit(
                "model_error",
                {**common, "api_request_id": "api-1", "kind": "authentication", "status_code": 401},
            )
            recorder.emit(
                "turn_finish",
                {**common, "outcome": None, "terminal_error": {"kind": "authentication"}, "unresolved_file_mutations": []},
            )
            result = evaluate_trace(load_trace_events(recorder.root))
            self.assertFalse(result.valid)
            self.assertIn("infrastructure_provider_error:authentication", result.invalid_reasons)
            self.assertNotIn("model_request_response_mismatch", result.invalid_reasons)

    def test_report_joins_incremental_judge_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self._successful_trace(tmp)
            judged = Path(tmp) / "judged"
            judged.mkdir()
            (judged / f"{recorder.trace_id}.json").write_text(
                json.dumps(
                    {
                        "trace_id": recorder.trace_id,
                        "status": "completed",
                        "result": {
                            "weighted_score": 82.5,
                            "failure_onset_step": 3,
                            "needs_review": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            evaluated = evaluate_run(tmp, Path(tmp) / "evaluated")
            report = build_report(evaluated)
            self.assertIn("Judge coverage: 1/1", report)
            self.assertIn("Mean Judge weighted score: 82.50/100", report)
            self.assertIn("Failure Onset detected: 1", report)
            self.assertIn("82.50 | 3", report)


if __name__ == "__main__":
    unittest.main()
