import json
import unittest
from types import SimpleNamespace

from evaluation.judge import JudgeClient, JudgeSettings, evaluate_semantic_rubric
from evaluation.rubric import load_rubric
from evaluation.trajectory import build_trajectory_steps, first_rule_failure_candidate


class _FakeMessages:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(output))])


class _RateLimitError(RuntimeError):
    status_code = 429


class _FakeOpenAICompletions:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.output)))]
        )


class EvaluationJudgeTests(unittest.TestCase):
    def _events(self):
        return [
            {
                "event_id": 1,
                "event_type": "model_response",
                "step_id": 1,
                "payload": {"content": "inspect code", "stop_reason": "tool_use"},
            },
            {
                "event_id": 2,
                "event_type": "tool_result",
                "step_id": 1,
                "payload": {
                    "tool_call_id": "c1",
                    "tool_name": "bash",
                    "tool_args": {"command": "pytest"},
                    "status": "failed",
                    "output": "failed",
                },
            },
            {
                "event_id": 3,
                "event_type": "model_response",
                "step_id": 2,
                "payload": {"content": "retry unchanged", "stop_reason": "tool_use"},
            },
            {
                "event_id": 4,
                "event_type": "tool_result",
                "step_id": 2,
                "payload": {
                    "tool_call_id": "c2",
                    "tool_name": "bash",
                    "tool_args": {"command": "pytest"},
                    "status": "failed",
                    "output": "failed",
                },
            },
        ]

    def _judge_result(self):
        rubric = load_rubric()
        return {
            "checks": [
                {
                    "dimension_id": dimension.dimension_id,
                    "score": 2,
                    "verdict": "partial",
                    "evidence_event_ids": [1],
                    "reason": "partial evidence",
                    "confidence": 0.8,
                }
                for dimension in rubric.dimensions
            ],
            "step_labels": [
                {"step_id": 1, "label": "FAILURE_ONSET", "evidence_event_ids": [1, 2], "reason": "unchanged retry"},
                {"step_id": 2, "label": "CASCADE_FAILURE", "evidence_event_ids": [3, 4], "reason": "same failure"},
            ],
            "failure_onset_step": 1,
            "failure_category": "TOOL_SELECTION",
            "summary": "Repeated a failed action.",
        }

    def test_step_builder_and_rule_candidate(self):
        steps = build_trajectory_steps(self._events())
        self.assertEqual([1, 2], [step["step_id"] for step in steps])
        self.assertEqual(1, first_rule_failure_candidate(steps))

    def test_judge_retries_429_and_validates_evidence(self):
        waits = []
        messages = _FakeMessages([_RateLimitError("limited"), self._judge_result()])
        judge = JudgeClient(
            SimpleNamespace(messages=messages),
            JudgeSettings(model="judge-test", max_retries=1, requests_per_minute=100),
            sleep=waits.append,
            random_value=lambda: 0.0,
        )
        result = evaluate_semantic_rubric(self._events(), load_rubric(), judge)
        self.assertEqual(1, result["failure_onset_step"])
        self.assertEqual(50.0, result["weighted_score"])
        self.assertFalse(result["needs_review"])
        self.assertEqual(2, len(messages.calls))
        self.assertEqual([1.0], waits)

    def test_judge_rejects_unknown_evidence_event(self):
        invalid = self._judge_result()
        invalid["checks"][0]["evidence_event_ids"] = [999]
        judge = JudgeClient(
            SimpleNamespace(messages=_FakeMessages([invalid])),
            JudgeSettings(model="judge-test", requests_per_minute=100),
        )
        with self.assertRaisesRegex(ValueError, "evidence"):
            evaluate_semantic_rubric(self._events(), load_rubric(), judge)

    def test_openai_compatible_judge_path(self):
        completions = _FakeOpenAICompletions(self._judge_result())
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        judge = JudgeClient(
            client,
            JudgeSettings(
                model="glm-test",
                api_format="openai",
                requests_per_minute=100,
            ),
        )
        result = evaluate_semantic_rubric(self._events(), load_rubric(), judge)
        self.assertEqual(50.0, result["weighted_score"])
        self.assertEqual("json_object", completions.calls[0]["response_format"]["type"])

    def test_not_applicable_dimension_is_excluded_from_weighted_denominator(self):
        output = self._judge_result()
        for check in output["checks"]:
            if check["dimension_id"] == "error_recovery":
                check["score"] = 4
                check["verdict"] = "not_applicable"
                check["evidence_event_ids"] = []
        judge = JudgeClient(
            SimpleNamespace(messages=_FakeMessages([output])),
            JudgeSettings(model="judge-test", requests_per_minute=100),
        )
        result = evaluate_semantic_rubric(self._events(), load_rubric(), judge)
        self.assertEqual(85.0, result["applicable_weight"])
        self.assertEqual(50.0, result["weighted_score"])

    def test_missing_full_response_falls_back_to_two_compact_requests(self):
        complete = self._judge_result()
        checks_only = {"checks": complete["checks"]}
        labels_only = {key: value for key, value in complete.items() if key != "checks"}
        messages = _FakeMessages([{}, checks_only, labels_only])
        judge = JudgeClient(
            SimpleNamespace(messages=messages),
            JudgeSettings(model="judge-test", requests_per_minute=100),
        )
        result = evaluate_semantic_rubric(self._events(), load_rubric(), judge)
        self.assertEqual(50.0, result["weighted_score"])
        self.assertEqual(3, len(messages.calls))
        second_payload = json.loads(messages.calls[1]["messages"][0]["content"])
        third_payload = json.loads(messages.calls[2]["messages"][0]["content"])
        self.assertEqual(["checks"], list(second_payload["required_output"]))
        self.assertNotIn("checks", third_payload["required_output"])


if __name__ == "__main__":
    unittest.main()
