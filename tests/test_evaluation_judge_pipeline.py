import tempfile
import unittest
from pathlib import Path

from evaluation.judge_pipeline import judge_plan
from evaluation.trace_store import JsonlTraceRecorder


class JudgePipelineTests(unittest.TestCase):
    def test_plan_counts_cached_and_pending_without_calling_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = JsonlTraceRecorder(root, trace_id="trace_one")
            first.emit("turn_start", {})
            second = JsonlTraceRecorder(root, trace_id="trace_two")
            second.emit("turn_start", {})
            output = root / "judge_output"
            output.mkdir()
            (output / "trace_one.json").write_text("{}\n", encoding="utf-8")
            plan = judge_plan(root, output)
            self.assertEqual(2, plan["total_traces"])
            self.assertEqual(1, plan["cached_traces"])
            self.assertEqual(1, plan["pending_traces"])


if __name__ == "__main__":
    unittest.main()
