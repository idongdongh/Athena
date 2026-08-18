import json
import tempfile
import unittest
from pathlib import Path

from agent.trace_events import emit_trace
from evaluation.trace_store import JsonlTraceRecorder, load_trace_events


class JsonlTraceRecorderTests(unittest.TestCase):
    def test_records_ordered_events_and_externalizes_large_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = JsonlTraceRecorder(
                tmp,
                task_id="task-1",
                session_id="session-1",
                inline_text_limit=256,
            )
            recorder.emit("turn_start", {"turn_id": "turn-1"})
            recorder.emit(
                "tool_result",
                {
                    "turn_id": "turn-1",
                    "step_id": 1,
                    "output": "x" * 300,
                    "api_key": "must-not-leak",
                },
            )

            events = load_trace_events(recorder.root)
            self.assertEqual([1, 2], [event["event_id"] for event in events])
            self.assertEqual("[REDACTED]", events[1]["payload"]["api_key"])
            output = events[1]["payload"]["output"]
            self.assertTrue(output["truncated"])
            artifact = recorder.root / output["artifact_ref"]
            self.assertEqual("x" * 300, artifact.read_text(encoding="utf-8"))

    def test_emit_trace_is_fail_open(self):
        class BrokenSink:
            def emit(self, event_type, payload=None):
                raise RuntimeError("broken")

        emit_trace(BrokenSink(), "turn_start", turn_id="turn-1")

    def test_loader_rejects_event_id_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            rows = [
                {"event_id": 1, "trace_id": "x"},
                {"event_id": 3, "trace_id": "x"},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "sequence"):
                load_trace_events(path)


if __name__ == "__main__":
    unittest.main()
