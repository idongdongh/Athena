"""SessionRuntime 增量同步测试。"""

import tempfile
import unittest
from pathlib import Path

from agent.model_response import TokenUsage
from agent.session_db import SessionDB
from agent.session_runtime import SessionRuntime


class SessionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = SessionDB(Path(self.tempdir.name) / "state.db")
        self.runtime = SessionRuntime.start(
            self.db,
            model="claude-test",
            system_prompt="stable",
        )

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    def test_flush_is_incremental_and_idempotent(self):
        messages = [{"role": "user", "content": "one"}]
        self.assertEqual(self.runtime.flush_new_messages(messages), 1)
        self.assertEqual(self.runtime.flush_new_messages(messages), 0)
        messages.append({"role": "assistant", "content": "two"})
        self.assertEqual(self.runtime.flush_new_messages(messages), 1)
        self.assertEqual(
            len(self.db.get_messages(self.runtime.session_id)),
            2,
        )

    def test_resume_restores_messages_and_flush_cursor(self):
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        self.runtime.flush_new_messages(messages)

        resumed, restored = SessionRuntime.resume(self.db, self.runtime.session_id)
        self.assertEqual(restored, messages)
        self.assertEqual(resumed.flushed_count, 2)
        self.assertEqual(resumed.flush_new_messages(restored), 0)

    def test_usage_accumulates_in_session(self):
        self.runtime.record_usage(TokenUsage(input_tokens=10, output_tokens=3))
        self.runtime.record_usage(TokenUsage(input_tokens=20, output_tokens=4))
        session = self.db.get_session(self.runtime.session_id)
        self.assertEqual(session["input_tokens"], 30)
        self.assertEqual(session["output_tokens"], 7)
        self.assertEqual(session["api_call_count"], 2)

    def test_shortened_history_requires_explicit_rebaseline(self):
        messages = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        self.runtime.flush_new_messages(messages)
        with self.assertRaises(RuntimeError):
            self.runtime.flush_new_messages(messages[:1])


if __name__ == "__main__":
    unittest.main()
