"""SessionRuntime 增量同步测试。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.context_state import ContextSettings
from agent.model_response import TokenUsage
from session_db import SessionDB
from run_agent import AIAgent


class SessionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = SessionDB(Path(self.tempdir.name) / "state.db")
        self.runtime = AIAgent(
            model="claude-test",
            system_prompt="stable",
            context_settings=ContextSettings(),
            session_db=self.db,
            client=SimpleNamespace(),
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

        resumed, restored = AIAgent.resume(
            self.db,
            self.runtime.session_id,
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
        )
        self.assertEqual(restored, messages)
        self.assertEqual(resumed.flushed_count, 2)
        self.assertEqual(resumed.flush_new_messages(restored), 0)

    def test_resume_does_not_nest_stored_complete_system_prompt(self):
        self.db.create_session(
            "legacy-prompt",
            "cli",
            model="test",
            system_prompt="OLD COMPLETE PROMPT WITH STALE MEMORY",
        )
        self.db.append_message("legacy-prompt", "user", "hello")
        resumed, _ = AIAgent.resume(
            self.db,
            "legacy-prompt",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
        )
        self.assertNotIn("OLD COMPLETE PROMPT", resumed.system_prompt)
        self.assertEqual(resumed.system_prompt.count("You are Athena"), 1)

    def test_resume_uses_explicit_current_caller_prompt(self):
        self.db.create_session(
            "caller-prompt",
            "cli",
            model="test",
            system_prompt="old complete prompt",
        )
        resumed, _ = AIAgent.resume(
            self.db,
            "caller-prompt",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
            system_prompt="CURRENT CALLER RULE",
        )
        self.assertIn("CURRENT CALLER RULE", resumed.system_prompt)
        self.assertNotIn("old complete prompt", resumed.system_prompt)

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
