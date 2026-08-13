"""上下文压缩与 SessionDB 的一致性测试。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from agent.context_state import ContextSettings, ContextState
from agent.conversation_compression import compress_context
from session_db import SessionDB
from run_agent import AIAgent


def _messages(turns=8):
    result = []
    for index in range(turns):
        result.extend([
            {"role": "user", "content": f"user-{index} " + "x" * 200},
            {"role": "assistant", "content": f"assistant-{index} " + "y" * 200},
        ])
    return result


class SessionCompressionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = SessionDB(Path(self.tempdir.name) / "state.db")
        self.runtime = AIAgent(
            model="claude-test",
            system_prompt="system",
            context_settings=ContextSettings(),
            session_db=self.db,
            client=SimpleNamespace(),
        )
        self.state = ContextState(context_window=2000, compression_threshold=0.2)

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    @staticmethod
    def _compressor():
        return ContextCompressor(
            context_length=2000,
            max_tokens=200,
            threshold_percent=0.2,
            protect_first_n=2,
            protect_last_n=2,
            summary_target_ratio=0.2,
            summary_callback=lambda _prompt, _budget: "checkpoint",
        )

    def _compress(self, messages, *, in_place):
        return compress_context(
            self._compressor(),
            self.state,
            messages,
            system="system",
            tools=[],
            current_tokens=1500,
            session_runtime=self.runtime,
            in_place=in_place,
        )

    def test_rotation_preserves_parent_and_resumes_child(self):
        messages = _messages()
        old_id = self.runtime.session_id

        result = self._compress(messages, in_place=False)

        self.assertTrue(result.changed)
        new_id = self.runtime.session_id
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(self.db.get_session(old_id)["end_reason"], "compression")
        self.assertEqual(self.db.get_session(new_id)["parent_session_id"], old_id)
        self.assertEqual(self.db.resolve_resume_session_id(old_id), new_id)
        self.assertEqual(
            self.db.get_messages_as_conversation(new_id),
            messages,
        )
        self.assertGreater(len(self.db.get_messages(old_id)), len(messages))

    def test_in_place_archives_old_history_under_same_id(self):
        messages = _messages()
        session_id = self.runtime.session_id

        result = self._compress(messages, in_place=True)

        self.assertTrue(result.changed)
        self.assertEqual(self.runtime.session_id, session_id)
        self.assertEqual(
            self.db.get_messages_as_conversation(session_id),
            messages,
        )
        all_rows = self.db.get_messages(session_id, include_inactive=True)
        self.assertGreater(len(all_rows), len(messages))
        self.assertTrue(any(row["compacted"] for row in all_rows))

    def test_persistence_failure_keeps_history_and_compressor_state(self):
        messages = _messages()
        original = list(messages)
        compressor = self._compressor()
        self.db.close()

        result = compress_context(
            compressor,
            self.state,
            messages,
            system="system",
            tools=[],
            current_tokens=1500,
            session_runtime=self.runtime,
        )

        self.assertFalse(result.changed)
        self.assertEqual(messages, original)
        self.assertIn("持久化失败", result.error)
        self.assertEqual(compressor.compression_count, 0)
        self.assertIsNone(compressor._previous_summary)
        self.assertFalse(compressor.awaiting_real_usage_after_compression)


if __name__ == "__main__":
    unittest.main()
