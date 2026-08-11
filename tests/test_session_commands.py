"""REPL 会话命令测试。"""

import tempfile
import unittest
from pathlib import Path

from agent.session_commands import handle_session_command
from agent.session_db import SessionDB
from agent.session_runtime import SessionRuntime, SessionSettings


class SessionCommandTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = SessionDB(Path(self.tempdir.name) / "state.db")
        self.runtime = SessionRuntime.start(self.db, model="model")

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    def command(self, text):
        return handle_session_command(
            text,
            db=self.db,
            runtime=self.runtime,
            model="model",
            system_prompt="system",
        )

    def test_settings_parse_and_resolve_relative_path(self):
        settings = SessionSettings.from_mapping({
            "session": {
                "enabled": False,
                "database": "data/state.db",
                "auto_resume": False,
            }
        })
        self.assertFalse(settings.enabled)
        self.assertFalse(settings.auto_resume)
        self.assertEqual(
            settings.resolve_database_path(Path("/tmp/project")),
            Path("/tmp/project/data/state.db"),
        )

    def test_new_ends_old_session_and_returns_empty_history(self):
        old_id = self.runtime.session_id
        result = self.command("/new")
        self.assertTrue(result.handled)
        self.assertNotEqual(result.runtime.session_id, old_id)
        self.assertEqual(result.messages, [])
        self.assertTrue(result.reset_context)
        self.assertEqual(self.db.get_session(old_id)["end_reason"], "session_reset")

    def test_resume_restores_existing_messages(self):
        history = [{"role": "user", "content": "hello"}]
        self.runtime.flush_new_messages(history)
        result = self.command(f"/resume {self.runtime.session_id}")
        self.assertTrue(result.handled)
        self.assertIsNone(result.messages)
        self.assertIn("已经位于", result.output)

    def test_bare_resume_lists_sessions(self):
        result = self.command("/resume")
        self.assertIn("近期会话", result.output)
        self.assertIn(self.runtime.session_id, result.output)

    def test_sessions_target_delegates_to_resume(self):
        other = SessionRuntime.start(self.db, model="model")
        other.flush_new_messages([{"role": "user", "content": "other"}])
        result = self.command(f"/sessions {other.session_id}")
        self.assertEqual(result.runtime.session_id, other.session_id)
        self.assertEqual(result.messages[0]["content"], "other")
        self.assertEqual(
            self.db.get_session(self.runtime.session_id)["end_reason"],
            "resumed_other",
        )
        self.assertIsNone(self.db.get_session(other.session_id)["end_reason"])

    def test_list_search_and_archive(self):
        self.runtime.flush_new_messages([
            {"role": "user", "content": "中文会话搜索目标"},
        ])
        listed = self.command("/sessions")
        self.assertIn(self.runtime.session_id, listed.output)

        searched = self.command("/search 会话搜索")
        if self.db.fts_enabled:
            self.assertIn(self.runtime.session_id, searched.output)

        archived = self.command(f"/archive {self.runtime.session_id}")
        self.assertIn("已归档", archived.output)
        self.assertEqual(self.db.list_sessions_rich(), [])

    def test_plain_input_is_not_consumed(self):
        self.assertFalse(self.command("hello").handled)


if __name__ == "__main__":
    unittest.main()
