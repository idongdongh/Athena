"""CLI 会话斜杠命令测试。"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from agent.context_state import ContextSettings
from agent.tool_guardrails import ToolCallGuardrailConfig
from athena_cli.commands import command_names
from athena_cli.config import MemorySettings, SessionSettings
from cli import AthenaCLI
from run_agent import AIAgent
from session_db import SessionDB


class _CLI(AthenaCLI):
    def __init__(self, db):
        self.model = "model"
        self.system_prompt = "system"
        self.context_settings = ContextSettings()
        self.tool_guardrail_config = ToolCallGuardrailConfig()
        self.memory_settings = MemorySettings(
            memory_enabled=False,
            user_profile_enabled=False,
        )
        self.api_key = "test"
        self.base_url = None
        self._session_db = db
        self.command_names = command_names()
        self._pending_resume_sessions = None
        self.conversation_history = []
        self.agent = AIAgent(
            model=self.model,
            system_prompt=self.system_prompt,
            context_settings=self.context_settings,
            session_db=db,
            client=SimpleNamespace(),
        )

    def _create_agent(self):
        return AIAgent(
            model=self.model,
            system_prompt=self.system_prompt,
            context_settings=self.context_settings,
            session_db=self._session_db,
            client=SimpleNamespace(),
        )


class SessionCommandTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = SessionDB(Path(self.tempdir.name) / "state.db")
        self.cli = _CLI(self.db)

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    def command(self, text):
        output = io.StringIO()
        with redirect_stdout(output):
            handled = self.cli.process_command(text)
        return handled, output.getvalue()

    def test_settings_parse_and_resolve_relative_path(self):
        settings = SessionSettings.from_mapping({
            "session": {"enabled": False, "database": "data/state.db"}
        })
        self.assertFalse(settings.enabled)
        self.assertEqual(
            settings.resolve_database_path(Path("/tmp/project")),
            Path("/tmp/project/data/state.db"),
        )

    def test_new_ends_old_session_and_clears_history(self):
        old_id = self.cli.agent.session_id
        self.cli.conversation_history.append({"role": "user", "content": "hello"})
        handled, _ = self.command("/new")
        self.assertTrue(handled)
        self.assertNotEqual(self.cli.agent.session_id, old_id)
        self.assertEqual(self.cli.conversation_history, [])
        self.assertEqual(self.db.get_session(old_id)["end_reason"], "session_reset")

    def test_sessions_target_delegates_to_resume(self):
        other = AIAgent(
            model="model",
            system_prompt="system",
            context_settings=ContextSettings(),
            session_db=self.db,
            client=SimpleNamespace(),
        )
        other.flush_new_messages([{"role": "user", "content": "other"}])
        old_id = self.cli.agent.session_id
        self.command(f"/sessions {other.session_id}")
        self.assertEqual(self.cli.agent.session_id, other.session_id)
        self.assertEqual(self.cli.conversation_history[0]["content"], "other")
        self.assertEqual(self.db.get_session(old_id)["end_reason"], "resumed_other")

    def test_list_search_archive_and_plain_input(self):
        self.cli.conversation_history.append(
            {"role": "user", "content": "中文会话搜索目标"}
        )
        self.cli.agent.flush_new_messages(self.cli.conversation_history)
        self.assertIn(self.cli.agent.session_id, self.command("/sessions")[1])
        searched = self.command("/search 会话搜索")[1]
        if self.db.fts_enabled:
            self.assertIn(self.cli.agent.session_id, searched)
        self.assertIn("已归档", self.command("/archive")[1])
        self.assertFalse(self.cli.process_command("hello"))


if __name__ == "__main__":
    unittest.main()
