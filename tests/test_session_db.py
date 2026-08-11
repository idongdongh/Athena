"""SessionDB 的真实 SQLite 集成测试。"""

import tempfile
import unittest
from pathlib import Path

from agent.session_db import SessionDB, new_session_id


class SessionDBTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "state.db"
        self.db = SessionDB(self.db_path)
        self.session_id = new_session_id()
        self.db.create_session(
            self.session_id,
            "cli",
            model="claude-test",
            system_prompt="stable prompt",
        )

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    def test_schema_uses_wal_and_session_crud(self):
        mode = self.db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")

        self.assertTrue(self.db.set_session_title(self.session_id, "demo"))
        self.db.update_session_usage(
            self.session_id,
            input_tokens=12,
            output_tokens=3,
            api_calls=1,
        )
        session = self.db.get_session(self.session_id)

        self.assertEqual(session["title"], "demo")
        self.assertEqual(session["input_tokens"], 12)
        self.assertEqual(session["output_tokens"], 3)
        self.assertEqual(session["api_call_count"], 1)

    def test_structured_content_round_trips_and_counts_tool_calls(self):
        content = [{
            "type": "tool_use",
            "id": "call-1",
            "name": "read_file",
            "input": {"path": "README.md"},
        }]
        self.db.append_message(self.session_id, "assistant", content)
        self.db.append_message(
            self.session_id,
            "user",
            [{
                "type": "tool_result",
                "tool_use_id": "call-1",
                "content": "file body",
            }],
        )

        restored = self.db.get_messages_as_conversation(self.session_id)
        session = self.db.get_session(self.session_id)

        self.assertEqual(restored[0]["content"], content)
        self.assertEqual(restored[1]["content"][0]["content"], "file body")
        self.assertEqual(session["message_count"], 2)
        self.assertEqual(session["tool_call_count"], 1)

    def test_append_rolls_back_when_session_does_not_exist(self):
        with self.assertRaises(Exception):
            self.db.append_message("missing", "user", "must fail")

        count = self.db._conn.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_archive_and_compact_preserves_searchable_history(self):
        self.db.append_message(self.session_id, "user", "旧消息：部署到杭州")
        self.db.append_message(self.session_id, "assistant", "已记录旧方案")

        inserted = self.db.archive_and_compact(
            self.session_id,
            [
                {"role": "user", "content": "[CONTEXT COMPACTION] 部署摘要"},
                {"role": "assistant", "content": "继续执行"},
            ],
        )

        active = self.db.get_messages_as_conversation(self.session_id)
        all_rows = self.db.get_messages(self.session_id, include_inactive=True)
        self.assertEqual(inserted, 2)
        self.assertEqual(len(active), 2)
        self.assertEqual(len(all_rows), 4)
        self.assertEqual(sum(row["compacted"] for row in all_rows), 2)
        if self.db.fts_enabled:
            matches = self.db.search_messages("旧消息")
            self.assertEqual(matches[0]["session_id"], self.session_id)

    def test_compression_chain_resolves_to_latest_tip(self):
        self.db.end_session(self.session_id, "compression")
        child = new_session_id()
        self.db.create_session(
            child,
            "cli",
            parent_session_id=self.session_id,
        )
        self.db.end_session(child, "compression")
        grandchild = new_session_id()
        self.db.create_session(grandchild, "cli", parent_session_id=child)

        self.assertEqual(self.db.get_compression_tip(self.session_id), grandchild)
        self.assertEqual(
            self.db.resolve_resume_session_id(self.session_id),
            grandchild,
        )

    def test_fts_searches_text_and_handles_special_characters(self):
        if not self.db.fts_enabled:
            self.skipTest("SQLite runtime has no FTS5")
        self.db.append_message(self.session_id, "user", "docker deployment failed")
        self.assertEqual(len(self.db.search_messages("docker")), 1)
        self.assertEqual(self.db.search_messages('"(()'), [])

    def test_close_is_idempotent(self):
        self.db.close()
        self.db.close()


if __name__ == "__main__":
    unittest.main()
