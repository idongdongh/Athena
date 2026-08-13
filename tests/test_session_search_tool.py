"""session_search 的真实 SQLite 查询与 Agent 注入契约。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.context_state import ContextSettings
from agent.tool_executor import execute_tool_calls
from run_agent import AIAgent
from session_db import SessionDB
from tools.registry import discover
from tools.session_search_tool import session_search


class SessionSearchToolTests(unittest.TestCase):
    def setUp(self):
        discover()
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = SessionDB(Path(self.tempdir.name) / "state.db")
        self.db.create_session("current", "cli", title="当前")
        self.db.append_message("current", "user", "当前讨论记忆搜索")
        self.db.create_session("parent", "cli", title="工具失败处理", started_at=1)
        self.first_id = self.db.append_message("parent", "user", "开始研究工具调用失败")
        self.anchor_id = self.db.append_message("parent", "assistant", "使用 guardrail 处理重复失败")
        self.db.append_message("parent", "user", "最后决定保留四级动作")
        self.db.create_session("child", "cli", parent_session_id="parent", started_at=2)
        self.child_anchor_id = self.db.append_message("child", "assistant", "guardrail 后续实现完成")
        self.db.create_session("other", "cli", title="Docker", started_at=3)
        self.db.append_message("other", "user", "杭州部署使用 Docker")

    def tearDown(self):
        self.db.close()
        self.tempdir.cleanup()

    def payload(self, **kwargs):
        return json.loads(session_search(db=self.db, current_session_id="current", **kwargs))

    def test_discovery_returns_bookends_window_and_dedupes_lineage(self):
        result = self.payload(query="guardrail", limit=3)
        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "discover")
        self.assertEqual(result["count"], 1)
        hit = result["results"][0]
        self.assertIn(hit["session_id"], {"parent", "child"})
        self.assertIn(hit["match_message_id"], {self.anchor_id, self.child_anchor_id})
        self.assertTrue(hit["bookend_start"])
        self.assertTrue(hit["bookend_end"])
        self.assertTrue(any(message.get("anchor") for message in hit["messages"]))

    def test_chinese_discovery_and_current_lineage_exclusion(self):
        result = self.payload(query="杭州部署")
        self.assertEqual(result["results"][0]["session_id"], "other")
        current = self.payload(query="记忆搜索")
        self.assertEqual(current["count"], 0)

    def test_discovery_can_match_session_title(self):
        result = self.payload(query="工具失败处理")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["matched_role"], "session_title")

    def test_invalid_fts_expression_is_recoverable_as_empty_result(self):
        result = self.payload(query="(")
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)

    def test_browse_read_and_scroll_shapes(self):
        browse = self.payload(limit=10)
        self.assertEqual(browse["mode"], "browse")
        self.assertNotIn("current", {item["session_id"] for item in browse["results"]})

        read = self.payload(session_id="parent")
        self.assertEqual(read["mode"], "read")
        self.assertEqual(read["total_messages"], 3)

        scroll = self.payload(session_id="parent", around_message_id=self.anchor_id, window=1)
        self.assertEqual(scroll["mode"], "scroll")
        self.assertEqual(len(scroll["messages"]), 3)
        self.assertEqual(scroll["messages_before"], 1)
        self.assertEqual(scroll["messages_after"], 1)

    def test_invalid_anchor_and_active_session_are_recoverable(self):
        missing = self.payload(session_id="parent", around_message_id=999)
        self.assertFalse(missing["success"])
        active = self.payload(session_id="current")
        self.assertFalse(active["success"])

    def test_read_truncates_long_session_and_archived_sessions_disappear(self):
        self.db.create_session("long", "cli", title="长会话")
        for index in range(35):
            self.db.append_message("long", "user", f"message {index}")
        read = self.payload(session_id="long")
        self.assertTrue(read["truncated"])
        self.assertEqual(read["total_messages"], 35)
        self.assertEqual(read["returned_messages"], 30)

        self.db.archive_session("long")
        archived = self.payload(session_id="long")
        self.assertFalse(archived["success"])

    def test_executor_injects_current_database(self):
        call = SimpleNamespace(
            type="tool_use",
            id="search-1",
            name="session_search",
            input={"query": "Docker"},
        )
        messages = []
        execute_tool_calls(
            [call],
            messages,
            session_db=self.db,
            current_session_id="current",
            show_progress=False,
        )
        result = json.loads(messages[-1]["content"][0]["content"])
        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "discover")

    def test_tool_visibility_and_prompt_are_database_aware(self):
        enabled = AIAgent(
            model="test",
            system_prompt="",
            context_settings=ContextSettings(),
            session_db=self.db,
            session_id="current",
            client=SimpleNamespace(),
        )
        self.assertIn("session_search", {tool["name"] for tool in enabled.tool_definitions()})
        self.assertIn("past conversation", enabled.system_prompt)

        disabled = AIAgent(
            model="test",
            system_prompt="",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
        )
        self.assertNotIn("session_search", {tool["name"] for tool in disabled.tool_definitions()})
        self.assertNotIn("past conversation", disabled.system_prompt)


if __name__ == "__main__":
    unittest.main()
