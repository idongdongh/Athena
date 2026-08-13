"""MemoryStore 与 AIAgent、工具执行器的集成。"""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from agent.context_state import ContextSettings
from agent.tool_executor import execute_tool_calls
from athena_cli.config import MemorySettings
from run_agent import AIAgent
from tools.registry import discover


class MemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.settings = MemorySettings(
            memory_enabled=True,
            user_profile_enabled=True,
            directory=str(self.root),
        )
        discover()

    def tearDown(self):
        self.tempdir.cleanup()

    def make_agent(self):
        return AIAgent(
            model="test",
            system_prompt="base prompt",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
            memory_settings=self.settings,
        )

    def test_new_agent_injects_snapshot_but_mid_session_write_does_not(self):
        first = self.make_agent()
        first._memory_store.add("user", "用户喜欢简洁回答")
        self.assertNotIn("用户喜欢简洁回答", first.system_prompt)

        second = self.make_agent()
        self.assertIn("用户喜欢简洁回答", second.system_prompt)
        self.assertIn("USER PROFILE", second.system_prompt)

    def test_memory_tool_receives_current_agent_store(self):
        agent = self.make_agent()
        call = SimpleNamespace(
            type="tool_use",
            id="tool-1",
            name="memory",
            input={"action": "add", "target": "memory", "content": "use uv"},
        )
        messages = []
        execute_tool_calls([call], messages, memory_store=agent._memory_store)
        payload = json.loads(messages[-1]["content"][0]["content"])
        self.assertTrue(payload["success"])
        self.assertEqual(agent._memory_store.memory_entries, ["use uv"])

    def test_disabled_memory_is_not_exposed_to_model(self):
        agent = AIAgent(
            model="test",
            system_prompt="base prompt",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
            memory_settings=MemorySettings(),
        )
        self.assertIsNone(agent._memory_store)
        self.assertNotIn("memory", {item["name"] for item in agent.tool_definitions()})

    def test_background_review_agent_only_exposes_memory(self):
        agent = AIAgent(
            model="test",
            system_prompt="",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
            memory_settings=self.settings,
            tool_allowlist=frozenset({"memory"}),
            is_background_review=True,
        )
        self.assertEqual({item["name"] for item in agent.tool_definitions()}, {"memory"})
        self.assertFalse(agent.begin_memory_review_cycle())

    def test_memory_review_counter_triggers_and_memory_call_resets(self):
        settings = MemorySettings(
            memory_enabled=True,
            user_profile_enabled=True,
            directory=str(self.root),
            nudge_interval=2,
        )
        agent = AIAgent(
            model="test",
            system_prompt="",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
            memory_settings=settings,
        )
        self.assertFalse(agent.begin_memory_review_cycle())
        self.assertTrue(agent.begin_memory_review_cycle())
        agent.begin_memory_review_cycle()
        agent.note_memory_tool_call()
        self.assertEqual(agent._turns_since_memory, 0)

    def test_resume_counter_ignores_tool_result_user_messages(self):
        settings = MemorySettings(
            memory_enabled=True,
            directory=str(self.root),
            nudge_interval=10,
        )
        agent = AIAgent(
            model="test",
            system_prompt="",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
            memory_settings=settings,
        )
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "tool_result", "content": "done"}]},
            {"role": "user", "content": "two"},
        ]
        agent.reset_session_state(history)
        self.assertEqual(agent._turns_since_memory, 2)

    def test_only_one_background_review_runs_at_a_time(self):
        agent = self.make_agent()
        running = SimpleNamespace(is_alive=lambda: True)
        agent._background_review_thread = running
        with patch("agent.background_review.spawn_background_review_thread") as spawn:
            self.assertFalse(agent.spawn_background_memory_review([]))
        spawn.assert_not_called()

    def test_refresh_boundary_loads_mid_session_write(self):
        agent = self.make_agent()
        agent._memory_store.add("memory", "new durable fact")
        self.assertNotIn("new durable fact", agent.system_prompt)
        agent.refresh_memory_snapshot()
        self.assertIn("new durable fact", agent.system_prompt)

    def test_system_prompt_is_layered_and_tool_aware(self):
        enabled = self.make_agent()
        self.assertIn("You are Athena", enabled.system_prompt)
        self.assertIn("# Finishing the job", enabled.system_prompt)
        self.assertIn("# Tool-call batching", enabled.system_prompt)
        self.assertIn("persistent memory across sessions", enabled.system_prompt)
        self.assertIn("Working directory:", enabled.system_prompt)

        disabled = AIAgent(
            model="test",
            system_prompt="",
            context_settings=ContextSettings(),
            client=SimpleNamespace(),
            memory_settings=MemorySettings(),
        )
        self.assertNotIn("persistent memory across sessions", disabled.system_prompt)


if __name__ == "__main__":
    unittest.main()
