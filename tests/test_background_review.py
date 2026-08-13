"""后台记忆复盘的触发、白名单和结果汇总。"""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from anthropic.types import TextBlock, ToolUseBlock

from agent import conversation_loop
from agent.background_review import _successful_memory_actions, spawn_background_review_thread
from agent.context_state import ContextSettings
from agent.interrupt_controller import InterruptController
from athena_cli.config import MemorySettings
from run_agent import AIAgent
from tools.registry import discover


class BackgroundReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        discover()

    def test_successful_action_summary_only_reads_new_successful_results(self):
        messages = [
            {"role": "user", "content": "old"},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "content": json.dumps({"success": True, "target": "user"}),
                }],
            },
        ]
        self.assertEqual(_successful_memory_actions(messages, 1), ["已更新用户画像"])

    def test_restricted_tool_name_is_rejected_even_if_globally_registered(self):
        call = ToolUseBlock(id="x", name="bash", input={}, type="tool_use")
        results = conversation_loop._invalid_tool_results([call], {"memory"})
        self.assertIsNotNone(results)
        self.assertIn("Available tools: memory", results[0]["content"])

    def test_completed_turn_triggers_review_after_response(self):
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="done")],
            usage=None,
        )
        agent = SimpleNamespace(
            client=SimpleNamespace(),
            model="test",
            system_prompt="system",
            context_settings=SimpleNamespace(
                max_output_tokens=100,
                max_length_continuations=1,
                context_window=1000,
                compression_enabled=False,
                max_compressions_per_turn=0,
            ),
            context_state=SimpleNamespace(
                update_from_response=lambda *_args, **_kwargs: SimpleNamespace(
                    effective_input_tokens=1, output_tokens=1, estimated=False
                )
            ),
            context_compressor=SimpleNamespace(update_from_response=lambda *_args: None),
            tool_guardrails=SimpleNamespace(reset_for_turn=lambda: None, halt_decision=None),
            file_mutation_tracker=SimpleNamespace(reset_for_turn=lambda: None, format_notice=lambda: ""),
            _memory_store=SimpleNamespace(),
            _is_background_review=False,
            interrupt_controller=InterruptController(),
            begin_memory_review_cycle=lambda: True,
            note_memory_tool_call=lambda: None,
            tool_definitions=lambda: [],
            flush_new_messages=lambda _messages: 0,
            record_usage=lambda _usage: None,
            spawn_background_memory_review=lambda messages: triggered.append(list(messages)) or True,
        )
        triggered = []
        messages = [{"role": "user", "content": "remember me"}]
        with patch.object(conversation_loop, "_stream_message_with_recovery", return_value=(response, False)):
            conversation_loop.run_conversation(agent, messages)
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0][-1]["role"], "assistant")

    def test_review_agent_can_only_write_shared_memory_without_parent_messages(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            settings = MemorySettings(
                memory_enabled=True,
                user_profile_enabled=True,
                directory=task_tmp,
                nudge_interval=1,
            )
            parent = AIAgent(
                model="test",
                system_prompt="",
                context_settings=ContextSettings(compression_enabled=False),
                client=SimpleNamespace(),
                memory_settings=settings,
                memory_root=Path(task_tmp),
            )
            snapshot = [
                {"role": "user", "content": "以后回答简洁"},
                {"role": "assistant", "content": [TextBlock(type="text", text="知道了")]},
            ]
            original = list(snapshot)
            responses = [
                SimpleNamespace(
                    stop_reason="tool_use",
                    content=[ToolUseBlock(
                        id="mem-1",
                        name="memory",
                        input={"action": "add", "target": "user", "content": "用户偏好简洁回答"},
                        type="tool_use",
                    )],
                    usage=None,
                ),
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[TextBlock(type="text", text="saved")],
                    usage=None,
                ),
            ]
            with patch.object(
                conversation_loop,
                "_stream_message_with_recovery",
                side_effect=[(responses[0], False), (responses[1], False)],
            ), redirect_stdout(io.StringIO()):
                thread = spawn_background_review_thread(parent, snapshot)
                thread.start()
                thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(snapshot, original)
            self.assertEqual(parent._memory_store.user_entries, ["用户偏好简洁回答"])


if __name__ == "__main__":
    unittest.main()
