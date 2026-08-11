"""上下文压缩与 conversation loop 的触发路径测试。"""

import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from anthropic.types import TextBlock, ToolUseBlock

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("MODEL_ID", "test-model")

import agent.conversation_loop as conversation_loop
from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor
from agent.context_state import ContextState


class _Stream:
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.response

    def close(self):
        pass


class _Manager:
    def __init__(self, response):
        self.stream = _Stream(response)

    def __enter__(self):
        return self.stream

    def __exit__(self, *_args):
        self.stream.close()


class _MessagesAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return _Manager(self.responses.pop(0))


def _history(turns=8):
    messages = []
    for index in range(turns):
        messages.extend([
            {"role": "user", "content": f"user {index} " + "x" * 200},
            {
                "role": "assistant",
                "content": [TextBlock(
                    type="text",
                    text=f"assistant {index} " + "y" * 200,
                )],
            },
        ])
    return messages


def _compressor():
    return ContextCompressor(
        context_length=2000,
        max_tokens=200,
        threshold_percent=0.2,
        protect_first_n=2,
        protect_last_n=2,
        summary_target_ratio=0.2,
        summary_callback=lambda _prompt, _budget: "compressed checkpoint",
    )


class ConversationCompressionIntegrationTests(unittest.TestCase):
    def tearDown(self):
        conversation_loop.interrupt_controller.clear()

    def test_preflight_compresses_before_sending_large_request(self):
        api = _MessagesAPI([
            SimpleNamespace(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="done")],
                usage=SimpleNamespace(input_tokens=100, output_tokens=2),
            )
        ])
        messages = [*_history(), {"role": "user", "content": "latest task"}]
        state = ContextState(context_window=2000, compression_threshold=0.2)

        with (
            patch.object(conversation_loop, "client", SimpleNamespace(messages=api)),
            patch.object(conversation_loop, "COMPRESSION_ENABLED", True),
        ):
            conversation_loop.agent_loop(
                messages,
                context_state=state,
                context_compressor=_compressor(),
            )

        self.assertEqual(len(api.calls), 1)
        self.assertIn(SUMMARY_PREFIX, repr(api.calls[0]["messages"]))
        self.assertIn("latest task", repr(api.calls[0]["messages"]))
        self.assertEqual(messages[-1]["content"][0].text, "done")

    def test_real_usage_triggers_compression_after_tool_results(self):
        api = _MessagesAPI([
            SimpleNamespace(
                stop_reason="tool_use",
                content=[ToolUseBlock(
                    type="tool_use",
                    id="read-1",
                    name="read_file",
                    input={"path": "a.py"},
                )],
                usage=SimpleNamespace(input_tokens=500, output_tokens=10),
            ),
            SimpleNamespace(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="finished")],
                usage=SimpleNamespace(input_tokens=120, output_tokens=3),
            ),
        ])
        messages = [*_history(), {"role": "user", "content": "read a.py"}]
        state = ContextState(context_window=2000, compression_threshold=0.2)

        def append_tool_result(_blocks, target_messages, **_kwargs):
            target_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "read-1",
                    "content": "file contents",
                }],
            })

        # 首次调用不走 preflight，专门验证真实 usage 的 post-response 触发。
        compressor = _compressor()
        compressor.awaiting_real_usage_after_compression = True
        with (
            patch.object(conversation_loop, "client", SimpleNamespace(messages=api)),
            patch.object(conversation_loop, "execute_tool_calls", side_effect=append_tool_result),
            patch.object(conversation_loop, "COMPRESSION_ENABLED", True),
        ):
            conversation_loop.agent_loop(
                messages,
                context_state=state,
                context_compressor=compressor,
            )

        self.assertEqual(len(api.calls), 2)
        second_request = repr(api.calls[1]["messages"])
        self.assertIn(SUMMARY_PREFIX, second_request)
        self.assertIn("read-1", second_request)
        self.assertIn("file contents", second_request)
        self.assertEqual(messages[-1]["content"][0].text, "finished")

    def test_compression_summary_helper_rejects_truncated_summary(self):
        response = SimpleNamespace(
            stop_reason="max_tokens",
            content=[TextBlock(type="text", text="partial summary")],
        )
        with patch.object(
            conversation_loop,
            "_stream_message_with_recovery",
            return_value=(response, False),
        ):
            with self.assertRaisesRegex(RuntimeError, "max_tokens"):
                conversation_loop._generate_context_summary("prompt", 256)


if __name__ == "__main__":
    unittest.main()
