"""Hermes 风格上下文压缩器的核心回归测试。"""

import unittest

from agent.context_compressor import (
    SUMMARY_END_MARKER,
    SUMMARY_PREFIX,
    ContextCompressor,
)
from agent.context_state import ContextState
from agent.conversation_compression import compress_context


def _long_conversation(turns=8):
    messages = []
    for index in range(turns):
        messages.extend([
            {"role": "user", "content": f"user {index} " + "x" * 200},
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"assistant {index} " + "y" * 200,
                }],
            },
        ])
    return messages


class ContextCompressorTests(unittest.TestCase):
    def _compressor(self, callback=lambda _prompt, _budget: "checkpoint"):
        return ContextCompressor(
            context_length=2000,
            max_tokens=200,
            threshold_percent=0.2,
            protect_first_n=2,
            protect_last_n=2,
            summary_target_ratio=0.2,
            summary_callback=callback,
        )

    def test_threshold_reserves_output_budget(self):
        compressor = ContextCompressor(
            context_length=1000,
            max_tokens=200,
            threshold_percent=0.5,
        )
        self.assertEqual(compressor.threshold_tokens, 400)

    def test_preflight_defers_until_real_usage_after_compression(self):
        compressor = self._compressor()
        compressor.last_compression_rough_tokens = 500
        compressor.last_prompt_tokens = -1
        compressor.awaiting_real_usage_after_compression = True

        self.assertTrue(compressor.should_defer_preflight_to_real_usage(600))

        compressor.update_from_response({
            "prompt_tokens": 300,
            "completion_tokens": 20,
            "total_tokens": 320,
        })
        self.assertTrue(compressor.should_defer_preflight_to_real_usage(600))
        self.assertFalse(compressor.should_defer_preflight_to_real_usage(5000))

    def test_compress_protects_head_and_latest_exchange(self):
        messages = _long_conversation()
        compressor = self._compressor()

        compressed = compressor.compress(messages, current_tokens=1500)

        self.assertIsNot(compressed, messages)
        self.assertEqual(compressed[0], messages[0])
        self.assertEqual(compressed[1], messages[1])
        self.assertEqual(compressed[-2]["role"], "user")
        self.assertIn("user 7", str(compressed[-2]["content"]))
        self.assertEqual(compressed[-1], messages[-1])
        self.assertIn(SUMMARY_PREFIX, str(compressed))
        self.assertIn(SUMMARY_END_MARKER, str(compressed))
        self.assertEqual(compressor.compression_count, 1)

    def test_summary_failure_preserves_original_messages(self):
        messages = _long_conversation()
        snapshot = repr(messages)

        def fail(_prompt, _budget):
            raise RuntimeError("summary unavailable")

        compressor = self._compressor(fail)
        compressed = compressor.compress(messages, current_tokens=1500)

        self.assertIs(compressed, messages)
        self.assertEqual(repr(messages), snapshot)
        self.assertTrue(compressor._last_compress_aborted)
        self.assertIn("summary unavailable", compressor._last_summary_error)

    def test_summary_input_and_output_redact_credentials(self):
        prompts = []

        def summarize(prompt, _budget):
            prompts.append(prompt)
            return "checkpoint token=secret-value sk-abcdefghijklmnop"

        messages = _long_conversation()
        messages[2]["content"] = "api_key=top-secret-value"
        compressor = self._compressor(summarize)

        compressed = compressor.compress(messages, current_tokens=1500)

        self.assertNotIn("top-secret-value", prompts[0])
        self.assertNotIn("secret-value", repr(compressed))
        self.assertNotIn("sk-abcdefghijklmnop", repr(compressed))

    def test_previous_summary_is_used_for_iterative_update(self):
        prompts = []

        def summarize(prompt, _budget):
            prompts.append(prompt)
            return f"checkpoint {len(prompts)}"

        compressor = self._compressor(summarize)
        first = compressor.compress(_long_conversation(), current_tokens=1500)
        second_input = [
            *first,
            {"role": "user", "content": "new task " + "z" * 500},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "new answer " + "q" * 500}],
            },
            {"role": "user", "content": "latest task " + "w" * 500},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "latest answer " + "e" * 500}],
            },
        ]
        compressor.compress(second_input, current_tokens=1600)

        self.assertEqual(len(prompts), 2)
        self.assertIn("PREVIOUS CHECKPOINT", prompts[1])
        self.assertIn("checkpoint 1", prompts[1])

        second = compressor.compress(
            [
                *second_input,
                {"role": "user", "content": "third task " + "t" * 500},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "third answer " + "a" * 500}],
                },
            ],
            current_tokens=1800,
        )
        self.assertEqual(second[0]["role"], "user")
        self.assertTrue(all(
            left["role"] != right["role"]
            for left, right in zip(second, second[1:])
        ))

    def test_tool_use_and_result_remain_protocol_complete(self):
        messages = _long_conversation(4)
        messages.extend([
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "read_file",
                    "input": {"path": "large.py"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "result " + "r" * 1000,
                }],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ])
        compressor = self._compressor()

        compressed = compressor.compress(messages, current_tokens=1800)
        surviving_calls = {
            block.get("id")
            for message in compressed
            if message.get("role") == "assistant"
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        surviving_results = {
            block.get("tool_use_id")
            for message in compressed
            if message.get("role") == "user"
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        }

        self.assertEqual(surviving_calls, surviving_results)

    def test_orchestrator_replaces_history_in_place_only_on_success(self):
        messages = _long_conversation()
        original_identity = id(messages)
        state = ContextState(context_window=2000, compression_threshold=0.2)
        compressor = self._compressor()

        result = compress_context(
            compressor,
            state,
            messages,
            system="system",
            tools=[],
            current_tokens=1500,
        )

        self.assertTrue(result.changed)
        self.assertEqual(id(messages), original_identity)
        self.assertEqual(result.after_messages, len(messages))
        self.assertGreater(state.usage_ratio, 0)


if __name__ == "__main__":
    unittest.main()
