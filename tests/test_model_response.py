"""模型响应归一化测试。"""

import unittest
from types import SimpleNamespace

from agent.model_response import ModelStopReason, inspect_model_response


class ModelResponseTests(unittest.TestCase):
    def test_normalizes_stop_reason_and_content_kinds(self):
        response = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="text", text="working"),
                SimpleNamespace(type="tool_use", name="read_file", input={}),
            ],
        )

        state = inspect_model_response(response)

        self.assertIs(state.stop_reason, ModelStopReason.TOOL_USE)
        self.assertTrue(state.has_text)
        self.assertTrue(state.has_tool_calls)

    def test_unknown_and_missing_stop_reasons_are_explicit(self):
        unknown = inspect_model_response(
            SimpleNamespace(stop_reason="future_reason", content=[])
        )
        missing = inspect_model_response(SimpleNamespace(content=[]))

        self.assertIs(unknown.stop_reason, ModelStopReason.UNKNOWN)
        self.assertEqual(unknown.raw_stop_reason, "future_reason")
        self.assertIs(missing.stop_reason, ModelStopReason.UNKNOWN)
        self.assertIsNone(missing.raw_stop_reason)

    def test_extracts_anthropic_usage_including_cache_tokens(self):
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[],
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=4,
                cache_creation_input_tokens=3,
                cache_read_input_tokens=7,
            ),
        )

        usage = inspect_model_response(response).usage

        self.assertIsNotNone(usage)
        self.assertEqual(usage.effective_input_tokens, 20)
        self.assertEqual(usage.output_tokens, 4)
        self.assertFalse(usage.estimated)

    def test_missing_usage_remains_none_for_context_estimator(self):
        state = inspect_model_response(
            SimpleNamespace(stop_reason="end_turn", content=[])
        )
        self.assertIsNone(state.usage)


if __name__ == "__main__":
    unittest.main()
