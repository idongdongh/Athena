"""会话级上下文状态测试。"""

import unittest
from types import SimpleNamespace

from agent.context_state import ContextSettings, ContextState
from agent.model_response import inspect_model_response


class ContextStateTests(unittest.TestCase):
    def test_settings_load_valid_yaml_sections(self):
        settings = ContextSettings.from_mapping({
            "model": {"max_output_tokens": 8192, "context_window": 200_000},
            "context": {
                "max_length_continuations": 3,
            },
            "compression": {
                "enabled": False,
                "threshold": 0.6,
                "target_ratio": 0.25,
                "protect_first_n": 4,
                "protect_last_n": 8,
                "abort_on_summary_failure": False,
                "max_per_turn": 3,
            },
        })

        self.assertEqual(settings.max_output_tokens, 8192)
        self.assertEqual(settings.context_window, 200_000)
        self.assertEqual(settings.compression_threshold, 0.6)
        self.assertEqual(settings.max_length_continuations, 3)
        self.assertFalse(settings.compression_enabled)
        self.assertEqual(settings.compression_target_ratio, 0.25)
        self.assertEqual(settings.protect_first_n, 4)
        self.assertEqual(settings.protect_last_n, 8)
        self.assertFalse(settings.abort_on_summary_failure)
        self.assertEqual(settings.max_compressions_per_turn, 3)

    def test_invalid_settings_fall_back_to_defaults(self):
        defaults = ContextSettings()
        settings = ContextSettings.from_mapping({
            "model": {"max_output_tokens": True, "context_window": -1},
            "context": {
                "compression_threshold": 1.5,
                "max_length_continuations": -1,
            },
            "compression": {
                "enabled": "yes",
                "threshold": 2.0,
                "target_ratio": 0.0,
                "protect_first_n": -1,
                "protect_last_n": -1,
                "abort_on_summary_failure": "no",
                "max_per_turn": 0,
            },
        })

        self.assertEqual(settings, defaults)

    def test_real_usage_updates_current_pressure_and_session_totals(self):
        state = ContextState(context_window=100, compression_threshold=0.5)
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[],
            usage=SimpleNamespace(
                input_tokens=20,
                output_tokens=5,
                cache_creation_input_tokens=10,
                cache_read_input_tokens=20,
            ),
        )

        usage = state.update_from_response(
            inspect_model_response(response),
            system="system",
            messages=[],
            tools=[],
            response_content=response.content,
        )

        self.assertEqual(usage.effective_input_tokens, 50)
        self.assertEqual(state.usage_ratio, 0.5)
        self.assertTrue(state.should_compress)
        self.assertEqual(state.session_input_tokens, 50)
        self.assertEqual(state.session_output_tokens, 5)

    def test_missing_usage_uses_marked_estimate(self):
        state = ContextState(context_window=1000, compression_threshold=0.75)
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="answer")],
        )

        usage = state.update_from_response(
            inspect_model_response(response),
            system="system",
            messages=[{"role": "user", "content": "你好"}],
            tools=[{"name": "read_file"}],
            response_content=response.content,
        )

        self.assertTrue(usage.estimated)
        self.assertGreater(usage.input_tokens, 0)
        self.assertGreater(usage.output_tokens, 0)
        self.assertIs(state.last_usage, usage)


if __name__ == "__main__":
    unittest.main()
