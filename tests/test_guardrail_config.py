"""tool_loop_guardrails YAML 配置测试。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config_loader import load_config
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


class GuardrailConfigTests(unittest.TestCase):
    def test_missing_and_empty_config_return_empty_mapping(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            self.assertEqual(load_config(root / "missing.yaml"), {})
            empty = root / "empty.yaml"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(load_config(empty), {})

    def test_loads_yaml_mapping(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            path = Path(task_tmp) / "config.yaml"
            path.write_text(
                "tool_loop_guardrails:\n  warnings_enabled: false\n",
                encoding="utf-8",
            )
            data = load_config(path)
        self.assertFalse(data["tool_loop_guardrails"]["warnings_enabled"])

    def test_invalid_yaml_and_non_mapping_root_fail_clearly(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            invalid = root / "invalid.yaml"
            invalid.write_text("tool_loop_guardrails: [", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "无法读取配置文件"):
                load_config(invalid)

            sequence = root / "sequence.yaml"
            sequence.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "顶层必须是 mapping"):
                load_config(sequence)

            invalid_encoding = root / "invalid-encoding.yaml"
            invalid_encoding.write_bytes(b"\xff\xfe\x00")
            with self.assertRaisesRegex(RuntimeError, "无法读取配置文件"):
                load_config(invalid_encoding)

    def test_nested_hermes_mapping_sets_all_thresholds(self):
        config = ToolCallGuardrailConfig.from_mapping({
            "warnings_enabled": False,
            "hard_stop_enabled": False,
            "warn_after": {
                "exact_failure": 4,
                "same_tool_failure": 5,
                "idempotent_no_progress": 6,
            },
            "hard_stop_after": {
                "exact_failure": 7,
                "same_tool_failure": 8,
                "idempotent_no_progress": 9,
            },
        })
        self.assertEqual(
            (
                config.warnings_enabled,
                config.hard_stop_enabled,
                config.exact_failure_warn_after,
                config.same_tool_failure_warn_after,
                config.no_progress_warn_after,
                config.exact_failure_block_after,
                config.same_tool_failure_halt_after,
                config.no_progress_block_after,
            ),
            (False, False, 4, 5, 6, 7, 8, 9),
        )

    def test_flat_legacy_fields_remain_supported(self):
        config = ToolCallGuardrailConfig.from_mapping({
            "exact_failure_warn_after": 4,
            "same_tool_failure_halt_after": 10,
        })
        self.assertEqual(config.exact_failure_warn_after, 4)
        self.assertEqual(config.same_tool_failure_halt_after, 10)

    def test_invalid_thresholds_fall_back_to_defaults(self):
        defaults = ToolCallGuardrailConfig()
        config = ToolCallGuardrailConfig.from_mapping({
            "warn_after": {
                "exact_failure": 0,
                "same_tool_failure": -1,
                "idempotent_no_progress": "invalid",
            },
        })
        self.assertEqual(config.exact_failure_warn_after, defaults.exact_failure_warn_after)
        self.assertEqual(config.same_tool_failure_warn_after, defaults.same_tool_failure_warn_after)
        self.assertEqual(config.no_progress_warn_after, defaults.no_progress_warn_after)

    def test_environment_only_overrides_explicitly_set_switches(self):
        yaml_config = ToolCallGuardrailConfig.from_mapping({
            "warnings_enabled": False,
            "hard_stop_enabled": False,
        })
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(yaml_config.with_environment_overrides(), yaml_config)
        with patch.dict(
            os.environ,
            {"TOOL_GUARDRAIL_WARNINGS": "true", "TOOL_GUARDRAIL_HARD_STOP": "on"},
            clear=True,
        ):
            overridden = yaml_config.with_environment_overrides()
        self.assertTrue(overridden.warnings_enabled)
        self.assertTrue(overridden.hard_stop_enabled)

    def test_yaml_threshold_changes_controller_behavior(self):
        config = ToolCallGuardrailConfig.from_mapping({
            "hard_stop_enabled": True,
            "hard_stop_after": {"exact_failure": 1},
        })
        controller = ToolCallGuardrailController(config)
        controller.after_call("read_file", {"path": "x"}, "Error", failed=True)
        decision = controller.before_call("read_file", {"path": "x"})
        self.assertEqual(decision.code, "repeated_exact_failure_block")

    def test_unknown_tool_limit_is_not_guardrail_configuration(self):
        config = ToolCallGuardrailConfig.from_mapping({
            "hard_stop_after": {"invalid_tool": 99},
        })
        self.assertFalse(hasattr(config, "invalid_tool_halt_after"))


if __name__ == "__main__":
    unittest.main()
