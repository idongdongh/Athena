"""tool_loop_guardrails YAML 配置测试。"""

import tempfile
import unittest
from pathlib import Path

from athena_cli.config import MemorySettings, load_config
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


class GuardrailConfigTests(unittest.TestCase):
    def test_memory_settings_follow_hermes_defaults_and_validate_values(self):
        defaults = MemorySettings.from_mapping({})
        self.assertFalse(defaults.memory_enabled)
        self.assertFalse(defaults.user_profile_enabled)
        self.assertEqual(defaults.nudge_interval, 10)
        self.assertEqual(defaults.directory, "memories")
        self.assertEqual(
            defaults.resolve_directory(Path("/project")),
            Path("/project/memories"),
        )

        configured = MemorySettings.from_mapping({
            "memory": {
                "memory_enabled": True,
                "user_profile_enabled": True,
                "memory_char_limit": 3000,
                "user_char_limit": -1,
                "nudge_interval": 0,
                "directory": "~/custom-memory",
            }
        })
        self.assertTrue(configured.memory_enabled)
        self.assertEqual(configured.memory_char_limit, 3000)
        self.assertEqual(configured.user_char_limit, 1375)
        self.assertEqual(configured.nudge_interval, 0)
        self.assertEqual(
            configured.resolve_directory(Path("/project")),
            Path.home() / "custom-memory",
        )

        invalid_interval = MemorySettings.from_mapping({
            "memory": {"nudge_interval": -1},
        })
        self.assertEqual(invalid_interval.nudge_interval, 10)

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
