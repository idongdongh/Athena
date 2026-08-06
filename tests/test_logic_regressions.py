"""核心调用链的逻辑回归测试（仅使用标准库 unittest）。"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from anthropic.types import TextBlock, ToolUseBlock

# conversation_loop 在导入期校验配置；测试只构造 fake client，不会发送网络请求。
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("MODEL_ID", "test-model")

import agent.conversation_loop as conversation_loop
import agent.tool_executor as tool_executor
import agent.tracer as tracer_module
from agent.file_mutation_tracker import FileMutationTracker
from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    classify_tool_failure,
)
from agent.tool_result_classification import (
    BLOCKED,
    CANCELLED,
    FAILED,
    INTERNAL_ERROR,
    SUCCESS,
    classify_tool_result,
)
from tools.registry import registry
from tools.approval import _check_sudo_stdin_guard
from tools.path_security import check_write_path
from tools.read_file_tool import read_file
import tools.web_extract_tool as web_extract_module


class _FakeTracer:
    def step_start(self, *args, **kwargs):
        pass

    def step_done(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass

    def tool_call(self, *args, **kwargs):
        pass

    def tool_result(self, *args, **kwargs):
        pass


class _FakeMessagesAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="done")],
        )


class _SingleToolBatchAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) > 1:
            raise AssertionError("guardrail halt 后不应再次调用模型")
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                ToolUseBlock(type="tool_use", id=f"call-{i}", name="probe_fail", input={"x": 1})
                for i in range(6)
            ],
        )


class _FailingPatchThenDoneAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[ToolUseBlock(type="tool_use", id="patch-1", name="patch", input={"path": "src/a.py"})],
            )
        return SimpleNamespace(stop_reason="end_turn", content=[TextBlock(type="text", text="all done")])


class _PostToolEmptyThenDoneAPI:
    def __init__(self, stay_empty=False):
        self.calls = []
        self.stay_empty = stay_empty

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[ToolUseBlock(
                    type="tool_use",
                    id="empty-followup-call",
                    name="empty_probe",
                    input={},
                )],
            )
        if len(self.calls) == 2 or self.stay_empty:
            return SimpleNamespace(stop_reason="end_turn", content=[])
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="recovered answer")],
        )


class _FakeTavily:
    def __init__(self, **kwargs):
        pass

    def extract(self, **kwargs):
        return {
            "results": [{"url": "https://example.test", "raw_content": "x" * 100}],
            "failed_results": [],
            "failed_urls": [],
        }


class LogicRegressionTests(unittest.TestCase):
    def test_anthropic_sdk_retries_are_disabled(self):
        self.assertEqual(conversation_loop.client.max_retries, 0)

    def test_normal_end_turn_does_not_request_summary(self):
        fake_client = SimpleNamespace(messages=_FakeMessagesAPI())
        messages = [{"role": "user", "content": "hello"}]
        fake_tracer = _FakeTracer()

        with (
            patch.object(conversation_loop, "client", fake_client),
            patch.object(conversation_loop, "reset_tracer", return_value=fake_tracer),
        ):
            conversation_loop.agent_loop(messages)

        self.assertEqual(len(fake_client.messages.calls), 1)
        self.assertEqual(messages[-1]["content"][0].text, "done")

    def test_permanent_provider_error_returns_controlled_assistant_message(self):
        class AuthenticationError(Exception):
            status_code = 401
            response = None

        class FailingMessagesAPI:
            def create(self, **_kwargs):
                raise AuthenticationError("invalid API key")

        messages = [{"role": "user", "content": "hello"}]
        fake_client = SimpleNamespace(messages=FailingMessagesAPI())
        with (
            patch.object(conversation_loop, "client", fake_client),
            patch.object(conversation_loop, "reset_tracer", return_value=_FakeTracer()),
        ):
            conversation_loop.agent_loop(messages)

        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIn("authentication", messages[-1]["content"][0].text)
        self.assertIn("HTTP 401", messages[-1]["content"][0].text)

    def test_bash_nonzero_exit_is_failure(self):
        self.assertEqual(
            classify_tool_failure("bash", '{"exit_code": 1, "stderr": "bad"}'),
            (True, " [exit 1]"),
        )
        self.assertEqual(
            classify_tool_failure("bash", '{"exit_code": 0, "stdout": "ok"}'),
            (False, ""),
        )

    def test_outcome_classifies_execution_and_policy_states(self):
        self.assertEqual(classify_tool_result("bash", '{"exit_code": 2}').status, FAILED)
        self.assertEqual(classify_tool_result("read_file", '{"content": "ok"}').status, SUCCESS)
        self.assertEqual(
            classify_tool_result("read_file", "Error: bad handler", status=INTERNAL_ERROR).status,
            INTERNAL_ERROR,
        )
        self.assertEqual(
            classify_tool_result("write_file", "denied", status=BLOCKED).status,
            BLOCKED,
        )

    def test_empty_tool_results_are_explicit_successes(self):
        for content in (None, "", "   \n"):
            outcome = classify_tool_result("read_file", content)
            self.assertEqual(outcome.status, SUCCESS)
            self.assertEqual(outcome.content, "(no output)")

    def test_empty_file_mutation_result_is_failure(self):
        for tool_name in ("write_file", "patch"):
            outcome = classify_tool_result(tool_name, None)
            self.assertEqual(outcome.status, FAILED)
            self.assertEqual(outcome.error_code, "empty_mutation_result")

        tracker = FileMutationTracker()
        outcome = classify_tool_result("write_file", None)
        tracker.record(outcome, {"path": "empty-result.txt"})
        self.assertEqual(len(tracker.unresolved_failures()), 1)

    def test_cancelled_outcome_does_not_change_guardrail_counts(self):
        controller = tool_executor.tool_guardrails
        controller.reset_for_turn()
        args = {"path": "missing.txt"}
        controller.after_call("read_file", args, '{"error": "missing"}', failed=True)
        counts_before = dict(controller._exact_failure_counts)

        with patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()):
            outcome = tool_executor._finalize_outcome(
                tool_executor._cancelled_outcome("read_file", "test interrupt"),
                args,
            )

        self.assertEqual(outcome.status, CANCELLED)
        self.assertEqual(controller._exact_failure_counts, counts_before)

    def test_sequential_interrupt_cancels_all_remaining_handlers(self):
        interrupted = False
        executions = 0

        def handler():
            nonlocal interrupted, executions
            executions += 1
            interrupted = True
            return '{"success": true}'

        previous = registry.get_entry("cancel_probe")
        registry.register(
            "cancel_probe",
            {"name": "cancel_probe", "input_schema": {"type": "object"}},
            handler,
        )
        calls = [
            SimpleNamespace(type="tool_use", id=f"cancel-{i}", name="cancel_probe", input={})
            for i in range(3)
        ]
        messages = []
        try:
            with (
                patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "check_tool_permission", return_value=None),
            ):
                tool_executor.execute_tool_calls(
                    calls,
                    messages,
                    is_cancelled=lambda: interrupted,
                )
        finally:
            if previous is None:
                registry._tools.pop("cancel_probe", None)
            else:
                registry._tools["cancel_probe"] = previous

        self.assertEqual(executions, 1)
        self.assertEqual(len(messages[-1]["content"]), 3)
        self.assertNotIn("cancelled", messages[-1]["content"][0]["content"])
        self.assertIn("cancelled", messages[-1]["content"][1]["content"])
        self.assertIn("cancelled", messages[-1]["content"][2]["content"])

    def test_concurrent_interrupt_only_cancels_pending_handler(self):
        started = threading.Event()
        release = threading.Event()
        second_executed = False

        def running_handler():
            started.set()
            self.assertTrue(release.wait(1))
            return '{"content": "finished"}'

        def pending_handler():
            nonlocal second_executed
            second_executed = True
            return '{"results": ["unexpected"]}'

        previous_read = registry.get_entry("read_file")
        previous_search = registry.get_entry("search_files")
        registry.register(
            "read_file",
            {"name": "read_file", "input_schema": {"type": "object"}},
            running_handler,
        )
        registry.register(
            "search_files",
            {"name": "search_files", "input_schema": {"type": "object"}},
            pending_handler,
        )
        calls = [
            SimpleNamespace(type="tool_use", id="running", name="read_file", input={}),
            SimpleNamespace(type="tool_use", id="pending", name="search_files", input={}),
        ]
        messages = []

        def is_cancelled():
            if started.is_set():
                release.set()
                return True
            return False

        try:
            with (
                patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "check_tool_permission", return_value=None),
            ):
                tool_executor.execute_tool_calls(
                    calls,
                    messages,
                    concurrent=True,
                    max_workers=1,
                    is_cancelled=is_cancelled,
                )
        finally:
            registry._tools["read_file"] = previous_read
            registry._tools["search_files"] = previous_search

        self.assertFalse(second_executed)
        self.assertIn("finished", messages[-1]["content"][0]["content"])
        self.assertIn("cancelled", messages[-1]["content"][1]["content"])

    def test_post_tool_empty_response_is_retried_once(self):
        fake_api = _PostToolEmptyThenDoneAPI()
        previous = registry.get_entry("empty_probe")
        registry.register(
            "empty_probe",
            {"name": "empty_probe", "input_schema": {"type": "object"}},
            lambda: None,
        )
        try:
            with (
                patch.object(conversation_loop, "client", SimpleNamespace(messages=fake_api)),
                patch.object(conversation_loop, "reset_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "check_tool_permission", return_value=None),
            ):
                messages = [{"role": "user", "content": "run empty probe"}]
                conversation_loop.agent_loop(messages)
        finally:
            if previous is None:
                registry._tools.pop("empty_probe", None)
            else:
                registry._tools["empty_probe"] = previous

        self.assertEqual(len(fake_api.calls), 3)
        self.assertEqual(messages[-1]["content"][0].text, "recovered answer")
        self.assertIn("(no output)", str(fake_api.calls[1]["messages"]))
        self.assertIn("previous response was empty", str(fake_api.calls[2]["messages"]))

    def test_repeated_post_tool_empty_response_stops_cleanly(self):
        fake_api = _PostToolEmptyThenDoneAPI(stay_empty=True)
        previous = registry.get_entry("empty_probe")
        registry.register(
            "empty_probe",
            {"name": "empty_probe", "input_schema": {"type": "object"}},
            lambda: None,
        )
        try:
            with (
                patch.object(conversation_loop, "client", SimpleNamespace(messages=fake_api)),
                patch.object(conversation_loop, "reset_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "check_tool_permission", return_value=None),
            ):
                messages = [{"role": "user", "content": "run empty probe"}]
                conversation_loop.agent_loop(messages)
        finally:
            if previous is None:
                registry._tools.pop("empty_probe", None)
            else:
                registry._tools["empty_probe"] = previous

        self.assertEqual(len(fake_api.calls), 3)
        self.assertIn("连续返回空回复", messages[-1]["content"][0].text)

    def test_guardrail_environment_switches(self):
        with patch.dict(
            os.environ,
            {"TOOL_GUARDRAIL_WARNINGS": "off", "TOOL_GUARDRAIL_HARD_STOP": "false"},
        ):
            config = ToolCallGuardrailConfig.from_environment()
        self.assertFalse(config.warnings_enabled)
        self.assertFalse(config.hard_stop_enabled)

    def test_dotenv_guardrail_switch_is_loaded_before_executor_import(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            Path(task_tmp, ".env").write_text(
                "ANTHROPIC_API_KEY=test-key\n"
                "MODEL_ID=test-model\n"
                "TOOL_GUARDRAIL_HARD_STOP=false\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.pop("TOOL_GUARDRAIL_HARD_STOP", None)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from agent.conversation_loop import tool_guardrails; "
                    "print(tool_guardrails.config.hard_stop_enabled)",
                ],
                cwd=task_tmp,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(result.stdout.strip(), "False")

    def test_permission_block_does_not_increment_guardrail(self):
        call = SimpleNamespace(type="tool_use", id="blocked-1", name="read_file", input={"path": "x.py"})
        messages = []
        fake_tracer = _FakeTracer()
        tool_executor.tool_guardrails.reset_for_turn()
        with (
            patch.object(tool_executor, "check_tool_permission", return_value="denied by policy"),
            patch.object(tool_executor, "get_tracer", return_value=fake_tracer),
        ):
            tool_executor.execute_tool_calls([call], messages)
        self.assertEqual(tool_executor.tool_guardrails._exact_failure_counts, {})
        self.assertIn("denied by policy", messages[-1]["content"][0]["content"])

    def test_dispatch_compatibility_entry_keeps_permission_boundary(self):
        executed = False

        def handler(**_kwargs):
            nonlocal executed
            executed = True
            return '{"content": "should not run"}'

        previous = registry.get_entry("read_file")
        registry.register(
            "read_file",
            {"name": "read_file", "input_schema": {"type": "object"}},
            handler,
        )
        tool_executor.tool_guardrails.reset_for_turn()
        try:
            with (
                patch.object(tool_executor, "check_tool_permission", return_value="denied"),
                patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
            ):
                result = tool_executor.dispatch_tool_call("read_file", {"path": "x.py"})
        finally:
            registry._tools["read_file"] = previous
        self.assertFalse(executed)
        self.assertEqual(result, "denied")
        self.assertEqual(tool_executor.tool_guardrails._exact_failure_counts, {})

    def test_unknown_tool_retries_are_counted_per_model_round(self):
        class InvalidRoundsAPI:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                index = len(self.calls)
                return SimpleNamespace(
                    stop_reason="tool_use",
                    content=[ToolUseBlock(
                        type="tool_use",
                        id=f"invalid-round-{index}",
                        name=f"not_registered_{index}",
                        input={},
                    )],
                )

        api = InvalidRoundsAPI()
        messages = [{"role": "user", "content": "use a missing tool"}]
        with (
            patch.object(conversation_loop, "client", SimpleNamespace(messages=api)),
            patch.object(conversation_loop, "reset_tracer", return_value=_FakeTracer()),
        ):
            conversation_loop.agent_loop(messages)

        self.assertEqual(len(api.calls), 3)
        self.assertIn("连续三轮调用未知工具", messages[-1]["content"][0].text)
        self.assertIsNone(tool_executor.tool_guardrails.halt_decision)

    def test_multiple_unknown_tools_in_one_batch_only_consume_one_retry(self):
        class OneInvalidBatchAPI:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return SimpleNamespace(
                        stop_reason="tool_use",
                        content=[ToolUseBlock(
                            type="tool_use",
                            id=f"unknown-{index}",
                            name=f"not_registered_{index}",
                            input={},
                        ) for index in range(3)],
                    )
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[TextBlock(type="text", text="recovered")],
                )

        api = OneInvalidBatchAPI()
        messages = [{"role": "user", "content": "use tools"}]
        with (
            patch.object(conversation_loop, "client", SimpleNamespace(messages=api)),
            patch.object(conversation_loop, "reset_tracer", return_value=_FakeTracer()),
        ):
            conversation_loop.agent_loop(messages)

        self.assertEqual(len(api.calls), 2)
        self.assertEqual(messages[-1]["content"][0].text, "recovered")

    def test_invalid_tool_batch_skips_valid_handlers(self):
        executed = False

        def handler():
            nonlocal executed
            executed = True
            return '{"content": "unexpected"}'

        calls = [
            ToolUseBlock(type="tool_use", id="valid", name="mixed_valid_probe", input={}),
            ToolUseBlock(type="tool_use", id="invalid", name="not_registered", input={}),
        ]
        previous = registry.get_entry("mixed_valid_probe")
        registry.register(
            "mixed_valid_probe",
            {"name": "mixed_valid_probe", "input_schema": {"type": "object"}},
            handler,
        )
        try:
            results = conversation_loop._invalid_tool_results(calls)
        finally:
            if previous is None:
                registry._tools.pop("mixed_valid_probe", None)
            else:
                registry._tools["mixed_valid_probe"] = previous

        self.assertFalse(executed)
        self.assertIsNotNone(results)
        self.assertIn("Skipped", results[0]["content"])
        self.assertIn("Unknown tool", results[1]["content"])

    def test_valid_tool_round_resets_unknown_tool_retries(self):
        class ResettingAPI:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                index = len(self.calls)
                if index in {1, 3, 4}:
                    return SimpleNamespace(
                        stop_reason="tool_use",
                        content=[ToolUseBlock(
                            type="tool_use",
                            id=f"invalid-{index}",
                            name="not_registered",
                            input={},
                        )],
                    )
                if index == 2:
                    return SimpleNamespace(
                        stop_reason="tool_use",
                        content=[ToolUseBlock(
                            type="tool_use",
                            id="valid-reset",
                            name="reset_probe",
                            input={},
                        )],
                    )
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[TextBlock(type="text", text="done")],
                )

        api = ResettingAPI()
        previous = registry.get_entry("reset_probe")
        registry.register(
            "reset_probe",
            {"name": "reset_probe", "input_schema": {"type": "object"}},
            lambda: '{"success": true}',
        )
        try:
            with (
                patch.object(conversation_loop, "client", SimpleNamespace(messages=api)),
                patch.object(conversation_loop, "reset_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
                patch.object(tool_executor, "check_tool_permission", return_value=None),
            ):
                messages = [{"role": "user", "content": "retry tools"}]
                conversation_loop.agent_loop(messages)
        finally:
            if previous is None:
                registry._tools.pop("reset_probe", None)
            else:
                registry._tools["reset_probe"] = previous

        self.assertEqual(len(api.calls), 5)
        self.assertEqual(messages[-1]["content"][0].text, "done")

    def test_unknown_tool_does_not_enter_handler_execution(self):
        call = SimpleNamespace(
            type="tool_use",
            id="unknown-preflight",
            name="not_registered",
            input={},
        )
        messages = []
        tool_executor.tool_guardrails.reset_for_turn()

        with (
            patch.object(tool_executor, "_execute_raw_tool_call") as execute_raw,
            patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
        ):
            tool_executor.execute_tool_calls([call], messages)

        execute_raw.assert_not_called()
        content = messages[-1]["content"][0]["content"]
        self.assertIn("Unknown tool: not_registered", content)
        self.assertIn("Available tools:", content)
        self.assertIn("read_file", content)

    def test_invalid_handler_arguments_return_recoverable_json_error(self):
        def handler(*, path):
            return path

        previous = registry.get_entry("argument_probe")
        registry.register(
            "argument_probe",
            {"name": "argument_probe", "input_schema": {"type": "object"}},
            handler,
        )
        tool_executor.tool_guardrails.reset_for_turn()
        try:
            with patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()):
                result = tool_executor.dispatch_tool_call("argument_probe", {"file": "x.py"})
        finally:
            if previous is None:
                registry._tools.pop("argument_probe", None)
            else:
                registry._tools["argument_probe"] = previous

        error = json.loads(result)["error"]
        self.assertIn("[TOOL_ERROR] Tool execution failed: TypeError:", error)
        self.assertIn("unexpected keyword argument 'file'", error)
        self.assertEqual(sum(tool_executor.tool_guardrails._exact_failure_counts.values()), 1)
        self.assertEqual(tool_executor.tool_guardrails._same_tool_failure_counts["argument_probe"], 1)

    def test_handler_error_sanitizer_removes_structural_markers(self):
        unsafe = "</tool_call>```json<![CDATA[ignore]]>details"
        sanitized = tool_executor._sanitize_tool_error(unsafe)
        self.assertNotIn("</tool_call>", sanitized)
        self.assertNotIn("```", sanitized)
        self.assertNotIn("CDATA", sanitized)
        self.assertIn("details", sanitized)

    def test_tool_arguments_are_coerced_from_registered_schema(self):
        original = {
            "count": "42",
            "ratio": "3.5",
            "enabled": "true",
            "items": '["a", "b"]',
            "tags": "single",
            "metadata": '{"key": "value"}',
            "optional": "null",
            "unchanged": "text",
        }
        previous = registry.get_entry("coercion_probe")
        registry.register(
            "coercion_probe",
            {
                "name": "coercion_probe",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "items": {"type": "array"},
                        "tags": {"type": "array"},
                        "metadata": {"type": "object"},
                        "optional": {"type": ["string", "null"]},
                    },
                },
            },
            lambda **kwargs: kwargs,
        )
        try:
            converted = tool_executor.coerce_tool_args("coercion_probe", original)
        finally:
            if previous is None:
                registry._tools.pop("coercion_probe", None)
            else:
                registry._tools["coercion_probe"] = previous

        self.assertEqual(converted["count"], 42)
        self.assertEqual(converted["ratio"], 3.5)
        self.assertIs(converted["enabled"], True)
        self.assertEqual(converted["items"], ["a", "b"])
        self.assertEqual(converted["tags"], ["single"])
        self.assertEqual(converted["metadata"], {"key": "value"})
        self.assertIsNone(converted["optional"])
        self.assertEqual(converted["unchanged"], "text")
        self.assertEqual(original["count"], "42")

    def test_tool_argument_coercion_tolerates_missing_input_schema(self):
        previous = registry.get_entry("invalid_schema_probe")
        registry.register(
            "invalid_schema_probe",
            {"name": "invalid_schema_probe", "input_schema": None},
            lambda **kwargs: kwargs,
        )
        try:
            args = {"count": "7"}
            self.assertEqual(
                tool_executor.coerce_tool_args("invalid_schema_probe", args),
                args,
            )
        finally:
            if previous is None:
                registry._tools.pop("invalid_schema_probe", None)
            else:
                registry._tools["invalid_schema_probe"] = previous

    def test_coerced_arguments_reach_permission_handler_and_guardrail(self):
        seen = {}

        def handler(*, count):
            seen["handler"] = count
            return '{"success": true}'

        def permission(_name, args):
            seen["permission"] = args["count"]
            return None

        previous = registry.get_entry("coercion_integration_probe")
        registry.register(
            "coercion_integration_probe",
            {
                "name": "coercion_integration_probe",
                "input_schema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            },
            handler,
        )
        tool_executor.tool_guardrails.reset_for_turn()
        try:
            with (
                patch.object(tool_executor, "check_tool_permission", side_effect=permission),
                patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()),
            ):
                tool_executor.dispatch_tool_call("coercion_integration_probe", {"count": "7"})
        finally:
            if previous is None:
                registry._tools.pop("coercion_integration_probe", None)
            else:
                registry._tools["coercion_integration_probe"] = previous

        self.assertEqual(seen, {"permission": 7, "handler": 7})
        self.assertEqual(tool_executor.tool_guardrails._same_tool_failure_counts, {})

    def test_json_equivalent_results_count_as_no_progress(self):
        controller = ToolCallGuardrailController()
        args = {"path": "x.py"}
        controller.after_call("read_file", args, '{"count": 0, "results": []}', failed=False)
        controller.after_call("read_file", args, '{ "results": [], "count": 0 }', failed=False)
        signature = next(iter(controller._no_progress))
        self.assertEqual(controller._no_progress[signature][1], 2)

    def test_parallel_results_are_finalized_in_model_order(self):
        started = threading.Event()
        release = threading.Event()
        trace_names = []
        classification_threads = []
        original_classifier = tool_executor.classify_tool_result

        class RecordingTracer(_FakeTracer):
            def tool_result(self, name, *_args, **_kwargs):
                trace_names.append(name)

        def slow_read(**_kwargs):
            started.set()
            self.assertTrue(release.wait(1))
            return '{"content": "slow"}'

        def fast_search(**_kwargs):
            self.assertTrue(started.wait(1))
            release.set()
            return '{"results": ["fast"]}'

        def record_classification(*args, **kwargs):
            classification_threads.append(threading.current_thread().name)
            return original_classifier(*args, **kwargs)

        old_read = registry.get_entry("read_file")
        old_search = registry.get_entry("search_files")
        registry.register("read_file", {"name": "read_file", "input_schema": {"type": "object"}}, slow_read)
        registry.register("search_files", {"name": "search_files", "input_schema": {"type": "object"}}, fast_search)
        calls = [
            SimpleNamespace(type="tool_use", id="slow", name="read_file", input={"path": "x"}),
            SimpleNamespace(type="tool_use", id="fast", name="search_files", input={"query": "x"}),
        ]
        try:
            with (
                patch.object(tool_executor, "get_tracer", return_value=RecordingTracer()),
                patch.object(tool_executor, "classify_tool_result", side_effect=record_classification),
            ):
                messages = []
                tool_executor.execute_tool_calls(calls, messages, concurrent=True)
        finally:
            registry._tools["read_file"] = old_read
            registry._tools["search_files"] = old_search
        self.assertEqual(trace_names, ["read_file", "search_files"])
        self.assertEqual(classification_threads, ["MainThread", "MainThread"])
        self.assertIn("slow", messages[-1]["content"][0]["content"])
        self.assertIn("fast", messages[-1]["content"][1]["content"])

    def test_file_mutation_tracker_clears_recovered_path(self):
        tracker = FileMutationTracker()
        failed = classify_tool_result("patch", '{"error": "old string missing"}')
        recovered = classify_tool_result("patch", '{"success": true, "diff": ""}')
        tracker.record(failed, {"path": "src/a.py"})
        self.assertTrue(tracker.unresolved_failures())
        tracker.record(recovered, {"path": "src/a.py"})
        self.assertEqual(tracker.unresolved_failures(), [])

    def test_file_mutation_tracker_resolves_symlink_aliases(self):
        tracker = FileMutationTracker()
        failed = classify_tool_result("patch", '{"error": "old string missing"}')
        recovered = classify_tool_result("patch", '{"success": true, "diff": ""}')
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            real_dir = root / "real"
            real_dir.mkdir()
            alias = root / "alias"
            alias.symlink_to(real_dir, target_is_directory=True)
            tracker.record(failed, {"path": str(real_dir / "a.py")})
            tracker.record(recovered, {"path": str(alias / "a.py")})
        self.assertEqual(tracker.unresolved_failures(), [])

    def test_final_answer_exposes_unrecovered_file_failure(self):
        fake_client = SimpleNamespace(messages=_FailingPatchThenDoneAPI())
        fake_tracer = _FakeTracer()
        previous = registry.get_entry("patch")
        registry.register(
            "patch", {"name": "patch", "input_schema": {"type": "object"}},
            lambda **_kwargs: '{"error": "old string missing"}',
        )
        try:
            with (
                patch.object(conversation_loop, "client", fake_client),
                patch.object(conversation_loop, "reset_tracer", return_value=fake_tracer),
                patch.object(tool_executor, "get_tracer", return_value=fake_tracer),
                patch.object(tool_executor, "check_tool_permission", return_value=None),
            ):
                messages = [{"role": "user", "content": "edit a.py"}]
                conversation_loop.agent_loop(messages)
        finally:
            registry._tools["patch"] = previous
        self.assertEqual(len(fake_client.messages.calls), 2)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIn("以下文件修改没有成功完成", messages[-1]["content"][-1].text)

    def test_guardrail_halt_returns_controlled_response_without_summary(self):
        fake_client = SimpleNamespace(messages=_SingleToolBatchAPI())
        fake_tracer = _FakeTracer()
        messages = [{"role": "user", "content": "run failing tool"}]
        executions = 0

        def fail_probe(**_kwargs):
            nonlocal executions
            executions += 1
            return '{"error": "boom"}'

        previous = registry.get_entry("probe_fail")
        registry.register(
            "probe_fail",
            {"name": "probe_fail", "input_schema": {"type": "object"}},
            fail_probe,
        )
        try:
            with (
                patch.object(conversation_loop, "client", fake_client),
                patch.object(conversation_loop, "reset_tracer", return_value=fake_tracer),
                patch.object(tool_executor, "get_tracer", return_value=fake_tracer),
            ):
                conversation_loop.agent_loop(messages)
        finally:
            if previous is None:
                registry._tools.pop("probe_fail", None)
            else:
                registry._tools["probe_fail"] = previous

        self.assertEqual(len(fake_client.messages.calls), 1)
        self.assertEqual(executions, 5)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIn("工具调用已停止", messages[-1]["content"][0].text)

    def test_turn_budget_is_a_hard_cap(self):
        results = [
            {"content": "123456789"},
            {"content": "abcdefghij"},
            {"content": "xyz"},
        ]
        with patch.object(tool_executor, "TURN_BUDGET_CHARS", 10):
            tool_executor._enforce_turn_budget(results)
        self.assertLessEqual(sum(len(item["content"]) for item in results), 10)

    def test_read_past_eof_returns_empty_content(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            path = Path(task_tmp) / "one.txt"
            path.write_text("only line", encoding="utf-8")
            result = json.loads(read_file(str(path), offset=10, limit=5))
        self.assertEqual(result["content"], "")
        self.assertEqual(result["total_lines"], 1)

    def test_web_extract_enforces_max_chars_for_short_content(self):
        with patch.object(web_extract_module, "TavilyClient", _FakeTavily):
            result = json.loads(
                web_extract_module.web_extract(["https://example.test"], max_chars=10)
            )
        self.assertLessEqual(len(result["results"][0]["content"]), 10)
        self.assertTrue(result["results"][0]["truncated"])

    def test_root_relative_sensitive_files_are_blocked(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as task_tmp:
            old_cwd = os.getcwd()
            os.chdir(task_tmp)
            try:
                self.assertIsNotNone(check_write_path(".env"))
                self.assertIsNotNone(check_write_path("config.yaml"))
                self.assertIsNone(check_write_path(".env.example"))
            finally:
                os.chdir(old_cwd)

    def test_symlink_to_sensitive_user_path_is_blocked(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as task_tmp:
            root = Path(task_tmp)
            fake_home = root / "home"
            work = root / "work"
            ssh_dir = fake_home / ".ssh"
            ssh_dir.mkdir(parents=True)
            work.mkdir()
            (work / "alias").symlink_to(ssh_dir, target_is_directory=True)
            old_cwd = os.getcwd()
            os.chdir(work)
            try:
                with patch.dict(os.environ, {"HOME": str(fake_home)}):
                    self.assertIsNotNone(check_write_path("alias/authorized_keys"))
            finally:
                os.chdir(old_cwd)

    def test_sudo_stdin_is_blocked_even_if_password_env_exists(self):
        with patch.dict(os.environ, {"SUDO_PASSWORD": "configured"}):
            blocked, _ = _check_sudo_stdin_guard("sudo -S whoami")
        self.assertTrue(blocked)

    def test_trace_paths_are_unique(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            with patch.object(tracer_module, "TRACE_DIR", task_tmp):
                first = tracer_module.Tracer()
                second = tracer_module.Tracer()
        self.assertNotEqual(first.path, second.path)

    def test_trace_records_outcome_status(self):
        outcome = classify_tool_result("bash", '{"exit_code": 3, "stderr": "bad"}')
        with tempfile.TemporaryDirectory() as task_tmp:
            with patch.object(tracer_module, "TRACE_DIR", task_tmp):
                trace = tracer_module.Tracer()
                trace.tool_result("bash", outcome.content, outcome=outcome)
                record = json.loads(Path(trace.path).read_text(encoding="utf-8"))
        self.assertEqual(record["status"], FAILED)
        self.assertEqual(record["error_code"], "nonzero_exit")


if __name__ == "__main__":
    unittest.main()
