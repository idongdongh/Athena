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

    def test_unknown_tool_is_guarded_as_failure(self):
        calls = [
            SimpleNamespace(type="tool_use", id=f"unknown-{i}", name=f"not_registered_{i}", input={})
            for i in range(3)
        ]
        messages = []
        tool_executor.tool_guardrails.reset_for_turn()
        with patch.object(tool_executor, "get_tracer", return_value=_FakeTracer()):
            tool_executor.execute_tool_calls(calls, messages)
        self.assertIsNotNone(tool_executor.tool_guardrails.halt_decision)
        self.assertEqual(tool_executor.tool_guardrails.halt_decision.code, "invalid_tool_halt")
        self.assertEqual(tool_executor.tool_guardrails._exact_failure_counts, {})
        self.assertEqual(tool_executor.tool_guardrails._same_tool_failure_counts, {})
        self.assertIn("invalid_tool_halt", messages[-1]["content"][-1]["content"])

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

    def test_reset_for_turn_clears_unknown_tool_count(self):
        controller = ToolCallGuardrailController()
        controller.record_unknown_tool("missing")
        controller.reset_for_turn()
        self.assertEqual(controller._invalid_tool_count, 0)
        self.assertIsNone(controller.halt_decision)

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
