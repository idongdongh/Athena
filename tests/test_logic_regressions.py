"""核心调用链的逻辑回归测试（仅使用标准库 unittest）。"""

import json
import os
import tempfile
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
from agent.tool_guardrails import is_tool_failure
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
        self.assertTrue(is_tool_failure("bash", '{"exit_code": 1, "stderr": "bad"}'))
        self.assertFalse(is_tool_failure("bash", '{"exit_code": 0, "stdout": "ok"}'))

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


if __name__ == "__main__":
    unittest.main()
