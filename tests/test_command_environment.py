"""进程级命令环境 manager 测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from athena_cli.config import TerminalSettings
from tools.command_environment import CommandEnvironmentManager
from tools.environments.base import CommandResult


class CommandEnvironmentManagerTests(unittest.TestCase):
    def test_environment_is_lazy_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CommandEnvironmentManager(TerminalSettings(), Path(tmp))
            environment = Mock()
            environment.execute.return_value = CommandResult(0, "ok", "")
            with patch.object(manager, "_create_environment", return_value=environment) as create:
                first = manager.execute("one")
                second = manager.execute("two", timeout=5)
            create.assert_called_once()
            self.assertEqual(first.stdout, "ok")
            self.assertEqual(second.stdout, "ok")
            environment.execute.assert_any_call("one", timeout=120)
            environment.execute.assert_any_call("two", timeout=5)

    def test_close_detaches_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CommandEnvironmentManager(TerminalSettings(), Path(tmp))
            environment = Mock()
            manager._environment = environment
            manager.close()
            manager.close()
            environment.close.assert_called_once()
            self.assertIsNone(manager._environment)

    def test_invalid_call_timeout_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CommandEnvironmentManager(TerminalSettings(), Path(tmp))
            for timeout in (0, -1, True, 1.5):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    manager.execute("true", timeout=timeout)


if __name__ == "__main__":
    unittest.main()
