"""LocalEnvironment 行为契约。"""

import os
import tempfile
import unittest
from pathlib import Path

from tools.environments.local import LocalEnvironment


class LocalEnvironmentTests(unittest.TestCase):
    def test_stdout_stderr_and_exit_code(self):
        result = LocalEnvironment().execute(
            "printf out; printf err >&2; exit 7",
            timeout=5,
        )
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.stdout, "out")
        self.assertEqual(result.stderr, "err")

    def test_command_uses_process_cwd_and_shell_state_is_not_persistent(self):
        environment = LocalEnvironment()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            previous = Path.cwd()
            os.chdir(tmp)
            try:
                first = environment.execute("pwd; export ATHENA_EPHEMERAL=yes", timeout=5)
                second = environment.execute("printf %s \"${ATHENA_EPHEMERAL-unset}\"", timeout=5)
            finally:
                os.chdir(previous)
        self.assertEqual(first.stdout.splitlines()[0], tmp)
        self.assertEqual(second.stdout, "unset")

    def test_timeout_is_explicit(self):
        with self.assertRaisesRegex(TimeoutError, r"Timeout \(1s\)"):
            LocalEnvironment().execute("sleep 5", timeout=1)

    def test_close_is_idempotent(self):
        environment = LocalEnvironment()
        environment.close()
        environment.close()


if __name__ == "__main__":
    unittest.main()
