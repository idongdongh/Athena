"""DockerEnvironment 的 Docker CLI 单元测试，不要求 daemon。"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from athena_cli.config import DockerSettings
from agent.interrupt_controller import ToolExecutionCancelled, interrupt_controller
from tools.environments.docker import DockerEnvironment, find_container_runtime


class _Completed(SimpleNamespace):
    returncode: int
    stdout: str
    stderr: str


def completed(returncode=0, stdout="", stderr=""):
    return _Completed(returncode=returncode, stdout=stdout, stderr=stderr)


class _FakePopen:
    last_args = None

    def __init__(self, args, *, stdout, stderr, **_kwargs):
        self.args = args
        type(self).last_args = args
        self.returncode = 0
        stdout.write(b"container-out")
        stderr.write(b"container-err")

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


class DockerEnvironmentTests(unittest.TestCase):
    def test_runtime_auto_prefers_docker_then_podman(self):
        with patch("tools.environments.docker.shutil.which", side_effect=lambda name: f"/bin/{name}" if name == "podman" else None):
            self.assertEqual(find_container_runtime("auto"), "/bin/podman")
        with patch("tools.environments.docker.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "找不到容器运行时"):
                find_container_runtime("docker")

    def _new_environment(self, root: Path, settings: DockerSettings | None = None):
        with (
            patch("tools.environments.docker.find_container_runtime", return_value="/usr/bin/docker"),
            patch("tools.environments.docker.subprocess.run", return_value=completed()),
        ):
            return DockerEnvironment(settings or DockerSettings(), root)

    def test_daemon_probe_failure_is_explicit(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tools.environments.docker.find_container_runtime", return_value="/usr/bin/docker"),
            patch("tools.environments.docker.subprocess.run", return_value=completed(1, stderr="daemon down")),
        ):
            with self.assertRaisesRegex(RuntimeError, "daemon down"):
                DockerEnvironment(DockerSettings(), Path(tmp))

    def test_container_run_uses_hardened_args_and_same_workspace_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            environment = self._new_environment(root)
            calls = []

            def control(args, *, timeout=30):
                calls.append(args)
                if args[0] == "run":
                    return completed(stdout="container-id\n")
                return completed()

            environment._run_control = control
            self.assertEqual(environment._ensure_container(), "container-id")
            run = calls[0]
            self.assertIn("--cap-drop", run)
            self.assertIn("ALL", run)
            self.assertIn("no-new-privileges", run)
            self.assertIn("--network", run)
            self.assertIn("none", run)
            self.assertIn("--pids-limit", run)
            self.assertIn("--memory", run)
            self.assertIn("--cpus", run)
            if hasattr(os, "getuid"):
                self.assertIn("--user", run)
            self.assertIn(f"{root}:{root}:rw", run)
            self.assertEqual(run[run.index("--workdir") + 1], str(root))
            joined = " ".join(run)
            self.assertNotIn("docker.sock", joined)
            self.assertNotIn("ANTHROPIC_API_KEY", joined)

    def test_read_only_and_network_enabled_adjust_run_args(self):
        settings = DockerSettings(network=True, workspace_read_only=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            environment = self._new_environment(root, settings)
            calls = []

            def control(args, *, timeout=30):
                calls.append(args)
                return completed(stdout="cid") if args[0] == "run" else completed()

            environment._run_control = control
            environment._ensure_container()
            run = calls[0]
            self.assertNotIn("--network", run)
            self.assertIn(f"{root}:{root}:ro", run)

    def test_create_failure_removes_partial_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = self._new_environment(Path(tmp))
            environment._run_control = Mock(return_value=completed(125, stderr="bad image"))
            with patch.object(environment, "_remove_by_name") as cleanup:
                with self.assertRaisesRegex(RuntimeError, "bad image"):
                    environment._ensure_container()
            cleanup.assert_called_once()

    def test_exec_passes_user_command_as_argument_not_host_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = self._new_environment(Path(tmp))
            command = "printf '%s\\n' \"hello world\"; echo done"
            with (
                patch.object(environment, "_ensure_container", return_value="cid"),
                patch("tools.environments.docker.subprocess.Popen", _FakePopen),
            ):
                result = environment.execute(command, timeout=5)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "container-out")
            self.assertEqual(result.stderr, "container-err")
            self.assertIn(command, _FakePopen.last_args)
            self.assertNotIn("shell=True", repr(_FakePopen.last_args))

    def test_timeout_kills_container_process_group(self):
        class BlockingPopen(_FakePopen):
            def __init__(self, args, *, stdout, stderr, **kwargs):
                super().__init__(args, stdout=stdout, stderr=stderr, **kwargs)
                self.returncode = None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(self.args, timeout)

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            environment = self._new_environment(Path(tmp))
            with (
                patch.object(environment, "_ensure_container", return_value="cid"),
                patch.object(environment, "_kill_container_process_group", return_value=True) as kill,
                patch("tools.environments.docker._stop_host_process"),
                patch("tools.environments.docker.subprocess.Popen", BlockingPopen),
            ):
                with self.assertRaisesRegex(TimeoutError, "Timeout"):
                    environment.execute("sleep 30", timeout=0.001)
            kill.assert_called_once()

    def test_interrupt_kills_container_process_group_and_preserves_cancelled_status(self):
        started = threading.Event()

        class BlockingPopen(_FakePopen):
            def __init__(self, args, *, stdout, stderr, **kwargs):
                super().__init__(args, stdout=stdout, stderr=stderr, **kwargs)
                self.returncode = None
                started.set()

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(self.args, timeout)

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            environment = self._new_environment(Path(tmp))
            result = {}
            with (
                patch.object(environment, "_ensure_container", return_value="cid"),
                patch.object(environment, "_kill_container_process_group", return_value=True) as kill,
                patch("tools.environments.docker._stop_host_process"),
                patch("tools.environments.docker.subprocess.Popen", BlockingPopen),
            ):
                def run():
                    try:
                        environment.execute("sleep 30", timeout=30)
                    except Exception as exc:
                        result["error"] = exc

                worker = threading.Thread(target=run)
                worker.start()
                self.assertTrue(started.wait(1))
                interrupt_controller.request()
                worker.join(2)
                interrupt_controller.clear()

            self.assertFalse(worker.is_alive())
            self.assertIsInstance(result.get("error"), ToolExecutionCancelled)
            kill.assert_called_once()

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = self._new_environment(Path(tmp))
            environment._container_id = "cid"
            with patch.object(environment, "_run_control", return_value=completed()) as control:
                environment.close()
                environment.close()
            control.assert_called_once_with(["rm", "-f", "cid"], timeout=15)


if __name__ == "__main__":
    unittest.main()
