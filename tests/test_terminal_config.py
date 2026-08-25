"""本地命令沙箱配置测试。"""

import unittest

from athena_cli.config import DockerSettings, TerminalSettings


class TerminalSettingsTests(unittest.TestCase):
    def test_defaults_preserve_local_backend(self):
        settings = TerminalSettings.from_mapping({})
        self.assertEqual(settings.backend, "local")
        self.assertEqual(settings.timeout, 120)
        self.assertEqual(settings.docker, DockerSettings())

    def test_valid_docker_config(self):
        settings = TerminalSettings.from_mapping({
            "terminal": {
                "backend": "DOCKER",
                "timeout": 30,
                "docker": {
                    "runtime": "podman",
                    "image": "example/sandbox:latest",
                    "network": True,
                    "cpu": 2,
                    "memory_mb": 2048,
                    "pids_limit": 64,
                    "workspace_read_only": True,
                    "startup_timeout": 45,
                },
            }
        })
        self.assertEqual(settings.backend, "docker")
        self.assertEqual(settings.docker.runtime, "podman")
        self.assertEqual(settings.docker.cpu, 2.0)
        self.assertTrue(settings.docker.workspace_read_only)

    def test_terminal_and_docker_sections_must_be_mappings(self):
        for config in ({"terminal": []}, {"terminal": {"docker": []}}):
            with self.subTest(config=config), self.assertRaises(RuntimeError):
                TerminalSettings.from_mapping(config)

    def test_invalid_scalar_values_fail_with_field_name(self):
        cases = [
            ({"backend": "ssh"}, "terminal.backend"),
            ({"timeout": True}, "terminal.timeout"),
            ({"timeout": 0}, "terminal.timeout"),
            ({"docker": {"runtime": "containerd"}}, "terminal.docker.runtime"),
            ({"docker": {"image": " "}}, "terminal.docker.image"),
            ({"docker": {"network": 1}}, "terminal.docker.network"),
            ({"docker": {"cpu": False}}, "terminal.docker.cpu"),
            ({"docker": {"memory_mb": -1}}, "terminal.docker.memory_mb"),
            ({"docker": {"pids_limit": 0}}, "terminal.docker.pids_limit"),
            ({"docker": {"startup_timeout": 1.5}}, "terminal.docker.startup_timeout"),
        ]
        for section, field in cases:
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field.replace(".", r"\.")):
                TerminalSettings.from_mapping({"terminal": section})


if __name__ == "__main__":
    unittest.main()
