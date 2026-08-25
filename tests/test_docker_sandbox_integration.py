"""真实 Docker/Podman 沙箱集成测试，默认不拉镜像。"""

import os
import tempfile
import unittest
from pathlib import Path

from athena_cli.config import DockerSettings
from tools.environments.docker import DockerEnvironment


@unittest.skipUnless(
    os.getenv("ATHENA_RUN_DOCKER_TESTS") == "1",
    "设置 ATHENA_RUN_DOCKER_TESTS=1 后运行真实容器测试",
)
class DockerSandboxIntegrationTests(unittest.TestCase):
    def test_workspace_round_trip_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            environment = DockerEnvironment(DockerSettings(), root)
            try:
                result = environment.execute("pwd; printf sandbox > result.txt", timeout=30)
                self.assertEqual(result.exit_code, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[0], str(root))
                self.assertEqual((root / "result.txt").read_text(), "sandbox")
            finally:
                environment.close()


if __name__ == "__main__":
    unittest.main()
