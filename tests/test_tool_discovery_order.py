"""工具发现必须发生在首个 AIAgent system prompt 构建之前。"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ToolDiscoveryOrderTests(unittest.TestCase):
    def test_fresh_process_builds_prompt_from_complete_tool_registry(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as task_tmp:
            db_path = Path(task_tmp) / "state.db"
            code = """
from pathlib import Path
from types import SimpleNamespace
from agent.context_state import ContextSettings
from run_agent import AIAgent
from session_db import SessionDB
db = SessionDB(Path(r'__DB__'))
agent = AIAgent(model='test', system_prompt='', context_settings=ContextSettings(), session_db=db, client=SimpleNamespace())
assert '# Finishing the job' in agent.system_prompt
assert '# Tool-call batching' in agent.system_prompt
assert 'past conversation' in agent.system_prompt
db.close()
""".replace("__DB__", str(db_path))
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
