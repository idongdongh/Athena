import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.runner import prepare_workspace, run_manifest
from evaluation.tasks import load_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EvaluationRunnerTests(unittest.TestCase):
    def test_loads_smoke_manifest_and_prepares_git_workspace(self):
        task = load_tasks(PROJECT_ROOT / "eval_tasks" / "smoke.yaml")[0]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            prepare_workspace(task, workspace)
            self.assertTrue((workspace / ".git").is_dir())
            self.assertTrue((workspace / "calculator.py").is_file())
            self.assertFalse((workspace / ".env").exists())

    def test_batch_runner_writes_result_and_resumes_finished_run(self):
        manifest = PROJECT_ROOT / "eval_tasks" / "smoke.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.runner._run_worker", return_value=(0, "ok", "", False)) as worker:
                root = run_manifest(manifest, tmp)
                result_path = root / "arithmetic-percentage-001__r1" / "runner_result.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertEqual("finished", result["status"])
                self.assertEqual(1, worker.call_count)
                run_manifest(manifest, tmp, resume=True)
                self.assertEqual(1, worker.call_count)


if __name__ == "__main__":
    unittest.main()
