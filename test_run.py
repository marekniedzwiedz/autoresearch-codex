from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run import (
    build_evaluator_context_env,
    extract_ollama_structured_output,
    load_project_config,
    normalize_patch_text,
    select_preferred_ollama_model,
)


class EvaluatorContextEnvTests(unittest.TestCase):
    def test_build_evaluator_context_env_includes_expected_values(self) -> None:
        env = build_evaluator_context_env(
            run_id="run-123",
            round_index=7,
            artifact_dir=Path("/tmp/artifacts"),
            artifacts_root=Path("/tmp/.evoloza-campaign"),
            worktree=Path("/tmp/worktree"),
            base_branch="main",
            champion_branch="evoloza/run-123/r006",
            champion_score=2512.25,
        )
        self.assertEqual(env["EVOLOZA_RUN_ID"], "run-123")
        self.assertEqual(env["EVOLOZA_ROUND"], "7")
        self.assertEqual(env["EVOLOZA_ARTIFACT_DIR"], "/tmp/artifacts")
        self.assertEqual(env["EVOLOZA_ARTIFACTS_ROOT"], "/tmp/.evoloza-campaign")
        self.assertEqual(env["EVOLOZA_WORKTREE"], "/tmp/worktree")
        self.assertEqual(env["EVOLOZA_BASE_BRANCH"], "main")
        self.assertEqual(env["EVOLOZA_CHAMPION_BRANCH"], "evoloza/run-123/r006")
        self.assertEqual(env["EVOLOZA_CHAMPION_SCORE"], "2512.250000")

    def test_build_evaluator_context_env_omits_unknown_champion_fields(self) -> None:
        env = build_evaluator_context_env(
            run_id="run-123",
            round_index=0,
            artifact_dir=Path("/tmp/artifacts"),
            artifacts_root=Path("/tmp/.evoloza-campaign"),
            worktree=Path("/tmp/repo"),
            base_branch="main",
        )
        self.assertNotIn("EVOLOZA_CHAMPION_BRANCH", env)
        self.assertNotIn("EVOLOZA_CHAMPION_SCORE", env)


class ConfigLoadingTests(unittest.TestCase):
    def test_load_project_config_supports_worker_ollama_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "config.toml").write_text(
                """
[worker]
backend = "ollama"
model = "qwen2.5-coder:32b"
ollama_host = "http://127.0.0.1:11434"
context_files = ["solver.py", "tests/*.py"]
max_context_bytes = 9000
max_file_bytes = 3000
max_files = 5
temperature = 0.1

[search]
max_rounds = 2

[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>[0-9]+)"

[git]
artifacts_dir = ".evoloza"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_project_config(repo)
        self.assertEqual(config.worker.backend, "ollama")
        self.assertEqual(config.worker.model, "qwen2.5-coder:32b")
        self.assertEqual(config.worker.ollama_host, "http://127.0.0.1:11434")
        self.assertEqual(config.worker.context_files, ["solver.py", "tests/*.py"])
        self.assertEqual(config.worker.max_context_bytes, 9000)
        self.assertEqual(config.worker.max_file_bytes, 3000)
        self.assertEqual(config.worker.max_files, 5)
        self.assertAlmostEqual(config.worker.temperature, 0.1)

    def test_load_project_config_supports_legacy_codex_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "config.toml").write_text(
                """
[codex]
binary = "/usr/local/bin/codex"
model = "gpt-5"
extra_args = ["--profile", "default"]

[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>[0-9]+)"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_project_config(repo)
        self.assertEqual(config.worker.backend, "codex")
        self.assertEqual(config.worker.binary, "/usr/local/bin/codex")
        self.assertEqual(config.worker.model, "gpt-5")
        self.assertEqual(config.worker.extra_args, ["--profile", "default"])


class OllamaModelSelectionTests(unittest.TestCase):
    def test_select_preferred_ollama_model_prefers_coder_models(self) -> None:
        selected = select_preferred_ollama_model(
            ["qwen3:30b", "codestral:latest", "qwen2.5-coder:32b"]
        )
        self.assertEqual(selected, "qwen2.5-coder:32b")


class PatchNormalizationTests(unittest.TestCase):
    def test_normalize_patch_text_repairs_missing_context_prefixes(self) -> None:
        patch = normalize_patch_text(
            """diff --git a/solver.py b/solver.py
index 3b8d6ad..c019452 100644
--- a/solver.py
+++ b/solver.py
@@ -1,2 +1,2 @@
def score_value():
-    return 1
+    return 2
"""
        )
        self.assertIn("\n def score_value():\n", patch)


class OllamaResponseParsingTests(unittest.TestCase):
    def test_extract_ollama_structured_output_falls_back_to_thinking(self) -> None:
        payload = extract_ollama_structured_output(
            {
                "response": "",
                "thinking": '{"hypothesis":"x","summary":"y","files_touched":[],"local_checks_run":[],"risks":[],"patch":""}',
            }
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["hypothesis"], "x")


if __name__ == "__main__":
    unittest.main()
