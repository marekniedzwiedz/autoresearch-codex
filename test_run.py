from __future__ import annotations

import unittest
from pathlib import Path

from run import build_evaluator_context_env


class EvaluatorContextEnvTests(unittest.TestCase):
    def test_build_evaluator_context_env_includes_expected_values(self) -> None:
        env = build_evaluator_context_env(
            run_id="run-123",
            round_index=7,
            artifact_dir=Path("/tmp/artifacts"),
            worktree=Path("/tmp/worktree"),
            base_branch="main",
            champion_branch="evoloza/run-123/r006",
            champion_score=2512.25,
        )
        self.assertEqual(env["EVOLOZA_RUN_ID"], "run-123")
        self.assertEqual(env["EVOLOZA_ROUND"], "7")
        self.assertEqual(env["EVOLOZA_ARTIFACT_DIR"], "/tmp/artifacts")
        self.assertEqual(env["EVOLOZA_WORKTREE"], "/tmp/worktree")
        self.assertEqual(env["EVOLOZA_BASE_BRANCH"], "main")
        self.assertEqual(env["EVOLOZA_CHAMPION_BRANCH"], "evoloza/run-123/r006")
        self.assertEqual(env["EVOLOZA_CHAMPION_SCORE"], "2512.250000")

    def test_build_evaluator_context_env_omits_unknown_champion_fields(self) -> None:
        env = build_evaluator_context_env(
            run_id="run-123",
            round_index=0,
            artifact_dir=Path("/tmp/artifacts"),
            worktree=Path("/tmp/repo"),
            base_branch="main",
        )
        self.assertNotIn("EVOLOZA_CHAMPION_BRANCH", env)
        self.assertNotIn("EVOLOZA_CHAMPION_SCORE", env)


if __name__ == "__main__":
    unittest.main()
