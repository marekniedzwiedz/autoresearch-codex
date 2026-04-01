from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


fake_chess = types.ModuleType("chess")
fake_engine = types.ModuleType("chess.engine")
fake_engine.SimpleEngine = object
fake_chess.engine = fake_engine
sys.modules.setdefault("chess", fake_chess)
sys.modules.setdefault("chess.engine", fake_engine)

benchmark = importlib.import_module("benchmark")


class BenchmarkScoreTests(unittest.TestCase):
    def test_match_win_above_five_points_advances(self) -> None:
        self.assertEqual(benchmark.score_from_match(7.0, 5.5, 10), 7.5)

    def test_match_draw_stays_flat(self) -> None:
        self.assertEqual(benchmark.score_from_match(7.0, 5.0, 10), 7.0)

    def test_shared_results_path_uses_git_common_dir(self) -> None:
        worktree_root = Path("/tmp/chess-demo/.autoresearch/worktrees/run-123/r001")
        git_result = mock.Mock(returncode=0, stdout="../../../../.git\n")
        with mock.patch.object(benchmark.subprocess, "run", return_value=git_result):
            results_path = benchmark.shared_results_path(worktree_root)
        self.assertEqual(
            results_path,
            Path("/tmp/chess-demo/.autoresearch/results.tsv").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
