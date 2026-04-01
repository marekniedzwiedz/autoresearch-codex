from __future__ import annotations

import unittest

from solver import SearchResult, path_cost, solve


class SolverTests(unittest.TestCase):
    def test_solver_returns_valid_path_on_simple_grid(self) -> None:
        grid = (
            "S111",
            "1#11",
            "1111",
            "111G",
        )
        result = solve(grid, budget=80)
        self.assertTrue(result.solved)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (3, 3))
        self.assertEqual(result.cost, path_cost(grid, result.path))

    def test_solver_reports_budget_exhausted_when_too_small(self) -> None:
        grid = (
            "S1111",
            "1###1",
            "11111",
            "1###1",
            "1111G",
        )
        result = solve(grid, budget=1)
        self.assertFalse(result.solved)
        self.assertIn(result.status, {"budget_exhausted", "no_path"})

    def test_path_cost_counts_weighted_cells(self) -> None:
        grid = (
            "S23",
            "1#4",
            "11G",
        )
        path = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2))
        self.assertEqual(path_cost(grid, path), 4)

    def test_search_result_shape_stays_stable(self) -> None:
        fields = tuple(SearchResult.__dataclass_fields__.keys())
        self.assertEqual(fields, ("solved", "path", "cost", "expanded", "status"))


if __name__ == "__main__":
    unittest.main()
