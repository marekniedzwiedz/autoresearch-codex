from __future__ import annotations

import unittest

from benchmark import case_score, normalize_result
from solver import SearchResult


class WrappedInt(int):
    def __rsub__(self, other: int) -> int:
        return other + 10**9


class BenchmarkTests(unittest.TestCase):
    def test_case_score_depends_on_solve_and_quality_only(self) -> None:
        result = SearchResult(True, ((0, 0), (0, 1)), 12, 999_999, "solved")
        score, label = case_score(12, result)
        self.assertEqual(score, 12_000)
        self.assertEqual(label, "solved gap=0 expanded=999999")

    def test_normalize_result_rejects_int_subclasses(self) -> None:
        result = SearchResult(True, ((0, 0),), WrappedInt(12), WrappedInt(0), "solved")
        with self.assertRaisesRegex(TypeError, "plain int"):
            normalize_result(result)


if __name__ == "__main__":
    unittest.main()
