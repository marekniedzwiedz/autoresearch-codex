# Mission
Improve `solver.py` so it solves more weighted mazes under a tight search budget
and gets closer to the exact optimal path cost.

## Goal
- Increase the benchmark score on the fixed maze suite in `benchmark.py`.
- Preserve correctness of returned paths.

## Constraints
- Only edit `solver.py`.
- Keep the public `solve(grid_lines, budget=DEFAULT_BUDGET)` entrypoint.
- Keep the `SearchResult` dataclass fields intact.
- Do not add new dependencies.
- Keep the solver deterministic.
- Treat `python3 benchmark.py` as expensive relative to `python3 -m unittest -q`.

## Strategy
- Favor heuristics and data structures that help under a tight expansion budget.
- Solve rate matters more than perfect path quality.
- After solve rate, focus on reducing cost gap to the exact optimum.
- Use node expansions carefully; the benchmark rewards efficiency.
- Run `python3 -m unittest -q` freely while iterating.
- Run `python3 benchmark.py` only after a meaningful solver change that is ready for evaluation.
- Avoid rerunning the full benchmark repeatedly for tiny tweaks inside one round.
