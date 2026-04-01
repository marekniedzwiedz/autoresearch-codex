# Maze Solver Example

This is a template target repo for Codex-AutoResearch.

Copy it to a separate directory first, then initialize it as its own git repo:

```bash
cp -R examples/maze_solver /tmp/maze-solver-demo
cd /tmp/maze-solver-demo
git init -b main
git add .
git commit -m "Initialize maze solver example"
```

Then run the harness against the copied repo:

```bash
python3 /path/to/run.py run --repo /tmp/maze-solver-demo
```

## How the benchmark works

- `solver.py` is the only file the agent should edit.
- `benchmark.py` generates a fixed set of weighted grid mazes from deterministic seeds.
- Each maze has an exact reference cost computed by an unrestricted Dijkstra search inside the benchmark.
- The candidate solver only gets a tight node-expansion budget for each maze.
- The score rewards two things, in this order:
  - solving more mazes
  - getting path cost closer to the exact optimum
- This makes it an optimization problem rather than a one-shot perfect shortest-path exercise.
- The benchmark still prints the solver's reported `expanded` count for debugging, but it does not score it because that number comes from candidate code and is not independently verifiable.

The score is absolute:

- better solvers get a higher `AUTORESEARCH_SCORE`
- unchanged or weaker solvers stay flat or drop
- the harness advances only when the new score beats the current champion

## Quick checks

```bash
python3 -m unittest -q
python3 benchmark.py
python3 solver.py
```
