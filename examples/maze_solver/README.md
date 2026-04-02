# Maze Solver Example

This is a template target repo for Evoloza.

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
evoloza run --repo /tmp/maze-solver-demo
```

## How the benchmark works

- `solver.py` is the only file the agent should edit.
- `benchmark.py` generates a deterministic mixed suite of much larger weighted mazes.
- Each maze has an exact reference cost computed by an unrestricted Dijkstra search inside the benchmark.
- The candidate solver gets a per-case search budget and an independently measured wall-time limit.
- The score rewards three things, in this order:
  - solving more mazes
  - getting path cost closer to the exact optimum
  - finishing faster according to the benchmark's own timing harness
- This makes it an optimization problem rather than a one-shot perfect shortest-path exercise.
- The benchmark still prints the solver's reported `expanded` count for debugging, but it does not score it because that number comes from candidate code and is not independently verifiable.
- The latency bonus loses at least one point on any non-zero runtime, so the nominal top score is intentionally not cleanly reachable in practice.

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
