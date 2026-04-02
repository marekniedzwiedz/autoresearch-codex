# Examples

These are template target repos for Evoloza.

Copy an example to its own directory before running the harness against it.
Do not point the harness at `examples/...` in-place, because this directory
lives inside the harness repo and is not meant to be the target git repo.

Available templates:

- `chess_engine/` - a tiny Python chess engine example that benchmarks the
  current working tree against the previous committed engine in self-play.
- `maze_solver/` - a weighted-grid pathfinding example where the solver is
  scored on solved count, path quality, and benchmark-measured latency across
  a larger mixed suite of mazes with tight per-case budgets and time limits.
