# Chess Engine Example

This is a template target repo for Codex-AutoResearch.

Copy it to a separate directory first, then initialize it as its own git repo:

```bash
cp -R examples/chess_engine /tmp/chess-engine-demo
cd /tmp/chess-engine-demo
git init -b main
python3 -m pip install -r requirements.txt
git add .
git commit -m "Initialize chess engine example"
```

Then run the harness against the copied repo:

```bash
python3 /path/to/run.py run --repo /tmp/chess-engine-demo
```

## How the benchmark works

- `engine.py` is the only file the agent should edit.
- `engine.py` exposes a small UCI engine over stdin/stdout.
- `benchmark.py` launches the current working tree engine as a UCI subprocess.
- It also writes `HEAD:engine.py` to a temporary file and launches that as a
  second UCI subprocess.
- The two engines play a short deterministic self-play match from five fixed
  opening positions after three full moves, with the candidate playing both
  White and Black, for 10 games total at `100ms` per move by default.
- Finished games are scored only by normal chess outcomes: win, loss, or draw.
- If a game reaches the benchmark cap of 200 full moves, it is treated as a
  draw so the loop stays bounded without inventing a winner from material.
- If the current engine scores more than 50% of the match points against the
  previous committed engine, it earns promotion.
- If it scores 50% or less, promotion is not earned and the harness rejects
  it.

The benchmark still emits a numeric `AUTORESEARCH_SCORE` for the harness, but
that number is now just an Elo-like champion rating anchored at `0.0` for the
initial engine:

- baseline starts at `0.0`
- ties and losses leave the rating unchanged
- a promoted candidate adds a smoothed Elo-style delta inferred from that
  match result against the current champion

In other words, promotion is binary, while the stored score is only a rough
strength estimate for champions that did earn promotion.

## Quick checks

```bash
python3 -m unittest -q
python3 benchmark.py
python3 engine.py
```
