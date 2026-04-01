# Mission
Improve `engine.py` so the current working tree engine wins short self-play
matches against the previous committed version.

## Goal
- Earn promotion by scoring above 50% against the engine in `HEAD:engine.py`.
- Treat the reported benchmark score as an Elo-like champion estimate, not as
  the promotion rule itself.
- Preserve legal move generation.

## Constraints
- Only edit `engine.py`.
- Keep the public entrypoint `choose_move(board, depth=DEFAULT_DEPTH)`.
- Keep `engine.py` speaking basic UCI so `benchmark.py` can launch it.
- Do not add new dependencies beyond `requirements.txt`.
- Keep the benchmark fast enough for many rounds.
- Treat `python3 benchmark.py` as expensive.

## Strategy
- Prefer small search and evaluation improvements.
- Favor legal, deterministic play over risky complexity.
- Do not overfit to one starting position.
- Focus on reliably winning the match, not on running up margin after a win is
  already likely.
- If an idea draws too often, try sharper move ordering or a better evaluation.
- Run `python3 -m unittest -q` freely while iterating.
- Run `python3 benchmark.py` only after a meaningful engine change that is ready for evaluation.
- Avoid running the full benchmark repeatedly for tiny tweaks inside one round.
- If a change clearly failed tests or is obviously weaker, fix it before rerunning the benchmark.
