# Evoloza

Evoloza is a reusable harness for running agent-driven improvement loops
against arbitrary git repositories.

You point it at a target repo, give it instructions in `program.md`, and define
an evaluator command that emits a numeric score. The harness then creates
isolated candidate branches, asks Codex to make one focused improvement
attempt, runs the evaluator, and only keeps the change if it beats the current
champion.

This is not a tiny single-repo demo like `karpathy/autoresearch`, where the
agent edits one training file inside one research target. This project is the
generic orchestration layer for applying the same pattern to many repos, which
is why it includes git isolation, run state, resumable history, artifacts, and
reporting.

The current implementation is mostly contained in `run.py`.

## Example Run

Here is a real run on the hardened maze benchmark from `examples/maze_solver`.
The first few rounds found the major algorithmic improvement; the later rounds
mostly shaved latency while preserving exact benchmark quality.

![Evoloza maze solver example progress](assets/evoloza-maze-example-progress.svg)

## What The Harness Manages

- A clean baseline score from the current branch
- One candidate branch and one git worktree per round
- Codex invocation and structured experiment summaries
- Evaluator execution and score extraction
- Champion promotion when a candidate improves the score
- Append-only experiment history in `.evoloza/results.tsv`
- Per-run state and artifacts so runs can be resumed or inspected later

## Core Loop

1. Measure the baseline on the current branch.
2. Create a candidate branch and worktree from the current champion.
3. Ask Codex to make one focused improvement attempt.
4. Run the evaluator command(s) and extract the score.
5. If the score improves, commit and promote the candidate.
6. If the score does not improve, discard the candidate branch.
7. Record the outcome and continue until a stopping condition is reached.

## How This Differs From `karpathy/autoresearch`

`karpathy/autoresearch` is a compact research target. Most of the policy lives
in `program.md`, the editable surface is intentionally tiny, and the repo is
optimized for legibility.

Evoloza takes the same basic hill-climbing pattern but packages it as a
reusable harness:

- It works on arbitrary git repos, not one training script.
- It supports arbitrary evaluator commands, not one fixed benchmark.
- It tracks explicit run state, candidate state, and champion state.
- It uses git worktrees and candidate branches for isolation.
- It preserves per-round artifacts and cross-run history under `.evoloza/`.

The extra code is there to make the loop repeatable across different target
repositories, not because the underlying search idea is more complicated.

## Commands

- `evoloza init --repo /path/to/repo`
- `evoloza run --repo /path/to/repo`
- `evoloza status --repo /path/to/repo`
- `evoloza report --repo /path/to/repo`

The installed CLI is `evoloza`.

For local development in this repository, you can also run:

- `python3 run.py init --repo /path/to/repo`
- `python3 run.py run --repo /path/to/repo`
- `python3 run.py status --repo /path/to/repo`
- `python3 run.py report --repo /path/to/repo`

## Target Repo Contract

Each target repo supplies:

- `program.md` with human-written guidance for Codex
- `config.toml` with loop and evaluator settings
- At least one evaluator command that exits successfully and prints a parseable score

The harness writes its own state under `.evoloza/`, including:

- `.evoloza/results.tsv` for cross-run experiment history
- `.evoloza/runs/<run_id>/state.json` for the latest run state
- `.evoloza/runs/<run_id>/rounds/...` for per-round prompts, logs, and results
- `.evoloza/worktrees/...` for temporary candidate worktrees

## Quick Start

Assume you have a repo at `/tmp/demo-repo` with some code and a benchmark
command that prints a score like `AUTORESEARCH_SCORE=123`.

Initialize the repo:

```bash
evoloza init --repo /tmp/demo-repo
```

If you skip `init` and call `run` first, the tool will scaffold missing
`program.md` and `config.toml`, then stop so you can edit them.

Edit `/tmp/demo-repo/program.md`:

```md
# Mission
Improve the benchmark score without breaking existing behavior.

## Goal
- Make the app faster.

## Constraints
- Do not change the public API.
- Keep the project runnable with the existing commands.

## Strategy
- Prefer small, focused improvements.
```

Edit `/tmp/demo-repo/config.toml`:

```toml
# Codex execution settings.
[codex]
binary = "codex"
model = ""
extra_args = []

# Loop stopping conditions.
[search]
max_rounds = 5
max_wall_time_minutes = 60
max_stagnation_rounds = 3

# How the harness evaluates a candidate branch.
[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "AUTORESEARCH_SCORE=(?P<score>-?[0-9]+(?:\\.[0-9]+)?)"
direction = "maximize"

# Git and artifact layout.
[git]
base_branch = ""
artifacts_dir = ".evoloza"
```

Run the loop:

```bash
evoloza run --repo /tmp/demo-repo
evoloza status --repo /tmp/demo-repo
evoloza report --repo /tmp/demo-repo
```

## What Happens During `run`

- The harness measures a baseline score from the current branch.
- Each round runs in its own candidate branch and git worktree.
- Codex is asked to make one improvement attempt and return a hypothesis.
- The evaluator command(s) run against the candidate worktree.
- If the score improves, the candidate is committed and becomes the new champion.
- If the score does not improve, the candidate branch is discarded.
- The full experiment history is fed back into the next prompt, and exact
  repeated hypotheses are rejected as duplicates.
- `run` stays in the foreground and shows progress, elapsed time, and token
  usage when Codex session files are available.
- If the latest run already completed or stopped, `run` starts a fresh run from
  the last committed champion and continues using the full experiment history in
  `.evoloza/results.tsv`.

## Why The Repo Is Larger Than A Demo

This repository is intentionally opinionated, but it is still a harness rather
than a toy example. The implementation complexity mostly comes from:

- generic target-repo support
- git worktree and branch lifecycle management
- persistent run/champion/candidate state
- artifact capture for debugging and inspection
- resumable experiment history across runs
- Codex process integration and live progress reporting

If you want the smallest possible autoresearch repo, follow Karpathy's design
and bake the loop into one specific target repository. If you want the loop as
reusable infrastructure, this repo is aiming at that second category.

## Notes

- The target repo can be created automatically if it does not exist yet.
- If the target repo is missing git, `init` and `run` will initialize it.
- The target repo must be clean before `run` starts.
- This project uses the Python standard library plus `tomli` on Python versions
  older than 3.11.
- To actually run experiments, you still need `git` and the Codex CLI
  installed, or another compatible binary configured via `config.toml` under
  `[codex].binary`.
