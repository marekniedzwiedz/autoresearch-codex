# Evoloza

Evoloza is a reusable harness for running agent-driven improvement loops
against arbitrary git repositories.

You point it at a target repo, give it instructions in `program.md`, and define
an evaluator command that emits a numeric score. The harness then creates
isolated candidate branches, asks a configured worker to make one focused improvement
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
- Worker invocation and structured experiment summaries
- Evaluator execution and score extraction
- Champion promotion when a candidate improves the score
- Append-only experiment history in `.evoloza/results.tsv`
- Per-run state and artifacts so runs can be resumed or inspected later

## Core Loop

1. Measure the baseline on the current branch.
2. Create a candidate branch and worktree from the current champion.
3. Ask the configured worker to make one focused improvement attempt.
4. Run the evaluator command(s) and extract the score.
5. If the score improves, commit and promote the candidate.
6. If the score does not improve, discard the candidate branch.
7. Record the outcome and continue until a stopping condition is reached.

## Modes

Evoloza now supports three operating modes:

- `run`: mode 0, the current all-in-one loop where one worker handles ideation, patching, and evaluation.
- `plan`: mode 1, generate a backlog of atomic experiment cards without executing them.
- `execute`: mode 2, consume a saved plan card-by-card and evaluate each card normally.

The intended split is:

- use an expensive planner such as `gpt-5.4` for `plan`
- use a cheaper or local executor such as Ollama `qwen3.5:35b` for `execute`
- keep `run` for the original single-worker behavior

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
- `evoloza plan --repo /path/to/repo`
- `evoloza execute --repo /path/to/repo --plan /path/to/plan.json`
- `evoloza status --repo /path/to/repo`
- `evoloza report --repo /path/to/repo`

You can also point Evoloza at a specific config file when a target repo keeps
multiple worker profiles side by side:

- `evoloza init --repo /path/to/repo --config config.ollama.toml`
- `evoloza run --repo /path/to/repo --config config.ollama.toml`
- `evoloza plan --repo /path/to/repo --config config.codex.toml`
- `evoloza execute --repo /path/to/repo --config config.ollama.toml --plan .evoloza/plans/<plan_id>/plan.json`
- `evoloza status --repo /path/to/repo --config config.ollama.toml`
- `evoloza report --repo /path/to/repo --config config.ollama.toml`

The installed CLI is `evoloza`.

For local development in this repository, you can also run:

- `python3 run.py init --repo /path/to/repo`
- `python3 run.py run --repo /path/to/repo`
- `python3 run.py plan --repo /path/to/repo`
- `python3 run.py execute --repo /path/to/repo --plan /path/to/plan.json`
- `python3 run.py status --repo /path/to/repo`
- `python3 run.py report --repo /path/to/repo`

Separate configs such as `config.codex.toml` and `config.ollama.toml` can share
the same `program.md`. Pass the desired file with `--config`; relative paths are
resolved from the target repo root. `execute` also resolves `--plan` relative to
the target repo root, and if `--plan` is omitted it uses the latest saved plan
under `.evoloza/plans/`. `status` and `report` also look up runs through the
selected config's `git.artifacts_dir`, so use the same config that created the
run when you inspect it later.

## Target Repo Contract

Each target repo supplies:

- `program.md` with human-written guidance for the worker
- `config.toml` with loop and evaluator settings
- At least one evaluator command that exits successfully and prints a parseable score

The harness writes its own state under `.evoloza/`, including:

- `.evoloza/results.tsv` for cross-run experiment history
- `.evoloza/plans/<plan_id>/plan.json` for saved experiment-card backlogs
- `.evoloza/runs/<run_id>/state.json` for the latest run state
- `.evoloza/runs/<run_id>/rounds/...` for per-round prompts, logs, and results
- `.evoloza/worktrees/...` for temporary candidate worktrees
- `preserved-worktree/` snapshots inside round artifacts when
  `git.preserve_candidate_worktrees = true`

Evaluator commands also receive run-local context through environment variables:

- `EVOLOZA_RUN_ID`
- `EVOLOZA_ROUND`
- `EVOLOZA_ARTIFACT_DIR`
- `EVOLOZA_ARTIFACTS_ROOT`
- `EVOLOZA_WORKTREE`
- `EVOLOZA_BASE_BRANCH`
- `EVOLOZA_CHAMPION_BRANCH` when a champion already exists
- `EVOLOZA_CHAMPION_SCORE` when a champion score already exists

## Quick Start

Assume you have a repo at `/tmp/demo-repo` with some code and a benchmark
command that prints a score like `EVOLOZA_SCORE=123`.

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
# Worker execution settings.
[worker]
backend = "codex"
binary = "codex"
model = ""
extra_args = []

[planner]
backend = "codex"
binary = "codex"
model = "gpt-5.4"
cards_per_plan = 6

[executor]
backend = "ollama"
model = "qwen3.5:35b"
context_files = ["solver.py", "benchmark.py", "test_*.py"]
temperature = 0.15
keep_alive = "30m"
think = false

[executor.ollama_options]
num_ctx = 131072
num_predict = 1024

# Loop stopping conditions.
[search]
max_rounds = 5
max_wall_time_minutes = 60
max_stagnation_rounds = 3

# How the harness evaluates a candidate branch.
[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>-?[0-9]+(?:\\.[0-9]+)?)"
direction = "maximize"

# Git and artifact layout.
[git]
base_branch = ""
artifacts_dir = ".evoloza"
preserve_candidate_worktrees = true
```

The same file can be switched to Ollama by replacing the worker section:

```toml
[worker]
backend = "ollama"
model = "qwen3.5:35b"
ollama_host = "http://127.0.0.1:11434"
context_files = ["solver.py", "benchmark.py", "test_*.py"]
max_context_bytes = 120000
max_file_bytes = 24000
max_files = 24
temperature = 0.2
keep_alive = "30m"
request_timeout_seconds = 3600
think = false
forbidden_hypotheses = ["continuation history for quiet ordering"]

[worker.ollama_options]
num_ctx = 131072
num_predict = 512

[git]
base_branch = ""
artifacts_dir = ".evoloza"
preserve_candidate_worktrees = true
```

Planner / executor split sample:

```toml
[planner]
backend = "codex"
binary = "codex"
model = "gpt-5.4"
cards_per_plan = 8

[executor]
backend = "ollama"
model = "qwen3.5:35b"
ollama_host = "http://127.0.0.1:11434"
context_files = ["solver.py", "benchmark.py", "test_*.py"]
max_context_bytes = 120000
max_file_bytes = 24000
max_files = 24
temperature = 0.15
keep_alive = "30m"
request_timeout_seconds = 3600
think = false
forbidden_hypotheses = ["continuation history for quiet ordering"]

[executor.ollama_options]
num_ctx = 131072
num_predict = 1024
```

Codex backend sample:

```toml
[worker]
backend = "codex"
binary = "codex"
model = ""
extra_args = []
```

Ollama backend sample:

```toml
[worker]
backend = "ollama"
model = "qwen2.5-coder:32b"
ollama_host = "http://127.0.0.1:11434"
context_files = ["solver.py", "benchmark.py", "test_*.py"]
max_context_bytes = 120000
max_file_bytes = 24000
max_files = 24
temperature = 0.2
keep_alive = "30m"
request_timeout_seconds = 3600
think = false
forbidden_hypotheses = ["continuation history for quiet ordering"]

[worker.ollama_options]
num_ctx = 131072
num_predict = 1024
```

The Ollama backend is still patch-based in `run`, but `execute` now uses
anchored structured edit operations instead of raw unified diffs. That split is
intentional: open-ended local-model rounds still behave like the legacy worker,
while card-driven execution asks the model for exact `edit_ops` against one
file and lets Evoloza materialize the final diff locally.

It works best when `program.md` narrows the editable surface or
`worker.context_files` points at the most relevant files. In `execute` mode,
Evoloza narrows the prompt further by replacing the full `target_file` with
focused excerpts around the card's `target_symbols`, `anchor_snippets`, and
code identifiers mentioned in the card notes, while still including any extra
context files such as `Cargo.toml` or compact journals.

Any Ollama `/api/generate` option supported by your local server can be passed
through under `[worker.ollama_options]`. This is the preferred place to set
request-local controls such as `num_ctx`, `num_predict`, `seed`, or stop
sequences from the target repository config.

Practical Ollama controls:

- `keep_alive` keeps the model loaded between rounds so repeated requests do not
  pay a fresh load penalty.
- `request_timeout_seconds` is the client-side timeout for one generation
  request. Increase it when planning or large edit generation can take many
  minutes.
- `num_ctx` controls the server-side context window budget. Set it high enough
  to avoid truncation, but do not max it blindly because larger KV caches cost
  RAM and can increase latency.
- `num_predict` is the easiest way to cap runaway patch or `edit_ops` output
  from a local model.

`worker.forbidden_hypotheses` lets the target repo seed off-limits idea
directions for local-model runs. Evoloza uses those seeds both in the prompt
and in duplicate detection, so Round 1 can reject known bad or already-covered
families before they waste evaluation time.

Run the loop:

```bash
evoloza run --repo /tmp/demo-repo
evoloza status --repo /tmp/demo-repo
evoloza report --repo /tmp/demo-repo
```

Or split planning and execution:

```bash
evoloza plan --repo /tmp/demo-repo --config config.codex.toml
evoloza execute --repo /tmp/demo-repo --config config.ollama.toml
evoloza status --repo /tmp/demo-repo --config config.ollama.toml
evoloza report --repo /tmp/demo-repo --config config.ollama.toml
```

`execute` defaults to the latest saved plan when `--plan` is omitted.

`plan` now uses the planner worker's bounded repository snapshot, not open-ended
tool exploration. In practice that means `[planner].context_files`,
`max_context_bytes`, `max_file_bytes`, and `max_files` matter for Codex
planners the same way they already mattered for Ollama prompts.

## Experiment Cards

`plan` writes a JSON backlog of atomic cards. Each card names:

- one `target_file`
- one short `hypothesis`
- a few `target_symbols`
- one to three exact `anchor_snippets` copied from the target file
- an `allowed_edit_scope`
- nearby `forbidden_families`
- short `implementation_notes`
- a `max_patch_lines` budget

`execute` uses those cards to build a bounded worker prompt. For Ollama
executors, the model must return structured `edit_ops` with exact
`anchor_snippet` matches inside the card's `target_file`; Evoloza applies those
ops locally, writes `candidate.patch`, and rejects anything that drifts outside
the card scope.

## What Happens During `run`

- The harness measures a baseline score from the current branch.
- Each round runs in its own candidate branch and git worktree.
- The configured worker is asked to make one improvement attempt and return a hypothesis.
- The evaluator command(s) run against the candidate worktree.
- If `git.preserve_candidate_worktrees = true`, Evoloza copies the evaluated
  candidate worktree into the round artifacts before cleanup so the exact
  source tree and built binaries remain available later.
- If the score improves, the candidate is committed and becomes the new champion.
- If the score does not improve, the candidate branch is discarded.
- The full experiment history is fed back into the next prompt, and exact
  repeated hypotheses are rejected as duplicates.
- `run` stays in the foreground and shows progress, elapsed time, and token
  usage when the selected backend reports it.
- If the latest run already completed or stopped, `run` starts a fresh run from
  the last committed champion and continues using the full experiment history in
  `.evoloza/results.tsv`.

## What Happens During `plan`

- Evoloza reads `program.md`, prior experiment history, and a bounded snapshot
  of the target repo built from the planner worker settings.
- The planner backend returns a JSON backlog of atomic experiment cards.
- The plan is saved under `.evoloza/plans/<plan_id>/plan.json`.
- Planner prompt and backend artifacts are stored alongside the plan.

## What Happens During `execute`

- Evoloza loads a saved plan and treats each card as one future round.
- The executor backend is prompted with one card at a time.
- The executor prompt is scoped to the card's target file and target symbols.
- For Ollama executors, the target file is shown as focused excerpts around the
  card symbols, anchor snippets, and backticked identifiers instead of dumping
  the whole file.
- Ollama execute mode returns anchored `edit_ops`, not a raw patch. Evoloza
  applies those ops locally, writes the resulting `candidate.patch`, and can run
  one repair pass if the anchors do not apply cleanly on the first attempt.
- Evaluation, promotion, artifact capture, and result history work the same way
  as in `run`.
- If `git.preserve_candidate_worktrees = true`, execute-mode rounds also keep a
  preserved snapshot of the evaluated worktree under the round artifacts.

## Why The Repo Is Larger Than A Demo

This repository is intentionally opinionated, but it is still a harness rather
than a toy example. The implementation complexity mostly comes from:

- generic target-repo support
- git worktree and branch lifecycle management
- persistent run/champion/candidate state
- artifact capture for debugging and inspection
- resumable experiment history across runs
- worker process integration and progress reporting

If you want the smallest possible repo in this style, follow Karpathy's design
and bake the loop into one specific target repository. If you want the loop as
reusable infrastructure, this repo is aiming at that second category.

## Notes

- The target repo can be created automatically if it does not exist yet.
- If the target repo is missing git, `init` and `run` will initialize it.
- The target repo must be clean before `run` starts.
- This project uses the Python standard library plus `tomli` on Python versions
  older than 3.11.
- To actually run experiments, you still need `git` plus either the Codex CLI
  or a local Ollama server with at least one installed model.
- Legacy `[codex]` configs are still accepted, but new configs should use
  `[worker]`.
