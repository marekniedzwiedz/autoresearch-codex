from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tomllib as _toml
except ModuleNotFoundError:
    try:
        import tomli as _toml
    except ModuleNotFoundError:
        _toml = None


PROGRAM_FILENAME = "program.md"
CONFIG_FILENAME = "config.toml"
LEGACY_CONFIG_FILENAME = "autoresearch.toml"
APP_NAME = "Evoloza"
CLI_NAME = "evoloza"
DEFAULT_ARTIFACTS_DIR = ".evoloza"
LEGACY_ARTIFACTS_DIR = ".autoresearch"
BRANCH_PREFIX = "evoloza"
RESULT_COLUMNS = [
    "run_id",
    "round",
    "parent_branch",
    "branch",
    "commit",
    "score",
    "status",
    "files_changed",
    "hypothesis",
    "summary",
]
WORKER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "summary": {"type": "string"},
        "files_touched": {"type": "array", "items": {"type": "string"}},
        "local_checks_run": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "patch": {"type": "string"},
    },
    "required": ["hypothesis", "summary", "files_touched", "local_checks_run", "risks"],
    "additionalProperties": False,
}
SUPPORTED_WORKER_BACKENDS = {"codex", "ollama"}
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TEMPERATURE = 0.2
DEFAULT_CONTEXT_MAX_BYTES = 120000
DEFAULT_CONTEXT_FILE_BYTES = 24000
DEFAULT_CONTEXT_FILE_COUNT = 24
TEXT_FILE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lua",
    ".m",
    ".md",
    ".php",
    ".pl",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".scala",
    ".scss",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
TEXT_FILE_NAMES = {
    ".editorconfig",
    ".gitignore",
    ".prettierrc",
    "Dockerfile",
    "Makefile",
    "README",
    "README.md",
    "requirements.txt",
}

DEFAULT_PROGRAM = """# Mission
Describe the objective the worker should optimize for in this repository.

## Goal
- State the desired outcome.

## Constraints
- List files, modules, or behaviors that must stay unchanged.

## Strategy
- Explain what kinds of changes are allowed and what tradeoffs matter.
"""

DEFAULT_CONFIG = """# Worker execution settings.
[worker]
# Backend name: `codex` or `ollama`.
backend = "codex"
# Path to the Codex CLI binary when backend = "codex".
binary = "codex"
# Optional model override. Leave empty to use the backend default.
model = ""
# Extra CLI args passed through to `codex exec` when backend = "codex".
extra_args = []
# Ollama backend only: local API base URL.
ollama_host = "http://127.0.0.1:11434"
# Ollama backend only: repo-relative paths or globs to include in prompt context.
context_files = []
# Ollama backend only: prompt snapshot limits.
max_context_bytes = 120000
max_file_bytes = 24000
max_files = 24
# Ollama backend only: sampling temperature.
temperature = 0.2

# Example Codex settings:
# backend = "codex"
# binary = "codex"
# model = ""
# extra_args = []
#
# Example Ollama settings:
# backend = "ollama"
# model = "qwen2.5-coder:32b"
# ollama_host = "http://127.0.0.1:11434"
# context_files = ["solver.py", "benchmark.py", "test_*.py"]
# max_context_bytes = 120000
# max_file_bytes = 24000
# max_files = 24
# temperature = 0.2

# Loop stopping conditions.
[search]
# Maximum number of candidate rounds to try.
max_rounds = 5
# Maximum wall clock time for a run, in minutes.
max_wall_time_minutes = 60
# Stop after this many non-improving rounds in a row.
max_stagnation_rounds = 3

# How the harness evaluates a candidate branch.
[evaluator]
# Commands run after each worker attempt. All must exit with code 0.
commands = ["python3 -c \\"print('EVOLOZA_SCORE=0')\\""]
# Regex used to extract the numeric score from evaluator output.
score_regex = "EVOLOZA_SCORE=(?P<score>-?[0-9]+(?:\\\\.[0-9]+)?)"
# Use `maximize` when bigger is better, `minimize` when smaller is better.
direction = "maximize"

# Git and artifact layout.
[git]
# Optional base branch override. Leave empty to auto-detect.
base_branch = ""
# Directory inside the target repo where logs, state, and worktrees are stored.
artifacts_dir = ".evoloza"
"""

class TomlDecodeError(ValueError):
    pass


class GitError(RuntimeError):
    pass


@dataclass
class WorkerSettings:
    backend: str = "codex"
    binary: str = "codex"
    model: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    ollama_host: str = DEFAULT_OLLAMA_HOST
    context_files: List[str] = field(default_factory=list)
    max_context_bytes: int = DEFAULT_CONTEXT_MAX_BYTES
    max_file_bytes: int = DEFAULT_CONTEXT_FILE_BYTES
    max_files: int = DEFAULT_CONTEXT_FILE_COUNT
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE


@dataclass
class EvaluatorSettings:
    commands: List[str]
    score_regex: str
    direction: str = "maximize"


@dataclass
class SearchSettings:
    max_rounds: int = 5
    max_wall_time_minutes: int = 60
    max_stagnation_rounds: int = 3


@dataclass
class GitSettings:
    base_branch: Optional[str] = None
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR


@dataclass
class ProjectConfig:
    worker: WorkerSettings
    evaluator: EvaluatorSettings
    search: SearchSettings
    git: GitSettings


@dataclass
class ChampionState:
    branch: str
    commit: str
    score: float
    summary: str
    files_changed: int = 0
    source: str = "baseline"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChampionState":
        return cls(**data)


@dataclass
class CandidateResult:
    run_id: str
    round_index: int
    parent_branch: str
    branch: str
    commit: Optional[str]
    score: Optional[float]
    status: str
    files_changed: int
    hypothesis: str
    summary: str
    artifact_dir: str


@dataclass
class RunState:
    run_id: str
    created_at: str
    updated_at: str
    repo_path: str
    status: str
    phase: str
    base_branch: str
    current_round: int
    rounds_without_improvement: int
    champion: Optional[ChampionState] = None
    pending_candidate: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["champion"] = self.champion.to_dict() if self.champion else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        payload = dict(data)
        champion = payload.get("champion")
        if champion is None and payload.get("beam"):
            champion = payload["beam"][0]
        payload["champion"] = ChampionState.from_dict(champion) if champion else None
        payload.pop("beam", None)
        payload.pop("round_plan", None)
        return cls(**payload)


@dataclass
class EvaluationResult:
    passed: bool
    score: Optional[float]
    log_path: str
    failure_reason: Optional[str] = None


@dataclass
class WorkerInvocationResult:
    returncode: int
    output_path: str
    stderr_path: str
    last_message_path: str
    structured_output: Optional[Dict[str, Any]]
    usage: Optional[Dict[str, int]]


class ProgressReporter:
    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stderr
        self.enabled = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.start_time = time.monotonic()
        self.last_update_time = self.start_time
        self._spinner_thread: Optional[threading.Thread] = None
        self._spinner_stop: Optional[threading.Event] = None
        self._spinner_message = ""
        self._line_width = 0
        self._lock = threading.Lock()
        self._frames = "|/-\\"
        self._frame_index = 0
        self.completed_input_tokens = 0
        self.completed_output_tokens = 0
        self.completed_cached_input_tokens = 0
        self.live_input_tokens = 0
        self.live_output_tokens = 0
        self.live_cached_input_tokens = 0
        self.live_usage_active = False
        self.has_usage = False
        self.current_phase: Optional[str] = None
        self.current_action: Optional[str] = None
        self.phase_started_time: Optional[float] = None
        self.phase_context_label: Optional[str] = None

    def event(self, message: str) -> None:
        with self._lock:
            self.last_update_time = time.monotonic()
            line = self._format_event_line_locked(message)
            self._emit_line_locked(line)

    def spin(self, message: str):
        return _SpinnerContext(self, message)

    def finish(self, message: str) -> None:
        self.end_phase()
        self.event(message)

    def _start_spinner(self, message: str) -> None:
        self._stop_spinner()
        self._spinner_message = message
        self.last_update_time = time.monotonic()
        if not self.enabled:
            return
        self._spinner_stop = threading.Event()
        self._spinner_thread = threading.Thread(target=self._spinner_loop, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self) -> None:
        if self._spinner_stop is not None:
            self._spinner_stop.set()
        if self._spinner_thread is not None:
            self._spinner_thread.join()
        self._spinner_thread = None
        self._spinner_stop = None
        if self.enabled:
            with self._lock:
                self._clear_line_locked()
                self.stream.flush()

    def _spinner_loop(self) -> None:
        assert self._spinner_stop is not None
        while not self._spinner_stop.is_set():
            with self._lock:
                now = time.monotonic()
                frame = self._frames[self._frame_index % len(self._frames)]
                self._frame_index += 1
                line = self._format_spinner_line_locked(frame, now)
                self.stream.write("\r" + line)
                padding = max(0, self._line_width - len(line))
                if padding:
                    self.stream.write(" " * padding)
                self.stream.flush()
                self._line_width = max(self._line_width, len(line))
            self._spinner_stop.wait(0.2)

    def _clear_line_locked(self) -> None:
        if self._line_width > 0:
            self.stream.write("\r" + " " * self._line_width + "\r")
            self._line_width = 0

    def _emit_line_locked(self, line: str) -> None:
        if self.enabled:
            self._clear_line_locked()
        self.stream.write(line + "\n")
        self.stream.flush()

    def _format_event_line_locked(self, message: str) -> str:
        line = "[{elapsed} | {tokens}] {message}".format(
            elapsed=format_duration(self.last_update_time - self.start_time),
            tokens=self.token_label(),
            message=message,
        )
        return self._fit_line_locked(line)

    def _format_spinner_line_locked(self, frame: str, now: float) -> str:
        prefix = "[{0}] ".format(frame)
        suffix = " | t {elapsed} | idle {since} | {tokens}".format(
            elapsed=format_duration(now - self.start_time),
            since=format_duration(now - self.last_update_time),
            tokens=self.token_label(),
        )
        message = self._spinner_status_message_locked()
        width = self._terminal_width_locked()
        available = min(40, width - len(prefix) - len(suffix))
        if available < 12:
            return self._fit_line_locked(prefix + message + suffix)
        return prefix + truncate_middle(message, available) + suffix

    def _spinner_status_message_locked(self) -> str:
        context = progress_context_label(self._spinner_message)
        if self.current_phase:
            parts = [part for part in (context, self.current_phase, self.current_action) if part]
            if parts:
                return " | ".join(parts)
        return compact_progress_message(self._spinner_message)

    def _fit_line_locked(self, line: str) -> str:
        width = self._terminal_width_locked()
        if len(line) <= width:
            return line
        return truncate_middle(line, width)

    def _terminal_width_locked(self) -> int:
        columns = shutil.get_terminal_size(fallback=(120, 20)).columns
        return max(40, columns - 1)

    def add_usage(self, usage: Optional[Dict[str, int]]) -> None:
        if not usage:
            return
        usage = normalize_token_usage(usage)
        with self._lock:
            self.has_usage = True
            self.completed_input_tokens += int(usage.get("input_tokens", 0))
            self.completed_output_tokens += int(usage.get("output_tokens", 0))
            self.completed_cached_input_tokens += int(usage.get("cached_input_tokens", 0))

    def set_live_usage(self, usage: Optional[Dict[str, int]]) -> None:
        if not usage:
            return
        usage = normalize_token_usage(usage)
        with self._lock:
            self.has_usage = True
            self.last_update_time = time.monotonic()
            self.live_input_tokens = int(usage.get("input_tokens", 0))
            self.live_output_tokens = int(usage.get("output_tokens", 0))
            self.live_cached_input_tokens = int(usage.get("cached_input_tokens", 0))
            self.live_usage_active = True

    def finalize_live_usage(self, fallback: Optional[Dict[str, int]] = None) -> None:
        fallback_usage = normalize_token_usage(fallback) if fallback else None
        with self._lock:
            if self.live_usage_active:
                self.completed_input_tokens += max(
                    self.live_input_tokens,
                    0 if fallback_usage is None else fallback_usage["input_tokens"],
                )
                self.completed_output_tokens += max(
                    self.live_output_tokens,
                    0 if fallback_usage is None else fallback_usage["output_tokens"],
                )
                self.completed_cached_input_tokens += max(
                    self.live_cached_input_tokens,
                    0 if fallback_usage is None else fallback_usage["cached_input_tokens"],
                )
                self.live_input_tokens = 0
                self.live_output_tokens = 0
                self.live_cached_input_tokens = 0
                self.live_usage_active = False
                self.has_usage = True
                return
        self.add_usage(fallback_usage)

    def set_phase(
        self,
        phase: Optional[str],
        action: Optional[str] = None,
        context_label: Optional[str] = None,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self.last_update_time = now
            context = context_label or progress_context_label(self._spinner_message)
            if phase == self.current_phase and action == self.current_action:
                return
            previous_phase = self.current_phase
            previous_action = self.current_action
            previous_started = self.phase_started_time
            previous_context = self.phase_context_label or context
            phase_changed = phase != self.current_phase
            self.current_phase = phase
            self.current_action = action
            self.phase_context_label = context
            if not phase_changed:
                return
            self.phase_started_time = now if phase else None
            if previous_phase and previous_started is not None:
                finish_message = "{context} {phase} finished in {duration}".format(
                    context=previous_context or "run",
                    phase=previous_phase,
                    duration=format_duration(now - previous_started),
                )
                if previous_action:
                    finish_message += ": {0}".format(previous_action)
                self._emit_line_locked(self._format_event_line_locked(finish_message))
            if phase and not self.enabled:
                start_message = "{context} {phase}".format(
                    context=context or "run",
                    phase=phase,
                )
                if action:
                    start_message += ": {0}".format(action)
                self._emit_line_locked(self._format_event_line_locked(start_message))

    def end_phase(self) -> None:
        now = time.monotonic()
        with self._lock:
            if not self.current_phase or self.phase_started_time is None:
                return
            self.last_update_time = now
            finish_message = "{context} {phase} finished in {duration}".format(
                context=self.phase_context_label or "run",
                phase=self.current_phase,
                duration=format_duration(now - self.phase_started_time),
            )
            if self.current_action:
                finish_message += ": {0}".format(self.current_action)
            self._emit_line_locked(self._format_event_line_locked(finish_message))
            self.current_phase = None
            self.current_action = None
            self.phase_started_time = None
            self.phase_context_label = None

    def token_label(self) -> str:
        if not self.has_usage:
            return "tok pending"
        input_tokens = self.completed_input_tokens + self.live_input_tokens
        output_tokens = self.completed_output_tokens + self.live_output_tokens
        parts = [
            "tok in {0}".format(format_token_count(input_tokens)),
            "out {0}".format(format_token_count(output_tokens)),
        ]
        return " ".join(parts)


class _SpinnerContext:
    def __init__(self, reporter: ProgressReporter, message: str) -> None:
        self.reporter = reporter
        self.message = message

    def __enter__(self):
        self.reporter._start_spinner(self.message)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.reporter._stop_spinner()


class CodexSessionUsageWatcher:
    def __init__(self, worktree: Path, progress: ProgressReporter, started_at_wall: float) -> None:
        self.worktree = str(worktree.resolve())
        self.progress = progress
        self.started_at_wall = started_at_wall
        self.sessions_root = Path.home() / ".codex" / "sessions"
        self.session_path: Optional[Path] = None
        self._offset = 0
        self._buffer = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.sessions_root.exists():
            self._thread.start()

    def stop(self) -> None:
        if not self.sessions_root.exists():
            return
        self._poll_once()
        self._stop.set()
        self._thread.join()
        self._poll_once()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self._poll_once()

    def _poll_once(self) -> None:
        if self.session_path is None:
            self.session_path = find_codex_session_file(self.worktree, self.started_at_wall, self.sessions_root)
            if self.session_path is None:
                return
        try:
            with self.session_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except FileNotFoundError:
            self.session_path = None
            self._offset = 0
            self._buffer = ""
            return
        if not chunk:
            return
        text = self._buffer + chunk
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            self._buffer = lines.pop()
        else:
            self._buffer = ""
        for line in lines:
            usage = parse_live_usage_from_session_line(line)
            if usage is not None:
                self.progress.set_live_usage(usage)
            phase_update = parse_live_phase_from_session_line(line)
            if phase_update is not None:
                self.progress.set_phase(*phase_update)


def loads_toml(text: str) -> Dict[str, Any]:
    if _toml is None:
        raise RuntimeError("Install tomli to load TOML config files on Python < 3.11")
    try:
        return _toml.loads(text)
    except _toml.TOMLDecodeError as exc:
        raise TomlDecodeError(str(exc)) from exc


def ensure_project_files(repo: Path, force: bool = False) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _write_if_needed(repo / PROGRAM_FILENAME, DEFAULT_PROGRAM, force)
    _write_if_needed(repo / CONFIG_FILENAME, DEFAULT_CONFIG, force)


def scaffold_missing_project_files(repo: Path) -> List[Path]:
    created = []
    program_path = repo / PROGRAM_FILENAME
    config_path = repo / CONFIG_FILENAME
    had_program = program_path.exists()
    had_config = config_path.exists() or (repo / LEGACY_CONFIG_FILENAME).exists()
    ensure_project_files(repo, force=False)
    if not had_program and program_path.exists():
        created.append(program_path)
    if not had_config and config_path.exists():
        created.append(config_path)
    return created


def load_project_config(repo: Path) -> ProjectConfig:
    config_path = find_config_path(repo)
    if config_path is None:
        raise FileNotFoundError(
            "Missing config file: expected {0} or {1}".format(
                repo / CONFIG_FILENAME,
                repo / LEGACY_CONFIG_FILENAME,
            )
        )
    data = loads_toml(config_path.read_text(encoding="utf-8"))
    legacy_codex_section = data.get("codex", {})
    worker_section = data.get("worker")
    evaluator_section = data.get("evaluator", {})
    search_section = data.get("search", {})
    git_section = data.get("git", {})

    if worker_section is None:
        worker_section = legacy_codex_section
        backend = "codex"
    else:
        backend = str(worker_section.get("backend", "codex")).strip().lower() or "codex"
    if backend not in SUPPORTED_WORKER_BACKENDS:
        raise ValueError(
            "worker.backend must be one of {0}".format(", ".join(sorted(SUPPORTED_WORKER_BACKENDS)))
        )

    commands = evaluator_section.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("evaluator.commands must be a non-empty TOML array")
    direction = evaluator_section.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("evaluator.direction must be 'maximize' or 'minimize'")
    worker_extra_args = worker_section.get("extra_args", legacy_codex_section.get("extra_args", []))
    if not isinstance(worker_extra_args, list):
        raise ValueError("worker.extra_args must be a TOML array")
    worker_context_files = worker_section.get("context_files", [])
    if not isinstance(worker_context_files, list):
        raise ValueError("worker.context_files must be a TOML array")

    return ProjectConfig(
        worker=WorkerSettings(
            backend=backend,
            binary=str(
                worker_section.get(
                    "binary",
                    legacy_codex_section.get("binary", "codex" if backend == "codex" else "ollama"),
                )
            ),
            model=_empty_to_none(worker_section.get("model", legacy_codex_section.get("model"))),
            extra_args=[str(item) for item in worker_extra_args],
            ollama_host=str(
                worker_section.get(
                    "ollama_host",
                    worker_section.get("host", DEFAULT_OLLAMA_HOST),
                )
            ),
            context_files=[str(item) for item in worker_context_files],
            max_context_bytes=int(worker_section.get("max_context_bytes", DEFAULT_CONTEXT_MAX_BYTES)),
            max_file_bytes=int(worker_section.get("max_file_bytes", DEFAULT_CONTEXT_FILE_BYTES)),
            max_files=int(worker_section.get("max_files", DEFAULT_CONTEXT_FILE_COUNT)),
            temperature=float(worker_section.get("temperature", DEFAULT_OLLAMA_TEMPERATURE)),
        ),
        evaluator=EvaluatorSettings(
            commands=[str(item) for item in commands],
            score_regex=str(evaluator_section["score_regex"]),
            direction=direction,
        ),
        search=SearchSettings(
            max_rounds=int(search_section.get("max_rounds", 5)),
            max_wall_time_minutes=int(search_section.get("max_wall_time_minutes", 60)),
            max_stagnation_rounds=int(search_section.get("max_stagnation_rounds", 3)),
        ),
        git=GitSettings(
            base_branch=_empty_to_none(git_section.get("base_branch")),
            artifacts_dir=resolve_artifacts_dir(repo, git_section),
        ),
    )


def resolve_artifacts_dir(repo: Path, git_section: Dict[str, Any]) -> str:
    configured = _empty_to_none(git_section.get("artifacts_dir"))
    if configured is not None:
        return str(configured)
    if (repo / DEFAULT_ARTIFACTS_DIR).exists():
        return DEFAULT_ARTIFACTS_DIR
    if (repo / LEGACY_ARTIFACTS_DIR).exists():
        return LEGACY_ARTIFACTS_DIR
    return DEFAULT_ARTIFACTS_DIR


def program_text(repo: Path) -> str:
    path = repo / PROGRAM_FILENAME
    if not path.exists():
        raise FileNotFoundError("Missing program file: {0}".format(path))
    return path.read_text(encoding="utf-8")


def find_config_path(repo: Path) -> Optional[Path]:
    preferred = repo / CONFIG_FILENAME
    legacy = repo / LEGACY_CONFIG_FILENAME
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return None


def is_git_repo(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def ensure_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    if not is_git_repo(repo):
        result = subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            run_git(repo, "init")
            run_git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    if not has_commits(repo):
        run_git(repo, "add", "-A")
        run_git(
            repo,
            "commit",
            "--allow-empty",
            "-m",
            "Initialize repository for {0}".format(APP_NAME),
            env=git_commit_env(),
        )


def has_commits(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def ensure_clean_worktree(repo: Path) -> None:
    if run_git(repo, "status", "--porcelain").strip():
        raise GitError("Target repo must be clean before running {0}".format(APP_NAME))


def determine_base_branch(repo: Path, configured: Optional[str]) -> str:
    if configured:
        return configured
    branch = run_git(repo, "branch", "--show-current").strip()
    if branch:
        return branch
    for candidate in ("main", "master"):
        if branch_exists(repo, candidate):
            return candidate
    raise GitError("Unable to determine base branch")


def head_commit(repo: Path, ref: str = "HEAD") -> str:
    return run_git(repo, "rev-parse", ref).strip()


def create_worktree(repo: Path, worktree_path: Path, branch: str, start_point: str) -> None:
    if worktree_path.exists():
        remove_worktree(repo, worktree_path)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    run_git(repo, "worktree", "add", "-b", branch, str(worktree_path), start_point)


def remove_worktree(repo: Path, worktree_path: Path) -> None:
    if not worktree_path.exists():
        return
    result = subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and worktree_path.exists():
        for child in sorted(worktree_path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        if worktree_path.exists():
            worktree_path.rmdir()


def branch_exists(repo: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", "refs/heads/{0}".format(branch)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def delete_branch(repo: Path, branch: str) -> None:
    result = subprocess.run(
        ["git", "branch", "-D", branch],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise GitError(result.stderr.strip() or result.stdout.strip())


def create_branch(repo: Path, branch: str, start_point: str) -> None:
    run_git(repo, "branch", branch, start_point)


def tracked_changes(repo: Path) -> List[str]:
    files = []
    for line in run_git(repo, "status", "--porcelain").splitlines():
        if not line:
            continue
        payload = line[3:]
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        files.append(payload.strip())
    return sorted(set(files))


def stage_paths(repo: Path, paths: List[str]) -> None:
    if paths:
        run_git(repo, "add", "-A", "--", *paths)


def commit_paths(repo: Path, message: str) -> str:
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=git_commit_env(),
        check=False,
    )
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or result.stdout.strip())
    return head_commit(repo)


def run_evaluator(
    repo: Path,
    settings: EvaluatorSettings,
    artifact_dir: Path,
    progress: Optional[ProgressReporter] = None,
    stage_prefix: str = "Evaluator",
    context_env: Optional[Dict[str, str]] = None,
) -> EvaluationResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    command_env = os.environ.copy()
    if context_env:
        command_env.update(context_env)
    try:
        for index, command in enumerate(settings.commands, start=1):
            stage_message = "{prefix} {index}/{total}: {command}".format(
                prefix=stage_prefix,
                index=index,
                total=len(settings.commands),
                command=command,
            )
            if progress is not None:
                progress.set_phase(
                    classify_command_phase(command),
                    summarize_command_action(command),
                    context_label=progress_context_label(stage_message),
                )
                if not progress.enabled:
                    progress.event(stage_message)
            with progress.spin(stage_message) if progress is not None else _nullcontext():
                result = subprocess.run(
                    command,
                    cwd=str(repo),
                    shell=True,
                    executable="/bin/zsh",
                    capture_output=True,
                    text=True,
                    env=command_env,
                    check=False,
                )
            logs.append(
                {
                    "index": index,
                    "command": command,
                    "returncode": result.returncode,
                    "context_env": context_env or {},
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            if result.returncode != 0:
                log_path = artifact_dir / "evaluator.json"
                log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
                return EvaluationResult(
                    passed=False,
                    score=None,
                    log_path=str(log_path),
                    failure_reason="Command failed: {0}".format(command),
                )

        combined = "\n".join(entry["stdout"] + "\n" + entry["stderr"] for entry in logs)
        match = re.search(settings.score_regex, combined, re.MULTILINE)
        log_path = artifact_dir / "evaluator.json"
        log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
        if not match:
            return EvaluationResult(
                passed=False,
                score=None,
                log_path=str(log_path),
                failure_reason="Evaluator output did not match score_regex",
            )
        score_text = match.group("score") if "score" in match.groupdict() else match.group(1)
        return EvaluationResult(passed=True, score=float(score_text), log_path=str(log_path))
    finally:
        if progress is not None:
            progress.end_phase()


def build_evaluator_context_env(
    *,
    run_id: str,
    round_index: int,
    artifact_dir: Path,
    artifacts_root: Path,
    worktree: Path,
    base_branch: str,
    champion_branch: Optional[str] = None,
    champion_score: Optional[float] = None,
) -> Dict[str, str]:
    env = {
        "EVOLOZA_RUN_ID": run_id,
        "EVOLOZA_ROUND": str(round_index),
        "EVOLOZA_ARTIFACT_DIR": str(artifact_dir),
        "EVOLOZA_ARTIFACTS_ROOT": str(artifacts_root),
        "EVOLOZA_WORKTREE": str(worktree),
        "EVOLOZA_BASE_BRANCH": base_branch,
    }
    if champion_branch:
        env["EVOLOZA_CHAMPION_BRANCH"] = champion_branch
    if champion_score is not None:
        env["EVOLOZA_CHAMPION_SCORE"] = "{0:.6f}".format(champion_score)
    return env


def build_codex_prompt(
    program: str,
    config: ProjectConfig,
    run_id: str,
    round_index: int,
    champion: ChampionState,
    branch_name: str,
    history_rows: List[Dict[str, str]],
    ) -> str:
    history_block = render_history_for_prompt(history_rows)
    prompt = """
    You are running one {app_name} experiment.

    Run id: {run_id}
    Round: {round_index}
    Champion branch: {champion_branch}
    Candidate branch: {branch_name}
    Current champion score: {champion_score:.6f}
    Score direction: {direction}
    Current champion summary: {champion_summary}

    Previous experiment log:
    {history_block}

    Official evaluator commands:
    {commands}

    Hard rules:
    - Work only inside the current repository.
    - You may edit files and run local commands as needed.
    - Do not create commits, branches, worktrees, or reset git state.
    - Leave your best candidate diff in the working tree when you stop.
    - Keep changes coherent and focused on one experiment.
    - Read the previous experiment log and avoid repeating the same idea.
    - Only retry an earlier direction if you are clearly extending it in a meaningfully different way.
    - In your final `hypothesis` field, describe exactly what this experiment was trying.

    Mission:
    {program}
    """
    return textwrap.dedent(
        prompt.format(
            app_name=APP_NAME,
            run_id=run_id,
            round_index=round_index,
            champion_branch=champion.branch,
            branch_name=branch_name,
            champion_score=champion.score,
            direction=config.evaluator.direction,
            champion_summary=champion.summary,
            history_block=history_block,
            commands="\n".join("- {0}".format(item) for item in config.evaluator.commands),
            program=program.strip(),
        )
    ).strip() + "\n"


def worker_display_name(settings: WorkerSettings) -> str:
    if settings.backend == "ollama":
        return "Ollama"
    return "Codex"


def context_path_matches(path: str, patterns: List[str]) -> bool:
    normalized = path.replace(os.sep, "/")
    basename = Path(normalized).name
    for pattern in patterns:
        candidate = str(pattern).strip()
        if not candidate:
            continue
        if fnmatch.fnmatch(normalized, candidate) or fnmatch.fnmatch(basename, candidate):
            return True
    return False


def is_probably_text_file(path: Path) -> bool:
    if path.name in TEXT_FILE_NAMES or path.suffix.lower() in TEXT_FILE_EXTENSIONS:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\0" not in sample


def score_context_file(relpath: str, hint_text: str, size_bytes: int, forced: bool) -> int:
    path_text = relpath.lower()
    basename = Path(relpath).name.lower()
    score = 0
    if forced:
        score += 10000
    if path_text in hint_text or basename in hint_text:
        score += 1500
    if Path(relpath).suffix.lower() in TEXT_FILE_EXTENSIONS or basename in {item.lower() for item in TEXT_FILE_NAMES}:
        score += 200
    if "/test" in path_text or basename.startswith("test_"):
        score += 75
    if basename in {"pyproject.toml", "package.json", "requirements.txt", "setup.py"}:
        score += 125
    score -= min(size_bytes, 50000) // 200
    return score


def truncate_text_to_bytes(text: str, limit: int) -> Tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    truncated = encoded[: max(limit, 0)]
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]
    payload = truncated.decode("utf-8", errors="ignore").rstrip()
    return payload + "\n... [truncated]\n", True


def render_repo_file_list(paths: List[str], limit: int = 200) -> str:
    if not paths:
        return "- No tracked files."
    visible = paths[:limit]
    lines = ["- {0}".format(path) for path in visible]
    remaining = len(paths) - len(visible)
    if remaining > 0:
        lines.append("- ... ({0} more files omitted)".format(remaining))
    return "\n".join(lines)


def render_repo_snapshot_entry(relpath: str, content: str, truncated: bool) -> str:
    trailer = "\n[truncated]\n" if truncated and not content.rstrip().endswith("[truncated]") else ""
    body = content.rstrip("\n")
    return "=== FILE: {0} ===\n{1}{2}\n=== END FILE ===\n".format(relpath, body, trailer)


def build_repo_snapshot(worktree: Path, config: ProjectConfig, program: str) -> Tuple[str, List[str], List[str]]:
    all_paths = [line.strip() for line in run_git(worktree, "ls-files").splitlines() if line.strip()]
    included_entries: List[str] = []
    included_paths: List[str] = []
    omitted_paths: List[str] = []
    total_bytes = 0
    artifact_prefix = config.git.artifacts_dir.rstrip("/") + "/"
    hint_text = (program + "\n" + "\n".join(config.evaluator.commands)).lower()
    candidates = []

    for relpath in all_paths:
        if relpath.startswith(".git/") or relpath.startswith(artifact_prefix):
            continue
        forced = context_path_matches(relpath, config.worker.context_files)
        if relpath in {PROGRAM_FILENAME, CONFIG_FILENAME, LEGACY_CONFIG_FILENAME} and not forced:
            continue
        path = worktree / relpath
        if not path.is_file() or not is_probably_text_file(path):
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        if size_bytes > config.worker.max_file_bytes and not forced:
            omitted_paths.append(relpath)
            continue
        candidates.append(
            (
                0 if forced else 1,
                -score_context_file(relpath, hint_text, size_bytes, forced),
                size_bytes,
                relpath,
                forced,
            )
        )

    for _, _, _, relpath, forced in sorted(candidates):
        if len(included_paths) >= config.worker.max_files:
            omitted_paths.append(relpath)
            continue
        path = worktree / relpath
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            omitted_paths.append(relpath)
            continue
        content_limit = config.worker.max_file_bytes
        if forced:
            content_limit = max(content_limit, min(config.worker.max_context_bytes, content_limit * 2))
        rendered_content, truncated = truncate_text_to_bytes(content, content_limit)
        entry = render_repo_snapshot_entry(relpath, rendered_content, truncated)
        entry_bytes = len(entry.encode("utf-8"))
        if included_entries and total_bytes + entry_bytes > config.worker.max_context_bytes:
            omitted_paths.append(relpath)
            continue
        if not included_entries and entry_bytes > config.worker.max_context_bytes:
            reduced_content, reduced_truncated = truncate_text_to_bytes(
                content,
                max(1024, config.worker.max_context_bytes - 256),
            )
            entry = render_repo_snapshot_entry(relpath, reduced_content, reduced_truncated)
            entry_bytes = len(entry.encode("utf-8"))
        if entry_bytes > config.worker.max_context_bytes or total_bytes + entry_bytes > config.worker.max_context_bytes:
            omitted_paths.append(relpath)
            continue
        included_entries.append(entry)
        included_paths.append(relpath)
        total_bytes += entry_bytes

    if not included_entries:
        return "No repository files fit inside the configured Ollama prompt budget.\n", [], omitted_paths
    return "\n".join(included_entries).rstrip() + "\n", included_paths, omitted_paths


def build_ollama_prompt(
    worktree: Path,
    program: str,
    config: ProjectConfig,
    run_id: str,
    round_index: int,
    champion: ChampionState,
    branch_name: str,
    history_rows: List[Dict[str, str]],
) -> str:
    history_block = render_history_for_prompt(history_rows)
    all_paths = [line.strip() for line in run_git(worktree, "ls-files").splitlines() if line.strip()]
    snapshot, included_paths, omitted_paths = build_repo_snapshot(worktree, config, program)
    prompt = """
    You are preparing one {app_name} experiment patch for a git repository.

    Run id: {run_id}
    Round: {round_index}
    Champion branch: {champion_branch}
    Candidate branch: {branch_name}
    Current champion score: {champion_score:.6f}
    Score direction: {direction}
    Current champion summary: {champion_summary}

    Previous experiment log:
    {history_block}

    Official evaluator commands:
    {commands}

    Hard rules:
    - You cannot execute commands or inspect the filesystem directly.
    - Use only the repository snapshot included below.
    - Return exactly one JSON object and nothing else.
    - The JSON object must include: hypothesis, summary, files_touched, local_checks_run, risks, patch.
    - The `patch` field must be a unified git diff that can be applied with `git apply`.
    - Only modify existing text files whose contents are included below, unless you are creating a small supporting text file.
    - If the snapshot is not sufficient, return an empty patch and explain the missing context in `risks`.
    - Keep the change focused on one experiment.

    Mission:
    {program}

    Repository file list:
    {file_list}

    Included snapshot files:
    {included_paths}

    Omitted snapshot files:
    {omitted_paths}

    Repository snapshot:
    {snapshot}
    """
    return textwrap.dedent(
        prompt.format(
            app_name=APP_NAME,
            run_id=run_id,
            round_index=round_index,
            champion_branch=champion.branch,
            branch_name=branch_name,
            champion_score=champion.score,
            direction=config.evaluator.direction,
            champion_summary=champion.summary,
            history_block=history_block,
            commands="\n".join("- {0}".format(item) for item in config.evaluator.commands),
            program=program.strip(),
            file_list=render_repo_file_list(all_paths),
            included_paths=render_repo_file_list(included_paths, limit=max(len(included_paths), 1)),
            omitted_paths=render_repo_file_list(omitted_paths, limit=50),
            snapshot=snapshot.rstrip(),
        )
    ).strip() + "\n"


def build_worker_prompt(
    worktree: Path,
    program: str,
    config: ProjectConfig,
    run_id: str,
    round_index: int,
    champion: ChampionState,
    branch_name: str,
    history_rows: List[Dict[str, str]],
) -> str:
    if config.worker.backend == "ollama":
        return build_ollama_prompt(
            worktree,
            program,
            config,
            run_id,
            round_index,
            champion,
            branch_name,
            history_rows,
        )
    return build_codex_prompt(
        program,
        config,
        run_id,
        round_index,
        champion,
        branch_name,
        history_rows,
    )


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidates = [text.strip(), strip_code_fences(text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_ollama_structured_output(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for field in ("response", "thinking"):
        value = response.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        parsed = extract_json_object(value)
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    items = []
    for entry in value:
        text = str(entry).strip()
        if text:
            items.append(text)
    return items


def normalize_patch_text(text: str) -> str:
    stripped = strip_code_fences(text)
    diff_index = stripped.find("diff --git ")
    if diff_index != -1:
        stripped = stripped[diff_index:]
    stripped = stripped.strip()
    if not stripped:
        return ""
    return repair_unified_diff_hunks(stripped) + "\n"


def repair_unified_diff_hunks(patch_text: str) -> str:
    lines = patch_text.splitlines()
    repaired = []
    in_hunk = False
    for line in lines:
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@ "):
            in_hunk = True
        elif in_hunk and not line.startswith((" ", "+", "-", "\\")):
            line = " " + line
        repaired.append(line)
    return "\n".join(repaired)


def normalize_worker_output(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    return {
        "hypothesis": str(payload.get("hypothesis") or "No hypothesis provided.").strip()
        or "No hypothesis provided.",
        "summary": str(payload.get("summary") or "No summary provided.").strip()
        or "No summary provided.",
        "files_touched": normalize_string_list(payload.get("files_touched")),
        "local_checks_run": normalize_string_list(payload.get("local_checks_run")),
        "risks": normalize_string_list(payload.get("risks")),
        "patch": normalize_patch_text(str(payload.get("patch", ""))),
    }


def apply_unified_diff(worktree: Path, artifact_dir: Path, patch_text: str) -> Optional[str]:
    patch_path = artifact_dir / "candidate.patch"
    patch_path.write_text(patch_text, encoding="utf-8")
    for args in (
        ["git", "apply", "--check", "--recount", "--whitespace=nowarn", str(patch_path)],
        ["git", "apply", "--recount", "--whitespace=nowarn", str(patch_path)],
    ):
        result = subprocess.run(
            args,
            cwd=str(worktree),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return result.stderr.strip() or result.stdout.strip() or "git apply failed"
    return None


def select_preferred_ollama_model(model_names: List[str]) -> Optional[str]:
    if not model_names:
        return None
    preferences = (
        "qwen2.5-coder",
        "codestral",
        "deepseek-coder",
        "codellama",
        "starcoder",
        "qwen",
        "mistral",
        "llama",
    )
    lowered = [(name, name.lower()) for name in model_names]
    for prefix in preferences:
        for original, normalized in lowered:
            if prefix in normalized:
                return original
    return model_names[0]


def ollama_api_json(base_url: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Ollama API error {0}: {1}".format(exc.code, detail or exc.reason))
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama API unavailable at {0}: {1}".format(url, exc.reason))
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama API returned invalid JSON: {0}".format(exc))
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama API returned an unexpected payload")
    return parsed


def resolve_ollama_model(settings: WorkerSettings) -> str:
    if settings.model:
        return settings.model
    tags = ollama_api_json(settings.ollama_host, "/api/tags")
    models = tags.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Ollama /api/tags did not return a model list")
    names = []
    for item in models:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    selected = select_preferred_ollama_model(names)
    if selected is None:
        raise RuntimeError("No Ollama models are available at {0}".format(settings.ollama_host))
    return selected


def run_codex(
    worktree: Path,
    artifact_dir: Path,
    prompt: str,
    settings: WorkerSettings,
    progress: Optional[ProgressReporter] = None,
    stage_message: str = "Codex working",
) -> WorkerInvocationResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    schema_path = artifact_dir / "worker_output_schema.json"
    output_path = artifact_dir / "codex.jsonl"
    stderr_path = artifact_dir / "codex.stderr.log"
    last_message_path = artifact_dir / "last_message.json"
    schema_path.write_text(json.dumps(WORKER_OUTPUT_SCHEMA, indent=2), encoding="utf-8")

    command = [
        settings.binary,
        "exec",
        "--json",
        "--full-auto",
        "-C",
        str(worktree),
        "--output-schema",
        str(schema_path),
        "-o",
        str(last_message_path),
        "-",
    ]
    if settings.model:
        command.extend(["-m", settings.model])
    command.extend(settings.extra_args)

    if progress is not None and not progress.enabled:
        progress.event(stage_message)
    process = None
    watcher = None
    with progress.spin(stage_message) if progress is not None else _nullcontext():
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if progress is not None:
            watcher = CodexSessionUsageWatcher(worktree, progress, time.time())
            watcher.start()
        try:
            stdout_text, stderr_text = process.communicate(prompt)
        finally:
            if watcher is not None:
                watcher.stop()
    output_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    structured_output = None
    if last_message_path.exists():
        content = last_message_path.read_text(encoding="utf-8").strip()
        if content:
            structured_output = normalize_worker_output(extract_json_object(content))
    usage = parse_usage_from_jsonl(stdout_text)
    return WorkerInvocationResult(
        returncode=process.returncode if process is not None else 1,
        output_path=str(output_path),
        stderr_path=str(stderr_path),
        last_message_path=str(last_message_path),
        structured_output=structured_output,
        usage=usage,
    )


def run_ollama(
    worktree: Path,
    artifact_dir: Path,
    prompt: str,
    settings: WorkerSettings,
    progress: Optional[ProgressReporter] = None,
    stage_message: str = "Ollama working",
) -> WorkerInvocationResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request_path = artifact_dir / "ollama.request.json"
    response_path = artifact_dir / "ollama.response.json"
    stderr_path = artifact_dir / "ollama.stderr.log"
    last_message_path = artifact_dir / "last_message.json"
    if progress is not None and not progress.enabled:
        progress.event(stage_message)
    structured_output = None
    usage = None
    try:
        model = resolve_ollama_model(settings)
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": settings.temperature,
            },
        }
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with progress.spin(stage_message) if progress is not None else _nullcontext():
            response = ollama_api_json(settings.ollama_host, "/api/generate", payload)
        response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        structured_output = normalize_worker_output(extract_ollama_structured_output(response))
        if structured_output is None:
            message = "Ollama response did not contain a valid JSON object"
            stderr_path.write_text(message, encoding="utf-8")
            return WorkerInvocationResult(
                returncode=1,
                output_path=str(response_path),
                stderr_path=str(stderr_path),
                last_message_path=str(last_message_path),
                structured_output=None,
                usage=None,
            )
        last_message_path.write_text(json.dumps(structured_output, indent=2), encoding="utf-8")
        patch_error = None
        if structured_output.get("patch"):
            patch_error = apply_unified_diff(worktree, artifact_dir, str(structured_output["patch"]))
        if patch_error is not None:
            structured_output["summary"] = "Patch apply failed: {0}".format(patch_error)
            last_message_path.write_text(json.dumps(structured_output, indent=2), encoding="utf-8")
            stderr_path.write_text(patch_error, encoding="utf-8")
            return WorkerInvocationResult(
                returncode=1,
                output_path=str(response_path),
                stderr_path=str(stderr_path),
                last_message_path=str(last_message_path),
                structured_output=structured_output,
                usage=None,
            )
        usage = normalize_token_usage(
            {
                "input_tokens": response.get("prompt_eval_count", 0),
                "cached_input_tokens": 0,
                "output_tokens": response.get("eval_count", 0),
            }
        )
        return WorkerInvocationResult(
            returncode=0,
            output_path=str(response_path),
            stderr_path=str(stderr_path),
            last_message_path=str(last_message_path),
            structured_output=structured_output,
            usage=usage,
        )
    except Exception as exc:
        stderr_path.write_text(str(exc), encoding="utf-8")
        return WorkerInvocationResult(
            returncode=1,
            output_path=str(response_path),
            stderr_path=str(stderr_path),
            last_message_path=str(last_message_path),
            structured_output=structured_output,
            usage=usage,
        )


def run_worker(
    worktree: Path,
    artifact_dir: Path,
    prompt: str,
    settings: WorkerSettings,
    progress: Optional[ProgressReporter] = None,
    stage_message: Optional[str] = None,
) -> WorkerInvocationResult:
    label = worker_display_name(settings)
    message = stage_message or "{0} working".format(label)
    if settings.backend == "ollama":
        return run_ollama(worktree, artifact_dir, prompt, settings, progress=progress, stage_message=message)
    return run_codex(worktree, artifact_dir, prompt, settings, progress=progress, stage_message=message)


def ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t").writeheader()


def append_results(path: Path, rows: List[Dict[str, str]]) -> None:
    ensure_results_file(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t").writerows(rows)


def read_results(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def render_status(state: RunState) -> str:
    lines = [
        "Run: {0}".format(state.run_id),
        "Status: {0} ({1})".format(state.status, state.phase),
        "Base branch: {0}".format(state.base_branch),
        "Current round: {0}".format(state.current_round),
        "Rounds without improvement: {0}".format(state.rounds_without_improvement),
        "Champion:",
    ]
    if not state.champion:
        lines.append("- none")
    else:
        lines.append(
            "- {branch}: score={score:.6f} files_changed={files_changed} summary={summary}".format(
                branch=state.champion.branch,
                score=state.champion.score,
                files_changed=state.champion.files_changed,
                summary=state.champion.summary,
            )
        )
    return "\n".join(lines)


def render_report(state: RunState, results: List[Dict[str, str]]) -> str:
    lines = [
        "# {0} Report".format(APP_NAME),
        "",
        "- Run id: `{0}`".format(state.run_id),
        "- Status: `{0}`".format(state.status),
        "- Base branch: `{0}`".format(state.base_branch),
        "- Final champion: `{0}`".format(state.champion.branch if state.champion else "none"),
        "",
        "## Champion",
        "",
    ]
    if state.champion:
        lines.append(
            "- `{branch}` score={score:.6f} files_changed={files_changed} summary={summary}".format(
                branch=state.champion.branch,
                score=state.champion.score,
                files_changed=state.champion.files_changed,
                summary=state.champion.summary,
            )
        )
    lines.extend(["", "## Results", ""])
    for row in results:
        lines.append(
            "- round {round} `{branch}` status={status} score={score} hypothesis={hypothesis} summary={summary}".format(
                **row
            )
        )
    return "\n".join(lines)


class Orchestrator:
    def __init__(self, repo: Path, progress: Optional[ProgressReporter] = None):
        self.repo = repo.resolve()
        self.progress = progress

    def run(self, run_id: Optional[str] = None) -> RunState:
        if self.progress is not None:
            self.progress.event("Preparing run in {0}".format(self.repo))
        ensure_git_repo(self.repo)
        config = load_project_config(self.repo)
        ensure_clean_worktree(self.repo)

        state = self._load_or_create_state(config, run_id)
        if self.progress is not None:
            self.progress.event("Run id {0} on base branch {1}".format(state.run_id, state.base_branch))
            if state.status == "created" and state.current_round == 1 and state.champion and state.champion.source == "seeded":
                self.progress.event(
                    "Seeded from previous champion {0} score={1:.6f}".format(
                        state.champion.branch,
                        state.champion.score,
                    )
                )
        if state.phase == "candidate_in_progress" and state.pending_candidate:
            if self.progress is not None:
                self.progress.event("Cleaning up unfinished candidate from previous run state")
            self._cleanup_pending_candidate(state)
            state.phase = "idle"
            state.pending_candidate = None
            state.updated_at = now_iso()
            self._write_state(config, state)

        if not state.champion:
            if self.progress is not None:
                self.progress.event("Measuring baseline")
            state.champion = self._evaluate_baseline(config, state)
            state.updated_at = now_iso()
            self._write_state(config, state)
            if self.progress is not None:
                self.progress.event(
                    "Baseline ready: score={0:.6f}".format(state.champion.score)
                )

        started_at = time.monotonic()
        state.status = "running"
        self._write_state(config, state)

        while state.current_round <= config.search.max_rounds:
            if minutes_elapsed(started_at) >= config.search.max_wall_time_minutes:
                state.status = "stopped"
                break
            if state.rounds_without_improvement >= config.search.max_stagnation_rounds:
                state.status = "completed"
                break

            if self.progress is not None:
                self.progress.event(
                    "Round {0}/{1}: champion score={2:.6f}".format(
                        state.current_round,
                        config.search.max_rounds,
                        state.champion.score,
                    )
                )
            candidate = self._plan_round(config, state)
            state.phase = "candidate_in_progress"
            state.pending_candidate = candidate
            state.updated_at = now_iso()
            self._write_state(config, state)
            history_rows = self._history_rows(config)

            result = self._run_candidate(
                config=config,
                run_id=state.run_id,
                round_index=state.current_round,
                base_branch=state.base_branch,
                champion=state.champion,
                branch_name=candidate["branch"],
                worktree_path=Path(candidate["worktree"]),
                artifact_dir=Path(candidate["artifact_dir"]),
                history_rows=history_rows,
            )
            self._append_result(config, state.run_id, result)

            if result.status == "accepted" and result.commit and result.score is not None:
                state.champion = ChampionState(
                    branch=result.branch,
                    commit=result.commit,
                    score=result.score,
                    summary=result.summary,
                    files_changed=result.files_changed,
                    source="accepted",
                )
                state.rounds_without_improvement = 0
                if self.progress is not None:
                    self.progress.event(
                        "Round {0} accepted: score={1:.6f} hypothesis={2}".format(
                            result.round_index,
                            result.score,
                            result.hypothesis,
                        )
                    )
            else:
                state.rounds_without_improvement += 1
                if branch_exists(self.repo, result.branch):
                    delete_branch(self.repo, result.branch)
                if self.progress is not None:
                    score_text = "n/a" if result.score is None else "{0:.6f}".format(result.score)
                    self.progress.event(
                        "Round {0} {1}: score={2} hypothesis={3}".format(
                            result.round_index,
                            result.status,
                            score_text,
                            result.hypothesis,
                        )
                    )

            state.current_round += 1
            state.phase = "idle"
            state.pending_candidate = None
            state.updated_at = now_iso()
            self._write_state(config, state)

        if state.status == "running":
            state.status = "completed"
        state.phase = "idle"
        state.updated_at = now_iso()
        self._write_state(config, state)
        if self.progress is not None:
            final_score = "n/a" if state.champion is None else "{0:.6f}".format(state.champion.score)
            self.progress.finish("Run {0} finished with status={1}, champion score={2}".format(
                state.run_id,
                state.status,
                final_score,
            ))
        return state

    def status(self, run_id: Optional[str] = None) -> RunState:
        return self._load_state(load_project_config(self.repo), run_id)

    def report(self, run_id: Optional[str] = None) -> Tuple[RunState, List[Dict[str, str]]]:
        config = load_project_config(self.repo)
        state = self._load_state(config, run_id)
        return state, read_results(self._results_path(config, state.run_id))

    def _evaluate_baseline(self, config: ProjectConfig, state: RunState) -> ChampionState:
        artifact_dir = self._round_dir(config, state.run_id, 0) / "baseline"
        evaluation = run_evaluator(
            self.repo,
            config.evaluator,
            artifact_dir,
            progress=self.progress,
            stage_prefix="Baseline evaluator",
            context_env=build_evaluator_context_env(
                run_id=state.run_id,
                round_index=0,
                artifact_dir=artifact_dir,
                artifacts_root=self.repo / config.git.artifacts_dir,
                worktree=self.repo,
                base_branch=state.base_branch,
                champion_branch=state.base_branch,
            ),
        )
        if not evaluation.passed or evaluation.score is None:
            raise RuntimeError("Baseline evaluation failed: {0}".format(evaluation.failure_reason))
        baseline = ChampionState(
            branch=state.base_branch,
            commit=head_commit(self.repo),
            score=evaluation.score,
            summary="Baseline",
            files_changed=0,
            source="baseline",
        )
        append_results(
            self._results_path(config, state.run_id),
            [
                {
                    "run_id": state.run_id,
                    "round": "0",
                    "parent_branch": state.base_branch,
                    "branch": state.base_branch,
                    "commit": baseline.commit,
                    "score": "{0:.6f}".format(baseline.score),
                    "status": "baseline",
                    "files_changed": "0",
                    "hypothesis": "Baseline score measurement.",
                    "summary": "Baseline",
                }
            ],
        )
        append_results(
            self._global_results_path(config),
            [
                {
                    "run_id": state.run_id,
                    "round": "0",
                    "parent_branch": state.base_branch,
                    "branch": state.base_branch,
                    "commit": baseline.commit,
                    "score": "{0:.6f}".format(baseline.score),
                    "status": "baseline",
                    "files_changed": "0",
                    "hypothesis": "Baseline score measurement.",
                    "summary": "Baseline",
                }
            ],
        )
        (artifact_dir / "baseline.json").write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")
        return baseline

    def _plan_round(self, config: ProjectConfig, state: RunState) -> Dict[str, str]:
        round_index = state.current_round
        return {
            "branch": "{prefix}/{run}/r{round:03d}".format(
                prefix=BRANCH_PREFIX,
                run=state.run_id,
                round=round_index,
            ),
            "worktree": str(self._worktree_path(config, state.run_id, round_index)),
            "artifact_dir": str(self._worker_dir(config, state.run_id, round_index)),
        }

    def _run_candidate(
        self,
        config: ProjectConfig,
        run_id: str,
        round_index: int,
        base_branch: str,
        champion: ChampionState,
        branch_name: str,
        worktree_path: Path,
        artifact_dir: Path,
        history_rows: List[Dict[str, str]],
    ) -> CandidateResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        create_worktree(self.repo, worktree_path, branch_name, champion.branch)
        if self.progress is not None:
            self.progress.event(
                "Round {0}: candidate branch {1}".format(round_index, branch_name)
            )
        prompt = build_worker_prompt(
            worktree_path,
            program=program_text(self.repo),
            config=config,
            run_id=run_id,
            round_index=round_index,
            champion=champion,
            branch_name=branch_name,
            history_rows=history_rows,
        )
        (artifact_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        status = "failed"
        score = None
        commit = None
        hypothesis = "No hypothesis provided."
        summary = "No summary provided."
        files_changed = 0
        try:
            worker_name = worker_display_name(config.worker)
            invocation = run_worker(
                worktree_path,
                artifact_dir,
                prompt,
                config.worker,
                progress=self.progress,
                stage_message="Round {0}: {1} working on {2}".format(round_index, worker_name, branch_name),
            )
            if self.progress is not None:
                self.progress.finalize_live_usage(invocation.usage)
                self.progress.end_phase()
            if invocation.structured_output and invocation.structured_output.get("hypothesis"):
                hypothesis = str(invocation.structured_output["hypothesis"])
            if invocation.structured_output and invocation.structured_output.get("summary"):
                summary = str(invocation.structured_output["summary"])
            changed_paths = tracked_changes(worktree_path)
            files_changed = len(changed_paths)
            if invocation.returncode != 0:
                summary = "{0} worker failed. {1}".format(worker_name, summary)
            elif hypothesis_seen_before(history_rows, hypothesis):
                status = "duplicate"
                summary = "Rejected as duplicate hypothesis. {0}".format(summary)
            elif not changed_paths:
                status = "unchanged"
            else:
                evaluation = run_evaluator(
                    worktree_path,
                    config.evaluator,
                    artifact_dir,
                    progress=self.progress,
                    stage_prefix="Round {0} evaluator".format(round_index),
                    context_env=build_evaluator_context_env(
                        run_id=run_id,
                        round_index=round_index,
                        artifact_dir=artifact_dir,
                        artifacts_root=self.repo / config.git.artifacts_dir,
                        worktree=worktree_path,
                        base_branch=base_branch,
                        champion_branch=champion.branch,
                        champion_score=champion.score,
                    ),
                )
                if not evaluation.passed or evaluation.score is None:
                    summary = evaluation.failure_reason or summary
                else:
                    score = evaluation.score
                    if is_better(score, champion.score, config):
                        stage_paths(worktree_path, changed_paths)
                        commit = commit_paths(worktree_path, "{0} round {1}".format(APP_NAME, round_index))
                        status = "accepted"
                    else:
                        status = "rejected"
            (artifact_dir / "result.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "hypothesis": hypothesis,
                        "summary": summary,
                        "commit": commit,
                        "score": score,
                        "files_changed": files_changed,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return CandidateResult(
                run_id=run_id,
                round_index=round_index,
                parent_branch=champion.branch,
                branch=branch_name,
                commit=commit,
                score=score,
                status=status,
                files_changed=files_changed,
                hypothesis=hypothesis,
                summary=summary,
                artifact_dir=str(artifact_dir),
            )
        finally:
            remove_worktree(self.repo, worktree_path)

    def _cleanup_pending_candidate(self, state: RunState) -> None:
        pending = state.pending_candidate or {}
        worktree = pending.get("worktree")
        branch = pending.get("branch")
        if worktree:
            remove_worktree(self.repo, Path(worktree))
        if branch and branch_exists(self.repo, branch):
            delete_branch(self.repo, branch)

    def _result_row(self, result: CandidateResult) -> Dict[str, str]:
        return {
            "run_id": result.run_id,
            "round": str(result.round_index),
            "parent_branch": result.parent_branch,
            "branch": result.branch,
            "commit": result.commit or "",
            "score": "" if result.score is None else "{0:.6f}".format(result.score),
            "status": result.status,
            "files_changed": str(result.files_changed),
            "hypothesis": result.hypothesis,
            "summary": result.summary,
        }

    def _append_result(self, config: ProjectConfig, run_id: str, result: CandidateResult) -> None:
        row = self._result_row(result)
        append_results(self._results_path(config, run_id), [row])
        append_results(self._global_results_path(config), [row])

    def _load_or_create_state(self, config: ProjectConfig, run_id: Optional[str]) -> RunState:
        if run_id:
            existing = self._find_run_id(config, run_id)
            if existing is not None:
                return self._read_state(config, existing)
            return self._create_state(config, run_id, self._latest_seed_source_state(config))
        active_run_id = self._find_active_run_id(config)
        if active_run_id is not None:
            return self._read_state(config, active_run_id)
        return self._create_state(config, make_run_id(), self._latest_seed_source_state(config))

    def _create_state(
        self,
        config: ProjectConfig,
        run_id: str,
        seed_state: Optional[RunState] = None,
    ) -> RunState:
        champion = self._seed_champion(run_id, seed_state)
        base_branch = champion.branch if champion is not None else determine_base_branch(self.repo, config.git.base_branch)
        state = RunState(
            run_id=run_id,
            created_at=now_iso(),
            updated_at=now_iso(),
            repo_path=str(self.repo),
            status="created",
            phase="idle",
            base_branch=base_branch,
            current_round=1,
            rounds_without_improvement=0,
            champion=champion,
            pending_candidate=None,
        )
        self._write_state(config, state)
        ensure_results_file(self._results_path(config, run_id))
        ensure_results_file(self._global_results_path(config))
        if champion is not None and seed_state is not None:
            row = {
                "run_id": run_id,
                "round": "0",
                "parent_branch": champion.branch,
                "branch": champion.branch,
                "commit": champion.commit,
                "score": "{0:.6f}".format(champion.score),
                "status": "baseline",
                "files_changed": str(champion.files_changed),
                "hypothesis": "Seeded from previous champion.",
                "summary": "Seeded from run {0}: {1}".format(seed_state.run_id, champion.summary),
            }
            append_results(self._results_path(config, run_id), [row])
            append_results(self._global_results_path(config), [row])
            artifact_dir = self._round_dir(config, run_id, 0) / "baseline"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_payload = champion.to_dict()
            artifact_payload["seeded_from_run"] = seed_state.run_id
            artifact_payload["seeded_from_branch"] = seed_state.champion.branch if seed_state.champion else ""
            (artifact_dir / "baseline.json").write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
        return state

    def _seed_champion(self, run_id: str, seed_state: Optional[RunState]) -> Optional[ChampionState]:
        if seed_state is None or seed_state.champion is None:
            return None
        source = seed_state.champion
        branch = source.branch
        if branch_exists(self.repo, branch):
            if head_commit(self.repo, branch) != source.commit:
                branch = "{0}/{1}/seed".format(BRANCH_PREFIX, run_id)
                self._ensure_branch_points_to_commit(branch, source.commit)
        else:
            branch = "{0}/{1}/seed".format(BRANCH_PREFIX, run_id)
            self._ensure_branch_points_to_commit(branch, source.commit)
        return ChampionState(
            branch=branch,
            commit=source.commit,
            score=source.score,
            summary=source.summary,
            files_changed=source.files_changed,
            source="seeded",
        )

    def _ensure_branch_points_to_commit(self, branch: str, commit: str) -> None:
        if branch_exists(self.repo, branch):
            if head_commit(self.repo, branch) != commit:
                raise GitError(
                    "Seed branch {0} already exists at a different commit".format(branch)
                )
            return
        create_branch(self.repo, branch, commit)

    def _load_state(self, config: ProjectConfig, run_id: Optional[str]) -> RunState:
        existing = self._find_run_id(config, run_id)
        if existing is None:
            raise FileNotFoundError("No runs found for repository {0}".format(self.repo))
        return self._read_state(config, existing)

    def _find_active_run_id(self, config: ProjectConfig) -> Optional[str]:
        for run_id in self._list_run_ids(config):
            state = self._read_state(config, run_id)
            if state.status in {"created", "running"}:
                return run_id
        return None

    def _latest_seed_source_state(self, config: ProjectConfig) -> Optional[RunState]:
        for run_id in self._list_run_ids(config):
            state = self._read_state(config, run_id)
            if state.champion is not None:
                return state
        return None

    def _list_run_ids(self, config: ProjectConfig) -> List[str]:
        runs_dir = self._runs_dir(config)
        if not runs_dir.exists():
            return []
        return [
            path.name
            for path in sorted(
                [path for path in runs_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
                reverse=True,
            )
        ]

    def _find_run_id(self, config: ProjectConfig, run_id: Optional[str]) -> Optional[str]:
        runs_dir = self._runs_dir(config)
        if run_id:
            state_path = runs_dir / run_id / "state.json"
            return run_id if state_path.exists() else None
        run_ids = self._list_run_ids(config)
        if not run_ids:
            return None
        for run_id in run_ids:
            state = self._read_state(config, run_id)
            if state.status in {"created", "running"}:
                return run_id
        return run_ids[0]

    def _read_state(self, config: ProjectConfig, run_id: str) -> RunState:
        return RunState.from_dict(
            json.loads((self._runs_dir(config) / run_id / "state.json").read_text(encoding="utf-8"))
        )

    def _write_state(self, config: ProjectConfig, state: RunState) -> None:
        path = self._runs_dir(config) / state.run_id / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def _runs_dir(self, config: ProjectConfig) -> Path:
        return self.repo / config.git.artifacts_dir / "runs"

    def _results_path(self, config: ProjectConfig, run_id: str) -> Path:
        return self._runs_dir(config) / run_id / "results.tsv"

    def _global_results_path(self, config: ProjectConfig) -> Path:
        return self.repo / config.git.artifacts_dir / "results.tsv"

    def _history_rows(self, config: ProjectConfig) -> List[Dict[str, str]]:
        return read_results(self._global_results_path(config))

    def _round_dir(self, config: ProjectConfig, run_id: str, round_index: int) -> Path:
        return self._runs_dir(config) / run_id / "rounds" / "round-{0:03d}".format(round_index)

    def _worker_dir(self, config: ProjectConfig, run_id: str, round_index: int) -> Path:
        return self._round_dir(config, run_id, round_index) / "candidate"

    def _worktree_path(self, config: ProjectConfig, run_id: str, round_index: int) -> Path:
        return self.repo / config.git.artifacts_dir / "worktrees" / run_id / "r{0:03d}".format(round_index)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog=CLI_NAME)
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Scaffold target repo files")
    init_parser.add_argument("--repo", required=True, help="Path to the target repository")
    init_parser.add_argument("--force", action="store_true", help="Overwrite scaffold files")

    run_parser = subparsers.add_parser("run", help="Run or resume the improvement loop")
    run_parser.add_argument("--repo", required=True, help="Path to the target repository")
    run_parser.add_argument("--run-id", help="Resume a specific run id")

    status_parser = subparsers.add_parser("status", help="Show latest run status")
    status_parser.add_argument("--repo", required=True, help="Path to the target repository")
    status_parser.add_argument("--run-id", help="Show a specific run id")

    report_parser = subparsers.add_parser("report", help="Generate a markdown report")
    report_parser.add_argument("--repo", required=True, help="Path to the target repository")
    report_parser.add_argument("--run-id", help="Report a specific run id")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            repo = Path(args.repo).expanduser()
            ensure_project_files(repo, force=args.force)
            ensure_git_repo(repo)
            print("Initialized target repo at {0}".format(repo.resolve()))
            print("Created {0} and {1}".format(repo / PROGRAM_FILENAME, repo / CONFIG_FILENAME))
            return 0
        if args.command == "run":
            repo = Path(args.repo).expanduser()
            created = scaffold_missing_project_files(repo)
            ensure_git_repo(repo)
            if created:
                print("Scaffolded missing project files:", file=sys.stderr)
                for path in created:
                    print("- {0}".format(path), file=sys.stderr)
                print(
                    "Edit them and rerun `{0} run --repo {1}` (or `python3 run.py run --repo {1}` for local development).".format(
                        CLI_NAME,
                        repo,
                    ),
                    file=sys.stderr,
                )
                return 1
            load_project_config(repo)
            progress = ProgressReporter()
            print(render_status(Orchestrator(repo, progress=progress).run(run_id=args.run_id)))
            return 0
        if args.command == "status":
            repo = Path(args.repo).expanduser()
            print(render_status(Orchestrator(repo).status(run_id=args.run_id)))
            return 0
        if args.command == "report":
            repo = Path(args.repo).expanduser()
            state, results = Orchestrator(repo).report(run_id=args.run_id)
            print(render_report(state, results))
            return 0
    except Exception as exc:  # pragma: no cover
        print("error: {0}".format(exc), file=sys.stderr)
        return 1
    parser.print_help()
    return 1


def _write_if_needed(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.write_text(content, encoding="utf-8")


def _empty_to_none(value):
    if value in ("", None):
        return None
    return value


def git_commit_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", APP_NAME)
    env.setdefault("GIT_AUTHOR_EMAIL", "evoloza@example.com")
    env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    return env


def run_git(repo: Path, *args: str, env: Optional[Dict[str, str]] = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def minutes_elapsed(started_at: float) -> float:
    return (time.monotonic() - started_at) / 60.0


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "{0:02d}:{1:02d}:{2:02d}".format(hours, minutes, secs)
    return "{0:02d}:{1:02d}".format(minutes, secs)


def format_token_count(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        value = count / 1000.0
        if value < 10:
            return "{0:.1f}".format(value).rstrip("0").rstrip(".") + "k"
        return "{0:.0f}k".format(value)
    value = count / 1_000_000.0
    return "{0:.1f}".format(value).rstrip("0").rstrip(".") + "M"


def truncate_middle(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    left = (max_width - 3) // 2
    right = max_width - 3 - left
    return text[:left] + "..." + text[-right:]


def compact_progress_message(message: str) -> str:
    round_worker = re.match(r"^Round (\d+): ([A-Za-z0-9_-]+) working on .*/(r\d+)$", message)
    if round_worker:
        return "r{0} {1} {2}".format(
            round_worker.group(1),
            round_worker.group(2).lower(),
            round_worker.group(3),
        )
    round_eval = re.match(r"^Round (\d+) evaluator (\d+)/(\d+):", message)
    if round_eval:
        return "r{0} eval {1}/{2}".format(
            round_eval.group(1),
            round_eval.group(2),
            round_eval.group(3),
        )
    baseline_eval = re.match(r"^Baseline evaluator (\d+)/(\d+):", message)
    if baseline_eval:
        return "baseline eval {0}/{1}".format(
            baseline_eval.group(1),
            baseline_eval.group(2),
        )
    return message


def parse_usage_from_jsonl(text: str) -> Optional[Dict[str, int]]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
    }
    found = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        usage = record.get("usage")
        if record.get("type") != "turn.completed" or not isinstance(usage, dict):
            continue
        found = True
        for key in totals:
            value = usage.get(key, 0)
            try:
                totals[key] += int(value)
            except (TypeError, ValueError):
                continue
    return totals if found else None


def parse_live_usage_from_session_line(line: str) -> Optional[Dict[str, int]]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or record.get("type") != "event_msg":
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "token_count":
        return None
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    total_usage = info.get("total_token_usage")
    if not isinstance(total_usage, dict):
        return None
    return normalize_token_usage(total_usage)


def normalize_token_usage(payload: Dict[str, Any]) -> Dict[str, int]:
    normalized = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        value = payload.get(key, 0)
        try:
            normalized[key] = int(value)
        except (TypeError, ValueError):
            normalized[key] = 0
    return normalized


def find_codex_session_file(worktree: str, started_at_wall: float, sessions_root: Path) -> Optional[Path]:
    candidates = []
    for path in sessions_root.glob("*/*/*/rollout-*.jsonl"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime < started_at_wall - 5:
            continue
        candidates.append((stat.st_mtime, path))
    for _, path in sorted(candidates, reverse=True):
        if session_file_matches_worktree(path, worktree):
            return path
    return None


def session_file_matches_worktree(path: Path, worktree: str) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
    except OSError:
        return False
    if not first_line:
        return False
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return False
    if not isinstance(record, dict) or record.get("type") != "session_meta":
        return False
    payload = record.get("payload")
    return isinstance(payload, dict) and payload.get("cwd") == worktree


def parse_live_phase_from_session_line(line: str) -> Optional[Tuple[str, Optional[str]]]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    record_type = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if record_type == "response_item":
        payload_type = payload.get("type")
        if payload_type == "reasoning":
            return ("thinking", "reasoning")
        if payload_type == "function_call" and payload.get("name") == "exec_command":
            arguments = parse_session_call_arguments(payload.get("arguments"))
            command = str(arguments.get("cmd", "")).strip()
            if not command:
                return ("thinking", "work")
            return (classify_command_phase(command), summarize_command_action(command))
        if payload_type == "custom_tool_call" and payload.get("name") == "apply_patch":
            return ("editing", "apply patch")
    if record_type == "event_msg" and payload.get("type") == "agent_message":
        return ("finalizing", "final answer")
    return None


def parse_session_call_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def classify_command_phase(command: str) -> str:
    lowered = command.lower()
    if "benchmark.py" in lowered or "evoloza_score" in lowered or "autoresearch_score" in lowered:
        return "benchmarking"
    if "pytest" in lowered or "unittest" in lowered or "cargo test" in lowered or "npm test" in lowered:
        return "testing"
    if "apply_patch" in lowered or "perl -0pi" in lowered or "perl -pi" in lowered or "sed -i" in lowered:
        return "editing"
    if (
        "rg --files" in lowered
        or re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)rg\b", lowered)
        or "sed -n" in lowered
        or re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)cat\b", lowered)
        or re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)ls\b", lowered)
        or re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)find\b", lowered)
        or re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)wc\b", lowered)
        or "git show" in lowered
        or "git diff" in lowered
        or "git status" in lowered
    ):
        return "reading"
    if "python3 - <<" in lowered or "python - <<" in lowered:
        return "thinking"
    return "thinking"


def summarize_command_action(command: str) -> str:
    lowered = command.lower()
    if "python3 -m unittest" in lowered or "python -m unittest" in lowered:
        return "unittest"
    if "pytest" in lowered:
        return "pytest"
    if "benchmark.py" in lowered:
        return "benchmark.py"
    if "rg --files" in lowered:
        return "list files"
    if re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)rg\b", lowered):
        return "search"
    if re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)ls\b", lowered):
        return "list files"
    if re.search(r"(^|[;&|]\s*|&&\s*|\|\|\s*)find\b", lowered):
        return "find files"
    edit_match = re.search(
        r"(?:perl -0pi -e [^\n]+\s+|perl -pi -e [^\n]+\s+|sed -i(?:\s+\S+)?\s+)([A-Za-z0-9_./-]+)",
        command,
    )
    if edit_match:
        return "edit {0}".format(edit_match.group(1))
    diff_match = re.search(r"git diff --\s+([A-Za-z0-9_./-]+)", command)
    if diff_match:
        return "diff {0}".format(diff_match.group(1))
    wc_match = re.search(r"wc -c\s+([A-Za-z0-9_./-]+)", command)
    if wc_match:
        return "count bytes {0}".format(wc_match.group(1))
    read_match = re.search(
        r"(?:sed -n '[^']+'\s+|cat\s+|git show HEAD:)([A-Za-z0-9_./-]+)",
        command,
    )
    if read_match:
        return "read {0}".format(read_match.group(1))
    if "python3 - <<" in lowered or "python - <<" in lowered:
        return "explore variants"
    return truncate_middle(command, 24)


def progress_context_label(message: str) -> Optional[str]:
    round_match = re.match(r"^Round (\d+)", message)
    if round_match:
        return "r{0}".format(round_match.group(1))
    if message.startswith("Baseline"):
        return "baseline"
    return None


def is_better(candidate: float, baseline: float, config: ProjectConfig) -> bool:
    if config.evaluator.direction == "maximize":
        return candidate > baseline
    return candidate < baseline


def render_history_for_prompt(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "- No previous experiments yet."
    lines = []
    for row in rows:
        lines.append(
            "- run={run_id} round={round} status={status} score={score} hypothesis={hypothesis} summary={summary}".format(
                run_id=row.get("run_id", ""),
                round=row.get("round", ""),
                status=row.get("status", ""),
                score=row.get("score", ""),
                hypothesis=row.get("hypothesis", ""),
                summary=row.get("summary", ""),
            )
        )
    return "\n".join(lines)


def normalize_hypothesis(text: str) -> str:
    return " ".join(text.lower().split())


def hypothesis_seen_before(rows: List[Dict[str, str]], hypothesis: str) -> bool:
    normalized = normalize_hypothesis(hypothesis)
    if not normalized or normalized == normalize_hypothesis("No hypothesis provided."):
        return False
    for row in rows:
        previous = normalize_hypothesis(row.get("hypothesis", ""))
        if previous and previous == normalized:
            return True
    return False


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
