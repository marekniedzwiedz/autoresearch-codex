from __future__ import annotations

import argparse
import csv
import difflib
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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
HYPOTHESIS_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "use",
    "with",
    "experiment",
    "candidate",
    "patch",
    "improve",
    "improved",
    "improving",
    "improvement",
    "engine",
    "search",
    "efficiency",
    "better",
    "added",
    "adding",
    "introduce",
    "introduced",
    "adjust",
    "adjusted",
    "tune",
    "tuned",
}
HYPOTHESIS_COMMON_OVERLAP_TOKENS = {
    "add",
    "bonus",
    "improv",
    "order",
    "score",
    "search",
}
HYPOTHESIS_THEME_PATTERNS = (
    ("continuation-history", ("continuation", "history")),
    ("quiet-history", ("quiet", "history")),
    ("low-ply-history", ("low", "ply", "history")),
    ("pawn-history", ("pawn", "history")),
    ("capture-history", ("capture", "history")),
    ("correction-history", ("correction", "history")),
    ("move-ordering", ("move", "order")),
    ("late-move-reduction", ("late", "move", "reduction")),
    ("lmr", ("lmr",)),
    ("null-move-pruning", ("null", "move", "prun")),
    ("futility-pruning", ("futility", "prun")),
    ("reverse-futility-pruning", ("reverse", "futility")),
    ("delta-pruning", ("delta", "prun")),
    ("singular-extension", ("singular", "extension")),
    ("aspiration-window", ("aspiration", "window")),
    ("transposition-table", ("transposition", "table")),
    ("tt-probe", ("tt", "probe")),
    ("killer-moves", ("killer", "move")),
    ("quiescence-search", ("quiescence", "search")),
    ("see-pruning", ("see", "prun")),
)
HYPOTHESIS_THEME_TOKEN_SEQUENCE_REGEXES = {
    "quiet-history": (
        re.compile(r"\bquiet(?:\s+\w+){0,2}\s+history\b"),
        re.compile(r"\bhistory(?:\s+\w+){0,4}\s+quiet(?:\s+\w+){0,1}\s+move\b"),
    ),
}
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
    "required": ["hypothesis", "summary", "files_touched", "local_checks_run", "risks", "patch"],
    "additionalProperties": False,
}
OLLAMA_WORKER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "summary": {"type": "string"},
        "patch": {"type": "string"},
    },
    "required": ["hypothesis", "patch"],
    "additionalProperties": False,
}
EXECUTOR_EDIT_ACTIONS = ("replace_block", "insert_before", "insert_after")
EXECUTOR_EDIT_OP_SCHEMA = {
    "type": "object",
    "properties": {
        "file": {"type": "string"},
        "symbol": {"type": "string"},
        "action": {"type": "string", "enum": list(EXECUTOR_EDIT_ACTIONS)},
        "anchor_snippet": {"type": "string"},
        "occurrence": {"type": "integer"},
        "new_text": {"type": "string"},
    },
    "required": ["file", "action", "anchor_snippet", "new_text"],
    "additionalProperties": False,
}
EXECUTOR_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "summary": {"type": "string"},
        "edit_ops": {"type": "array", "items": EXECUTOR_EDIT_OP_SCHEMA},
    },
    "required": ["hypothesis", "edit_ops"],
    "additionalProperties": False,
}
PLAN_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "hypothesis": {"type": "string"},
        "summary": {"type": "string"},
        "target_file": {"type": "string"},
        "target_symbols": {"type": "array", "items": {"type": "string"}},
        "anchor_snippets": {"type": "array", "items": {"type": "string"}},
        "allowed_edit_scope": {"type": "string"},
        "forbidden_families": {"type": "array", "items": {"type": "string"}},
        "implementation_notes": {"type": "string"},
        "max_patch_lines": {"type": "integer"},
    },
    "required": [
        "id",
        "hypothesis",
        "summary",
        "target_file",
        "target_symbols",
        "anchor_snippets",
        "allowed_edit_scope",
        "forbidden_families",
        "implementation_notes",
        "max_patch_lines",
    ],
    "additionalProperties": False,
}
PLAN_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {"type": "array", "items": PLAN_CARD_SCHEMA},
    },
    "required": ["cards"],
    "additionalProperties": False,
}
SUPPORTED_WORKER_BACKENDS = {"codex", "ollama"}
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TEMPERATURE = 0.2
DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS = 3600
DEFAULT_OLLAMA_PATCH_REPAIR_ATTEMPTS = 1
DEFAULT_CONTEXT_MAX_BYTES = 120000
DEFAULT_CONTEXT_FILE_BYTES = 24000
DEFAULT_CONTEXT_FILE_COUNT = 24
DEFAULT_PLANNER_CARD_COUNT = 8
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
# Ollama backend only: request controls.
keep_alive = ""
request_timeout_seconds = 3600
# Ollama thinking is model-dependent. Leave unset to use the model default.
# Set to false for coding-style patch generation when you do not want traces.
# think = false
# Optional repo-configured hypothesis seeds to treat as off-limits.
# forbidden_hypotheses = ["continuation history for quiet ordering"]
# Ollama backend only: extra /api/generate options such as num_ctx or seed.
# [worker.ollama_options]
# num_ctx = 131072
# num_predict = 512

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
# keep_alive = "30m"
# request_timeout_seconds = 3600
# think = false
# forbidden_hypotheses = ["continuation history for quiet ordering"]
# [worker.ollama_options]
# num_ctx = 131072
# num_predict = 512

# Optional planner-only settings for `evoloza plan`.
# Falls back to `[worker]` when omitted.
# [planner]
# backend = "codex"
# binary = "codex"
# model = "gpt-5.4"
# cards_per_plan = 8
#
# Optional executor-only settings for `evoloza execute`.
# Falls back to `[worker]` when omitted.
# [executor]
# backend = "ollama"
# model = "qwen3.5:35b"
# keep_alive = "30m"
# think = false
# [executor.ollama_options]
# num_ctx = 131072
# num_predict = 1024

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
# Copy each evaluated candidate worktree into the round artifacts before cleanup
# so the exact source tree and built binaries remain inspectable later.
preserve_candidate_worktrees = false
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
    keep_alive: Optional[Any] = None
    think: Optional[bool] = None
    request_timeout_seconds: int = DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS
    forbidden_hypotheses: List[str] = field(default_factory=list)
    ollama_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerSettings:
    worker: WorkerSettings = field(default_factory=WorkerSettings)
    cards_per_plan: int = DEFAULT_PLANNER_CARD_COUNT


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
    preserve_candidate_worktrees: bool = False


@dataclass
class ProjectConfig:
    worker: WorkerSettings
    planner: PlannerSettings
    executor: WorkerSettings
    evaluator: EvaluatorSettings
    search: SearchSettings
    git: GitSettings


@dataclass
class ExperimentCard:
    id: str
    hypothesis: str
    summary: str
    target_file: str
    target_symbols: List[str]
    anchor_snippets: List[str]
    allowed_edit_scope: str
    forbidden_families: List[str]
    implementation_notes: str
    max_patch_lines: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentCard":
        payload = dict(data)
        payload["target_symbols"] = [str(item) for item in payload.get("target_symbols", [])]
        payload["anchor_snippets"] = normalize_multiline_string_list(payload.get("anchor_snippets"))
        payload["forbidden_families"] = [str(item) for item in payload.get("forbidden_families", [])]
        payload["max_patch_lines"] = int(payload.get("max_patch_lines", 80))
        return cls(**payload)


@dataclass
class ExperimentPlan:
    plan_id: str
    created_at: str
    repo_path: str
    planner_backend: str
    planner_model: str
    program_path: str
    artifact_dir: str
    cards: List[ExperimentCard]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cards"] = [card.to_dict() for card in self.cards]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentPlan":
        payload = dict(data)
        payload["cards"] = [ExperimentCard.from_dict(item) for item in payload.get("cards", [])]
        return cls(**payload)


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
    preserved_worktree: Optional[str] = None


@dataclass
class UnifiedDiffHunk:
    old_start: int
    lines: List[str]


@dataclass
class UnifiedDiffFilePatch:
    relpath: str
    hunks: List[UnifiedDiffHunk]


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
    mode: str = "run"
    plan_path: Optional[str] = None
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


def resolve_cli_config_path(repo: Path, configured: Optional[Any]) -> Optional[Path]:
    if configured in ("", None):
        return None
    candidate = Path(str(configured)).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    return candidate


def resolve_cli_repo_path(repo: Path, configured: Optional[Any]) -> Optional[Path]:
    if configured in ("", None):
        return None
    candidate = Path(str(configured)).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    return candidate


def ensure_project_files(repo: Path, force: bool = False, config_path: Optional[Path] = None) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _write_if_needed(repo / PROGRAM_FILENAME, DEFAULT_PROGRAM, force)
    _write_if_needed(config_path or (repo / CONFIG_FILENAME), DEFAULT_CONFIG, force)


def scaffold_missing_project_files(repo: Path, config_path: Optional[Path] = None) -> List[Path]:
    created = []
    program_path = repo / PROGRAM_FILENAME
    target_config_path = config_path or (repo / CONFIG_FILENAME)
    had_program = program_path.exists()
    had_config = target_config_path.exists() if config_path is not None else (
        target_config_path.exists() or (repo / LEGACY_CONFIG_FILENAME).exists()
    )
    ensure_project_files(repo, force=False, config_path=target_config_path)
    if not had_program and program_path.exists():
        created.append(program_path)
    if not had_config and target_config_path.exists():
        created.append(target_config_path)
    return created


def _normalize_ollama_options(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("worker.ollama_options must be a TOML table")
    try:
        return json.loads(json.dumps(value))
    except TypeError as exc:
        raise ValueError("worker.ollama_options must contain only JSON-serializable values") from exc


def _normalize_ollama_keep_alive(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        return value
    raise ValueError("worker.keep_alive must be a string or number")


def _normalize_string_list(value: Any, field_name: str) -> List[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("{0} must be a TOML array".format(field_name))
    normalized = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def clone_worker_settings(settings: WorkerSettings) -> WorkerSettings:
    return WorkerSettings(**asdict(settings))


def _load_worker_settings(
    section_name: str,
    section: Optional[Dict[str, Any]],
    fallback_section: Optional[Dict[str, Any]] = None,
    *,
    default_backend: str = "codex",
) -> WorkerSettings:
    section = dict(section or {})
    fallback_section = dict(fallback_section or {})

    def get_value(key: str, default: Any = None) -> Any:
        if key in section:
            return section.get(key)
        if key in fallback_section:
            return fallback_section.get(key)
        return default

    backend = str(get_value("backend", default_backend)).strip().lower() or default_backend
    if backend not in SUPPORTED_WORKER_BACKENDS:
        raise ValueError(
            "{0}.backend must be one of {1}".format(
                section_name,
                ", ".join(sorted(SUPPORTED_WORKER_BACKENDS)),
            )
        )

    extra_args = get_value("extra_args", [])
    if not isinstance(extra_args, list):
        raise ValueError("{0}.extra_args must be a TOML array".format(section_name))
    context_files = get_value("context_files", [])
    if not isinstance(context_files, list):
        raise ValueError("{0}.context_files must be a TOML array".format(section_name))
    keep_alive = _normalize_ollama_keep_alive(get_value("keep_alive"))
    think = get_value("think")
    if think is not None and not isinstance(think, bool):
        raise ValueError("{0}.think must be true or false".format(section_name))
    request_timeout_seconds = int(
        get_value("request_timeout_seconds", DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS)
    )
    if request_timeout_seconds <= 0:
        raise ValueError(
            "{0}.request_timeout_seconds must be a positive integer".format(section_name)
        )
    forbidden_hypotheses = _normalize_string_list(
        get_value("forbidden_hypotheses"),
        "{0}.forbidden_hypotheses".format(section_name),
    )
    ollama_options = _normalize_ollama_options(get_value("ollama_options"))

    return WorkerSettings(
        backend=backend,
        binary=str(get_value("binary", "codex" if backend == "codex" else "ollama")),
        model=_empty_to_none(get_value("model")),
        extra_args=[str(item) for item in extra_args],
        ollama_host=str(get_value("ollama_host", get_value("host", DEFAULT_OLLAMA_HOST))),
        context_files=[str(item) for item in context_files],
        max_context_bytes=int(get_value("max_context_bytes", DEFAULT_CONTEXT_MAX_BYTES)),
        max_file_bytes=int(get_value("max_file_bytes", DEFAULT_CONTEXT_FILE_BYTES)),
        max_files=int(get_value("max_files", DEFAULT_CONTEXT_FILE_COUNT)),
        temperature=float(get_value("temperature", DEFAULT_OLLAMA_TEMPERATURE)),
        keep_alive=keep_alive,
        think=think,
        request_timeout_seconds=request_timeout_seconds,
        forbidden_hypotheses=forbidden_hypotheses,
        ollama_options=ollama_options,
    )


def load_project_config(repo: Path, config_path: Optional[Path] = None) -> ProjectConfig:
    requested_config_path = config_path
    config_path = find_config_path(repo, config_path)
    if config_path is None:
        if requested_config_path is not None:
            raise FileNotFoundError("Missing config file: expected {0}".format(requested_config_path))
        raise FileNotFoundError(
            "Missing config file: expected {0} or {1}".format(
                repo / CONFIG_FILENAME,
                repo / LEGACY_CONFIG_FILENAME,
            )
        )
    data = loads_toml(config_path.read_text(encoding="utf-8"))
    legacy_codex_section = data.get("codex", {})
    worker_section = data.get("worker")
    planner_section = data.get("planner")
    executor_section = data.get("executor")
    evaluator_section = data.get("evaluator", {})
    search_section = data.get("search", {})
    git_section = data.get("git", {})
    preserve_candidate_worktrees = git_section.get("preserve_candidate_worktrees", False)
    if not isinstance(preserve_candidate_worktrees, bool):
        raise ValueError("git.preserve_candidate_worktrees must be a boolean")

    if worker_section is None:
        worker_settings = _load_worker_settings(
            "worker",
            legacy_codex_section,
            default_backend="codex",
        )
        worker_fallback_section = legacy_codex_section
    else:
        worker_settings = _load_worker_settings(
            "worker",
            worker_section,
            legacy_codex_section,
            default_backend="codex",
        )
        worker_fallback_section = worker_section

    commands = evaluator_section.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("evaluator.commands must be a non-empty TOML array")
    direction = evaluator_section.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("evaluator.direction must be 'maximize' or 'minimize'")
    if planner_section is None:
        planner_settings = PlannerSettings(worker=clone_worker_settings(worker_settings))
    else:
        cards_per_plan = int(planner_section.get("cards_per_plan", DEFAULT_PLANNER_CARD_COUNT))
        if cards_per_plan <= 0:
            raise ValueError("planner.cards_per_plan must be a positive integer")
        planner_settings = PlannerSettings(
            worker=_load_worker_settings(
                "planner",
                planner_section,
                worker_fallback_section,
                default_backend=worker_settings.backend,
            ),
            cards_per_plan=cards_per_plan,
        )
    if executor_section is None:
        executor_settings = clone_worker_settings(worker_settings)
    else:
        executor_settings = _load_worker_settings(
            "executor",
            executor_section,
            worker_fallback_section,
            default_backend=worker_settings.backend,
        )

    return ProjectConfig(
        worker=worker_settings,
        planner=planner_settings,
        executor=executor_settings,
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
            preserve_candidate_worktrees=preserve_candidate_worktrees,
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


def load_experiment_plan(path: Path) -> ExperimentPlan:
    return ExperimentPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def find_config_path(repo: Path, configured: Optional[Path] = None) -> Optional[Path]:
    if configured is not None:
        return configured if configured.exists() else None
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


def preserve_worktree_snapshot(
    worktree_path: Path,
    artifact_dir: Path,
    *,
    label: str = "preserved-worktree",
) -> Tuple[Optional[str], Optional[str]]:
    if not worktree_path.exists():
        return None, "worktree does not exist: {0}".format(worktree_path)
    snapshot_dir = artifact_dir / label
    metadata_path = artifact_dir / "{0}.json".format(label)
    error_path = artifact_dir / "{0}.error.txt".format(label)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    try:
        shutil.copytree(
            worktree_path,
            snapshot_dir,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        metadata = {
            "source_worktree": str(worktree_path),
            "snapshot_dir": str(snapshot_dir),
            "preserved_at": now_iso(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if error_path.exists():
            error_path.unlink()
        return str(snapshot_dir), None
    except Exception as exc:
        error_path.write_text(str(exc) + "\n", encoding="utf-8")
        return None, str(error_path)


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


def merge_line_ranges(ranges: List[Tuple[int, int, str]], gap: int = 2) -> List[Tuple[int, int, List[str]]]:
    if not ranges:
        return []
    merged: List[Tuple[int, int, List[str]]] = []
    for start, end, label in sorted(ranges, key=lambda item: (item[0], item[1], item[2])):
        if not merged or start > merged[-1][1] + gap:
            merged.append((start, end, [label]))
            continue
        prev_start, prev_end, prev_labels = merged[-1]
        if label not in prev_labels:
            prev_labels.append(label)
        merged[-1] = (prev_start, max(prev_end, end), prev_labels)
    return merged


def collect_card_focus_terms(card: ExperimentCard, limit: int = 16) -> List[str]:
    terms: List[str] = []
    seen = set()

    def add_term(term: str) -> None:
        candidate = str(term or "").strip()
        if not candidate:
            return
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
            return
        lowered = candidate.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        terms.append(candidate)

    for symbol in card.target_symbols:
        add_term(symbol)
    for snippet in card.anchor_snippets:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", snippet or ""):
            add_term(token)
            if len(terms) >= limit:
                return terms
    for field_text in (
        card.summary,
        card.allowed_edit_scope,
        card.implementation_notes,
        card.hypothesis,
    ):
        for snippet in re.findall(r"`([^`]+)`", field_text or ""):
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", snippet):
                add_term(token)
                if len(terms) >= limit:
                    return terms
    return terms


def find_symbol_occurrence_lines(lines: List[str], symbol: str, limit: int = 4) -> List[int]:
    if not symbol:
        return []
    pattern = re.compile(r"\b{0}\b".format(re.escape(symbol)))
    matches = []
    for index, line in enumerate(lines):
        if pattern.search(line):
            matches.append(index)
            if len(matches) >= limit:
                break
    return matches


def find_definition_lines(lines: List[str], symbol: str, limit: int = 2) -> List[int]:
    if not symbol:
        return []
    patterns = (
        re.compile(r"\bfn\s+{0}\b".format(re.escape(symbol))),
        re.compile(r"\b(?:struct|enum|trait|impl|const|static)\s+{0}\b".format(re.escape(symbol))),
    )
    matches = []
    for index, line in enumerate(lines):
        if any(pattern.search(line) for pattern in patterns):
            matches.append(index)
            if len(matches) >= limit:
                break
    return matches


def find_anchor_snippet_ranges(
    lines: List[str],
    anchor_snippet: str,
    limit: int = 2,
) -> List[Tuple[int, int]]:
    snippet_lines = str(anchor_snippet or "").strip("\n").splitlines()
    if not snippet_lines:
        return []
    matches = find_subsequence_matches(lines, snippet_lines)
    ranges = []
    for start in matches[:limit]:
        ranges.append((start, start + len(snippet_lines) - 1))
    return ranges


def render_focused_snapshot_entry(
    relpath: str,
    lines: List[str],
    regions: List[Tuple[int, int, List[str]]],
    truncated: bool,
) -> str:
    blocks = ["=== FILE: {0} (focused excerpts) ===".format(relpath)]
    for start, end, labels in regions:
        label_text = ", ".join(labels)
        blocks.append(
            "--- EXCERPT: lines {0}-{1} | focus: {2} ---".format(
                start + 1,
                end + 1,
                label_text,
            )
        )
        blocks.append("\n".join(lines[start : end + 1]).rstrip())
        blocks.append("--- END EXCERPT ---")
    if truncated:
        blocks.append("[additional focused excerpts omitted]")
    blocks.append("=== END FILE ===")
    return "\n".join(blocks).rstrip() + "\n"


def build_focused_target_snapshot_entry(
    worktree: Path,
    relpath: str,
    card: ExperimentCard,
    settings: WorkerSettings,
) -> Optional[str]:
    path = worktree / relpath
    if not path.is_file() or not is_probably_text_file(path):
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = content.splitlines()
    if not lines:
        return render_repo_snapshot_entry(relpath, "", False)

    focus_terms = collect_card_focus_terms(card)
    ranges: List[Tuple[int, int, str]] = []
    for symbol in card.target_symbols:
        for index in find_definition_lines(lines, symbol):
            ranges.append((max(0, index - 20), min(len(lines) - 1, index + 28), "{0} definition".format(symbol)))
    for snippet_index, snippet in enumerate(card.anchor_snippets, start=1):
        for start, end in find_anchor_snippet_ranges(lines, snippet):
            ranges.append((max(0, start - 10), min(len(lines) - 1, end + 12), "anchor {0}".format(snippet_index)))
    for term in focus_terms:
        for index in find_symbol_occurrence_lines(lines, term):
            ranges.append((max(0, index - 14), min(len(lines) - 1, index + 18), term))

    merged = merge_line_ranges(ranges)
    if not merged:
        rendered_content, truncated = truncate_text_to_bytes(content, settings.max_file_bytes)
        return render_repo_snapshot_entry(relpath, rendered_content, truncated)

    entry_budget = max(
        4096,
        min(settings.max_context_bytes, max(settings.max_file_bytes, settings.max_context_bytes // 2)),
    )
    selected_regions: List[Tuple[int, int, List[str]]] = []
    truncated = False
    for region in merged:
        candidate = render_focused_snapshot_entry(relpath, lines, selected_regions + [region], False)
        if len(candidate.encode("utf-8")) > entry_budget:
            truncated = True
            break
        selected_regions.append(region)
    if not selected_regions:
        rendered_content, reduced_truncated = truncate_text_to_bytes(content, entry_budget)
        return render_repo_snapshot_entry(relpath, rendered_content, reduced_truncated)
    return render_focused_snapshot_entry(relpath, lines, selected_regions, truncated)


def build_repo_snapshot(
    worktree: Path,
    config: ProjectConfig,
    program: str,
    exclude_paths: Optional[List[str]] = None,
) -> Tuple[str, List[str], List[str]]:
    all_paths = [line.strip() for line in run_git(worktree, "ls-files").splitlines() if line.strip()]
    included_entries: List[str] = []
    included_paths: List[str] = []
    omitted_paths: List[str] = []
    total_bytes = 0
    artifact_prefix = config.git.artifacts_dir.rstrip("/") + "/"
    hint_text = (program + "\n" + "\n".join(config.evaluator.commands)).lower()
    candidates = []
    excluded = {item.replace(os.sep, "/") for item in (exclude_paths or []) if str(item).strip()}

    for relpath in all_paths:
        if relpath.startswith(".git/") or relpath.startswith(artifact_prefix):
            continue
        if relpath in excluded:
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
    forbidden_repeat_block = render_forbidden_repeat_guidance(
        history_rows,
        config.worker.forbidden_hypotheses,
    )
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

    Repo-configured and already-explored idea families you must avoid repeating:
    {forbidden_repeat_block}

    Official evaluator commands:
    {commands}

    Hard rules:
    - You cannot execute commands or inspect the filesystem directly.
    - Use only the repository snapshot included below.
    - Return exactly one JSON object and nothing else.
    - The JSON object may only include: hypothesis, summary, patch.
    - The JSON object must include: hypothesis and patch.
    - The `patch` field must be a unified git diff that can be applied with `git apply`.
    - Only modify existing text files whose contents are included below, unless you are creating a small supporting text file.
    - If the snapshot is not sufficient, return an empty patch and explain the blocker briefly in `summary`.
    - Keep the change focused on one experiment.
    - Keep the JSON compact. Do not spend tokens on long explanations.
    - Before patching, verify that the targeted code and the intended missing behavior are actually absent from the snapshot.
    - If the intended logic already exists in the snapshot, do not emit a redundant patch. Choose a different focused idea or return an empty patch.
    - Treat the repo-configured and already-explored idea families above as off-limits for this round unless you are clearly changing the underlying mechanism.
    - If your first idea lands in one of those families, discard it and choose a different idea before writing the patch.
    - Every diff hunk must use exact context lines copied from the snapshot below.
    - Keep `hypothesis` to one short sentence.
    - Keep `summary` to at most one short sentence, or omit it entirely.
    - Prefer a complete valid JSON object with a small patch over detailed prose.
    - If output budget is tight, shorten or omit `summary` first, not `patch`.
    - Do not use markdown fences.

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
            forbidden_repeat_block=forbidden_repeat_block,
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


def render_experiment_card(card: ExperimentCard) -> str:
    lines = [
        "- id: {0}".format(card.id),
        "- hypothesis: {0}".format(card.hypothesis),
        "- summary: {0}".format(card.summary),
        "- target_file: {0}".format(card.target_file),
        "- target_symbols: {0}".format(", ".join(card.target_symbols) or "(none)"),
        "- anchor_snippets: {0}".format(len(card.anchor_snippets)),
        "- allowed_edit_scope: {0}".format(card.allowed_edit_scope),
        "- forbidden_families: {0}".format(", ".join(card.forbidden_families) or "(none)"),
        "- implementation_notes: {0}".format(card.implementation_notes),
        "- max_patch_lines: {0}".format(card.max_patch_lines),
    ]
    for index, snippet in enumerate(card.anchor_snippets, start=1):
        lines.append("  anchor[{0}]:".format(index))
        for snippet_line in str(snippet).strip("\n").splitlines() or [""]:
            lines.append("    {0}".format(snippet_line))
    return "\n".join(lines)


def build_codex_planner_prompt(
    worktree: Path,
    program: str,
    config: ProjectConfig,
    card_count: int,
    history_rows: List[Dict[str, str]],
    settings: WorkerSettings,
) -> str:
    forbidden_repeat_block = render_forbidden_repeat_guidance(
        history_rows,
        settings.forbidden_hypotheses,
    )
    snapshot_config = project_config_with_worker(config, settings)
    all_paths = [line.strip() for line in run_git(worktree, "ls-files").splitlines() if line.strip()]
    snapshot, included_paths, omitted_paths = build_repo_snapshot(worktree, snapshot_config, program)
    prompt = """
    You are preparing an Evoloza experiment backlog for this repository.

    Create exactly {card_count} atomic experiment cards for future rounds.

    Previous experiment log:
    {history_block}

    Repo-configured and already-explored idea families you must avoid repeating:
    {forbidden_repeat_block}

    Hard rules:
    - You cannot execute commands or inspect the filesystem directly.
    - Use only the repository snapshot included below.
    - Return exactly one JSON object and nothing else.
    - The JSON object must match the provided schema.
    - Every card must be independently executable in one future round.
    - Each card must name exactly one target file.
    - Prefer one small mechanism over bundles.
    - Prefer one to three target symbols per card.
    - Every card must include one to three exact `anchor_snippets` copied verbatim from the target file.
    - Each `anchor_snippet` should be a compact 1-6 line block that gives the executor an exact insertion or replacement point.
    - Use `forbidden_families` to list nearby explored families the executor must avoid drifting into.
    - Avoid cards that mostly restate already accepted or already rejected ideas.
    - Use only target files whose contents are included below.
    - If the snapshot is insufficient, still return the strongest cards you can support from the available context.

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
            card_count=card_count,
            history_block=render_history_for_prompt(history_rows),
            forbidden_repeat_block=forbidden_repeat_block,
            program=program.strip(),
            file_list=render_repo_file_list(all_paths),
            included_paths=render_repo_file_list(included_paths, limit=max(len(included_paths), 1)),
            omitted_paths=render_repo_file_list(omitted_paths, limit=50),
            snapshot=snapshot.rstrip(),
        )
    ).strip() + "\n"


def build_ollama_planner_prompt(
    worktree: Path,
    program: str,
    config: ProjectConfig,
    card_count: int,
    history_rows: List[Dict[str, str]],
    settings: WorkerSettings,
) -> str:
    history_block = render_history_for_prompt(history_rows)
    forbidden_repeat_block = render_forbidden_repeat_guidance(
        history_rows,
        settings.forbidden_hypotheses,
    )
    snapshot_config = project_config_with_worker(config, settings)
    all_paths = [line.strip() for line in run_git(worktree, "ls-files").splitlines() if line.strip()]
    snapshot, included_paths, omitted_paths = build_repo_snapshot(worktree, snapshot_config, program)
    prompt = """
    You are preparing an Evoloza experiment backlog for a git repository.

    Create exactly {card_count} atomic experiment cards for future rounds.

    Previous experiment log:
    {history_block}

    Repo-configured and already-explored idea families you must avoid repeating:
    {forbidden_repeat_block}

    Hard rules:
    - You cannot execute commands or inspect the filesystem directly.
    - Use only the repository snapshot included below.
    - Return exactly one JSON object and nothing else.
    - The JSON object must match the provided schema.
    - Every card must be independently executable in one future round.
    - Each card must name exactly one target file from the snapshot.
    - Prefer one to three target symbols per card.
    - Every card must include one to three exact `anchor_snippets` copied verbatim from the target file snapshot.
    - Each `anchor_snippet` should be a compact 1-6 line block that gives the executor an exact insertion or replacement point.
    - Keep cards atomic and implementation-oriented.
    - Use `forbidden_families` to name nearby explored families the executor must avoid drifting into.
    - Avoid cards that mostly restate already accepted or already rejected ideas.

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
            card_count=card_count,
            history_block=history_block,
            forbidden_repeat_block=forbidden_repeat_block,
            program=program.strip(),
            file_list=render_repo_file_list(all_paths),
            included_paths=render_repo_file_list(included_paths, limit=max(len(included_paths), 1)),
            omitted_paths=render_repo_file_list(omitted_paths, limit=50),
            snapshot=snapshot.rstrip(),
        )
    ).strip() + "\n"


def build_planner_prompt(
    worktree: Path,
    program: str,
    config: ProjectConfig,
    card_count: int,
    history_rows: List[Dict[str, str]],
    settings: WorkerSettings,
) -> str:
    if settings.backend == "ollama":
        return build_ollama_planner_prompt(
            worktree,
            program,
            config,
            card_count,
            history_rows,
            settings,
        )
    return build_codex_planner_prompt(worktree, program, config, card_count, history_rows, settings)


def build_codex_execute_prompt(
    program: str,
    config: ProjectConfig,
    run_id: str,
    round_index: int,
    champion: ChampionState,
    branch_name: str,
    history_rows: List[Dict[str, str]],
    card: ExperimentCard,
    settings: WorkerSettings,
) -> str:
    forbidden_repeat_block = render_forbidden_repeat_guidance(
        history_rows,
        settings.forbidden_hypotheses,
    )
    prompt = """
    You are executing one preplanned Evoloza experiment card.

    Run id: {run_id}
    Round: {round_index}
    Champion branch: {champion_branch}
    Candidate branch: {branch_name}
    Current champion score: {champion_score:.6f}
    Score direction: {direction}
    Current champion summary: {champion_summary}

    Previous experiment log:
    {history_block}

    Repo-configured and already-explored idea families you must avoid repeating:
    {forbidden_repeat_block}

    Execution card:
    {card_block}

    Official evaluator commands:
    {commands}

    Hard rules:
    - Work only inside the current repository.
    - You may edit files and run local commands as needed.
    - Do not create commits, branches, worktrees, or reset git state.
    - Do not invent a new experiment. Execute the card above.
    - Only modify `{target_file}`. If the card cannot be implemented within that file, leave the working tree unchanged.
    - Stay close to the named target symbols unless the current file structure forces a tiny nearby adjustment.
    - Keep the final patch within about {max_patch_lines} changed lines unless a slightly larger patch is clearly necessary.
    - In the final `hypothesis`, describe the card you actually executed, not a new direction.

    Mission:
    {program}
    """
    return textwrap.dedent(
        prompt.format(
            run_id=run_id,
            round_index=round_index,
            champion_branch=champion.branch,
            branch_name=branch_name,
            champion_score=champion.score,
            direction=config.evaluator.direction,
            champion_summary=champion.summary,
            history_block=render_history_for_prompt(history_rows),
            forbidden_repeat_block=forbidden_repeat_block,
            card_block=render_experiment_card(card),
            commands="\n".join("- {0}".format(item) for item in config.evaluator.commands),
            target_file=card.target_file,
            max_patch_lines=card.max_patch_lines,
            program=program.strip(),
        )
    ).strip() + "\n"


def build_ollama_execute_prompt(
    worktree: Path,
    program: str,
    config: ProjectConfig,
    run_id: str,
    round_index: int,
    champion: ChampionState,
    branch_name: str,
    history_rows: List[Dict[str, str]],
    card: ExperimentCard,
    settings: WorkerSettings,
) -> str:
    history_block = render_history_for_prompt(history_rows)
    forbidden_repeat_block = render_forbidden_repeat_guidance(
        history_rows,
        settings.forbidden_hypotheses,
    )
    scoped_settings = clone_worker_settings(settings)
    scoped_settings.context_files = [item for item in settings.context_files if item != card.target_file]
    snapshot_config = project_config_with_worker(config, scoped_settings)
    focused_entry = build_focused_target_snapshot_entry(worktree, card.target_file, card, scoped_settings)
    focused_header = "Target file view: focused excerpts around the card symbols and code identifiers.\n"
    snapshot, included_paths, omitted_paths = build_repo_snapshot(
        worktree,
        snapshot_config,
        program,
        exclude_paths=[card.target_file],
    )
    if focused_entry is not None:
        snapshot = focused_header + "\n" + focused_entry.rstrip() + "\n\n" + snapshot.lstrip()
        included_paths = ["{0} (focused excerpts)".format(card.target_file), *included_paths]
    all_paths = [card.target_file, *[item for item in scoped_settings.context_files if item != card.target_file]]
    prompt = """
    You are executing one preplanned Evoloza experiment card for a git repository.

    Run id: {run_id}
    Round: {round_index}
    Champion branch: {champion_branch}
    Candidate branch: {branch_name}
    Current champion score: {champion_score:.6f}
    Score direction: {direction}
    Current champion summary: {champion_summary}

    Previous experiment log:
    {history_block}

    Repo-configured and already-explored idea families you must avoid repeating:
    {forbidden_repeat_block}

    Execution card:
    {card_block}

    Official evaluator commands:
    {commands}

    Hard rules:
    - You cannot execute commands or inspect the filesystem directly.
    - Use only the repository snapshot included below.
    - Return exactly one JSON object and nothing else.
    - The JSON object may only include: hypothesis, summary, edit_ops.
    - The JSON object must include: hypothesis and edit_ops.
    - `edit_ops` must be an array of anchored operations with fields: file, action, anchor_snippet, new_text.
    - Valid actions are: replace_block, insert_before, insert_after.
    - Every `anchor_snippet` must be copied verbatim from `{target_file}` in the snapshot below.
    - Use the card's `anchor_snippets` whenever possible instead of inventing new anchors.
    - Do not invent a new experiment. Execute the card above.
    - Only modify `{target_file}`. If the card cannot be implemented within that file, return an empty `edit_ops` array.
    - Stay close to the named target symbols unless the current file structure forces a tiny nearby adjustment.
    - Keep the final materialized patch within about {max_patch_lines} changed lines unless a slightly larger patch is clearly necessary.
    - Keep the JSON compact. Use `summary` only for blockers or short execution notes.
    - In the final `hypothesis`, describe the card you actually executed, not a new direction.

    Mission:
    {program}

    Focused repository files:
    {included_paths}

    Omitted snapshot files:
    {omitted_paths}

    Repository snapshot:
    {snapshot}
    """
    return textwrap.dedent(
        prompt.format(
            run_id=run_id,
            round_index=round_index,
            champion_branch=champion.branch,
            branch_name=branch_name,
            champion_score=champion.score,
            direction=config.evaluator.direction,
            champion_summary=champion.summary,
            history_block=history_block,
            forbidden_repeat_block=forbidden_repeat_block,
            card_block=render_experiment_card(card),
            commands="\n".join("- {0}".format(item) for item in config.evaluator.commands),
            target_file=card.target_file,
            max_patch_lines=card.max_patch_lines,
            program=program.strip(),
            included_paths=render_repo_file_list(included_paths, limit=max(len(included_paths), 1)),
            omitted_paths=render_repo_file_list(omitted_paths, limit=50),
            snapshot=snapshot.rstrip(),
        )
    ).strip() + "\n"


def build_execute_prompt(
    worktree: Path,
    program: str,
    config: ProjectConfig,
    run_id: str,
    round_index: int,
    champion: ChampionState,
    branch_name: str,
    history_rows: List[Dict[str, str]],
    card: ExperimentCard,
    settings: WorkerSettings,
) -> str:
    if settings.backend == "ollama":
        return build_ollama_execute_prompt(
            worktree,
            program,
            config,
            run_id,
            round_index,
            champion,
            branch_name,
            history_rows,
            card,
            settings,
        )
    return build_codex_execute_prompt(
        program,
        config,
        run_id,
        round_index,
        champion,
        branch_name,
        history_rows,
        card,
        settings,
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


def normalize_multiline_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    items = []
    for entry in value:
        text = str(entry).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if text.strip():
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


def extract_hunk_only_patch_anchors(patch_text: str, limit: int = 12) -> List[str]:
    anchors: List[str] = []
    seen = set()
    for line in patch_text.splitlines():
        if line.startswith("@@ "):
            match = re.match(r"^@@ .* @@\s*(.*)$", line)
            trailer = match.group(1).strip() if match else ""
            if trailer and trailer not in seen:
                anchors.append(trailer)
                seen.add(trailer)
        elif line.startswith((" ", "-")) and not line.startswith("---"):
            snippet = line[1:].strip()
            if snippet and snippet not in seen:
                anchors.append(snippet)
                seen.add(snippet)
        if len(anchors) >= limit:
            break
    return anchors


def patch_target_path_bias(relpath: str) -> int:
    normalized = relpath.replace(os.sep, "/")
    bias = 0
    if normalized.startswith("tools/sota/"):
        bias -= 500
    if normalized.startswith("tools/"):
        bias -= 100
    if normalized.startswith("src/") or "/src/" in normalized:
        bias += 25
    if normalized.startswith("rust_sota/"):
        bias += 250
    bias -= normalized.count("/")
    bias -= len(normalized) // 8
    return bias


def infer_hunk_only_patch_target_path(worktree: Path, patch_text: str) -> Optional[str]:
    stripped = patch_text.lstrip()
    if not stripped.startswith("@@ "):
        return None
    anchors = extract_hunk_only_patch_anchors(patch_text)
    if not anchors:
        return None
    candidates: List[Tuple[int, int, int, str]] = []
    relpaths = [line.strip() for line in run_git(worktree, "ls-files").splitlines() if line.strip()]
    if not relpaths:
        relpaths = [
            str(path.relative_to(worktree)).replace(os.sep, "/")
            for path in worktree.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
    for relpath in relpaths:
        path = worktree / relpath
        if not path.is_file() or not is_probably_text_file(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = sum(1 for anchor in anchors if anchor in content)
        if score <= 0:
            continue
        size_bytes = path.stat().st_size
        candidates.append((score, patch_target_path_bias(relpath), -size_bytes, relpath))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_score, best_bias, _, best_path = candidates[0]
    if best_score < min(3, len(anchors)):
        return None
    if len(candidates) > 1 and candidates[1][0] == best_score and candidates[1][1] == best_bias:
        return None
    return best_path


def wrap_hunk_only_patch(worktree: Path, patch_text: str) -> Optional[str]:
    relpath = infer_hunk_only_patch_target_path(worktree, patch_text)
    if relpath is None:
        return None
    body = patch_text.rstrip("\n")
    return (
        "diff --git a/{0} b/{0}\n"
        "--- a/{0}\n"
        "+++ b/{0}\n"
        "{1}\n"
    ).format(relpath, body)


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


def normalize_executor_output(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    raw_ops = payload.get("edit_ops")
    if raw_ops is None:
        raw_ops = []
    if not isinstance(raw_ops, list):
        return None
    edit_ops = []
    for item in raw_ops:
        if not isinstance(item, dict):
            continue
        occurrence = item.get("occurrence")
        try:
            normalized_occurrence = int(occurrence) if occurrence not in (None, "") else None
        except (TypeError, ValueError):
            normalized_occurrence = None
        edit_ops.append(
            {
                "file": str(item.get("file") or "").strip(),
                "symbol": str(item.get("symbol") or "").strip(),
                "action": str(item.get("action") or "").strip(),
                "anchor_snippet": str(item.get("anchor_snippet") or "")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .strip("\n"),
                "occurrence": normalized_occurrence,
                "new_text": str(item.get("new_text") or "").replace("\r\n", "\n").replace("\r", "\n"),
            }
        )
    return {
        "hypothesis": str(payload.get("hypothesis") or "No hypothesis provided.").strip()
        or "No hypothesis provided.",
        "summary": str(payload.get("summary") or "No summary provided.").strip()
        or "No summary provided.",
        "edit_ops": edit_ops,
    }


def normalize_plan_output(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        return None
    cards = []
    for index, item in enumerate(raw_cards, start=1):
        if not isinstance(item, dict):
            continue
        hypothesis = str(item.get("hypothesis") or "").strip()
        target_file = str(item.get("target_file") or "").strip()
        anchor_snippets = normalize_multiline_string_list(item.get("anchor_snippets"))
        if not hypothesis or not target_file or not anchor_snippets:
            continue
        try:
            max_patch_lines = int(item.get("max_patch_lines", 80))
        except (TypeError, ValueError):
            max_patch_lines = 80
        cards.append(
            {
                "id": str(item.get("id") or "card-{0:03d}".format(index)).strip()
                or "card-{0:03d}".format(index),
                "hypothesis": hypothesis,
                "summary": str(item.get("summary") or "No summary provided.").strip()
                or "No summary provided.",
                "target_file": target_file,
                "target_symbols": normalize_string_list(item.get("target_symbols")),
                "anchor_snippets": anchor_snippets,
                "allowed_edit_scope": str(
                    item.get("allowed_edit_scope")
                    or "Only touch {0}.".format(target_file)
                ).strip(),
                "forbidden_families": normalize_string_list(item.get("forbidden_families")),
                "implementation_notes": str(
                    item.get("implementation_notes") or "Keep the patch focused and minimal."
                ).strip(),
                "max_patch_lines": max(1, min(max_patch_lines, 400)),
            }
        )
    if not cards:
        return None
    return {"cards": cards}


def project_config_with_worker(config: ProjectConfig, worker: WorkerSettings) -> ProjectConfig:
    return ProjectConfig(
        worker=worker,
        planner=config.planner,
        executor=config.executor,
        evaluator=config.evaluator,
        search=config.search,
        git=config.git,
    )


def strip_diff_path_prefix(path: str) -> str:
    candidate = str(path or "").strip()
    if candidate.startswith(("a/", "b/")):
        return candidate[2:]
    return candidate


def parse_unified_diff_file_patches(patch_text: str) -> List[UnifiedDiffFilePatch]:
    patches: List[UnifiedDiffFilePatch] = []
    current_path: Optional[str] = None
    current_hunks: List[UnifiedDiffHunk] = []
    current_hunk_lines: List[str] = []
    current_hunk_start: Optional[int] = None
    in_hunk = False

    def flush_hunk() -> None:
        nonlocal current_hunk_lines, current_hunk_start, in_hunk
        if current_hunk_start is not None:
            current_hunks.append(UnifiedDiffHunk(old_start=current_hunk_start, lines=current_hunk_lines))
        current_hunk_lines = []
        current_hunk_start = None
        in_hunk = False

    def flush_file() -> None:
        nonlocal current_path, current_hunks
        flush_hunk()
        if current_path and current_hunks:
            patches.append(UnifiedDiffFilePatch(relpath=current_path, hunks=current_hunks))
        current_path = None
        current_hunks = []

    for raw_line in patch_text.splitlines():
        if raw_line.startswith("diff --git "):
            flush_file()
            continue
        if raw_line.startswith("--- "):
            flush_hunk()
            continue
        if raw_line.startswith("+++ "):
            candidate = strip_diff_path_prefix(raw_line[4:].strip())
            current_path = None if candidate == "/dev/null" else candidate
            continue
        if raw_line.startswith("@@ "):
            flush_hunk()
            match = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", raw_line)
            if match:
                current_hunk_start = max(1, int(match.group(1)))
                current_hunk_lines = [raw_line]
                in_hunk = True
            continue
        if in_hunk and raw_line.startswith((" ", "+", "-", "\\")):
            current_hunk_lines.append(raw_line)
            continue
        if in_hunk:
            flush_hunk()
    flush_file()
    return patches


def find_subsequence_matches(haystack: List[str], needle: List[str]) -> List[int]:
    if not needle or len(needle) > len(haystack):
        return []
    width = len(needle)
    return [index for index in range(len(haystack) - width + 1) if haystack[index : index + width] == needle]


def select_best_hunk_match(matches: List[int], expected_start: int) -> Optional[int]:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    ranked = sorted(matches, key=lambda item: (abs(item - expected_start), item))
    if len(ranked) >= 2 and abs(ranked[0] - expected_start) == abs(ranked[1] - expected_start):
        return None
    return ranked[0]


def render_git_style_unified_diff(relpath: str, before_text: str, after_text: str) -> str:
    diff_lines = list(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="a/{0}".format(relpath),
            tofile="b/{0}".format(relpath),
            n=3,
        )
    )
    if not diff_lines:
        return ""
    return "diff --git a/{0} b/{0}\n".format(relpath) + "".join(diff_lines)


def find_text_occurrences(text: str, needle: str) -> List[int]:
    if not needle:
        return []
    matches = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        matches.append(index)
        start = index + max(1, len(needle))
    return matches


def count_patch_changed_lines(patch_text: str) -> int:
    count = 0
    for line in patch_text.splitlines():
        if line.startswith(("+++", "---", "@@", "diff --git ")):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def apply_executor_edit_ops(
    worktree: Path,
    artifact_dir: Path,
    target_file: str,
    edit_ops: List[Dict[str, Any]],
    max_patch_lines: Optional[int] = None,
) -> Optional[str]:
    relpath = strip_diff_path_prefix(target_file).replace(os.sep, "/")
    path = worktree / relpath
    patch_path = artifact_dir / "candidate.patch"
    if not path.exists() or not path.is_file():
        return "executor edit ops require an existing file ({0})".format(relpath)
    try:
        before_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return str(exc)
    current_text = before_text
    for index, op in enumerate(edit_ops, start=1):
        op_file = strip_diff_path_prefix(str(op.get("file") or relpath)).replace(os.sep, "/")
        if op_file != relpath:
            return "edit op {0} targeted {1} outside planned scope {2}".format(index, op_file, relpath)
        action = str(op.get("action") or "").strip()
        if action not in EXECUTOR_EDIT_ACTIONS:
            return "edit op {0} used unsupported action {1}".format(index, action or "<empty>")
        anchor_snippet = str(op.get("anchor_snippet") or "")
        if not anchor_snippet:
            return "edit op {0} did not include an anchor snippet".format(index)
        new_text = str(op.get("new_text") or "")
        matches = find_text_occurrences(current_text, anchor_snippet)
        if not matches:
            return "edit op {0} anchor snippet was not found in {1}".format(index, relpath)
        occurrence = op.get("occurrence")
        if occurrence in (None, ""):
            if len(matches) != 1:
                return "edit op {0} anchor snippet matched {1} locations; specify occurrence".format(
                    index,
                    len(matches),
                )
            match_index = matches[0]
        else:
            try:
                occurrence_index = int(occurrence)
            except (TypeError, ValueError):
                return "edit op {0} used an invalid occurrence".format(index)
            if occurrence_index < 1 or occurrence_index > len(matches):
                return "edit op {0} occurrence {1} is out of range for {2} matches".format(
                    index,
                    occurrence_index,
                    len(matches),
                )
            match_index = matches[occurrence_index - 1]
        anchor_end = match_index + len(anchor_snippet)
        if action == "replace_block":
            current_text = current_text[:match_index] + new_text + current_text[anchor_end:]
        elif action == "insert_before":
            current_text = current_text[:match_index] + new_text + current_text[match_index:]
        else:
            current_text = current_text[:anchor_end] + new_text + current_text[anchor_end:]
    rendered_patch = render_git_style_unified_diff(relpath, before_text, current_text)
    if max_patch_lines is not None and rendered_patch:
        changed_lines = count_patch_changed_lines(rendered_patch)
        if changed_lines > max_patch_lines + 20:
            return "executor edit ops expanded to {0} changed lines, exceeding budget {1}".format(
                changed_lines,
                max_patch_lines,
            )
    path.write_text(current_text, encoding="utf-8")
    patch_path.write_text(rendered_patch, encoding="utf-8")
    return None


def apply_hunk_with_trimmed_context(current_lines: List[str], hunk: UnifiedDiffHunk) -> Optional[List[str]]:
    body = [line for line in hunk.lines[1:] if line and not line.startswith("\\")]
    if not body:
        return None
    leading_context = 0
    while leading_context < len(body) and body[leading_context].startswith(" "):
        leading_context += 1
    trailing_context = 0
    while trailing_context < len(body) and body[len(body) - 1 - trailing_context].startswith(" "):
        trailing_context += 1

    for trim_total in range(leading_context + trailing_context + 1):
        for trim_leading in range(trim_total + 1):
            trim_trailing = trim_total - trim_leading
            if trim_leading > leading_context or trim_trailing > trailing_context:
                continue
            start_index = trim_leading
            end_index = len(body) - trim_trailing if trim_trailing else len(body)
            candidate = body[start_index:end_index]
            old_lines = [line[1:] for line in candidate if line.startswith((" ", "-"))]
            new_lines = [line[1:] for line in candidate if line.startswith((" ", "+"))]
            if not old_lines:
                continue
            matches = find_subsequence_matches(current_lines, old_lines)
            match_index = select_best_hunk_match(matches, max(0, hunk.old_start - 1))
            if match_index is None:
                continue
            return current_lines[:match_index] + new_lines + current_lines[match_index + len(old_lines) :]
    return None


def apply_patch_via_trimmed_hunks(worktree: Path, patch_text: str) -> Tuple[bool, Optional[str]]:
    file_patches = parse_unified_diff_file_patches(patch_text)
    if not file_patches:
        return False, None
    rewritten_files: Dict[str, str] = {}
    rendered_diffs: List[str] = []

    for file_patch in file_patches:
        relpath = file_patch.relpath
        path = worktree / relpath
        if not path.exists() or not path.is_file():
            return False, "trimmed-hunk fallback only supports existing files ({0})".format(relpath)
        try:
            before_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return False, str(exc)
        current_lines = before_text.splitlines()
        had_trailing_newline = before_text.endswith("\n")
        for hunk in file_patch.hunks:
            updated = apply_hunk_with_trimmed_context(current_lines, hunk)
            if updated is None:
                return False, "trimmed-hunk fallback could not anchor {0}:{1}".format(relpath, hunk.old_start)
            current_lines = updated
        after_text = "\n".join(current_lines)
        if current_lines and had_trailing_newline:
            after_text += "\n"
        elif not current_lines:
            after_text = ""
        rewritten_files[relpath] = after_text
        rendered = render_git_style_unified_diff(relpath, before_text, after_text)
        if rendered:
            rendered_diffs.append(rendered)

    for relpath, after_text in rewritten_files.items():
        (worktree / relpath).write_text(after_text, encoding="utf-8")
    return True, "".join(rendered_diffs)


def apply_unified_diff(worktree: Path, artifact_dir: Path, patch_text: str) -> Optional[str]:
    patch_path = artifact_dir / "candidate.patch"
    patch_payload = patch_text
    if patch_payload.lstrip().startswith("@@ "):
        wrapped_patch = wrap_hunk_only_patch(worktree, patch_payload)
        if wrapped_patch is not None and wrapped_patch != patch_payload:
            patch_payload = wrapped_patch
    patch_path.write_text(patch_payload, encoding="utf-8")
    check_result = subprocess.run(
        ["git", "apply", "--check", "--recount", "--whitespace=nowarn", str(patch_path)],
        cwd=str(worktree),
        capture_output=True,
        text=True,
        check=False,
    )
    if check_result.returncode == 0:
        apply_result = subprocess.run(
            ["git", "apply", "--recount", "--whitespace=nowarn", str(patch_path)],
            cwd=str(worktree),
            capture_output=True,
            text=True,
            check=False,
        )
        if apply_result.returncode == 0:
            return None
        return apply_result.stderr.strip() or apply_result.stdout.strip() or "git apply failed"
    fallback_ok, fallback_detail = apply_patch_via_trimmed_hunks(worktree, patch_payload)
    if fallback_ok:
        if fallback_detail:
            patch_path.write_text(fallback_detail, encoding="utf-8")
        return None
    return check_result.stderr.strip() or check_result.stdout.strip() or fallback_detail or "git apply failed"


def extract_patch_addition_blocks(patch_text: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    current_path: Optional[str] = None
    current_block: List[str] = []
    in_hunk = False

    def flush_block() -> None:
        nonlocal current_block
        if current_path is None or not current_block:
            current_block = []
            return
        block_text = "\n".join(current_block)
        if any(char.isalnum() for char in block_text):
            blocks.append((current_path, block_text))
        current_block = []

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush_block()
            current_path = None
            in_hunk = False
            continue
        if line.startswith("+++ "):
            flush_block()
            target = line[4:].strip()
            if target == "/dev/null":
                current_path = None
            else:
                current_path = target[2:] if target.startswith("b/") else target
            continue
        if line.startswith("@@ "):
            flush_block()
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+") and not line.startswith("+++ "):
            current_block.append(line[1:])
            continue
        flush_block()
        if not line.startswith((" ", "-", "\\")):
            in_hunk = False

    flush_block()
    return blocks


def patch_additions_already_present(worktree: Path, patch_text: str) -> bool:
    blocks = extract_patch_addition_blocks(patch_text)
    if not blocks:
        return False
    cached_files: Dict[str, str] = {}
    for relpath, block_text in blocks:
        if relpath not in cached_files:
            path = worktree / relpath
            if not path.exists():
                return False
            try:
                cached_files[relpath] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
        if block_text not in cached_files[relpath]:
            return False
    return True


def mark_patch_as_redundant(
    invocation: WorkerInvocationResult,
    summary: str,
) -> WorkerInvocationResult:
    structured_output = dict(invocation.structured_output or {})
    structured_output["patch"] = ""
    structured_output["summary"] = summary
    Path(invocation.last_message_path).write_text(json.dumps(structured_output, indent=2), encoding="utf-8")
    Path(invocation.stderr_path).write_text(summary, encoding="utf-8")
    return WorkerInvocationResult(
        returncode=0,
        output_path=invocation.output_path,
        stderr_path=invocation.stderr_path,
        last_message_path=invocation.last_message_path,
        structured_output=structured_output,
        usage=invocation.usage,
    )


def merge_repaired_worker_output(
    previous_output: Dict[str, Any],
    repaired_output: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(repaired_output)
    original_hypothesis = str(previous_output.get("hypothesis") or "").strip()
    if original_hypothesis:
        merged["hypothesis"] = original_hypothesis
    if not str(merged.get("summary") or "").strip():
        original_summary = str(previous_output.get("summary") or "").strip()
        if original_summary:
            merged["summary"] = original_summary
    return merged


def preserve_original_hypothesis_for_repair(
    previous_output: Dict[str, Any],
    repair_result: WorkerInvocationResult,
) -> WorkerInvocationResult:
    if repair_result.structured_output is None:
        return repair_result
    merged_output = merge_repaired_worker_output(previous_output, repair_result.structured_output)
    Path(repair_result.last_message_path).write_text(json.dumps(merged_output, indent=2), encoding="utf-8")
    return WorkerInvocationResult(
        returncode=repair_result.returncode,
        output_path=repair_result.output_path,
        stderr_path=repair_result.stderr_path,
        last_message_path=repair_result.last_message_path,
        structured_output=merged_output,
        usage=repair_result.usage,
    )


def promote_repair_artifacts(
    artifact_dir: Path,
    invocation: WorkerInvocationResult,
) -> WorkerInvocationResult:
    source_dir = Path(invocation.last_message_path).parent
    promoted_last_message = Path(invocation.last_message_path)
    promoted_stderr = Path(invocation.stderr_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("candidate.patch", "last_message.json", "ollama.stderr.log"):
        source = source_dir / filename
        if not source.exists():
            continue
        destination = artifact_dir / filename
        shutil.copyfile(source, destination)
        if filename == "last_message.json":
            promoted_last_message = destination
        elif filename == "ollama.stderr.log":
            promoted_stderr = destination
    return WorkerInvocationResult(
        returncode=invocation.returncode,
        output_path=invocation.output_path,
        stderr_path=str(promoted_stderr),
        last_message_path=str(promoted_last_message),
        structured_output=invocation.structured_output,
        usage=invocation.usage,
    )


def extract_patch_failure_locations(stderr_text: str) -> List[Tuple[str, int]]:
    locations: List[Tuple[str, int]] = []
    seen = set()
    for match in re.finditer(r"^error: patch failed: ([^:\n]+):(\d+)$", stderr_text or "", re.MULTILINE):
        relpath = match.group(1).strip()
        line_no = int(match.group(2))
        key = (relpath, line_no)
        if key in seen:
            continue
        seen.add(key)
        locations.append(key)
    return locations


def render_patch_failure_context(
    worktree: Path,
    stderr_text: str,
    radius: int = 8,
    limit: int = 3,
) -> str:
    blocks = []
    for relpath, line_no in extract_patch_failure_locations(stderr_text)[:limit]:
        path = worktree / relpath
        if not path.exists():
            blocks.append(
                "File: {0}\nApprox failed line: {1}\n<file missing from current snapshot>".format(
                    relpath, line_no
                )
            )
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            blocks.append(
                "File: {0}\nApprox failed line: {1}\n<unable to read file: {2}>".format(
                    relpath, line_no, exc
                )
            )
            continue
        start = max(1, line_no - radius)
        end = min(len(lines), line_no + radius)
        snippet = "\n".join(
            "{0:>6}: {1}".format(index, lines[index - 1]) for index in range(start, end + 1)
        )
        blocks.append(
            "File: {0}\nApprox failed line: {1}\n{2}".format(relpath, line_no, snippet)
        )
    return "\n\n".join(blocks)


def build_ollama_patch_repair_prompt(
    previous_output: Dict[str, Any],
    patch_error: str,
    failure_context: str,
) -> str:
    previous_payload = {
        "hypothesis": previous_output.get("hypothesis", ""),
        "summary": previous_output.get("summary", ""),
        "files_touched": previous_output.get("files_touched", []),
        "local_checks_run": previous_output.get("local_checks_run", []),
        "risks": previous_output.get("risks", []),
    }
    prompt = """
    You previously returned a valid Evoloza worker JSON object, but the patch did not apply.

    Return exactly one JSON object and nothing else.
    The JSON object may only include: hypothesis, summary, patch.
    The JSON object must include: hypothesis and patch.
    The `patch` field must be a unified git diff that can be applied with `git apply`.
    Keep the JSON compact and prioritize a correct applying patch over explanation.
    Use exact context lines from the current snapshot excerpts below.
    If the intended logic already exists in the current snapshot, return an empty patch and say so briefly in `summary`.
    If you cannot repair the patch safely, return an empty patch and explain the blocker briefly in `summary`.
    Do not use markdown fences.

    Previous structured output:
    {previous_payload}

    Patch apply error:
    {patch_error}

    Current snapshot excerpts around failed hunks:
    {failure_context}

    Previous patch:
    {patch_text}
    """
    return textwrap.dedent(
        prompt.format(
            previous_payload=json.dumps(previous_payload, indent=2),
            patch_error=patch_error.strip(),
            failure_context=failure_context.strip() or "<no failure context available>",
            patch_text=str(previous_output.get("patch", "")).rstrip(),
        )
    ).strip() + "\n"


def build_ollama_executor_repair_prompt(
    card: ExperimentCard,
    previous_output: Dict[str, Any],
    apply_error: str,
    target_snapshot: str,
) -> str:
    previous_payload = {
        "hypothesis": previous_output.get("hypothesis", ""),
        "summary": previous_output.get("summary", ""),
        "edit_ops": previous_output.get("edit_ops", []),
    }
    prompt = """
    You previously returned a valid Evoloza executor JSON object, but the anchored edit operations did not apply locally.

    Return exactly one JSON object and nothing else.
    The JSON object may only include: hypothesis, summary, edit_ops.
    The JSON object must include: hypothesis and edit_ops.
    Each edit op must include: file, action, anchor_snippet, new_text.
    Valid actions are: replace_block, insert_before, insert_after.
    Every `anchor_snippet` must be copied verbatim from the current target snapshot below.
    Only modify `{target_file}`.
    Keep the JSON compact and prioritize exact anchors over explanation.
    If the intended logic already exists in the current snapshot, return an empty `edit_ops` array and say so briefly in `summary`.
    If you cannot repair the edit safely, return an empty `edit_ops` array and explain the blocker briefly in `summary`.
    Do not use markdown fences.

    Execution card:
    {card_block}

    Previous structured output:
    {previous_payload}

    Local apply error:
    {apply_error}

    Current target snapshot:
    {target_snapshot}
    """
    return textwrap.dedent(
        prompt.format(
            target_file=card.target_file,
            card_block=render_experiment_card(card),
            previous_payload=json.dumps(previous_payload, indent=2),
            apply_error=apply_error.strip(),
            target_snapshot=target_snapshot.strip() or "<no snapshot available>",
        )
    ).strip() + "\n"


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


def ollama_api_json(
    base_url: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
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
    tags = ollama_api_json(
        settings.ollama_host,
        "/api/tags",
        timeout_seconds=settings.request_timeout_seconds,
    )
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


def build_ollama_generate_payload(
    prompt: str,
    settings: WorkerSettings,
    response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    options = dict(settings.ollama_options)
    options.setdefault("temperature", settings.temperature)
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "format": response_schema or OLLAMA_WORKER_OUTPUT_SCHEMA,
        "stream": False,
        "options": options,
    }
    if settings.keep_alive is not None:
        payload["keep_alive"] = settings.keep_alive
    if settings.think is not None:
        payload["think"] = settings.think
    return payload


def invoke_ollama_structured(
    prompt: str,
    settings: WorkerSettings,
    request_path: Path,
    response_path: Path,
    stderr_path: Path,
    last_message_path: Path,
    response_schema: Optional[Dict[str, Any]] = None,
    output_normalizer: Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = normalize_worker_output,
    progress: Optional[ProgressReporter] = None,
    stage_message: str = "Ollama working",
) -> WorkerInvocationResult:
    structured_output = None
    usage = None
    try:
        model = resolve_ollama_model(settings)
        payload = build_ollama_generate_payload(prompt, settings, response_schema=response_schema)
        payload["model"] = model
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with progress.spin(stage_message) if progress is not None else _nullcontext():
            response = ollama_api_json(
                settings.ollama_host,
                "/api/generate",
                payload,
                timeout_seconds=settings.request_timeout_seconds,
            )
        response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        structured_output = output_normalizer(extract_ollama_structured_output(response))
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


def attempt_ollama_patch_repair(
    worktree: Path,
    artifact_dir: Path,
    settings: WorkerSettings,
    progress: Optional[ProgressReporter],
    stage_message: str,
    structured_output: Dict[str, Any],
    patch_error: str,
) -> Tuple[Optional[WorkerInvocationResult], Optional[str]]:
    failure_context = render_patch_failure_context(worktree, patch_error)
    if not failure_context:
        return None, None
    repair_prompt = build_ollama_patch_repair_prompt(structured_output, patch_error, failure_context)
    repair_dir = artifact_dir / "repair-01"
    repair_dir.mkdir(parents=True, exist_ok=True)
    (repair_dir / "prompt.md").write_text(repair_prompt, encoding="utf-8")
    repair_result = invoke_ollama_structured(
        repair_prompt,
        settings,
        request_path=repair_dir / "ollama.request.json",
        response_path=repair_dir / "ollama.response.json",
        stderr_path=repair_dir / "ollama.stderr.log",
        last_message_path=repair_dir / "last_message.json",
        progress=progress,
        stage_message="{0} repair 1".format(stage_message),
    )
    if repair_result.returncode != 0 or repair_result.structured_output is None:
        repair_error = Path(repair_result.stderr_path).read_text(encoding="utf-8").strip()
        return None, repair_error or "repair worker failed"
    repair_result = preserve_original_hypothesis_for_repair(structured_output, repair_result)
    repaired_patch = str(repair_result.structured_output.get("patch", ""))
    if repaired_patch:
        repair_apply_error = apply_unified_diff(worktree, repair_dir, repaired_patch)
        if repair_apply_error is not None:
            if patch_additions_already_present(worktree, repaired_patch):
                return (
                    mark_patch_as_redundant(
                        repair_result,
                        "Patch was redundant: additions already present in current snapshot.",
                    ),
                    None,
                )
            return None, "repair patch apply failed: {0}".format(repair_apply_error)
    return repair_result, None


def attempt_ollama_executor_repair(
    worktree: Path,
    artifact_dir: Path,
    settings: WorkerSettings,
    progress: Optional[ProgressReporter],
    stage_message: str,
    card: ExperimentCard,
    structured_output: Dict[str, Any],
    apply_error: str,
) -> Tuple[Optional[WorkerInvocationResult], Optional[str]]:
    target_snapshot = build_focused_target_snapshot_entry(worktree, card.target_file, card, settings)
    if target_snapshot is None:
        return None, None
    repair_prompt = build_ollama_executor_repair_prompt(card, structured_output, apply_error, target_snapshot)
    repair_dir = artifact_dir / "repair-01"
    repair_dir.mkdir(parents=True, exist_ok=True)
    (repair_dir / "prompt.md").write_text(repair_prompt, encoding="utf-8")
    repair_result = invoke_ollama_structured(
        repair_prompt,
        settings,
        request_path=repair_dir / "ollama.request.json",
        response_path=repair_dir / "ollama.response.json",
        stderr_path=repair_dir / "ollama.stderr.log",
        last_message_path=repair_dir / "last_message.json",
        response_schema=EXECUTOR_OUTPUT_SCHEMA,
        output_normalizer=normalize_executor_output,
        progress=progress,
        stage_message="{0} repair 1".format(stage_message),
    )
    if repair_result.returncode != 0 or repair_result.structured_output is None:
        repair_error = Path(repair_result.stderr_path).read_text(encoding="utf-8").strip()
        return None, repair_error or "repair worker failed"
    repair_result = preserve_original_hypothesis_for_repair(structured_output, repair_result)
    repair_apply_error = apply_executor_edit_ops(
        worktree,
        repair_dir,
        card.target_file,
        list(repair_result.structured_output.get("edit_ops", [])),
        max_patch_lines=card.max_patch_lines,
    )
    if repair_apply_error is not None:
        return None, "repair edit ops failed: {0}".format(repair_apply_error)
    return repair_result, None


def run_codex(
    worktree: Path,
    artifact_dir: Path,
    prompt: str,
    settings: WorkerSettings,
    output_schema: Optional[Dict[str, Any]] = None,
    output_normalizer: Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = normalize_worker_output,
    progress: Optional[ProgressReporter] = None,
    stage_message: str = "Codex working",
) -> WorkerInvocationResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    schema_path = artifact_dir / "worker_output_schema.json"
    output_path = artifact_dir / "codex.jsonl"
    stderr_path = artifact_dir / "codex.stderr.log"
    last_message_path = artifact_dir / "last_message.json"
    schema_path.write_text(json.dumps(output_schema or WORKER_OUTPUT_SCHEMA, indent=2), encoding="utf-8")

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
            structured_output = output_normalizer(extract_json_object(content))
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
    invocation = invoke_ollama_structured(
        prompt,
        settings,
        request_path=request_path,
        response_path=response_path,
        stderr_path=stderr_path,
        last_message_path=last_message_path,
        progress=progress,
        stage_message=stage_message,
    )
    if invocation.returncode != 0 or invocation.structured_output is None:
        return invocation
    patch_error = None
    if invocation.structured_output.get("patch"):
        patch_error = apply_unified_diff(worktree, artifact_dir, str(invocation.structured_output["patch"]))
    if patch_error is None:
        return invocation
    if patch_additions_already_present(worktree, str(invocation.structured_output.get("patch", ""))):
        return mark_patch_as_redundant(
            invocation,
            "Patch was redundant: additions already present in current snapshot.",
        )
    repaired_invocation = None
    repair_error = None
    for _ in range(DEFAULT_OLLAMA_PATCH_REPAIR_ATTEMPTS):
        repaired_invocation, repair_error = attempt_ollama_patch_repair(
            worktree,
            artifact_dir,
            settings,
            progress,
            stage_message,
            invocation.structured_output,
            patch_error,
        )
        if repaired_invocation is not None:
            return promote_repair_artifacts(artifact_dir, repaired_invocation)
    structured_output = dict(invocation.structured_output)
    summary = "Patch apply failed: {0}".format(patch_error)
    if repair_error:
        summary = "{0}. Repair attempt failed: {1}".format(summary, repair_error)
    structured_output["summary"] = summary
    Path(invocation.last_message_path).write_text(json.dumps(structured_output, indent=2), encoding="utf-8")
    Path(invocation.stderr_path).write_text(summary, encoding="utf-8")
    return WorkerInvocationResult(
        returncode=1,
        output_path=invocation.output_path,
        stderr_path=invocation.stderr_path,
        last_message_path=invocation.last_message_path,
        structured_output=structured_output,
        usage=invocation.usage,
    )


def run_ollama_execute(
    worktree: Path,
    artifact_dir: Path,
    prompt: str,
    settings: WorkerSettings,
    card: ExperimentCard,
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
    invocation = invoke_ollama_structured(
        prompt,
        settings,
        request_path=request_path,
        response_path=response_path,
        stderr_path=stderr_path,
        last_message_path=last_message_path,
        response_schema=EXECUTOR_OUTPUT_SCHEMA,
        output_normalizer=normalize_executor_output,
        progress=progress,
        stage_message=stage_message,
    )
    if invocation.returncode != 0 or invocation.structured_output is None:
        return invocation
    apply_error = apply_executor_edit_ops(
        worktree,
        artifact_dir,
        card.target_file,
        list(invocation.structured_output.get("edit_ops", [])),
        max_patch_lines=card.max_patch_lines,
    )
    if apply_error is None:
        return invocation
    repaired_invocation = None
    repair_error = None
    for _ in range(DEFAULT_OLLAMA_PATCH_REPAIR_ATTEMPTS):
        repaired_invocation, repair_error = attempt_ollama_executor_repair(
            worktree,
            artifact_dir,
            settings,
            progress,
            stage_message,
            card,
            invocation.structured_output,
            apply_error,
        )
        if repaired_invocation is not None:
            return promote_repair_artifacts(artifact_dir, repaired_invocation)
    structured_output = dict(invocation.structured_output)
    summary = "Executor edit ops failed: {0}".format(apply_error)
    if repair_error:
        summary = "{0}. Repair attempt failed: {1}".format(summary, repair_error)
    structured_output["summary"] = summary
    Path(invocation.last_message_path).write_text(json.dumps(structured_output, indent=2), encoding="utf-8")
    Path(invocation.stderr_path).write_text(summary, encoding="utf-8")
    return WorkerInvocationResult(
        returncode=1,
        output_path=invocation.output_path,
        stderr_path=invocation.stderr_path,
        last_message_path=invocation.last_message_path,
        structured_output=structured_output,
        usage=invocation.usage,
    )


def run_structured_task(
    worktree: Path,
    artifact_dir: Path,
    prompt: str,
    settings: WorkerSettings,
    *,
    codex_output_schema: Optional[Dict[str, Any]] = None,
    ollama_output_schema: Optional[Dict[str, Any]] = None,
    output_normalizer: Callable[[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = normalize_worker_output,
    apply_patch_output: bool = False,
    progress: Optional[ProgressReporter] = None,
    stage_message: Optional[str] = None,
) -> WorkerInvocationResult:
    label = worker_display_name(settings)
    message = stage_message or "{0} working".format(label)
    if settings.backend == "ollama":
        if apply_patch_output:
            return run_ollama(
                worktree,
                artifact_dir,
                prompt,
                settings,
                progress=progress,
                stage_message=message,
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_path = artifact_dir / "ollama.request.json"
        response_path = artifact_dir / "ollama.response.json"
        stderr_path = artifact_dir / "ollama.stderr.log"
        last_message_path = artifact_dir / "last_message.json"
        if progress is not None and not progress.enabled:
            progress.event(message)
        return invoke_ollama_structured(
            prompt,
            settings,
            request_path=request_path,
            response_path=response_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path,
            response_schema=ollama_output_schema,
            output_normalizer=output_normalizer,
            progress=progress,
            stage_message=message,
        )
    return run_codex(
        worktree,
        artifact_dir,
        prompt,
        settings,
        output_schema=codex_output_schema,
        output_normalizer=output_normalizer,
        progress=progress,
        stage_message=message,
    )


def run_worker(
    worktree: Path,
    artifact_dir: Path,
    prompt: str,
    settings: WorkerSettings,
    progress: Optional[ProgressReporter] = None,
    stage_message: Optional[str] = None,
) -> WorkerInvocationResult:
    return run_structured_task(
        worktree,
        artifact_dir,
        prompt,
        settings,
        codex_output_schema=WORKER_OUTPUT_SCHEMA,
        ollama_output_schema=OLLAMA_WORKER_OUTPUT_SCHEMA,
        output_normalizer=normalize_worker_output,
        apply_patch_output=True,
        progress=progress,
        stage_message=stage_message,
    )


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
        "Mode: {0}".format(state.mode),
        "Base branch: {0}".format(state.base_branch),
        "Current round: {0}".format(state.current_round),
        "Rounds without improvement: {0}".format(state.rounds_without_improvement),
    ]
    if state.plan_path:
        lines.append("Plan: {0}".format(state.plan_path))
    lines.append("Champion:")
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
        "- Mode: `{0}`".format(state.mode),
        "- Base branch: `{0}`".format(state.base_branch),
        "- Final champion: `{0}`".format(state.champion.branch if state.champion else "none"),
        "",
        "## Champion",
        "",
    ]
    if state.plan_path:
        lines.insert(5, "- Plan: `{0}`".format(state.plan_path))
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
    def __init__(
        self,
        repo: Path,
        progress: Optional[ProgressReporter] = None,
        config_path: Optional[Path] = None,
    ):
        self.repo = repo.resolve()
        self.progress = progress
        self.config_path = config_path

    def plan(
        self,
        output_path: Optional[Path] = None,
        card_count: Optional[int] = None,
    ) -> ExperimentPlan:
        if self.progress is not None:
            self.progress.event("Preparing plan in {0}".format(self.repo))
        ensure_git_repo(self.repo)
        config = load_project_config(self.repo, self.config_path)
        count = int(card_count or config.planner.cards_per_plan)
        if count <= 0:
            raise ValueError("Planner card count must be a positive integer")
        plan_id = make_run_id()
        artifact_dir = self._plan_dir(config, plan_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_planner_prompt(
            self.repo,
            program_text(self.repo),
            config,
            count,
            self._history_rows(config),
            config.planner.worker,
        )
        (artifact_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        invocation = run_structured_task(
            self.repo,
            artifact_dir / "planner",
            prompt,
            config.planner.worker,
            codex_output_schema=PLAN_OUTPUT_SCHEMA,
            ollama_output_schema=PLAN_OUTPUT_SCHEMA,
            output_normalizer=normalize_plan_output,
            apply_patch_output=False,
            progress=self.progress,
            stage_message="Planning experiment cards",
        )
        if self.progress is not None:
            self.progress.finalize_live_usage(invocation.usage)
            self.progress.end_phase()
        if invocation.returncode != 0 or invocation.structured_output is None:
            summary = Path(invocation.stderr_path).read_text(encoding="utf-8").strip()
            raise RuntimeError("Planner failed: {0}".format(summary or "invalid structured output"))
        cards = [
            ExperimentCard.from_dict(item)
            for item in invocation.structured_output.get("cards", [])[:count]
        ]
        if not cards:
            raise RuntimeError("Planner returned no valid experiment cards")
        planner_model = config.planner.worker.model or ""
        if config.planner.worker.backend == "ollama" and not planner_model:
            planner_model = resolve_ollama_model(config.planner.worker)
        plan = ExperimentPlan(
            plan_id=plan_id,
            created_at=now_iso(),
            repo_path=str(self.repo),
            planner_backend=config.planner.worker.backend,
            planner_model=planner_model,
            program_path=str(self.repo / PROGRAM_FILENAME),
            artifact_dir=str(artifact_dir),
            cards=cards,
        )
        canonical_path = artifact_dir / "plan.json"
        canonical_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        if self.progress is not None:
            self.progress.finish(
                "Plan {0} created with {1} cards".format(plan.plan_id, len(plan.cards))
            )
        return plan

    def run(self, run_id: Optional[str] = None) -> RunState:
        if self.progress is not None:
            self.progress.event("Preparing run in {0}".format(self.repo))
        ensure_git_repo(self.repo)
        config = load_project_config(self.repo, self.config_path)
        ensure_clean_worktree(self.repo)
        state = self._load_or_create_state(config, run_id, mode="run", plan_path=None)
        self._prepare_state(config, state)
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
            self._advance_state_after_result(config, state, result)

        return self._finish_state(config, state)

    def execute(self, plan_path: Path, run_id: Optional[str] = None) -> RunState:
        if self.progress is not None:
            self.progress.event("Preparing execute run in {0}".format(self.repo))
        ensure_git_repo(self.repo)
        config = load_project_config(self.repo, self.config_path)
        ensure_clean_worktree(self.repo)
        resolved_plan_path = plan_path.resolve()
        plan = load_experiment_plan(resolved_plan_path)
        if not plan.cards:
            raise RuntimeError("Plan contains no experiment cards: {0}".format(resolved_plan_path))
        state = self._load_or_create_state(
            config,
            run_id,
            mode="execute",
            plan_path=str(resolved_plan_path),
        )
        self._prepare_state(config, state)
        started_at = time.monotonic()
        total_cards = len(plan.cards)
        state.status = "running"
        self._write_state(config, state)

        while state.current_round <= total_cards:
            if minutes_elapsed(started_at) >= config.search.max_wall_time_minutes:
                state.status = "stopped"
                break
            card = plan.cards[state.current_round - 1]
            if self.progress is not None:
                self.progress.event(
                    "Round {0}/{1}: card {2} target={3}".format(
                        state.current_round,
                        total_cards,
                        card.id,
                        card.target_file,
                    )
                )
            candidate = self._plan_round(config, state)
            candidate["card_id"] = card.id
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
                worker_settings=config.executor,
                planned_card=card,
            )
            self._append_result(config, state.run_id, result)
            self._advance_state_after_result(config, state, result)

        return self._finish_state(config, state)

    def status(self, run_id: Optional[str] = None) -> RunState:
        return self._load_state(load_project_config(self.repo, self.config_path), run_id)

    def report(self, run_id: Optional[str] = None) -> Tuple[RunState, List[Dict[str, str]]]:
        config = load_project_config(self.repo, self.config_path)
        state = self._load_state(config, run_id)
        return state, read_results(self._results_path(config, state.run_id))

    def latest_plan_path(self) -> Optional[Path]:
        config = load_project_config(self.repo, self.config_path)
        return self._latest_plan_path(config)

    def _prepare_state(self, config: ProjectConfig, state: RunState) -> None:
        if self.progress is not None:
            self.progress.event("Run id {0} on base branch {1}".format(state.run_id, state.base_branch))
            if (
                state.status == "created"
                and state.current_round == 1
                and state.champion
                and state.champion.source == "seeded"
            ):
                self.progress.event(
                    "Seeded from previous champion {0} score={1:.6f}".format(
                        state.champion.branch,
                        state.champion.score,
                    )
                )
        if state.phase == "candidate_in_progress" and state.pending_candidate:
            if self.progress is not None:
                self.progress.event("Cleaning up unfinished candidate from previous run state")
            self._cleanup_pending_candidate(config, state)
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
                self.progress.event("Baseline ready: score={0:.6f}".format(state.champion.score))

    def _advance_state_after_result(
        self,
        config: ProjectConfig,
        state: RunState,
        result: CandidateResult,
    ) -> None:
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

    def _finish_state(self, config: ProjectConfig, state: RunState) -> RunState:
        if state.status == "running":
            state.status = "completed"
        state.phase = "idle"
        state.updated_at = now_iso()
        self._write_state(config, state)
        if self.progress is not None:
            final_score = "n/a" if state.champion is None else "{0:.6f}".format(state.champion.score)
            self.progress.finish(
                "Run {0} finished with status={1}, champion score={2}".format(
                    state.run_id,
                    state.status,
                    final_score,
                )
            )
        return state

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
        worker_settings: Optional[WorkerSettings] = None,
        prompt: Optional[str] = None,
        planned_card: Optional[ExperimentCard] = None,
    ) -> CandidateResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        create_worktree(self.repo, worktree_path, branch_name, champion.branch)
        if self.progress is not None:
            self.progress.event(
                "Round {0}: candidate branch {1}".format(round_index, branch_name)
            )
        active_worker = worker_settings or config.worker
        if prompt is None:
            if planned_card is not None:
                prompt = build_execute_prompt(
                    worktree_path,
                    program=program_text(self.repo),
                    config=config,
                    run_id=run_id,
                    round_index=round_index,
                    champion=champion,
                    branch_name=branch_name,
                    history_rows=history_rows,
                    card=planned_card,
                    settings=active_worker,
                )
            else:
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
        if planned_card is not None:
            (artifact_dir / "card.json").write_text(
                json.dumps(planned_card.to_dict(), indent=2),
                encoding="utf-8",
            )
        status = "failed"
        score = None
        commit = None
        hypothesis = "No hypothesis provided."
        summary = "No summary provided."
        files_changed = 0
        scope_violation = False
        preserved_worktree = None
        try:
            worker_name = worker_display_name(active_worker)
            if planned_card is not None and active_worker.backend == "ollama":
                invocation = run_ollama_execute(
                    worktree_path,
                    artifact_dir,
                    prompt,
                    active_worker,
                    planned_card,
                    progress=self.progress,
                    stage_message="Round {0}: {1} working on {2}".format(round_index, worker_name, branch_name),
                )
            else:
                invocation = run_worker(
                    worktree_path,
                    artifact_dir,
                    prompt,
                    active_worker,
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
            else:
                if planned_card is not None:
                    outside_scope = [
                        relpath for relpath in changed_paths if relpath != planned_card.target_file
                    ]
                    if outside_scope:
                        scope_violation = True
                        summary = "Patch touched files outside planned scope: {0}".format(
                            ", ".join(outside_scope[:5])
                        )
                if scope_violation:
                    status = "failed"
                    pass
                else:
                    repeat_reason = None
                    if planned_card is None:
                        repeat_reason = hypothesis_repeat_reason(
                            history_rows,
                            hypothesis,
                            summary,
                            active_worker.forbidden_hypotheses,
                            allow_family_overlap=True,
                            allow_distinctive_token_overlap=True,
                        )
                    if repeat_reason is not None:
                        status = "duplicate"
                        summary = "Rejected as repeated idea. {0}. {1}".format(repeat_reason, summary)
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
            if config.git.preserve_candidate_worktrees:
                preserved_worktree, preservation_error = preserve_worktree_snapshot(
                    worktree_path,
                    artifact_dir,
                )
                if preservation_error and self.progress is not None:
                    self.progress.event(
                        "Round {0}: failed to preserve candidate snapshot ({1})".format(
                            round_index,
                            preservation_error,
                        )
                    )
            (artifact_dir / "result.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "hypothesis": hypothesis,
                        "summary": summary,
                        "commit": commit,
                        "score": score,
                        "files_changed": files_changed,
                        "preserved_worktree": preserved_worktree,
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
                preserved_worktree=preserved_worktree,
            )
        finally:
            remove_worktree(self.repo, worktree_path)

    def _cleanup_pending_candidate(self, config: ProjectConfig, state: RunState) -> None:
        pending = state.pending_candidate or {}
        worktree = pending.get("worktree")
        artifact_dir = pending.get("artifact_dir")
        branch = pending.get("branch")
        if config.git.preserve_candidate_worktrees and worktree and artifact_dir:
            preserved_worktree, preservation_error = preserve_worktree_snapshot(
                Path(worktree),
                Path(artifact_dir),
                label="preserved-worktree-interrupted",
            )
            if self.progress is not None and preserved_worktree:
                self.progress.event(
                    "Preserved unfinished candidate snapshot at {0}".format(preserved_worktree)
                )
            elif self.progress is not None and preservation_error:
                self.progress.event(
                    "Failed to preserve unfinished candidate snapshot ({0})".format(
                        preservation_error
                    )
                )
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

    def _load_or_create_state(
        self,
        config: ProjectConfig,
        run_id: Optional[str],
        *,
        mode: str = "run",
        plan_path: Optional[str] = None,
    ) -> RunState:
        if run_id:
            existing = self._find_run_id(config, run_id)
            if existing is not None:
                state = self._read_state(config, existing)
                if state.mode != mode:
                    raise RuntimeError(
                        "Run {0} uses mode {1}, not {2}".format(existing, state.mode, mode)
                    )
                if mode == "execute" and state.plan_path != plan_path:
                    raise RuntimeError(
                        "Run {0} is bound to a different plan: {1}".format(
                            existing,
                            state.plan_path,
                        )
                    )
                return state
            return self._create_state(
                config,
                run_id,
                self._latest_seed_source_state(config),
                mode=mode,
                plan_path=plan_path,
            )
        active_run_id = self._find_active_run_id(config, mode)
        if active_run_id is not None:
            state = self._read_state(config, active_run_id)
            if mode == "execute" and state.plan_path != plan_path:
                raise RuntimeError(
                    "Active execute run {0} is bound to a different plan: {1}".format(
                        active_run_id,
                        state.plan_path,
                    )
                )
            return state
        return self._create_state(
            config,
            make_run_id(),
            self._latest_seed_source_state(config),
            mode=mode,
            plan_path=plan_path,
        )

    def _create_state(
        self,
        config: ProjectConfig,
        run_id: str,
        seed_state: Optional[RunState] = None,
        *,
        mode: str = "run",
        plan_path: Optional[str] = None,
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
            mode=mode,
            plan_path=plan_path,
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

    def _find_active_run_id(self, config: ProjectConfig, mode: Optional[str] = None) -> Optional[str]:
        for run_id in self._list_run_ids(config):
            state = self._read_state(config, run_id)
            if mode is not None and state.mode != mode:
                continue
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

    def _plans_dir(self, config: ProjectConfig) -> Path:
        return self.repo / config.git.artifacts_dir / "plans"

    def _plan_dir(self, config: ProjectConfig, plan_id: str) -> Path:
        return self._plans_dir(config) / plan_id

    def _latest_plan_path(self, config: ProjectConfig) -> Optional[Path]:
        plans_dir = self._plans_dir(config)
        if not plans_dir.exists():
            return None
        candidates = sorted(
            [path / "plan.json" for path in plans_dir.iterdir() if path.is_dir() and (path / "plan.json").exists()],
            reverse=True,
        )
        return candidates[0] if candidates else None

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
    init_parser.add_argument("--config", help="Config path relative to the repo or absolute")
    init_parser.add_argument("--force", action="store_true", help="Overwrite scaffold files")

    run_parser = subparsers.add_parser("run", help="Run or resume the improvement loop")
    run_parser.add_argument("--repo", required=True, help="Path to the target repository")
    run_parser.add_argument("--config", help="Config path relative to the repo or absolute")
    run_parser.add_argument("--run-id", help="Resume a specific run id")

    plan_parser = subparsers.add_parser("plan", help="Generate experiment cards without executing them")
    plan_parser.add_argument("--repo", required=True, help="Path to the target repository")
    plan_parser.add_argument("--config", help="Config path relative to the repo or absolute")
    plan_parser.add_argument("--count", type=int, help="Override planner.cards_per_plan")
    plan_parser.add_argument("--output", help="Write the generated plan JSON to this path")

    execute_parser = subparsers.add_parser("execute", help="Execute a saved experiment plan")
    execute_parser.add_argument("--repo", required=True, help="Path to the target repository")
    execute_parser.add_argument("--config", help="Config path relative to the repo or absolute")
    execute_parser.add_argument("--plan", help="Plan path relative to the repo or absolute; defaults to the latest saved plan")
    execute_parser.add_argument("--run-id", help="Resume a specific execute-mode run id")

    status_parser = subparsers.add_parser("status", help="Show latest run status")
    status_parser.add_argument("--repo", required=True, help="Path to the target repository")
    status_parser.add_argument("--config", help="Config path relative to the repo or absolute")
    status_parser.add_argument("--run-id", help="Show a specific run id")

    report_parser = subparsers.add_parser("report", help="Generate a markdown report")
    report_parser.add_argument("--repo", required=True, help="Path to the target repository")
    report_parser.add_argument("--config", help="Config path relative to the repo or absolute")
    report_parser.add_argument("--run-id", help="Report a specific run id")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            repo = Path(args.repo).expanduser()
            config_path = resolve_cli_config_path(repo, args.config)
            ensure_project_files(repo, force=args.force, config_path=config_path)
            ensure_git_repo(repo)
            print("Initialized target repo at {0}".format(repo.resolve()))
            print(
                "Created {0} and {1}".format(
                    repo / PROGRAM_FILENAME,
                    config_path if config_path is not None else repo / CONFIG_FILENAME,
                )
            )
            return 0
        if args.command == "run":
            repo = Path(args.repo).expanduser()
            config_path = resolve_cli_config_path(repo, args.config)
            created = scaffold_missing_project_files(repo, config_path=config_path)
            ensure_git_repo(repo)
            if created:
                print("Scaffolded missing project files:", file=sys.stderr)
                for path in created:
                    print("- {0}".format(path), file=sys.stderr)
                rerun_command = "{0} run --repo {1}".format(CLI_NAME, repo)
                local_rerun_command = "python3 run.py run --repo {0}".format(repo)
                if args.config:
                    rerun_command += " --config {0}".format(args.config)
                    local_rerun_command += " --config {0}".format(args.config)
                print(
                    "Edit them and rerun `{0}` (or `{1}` for local development).".format(
                        rerun_command,
                        local_rerun_command,
                    ),
                    file=sys.stderr,
                )
                return 1
            load_project_config(repo, config_path)
            progress = ProgressReporter()
            print(
                render_status(
                    Orchestrator(repo, progress=progress, config_path=config_path).run(
                        run_id=args.run_id
                    )
                )
            )
            return 0
        if args.command == "plan":
            repo = Path(args.repo).expanduser()
            config_path = resolve_cli_config_path(repo, args.config)
            output_path = resolve_cli_repo_path(repo, args.output)
            created = scaffold_missing_project_files(repo, config_path=config_path)
            ensure_git_repo(repo)
            if created:
                print("Scaffolded missing project files:", file=sys.stderr)
                for path in created:
                    print("- {0}".format(path), file=sys.stderr)
                rerun_command = "{0} plan --repo {1}".format(CLI_NAME, repo)
                local_rerun_command = "python3 run.py plan --repo {0}".format(repo)
                if args.config:
                    rerun_command += " --config {0}".format(args.config)
                    local_rerun_command += " --config {0}".format(args.config)
                print(
                    "Edit them and rerun `{0}` (or `{1}` for local development).".format(
                        rerun_command,
                        local_rerun_command,
                    ),
                    file=sys.stderr,
                )
                return 1
            load_project_config(repo, config_path)
            progress = ProgressReporter()
            plan = Orchestrator(repo, progress=progress, config_path=config_path).plan(
                output_path=output_path,
                card_count=args.count,
            )
            final_path = output_path or (Path(plan.artifact_dir) / "plan.json")
            print("Plan: {0}".format(final_path))
            print("Cards: {0}".format(len(plan.cards)))
            return 0
        if args.command == "execute":
            repo = Path(args.repo).expanduser()
            config_path = resolve_cli_config_path(repo, args.config)
            ensure_git_repo(repo)
            load_project_config(repo, config_path)
            orchestrator = Orchestrator(repo, progress=ProgressReporter(), config_path=config_path)
            plan_path = resolve_cli_repo_path(repo, args.plan)
            if plan_path is None:
                plan_path = orchestrator.latest_plan_path()
            if plan_path is None or not plan_path.exists():
                raise FileNotFoundError("No plan found. Use `plan` first or pass --plan.")
            print(
                render_status(
                    orchestrator.execute(
                        plan_path=plan_path,
                        run_id=args.run_id,
                    )
                )
            )
            return 0
        if args.command == "status":
            repo = Path(args.repo).expanduser()
            config_path = resolve_cli_config_path(repo, args.config)
            print(
                render_status(
                    Orchestrator(repo, config_path=config_path).status(run_id=args.run_id)
                )
            )
            return 0
        if args.command == "report":
            repo = Path(args.repo).expanduser()
            config_path = resolve_cli_config_path(repo, args.config)
            state, results = Orchestrator(repo, config_path=config_path).report(run_id=args.run_id)
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


def normalize_hypothesis_token(token: str) -> str:
    value = token.lower().strip()
    if not value:
        return ""
    if value.endswith("ies") and len(value) > 4:
        value = value[:-3] + "y"
    elif value.endswith("ing") and len(value) > 5:
        value = value[:-3]
    elif value.endswith("ed") and len(value) > 4:
        value = value[:-2]
    elif value.endswith("es") and len(value) > 4:
        value = value[:-2]
    elif value.endswith("s") and len(value) > 3:
        value = value[:-1]
    replacements = {
        "ordering": "order",
        "ordered": "order",
        "reduced": "reduction",
        "reducing": "reduction",
        "reductions": "reduction",
        "prune": "prun",
        "prunes": "prun",
        "pruning": "prun",
        "histori": "history",
        "histories": "history",
        "moves": "move",
        "extensions": "extension",
    }
    return replacements.get(value, value)


def extract_hypothesis_tokens(text: str) -> List[str]:
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        token = normalize_hypothesis_token(raw)
        if not token or token in HYPOTHESIS_STOPWORDS:
            continue
        if len(token) < 2 and token != "tt":
            continue
        tokens.append(token)
    return tokens


def build_hypothesis_profile(hypothesis: str, summary: str = "") -> Dict[str, Any]:
    normalized = normalize_hypothesis(hypothesis)
    source = " ".join(part for part in (hypothesis, summary) if str(part).strip())
    tokens = extract_hypothesis_tokens(source)
    token_set = set(tokens)
    token_sequence = " ".join(tokens)
    families = []
    for label, required_tokens in HYPOTHESIS_THEME_PATTERNS:
        regexes = HYPOTHESIS_THEME_TOKEN_SEQUENCE_REGEXES.get(label)
        if regexes:
            if any(regex.search(token_sequence) for regex in regexes):
                families.append(label)
            continue
        if all(token in token_set for token in required_tokens):
            families.append(label)
    distinctive = {
        token
        for token in token_set
        if token not in HYPOTHESIS_COMMON_OVERLAP_TOKENS and token not in {"search", "engine"}
    }
    return {
        "normalized": normalized,
        "tokens": token_set,
        "families": families,
        "distinctive_tokens": distinctive,
    }


def build_repeat_reference_rows(
    rows: List[Dict[str, str]],
    forbidden_hypotheses: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    references = []
    for index, hypothesis in enumerate(forbidden_hypotheses or [], start=1):
        references.append(
            {
                "run_id": "repo",
                "round": "seed-{0}".format(index),
                "hypothesis": hypothesis,
                "summary": "Repo-configured forbidden direction.",
            }
        )
    references.extend(rows)
    return references


def render_forbidden_repeat_guidance(
    rows: List[Dict[str, str]],
    forbidden_hypotheses: Optional[List[str]] = None,
    limit: int = 8,
) -> str:
    reference_rows = build_repeat_reference_rows(rows, forbidden_hypotheses)
    if not reference_rows:
        return "- None yet."
    families: Dict[str, Dict[str, Any]] = {}
    fallback_rows = []
    for index, row in enumerate(reference_rows):
        hypothesis = str(row.get("hypothesis", ""))
        summary = str(row.get("summary", ""))
        profile = build_hypothesis_profile(hypothesis, summary)
        if profile["families"]:
            for label in profile["families"]:
                entry = families.setdefault(
                    label,
                    {
                        "count": 0,
                        "index": index,
                        "example": hypothesis,
                    },
                )
                entry["count"] += 1
                entry["index"] = index
                entry["example"] = hypothesis
        elif profile["normalized"] and profile["normalized"] != normalize_hypothesis("No hypothesis provided."):
            fallback_rows.append((index, hypothesis))
    if families:
        ranked = sorted(
            families.items(),
            key=lambda item: (-item[1]["count"], -item[1]["index"], item[0]),
        )[:limit]
        return "\n".join(
            "- {0}: seen {1}x; latest example=\"{2}\"".format(
                label,
                entry["count"],
                truncate_middle(entry["example"], 96),
            )
            for label, entry in ranked
        )
    recent = fallback_rows[-limit:]
    if not recent:
        return "- None yet."
    return "\n".join(
        "- recent hypothesis=\"{0}\"".format(truncate_middle(hypothesis, 96))
        for _, hypothesis in reversed(recent)
    )


def normalize_hypothesis(text: str) -> str:
    return " ".join(text.lower().split())


def hypothesis_repeat_reason(
    rows: List[Dict[str, str]],
    hypothesis: str,
    summary: str = "",
    forbidden_hypotheses: Optional[List[str]] = None,
    *,
    allow_family_overlap: bool = True,
    allow_distinctive_token_overlap: bool = True,
) -> Optional[str]:
    profile = build_hypothesis_profile(hypothesis, summary)
    normalized = profile["normalized"]
    if not normalized or normalized == normalize_hypothesis("No hypothesis provided."):
        return None
    for row in build_repeat_reference_rows(rows, forbidden_hypotheses):
        previous_hypothesis = str(row.get("hypothesis", ""))
        previous_summary = str(row.get("summary", ""))
        previous_profile = build_hypothesis_profile(previous_hypothesis, previous_summary)
        previous_normalized = previous_profile["normalized"]
        is_repo_seed = row.get("run_id") == "repo"
        reference_label = (
            "repo-configured forbidden direction"
            if is_repo_seed
            else "run={0} round={1}".format(row.get("run_id", ""), row.get("round", ""))
        )
        if previous_normalized and previous_normalized == normalized:
            return "same hypothesis text as {0}".format(reference_label)
        if allow_family_overlap:
            family_overlap = sorted(set(profile["families"]) & set(previous_profile["families"]))
            if family_overlap:
                return "same idea family ({0}) as {1}".format(
                    ", ".join(family_overlap),
                    reference_label,
                )
        if allow_distinctive_token_overlap:
            shared_tokens = profile["distinctive_tokens"] & previous_profile["distinctive_tokens"]
            min_size = min(len(profile["distinctive_tokens"]), len(previous_profile["distinctive_tokens"]))
            if len(shared_tokens) >= 4 and min_size > 0 and len(shared_tokens) >= max(4, int(min_size * 0.6)):
                return "same key terms ({0}) as {1}".format(
                    ", ".join(sorted(shared_tokens)[:5]),
                    reference_label,
                )
    return None


def hypothesis_seen_before(rows: List[Dict[str, str]], hypothesis: str) -> bool:
    return hypothesis_repeat_reason(rows, hypothesis) is not None


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
