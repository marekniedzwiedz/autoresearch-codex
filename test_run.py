from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from run import (
    EXECUTOR_OUTPUT_SCHEMA,
    OLLAMA_WORKER_OUTPUT_SCHEMA,
    PLAN_OUTPUT_SCHEMA,
    WORKER_OUTPUT_SCHEMA,
    ChampionState,
    EvaluatorSettings,
    ExperimentCard,
    GitSettings,
    PlannerSettings,
    ProjectConfig,
    SearchSettings,
    WorkerInvocationResult,
    apply_executor_edit_ops,
    apply_unified_diff,
    build_focused_target_snapshot_entry,
    build_hypothesis_profile,
    build_ollama_execute_prompt,
    build_ollama_patch_repair_prompt,
    build_planner_prompt,
    build_repo_snapshot,
    build_evaluator_context_env,
    build_ollama_generate_payload,
    extract_patch_failure_locations,
    extract_ollama_structured_output,
    hypothesis_repeat_reason,
    load_project_config,
    merge_repaired_worker_output,
    normalize_executor_output,
    normalize_plan_output,
    normalize_patch_text,
    patch_additions_already_present,
    preserve_worktree_snapshot,
    promote_repair_artifacts,
    render_patch_failure_context,
    render_forbidden_repeat_guidance,
    resolve_cli_config_path,
    resolve_cli_repo_path,
    scaffold_missing_project_files,
    select_preferred_ollama_model,
    WorkerSettings,
)


class EvaluatorContextEnvTests(unittest.TestCase):
    def test_build_evaluator_context_env_includes_expected_values(self) -> None:
        env = build_evaluator_context_env(
            run_id="run-123",
            round_index=7,
            artifact_dir=Path("/tmp/artifacts"),
            artifacts_root=Path("/tmp/.evoloza-campaign"),
            worktree=Path("/tmp/worktree"),
            base_branch="main",
            champion_branch="evoloza/run-123/r006",
            champion_score=2512.25,
        )
        self.assertEqual(env["EVOLOZA_RUN_ID"], "run-123")
        self.assertEqual(env["EVOLOZA_ROUND"], "7")
        self.assertEqual(env["EVOLOZA_ARTIFACT_DIR"], "/tmp/artifacts")
        self.assertEqual(env["EVOLOZA_ARTIFACTS_ROOT"], "/tmp/.evoloza-campaign")
        self.assertEqual(env["EVOLOZA_WORKTREE"], "/tmp/worktree")
        self.assertEqual(env["EVOLOZA_BASE_BRANCH"], "main")
        self.assertEqual(env["EVOLOZA_CHAMPION_BRANCH"], "evoloza/run-123/r006")
        self.assertEqual(env["EVOLOZA_CHAMPION_SCORE"], "2512.250000")

    def test_build_evaluator_context_env_omits_unknown_champion_fields(self) -> None:
        env = build_evaluator_context_env(
            run_id="run-123",
            round_index=0,
            artifact_dir=Path("/tmp/artifacts"),
            artifacts_root=Path("/tmp/.evoloza-campaign"),
            worktree=Path("/tmp/repo"),
            base_branch="main",
        )
        self.assertNotIn("EVOLOZA_CHAMPION_BRANCH", env)
        self.assertNotIn("EVOLOZA_CHAMPION_SCORE", env)


class ConfigLoadingTests(unittest.TestCase):
    def test_resolve_cli_config_path_uses_repo_for_relative_paths(self) -> None:
        repo = Path("/tmp/example-repo")
        self.assertEqual(
            resolve_cli_config_path(repo, "config.ollama.toml"),
            repo / "config.ollama.toml",
        )
        self.assertEqual(
            resolve_cli_repo_path(repo, "plans/latest.json"),
            repo / "plans/latest.json",
        )

    def test_load_project_config_supports_worker_ollama_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "config.toml").write_text(
                """
[worker]
backend = "ollama"
model = "qwen2.5-coder:32b"
ollama_host = "http://127.0.0.1:11434"
context_files = ["solver.py", "tests/*.py"]
max_context_bytes = 9000
max_file_bytes = 3000
max_files = 5
temperature = 0.1
keep_alive = "30m"
request_timeout_seconds = 1800
think = false
forbidden_hypotheses = ["continuation history for quiet ordering"]

[worker.ollama_options]
num_ctx = 250000
num_predict = 512

[search]
max_rounds = 2

[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>[0-9]+)"

[git]
artifacts_dir = ".evoloza"
preserve_candidate_worktrees = true
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_project_config(repo)
        self.assertEqual(config.worker.backend, "ollama")
        self.assertEqual(config.worker.model, "qwen2.5-coder:32b")
        self.assertEqual(config.worker.ollama_host, "http://127.0.0.1:11434")
        self.assertEqual(config.worker.context_files, ["solver.py", "tests/*.py"])
        self.assertEqual(config.worker.max_context_bytes, 9000)
        self.assertEqual(config.worker.max_file_bytes, 3000)
        self.assertEqual(config.worker.max_files, 5)
        self.assertAlmostEqual(config.worker.temperature, 0.1)
        self.assertEqual(config.worker.keep_alive, "30m")
        self.assertEqual(config.worker.request_timeout_seconds, 1800)
        self.assertIs(config.worker.think, False)
        self.assertEqual(
            config.worker.forbidden_hypotheses,
            ["continuation history for quiet ordering"],
        )
        self.assertEqual(config.worker.ollama_options, {"num_ctx": 250000, "num_predict": 512})
        self.assertIs(config.git.preserve_candidate_worktrees, True)

    def test_load_project_config_supports_planner_and_executor_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "config.toml").write_text(
                """
[worker]
backend = "ollama"
model = "qwen3.5:35b"
context_files = ["engine.rs"]

[planner]
backend = "codex"
binary = "codex"
model = "gpt-5.4"
cards_per_plan = 6

[executor]
backend = "ollama"
model = "qwen3.5:35b"
forbidden_hypotheses = ["continuation history bonus for quiet moves"]

[worker.ollama_options]
num_ctx = 250000

[executor.ollama_options]
num_ctx = 131072
num_predict = 1024

[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>[0-9]+)"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_project_config(repo)
        self.assertEqual(config.planner.cards_per_plan, 6)
        self.assertEqual(config.planner.worker.backend, "codex")
        self.assertEqual(config.planner.worker.model, "gpt-5.4")
        self.assertEqual(config.executor.backend, "ollama")
        self.assertEqual(config.executor.model, "qwen3.5:35b")
        self.assertEqual(
            config.executor.forbidden_hypotheses,
            ["continuation history bonus for quiet moves"],
        )
        self.assertEqual(
            config.executor.ollama_options,
            {"num_ctx": 131072, "num_predict": 1024},
        )

    def test_load_project_config_supports_explicit_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            config_path = repo / "config.ollama.toml"
            config_path.write_text(
                """
[worker]
backend = "ollama"
model = "qwen3.5:35b"

[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>[0-9]+)"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_project_config(repo, config_path=config_path)
        self.assertEqual(config.worker.backend, "ollama")
        self.assertEqual(config.worker.model, "qwen3.5:35b")

    def test_load_project_config_supports_legacy_codex_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "config.toml").write_text(
                """
[codex]
binary = "/usr/local/bin/codex"
model = "gpt-5"
extra_args = ["--profile", "default"]

[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>[0-9]+)"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_project_config(repo)
        self.assertEqual(config.worker.backend, "codex")
        self.assertEqual(config.worker.binary, "/usr/local/bin/codex")
        self.assertEqual(config.worker.model, "gpt-5")
        self.assertEqual(config.worker.extra_args, ["--profile", "default"])

    def test_load_project_config_rejects_invalid_ollama_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "config.toml").write_text(
                """
[worker]
backend = "ollama"
request_timeout_seconds = 0

[evaluator]
commands = ["python3 benchmark.py"]
score_regex = "EVOLOZA_SCORE=(?P<score>[0-9]+)"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "worker.request_timeout_seconds"):
                load_project_config(repo)

    def test_scaffold_missing_project_files_uses_explicit_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            config_path = repo / "config.ollama.toml"
            created = scaffold_missing_project_files(repo, config_path=config_path)
            self.assertEqual(created, [repo / "program.md", config_path])
            self.assertTrue(config_path.exists())
            self.assertFalse((repo / "config.toml").exists())


class OllamaModelSelectionTests(unittest.TestCase):
    def test_select_preferred_ollama_model_prefers_coder_models(self) -> None:
        selected = select_preferred_ollama_model(
            ["qwen3:30b", "codestral:latest", "qwen2.5-coder:32b"]
        )
        self.assertEqual(selected, "qwen2.5-coder:32b")


class PatchNormalizationTests(unittest.TestCase):
    def test_preserve_worktree_snapshot_copies_worktree_without_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            worktree = repo / "worktree"
            artifact_dir = repo / "artifacts"
            (worktree / "rust_sota/target/release").mkdir(parents=True, exist_ok=True)
            (worktree / ".git").write_text("gitdir: /tmp/example\n", encoding="utf-8")
            (worktree / "rust_sota/src").mkdir(parents=True, exist_ok=True)
            (worktree / "rust_sota/src/engine.rs").write_text("fn main() {}\n", encoding="utf-8")
            (worktree / "rust_sota/target/release/rust_sota").write_text("binary\n", encoding="utf-8")
            snapshot_dir, error_path = preserve_worktree_snapshot(worktree, artifact_dir)
            self.assertIsNone(error_path)
            self.assertIsNotNone(snapshot_dir)
            snapshot = Path(snapshot_dir or "")
            self.assertTrue((snapshot / "rust_sota/src/engine.rs").exists())
            self.assertTrue((snapshot / "rust_sota/target/release/rust_sota").exists())
            self.assertFalse((snapshot / ".git").exists())
            self.assertTrue((artifact_dir / "preserved-worktree.json").exists())

    def test_normalize_patch_text_repairs_missing_context_prefixes(self) -> None:
        patch = normalize_patch_text(
            """diff --git a/solver.py b/solver.py
index 3b8d6ad..c019452 100644
--- a/solver.py
+++ b/solver.py
@@ -1,2 +1,2 @@
def score_value():
-    return 1
+    return 2
"""
        )
        self.assertIn("\n def score_value():\n", patch)

    def test_extract_patch_failure_locations_reads_git_apply_errors(self) -> None:
        stderr_text = (
            "error: patch failed: rust_sota/src/engine.rs:2077\n"
            "error: rust_sota/src/engine.rs: patch does not apply\n"
            "error: patch failed: rust_sota/src/main.rs:44\n"
        )
        self.assertEqual(
            extract_patch_failure_locations(stderr_text),
            [("rust_sota/src/engine.rs", 2077), ("rust_sota/src/main.rs", 44)],
        )

    def test_render_patch_failure_context_includes_nearby_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            target = repo / "rust_sota/src/engine.rs"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("alpha\nbeta\ngamma\ndelta\nepsilon\n", encoding="utf-8")
            context = render_patch_failure_context(
                repo,
                "error: patch failed: rust_sota/src/engine.rs:3\n",
                radius=1,
            )
        self.assertIn("File: rust_sota/src/engine.rs", context)
        self.assertIn("     2: beta", context)
        self.assertIn("     3: gamma", context)
        self.assertIn("     4: delta", context)

    def test_build_ollama_patch_repair_prompt_mentions_empty_patch_guard(self) -> None:
        prompt = build_ollama_patch_repair_prompt(
            {
                "hypothesis": "x",
                "summary": "y",
                "files_touched": ["rust_sota/src/engine.rs"],
                "local_checks_run": [],
                "risks": [],
                "patch": "--- a/file\n+++ b/file\n",
            },
            "error: patch failed: rust_sota/src/engine.rs:3",
            "File: rust_sota/src/engine.rs\nApprox failed line: 3\n     3: gamma",
        )
        self.assertIn("return an empty patch", prompt)
        self.assertIn("Current snapshot excerpts around failed hunks", prompt)

    def test_patch_additions_already_present_detects_redundant_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            target = repo / "rust_sota/src/engine.rs"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "fn score() {\n    let quiet_history_score = history_score + continuation_score;\n}\n",
                encoding="utf-8",
            )
            patch = normalize_patch_text(
                """
diff --git a/rust_sota/src/engine.rs b/rust_sota/src/engine.rs
--- a/rust_sota/src/engine.rs
+++ b/rust_sota/src/engine.rs
@@ -1,2 +1,3 @@
 fn score() {
+    let quiet_history_score = history_score + continuation_score;
 }
"""
            )
            self.assertTrue(patch_additions_already_present(repo, patch))

    def test_apply_unified_diff_wraps_hunk_only_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            target = repo / "rust_sota/src/engine.rs"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                (
                    "fn quiet_history_score(\n"
                    "    score: i32,\n"
                    "    current_key: usize,\n"
                    ") -> i32 {\n"
                    "    let mut score = score;\n"
                    "    score += 1;\n"
                    "    score\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            patch = normalize_patch_text(
                """
@@ -3,5 +3,6 @@ fn quiet_history_score(
     current_key: usize,
 ) -> i32 {
     let mut score = score;
+    let _unused = current_key;
     score += 1;
     score
"""
            )
            self.assertIsNone(apply_unified_diff(repo, repo, patch))
            self.assertIn("let _unused = current_key;", target.read_text(encoding="utf-8"))

    def test_apply_unified_diff_falls_back_to_trimmed_context_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            target = repo / "rust_sota/src/engine.rs"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                (
                    "fn negamax() {\n"
                    "    let raw = evaluate();\n"
                    "    let se = raw;\n"
                    "    raw_static_eval = Some(raw);\n"
                    "    static_eval = Some(se);\n"
                    "    if let Some(parent) = parent_static_eval {\n"
                    "        improving = se >= parent - 18;\n"
                    "    }\n"
                    "    if se - margin >= beta {\n"
                    "        return Ok(se);\n"
                    "    }\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            patch = normalize_patch_text(
                """
diff --git a/rust_sota/src/engine.rs b/rust_sota/src/engine.rs
--- a/rust_sota/src/engine.rs
+++ b/rust_sota/src/engine.rs
@@ -2,10 +2,15 @@ fn negamax() {
     let raw = evaluate();
     let se = raw;
     raw_static_eval = Some(raw);
     static_eval = Some(se);
+    let corrected_eval = apply_correction_to_eval(se, correction_history_value());
     if let Some(parent) = parent_static_eval {
-        improving = se >= parent - 18;
+        improving = corrected_eval >= parent - 18;
     }
+    // stale trailing context that should be trimmed away by fallback
+    if depth <= 2 {
+        return Ok(corrected_eval);
+    }
     if se - margin >= beta {
         return Ok(se);
     }
 """
            )
            self.assertIsNone(apply_unified_diff(repo, repo, patch))
            rewritten = target.read_text(encoding="utf-8")
            self.assertIn("let corrected_eval = apply_correction_to_eval(se, correction_history_value());", rewritten)
            self.assertIn("improving = corrected_eval >= parent - 18;", rewritten)

    def test_apply_unified_diff_prefers_active_engine_over_snapshot_tie(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            active = repo / "rust_sota/src/engine.rs"
            snapshot = repo / "tools/sota/2026-04-07_rust_sota_v0.9.1/src/engine.rs"
            active.parent.mkdir(parents=True, exist_ok=True)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            content = (
                "fn quiet_history_score(\n"
                "    score: i32,\n"
                "    current_key: usize,\n"
                ") -> i32 {\n"
                "    let mut score = score;\n"
                "    score += 1;\n"
                "    score\n"
                "}\n"
            )
            active.write_text(content, encoding="utf-8")
            snapshot.write_text(content, encoding="utf-8")
            patch = normalize_patch_text(
                """
@@ -3,5 +3,6 @@ fn quiet_history_score(
     current_key: usize,
 ) -> i32 {
     let mut score = score;
+    let _active_only = current_key;
     score += 1;
     score
"""
            )
            self.assertIsNone(apply_unified_diff(repo, repo, patch))
            self.assertIn("let _active_only = current_key;", active.read_text(encoding="utf-8"))
            self.assertNotIn("let _active_only = current_key;", snapshot.read_text(encoding="utf-8"))

    def test_apply_executor_edit_ops_materializes_anchored_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            target = repo / "rust_sota/src/engine.rs"
            target.parent.mkdir(parents=True, exist_ok=True)
            before = (
                "fn negamax() {\n"
                "    let raw = evaluate();\n"
                "    let se = raw;\n"
                "    raw_static_eval = Some(raw);\n"
                "    static_eval = Some(se);\n"
                "}\n"
            )
            target.write_text(before, encoding="utf-8")
            error = apply_executor_edit_ops(
                repo,
                repo,
                "rust_sota/src/engine.rs",
                [
                    {
                        "file": "rust_sota/src/engine.rs",
                        "action": "insert_after",
                        "anchor_snippet": "    let se = raw;\n",
                        "new_text": "    let corrected = apply_correction_to_eval(se, correction_history_value());\n",
                    }
                ],
                max_patch_lines=30,
            )
            self.assertIsNone(error)
            rewritten = target.read_text(encoding="utf-8")
            self.assertIn("let corrected = apply_correction_to_eval(se, correction_history_value());", rewritten)
            patch_text = (repo / "candidate.patch").read_text(encoding="utf-8")
            self.assertIn("diff --git a/rust_sota/src/engine.rs b/rust_sota/src/engine.rs", patch_text)
            self.assertIn("+    let corrected = apply_correction_to_eval(se, correction_history_value());", patch_text)

    def test_apply_executor_edit_ops_rejects_ambiguous_anchor_without_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            target = repo / "rust_sota/src/engine.rs"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                (
                    "fn first() {\n"
                    "    let se = raw;\n"
                    "}\n"
                    "fn second() {\n"
                    "    let se = raw;\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            error = apply_executor_edit_ops(
                repo,
                repo,
                "rust_sota/src/engine.rs",
                [
                    {
                        "file": "rust_sota/src/engine.rs",
                        "action": "insert_after",
                        "anchor_snippet": "    let se = raw;\n",
                        "new_text": "    let corrected = apply_correction_to_eval(se, 1);\n",
                    }
                ],
            )
            self.assertIsNotNone(error)
            self.assertIn("matched 2 locations", error or "")

    def test_promote_repair_artifacts_copies_final_executor_patch_to_candidate_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "candidate"
            repair_dir = artifact_dir / "repair-01"
            repair_dir.mkdir(parents=True, exist_ok=True)
            (repair_dir / "candidate.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
            (repair_dir / "last_message.json").write_text('{"hypothesis":"x","edit_ops":[]}', encoding="utf-8")
            (repair_dir / "ollama.stderr.log").write_text("", encoding="utf-8")
            promoted = promote_repair_artifacts(
                artifact_dir,
                WorkerInvocationResult(
                    returncode=0,
                    output_path=str(repair_dir / "ollama.response.json"),
                    stderr_path=str(repair_dir / "ollama.stderr.log"),
                    last_message_path=str(repair_dir / "last_message.json"),
                    structured_output={"hypothesis": "x", "edit_ops": []},
                    usage=None,
                ),
            )
            self.assertTrue((artifact_dir / "candidate.patch").exists())
            self.assertTrue((artifact_dir / "last_message.json").exists())
            self.assertEqual(promoted.last_message_path, str(artifact_dir / "last_message.json"))
            self.assertEqual(promoted.stderr_path, str(artifact_dir / "ollama.stderr.log"))


class OllamaResponseParsingTests(unittest.TestCase):
    def test_extract_ollama_structured_output_falls_back_to_thinking(self) -> None:
        payload = extract_ollama_structured_output(
            {
                "response": "",
                "thinking": '{"hypothesis":"x","summary":"y","files_touched":[],"local_checks_run":[],"risks":[],"patch":""}',
            }
        )
        self.assertIsNotNone(payload)
        self.assertEqual(payload["hypothesis"], "x")


class OllamaPayloadTests(unittest.TestCase):
    def test_worker_output_schema_requires_patch(self) -> None:
        self.assertIn("patch", WORKER_OUTPUT_SCHEMA["required"])
        self.assertEqual(WORKER_OUTPUT_SCHEMA["properties"]["patch"]["type"], "string")
        self.assertEqual(OLLAMA_WORKER_OUTPUT_SCHEMA["required"], ["hypothesis", "patch"])

    def test_build_ollama_generate_payload_supports_repo_configurable_request_fields(self) -> None:
        settings = WorkerSettings(
            backend="ollama",
            model="qwen3.5:35b",
            temperature=0.15,
            keep_alive="30m",
            think=False,
            ollama_options={"num_ctx": 250000, "num_predict": 512},
        )
        payload = build_ollama_generate_payload("prompt body", settings)
        self.assertEqual(payload["prompt"], "prompt body")
        self.assertEqual(payload["format"], OLLAMA_WORKER_OUTPUT_SCHEMA)
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertIs(payload["think"], False)
        self.assertEqual(
            payload["options"],
            {"num_ctx": 250000, "num_predict": 512, "temperature": 0.15},
        )

    def test_build_ollama_generate_payload_allows_ollama_options_to_override_temperature(self) -> None:
        settings = WorkerSettings(
            backend="ollama",
            temperature=0.2,
            ollama_options={"temperature": 0.05, "seed": 7},
        )
        payload = build_ollama_generate_payload("prompt body", settings)
        self.assertEqual(payload["options"], {"temperature": 0.05, "seed": 7})

    def test_build_ollama_generate_payload_accepts_custom_schema(self) -> None:
        settings = WorkerSettings(backend="ollama")
        payload = build_ollama_generate_payload(
            "prompt body",
            settings,
            response_schema=PLAN_OUTPUT_SCHEMA,
        )
        self.assertEqual(payload["format"], PLAN_OUTPUT_SCHEMA)


class PlannerOutputTests(unittest.TestCase):
    def test_plan_schema_requires_all_declared_card_fields(self) -> None:
        required = set(PLAN_OUTPUT_SCHEMA["properties"]["cards"]["items"]["required"])
        declared = set(PLAN_OUTPUT_SCHEMA["properties"]["cards"]["items"]["properties"].keys())
        self.assertEqual(required, declared)

    def test_executor_schema_requires_declared_top_level_fields(self) -> None:
        required = set(EXECUTOR_OUTPUT_SCHEMA["required"])
        declared = set(EXECUTOR_OUTPUT_SCHEMA["properties"].keys())
        self.assertEqual(required, {"hypothesis", "edit_ops"})
        self.assertTrue(required.issubset(declared))

    def test_normalize_plan_output_keeps_valid_cards(self) -> None:
        normalized = normalize_plan_output(
            {
                "cards": [
                    {
                        "hypothesis": "Add a recapture ordering heuristic.",
                        "target_file": "rust_sota/src/engine.rs",
                        "target_symbols": ["move_ordered"],
                        "anchor_snippets": [
                            "fn move_ordered(\n    moves: &[Move],\n) -> Move {\n"
                        ],
                        "allowed_edit_scope": "Touch only move_ordered in engine.rs.",
                        "forbidden_families": ["continuation-history"],
                        "implementation_notes": "Keep it tiny.",
                        "max_patch_lines": 55,
                    },
                    {
                        "hypothesis": "",
                        "target_file": "",
                    },
                ]
            }
        )
        self.assertIsNotNone(normalized)
        cards = normalized["cards"]
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["id"], "card-001")
        self.assertEqual(cards[0]["target_symbols"], ["move_ordered"])
        self.assertEqual(cards[0]["anchor_snippets"], ["fn move_ordered(\n    moves: &[Move],\n) -> Move {"])
        self.assertEqual(cards[0]["forbidden_families"], ["continuation-history"])
        self.assertEqual(cards[0]["max_patch_lines"], 55)

    def test_normalize_executor_output_preserves_multiline_anchor_snippets(self) -> None:
        normalized = normalize_executor_output(
            {
                "hypothesis": "Wire correction history into pruning eval.",
                "edit_ops": [
                    {
                        "file": "rust_sota/src/engine.rs",
                        "action": "insert_after",
                        "anchor_snippet": "    let se = raw;\n    raw_static_eval = Some(raw);\n",
                        "new_text": "    let corrected = apply_correction_to_eval(se, 12);\n",
                    }
                ],
            }
        )
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["edit_ops"][0]["anchor_snippet"], "    let se = raw;\n    raw_static_eval = Some(raw);")


class FocusedSnapshotTests(unittest.TestCase):
    def test_build_focused_target_snapshot_entry_prefers_symbol_and_identifier_excerpts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir)
            target = worktree / "engine.rs"
            target.write_text(
                "\n".join(
                    [
                        "fn helper() {",
                        '    let unrelated = "background";',
                        "}",
                        "",
                        "fn negamax() {",
                        "    let raw_static_eval = raw;",
                        "    let se = raw;",
                        "    if correction_history_value() > 0 {",
                        "        update_correction_histories();",
                        "    }",
                        "}",
                        "",
                        "fn correction_history_value() -> i32 {",
                        "    0",
                        "}",
                        "",
                        "fn update_correction_histories() {",
                        "    let marker = 1;",
                        "}",
                        "",
                        *["fn filler_{0}() {{}}".format(index) for index in range(40)],
                        "",
                        'fn unrelated_large() { let marker = "omit-me"; }',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            card = ExperimentCard(
                id="card-1",
                hypothesis="Use correction history inside negamax.",
                summary="Wire `raw_static_eval` through correction-history pruning.",
                target_file="engine.rs",
                target_symbols=["negamax", "correction_history_value", "update_correction_histories"],
                anchor_snippets=[
                    "fn negamax() {\n    let raw_static_eval = raw;\n    let se = raw;\n",
                    "fn correction_history_value() -> i32 {\n    0\n}\n",
                ],
                allowed_edit_scope="Touch only `negamax` and helpers in `engine.rs`.",
                forbidden_families=[],
                implementation_notes="Reuse `raw_static_eval` and `correction_history_value` before calling `update_correction_histories`.",
                max_patch_lines=80,
            )
            entry = build_focused_target_snapshot_entry(worktree, "engine.rs", card, WorkerSettings())
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertIn("focused excerpts", entry)
        self.assertIn("fn negamax()", entry)
        self.assertIn("let raw_static_eval = raw;", entry)
        self.assertIn("fn correction_history_value()", entry)
        self.assertIn("fn update_correction_histories()", entry)
        self.assertNotIn('omit-me', entry)

    def test_build_repo_snapshot_supports_excluding_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True)
            (worktree / "engine.rs").write_text("fn main() {}\n", encoding="utf-8")
            (worktree / "Cargo.toml").write_text("[package]\nname = 'x'\n", encoding="utf-8")
            subprocess.run(["git", "add", "engine.rs", "Cargo.toml"], cwd=worktree, check=True, capture_output=True)
            config = ProjectConfig(
                worker=WorkerSettings(backend="ollama", context_files=["engine.rs", "Cargo.toml"], max_files=4),
                planner=PlannerSettings(),
                executor=WorkerSettings(),
                evaluator=EvaluatorSettings(commands=["python3 bench.py"], score_regex="EVOLOZA_SCORE=(?P<score>[0-9]+)"),
                search=SearchSettings(),
                git=GitSettings(artifacts_dir=".evoloza"),
            )
            snapshot, included_paths, omitted_paths = build_repo_snapshot(
                worktree,
                config,
                "Touch only engine.rs.",
                exclude_paths=["engine.rs"],
            )
        self.assertIn("Cargo.toml", snapshot)
        self.assertNotIn("engine.rs", snapshot)
        self.assertEqual(included_paths, ["Cargo.toml"])
        self.assertEqual(omitted_paths, [])

    def test_build_ollama_execute_prompt_uses_focused_target_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True)
            (worktree / "engine.rs").write_text(
                "fn negamax() {\n    let raw_static_eval = raw;\n    let se = raw;\n}\n",
                encoding="utf-8",
            )
            (worktree / "Cargo.toml").write_text("[package]\nname = 'x'\n", encoding="utf-8")
            subprocess.run(["git", "add", "engine.rs", "Cargo.toml"], cwd=worktree, check=True, capture_output=True)
            config = ProjectConfig(
                worker=WorkerSettings(),
                planner=PlannerSettings(),
                executor=WorkerSettings(
                    backend="ollama",
                    context_files=["engine.rs", "Cargo.toml"],
                    max_files=4,
                    max_context_bytes=120000,
                    max_file_bytes=24000,
                ),
                evaluator=EvaluatorSettings(commands=["python3 bench.py"], score_regex="EVOLOZA_SCORE=(?P<score>[0-9]+)"),
                search=SearchSettings(),
                git=GitSettings(artifacts_dir=".evoloza"),
            )
            card = ExperimentCard(
                id="card-1",
                hypothesis="Use correction history inside negamax.",
                summary="Wire `raw_static_eval` into pruning.",
                target_file="engine.rs",
                target_symbols=["negamax"],
                anchor_snippets=["fn negamax() {\n    let raw_static_eval = raw;\n    let se = raw;\n}\n"],
                allowed_edit_scope="Touch only `negamax` in `engine.rs`.",
                forbidden_families=[],
                implementation_notes="Reuse `raw_static_eval`.",
                max_patch_lines=40,
            )
            prompt = build_ollama_execute_prompt(
                worktree,
                "Improve the engine.",
                config,
                "run-1",
                1,
                ChampionState(branch="main", commit="abc123", score=1.0, summary="Baseline"),
                "evoloza/run-1/r001",
                [],
                card,
                config.executor,
            )
        self.assertIn("Target file view: focused excerpts", prompt)
        self.assertIn("engine.rs (focused excerpts)", prompt)
        self.assertIn("Cargo.toml", prompt)
        self.assertIn("edit_ops", prompt)
        self.assertIn("anchor_snippets", prompt)
        self.assertNotIn("Repository file list:", prompt)

    def test_build_codex_planner_prompt_uses_bounded_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=worktree, check=True, capture_output=True)
            (worktree / "program.md").write_text("# Mission\nImprove the engine.\n", encoding="utf-8")
            (worktree / "engine.rs").write_text("fn negamax() {}\n", encoding="utf-8")
            (worktree / "Cargo.toml").write_text("[package]\nname = 'x'\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "program.md", "engine.rs", "Cargo.toml"],
                cwd=worktree,
                check=True,
                capture_output=True,
            )
            config = ProjectConfig(
                worker=WorkerSettings(),
                planner=PlannerSettings(
                    worker=WorkerSettings(
                        backend="codex",
                        model="gpt-5.4",
                        context_files=["engine.rs", "Cargo.toml"],
                        max_context_bytes=64000,
                        max_file_bytes=16000,
                        max_files=2,
                    ),
                    cards_per_plan=4,
                ),
                executor=WorkerSettings(),
                evaluator=EvaluatorSettings(commands=["python3 bench.py"], score_regex="EVOLOZA_SCORE=(?P<score>[0-9]+)"),
                search=SearchSettings(),
                git=GitSettings(artifacts_dir=".evoloza"),
            )
            prompt = build_planner_prompt(
                worktree,
                "Improve the engine.",
                config,
                4,
                [],
                config.planner.worker,
            )
        self.assertIn("Use only the repository snapshot included below.", prompt)
        self.assertIn("Repository snapshot:", prompt)
        self.assertIn("engine.rs", prompt)
        self.assertIn("Cargo.toml", prompt)
        self.assertNotIn("You may inspect files and run local commands as needed.", prompt)


class HypothesisRepeatTests(unittest.TestCase):
    def test_build_hypothesis_profile_extracts_engine_theme_families(self) -> None:
        profile = build_hypothesis_profile(
            "Introduce a continuation history bonus for quiet moves to improve move ordering and search efficiency."
        )
        self.assertIn("continuation-history", profile["families"])
        self.assertIn("quiet-history", profile["families"])
        self.assertIn("move-ordering", profile["families"])

    def test_build_hypothesis_profile_keeps_correction_history_distinct_from_quiet_history(self) -> None:
        profile = build_hypothesis_profile(
            "The Rust engine already allocates and ages correction-history tables but never applies them; "
            "using those tables only for pruning-side static eval in negamax, and training them conservatively "
            "from resolved quiet nodes, should improve fixed-movetime search efficiency."
        )
        self.assertIn("correction-history", profile["families"])
        self.assertNotIn("quiet-history", profile["families"])

    def test_hypothesis_repeat_reason_detects_same_family_with_different_wording(self) -> None:
        rows = [
            {
                "run_id": "run-1",
                "round": "4",
                "hypothesis": "Introduce a continuation history bonus for quiet moves to improve move ordering.",
                "summary": "No summary provided.",
            }
        ]
        reason = hypothesis_repeat_reason(
            rows,
            "Add quiet continuation history weighting to move ordering.",
            "No summary provided.",
        )
        self.assertIsNotNone(reason)
        self.assertIn("continuation-history", reason or "")

    def test_hypothesis_repeat_reason_allows_execute_mode_to_ignore_family_overlap(self) -> None:
        rows = [
            {
                "run_id": "run-1",
                "round": "4",
                "hypothesis": "Introduce a continuation history bonus for quiet moves to improve move ordering.",
                "summary": "No summary provided.",
            }
        ]
        reason = hypothesis_repeat_reason(
            rows,
            "Add quiet continuation history weighting to move ordering.",
            "No summary provided.",
            allow_family_overlap=False,
            allow_distinctive_token_overlap=False,
        )
        self.assertIsNone(reason)

    def test_hypothesis_repeat_reason_still_rejects_exact_text_when_family_overlap_is_disabled(self) -> None:
        rows = [
            {
                "run_id": "run-1",
                "round": "4",
                "hypothesis": "Introduce a continuation history bonus for quiet moves to improve move ordering.",
                "summary": "No summary provided.",
            }
        ]
        reason = hypothesis_repeat_reason(
            rows,
            "Introduce a continuation history bonus for quiet moves to improve move ordering.",
            "No summary provided.",
            allow_family_overlap=False,
            allow_distinctive_token_overlap=False,
        )
        self.assertIsNotNone(reason)
        self.assertIn("same hypothesis text", reason or "")

    def test_render_forbidden_repeat_guidance_prefers_family_summary(self) -> None:
        rows = [
            {
                "run_id": "run-1",
                "round": "1",
                "hypothesis": "Introduce a continuation history bonus for quiet moves.",
                "summary": "No summary provided.",
            },
            {
                "run_id": "run-1",
                "round": "2",
                "hypothesis": "Tune null move pruning margins in quiet positions.",
                "summary": "No summary provided.",
            },
            {
                "run_id": "run-1",
                "round": "3",
                "hypothesis": "Add continuation history weighting for quiet move ordering.",
                "summary": "No summary provided.",
            },
        ]
        guidance = render_forbidden_repeat_guidance(rows)
        self.assertIn("continuation-history", guidance)
        self.assertIn("seen 2x", guidance)
        self.assertIn("null-move-pruning", guidance)

    def test_render_forbidden_repeat_guidance_includes_repo_seeded_families(self) -> None:
        guidance = render_forbidden_repeat_guidance(
            [],
            ["continuation history for quiet ordering"],
        )
        self.assertIn("continuation-history", guidance)

    def test_hypothesis_repeat_reason_rejects_repo_seeded_family(self) -> None:
        reason = hypothesis_repeat_reason(
            [],
            "Add quiet continuation history weighting to move ordering.",
            "No summary provided.",
            ["continuation history for quiet ordering"],
        )
        self.assertIsNotNone(reason)
        self.assertIn("repo-configured forbidden direction", reason or "")

    def test_hypothesis_repeat_reason_allows_correction_history_when_quiet_history_is_forbidden(self) -> None:
        reason = hypothesis_repeat_reason(
            [],
            "The Rust engine already allocates and ages correction-history tables but never applies them; "
            "using those tables only for pruning-side static eval in negamax, and training them conservatively "
            "from resolved quiet nodes, should improve fixed-movetime search efficiency.",
            "No summary provided.",
            ["continuation history for quiet moves"],
        )
        self.assertIsNone(reason)

    def test_merge_repaired_worker_output_preserves_original_hypothesis(self) -> None:
        merged = merge_repaired_worker_output(
            {
                "hypothesis": "Introduce a continuation history bonus for quiet moves.",
                "summary": "Original summary.",
                "patch": "diff --git a/x b/x\n",
            },
            {
                "hypothesis": "Patch failed because line numbers drifted.",
                "summary": "Could not repair safely.",
                "patch": "",
            },
        )
        self.assertEqual(
            merged["hypothesis"],
            "Introduce a continuation history bonus for quiet moves.",
        )
        self.assertEqual(merged["summary"], "Could not repair safely.")


if __name__ == "__main__":
    unittest.main()
