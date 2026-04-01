from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import chess
import chess.engine


ROOT = Path(__file__).resolve().parent
ENGINE_PATH = ROOT / "engine.py"
ALLOWED_CHANGED_FILES = {"engine.py"}
MAX_PLIES = 400
DEFAULT_MOVETIME_MS = int(os.environ.get("UCI_MOVETIME_MS", "100"))

STARTING_POSITIONS = [
    ("giuoco_piano", "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("ruy_lopez", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"),
    ("sicilian_open", "rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4"),
    ("queens_gambit", "rnbqkbnr/pp2pppp/3p4/2p5/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4"),
    ("kings_indian", "rnbqkb1r/pppppp1p/5np1/8/3P4/2N2N2/PPP1PPPP/R1BQKB1R b KQkq - 3 3"),
]

def main() -> int:
    assert_allowed_changes()
    previous_score = read_previous_champion_score()
    with tempfile.TemporaryDirectory() as tmpdir:
        previous_path = write_previous_engine(Path(tmpdir))
        candidate_points, games = play_match(previous_path)

    previous_points = len(games) - candidate_points
    score_delta = candidate_points - half_match_points(len(games))
    score = score_from_match(previous_score, candidate_points, len(games))

    print("previous_score={0:.1f}".format(previous_score))
    print("movetime_ms={0}".format(DEFAULT_MOVETIME_MS))
    for label, candidate_is_white, points, detail in games:
        color = "white" if candidate_is_white else "black"
        print("game {0} candidate_as_{1}: {2} ({3})".format(label, color, format_points(points), detail))
    print("candidate_points={0:.1f}".format(candidate_points))
    print("previous_points={0:.1f}".format(previous_points))
    print("match_target={0:.1f}".format(half_match_points(len(games))))
    print("score_delta={0:.1f}".format(score_delta))
    print("AUTORESEARCH_SCORE={0:.1f}".format(score))
    return 0


def play_match(previous_path: Path) -> tuple[float, list[tuple[str, bool, float, str]]]:
    candidate_points = 0.0
    games = []
    with ExitStack() as stack:
        candidate_engine = launch_engine(ENGINE_PATH)
        stack.callback(safe_quit, candidate_engine)
        previous_engine = launch_engine(previous_path)
        stack.callback(safe_quit, previous_engine)
        for label, fen in STARTING_POSITIONS:
            for candidate_is_white in (True, False):
                points, detail = play_game(
                    fen=fen,
                    candidate_is_white=candidate_is_white,
                    candidate=candidate_engine,
                    previous=previous_engine,
                )
                candidate_points += points
                games.append((label, candidate_is_white, points, detail))
    return candidate_points, games


def assert_allowed_changes() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("Unable to inspect git diff against HEAD")
    changed = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    unexpected = sorted(changed - ALLOWED_CHANGED_FILES)
    if unexpected:
        raise SystemExit("Only engine.py may change in this example; found: {0}".format(", ".join(unexpected)))


def read_previous_champion_score(results_path: Path | None = None) -> float:
    path = results_path or shared_results_path()
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in reversed(rows):
        if row.get("status") not in {"baseline", "accepted"}:
            continue
        try:
            return float(row["score"])
        except (KeyError, TypeError, ValueError):
            return 0.0
    return 0.0


def shared_results_path(root: Path = ROOT) -> Path:
    return shared_repo_root(root) / ".autoresearch" / "results.tsv"


def shared_repo_root(root: Path = ROOT) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return root
    common_dir_text = result.stdout.strip()
    if not common_dir_text:
        return root
    common_dir = Path(common_dir_text)
    if not common_dir.is_absolute():
        common_dir = (root / common_dir).resolve()
    return common_dir.parent


def half_match_points(total_games: int) -> float:
    return total_games / 2.0


def score_from_match(previous_score: float, candidate_points: float, total_games: int) -> float:
    return previous_score + (candidate_points - half_match_points(total_games))


def write_previous_engine(tmpdir: Path) -> Path:
    result = subprocess.run(
        ["git", "show", "HEAD:engine.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("Unable to load HEAD:engine.py")
    path = tmpdir / "previous_engine.py"
    path.write_text(result.stdout, encoding="utf-8")
    return path


def launch_engine(path: Path) -> chess.engine.SimpleEngine:
    return chess.engine.SimpleEngine.popen_uci([sys.executable, str(path)])


def play_game(
    fen: str,
    candidate_is_white: bool,
    candidate: chess.engine.SimpleEngine,
    previous: chess.engine.SimpleEngine,
) -> tuple[float, str]:
    board = chess.Board(fen)
    candidate_color = chess.WHITE if candidate_is_white else chess.BLACK

    for _ in range(MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            break
        side_to_move = candidate if board.turn == candidate_color else previous
        try:
            result = side_to_move.play(
                board,
                chess.engine.Limit(time=DEFAULT_MOVETIME_MS / 1000.0),
                info=chess.engine.INFO_NONE,
            )
            move = result.move
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError) as exc:
            return (0.0 if board.turn == candidate_color else 1.0, "engine error: {0}".format(exc))
        if move is None or move not in board.legal_moves:
            return (0.0 if board.turn == candidate_color else 1.0, "illegal move: {0}".format(move))
        board.push(move)

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.5, "draw"
        return (1.0 if outcome.winner == candidate_color else 0.0, "checkmate")
    return 0.5, "ply-cap draw"


def format_points(points: float) -> str:
    if points == 1.0:
        return "win"
    if points == 0.0:
        return "loss"
    return "draw"


def safe_quit(engine: chess.engine.SimpleEngine) -> None:
    try:
        engine.quit()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
