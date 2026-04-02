from __future__ import annotations

import sys

import chess


ENGINE_NAME = "EvolozaExample"
ENGINE_AUTHOR = "OpenAI"
DEFAULT_DEPTH = 2
DEFAULT_MOVETIME_MS = 100
MATE_SCORE = 100_000

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
CENTER_CONTROL_BONUS = 12
DEVELOPMENT_BONUS = 8


def choose_move(board: chess.Board, depth: int = DEFAULT_DEPTH) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")

    best_move = legal_moves[0]
    best_score = -MATE_SCORE
    alpha = -MATE_SCORE
    beta = MATE_SCORE

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, max(0, depth - 1), -beta, -alpha)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
    return best_move


def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if board.is_checkmate():
        return -MATE_SCORE - depth
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0
    if depth == 0:
        return evaluate(board)

    best_score = -MATE_SCORE
    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if score > best_score:
            best_score = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best_score


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    return sorted(board.legal_moves, key=lambda move: move_order_score(board, move), reverse=True)


def move_order_score(board: chess.Board, move: chess.Move) -> int:
    score = 0
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        moving_piece = board.piece_at(move.from_square)
        if captured_piece is not None:
            score += 10 * PIECE_VALUES[captured_piece.piece_type]
        if moving_piece is not None:
            score -= PIECE_VALUES[moving_piece.piece_type]
    if move.promotion:
        score += PIECE_VALUES.get(move.promotion, 0)
    if board.gives_check(move):
        score += 50
    return score


def evaluate(board: chess.Board) -> int:
    white_score = side_score(board, chess.WHITE)
    black_score = side_score(board, chess.BLACK)
    score = white_score - black_score
    return score if board.turn == chess.WHITE else -score


def side_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
    score += center_control(board, color)
    score += development(board, color)
    return score


def center_control(board: chess.Board, color: chess.Color) -> int:
    return sum(CENTER_CONTROL_BONUS for square in CENTER_SQUARES if board.is_attacked_by(color, square))


def development(board: chess.Board, color: chess.Color) -> int:
    minor_pieces = board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color)
    if color == chess.WHITE:
        home_squares = {chess.B1, chess.G1, chess.C1, chess.F1}
    else:
        home_squares = {chess.B8, chess.G8, chess.C8, chess.F8}
    developed = sum(1 for square in minor_pieces if square not in home_squares)
    return developed * DEVELOPMENT_BONUS


def depth_for_movetime_ms(movetime_ms: int) -> int:
    if movetime_ms >= 800:
        return 4
    if movetime_ms >= 250:
        return 3
    return DEFAULT_DEPTH


def parse_go_depth(command: str) -> int:
    tokens = command.split()
    if "depth" in tokens:
        index = tokens.index("depth")
        if index + 1 < len(tokens):
            try:
                return max(1, int(tokens[index + 1]))
            except ValueError:
                return DEFAULT_DEPTH
    if "movetime" in tokens:
        index = tokens.index("movetime")
        if index + 1 < len(tokens):
            try:
                return depth_for_movetime_ms(int(tokens[index + 1]))
            except ValueError:
                return DEFAULT_DEPTH
    return depth_for_movetime_ms(DEFAULT_MOVETIME_MS)


def parse_position(command: str) -> chess.Board:
    tokens = command.split()
    if len(tokens) < 2:
        return chess.Board()
    if tokens[1] == "startpos":
        board = chess.Board()
        move_index = 2
    elif tokens[1] == "fen":
        move_index = len(tokens)
        if "moves" in tokens[2:]:
            move_index = tokens.index("moves")
        fen = " ".join(tokens[2:move_index])
        board = chess.Board(fen)
    else:
        board = chess.Board()
        move_index = len(tokens)
    if move_index < len(tokens) and tokens[move_index] == "moves":
        for move_text in tokens[move_index + 1 :]:
            board.push_uci(move_text)
    return board


def run_uci() -> int:
    board = chess.Board()
    for raw_line in sys.stdin:
        command = raw_line.strip()
        if not command:
            continue
        if command == "uci":
            print("id name {0}".format(ENGINE_NAME), flush=True)
            print("id author {0}".format(ENGINE_AUTHOR), flush=True)
            print("uciok", flush=True)
            continue
        if command == "isready":
            print("readyok", flush=True)
            continue
        if command == "ucinewgame":
            board = chess.Board()
            continue
        if command.startswith("position "):
            board = parse_position(command)
            continue
        if command.startswith("go"):
            if board.is_game_over(claim_draw=True):
                print("bestmove 0000", flush=True)
                continue
            depth = parse_go_depth(command)
            move = choose_move(board.copy(stack=False), depth=depth)
            print("bestmove {0}".format(move.uci()), flush=True)
            continue
        if command in {"stop", "ponderhit"}:
            continue
        if command.startswith("setoption") or command.startswith("debug "):
            continue
        if command == "quit":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(run_uci())
