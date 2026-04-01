from __future__ import annotations

import random
from heapq import heappop, heappush

from solver import SearchResult, solve


Coordinate = tuple[int, int]
GRID_SIZE = 18
CASE_BUDGET = 220
SOLVE_BONUS = 10_000
QUALITY_BONUS = 2_000
QUALITY_PENALTY = 35
EFFICIENCY_BONUS = 600
CASE_SEEDS = (7, 11, 19, 23, 31, 43, 59, 71, 83, 97)
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def build_case(seed: int, size: int = GRID_SIZE) -> tuple[str, ...]:
    rng = random.Random(seed)
    grid = [[str(weight_for(rng)) for _ in range(size)] for _ in range(size)]

    mandatory_path = {(0, 0)}
    row = 0
    col = 0
    while row < size - 1 or col < size - 1:
        if row == size - 1:
            col += 1
        elif col == size - 1:
            row += 1
        elif rng.random() < 0.5:
            row += 1
        else:
            col += 1
        mandatory_path.add((row, col))

    scenic_path = {(0, 0)}
    row = 0
    col = 0
    while row < size - 1 or col < size - 1:
        move_down = row < size - 1 and (col == size - 1 or rng.random() < 0.6)
        if move_down:
            row += 1
        else:
            col += 1
        scenic_path.add((row, col))

    for row_index in range(size):
        for col_index in range(size):
            if (row_index, col_index) in mandatory_path:
                grid[row_index][col_index] = str(rng.randint(3, 7))
                continue
            if (row_index, col_index) in scenic_path:
                grid[row_index][col_index] = str(rng.randint(1, 2))
                continue
            if rng.random() < 0.2:
                grid[row_index][col_index] = "#"
            else:
                grid[row_index][col_index] = str(weight_for(rng))

    grid[0][0] = "S"
    grid[size - 1][size - 1] = "G"
    return tuple("".join(row_cells) for row_cells in grid)


def weight_for(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.45:
        return 1
    if roll < 0.7:
        return 2
    if roll < 0.85:
        return 3
    if roll < 0.93:
        return 5
    return 8


def find_terminals(grid: tuple[str, ...]) -> tuple[Coordinate, Coordinate]:
    start: Coordinate | None = None
    goal: Coordinate | None = None
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == "S":
                start = (row_index, col_index)
            elif cell == "G":
                goal = (row_index, col_index)
    if start is None or goal is None:
        raise ValueError("Grid must contain S and G")
    return start, goal


def is_open(grid: tuple[str, ...], node: Coordinate) -> bool:
    row, col = node
    return 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != "#"


def cell_cost(grid: tuple[str, ...], node: Coordinate) -> int:
    row, col = node
    cell = grid[row][col]
    if cell in {"S", "G", "."}:
        return 1
    return int(cell)


def dijkstra_cost(grid: tuple[str, ...]) -> int:
    start, goal = find_terminals(grid)
    frontier: list[tuple[int, Coordinate]] = [(0, start)]
    best_cost = {start: 0}

    while frontier:
        current_cost, current = heappop(frontier)
        if current == goal:
            return current_cost
        if current_cost != best_cost[current]:
            continue
        row, col = current
        for row_delta, col_delta in ORTHOGONAL_STEPS:
            neighbor = (row + row_delta, col + col_delta)
            if not is_open(grid, neighbor):
                continue
            new_cost = current_cost + cell_cost(grid, neighbor)
            if new_cost >= best_cost.get(neighbor, 10**9):
                continue
            best_cost[neighbor] = new_cost
            heappush(frontier, (new_cost, neighbor))
    raise ValueError("Benchmark case is unsolved for exact search")


def validate_result(grid: tuple[str, ...], result: SearchResult) -> tuple[bool, int]:
    if not result.solved:
        return True, -1
    start, goal = find_terminals(grid)
    if not result.path or result.path[0] != start or result.path[-1] != goal:
        return False, -1

    total_cost = 0
    for previous, current in zip(result.path, result.path[1:]):
        if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) != 1:
            return False, -1
        if not is_open(grid, current):
            return False, -1
        total_cost += cell_cost(grid, current)

    if total_cost != result.cost:
        return False, -1
    return True, total_cost


def case_score(optimal_cost: int, result: SearchResult) -> tuple[int, str]:
    if not result.solved:
        return 0, result.status

    gap = max(0, result.cost - optimal_cost)
    quality_score = max(0, QUALITY_BONUS - gap * QUALITY_PENALTY)
    efficiency_score = max(0, EFFICIENCY_BONUS - result.expanded)
    total = SOLVE_BONUS + quality_score + efficiency_score
    label = "solved gap={0} expanded={1}".format(gap, result.expanded)
    return total, label


def main() -> int:
    total_score = 0
    solved_count = 0
    for seed in CASE_SEEDS:
        grid = build_case(seed)
        optimal_cost = dijkstra_cost(grid)
        result = solve(grid, budget=CASE_BUDGET)
        valid, _ = validate_result(grid, result)
        if result.solved and not valid:
            result = SearchResult(False, tuple(), -1, result.expanded, "invalid_path")

        score, label = case_score(optimal_cost, result)
        if result.solved:
            solved_count += 1
        total_score += score
        print(
            "seed={0} optimal={1} result={2} case_score={3}".format(
                seed,
                optimal_cost,
                label,
                score,
            )
        )

    print("solved={0}/{1} budget={2}".format(solved_count, len(CASE_SEEDS), CASE_BUDGET))
    print("AUTORESEARCH_SCORE={0}".format(total_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
