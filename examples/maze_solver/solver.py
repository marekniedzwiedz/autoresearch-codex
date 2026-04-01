from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush


Coordinate = tuple[int, int]
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DEFAULT_BUDGET = 220
HEURISTIC_WEIGHT = 1.8


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    path: tuple[Coordinate, ...]
    cost: int
    expanded: int
    status: str


def solve(grid_lines: list[str] | tuple[str, ...], budget: int = DEFAULT_BUDGET) -> SearchResult:
    grid = tuple(grid_lines)
    start, goal = find_terminals(grid)
    frontier: list[tuple[float, int, int, Coordinate]] = []
    best_cost = {start: 0}
    came_from: dict[Coordinate, Coordinate | None] = {start: None}
    closed: set[Coordinate] = set()
    expanded = 0
    ticket = 0

    heappush(frontier, (heuristic(start, goal) * HEURISTIC_WEIGHT, 0, ticket, start))
    ticket += 1

    while frontier and expanded < budget:
        _, current_cost, _, current = heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            path = rebuild_path(came_from, goal)
            return SearchResult(True, path, current_cost, expanded, "solved")

        for neighbor in ordered_neighbors(grid, current, goal):
            if neighbor in closed:
                continue
            new_cost = current_cost + cell_cost(grid, neighbor)
            if new_cost >= best_cost.get(neighbor, 10**9):
                continue
            best_cost[neighbor] = new_cost
            came_from[neighbor] = current
            priority = new_cost + heuristic(neighbor, goal) * HEURISTIC_WEIGHT
            heappush(frontier, (priority, new_cost, ticket, neighbor))
            ticket += 1

    status = "budget_exhausted" if frontier else "no_path"
    return SearchResult(False, tuple(), -1, expanded, status)


def ordered_neighbors(grid: tuple[str, ...], current: Coordinate, goal: Coordinate) -> list[Coordinate]:
    row, col = current
    neighbors: list[Coordinate] = []
    for row_delta, col_delta in ORTHOGONAL_STEPS:
        candidate = (row + row_delta, col + col_delta)
        if is_open(grid, candidate):
            neighbors.append(candidate)
    neighbors.sort(key=lambda node: (cell_cost(grid, node), heuristic(node, goal)))
    return neighbors


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
    if cell.isdigit():
        return int(cell)
    raise ValueError("Unsupported cell: {0}".format(cell))


def heuristic(node: Coordinate, goal: Coordinate) -> int:
    return abs(goal[0] - node[0]) + abs(goal[1] - node[1])


def rebuild_path(came_from: dict[Coordinate, Coordinate | None], goal: Coordinate) -> tuple[Coordinate, ...]:
    path: list[Coordinate] = []
    current: Coordinate | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return tuple(path)


def path_cost(grid_lines: list[str] | tuple[str, ...], path: tuple[Coordinate, ...]) -> int:
    grid = tuple(grid_lines)
    if not path:
        return 0
    total = 0
    for node in path[1:]:
        total += cell_cost(grid, node)
    return total


def demo_grid() -> tuple[str, ...]:
    return (
        "S2123",
        "11#13",
        "21113",
        "31#11",
        "3111G",
    )


if __name__ == "__main__":
    sample = demo_grid()
    result = solve(sample)
    print("status={0} cost={1} expanded={2}".format(result.status, result.cost, result.expanded))
    print("path={0}".format(list(result.path)))
