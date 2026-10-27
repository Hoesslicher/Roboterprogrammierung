# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

Maze benchmarks: a grid of cells, in which a randomized depth-first search removes walls until every cell can be
reached ("perfect maze": exactly one way between two cells). Additionally removed walls create loops.

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import random

from shapely.geometry import LineString
from shapely.ops import unary_union

from IPBenchmark import Benchmark, levelFromClearance
from IPEnvironment import CollisionChecker

# east, north, west, south
NEIGHBOURS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def mazeWalls(rows, cols, seed=0, extraOpenings=0):
    """Walls of a maze in grid coordinates: sorted list of segments ((x1, y1), (x2, y2)), every segment is one side
    of a cell. The cell (i, j) covers [i, i+1] x [j, j+1].

    Args:
        rows, cols: number of cells
        seed: seed of the random generator, the same seed gives the same maze
        extraOpenings: number of additionally removed inner walls (0: perfect maze)
    """
    # own random generator: the maze neither depends on nor changes the state of the global one
    rnd = random.Random(seed)

    walls = set()
    for i in range(cols):
        for j in range(rows):
            walls.add(((i, j), (i + 1, j)))  # lower side
            walls.add(((i, j), (i, j + 1)))  # left side
    walls.update(((i, rows), (i + 1, rows)) for i in range(cols))  # upper border
    walls.update(((cols, j), (cols, j + 1)) for j in range(rows))  # right border

    # randomized depth-first search: remove the wall to a random unvisited neighbour, go back if there is none
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        candidates = [(i + di, j + dj) for di, dj in NEIGHBOURS
                      if 0 <= i + di < cols and 0 <= j + dj < rows and (i + di, j + dj) not in visited]
        if not candidates:
            stack.pop()
            continue
        neighbour = rnd.choice(candidates)
        walls.discard(_wallBetween((i, j), neighbour))
        visited.add(neighbour)
        stack.append(neighbour)

    # additional openings: several ways between two cells
    innerWalls = sorted(wall for wall in walls if not _isBorder(wall, rows, cols))
    for wall in rnd.sample(innerWalls, min(extraOpenings, len(innerWalls))):
        walls.discard(wall)

    return sorted(walls)


def _wallBetween(cell1, cell2):
    """Wall between two neighbouring cells"""
    (i1, j1), (i2, j2) = cell1, cell2
    if i1 != i2:
        i = max(i1, i2)
        return ((i, j1), (i, j1 + 1))
    j = max(j1, j2)
    return ((i1, j), (i1 + 1, j))


def _isBorder(wall, rows, cols):
    (x1, y1), (x2, y2) = wall
    return (x1 == x2 and x1 in (0, cols)) or (y1 == y2 and y1 in (0, rows))


def mazeBenchmark(rows=8, cols=8, cellSize=2.5, wallThickness=0.3, seed=0, extraOpenings=0, mergeWalls=True,
                  limits=None):
    """Benchmark with a maze in the middle of the environment: start in the lower left cell, goal in the upper
    right cell.

    Args:
        rows, cols: number of cells
        cellSize: distance between the centers of two parallel walls
        wallThickness: thickness of the walls, the corridors are cellSize - wallThickness wide
        seed: seed of the random generator, the same seed gives the same maze
        extraOpenings: number of additionally removed inner walls (0: exactly one way from start to goal)
        mergeWalls: True: all walls are merged into one obstacle, False: every wall is an obstacle of its own
        limits: limits of the environment (default: the limits of all benchmarks)

    Example:

        benchmark = mazeBenchmark(rows=6, cols=6, seed=3)
    """
    limits = [[0.0, 22.0], [0.0, 22.0]] if limits is None else limits
    assert cols * cellSize < limits[0][1] - limits[0][0] and rows * cellSize < limits[1][1] - limits[1][0], \
        "maze does not fit into the limits"

    x0 = (limits[0][0] + limits[0][1] - cols * cellSize) / 2
    y0 = (limits[1][0] + limits[1][1] - rows * cellSize) / 2

    def toWorld(point):
        return (x0 + point[0] * cellSize, y0 + point[1] * cellSize)

    # square caps: the walls meet without notches at the corners
    walls = [LineString([toWorld(a), toWorld(b)]).buffer(wallThickness / 2, cap_style="square")
             for a, b in mazeWalls(rows, cols, seed, extraOpenings)]
    if mergeWalls:
        scene = {"maze": unary_union(walls)}
    else:
        scene = {"wall" + str(k): wall for k, wall in enumerate(walls)}

    start = list(toWorld((0.5, 0.5)))
    goal = list(toWorld((cols - 0.5, rows - 0.5)))
    corridor = cellSize - wallThickness
    name = f"Maze {rows}x{cols} s{seed}" + (f" +{extraOpenings}" if extraOpenings else "")
    description = (f"Maze with {rows}x{cols} cells (seed {seed}, {extraOpenings} extra openings), "
                   f"the corridors are {corridor:.2f} wide.")
    # the narrowest passage is a corridor: clearance = half its width
    return Benchmark(name, CollisionChecker(scene, limits), [start], [goal], description,
                     levelFromClearance(corridor / 2))
