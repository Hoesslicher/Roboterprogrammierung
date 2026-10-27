# coding: utf-8

"""
This code is part of the course 'Introduction to robot path planning' (Author: Bjoern Hein). 

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from shapely.geometry import Point, box
from shapely.ops import unary_union

class Benchmark(object):
        
    def __init__(self, name, collisionChecker, startList, goalList, description, level ):
        """Benchmark is used to define tests cases
        
        :name: Name of benchmark
        :environment: collision checker
        :startList: list of possible start configurations
        :goalList: list of possible goal configurations
        :description: textual description of the benchmark
        :level (int): 1 -"easy",2 - "normal",3 - "hard", 4 - "insane"
        """
        self.name = name
        self.collisionChecker = collisionChecker # environment (collision checker) 
        self.startList = startList # in limits, collision-free?
        self.goalList = goalList # in limits, collision-free?
        self.description = description
        self.level = level # in [1,2,3,4]


# Validation and assessment of benchmarks (explained in IP-4-0). Valid for a 2D point robot: the scene of the
# collision checker is the C-space.

def freeSpace(collisionChecker, radius=0.0):
    """Free space of a disc robot with the given radius (radius 0: point robot).
    Simplification: only the obstacles are grown, not the border of the environment."""
    limits = collisionChecker.getEnvironmentLimits()
    area = box(limits[0][0], limits[1][0], limits[0][1], limits[1][1])
    obstacles = unary_union(list(collisionChecker.scene.values()))
    return area.difference(obstacles.buffer(radius))

def isSolvable(benchmark, radius=0.0):
    """Is there a start and a goal in the same connected part of the free space?"""
    free = freeSpace(benchmark.collisionChecker, radius)
    for part in getattr(free, "geoms", [free]):
        if (any(part.covers(Point(start)) for start in benchmark.startList)
                and any(part.covers(Point(goal)) for goal in benchmark.goalList)):
            return True
    return False

def clearance(benchmark, maxRadius=5.0, tolerance=0.01):
    """Largest radius r (bisection), for which the benchmark is still solvable."""
    if not isSolvable(benchmark):
        return 0.0
    low, high = 0.0, maxRadius
    while high - low > tolerance:
        middle = (low + high) / 2
        if isSolvable(benchmark, middle):
            low = middle
        else:
            high = middle
    return low

def levelFromClearance(c):
    """Rule of thumb used for the levels in IPTestSuite.py"""
    return 1 if c >= 2.0 else 2 if c >= 0.75 else 3 if c >= 0.4 else 4

def validateBenchmark(benchmark):
    """Print validity, solvability and difficulty measures of a benchmark. Returns True if it is valid."""
    checker = benchmark.collisionChecker
    valid = True
    for kind, configurations in (("start", benchmark.startList), ("goal", benchmark.goalList)):
        for q in configurations:
            free = not checker.pointInCollision(q)
            print(f"  {kind} {q} valid: {free}")
            valid = valid and free
    limits = checker.getEnvironmentLimits()
    area = box(limits[0][0], limits[1][0], limits[0][1], limits[1][1])
    c = clearance(benchmark)
    print(f"  solvable: {isSolvable(benchmark)}")
    print(f"  free space: {freeSpace(checker).area / area.area:.0%} of the environment")
    print(f"  clearance: {c:.2f} -> level {levelFromClearance(c)} (given level: {benchmark.level})")
    return valid
